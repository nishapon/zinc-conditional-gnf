#!/usr/bin/env python3
"""Stage 5 — sample molecules from a trained flow and score them.

Pipeline per temperature: draw node counts from the TRAINING histogram, sample
latents from the prior scaled by the temperature, invert the flow, decode with
the frozen autoencoder, assemble bonds with all three decoders (raw argmax,
greedy valence-aware, degree-constrained) and compute the generation metrics.

The autoencoder used is the one recorded in the flow checkpoint; its SHA-256
is verified, so a flow can never be scored through a different decoder than
the one that produced its training embeddings.

Novelty is measured against the TRAINING split only (train+val is reported as
a secondary number). The test split is never loaded.

Several temperatures can be given at once (`--temperature 0.7,0.85,1.0`), each
written to its own subdirectory, so the temperature can be chosen on
validation data in a single job.

Writes under --output-dir, per temperature:
    metrics.json, metrics.txt, samples.jsonl, valid_smiles.txt

Exit codes: 0 ok; 2 missing checkpoint or data; 3 autoencoder mismatch;
4 bad arguments; 1 unexpected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch  # noqa: E402

from zinc_gnf.dataset import DatasetFormatError, load_training_splits  # noqa: E402
from zinc_gnf.generation import (  # noqa: E402
    DECODERS,
    NodeCountSampler,
    generate_molecules,
    node_count_agreement,
    size_distribution,
)
from zinc_gnf.metrics_generation import format_metrics, generation_metrics  # noqa: E402
from zinc_gnf.trainer_autoencoder import ResumeMismatchError, load_autoencoder, resolve_device  # noqa: E402
from zinc_gnf.trainer_flow import file_sha256, load_flow  # noqa: E402
from zinc_gnf.utils.logging import JobLogger  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="data/processed/zinc250k_charge_aware_moflow_fixed")
    parser.add_argument("--ae-checkpoint", default=None,
                        help="defaults to the autoencoder recorded in the flow checkpoint")
    parser.add_argument("--count", type=int, default=10000, help="samples per temperature")
    parser.add_argument("--temperature", default="1.0", help="comma-separated list")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-n", type=int, default=None,
                        help="fix every sample's atom count instead of drawing it")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--decoders", default=",".join(DECODERS))
    parser.add_argument("--headline", default="degree", choices=DECODERS)
    parser.add_argument("--no-diversity", action="store_true",
                        help="skip the O(k^2) Tanimoto diversity computation")
    parser.add_argument("--allow-ae-mismatch", action="store_true",
                        help="score with an autoencoder other than the recorded one (not for results)")
    parser.add_argument("--save-samples", action="store_true", default=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    output_root = Path(arguments.output_dir)
    try:
        temperatures = [float(t) for t in str(arguments.temperature).split(",") if t.strip()]
        decoders = [d.strip() for d in arguments.decoders.split(",") if d.strip()]
        if not temperatures:
            raise ValueError("no temperature given")
        if arguments.count < 1:
            raise ValueError("--count must be >= 1")
        unknown = set(decoders) - set(DECODERS)
        if unknown:
            raise ValueError(f"unknown decoders {sorted(unknown)}")
        output_root.mkdir(parents=True, exist_ok=True)
        logger = JobLogger("generate", output_root, filename="generate.log")
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4

    try:
        device = resolve_device(arguments.device)
        flow, probe = load_flow(arguments.flow_checkpoint, device=device)
        recorded = (probe.get("autoencoder") or {})
        ae_path = Path(arguments.ae_checkpoint or recorded.get("checkpoint", ""))
        if not ae_path.exists():
            raise FileNotFoundError(f"autoencoder checkpoint not found: {ae_path}")
        digest = file_sha256(ae_path)
        if recorded.get("sha256") and digest != recorded["sha256"]:
            message = (f"autoencoder mismatch: the flow was trained with sha256 "
                       f"{recorded['sha256'][:16]} but {ae_path} has {digest[:16]}")
            if not arguments.allow_ae_mismatch:
                raise ResumeMismatchError(message)
            logger.warn(message + " (continuing because --allow-ae-mismatch was given)")
        autoencoder, _ = load_autoencoder(ae_path, device=device)
        for parameter in autoencoder.parameters():
            parameter.requires_grad_(False)
        autoencoder.eval()
        if "degree" in decoders and not autoencoder.config.degree_head:
            raise ValueError("this autoencoder has no degree head; drop 'degree' from --decoders")
        if "degree" in decoders:
            # Without a MILP solver every single sample would fall back to the greedy
            # decoder and the "degree" column would silently be the greedy one. Fail loudly.
            try:
                from scipy.optimize import milp  # noqa: F401
            except ImportError as error:
                raise ValueError(
                    "the degree-constrained decoder needs scipy.optimize.milp (scipy >= 1.9); "
                    f"without it every molecule falls back to greedy decoding: {error}") from error

        train_records, val_records, fingerprint = load_training_splits(arguments.data_dir)
        train_counts = [int(r["n_node"]) for r in train_records]
        train_smiles = {str(r["smiles"]) for r in train_records}
        validation_smiles = {str(r["smiles"]) for r in val_records}
        del train_records, val_records
        node_counts = NodeCountSampler(train_counts)

        logger.header("Generation from the unconditional flow")
        logger.info(f"flow={arguments.flow_checkpoint} step={probe.get('step')} "
                    f"conditional={flow.config.conditional} noise_sigma={flow.config.noise_sigma}")
        logger.info(f"autoencoder={ae_path} sha256={digest[:16]} degree_head={autoencoder.config.degree_head}")
        logger.info(f"novelty reference: {len(train_smiles):,} training molecules "
                    f"(train+val secondary: {len(train_smiles | validation_smiles):,}); "
                    f"test split NOT loaded")
        logger.info(f"node counts from the training histogram: mean={node_counts.mean:.2f} "
                    f"sizes {int(node_counts.values.min())}..{int(node_counts.values.max())}")
        logger.info(f"device={device} count={arguments.count:,} temperatures={temperatures} "
                    f"decoders={decoders} headline={arguments.headline}")

        summary = {}
        for temperature in temperatures:
            directory = output_root / f"T{temperature:g}"
            directory.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            sampled = generate_molecules(
                flow, autoencoder, count=arguments.count, node_counts=node_counts,
                temperature=temperature, seed=arguments.seed, batch_size=arguments.batch_size,
                decoders=decoders, target_n=arguments.target_n, device=device)
            records = sampled["records"]
            metrics = generation_metrics(records, train_smiles=train_smiles,
                                         validation_smiles=validation_smiles, decoders=decoders,
                                         headline=arguments.headline,
                                         diversity=not arguments.no_diversity)
            metrics["settings"] = sampled["settings"]
            metrics["flow_checkpoint"] = str(arguments.flow_checkpoint)
            metrics["flow_step"] = probe.get("step")
            metrics["autoencoder"] = {"checkpoint": str(ae_path), "sha256": digest}
            metrics["data"] = {k: fingerprint.get(k) for k in
                               ("representation", "split_protocol", "counts", "counts_digest")}
            metrics["requested_size_distribution"] = size_distribution(records)
            if arguments.headline in decoders:
                metrics["node_count_agreement"] = node_count_agreement(records, arguments.headline)
            metrics["seconds"] = time.perf_counter() - started
            # Sampling-stability diagnostics. A large round-trip error means the inverse used
            # for sampling is not the inverse of the density being reported, and any metric
            # below would describe a different model than the one that was trained.
            round_trip = sampled["settings"]["latent_round_trip_relative_error"]
            if round_trip is not None and round_trip > 1e-3:
                logger.warn(f"T={temperature:g}: re-encoding the samples misses their latents by "
                            f"{round_trip:.2e} (relative). The flow may be in a numerically "
                            f"unstable regime; check log|det J|/node in the training log before "
                            f"trusting these metrics.")
            if sampled["settings"]["nonfinite_embeddings"]:
                logger.warn(f"T={temperature:g}: {sampled['settings']['nonfinite_embeddings']} of "
                            f"{arguments.count} samples had non-finite embeddings and are counted "
                            f"as invalid (rate "
                            f"{sampled['settings']['nonfinite_embedding_rate']:.4f})")
            atomic_write_json(metrics, directory / "metrics.json")
            report = format_metrics(metrics)
            (directory / "metrics.txt").write_text(report + "\n", encoding="utf-8")
            if arguments.save_samples:
                with (directory / "samples.jsonl").open("w", encoding="utf-8") as handle:
                    for record in records:
                        handle.write(json.dumps(record, default=float) + "\n")
                head = arguments.headline
                valid = [r[head]["smiles"] for r in records
                         if r.get(head, {}).get("valid") and r[head].get("smiles")]
                (directory / "valid_smiles.txt").write_text("\n".join(valid) + "\n", encoding="utf-8")
            logger.info(f"T={temperature:g} done in {metrics['seconds']:.1f}s -> {directory}")
            print(report, flush=True)
            summary[f"T{temperature:g}"] = metrics.get("headline", {})
        atomic_write_json({"flow_checkpoint": str(arguments.flow_checkpoint),
                           "count": arguments.count, "headline_decoder": arguments.headline,
                           "by_temperature": summary}, output_root / "summary.json")
        logger.info(f"summary: {output_root / 'summary.json'}")
    except FileNotFoundError as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 2
    except ResumeMismatchError as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 3
    except (DatasetFormatError, ValueError) as error:
        print(f"FATAL: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 4
    except Exception:  # noqa: BLE001
        print("FATAL: unexpected error\n" + traceback.format_exc(), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
