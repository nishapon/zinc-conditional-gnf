#!/usr/bin/env python3
"""Stage 12 — N-conditioned generation experiment and conditioning verification.

Two things, in one job, from one trained conditional checkpoint:

A. VERIFY THE CONDITIONING IS REAL, before any molecule is looked at.
   1. The same VALIDATION molecules scored twice with the same smoothing noise: once with
      their own standardised conditions, once with the conditions permuted across
      molecules. Same weights, same data, same marginal condition distribution - only the
      pairing differs. A model that ignores conditioning gets the identical likelihood
      (gap 0, ties 100%); a model that uses it is strictly worse under the permutation.
   2. `mu(N), sigma(N)` along the requested-N sweep. A dead conditional prior gives a flat
      curve with zero spread.
   3. Per-N empirical latent spread against the model's own `sigma(N)` - a DIAGNOSTIC that
      says whether a conditional calibration would be needed. Nothing is calibrated here.

   A run that merely finishes, or that produces molecules of the requested size, is NOT
   evidence: N also sets how many node vectors are drawn and the decoder emits one atom per
   vector, so the atom count is satisfied by construction. Section A.1 is the evidence.

B. GENERATE AT SEVERAL REQUESTED N and evaluate each size separately: requested-N delivery
   before and after decoding, raw / greedy / degree-constrained validity kept apart,
   degree-decoder fallback, QED, uniqueness, novelty against the TRAINING split and
   internal diversity. Per N it writes its own metrics.json / samples.jsonl, plus one
   summary.json and a readable report across the sweep.

The requested N values default to representative counts of the ZINC250K TRAINING
distribution (its deciles), so the sweep is not hand-picked. Novelty references the
training split only; the TEST split is never loaded.

Exit codes: 0 ok; 2 missing checkpoint or data; 3 autoencoder mismatch; 4 bad arguments;
5 the conditioning verification failed its minimum requirement; 1 unexpected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from zinc_gnf.conditioning import (  # noqa: E402
    condition_sensitivity,
    format_conditioning_check,
    latent_spread_by_n,
    load_condition_scaler,
    prior_response,
    standardise_condition,
)
from zinc_gnf.dataset import (  # noqa: E402
    DatasetFormatError,
    MolecularGraphDataset,
    load_training_splits,
    seeded_subset_indices,
)
from zinc_gnf.generation import (  # noqa: E402
    DECODERS,
    NodeCountSampler,
    generate_molecules,
    requested_n_report,
)
from zinc_gnf.metrics_generation import format_metrics, generation_metrics  # noqa: E402
from zinc_gnf.molecular_profile import profile_molecules  # noqa: E402
from zinc_gnf.trainer_autoencoder import ResumeMismatchError, load_autoencoder, resolve_device  # noqa: E402
from zinc_gnf.trainer_flow import VALIDATION_NOISE_SEED, VALIDATION_SUBSET_SEED, file_sha256, load_flow  # noqa: E402
from zinc_gnf.utils.logging import JobLogger  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402

# A conditional model must at least beat its own permuted-condition control. These are
# sanity floors, not success criteria: a smoke run with 300 steps may legitimately fail
# them, which is why --allow-weak-conditioning exists.
MIN_CONDITION_GAP = 1e-4
MIN_CONDITION_WIN_RATE = 0.55


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="data/processed/zinc250k_charge_aware_moflow_fixed")
    parser.add_argument("--ae-checkpoint", default=None,
                        help="defaults to the autoencoder recorded in the flow checkpoint")
    parser.add_argument("--target-n", default=None,
                        help="comma-separated requested atom counts; default: the deciles "
                             "of the TRAINING node-count distribution")
    parser.add_argument("--count", type=int, default=1000, help="samples per requested N")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="one temperature (the frozen unconditional result used 0.2)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decoders", default=",".join(DECODERS))
    parser.add_argument("--headline", default="degree", choices=DECODERS)
    parser.add_argument("--check-molecules", type=int, default=2000,
                        help="validation molecules for the conditioning verification")
    parser.add_argument("--no-diversity", action="store_true")
    parser.add_argument("--allow-ae-mismatch", action="store_true")
    parser.add_argument("--allow-weak-conditioning", action="store_true",
                        help="report a failed verification instead of exiting 5 (smoke tests)")
    parser.add_argument("--skip-generation", action="store_true",
                        help="run only the conditioning verification")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def default_target_n(counts: list[int]) -> list[int]:
    """Deciles of the training node-count distribution, deduplicated and sorted."""
    quantiles = np.quantile(np.asarray(counts, dtype=np.float64),
                            [0.05, 0.25, 0.5, 0.75, 0.95])
    return sorted({int(round(float(q))) for q in quantiles})


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    output_root = Path(arguments.output_dir)
    try:
        decoders = [d.strip() for d in arguments.decoders.split(",") if d.strip()]
        unknown = set(decoders) - set(DECODERS)
        if unknown:
            raise ValueError(f"unknown decoders {sorted(unknown)}")
        if arguments.count < 1:
            raise ValueError("--count must be >= 1")
        if arguments.temperature <= 0:
            raise ValueError("--temperature must be > 0")
        output_root.mkdir(parents=True, exist_ok=True)
        logger = JobLogger("generate_ncond", output_root, filename="generate_ncond.log")
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4

    started = time.perf_counter()
    try:
        device = resolve_device(arguments.device)
        flow, probe = load_flow(arguments.flow_checkpoint, device=device)
        features = flow.config.condition_features
        if not flow.config.conditional:
            raise ValueError("this flow is unconditional; use scripts/05_generate.py")
        if "n" not in features:
            raise ValueError(f"this flow conditions on {list(features)}, which does not "
                             f"include 'n'; stage 12 is the N-conditioned experiment")
        if "qed" in features:
            raise ValueError(f"this flow also conditions on QED ({list(features)}); use the "
                             f"joint N+QED script, not stage 12, so QED targets are not "
                             f"silently invented here")

        recorded = probe.get("autoencoder") or {}
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
        autoencoder.eval()
        for parameter in autoencoder.parameters():
            parameter.requires_grad_(False)
        if "degree" in decoders:
            if not autoencoder.config.degree_head:
                raise ValueError("this autoencoder has no degree head; drop 'degree' from --decoders")
            try:
                from scipy.optimize import milp  # noqa: F401
            except ImportError as error:
                raise ValueError("the degree-constrained decoder needs scipy.optimize.milp "
                                 f"(scipy >= 1.9): {error}") from error

        scaler = load_condition_scaler(arguments.data_dir)
        train_records, val_records, fingerprint = load_training_splits(arguments.data_dir)
        train_counts = [int(r["n_node"]) for r in train_records]
        train_smiles = {str(r["smiles"]) for r in train_records}
        validation_smiles = {str(r["smiles"]) for r in val_records}
        val_set = MolecularGraphDataset(val_records)
        real_by_n: dict[int, list[str]] = {}
        for record in train_records:
            real_by_n.setdefault(int(record["n_node"]), []).append(str(record["smiles"]))
        del train_records, val_records
        node_counts = NodeCountSampler(train_counts)

        targets = ([int(t) for t in str(arguments.target_n).split(",") if t.strip()]
                   if arguments.target_n else default_target_n(train_counts))
        if not targets:
            raise ValueError("no requested N values")
        if min(targets) < 1:
            raise ValueError("every requested N must be >= 1")
        unseen = [n for n in targets if n not in real_by_n]
        if unseen:
            logger.warn(f"requested N values absent from the training distribution: {unseen}; "
                        f"they are extrapolation, and are reported as such")

        logger.header("N-conditioned generation and conditioning verification")
        logger.info(f"flow={arguments.flow_checkpoint} step={probe.get('step')} "
                    f"features={list(features)} node_autoregressive="
                    f"{flow.config.node_autoregressive} temperature={arguments.temperature}")
        logger.info(f"autoencoder={ae_path} sha256={digest[:16]} (frozen, read-only)")
        logger.info(f"requested N: {targets} ({arguments.count} samples each); "
                    f"novelty reference = {len(train_smiles):,} training SMILES; "
                    f"the test split is not loaded")
        if flow.prior.is_calibrated:
            logger.warn("this checkpoint carries a prior calibration")

        # ---- A. verification -------------------------------------------------
        check_indices = seeded_subset_indices(len(val_set), arguments.check_molecules,
                                              VALIDATION_SUBSET_SEED)
        sensitivity = condition_sensitivity(flow, autoencoder, val_set, indices=check_indices,
                                            device=device, batch_size=arguments.batch_size,
                                            noise_seed=VALIDATION_NOISE_SEED)
        response = prior_response(flow, scaler=scaler, n_values=targets, device=device)
        spread = latent_spread_by_n(flow, autoencoder, val_set, n_values=targets,
                                    indices=check_indices, device=device,
                                    batch_size=arguments.batch_size,
                                    noise_seed=VALIDATION_NOISE_SEED)
        report = format_conditioning_check(sensitivity, response, spread)
        for line in report.splitlines():
            logger.info(line)
        passed = (sensitivity["gap_nats_per_node"] > MIN_CONDITION_GAP
                  and sensitivity["molecules_better_with_true_condition"] > MIN_CONDITION_WIN_RATE)
        verification = {
            "sensitivity": sensitivity, "prior_response": response, "latent_spread_by_n": spread,
            "thresholds": {"min_gap_nats_per_node": MIN_CONDITION_GAP,
                           "min_win_rate": MIN_CONDITION_WIN_RATE},
            "passed": bool(passed),
            "validation_molecules": len(check_indices),
        }
        atomic_write_json(verification, output_root / "conditioning_check.json")
        (output_root / "conditioning_check.txt").write_text(report + "\n", encoding="utf-8")
        if not passed:
            logger.error(f"CONDITIONING VERIFICATION FAILED: gap "
                         f"{sensitivity['gap_nats_per_node']:+.6f} nats/node, win rate "
                         f"{sensitivity['molecules_better_with_true_condition']:.3f}. The model "
                         f"is not using N; generated atom counts would still match by "
                         f"construction, which is why that is not the test.")
            if not arguments.allow_weak_conditioning:
                return 5
            logger.warn("continuing because --allow-weak-conditioning was given; every number "
                        "below must be reported as coming from a model with unproven "
                        "conditioning")
        else:
            logger.info(f"conditioning verified: the molecules' own N explains them "
                        f"{sensitivity['gap_nats_per_node']:+.4f} nats/node better than a "
                        f"permuted N, on "
                        f"{sensitivity['molecules_better_with_true_condition'] * 100:.1f}% of "
                        f"molecules")

        if arguments.skip_generation:
            atomic_write_json({"verification": verification, "generation": None,
                               "seconds": time.perf_counter() - started},
                              output_root / "summary.json")
            logger.info("--skip-generation: no molecules sampled")
            return 0

        # ---- B. generation at each requested N -------------------------------
        by_target: dict[str, Any] = {}
        for position, target in enumerate(targets):
            directory = output_root / f"N{target}"
            directory.mkdir(parents=True, exist_ok=True)
            values = standardise_condition(features, scaler=scaler,
                                           n=[target] * arguments.count)
            sampled = generate_molecules(
                flow, autoencoder, count=arguments.count, node_counts=node_counts,
                temperature=arguments.temperature, seed=arguments.seed + position,
                batch_size=arguments.batch_size, decoders=decoders,
                condition_values=values, target_n=target, device=device)
            records = sampled["records"]
            metrics = generation_metrics(records, train_smiles=train_smiles,
                                         validation_smiles=validation_smiles,
                                         decoders=decoders, headline=arguments.headline,
                                         diversity=not arguments.no_diversity)
            metrics["settings"] = sampled["settings"]
            metrics["requested_n"] = target
            metrics["requested_n_report"] = requested_n_report(records, target, decoders=decoders)
            metrics["flow_checkpoint"] = str(arguments.flow_checkpoint)
            metrics["flow_step"] = probe.get("step")
            metrics["autoencoder"] = {"checkpoint": str(ae_path), "sha256": digest}
            metrics["condition"] = {"features": list(features),
                                    "standardised_value": [float(v) for v in values[0]],
                                    "scaler": scaler.to_dict()}
            head = metrics["decoders"].get(arguments.headline, {})
            smiles = [r[arguments.headline].get("smiles") if
                      (r.get(arguments.headline) or {}).get("valid") else None for r in records]
            metrics["profile"] = profile_molecules(smiles, name=f"N{target}", scaffolds=True)
            # The real molecules of EXACTLY this size, as the honest reference for a
            # size-conditioned sample (the whole-split reference mixes sizes).
            reference = real_by_n.get(target, [])
            metrics["real_reference_same_n"] = (
                profile_molecules(reference[:2000], name=f"real_N{target}", scaffolds=False)
                if reference else {"note": "no training molecule has this N"})
            atomic_write_json(metrics, directory / "metrics.json")
            (directory / "metrics.txt").write_text(format_metrics(metrics) + "\n", encoding="utf-8")
            with (directory / "samples.jsonl").open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, default=float) + "\n")
            valid = [r[arguments.headline]["smiles"] for r in records
                     if (r.get(arguments.headline) or {}).get("valid")
                     and r[arguments.headline].get("smiles")]
            (directory / "valid_smiles.txt").write_text("\n".join(valid) + "\n", encoding="utf-8")
            by_target[f"N{target}"] = {
                "requested_n": target,
                "delivered_before_decoding": metrics["requested_n_report"]["before_decoding"],
                "delivered_after_decoding": metrics["requested_n_report"]["after_decoding"],
                "raw_validity": (metrics["decoders"].get("raw", {}).get("strict_validity") or {}
                                 ).get("value"),
                "headline_validity": (head.get("strict_validity") or {}).get("value"),
                "fallback_rate": (head.get("fallback_rate") or {}).get("value"),
                "uniqueness": (head.get("uniqueness") or {}).get("value"),
                "novelty_vs_train": (head.get("novelty_vs_train") or {}).get("value"),
                "internal_diversity": (head.get("diversity") or {}).get("internal_diversity"),
                "qed_mean": metrics["profile"]["qed"]["mean"],
                "qed_at_least_0p7": metrics["profile"]["qed_at_least_0p7"],
                "aromatic_rings_mean": metrics["profile"]["aromatic_rings"]["mean"],
                "real_qed_mean_same_n": (metrics["real_reference_same_n"].get("qed") or {}
                                         ).get("mean"),
                "real_aromatic_rings_same_n": (
                    metrics["real_reference_same_n"].get("aromatic_rings") or {}).get("mean"),
                "training_molecules_with_this_n": len(reference),
                "directory": str(directory),
            }
            logger.info(
                f"N={target}: raw validity "
                f"{by_target[f'N{target}']['raw_validity']}, "
                f"{arguments.headline} validity {by_target[f'N{target}']['headline_validity']}, "
                f"fallback {by_target[f'N{target}']['fallback_rate']}, "
                f"QED {by_target[f'N{target}']['qed_mean']:.4f} "
                f"(real at this N: {by_target[f'N{target}']['real_qed_mean_same_n']}), "
                f"N delivered after decoding "
                f"{metrics['requested_n_report']['after_decoding'][arguments.headline]['match_over_all_samples']:.4f}")

        summary = {
            "verification": verification,
            "by_target_n": by_target,
            "settings": {"targets": targets, "count": arguments.count,
                         "temperature": arguments.temperature, "seed": arguments.seed,
                         "headline_decoder": arguments.headline, "decoders": decoders},
            "flow": {"checkpoint": str(arguments.flow_checkpoint), "step": probe.get("step"),
                     "features": list(features),
                     "node_autoregressive": flow.config.node_autoregressive},
            "autoencoder": {"checkpoint": str(ae_path), "sha256": digest},
            "data_fingerprint": fingerprint,
            "seconds": time.perf_counter() - started,
        }
        atomic_write_json(summary, output_root / "summary.json")
        lines = [report, "", "Per requested N (headline decoder: "
                 f"{arguments.headline}, temperature {arguments.temperature}, "
                 f"{arguments.count} samples each)", ""]
        lines.append("|   N | N delivered (after decode) | raw validity | headline validity | "
                     "fallback | QED | real QED at this N | aromatic rings | real aromatic rings |")
        lines.append("|----:|---------------------------:|-------------:|------------------:|"
                     "---------:|----:|-------------------:|---------------:|--------------------:|")

        def cell(value: Any, template: str = "{:.4f}") -> str:
            return "N/A" if value is None or (isinstance(value, float) and np.isnan(value)) \
                else template.format(float(value))

        for key, row in by_target.items():
            delivered = row["delivered_after_decoding"][arguments.headline][
                "match_over_all_samples"]
            lines.append(f"| {row['requested_n']:>3} | {cell(delivered)} | "
                         f"{cell(row['raw_validity'])} | {cell(row['headline_validity'])} | "
                         f"{cell(row['fallback_rate'])} | {cell(row['qed_mean'])} | "
                         f"{cell(row['real_qed_mean_same_n'])} | "
                         f"{cell(row['aromatic_rings_mean'])} | "
                         f"{cell(row['real_aromatic_rings_same_n'])} |")
        lines += ["",
                  "'N delivered' is structural (one atom per sampled node vector) and counts an "
                  "invalid sample as a miss, so it tracks validity, not size control. Raw "
                  "validity is the only column that is purely the model's bond prediction; the "
                  "headline column includes the degree-constrained program AND its greedy "
                  "fallback, whose rate is given beside it."]
        (output_root / "ncond_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines), flush=True)
        logger.info(f"wrote {output_root}/summary.json, ncond_report.txt, "
                    f"conditioning_check.{{json,txt}} and one directory per requested N")
        return 0

    except FileNotFoundError as error:
        print(f"FATAL (missing input): {error}", file=sys.stderr, flush=True)
        return 2
    except DatasetFormatError as error:
        print(f"FATAL (dataset): {error}", file=sys.stderr, flush=True)
        return 2
    except ResumeMismatchError as error:
        print(f"FATAL (autoencoder mismatch): {error}", file=sys.stderr, flush=True)
        return 3
    except ValueError as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
