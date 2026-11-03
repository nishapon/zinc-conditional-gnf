#!/usr/bin/env python3
"""Stage 10 — measure a trained flow's latent calibration and install it in the prior.

Reads an EXISTING flow checkpoint, measures the per-dimension mean and std of the
latents it produces on TRAINING molecules (through the frozen autoencoder recorded in
that checkpoint, with the flow's own smoothing noise), and writes a NEW checkpoint
whose prior is that measured Gaussian. No retraining; the input checkpoint, the
autoencoder and every earlier result are untouched, and the output path is refused if
it already exists unless --force is given.

Why this is the right kind of change: for a diagonal Gaussian prior family the
maximum-likelihood parameters of the observed latents are their empirical mean and
std, and the trained prior belongs to that family, so on the measurement data the
calibrated model's likelihood is >= the trained model's, with equality exactly when
the flow was already calibrated. Both log_prob and sample use the calibrated prior, so
the model remains a normalized density rather than a sampling-time trick.

The validation split is scored before and after with a fixed noise stream, so the
effect on held-out likelihood is reported honestly (including when it is negative,
which would mean the training latents are not representative). The TEST split is never
loaded.

Exit codes: 0 ok; 2 missing checkpoint or data; 3 autoencoder mismatch;
4 bad arguments or refused overwrite; 1 unexpected.
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

import torch  # noqa: E402

from zinc_gnf.dataset import (  # noqa: E402
    DatasetFormatError,
    MolecularGraphDataset,
    load_training_splits,
    seeded_subset_indices,
)
from zinc_gnf.latent_calibration import (  # noqa: E402
    CALIBRATION_NOISE_SEED,
    CALIBRATION_SUBSET_SEED,
    calibrate_prior,
    dataset_negative_log_likelihood,
    format_calibration,
    latent_statistics,
)
from zinc_gnf.trainer_autoencoder import ResumeMismatchError, load_autoencoder, resolve_device  # noqa: E402
from zinc_gnf.trainer_flow import (  # noqa: E402
    VALIDATION_NOISE_SEED,
    VALIDATION_SUBSET_SEED,
    autoencoder_seen_indices,
    file_sha256,
    load_flow,
)
from zinc_gnf.utils.checkpointing import save_checkpoint  # noqa: E402
from zinc_gnf.utils.logging import JobLogger  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402

RESERVED_CHECKPOINT_KEYS = {
    "epoch", "best_metric", "configuration", "model_state_dict",
    "optimizer_state_dict", "torch_rng_state", "cuda_rng_state",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--output-checkpoint", default=None,
                        help="default: <input stem>_calibrated.pt next to the input")
    parser.add_argument("--output-dir", default=None,
                        help="where calibration.json/.txt go; default: the output checkpoint's directory")
    parser.add_argument("--data-dir", default="data/processed/zinc250k_charge_aware_moflow_fixed")
    parser.add_argument("--ae-checkpoint", default=None,
                        help="defaults to the autoencoder recorded in the flow checkpoint")
    parser.add_argument("--molecules", type=int, default=20000,
                        help="training molecules used for the measurement (0 = all)")
    parser.add_argument("--val-molecules", type=int, default=5000,
                        help="validation molecules used for the before/after likelihood check")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--std-floor", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-ae-mismatch", action="store_true")
    parser.add_argument("--measure-only", action="store_true",
                        help="report the calibration but write no checkpoint")
    parser.add_argument("--force", action="store_true",
                        help="allow overwriting an existing output checkpoint")
    return parser.parse_args(argv)


def training_indices(count: int, probe: dict[str, Any]) -> list[int] | None:
    """The training molecules the flow was trained on, in dataset order.

    Reproduces the checkpoint's own `ae_seen_only` / `max_train_molecules` restriction
    so the calibration is measured on the same population the flow was fitted to.
    """
    train_config = probe.get("train_config") or {}
    if train_config.get("ae_seen_only"):
        keep = autoencoder_seen_indices(count, int(train_config.get("ae_subset_seed", 0)),
                                        int(train_config.get("ae_subset_size", count)))
    else:
        keep = list(range(count))
    limit = train_config.get("max_train_molecules")
    if limit:
        keep = keep[: int(limit)]
    return keep


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        flow_path = Path(arguments.flow_checkpoint)
        output_checkpoint = Path(arguments.output_checkpoint) if arguments.output_checkpoint \
            else flow_path.with_name(f"{flow_path.stem}_calibrated.pt")
        output_dir = Path(arguments.output_dir) if arguments.output_dir else output_checkpoint.parent
        if output_checkpoint.resolve() == flow_path.resolve():
            raise ValueError("refusing to write the calibrated prior over the input checkpoint")
        if output_checkpoint.exists() and not (arguments.force or arguments.measure_only):
            raise ValueError(f"{output_checkpoint} already exists; pass --force to replace it")
        if arguments.molecules < 0 or arguments.val_molecules < 0:
            raise ValueError("--molecules and --val-molecules must be >= 0")
        if arguments.batch_size < 1:
            raise ValueError("--batch-size must be >= 1")
        output_dir.mkdir(parents=True, exist_ok=True)
        logger = JobLogger("calibrate", output_dir, filename="calibrate.log")
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4

    started = time.time()
    try:
        device = resolve_device(arguments.device)
        flow, probe = load_flow(flow_path, device=device)
        if flow.config.conditional:
            logger.error("this flow has a conditional prior; calibration here is "
                         "unconditional-only (see zinc_gnf/latent_calibration.py)")
            return 4
        recorded = probe.get("autoencoder") or {}
        ae_path = Path(arguments.ae_checkpoint or recorded.get("checkpoint", ""))
        if not ae_path.exists():
            raise FileNotFoundError(f"autoencoder checkpoint not found: {ae_path}")
        digest = file_sha256(ae_path)
        if recorded.get("sha256") and digest != recorded["sha256"]:
            message = (f"autoencoder mismatch: flow was trained with sha256 "
                       f"{recorded['sha256'][:16]} but {ae_path} has {digest[:16]}")
            if not arguments.allow_ae_mismatch:
                raise ResumeMismatchError(message)
            logger.warn(message + " (continuing because --allow-ae-mismatch was given)")
        autoencoder, _ = load_autoencoder(ae_path, device=device)
        autoencoder.eval()
        for parameter in autoencoder.parameters():
            parameter.requires_grad_(False)

        train_records, val_records, fingerprint = load_training_splits(arguments.data_dir)
        train_set = MolecularGraphDataset(train_records)
        val_set = MolecularGraphDataset(val_records)
        del train_records, val_records

        pool = training_indices(len(train_set), probe)
        if arguments.molecules and arguments.molecules < len(pool):
            chosen = seeded_subset_indices(len(pool), arguments.molecules, CALIBRATION_SUBSET_SEED)
            measurement_indices = [pool[i] for i in chosen]
        else:
            measurement_indices = list(pool)
        val_indices = seeded_subset_indices(len(val_set), arguments.val_molecules or len(val_set),
                                            VALIDATION_SUBSET_SEED)

        logger.header("Latent calibration of a trained flow")
        logger.info(f"flow={flow_path} step={probe.get('step')} "
                    f"node_autoregressive={flow.config.node_autoregressive} "
                    f"noise_sigma={flow.config.noise_sigma} device={device}")
        logger.info(f"autoencoder={ae_path} sha256={digest[:16]} (frozen, read-only)")
        logger.info(f"measuring on {len(measurement_indices):,} training molecules of "
                    f"{len(train_set):,}; held-out check on {len(val_indices):,} validation "
                    f"molecules; the test split is not loaded")
        if flow.prior.is_calibrated:
            logger.warn("the input checkpoint is ALREADY calibrated; this run re-measures the "
                        "latents and replaces the calibration (it does not compose)")

        before_val = dataset_negative_log_likelihood(
            flow, autoencoder, val_set, indices=val_indices, device=device,
            batch_size=arguments.batch_size, noise_seed=VALIDATION_NOISE_SEED)
        statistics = latent_statistics(
            flow, autoencoder, train_set, indices=measurement_indices, device=device,
            batch_size=arguments.batch_size, noise_seed=CALIBRATION_NOISE_SEED)
        before_train = dataset_negative_log_likelihood(
            flow, autoencoder, train_set, indices=measurement_indices, device=device,
            batch_size=arguments.batch_size, noise_seed=CALIBRATION_NOISE_SEED)

        report = format_calibration(statistics)
        for line in report.splitlines():
            logger.info(line)

        note = calibrate_prior(flow, statistics, std_floor=arguments.std_floor)
        after_train = dataset_negative_log_likelihood(
            flow, autoencoder, train_set, indices=measurement_indices, device=device,
            batch_size=arguments.batch_size, noise_seed=CALIBRATION_NOISE_SEED)
        after_val = dataset_negative_log_likelihood(
            flow, autoencoder, val_set, indices=val_indices, device=device,
            batch_size=arguments.batch_size, noise_seed=VALIDATION_NOISE_SEED)

        train_gain = before_train["nll_per_node"] - after_train["nll_per_node"]
        val_gain = before_val["nll_per_node"] - after_val["nll_per_node"]
        logger.info(f"train NLL/node {before_train['nll_per_node']:.4f} -> "
                    f"{after_train['nll_per_node']:.4f}  (gain {train_gain:+.4f} nats)")
        logger.info(f"val   NLL/node {before_val['nll_per_node']:.4f} -> "
                    f"{after_val['nll_per_node']:.4f}  (gain {val_gain:+.4f} nats)")
        if train_gain < -1e-4:
            logger.error("train likelihood got WORSE; that contradicts the maximum-likelihood "
                         "argument and means something is wrong - not writing a checkpoint")
            return 1
        if val_gain < 0:
            logger.warn("held-out likelihood got worse: the training latents are not "
                        "representative of validation latents. Report this, do not hide it.")
        if note["floored_dimensions"]:
            logger.warn(f"{note['floored_dimensions']} latent dimensions had std below "
                        f"{arguments.std_floor} and were floored")

        payload: dict[str, Any] = {
            "flow_checkpoint": str(flow_path),
            "flow_checkpoint_sha256": file_sha256(flow_path),
            "flow_step": probe.get("step"),
            "autoencoder": {"checkpoint": str(ae_path), "sha256": digest},
            "data_fingerprint": fingerprint,
            "measurement": {"molecules": len(measurement_indices),
                            "nodes": statistics["nodes"],
                            "subset_seed": CALIBRATION_SUBSET_SEED,
                            "noise_seed": CALIBRATION_NOISE_SEED},
            "statistics": statistics["summary"],
            "per_dimension": {
                "latent_mean": [float(v) for v in statistics["mean"]],
                "latent_std": [float(v) for v in statistics["std"]],
                "prior_mu": [float(v) for v in statistics["prior_mu"]],
                "prior_sigma": [float(v) for v in statistics["prior_sigma"]],
            },
            "calibration": note,
            "likelihood": {
                "train_before": before_train, "train_after": after_train,
                "train_gain_nats_per_node": train_gain,
                "val_before": before_val, "val_after": after_val,
                "val_gain_nats_per_node": val_gain,
                "val_molecules": len(val_indices),
                "val_subset_seed": VALIDATION_SUBSET_SEED,
                "val_noise_seed": VALIDATION_NOISE_SEED,
            },
            "measure_only": bool(arguments.measure_only),
            "seconds": time.time() - started,
        }

        if arguments.measure_only:
            logger.info("--measure-only: no checkpoint written")
        else:
            extra = {k: v for k, v in probe.items() if k not in RESERVED_CHECKPOINT_KEYS}
            extra["latent_calibration"] = {k: payload[k] for k in
                                           ("flow_checkpoint", "flow_checkpoint_sha256",
                                            "measurement", "statistics", "calibration",
                                            "likelihood")}
            save_checkpoint(output_checkpoint, flow, None,
                            epoch=int(probe.get("epoch", 0)),
                            best_metric=float(probe.get("best_metric", float("inf"))),
                            configuration=probe.get("configuration", {}),
                            extra=extra)
            payload["output_checkpoint"] = str(output_checkpoint)
            payload["output_checkpoint_sha256"] = file_sha256(output_checkpoint)
            logger.info(f"calibrated checkpoint written to {output_checkpoint} "
                        f"(the input checkpoint is unchanged)")

        atomic_write_json(payload, output_dir / "calibration.json")
        text = (report + "\n\n"
                f"train NLL/node {before_train['nll_per_node']:.4f} -> "
                f"{after_train['nll_per_node']:.4f}  (gain {train_gain:+.4f} nats)\n"
                f"val   NLL/node {before_val['nll_per_node']:.4f} -> "
                f"{after_val['nll_per_node']:.4f}  (gain {val_gain:+.4f} nats)\n")
        (output_dir / "calibration.txt").write_text(text, encoding="utf-8")
        logger.info(f"wrote {output_dir / 'calibration.json'} and "
                    f"{output_dir / 'calibration.txt'}")
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
