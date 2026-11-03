#!/usr/bin/env python3
"""Slice 1b — charge-aware ZINC250K preprocessing (representation v2). CPU only.

A separate path from the neutral-only baseline (scripts/01_preprocess.py),
which is unchanged. Keeps charged molecules by encoding atoms as
(element, formal charge) types, with explicit per-row accounting.

Outputs (never the baseline directory):
  data/processed/zinc250k_charge_aware/splits.pkl     train/val/test records
  data/processed/zinc250k_charge_aware/scaler.json    train-only condition scaler
  data/processed/zinc250k_charge_aware/report.json    full statistics
  data/processed/zinc250k_charge_aware/row_index.csv  every CSV row: status/reason/split
  outputs/metrics/preprocessing_charge_aware/*.json   rejections, duplicates, overlap, ...
  outputs/metrics/preprocessing_charge_aware/rejected_rows.csv
  outputs/metrics/preprocessing_charge_aware/run.log

Nothing is written to the processed directory unless the whole run succeeds.
Run on the cluster ONLY through sbatch (slurm/01b_preprocess_charge_aware.sh).
Exit codes: 0 ok; 2 CSV missing; 3 dataset identity mismatch; 4 invalid
arguments/config/MoFlow verification failure; 130 interrupted (nothing
written); 1 unexpected error (traceback is in the log).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from zinc_gnf.preprocessing_charge_aware import (  # noqa: E402
    SPLIT_PROTOCOLS,
    OutputCollisionError,
    PreprocessingInterrupted,
    prepare_zinc_dataset_charge_aware,
)
from zinc_gnf.split_audit import MoflowVerificationError  # noqa: E402
from zinc_gnf.utils.config import load_config  # noqa: E402
from zinc_gnf.utils.logging import JobLogger  # noqa: E402
from zinc_gnf.utils.runtime import StopFlag  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="configs/zinc250k_charge_aware.yaml")
    parser.add_argument("--csv", default=None, help="Override data.csv_path.")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--metrics-dir", default=None)
    parser.add_argument(
        "--expected-sha256", default=None,
        help="Override the pinned digest; 'skip' disables the check.",
    )
    parser.add_argument("--split-protocol", choices=SPLIT_PROTOCOLS, default=None)
    parser.add_argument("--moflow-csv", default=None,
                        help="MoFlow data/zinc250k.csv (copied to the login node).")
    parser.add_argument("--moflow-valid-idx", default=None,
                        help="MoFlow data/valid_idx_zinc.json.")
    parser.add_argument("--max-molecules", type=int, default=None,
                        help="Deterministic subsample for smoke tests only.")
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)

    try:
        configuration = load_config(arguments.config)
        data = configuration["data"]
        csv_path = Path(arguments.csv or data["csv_path"])
        processed_dir = Path(arguments.processed_dir or data["processed_dir"])
        metrics_dir = Path(arguments.metrics_dir or data["metrics_dir"])
        baseline_dir = Path(configuration["representation"]["baseline_processed_dir"])
        seed = int(configuration["experiment"]["seed"])
        split_protocol = arguments.split_protocol or data.get("split_protocol", "random_80_10_10")
        expected_sha = arguments.expected_sha256 or data.get("expected_sha256")
        if expected_sha == "skip":
            expected_sha = None
        expected_rows = data.get("expected_rows")
        if arguments.max_molecules is not None:
            expected_rows = None
        settings = dict(
            train_fraction=float(data["train_fraction"]),
            validation_fraction=float(data["validation_fraction"]),
            min_nodes=int(data["min_nodes"]),
            max_nodes=int(data["max_nodes"]),
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print(f"FATAL: invalid configuration: {error}", file=sys.stderr, flush=True)
        return 4

    if arguments.max_molecules is not None and arguments.max_molecules < 1:
        print("FATAL: --max-molecules must be >= 1", file=sys.stderr, flush=True)
        return 4

    try:
        logger = JobLogger("preprocess_charge_aware", metrics_dir, filename="run.log")
    except OSError as error:
        print(f"FATAL: cannot create {metrics_dir}: {error}", file=sys.stderr, flush=True)
        return 4

    try:
        with StopFlag(logger) as stop_flag:
            prepare_zinc_dataset_charge_aware(
                csv_path=csv_path,
                processed_dir=processed_dir,
                metrics_dir=metrics_dir,
                baseline_processed_dir=baseline_dir,
                seed=seed,
                logger=logger,
                expected_sha256=expected_sha,
                expected_rows=expected_rows,
                split_protocol=split_protocol,
                moflow_csv=arguments.moflow_csv,
                moflow_valid_idx=arguments.moflow_valid_idx,
                max_molecules=arguments.max_molecules,
                progress_every=arguments.progress_every,
                heartbeat_seconds=arguments.heartbeat_seconds,
                stop_flag=stop_flag,
                **settings,
            )
    except FileNotFoundError as error:
        logger.error(f"FATAL: {error}")
        return 2
    except PreprocessingInterrupted as error:
        logger.error(f"INTERRUPTED: {error}")
        return 130
    except (OutputCollisionError, MoflowVerificationError, ValueError) as error:
        logger.error(f"FATAL: {error}")
        return 4
    except RuntimeError as error:
        logger.error(f"FATAL: {error}")
        return 3 if "SHA-256 mismatch" in str(error) else 1
    except Exception:  # noqa: BLE001 - keep the traceback in the SLURM log.
        logger.error("FATAL: unexpected error\n" + traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
