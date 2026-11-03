#!/usr/bin/env python3
"""Slice 1d — derive the MoFlow-fixed ZINC250K dataset from existing records. CPU only.

Primary-experiment split: MoFlow's official held-out molecules are the TEST
set; a deterministic validation subset is carved only from the remaining
training pool; the condition scaler is refit on the new training subset only.

Reads the already-encoded random-split dataset (default
data/processed/zinc250k_charge_aware/, left untouched) and writes a separate
dataset (default data/processed/zinc250k_charge_aware_moflow_fixed/). Nothing
is re-encoded and nothing is deduplicated. Every assumption is verified first;
on any failed check nothing is written.

MoFlow's two files must already be on disk (copy them on the login node;
compute nodes have no internet):
  data/external/moflow/zinc250k.csv
  data/external/moflow/valid_idx_zinc.json

Exit codes: 0 ok; 2 input missing; 4 bad arguments/config or output collision;
5 verification failed (nothing written); 1 unexpected error (traceback in log).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from zinc_gnf.preprocessing_charge_aware import OutputCollisionError  # noqa: E402
from zinc_gnf.split_audit import MoflowVerificationError  # noqa: E402
from zinc_gnf.split_derivation import (  # noqa: E402
    MOFLOW_README_HELDOUT_COUNT,
    MOFLOW_README_TRAIN_COUNT,
    DerivationError,
    SourceCollisionError,
    derive_moflow_fixed_dataset,
)
from zinc_gnf.utils.config import load_config  # noqa: E402
from zinc_gnf.utils.logging import JobLogger  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/zinc250k_charge_aware.yaml")
    parser.add_argument("--source-dir", default=None,
                        help="Random-split dataset to derive from (default: config data.processed_dir).")
    parser.add_argument("--output-dir", default=None,
                        help="Default: config data.moflow_fixed_dir.")
    parser.add_argument("--metrics-dir", default=None)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--moflow-csv", default="data/external/moflow/zinc250k.csv")
    parser.add_argument("--moflow-valid-idx", default="data/external/moflow/valid_idx_zinc.json")
    parser.add_argument("--validation-fraction", type=float, default=None,
                        help="Fraction of the training pool used for validation (default 0.10).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-sha256", action="store_true")
    arguments = parser.parse_args(argv)

    try:
        configuration = load_config(arguments.config)
        data = configuration["data"]
        source_dir = Path(arguments.source_dir or data["processed_dir"])
        output_dir = Path(arguments.output_dir or data["moflow_fixed_dir"])
        metrics_dir = Path(arguments.metrics_dir or data["moflow_fixed_metrics_dir"])
        csv_path = Path(arguments.csv or data["csv_path"])
        baseline_dir = Path(configuration["representation"]["baseline_processed_dir"])
        seed = int(arguments.seed if arguments.seed is not None else configuration["experiment"]["seed"])
        validation_fraction = float(
            arguments.validation_fraction if arguments.validation_fraction is not None
            else data.get("moflow_fixed_validation_fraction", 0.10)
        )
        expected_sha = None if arguments.skip_sha256 else data.get("expected_sha256")
        expected_rows = data.get("expected_rows")
        # MoFlow's published counts (README: 224,568 train; 24,887 held out). They are a
        # verification gate; a config may set them to null only for synthetic test data.
        reference_train = data.get("moflow_reference_train_count", MOFLOW_README_TRAIN_COUNT)
        reference_heldout = data.get("moflow_reference_heldout_count", MOFLOW_README_HELDOUT_COUNT)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print(f"FATAL: invalid configuration: {error}", file=sys.stderr, flush=True)
        return 4

    try:
        logger = JobLogger("derive_moflow_fixed", metrics_dir, filename="derive.log")
    except OSError as error:
        print(f"FATAL: cannot create {metrics_dir}: {error}", file=sys.stderr, flush=True)
        return 4

    try:
        derive_moflow_fixed_dataset(
            source_dir=source_dir, output_dir=output_dir, metrics_dir=metrics_dir,
            csv_path=csv_path, moflow_csv=arguments.moflow_csv,
            moflow_valid_idx=arguments.moflow_valid_idx,
            baseline_processed_dir=baseline_dir, logger=logger, seed=seed,
            validation_fraction_of_pool=validation_fraction,
            expected_sha256=expected_sha, expected_rows=expected_rows,
            moflow_reference_train_count=reference_train,
            moflow_reference_heldout_count=reference_heldout,
        )
    except FileNotFoundError as error:
        logger.error(f"FATAL: {error}")
        return 2
    except (DerivationError, MoflowVerificationError) as error:
        logger.error(f"VERIFICATION FAILED, nothing written: {error}")
        return 5
    except (OutputCollisionError, SourceCollisionError, ValueError) as error:
        logger.error(f"FATAL: {error}")
        return 4
    except Exception:  # noqa: BLE001
        logger.error("FATAL: unexpected error\n" + traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
