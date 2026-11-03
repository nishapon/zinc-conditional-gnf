#!/usr/bin/env python3
"""Phase 0 — CPU profiler for the existing ZINC250K preprocessing path.

Profiles the real `zinc_gnf.chemistry.encode_zinc_smiles` on a deterministic
seeded sample (default 3000 molecules) and reports RDKit parse,
canonicalization, kekulization, reconstruction and QED time, clean
throughput, tail latencies, the usable CPU allocation, pipeline I/O costs
(CSV read, SHA-256, pickle write/read on each --scratch-dir), a fixed
machine-speed calibration, and an extrapolated full-dataset runtime.

It does NOT write or overwrite the production processed dataset: outputs go
to --output-dir (checked to be outside data.processed_dir) and scratch files
are deleted.

Outputs:
  <output-dir>/preprocessing_profile.json   machine-readable report
  <output-dir>/preprocessing_profile.txt    human-readable summary
  <output-dir>/run.log                      timestamped log (also on stdout)

Run on the cluster ONLY through sbatch (slurm/00_profile_preprocessing.sh).
Exit codes: 0 ok; 2 CSV missing; 3 dataset identity mismatch;
4 invalid arguments or configuration; 130 stopped early (partial report);
1 unexpected error (traceback is written to the log).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from zinc_gnf.diagnostics import (  # noqa: E402
    EXPECTED_ZINC250K_SHA256,
    SKIP_HASH_CHECK,
    DatasetIdentityError,
)
from zinc_gnf.profiling import (  # noqa: E402
    DEFAULT_SAMPLE_SIZE,
    run_preprocessing_profile,
)
from zinc_gnf.utils.config import load_config  # noqa: E402
from zinc_gnf.utils.logging import JobLogger  # noqa: E402
from zinc_gnf.utils.runtime import StopFlag  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="configs/zinc250k.yaml")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--output-dir", default="outputs/diagnostics/profiling")
    parser.add_argument(
        "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
        help="Molecules to profile (recommended 2000-5000).",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--scratch-dir", action="append", default=None,
        help=("Directory for the pickle write/read benchmark; repeat to "
              "compare locations (e.g. node-local $TMPDIR vs the shared "
              "home directory). Default: the system temp directory."),
    )
    parser.add_argument(
        "--expected-sha256", default=EXPECTED_ZINC250K_SHA256,
        help=f"Pinned dataset digest; pass '{SKIP_HASH_CHECK}' to disable.",
    )
    parser.add_argument("--skip-io", action="store_true",
                        help="Skip SHA-256 and pickle benchmarks (smoke tests).")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--heartbeat-seconds", type=float, default=15.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)

    try:
        configuration = load_config(arguments.config)
        data_section = configuration["data"]
        csv_path = Path(arguments.csv or data_section["csv_path"])
        processed_dir = Path(data_section["processed_dir"])
        min_nodes = int(data_section["min_nodes"])
        max_nodes = int(data_section["max_nodes"])
        split_seed = int(configuration["experiment"]["seed"])
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print(f"FATAL: invalid configuration: {error}", file=sys.stderr, flush=True)
        return 4

    if arguments.sample_size < 1:
        print("FATAL: --sample-size must be >= 1", file=sys.stderr, flush=True)
        return 4

    expected_sha = (
        None if arguments.expected_sha256 == SKIP_HASH_CHECK
        else arguments.expected_sha256
    )
    output_dir = Path(arguments.output_dir)

    try:
        logger = JobLogger("preprocessing_profile", output_dir, filename="run.log")
    except OSError as error:
        print(f"FATAL: cannot create output directory {output_dir}: {error}",
              file=sys.stderr, flush=True)
        return 4

    try:
        with StopFlag(logger) as stop_flag:
            report = run_preprocessing_profile(
                csv_path=csv_path,
                output_dir=output_dir,
                processed_dir=processed_dir,
                logger=logger,
                sample_size=arguments.sample_size,
                sample_seed=arguments.sample_seed,
                min_nodes=min_nodes,
                max_nodes=max_nodes,
                expected_sha256=expected_sha,
                scratch_dirs=arguments.scratch_dir,
                skip_io=arguments.skip_io,
                skip_calibration=arguments.skip_calibration,
                progress_every=arguments.progress_every,
                heartbeat_seconds=arguments.heartbeat_seconds,
                split_seed=split_seed,
                repository_dir=REPOSITORY_ROOT,
                stop_flag=stop_flag,
            )
    except FileNotFoundError as error:
        logger.error(f"FATAL: {error}")
        return 2
    except DatasetIdentityError as error:
        logger.error(f"FATAL: {error}")
        return 3
    except ValueError as error:
        logger.error(f"FATAL: {error}")
        return 4
    except Exception:  # noqa: BLE001 - we want the traceback in the SLURM log.
        logger.error("FATAL: unexpected error\n" + traceback.format_exc())
        return 1

    return 0 if report["complete"] else 130


if __name__ == "__main__":
    sys.exit(main())
