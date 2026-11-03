#!/usr/bin/env python3
"""Phase 0 — ZINC250K chemistry diagnostics (CPU only, read-only on data).

Reads the existing ZINC250K CSV and reports, WITHOUT changing the encoding
rules or removing any molecule:

  * total records, per-reason rejection counts (existing neutral-only rules),
  * formal-charge census and charge-typed atom frequencies,
  * charged vs neutral molecule counts,
  * CSV-vs-RDKit QED discrepancies,
  * all molecules vs accepted neutral-only molecules,
  * a diagnostic-only charge-stripping counterfactual.

Outputs (never inside data.processed_dir):
  <output-dir>/chemistry_report.json     machine-readable report
  <output-dir>/chemistry_summary.txt     human-readable summary
  <output-dir>/run.log                   timestamped log (also on stdout)
  <figures-dir>/*.png                    diagnostic figures

If the job is stopped by SIGTERM (SLURM time limit or scancel), a
`*.partial.json` report is written instead of losing the run.

Run on the cluster ONLY through sbatch (slurm/00_chemistry_diagnostics.sh).
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
    run_chemistry_diagnostics,
)
from zinc_gnf.utils.config import load_config  # noqa: E402
from zinc_gnf.utils.logging import JobLogger  # noqa: E402
from zinc_gnf.utils.runtime import StopFlag  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="configs/zinc250k.yaml")
    parser.add_argument(
        "--csv", default=None,
        help="Override data.csv_path from the config (default data/raw/zinc250k.csv).",
    )
    parser.add_argument("--output-dir", default="outputs/diagnostics/chemistry")
    parser.add_argument("--figures-dir", default="outputs/figures/diagnostics")
    parser.add_argument(
        "--expected-sha256", default=EXPECTED_ZINC250K_SHA256,
        help=f"Pinned dataset digest; pass '{SKIP_HASH_CHECK}' to disable.",
    )
    parser.add_argument(
        "--max-molecules", type=int, default=None,
        help="Deterministic subsample (smoke tests only). Default: all rows.",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument(
        "--no-counterfactual", action="store_true",
        help="Skip the diagnostic charge-stripping counterfactual.",
    )
    parser.add_argument("--no-figures", action="store_true")
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
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print(f"FATAL: invalid configuration: {error}", file=sys.stderr, flush=True)
        return 4

    if arguments.max_molecules is not None and arguments.max_molecules < 1:
        print("FATAL: --max-molecules must be >= 1", file=sys.stderr, flush=True)
        return 4

    expected_sha = (
        None if arguments.expected_sha256 == SKIP_HASH_CHECK
        else arguments.expected_sha256
    )
    output_dir = Path(arguments.output_dir)

    try:
        logger = JobLogger("chemistry_diagnostics", output_dir, filename="run.log")
    except OSError as error:
        print(f"FATAL: cannot create output directory {output_dir}: {error}",
              file=sys.stderr, flush=True)
        return 4

    try:
        with StopFlag(logger) as stop_flag:
            report = run_chemistry_diagnostics(
                csv_path=csv_path,
                output_dir=output_dir,
                processed_dir=processed_dir,
                figures_dir=None if arguments.no_figures else arguments.figures_dir,
                logger=logger,
                min_nodes=min_nodes,
                max_nodes=max_nodes,
                expected_sha256=expected_sha,
                max_molecules=arguments.max_molecules,
                sample_seed=arguments.sample_seed,
                progress_every=arguments.progress_every,
                heartbeat_seconds=arguments.heartbeat_seconds,
                counterfactual=not arguments.no_counterfactual,
                make_figures=not arguments.no_figures,
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
