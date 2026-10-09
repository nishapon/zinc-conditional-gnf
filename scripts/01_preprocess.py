#!/usr/bin/env python3
"""Slice 1 — Preprocess ZINC250K: encode, split, cross-check QED, write
splits.pkl + scaler.json + report.json.

CPU-only. Deterministic given --seed. No network calls — the ZINC250K CSV
must already exist on disk (ISML compute nodes have no internet).

Expected runtime: ~5-10 minutes single-process on 249,455 molecules.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

from zinc_gnf.preprocessing import (
    REFERENCE_MOFLOW_COUNT,
    prepare_zinc_dataset,
)
from zinc_gnf.utils.config import load_config
from zinc_gnf.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--config",
        default="configs/zinc250k.yaml",
        help="Experiment YAML configuration.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help=(
            "Override the config's data.csv_path. By default uses the "
            "path from the config file."
        ),
    )
    parser.add_argument(
        "--processed-dir",
        default=None,
        help=(
            "Override the config's data.processed_dir."
        ),
    )
    parser.add_argument(
        "--metrics-dir",
        default="outputs/metrics/preprocessing",
        help="Where to write auxiliary summary files.",
    )
    parser.add_argument(
        "--expected-sha256",
        default=None,
        help=(
            "Optional expected SHA-256 of the CSV. If provided and the "
            "file hash does not match, the run aborts. Use this to pin "
            "dataset identity across re-runs."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar (useful in SLURM logs).",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    configuration = load_config(arguments.config)

    seed = int(configuration["experiment"]["seed"])
    set_seed(seed)

    data_section = configuration["data"]
    csv_path = Path(arguments.csv or data_section["csv_path"])
    processed_dir = Path(
        arguments.processed_dir or data_section["processed_dir"]
    )
    metrics_dir = Path(arguments.metrics_dir)

    print("=" * 70)
    print("ZINC250K preprocessing")
    print(f"Start time:        {datetime.now().isoformat(timespec='seconds')}")
    print(f"CSV path:          {csv_path}")
    print(f"Processed dir:     {processed_dir}")
    print(f"Metrics dir:       {metrics_dir}")
    print(f"Seed:              {seed}")
    print(f"Train fraction:    {data_section['train_fraction']}")
    print(f"Validation frac:   {data_section['validation_fraction']}")
    print(f"Min / max nodes:   {data_section['min_nodes']} / {data_section['max_nodes']}")
    print(f"Reference moflow:  {REFERENCE_MOFLOW_COUNT} molecules")
    print("=" * 70)

    start_time = time.monotonic()
    try:
        report = prepare_zinc_dataset(
            csv_path=csv_path,
            processed_dir=processed_dir,
            metrics_dir=metrics_dir,
            seed=seed,
            train_fraction=float(data_section["train_fraction"]),
            validation_fraction=float(data_section["validation_fraction"]),
            min_nodes=int(data_section["min_nodes"]),
            max_nodes=int(data_section["max_nodes"]),
            expected_sha256=arguments.expected_sha256,
            progress=not arguments.no_progress,
        )
    except FileNotFoundError as error:
        print(f"\nFATAL: {error}", file=sys.stderr)
        return 2
    except RuntimeError as error:
        print(f"\nFATAL: {error}", file=sys.stderr)
        return 3

    elapsed = time.monotonic() - start_time

    print("=" * 70)
    print(f"Done in {elapsed:.1f}s.")
    print(f"Candidate rows:        {report.candidate_rows:,}")
    print(f"Accepted molecules:    {report.accepted_molecules:,} "
          f"({100 * report.accepted_fraction:.2f}%)")
    print(f"  Reference (MoFlow):  {REFERENCE_MOFLOW_COUNT:,}")
    print(f"Train / Val / Test:    "
          f"{report.train_molecules:,} / "
          f"{report.validation_molecules:,} / "
          f"{report.test_molecules:,}")
    print(f"CSV SHA-256:           {report.csv_sha256}")
    print()
    print("Rejection reasons:")
    for reason, count in sorted(
        report.rejection_reasons.items(),
        key=lambda kv: -kv[1],
    ):
        print(f"  {reason:<32} {count:>8,}")
    print()
    print("QED (RDKit) stats:")
    for key, value in report.qed_rdkit.items():
        print(f"  {key:<12} {value:.5f}")
    print()
    print("CSV QED vs RDKit QED discrepancy:")
    print(json.dumps(report.qed_csv_vs_rdkit, indent=2))
    print("=" * 70)
    print(f"splits.pkl -> {processed_dir / 'splits.pkl'}")
    print(f"scaler.json -> {processed_dir / 'scaler.json'}")
    print(f"report.json -> {processed_dir / 'report.json'}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
