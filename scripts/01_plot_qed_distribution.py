#!/usr/bin/env python3
"""Slice 1 companion script: produces the three Slice 1 diagnostic figures
from the preprocessing artifacts.

Figures:
    outputs/figures/qed_histogram.png        QED per split, overlaid
    outputs/figures/n_distribution.png       N per split, overlaid
    outputs/figures/qed_csv_vs_rdkit.png     CSV QED vs RDKit QED

Reads only files produced by scripts/01_preprocess.py; does not re-encode.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import pickle
import sys

from zinc_gnf.plots import (
    plot_n_distribution,
    plot_qed_csv_vs_rdkit,
    plot_qed_distribution,
)
from zinc_gnf.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/zinc250k.yaml",
        help="Experiment YAML configuration.",
    )
    parser.add_argument(
        "--processed-dir",
        default=None,
        help="Override the config's data.processed_dir.",
    )
    parser.add_argument(
        "--metrics-dir",
        default="outputs/metrics/preprocessing",
        help="Where qed_pairs.pkl lives (produced by 01_preprocess.py).",
    )
    parser.add_argument(
        "--figures-dir",
        default="outputs/figures",
        help="Where to write the PNGs.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    configuration = load_config(arguments.config)

    data_section = configuration["data"]
    processed_dir = Path(
        arguments.processed_dir or data_section["processed_dir"]
    )
    metrics_dir = Path(arguments.metrics_dir)
    figures_dir = Path(arguments.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    splits_path = processed_dir / "splits.pkl"
    qed_pairs_path = metrics_dir / "qed_pairs.pkl"

    if not splits_path.exists():
        print(
            f"FATAL: {splits_path} not found. "
            "Did scripts/01_preprocess.py finish successfully?",
            file=sys.stderr,
        )
        return 2
    if not qed_pairs_path.exists():
        print(
            f"FATAL: {qed_pairs_path} not found. "
            "Did scripts/01_preprocess.py finish successfully?",
            file=sys.stderr,
        )
        return 2

    print("=" * 70)
    print("ZINC250K preprocessing diagnostic plots")
    print(f"Start time:    {datetime.now().isoformat(timespec='seconds')}")
    print(f"Splits:        {splits_path}")
    print(f"QED pairs:     {qed_pairs_path}")
    print(f"Figures dir:   {figures_dir}")
    print("=" * 70)

    with splits_path.open("rb") as handle:
        payload = pickle.load(handle)

    splits = {k: payload[k] for k in ("train", "val", "test")}

    with qed_pairs_path.open("rb") as handle:
        qed_pairs = pickle.load(handle)

    qed_hist_path = plot_qed_distribution(
        splits, figures_dir / "qed_histogram.png",
    )
    n_hist_path = plot_n_distribution(
        splits,
        figures_dir / "n_distribution.png",
        max_nodes=int(data_section["max_nodes"]),
    )
    qed_scatter_path = plot_qed_csv_vs_rdkit(
        qed_pairs, figures_dir / "qed_csv_vs_rdkit.png",
    )

    print(f"  wrote {qed_hist_path}")
    print(f"  wrote {n_hist_path}")
    print(f"  wrote {qed_scatter_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
