#!/usr/bin/env python3
"""Create a scaffold-disjoint ZINC conditional-GNF dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from zinc_gnf.data.preprocessing import prepare_zinc_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to zinc250k.csv.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the processed split and scaler.",
    )
    parser.add_argument(
        "--target-molecules",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=12_000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20_260_928,
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=38,
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_args()

    split_path, scaler_path, report = prepare_zinc_dataset(
        arguments.csv,
        arguments.output_dir,
        target_molecules=arguments.target_molecules,
        candidate_count=arguments.candidate_count,
        seed=arguments.seed,
        max_nodes=arguments.max_nodes,
        train_fraction=arguments.train_fraction,
        validation_fraction=arguments.validation_fraction,
    )

    print("ZINC PREPROCESSING COMPLETE")
    print(json.dumps(asdict(report), indent=2))
    print("Split:", split_path)
    print("Scaler:", scaler_path)


if __name__ == "__main__":
    main()
