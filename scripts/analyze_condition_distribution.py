#!/usr/bin/env python3
"""Analyze QED and node-count conditioning distributions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import pickle
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from zinc_gnf.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--config",
        default="configs/zinc250k.yaml",
    )
    parser.add_argument(
        "--splits",
        default=None,
        help=(
            "Preprocessed zinc_splits.pkl. Defaults to "
            "<processed_dir>/zinc_splits.pkl."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/metrics/"
            "condition_distribution"
        ),
    )
    parser.add_argument(
        "--qed-bins",
        type=int,
        default=20,
    )

    return parser.parse_args()


def safe_correlation(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    if (
        len(first) < 2
        or np.std(first) == 0
        or np.std(second) == 0
    ):
        return float("nan")

    return float(
        np.corrcoef(
            first,
            second,
        )[0, 1]
    )


def json_safe(
    value: Any,
) -> Any:
    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): json_safe(item)
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        np.ndarray,
    ):
        return json_safe(
            value.tolist()
        )

    if isinstance(
        value,
        np.generic,
    ):
        return json_safe(
            value.item()
        )

    if isinstance(
        value,
        float,
    ) and not math.isfinite(value):
        return None

    return value


def write_json(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            json_safe(payload),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(
            rows
        )


def split_arrays(
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    qed = np.asarray(
        [
            float(record["qed"])
            for record in records
        ],
        dtype=np.float64,
    )

    n_node = np.asarray(
        [
            int(record["n_node"])
            for record in records
        ],
        dtype=np.int64,
    )

    if not np.isfinite(qed).all():
        raise FloatingPointError(
            "QED contains non-finite values."
        )

    if (n_node <= 0).any():
        raise ValueError(
            "Node counts must be positive."
        )

    return qed, n_node


def distribution_summary(
    qed: np.ndarray,
    n_node: np.ndarray,
) -> dict[str, Any]:
    quantiles = [
        0.00,
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        1.00,
    ]

    return {
        "molecules": int(
            len(qed)
        ),
        "qed": {
            "mean": float(
                np.mean(qed)
            ),
            "std": float(
                np.std(qed)
            ),
            "minimum": float(
                np.min(qed)
            ),
            "maximum": float(
                np.max(qed)
            ),
            "quantiles": {
                str(quantile): float(
                    np.quantile(
                        qed,
                        quantile,
                    )
                )
                for quantile
                in quantiles
            },
            "below_0_4_percent": float(
                100.0
                * np.mean(
                    qed < 0.4
                )
            ),
            "above_0_6_percent": float(
                100.0
                * np.mean(
                    qed >= 0.6
                )
            ),
            "above_0_7_percent": float(
                100.0
                * np.mean(
                    qed >= 0.7
                )
            ),
            "above_0_8_percent": float(
                100.0
                * np.mean(
                    qed >= 0.8
                )
            ),
            "above_0_9_percent": float(
                100.0
                * np.mean(
                    qed >= 0.9
                )
            ),
        },
        "n_node": {
            "mean": float(
                np.mean(n_node)
            ),
            "std": float(
                np.std(n_node)
            ),
            "minimum": int(
                np.min(n_node)
            ),
            "maximum": int(
                np.max(n_node)
            ),
            "median": float(
                np.median(n_node)
            ),
        },
        "qed_n_node_correlation": (
            safe_correlation(
                qed,
                n_node.astype(
                    np.float64
                ),
            )
        ),
    }


def main() -> None:
    arguments = parse_args()

    if arguments.qed_bins <= 1:
        raise ValueError(
            "qed-bins must be greater than one."
        )

    configuration = load_config(
        arguments.config
    )

    split_path = (
        Path(arguments.splits)
        if arguments.splits
        else Path(
            configuration["data"][
                "processed_dir"
            ]
        ) / "zinc_splits.pkl"
    )

    output_directory = Path(
        arguments.output_dir
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    with split_path.open(
        "rb"
    ) as handle:
        payload = pickle.load(
            handle
        )

    training_records = list(
        payload["train"]
    )
    validation_records = list(
        payload["val"]
    )

    train_qed, train_n = (
        split_arrays(
            training_records
        )
    )
    val_qed, val_n = (
        split_arrays(
            validation_records
        )
    )

    qed_edges = np.linspace(
        0.0,
        1.0,
        arguments.qed_bins + 1,
    )

    train_qed_count, _ = (
        np.histogram(
            train_qed,
            bins=qed_edges,
        )
    )
    val_qed_count, _ = (
        np.histogram(
            val_qed,
            bins=qed_edges,
        )
    )

    qed_rows = []

    for index in range(
        arguments.qed_bins
    ):
        lower = float(
            qed_edges[index]
        )
        upper = float(
            qed_edges[index + 1]
        )

        qed_rows.append({
            "bin_index": index,
            "qed_lower": lower,
            "qed_upper": upper,
            "qed_middle": (
                0.5
                * (lower + upper)
            ),
            "train_count": int(
                train_qed_count[index]
            ),
            "train_percent": float(
                100.0
                * train_qed_count[index]
                / len(train_qed)
            ),
            "validation_count": int(
                val_qed_count[index]
            ),
            "validation_percent": float(
                100.0
                * val_qed_count[index]
                / len(val_qed)
            ),
        })

    write_csv(
        output_directory
        / "qed_frequency.csv",
        list(
            qed_rows[0].keys()
        ),
        qed_rows,
    )

    minimum_n = int(
        min(
            train_n.min(),
            val_n.min(),
        )
    )
    maximum_n = int(
        max(
            train_n.max(),
            val_n.max(),
        )
    )

    node_rows = []

    for node_count in range(
        minimum_n,
        maximum_n + 1,
    ):
        train_count = int(
            np.sum(
                train_n == node_count
            )
        )
        validation_count = int(
            np.sum(
                val_n == node_count
            )
        )

        node_rows.append({
            "n_node": node_count,
            "train_count": train_count,
            "train_percent": float(
                100.0
                * train_count
                / len(train_n)
            ),
            "validation_count": (
                validation_count
            ),
            "validation_percent": float(
                100.0
                * validation_count
                / len(val_n)
            ),
        })

    write_csv(
        output_directory
        / "node_count_frequency.csv",
        list(
            node_rows[0].keys()
        ),
        node_rows,
    )

    node_edges = np.arange(
        minimum_n - 0.5,
        maximum_n + 1.5,
        1.0,
    )

    train_joint, _, _ = (
        np.histogram2d(
            train_n,
            train_qed,
            bins=[
                node_edges,
                qed_edges,
            ],
        )
    )

    joint_rows = []

    for node_index, node_count in enumerate(
        range(
            minimum_n,
            maximum_n + 1,
        )
    ):
        for qed_index in range(
            arguments.qed_bins
        ):
            joint_rows.append({
                "n_node": node_count,
                "qed_lower": float(
                    qed_edges[
                        qed_index
                    ]
                ),
                "qed_upper": float(
                    qed_edges[
                        qed_index + 1
                    ]
                ),
                "train_count": int(
                    train_joint[
                        node_index,
                        qed_index,
                    ]
                ),
            })

    write_csv(
        output_directory
        / "joint_qed_node_count_frequency.csv",
        list(
            joint_rows[0].keys()
        ),
        joint_rows,
    )

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    width = (
        qed_edges[1]
        - qed_edges[0]
    )

    axis.bar(
        qed_edges[:-1],
        100.0
        * train_qed_count
        / len(train_qed),
        width=width,
        align="edge",
        alpha=0.70,
        label="Training",
    )

    axis.step(
        qed_edges[:-1]
        + 0.5 * width,
        100.0
        * val_qed_count
        / len(val_qed),
        where="mid",
        linewidth=2.0,
        label="Validation",
    )

    axis.set_xlabel(
        "QED"
    )
    axis.set_ylabel(
        "Molecules (%)"
    )
    axis.set_title(
        "ZINC QED condition distribution"
    )
    axis.set_xlim(
        0.0,
        1.0,
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_directory
        / "qed_frequency.png",
        dpi=200,
    )
    plt.close(
        figure
    )

    figure, axis = plt.subplots(
        figsize=(10, 5)
    )

    node_values = np.arange(
        minimum_n,
        maximum_n + 1,
    )

    axis.bar(
        node_values - 0.2,
        [
            row["train_percent"]
            for row in node_rows
        ],
        width=0.4,
        label="Training",
    )

    axis.bar(
        node_values + 0.2,
        [
            row[
                "validation_percent"
            ]
            for row in node_rows
        ],
        width=0.4,
        label="Validation",
    )

    axis.set_xlabel(
        "Heavy-atom count"
    )
    axis.set_ylabel(
        "Molecules (%)"
    )
    axis.set_title(
        "ZINC node-count condition distribution"
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_directory
        / "node_count_frequency.png",
        dpi=200,
    )
    plt.close(
        figure
    )

    figure, axis = plt.subplots(
        figsize=(11, 7)
    )

    image = axis.imshow(
        np.log1p(
            train_joint
        ),
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[
            0.0,
            1.0,
            minimum_n - 0.5,
            maximum_n + 0.5,
        ],
    )

    axis.set_xlabel(
        "QED"
    )
    axis.set_ylabel(
        "Heavy-atom count"
    )
    axis.set_title(
        "Training joint condition frequency, log(1 + count)"
    )

    figure.colorbar(
        image,
        ax=axis,
        label="log(1 + molecules)",
    )
    figure.tight_layout()
    figure.savefig(
        output_directory
        / "joint_qed_node_count_frequency.png",
        dpi=200,
    )
    plt.close(
        figure
    )

    report = {
        "split_path": str(
            split_path
        ),
        "condition_names": (
            payload.get(
                "condition_names"
            )
        ),
        "condition_mean": (
            payload.get(
                "condition_mean"
            )
        ),
        "condition_std": (
            payload.get(
                "condition_std"
            )
        ),
        "training": (
            distribution_summary(
                train_qed,
                train_n,
            )
        ),
        "validation": (
            distribution_summary(
                val_qed,
                val_n,
            )
        ),
        "test_split_accessed": False,
    }

    write_json(
        output_directory
        / "condition_distribution.json",
        report,
    )

    print(
        "ZINC CONDITION DISTRIBUTION"
    )
    print(
        "Training molecules:",
        len(train_qed),
    )
    print(
        "Validation molecules:",
        len(val_qed),
    )
    print(
        "Training QED mean/std:",
        round(
            float(
                np.mean(train_qed)
            ),
            4,
        ),
        "/",
        round(
            float(
                np.std(train_qed)
            ),
            4,
        ),
    )
    print(
        "Training QED range:",
        round(
            float(
                np.min(train_qed)
            ),
            4,
        ),
        "to",
        round(
            float(
                np.max(train_qed)
            ),
            4,
        ),
    )
    print(
        "Training node-count range:",
        int(
            np.min(train_n)
        ),
        "to",
        int(
            np.max(train_n)
        ),
    )
    print(
        "QED/node-count correlation:",
        round(
            safe_correlation(
                train_qed,
                train_n.astype(
                    np.float64
                ),
            ),
            4,
        ),
    )
    print(
        "QED >= 0.8:",
        round(
            100.0
            * float(
                np.mean(
                    train_qed >= 0.8
                )
            ),
            3,
        ),
        "%",
    )
    print(
        "Output directory:",
        output_directory,
    )
    print(
        "Test split accessed: no"
    )
    print(
        "CONDITION DISTRIBUTION ANALYSIS COMPLETE"
    )


if __name__ == "__main__":
    main()
