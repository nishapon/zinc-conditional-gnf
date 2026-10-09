"""Headless matplotlib plots for Slice 1 and later diagnostic stages.

Why a dedicated module (not inline in each script): plotting code benefits
from being tested in isolation and reused across stages. Keeps the CLI
scripts short — each script is just "load inputs, call a function, save
output".

All functions explicitly set the matplotlib backend to Agg (file-only) at
module import so no X11 server is required on the compute node.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt      # noqa: E402  - after backend set
import numpy as np                   # noqa: E402


# ---------------------------------------------------------------------------
# QED distribution per split.
# ---------------------------------------------------------------------------
def plot_qed_distribution(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    output_path: str | Path,
    *,
    bins: int = 40,
    title: str = "QED distribution per split (train/val/test)",
) -> Path:
    """Overlaid histogram of QED values across train/val/test splits.

    Reads `splits["train"|"val"|"test"]`, each a list of records with a
    `qed` field. Each split's histogram is normalized by its own count so
    the three curves are directly comparable at the shape level — a visual
    sanity check that random splitting did not accidentally skew QED
    coverage between splits.
    """
    figure, axes = plt.subplots(figsize=(8, 5))

    palette = {
        "train": "#1f77b4",
        "val": "#2ca02c",
        "test": "#d62728",
    }

    for split_name, records in splits.items():
        values = np.asarray(
            [float(r["qed"]) for r in records], dtype=np.float64,
        )
        if values.size == 0:
            continue
        axes.hist(
            values,
            bins=bins,
            range=(0.0, 1.0),
            density=True,
            histtype="step",
            linewidth=2.0,
            label=f"{split_name} (n={values.size:,})",
            color=palette.get(split_name, None),
        )

    axes.set_xlabel("QED (RDKit-recomputed)")
    axes.set_ylabel("Density")
    axes.set_title(title)
    axes.set_xlim(0.0, 1.0)
    axes.legend()
    axes.grid(alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=130)
    plt.close(figure)
    return output_path


# ---------------------------------------------------------------------------
# Heavy-atom-count distribution per split.
# ---------------------------------------------------------------------------
def plot_n_distribution(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    output_path: str | Path,
    *,
    max_nodes: int = 38,
    title: str = "Heavy-atom count per split (train/val/test)",
) -> Path:
    figure, axes = plt.subplots(figsize=(8, 5))

    palette = {
        "train": "#1f77b4",
        "val": "#2ca02c",
        "test": "#d62728",
    }
    bin_edges = np.arange(0, max_nodes + 2) - 0.5

    for split_name, records in splits.items():
        values = np.asarray(
            [int(r["n_node"]) for r in records], dtype=np.int64,
        )
        if values.size == 0:
            continue
        axes.hist(
            values,
            bins=bin_edges,
            density=True,
            histtype="step",
            linewidth=2.0,
            label=f"{split_name} (n={values.size:,})",
            color=palette.get(split_name, None),
        )

    axes.set_xlabel("Heavy-atom count N")
    axes.set_ylabel("Density")
    axes.set_title(title)
    axes.legend()
    axes.grid(alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=130)
    plt.close(figure)
    return output_path


# ---------------------------------------------------------------------------
# CSV QED vs. RDKit QED — scatter plus residual histogram.
# ---------------------------------------------------------------------------
def plot_qed_csv_vs_rdkit(
    qed_pairs: Iterable[Mapping[str, float]],
    output_path: str | Path,
    *,
    title: str = "ZINC CSV QED vs. RDKit-recomputed QED",
) -> Path:
    """Diagnostic plot for the CSV-vs-RDKit QED cross-check.

    Left: scatter with y = x reference. Right: histogram of the absolute
    discrepancy. If the two sources agree perfectly, the scatter sits on
    the diagonal and the histogram concentrates at zero. If there's a
    systematic drift (RDKit version change), it will show as an offset
    cluster. If a handful of molecules disagree massively, they'll be
    outliers in both panels.
    """
    pairs = list(qed_pairs)
    if not pairs:
        raise ValueError("no QED pairs to plot")

    csv_qed = np.asarray(
        [p["csv_qed"] for p in pairs], dtype=np.float64,
    )
    rdkit_qed = np.asarray(
        [p["rdkit_qed"] for p in pairs], dtype=np.float64,
    )
    deltas = np.abs(csv_qed - rdkit_qed)

    figure, (ax_scatter, ax_hist) = plt.subplots(
        ncols=2, figsize=(12, 5),
    )

    ax_scatter.scatter(csv_qed, rdkit_qed, s=1, alpha=0.1, color="#1f77b4")
    ax_scatter.plot([0, 1], [0, 1], color="#999", linestyle="--", linewidth=1)
    ax_scatter.set_xlabel("CSV QED (ZINC250K file)")
    ax_scatter.set_ylabel("RDKit-recomputed QED")
    ax_scatter.set_title("Scatter (log-density under points may hide overlap)")
    ax_scatter.set_xlim(0.0, 1.0)
    ax_scatter.set_ylim(0.0, 1.0)
    ax_scatter.grid(alpha=0.3)

    ax_hist.hist(deltas, bins=60, color="#1f77b4")
    ax_hist.set_yscale("log")
    ax_hist.set_xlabel("|CSV QED - RDKit QED|")
    ax_hist.set_ylabel("Molecule count (log)")
    ax_hist.set_title(
        f"Mean abs delta={deltas.mean():.4f}, "
        f"max={deltas.max():.4f}, p99={np.quantile(deltas, 0.99):.4f}"
    )
    ax_hist.grid(alpha=0.3)

    figure.suptitle(title)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=130)
    plt.close(figure)
    return output_path


__all__ = [
    "plot_n_distribution",
    "plot_qed_csv_vs_rdkit",
    "plot_qed_distribution",
]
