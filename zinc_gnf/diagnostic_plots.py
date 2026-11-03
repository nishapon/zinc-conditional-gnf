"""Headless figures for the Phase 0 chemistry diagnostics.

Kept separate from `zinc_gnf.plots` (Slice 1) so the production plotting
module is untouched. All figures use the Agg backend and are written to a
caller-chosen directory (never the processed-data directory).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

COLOR_ALL = "#1f77b4"
COLOR_ACCEPTED = "#2ca02c"
COLOR_CHARGED = "#d62728"


def _save(figure: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def _bar(axes: Any, labels: list[str], values: list[float], color: str,
         *, log: bool = False, rotate: int = 0) -> None:
    positions = np.arange(len(labels))
    axes.bar(positions, values, color=color)
    axes.set_xticks(positions)
    axes.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center")
    if log:
        axes.set_yscale("log")
    axes.grid(axis="y", alpha=0.3)


def _empty_note(axes: Any, text: str) -> None:
    axes.text(0.5, 0.5, text, ha="center", va="center",
              transform=axes.transAxes, color="#666")
    axes.set_xticks([])
    axes.set_yticks([])


def plot_charge_overview(report: dict[str, Any], path: Path) -> Path:
    """Net charge per molecule, per-atom charge, and neutral vs charged."""
    figure, (ax_net, ax_atom, ax_class) = plt.subplots(ncols=3, figsize=(16, 4.8))
    census = report["charge_census"]

    net = census["net_charge_distribution"]
    if net:
        keys = sorted(net, key=int)
        _bar(ax_net, keys, [net[k] for k in keys], COLOR_ALL, log=True)
    else:
        _empty_note(ax_net, "no data")
    ax_net.set_title("Net formal charge per molecule")
    ax_net.set_xlabel("net charge")
    ax_net.set_ylabel("molecules (log)")

    atom = census["atom_formal_charge_distribution"]
    if atom:
        keys = sorted(atom, key=int)
        _bar(ax_atom, keys, [atom[k] for k in keys], COLOR_CHARGED, log=True)
    else:
        _empty_note(ax_atom, "no data")
    ax_atom.set_title("Formal charge per atom")
    ax_atom.set_xlabel("atom formal charge")
    ax_atom.set_ylabel("atoms (log)")

    classes = census["molecule_classes"]
    order = ["neutral", "net_cation", "net_anion", "charged_net_neutral"]
    labels = [name for name in order if classes.get(name)]
    if labels:
        _bar(ax_class, labels, [classes[n] for n in labels], COLOR_ACCEPTED,
             rotate=20)
    else:
        _empty_note(ax_class, "no data")
    counts = report["counts"]
    ax_class.set_title(
        "Neutral vs charged molecules\n"
        f"charged={counts['charged_molecules']:,} / "
        f"parseable={counts['parseable_molecules']:,}"
    )
    ax_class.set_ylabel("molecules")
    return _save(figure, path)


def plot_charged_atom_types(report: dict[str, Any], path: Path,
                            *, top: int = 15) -> Path:
    """Frequency of charge-typed atoms (e.g. N+1, O-1)."""
    figure, (ax_atoms, ax_sig) = plt.subplots(ncols=2, figsize=(14, 5))

    types = report["atom_types"]["charged_atom_types"][:top]
    if types:
        _bar(ax_atoms, [t["atom_type"] for t in types],
             [t["atoms"] for t in types], COLOR_CHARGED, log=True)
    else:
        _empty_note(ax_atoms, "no charged atoms")
    ax_atoms.set_title("Charge-typed atoms (atom count)")
    ax_atoms.set_ylabel("atoms (log)")

    signatures = report["charge_census"]["top_charge_signatures"][:top]
    if signatures:
        _bar(ax_sig, [s["signature"] or "(none)" for s in signatures],
             [s["molecules"] for s in signatures], COLOR_ALL, log=True,
             rotate=40)
    else:
        _empty_note(ax_sig, "no charged molecules")
    ax_sig.set_title("Most common charged-atom signatures (molecules)")
    ax_sig.set_ylabel("molecules (log)")
    return _save(figure, path)


def plot_rejection_reasons(report: dict[str, Any], path: Path) -> Path:
    """Rejection counts under the existing encoding rules."""
    figure, axes = plt.subplots(figsize=(9, 5))
    reasons = report["rejection_reasons"]
    if reasons:
        names = list(reasons)
        total = [reasons[n]["count"] for n in names]
        charged = [reasons[n]["of_which_charged"] for n in names]
        positions = np.arange(len(names))
        axes.barh(positions, total, color=COLOR_ALL, label="all rejected")
        axes.barh(positions, charged, color=COLOR_CHARGED,
                  label="of which charged")
        axes.set_yticks(positions)
        axes.set_yticklabels(names)
        axes.invert_yaxis()
        axes.set_xscale("log")
        axes.legend()
    else:
        _empty_note(axes, "no rejections")
    counts = report["counts"]
    axes.set_title(
        "Rejections under the existing neutral-only rules\n"
        f"rejected={counts['rejected_total']:,} of {counts['total_molecules']:,}"
    )
    axes.set_xlabel("molecules (log)")
    axes.grid(axis="x", alpha=0.3)
    return _save(figure, path)


def _hist_step(axes: Any, histogram: dict[str, Any], label: str, color: str) -> None:
    counts = np.asarray(histogram["counts"], dtype=np.float64)
    edges = np.asarray(histogram["bin_edges"], dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return
    density = counts / total / np.diff(edges)
    axes.stairs(density, edges, color=color, linewidth=2.0,
                label=f"{label} (n={int(histogram['n']):,})")


def plot_distribution_comparison(report: dict[str, Any], path: Path) -> Path:
    """All molecules vs accepted neutral-only vs charged: QED, N, aromaticity."""
    figure, axes_row = plt.subplots(ncols=3, figsize=(17, 4.8))
    histograms = report["distribution_comparison"]["histograms"]
    specs = [
        ("qed_csv", "CSV QED", "QED (CSV)"),
        ("n_atoms", "Heavy-atom count", "N atoms"),
        ("aromatic_fraction", "Aromatic atom fraction", "aromatic atoms / N"),
    ]
    for axes, (key, title, xlabel) in zip(axes_row, specs):
        _hist_step(axes, histograms["all"][key], "all", COLOR_ALL)
        _hist_step(axes, histograms["accepted"][key], "accepted (neutral-only)",
                   COLOR_ACCEPTED)
        _hist_step(axes, histograms["charged"][key], "charged", COLOR_CHARGED)
        metric_name = {"qed_csv": "qed_csv", "n_atoms": "n_atoms",
                       "aromatic_fraction": "aromatic_fraction"}[key]
        ks = report["distribution_comparison"]["metrics"][metric_name].get(
            "ks_all_vs_accepted"
        )
        subtitle = f"\nKS(all vs accepted)={ks['statistic']:.4f}" if ks else ""
        axes.set_title(title + subtitle)
        axes.set_xlabel(xlabel)
        axes.set_ylabel("density")
        axes.grid(alpha=0.3)
        axes.legend(fontsize=8)
    return _save(figure, path)


def plot_element_composition(report: dict[str, Any], path: Path) -> Path:
    """Element fractions: all parseable molecules vs accepted neutral-only."""
    figure, axes = plt.subplots(figsize=(10, 5))
    composition = report["distribution_comparison"]["element_composition"]
    everyone = composition["all_parseable"]
    accepted = composition["accepted_neutral_only"]
    elements = sorted(set(everyone) | set(accepted),
                      key=lambda e: -everyone.get(e, 0.0))
    if elements:
        positions = np.arange(len(elements))
        width = 0.4
        axes.bar(positions - width / 2, [everyone.get(e, 0.0) for e in elements],
                 width, color=COLOR_ALL, label="all parseable")
        axes.bar(positions + width / 2, [accepted.get(e, 0.0) for e in elements],
                 width, color=COLOR_ACCEPTED, label="accepted (neutral-only)")
        axes.set_xticks(positions)
        axes.set_xticklabels(elements)
        axes.set_yscale("log")
        axes.legend()
    else:
        _empty_note(axes, "no data")
    axes.set_title(
        "Element composition (atom fractions); total variation distance = "
        f"{composition['total_variation_distance']:.4f}"
    )
    axes.set_ylabel("fraction of atoms (log)")
    axes.grid(axis="y", alpha=0.3)
    return _save(figure, path)


def plot_qed_discrepancy(census: Any, report: dict[str, Any], path: Path) -> Path:
    """CSV QED vs RDKit QED: all parseable, and the charge-strip shift."""
    figure, (ax_all, ax_hist, ax_strip) = plt.subplots(ncols=3, figsize=(17, 4.8))

    csv_all, rdkit_all = census.qed_pairs("all_parseable")
    if csv_all.size:
        step = max(1, csv_all.size // 40000)
        ax_all.scatter(csv_all[::step], rdkit_all[::step], s=1, alpha=0.15,
                       color=COLOR_ALL)
        ax_all.plot([0, 1], [0, 1], color="#999", linestyle="--", linewidth=1)
    else:
        _empty_note(ax_all, "no data")
    ax_all.set_xlim(0, 1)
    ax_all.set_ylim(0, 1)
    ax_all.set_xlabel("CSV QED")
    ax_all.set_ylabel("RDKit QED on the CSV SMILES as-is")
    ax_all.set_title("All parseable molecules")
    ax_all.grid(alpha=0.3)

    for name, color, label in (
        ("uncharged", COLOR_ACCEPTED, "uncharged"),
        ("charged", COLOR_CHARGED, "charged"),
    ):
        csv_values, rdkit_values = census.qed_pairs(name)
        if csv_values.size:
            ax_hist.hist(np.abs(rdkit_values - csv_values), bins=60,
                         histtype="step", linewidth=2.0, color=color,
                         label=f"{label} (n={csv_values.size:,})")
    ax_hist.set_yscale("log")
    ax_hist.set_xlabel("|RDKit QED - CSV QED|")
    ax_hist.set_ylabel("molecules (log)")
    ax_hist.set_title("QED agreement, charged vs uncharged")
    ax_hist.legend()
    ax_hist.grid(alpha=0.3)

    csv_strip, stripped = census.qed_pairs("charge_stripped_recovered")
    if csv_strip.size:
        signed = stripped - csv_strip
        ax_strip.hist(signed, bins=60, color=COLOR_CHARGED)
        ax_strip.axvline(0.0, color="#333", linewidth=1)
        ax_strip.set_title(
            "Diagnostic counterfactual: QED(charge-stripped) - CSV QED\n"
            f"mean={signed.mean():+.4f}, mean|.|={np.abs(signed).mean():.4f}"
        )
    else:
        _empty_note(ax_strip, "no charge-strip counterfactual data")
        ax_strip.set_title("Charge-strip counterfactual")
    ax_strip.set_xlabel("signed QED shift")
    ax_strip.set_ylabel("molecules")
    ax_strip.grid(alpha=0.3)
    return _save(figure, path)


def plot_counterfactual(report: dict[str, Any], path: Path) -> Path:
    """Diagnostic charge-strip counterfactual: recovered vs still rejected."""
    figure, (ax_total, ax_reasons) = plt.subplots(ncols=2, figsize=(14, 5))
    cf = report["charge_strip_counterfactual"]
    counts = report["counts"]

    labels = ["accepted now\n(neutral-only)", "hypothetical\n(+ stripped charged)"]
    values = [counts["accepted_neutral_only"], cf["hypothetical_accepted_total"]]
    _bar(ax_total, labels, values, COLOR_ACCEPTED)
    ax_total.axhline(counts["total_molecules"], color="#999", linestyle="--",
                     label=f"all rows ({counts['total_molecules']:,})")
    ax_total.legend()
    ax_total.set_title("Molecules kept: current rule vs diagnostic what-if")
    ax_total.set_ylabel("molecules")

    reasons = cf["still_rejected_reasons"]
    if reasons:
        names = list(reasons)
        _bar(ax_reasons, names, [reasons[n] for n in names], COLOR_CHARGED,
             log=True, rotate=35)
    else:
        _empty_note(ax_reasons, "nothing still rejected")
    ax_reasons.set_title("Charged molecules NOT recovered by naive stripping")
    ax_reasons.set_ylabel("molecules (log)")
    return _save(figure, path)


def write_chemistry_figures(census: Any, report: dict[str, Any],
                            figures_dir: str | Path) -> list[Path]:
    """Write every Phase 0 chemistry figure; returns the written paths."""
    directory = Path(figures_dir)
    return [
        plot_charge_overview(report, directory / "charge_overview.png"),
        plot_charged_atom_types(report, directory / "charged_atom_types.png"),
        plot_rejection_reasons(report, directory / "rejection_reasons.png"),
        plot_distribution_comparison(
            report, directory / "distribution_all_vs_accepted.png"),
        plot_element_composition(
            report, directory / "element_composition_all_vs_accepted.png"),
        plot_qed_discrepancy(
            census, report, directory / "qed_csv_vs_rdkit_charge.png"),
        plot_counterfactual(
            report, directory / "charge_strip_counterfactual.png"),
    ]


__all__ = [
    "plot_charge_overview",
    "plot_charged_atom_types",
    "plot_counterfactual",
    "plot_distribution_comparison",
    "plot_element_composition",
    "plot_qed_discrepancy",
    "plot_rejection_reasons",
    "write_chemistry_figures",
]
