"""Is the QED we asked for actually IN the training data at that N?

Part A of the stage-14 investigation. Everything here reads the processed dataset and
measures; nothing is trained and no model is needed (the generated-distribution overlay
takes already-saved `samples.jsonl` files as input).

The four candidate explanations for weak QED control, and what in this module tests each:

  data imbalance          `qed_histogram`, `support_table` - how many molecules exist at
                          each (N, QED) and how thin the high-QED tail is per N
  infeasible targets      `feasibility` - a requested (N, QED) is classified against the
                          EMPIRICAL conditional distribution at that N: inside the
                          central mass, in a thin tail, or outside the observed range.
                          Asking for QED 0.8 at an N where the data's 99th percentile is
                          0.72 is an impossible request, and the resulting MAE says
                          nothing about the model
  wrong labels / scaler   `verify_labels` - recompute QED with RDKit from the stored
                          graph (atom types + bond matrix, i.e. the representation the
                          model is actually trained on, not the original SMILES string)
                          and re-derive the standardised condition from scaler.json
  the model               `compare_distributions` - the generated QED distribution at a
                          requested target against the real distribution at the same N,
                          compared as DISTRIBUTIONS (quantiles, overlap, Wasserstein),
                          not as means

A mean can match while the distribution is wrong, and a distribution can be wrong for
either of two different reasons - the model missing the target, or the target not
existing in the data. Keeping those apart is the whole point of this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import QED

from zinc_gnf.chemistry_charge_aware import graph_to_mol_charge_aware
from zinc_gnf.preprocessing import ConditionScaler

QED_BINS = np.linspace(0.0, 1.0, 51)              # 0.02-wide bins
QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
HIGH_QED_THRESHOLDS = (0.7, 0.8)
# A requested QED is "thin tail" if fewer than this fraction of molecules at that N lie
# within +-0.05 of it, and "outside support" if none do and it is beyond the observed range.
THIN_TAIL_FRACTION = 0.01
NEIGHBOURHOOD = 0.05


def qed_histogram(qed: Sequence[float], *, bins: np.ndarray = QED_BINS) -> dict[str, Any]:
    """Counts, density and quantiles of one QED population."""
    values = np.asarray([q for q in qed if np.isfinite(q)], dtype=np.float64)
    counts, edges = np.histogram(values, bins=bins)
    return {
        "molecules": int(values.size),
        "bin_edges": [float(e) for e in edges],
        "counts": [int(c) for c in counts],
        "density": [float(c) / max(1, values.size) for c in counts],
        "mean": float(values.mean()) if values.size else float("nan"),
        "std": float(values.std()) if values.size else float("nan"),
        "quantiles": {str(q): float(np.quantile(values, q)) for q in QUANTILES}
        if values.size else {},
        **{f"fraction_at_least_{str(t).replace('.', 'p')}":
           float((values >= t).mean()) if values.size else float("nan")
           for t in HIGH_QED_THRESHOLDS},
        **{f"count_at_least_{str(t).replace('.', 'p')}": int((values >= t).sum())
           for t in HIGH_QED_THRESHOLDS},
    }


def by_n(records: Iterable[dict[str, Any]]) -> dict[int, list[float]]:
    """QED values grouped by heavy-atom count."""
    grouped: dict[int, list[float]] = {}
    for record in records:
        grouped.setdefault(int(record["n_node"]), []).append(float(record["qed"]))
    return grouped


def support_table(grouped: dict[int, list[float]], *,
                  thresholds: Sequence[float] = HIGH_QED_THRESHOLDS) -> list[dict[str, Any]]:
    """Per-N: how many molecules, and how much high-QED mass there actually is."""
    rows = []
    total = sum(len(v) for v in grouped.values())
    for n in sorted(grouped):
        values = np.asarray(grouped[n], dtype=np.float64)
        row: dict[str, Any] = {
            "n": int(n),
            "molecules": int(values.size),
            "share_of_split": float(values.size / total) if total else float("nan"),
            "qed_mean": float(values.mean()),
            "qed_std": float(values.std()),
        }
        row.update({f"q{int(q * 100)}": float(np.quantile(values, q)) for q in QUANTILES})
        for threshold in thresholds:
            key = str(threshold).replace(".", "p")
            row[f"count_ge_{key}"] = int((values >= threshold).sum())
            row[f"fraction_ge_{key}"] = float((values >= threshold).mean())
        rows.append(row)
    return rows


def n_qed_heatmap(grouped: dict[int, list[float]], *,
                  bins: np.ndarray = QED_BINS) -> dict[str, Any]:
    """Counts on the (N, QED) grid plus the conditional quantiles of QED given N."""
    counts = sorted(grouped)
    matrix = np.zeros((len(counts), len(bins) - 1), dtype=np.int64)
    for row, n in enumerate(counts):
        matrix[row], _ = np.histogram(np.asarray(grouped[n], dtype=np.float64), bins=bins)
    row_sums = matrix.sum(axis=1, keepdims=True)
    return {
        "n_values": [int(n) for n in counts],
        "qed_bin_edges": [float(e) for e in bins],
        "counts": matrix.tolist(),
        # Density WITHIN each N, which is what a conditional model is asked to match.
        "conditional_density": (matrix / np.maximum(row_sums, 1)).tolist(),
        "conditional_quantiles": {
            str(q): [float(np.quantile(np.asarray(grouped[n]), q)) for n in counts]
            for q in QUANTILES},
        "molecules_per_n": [int(len(grouped[n])) for n in counts],
    }


def feasibility(grouped: dict[int, list[float]], *, requests: Sequence[tuple[int, float]],
                neighbourhood: float = NEIGHBOURHOOD,
                thin_fraction: float = THIN_TAIL_FRACTION) -> list[dict[str, Any]]:
    """Classify each requested (N, QED) against the EMPIRICAL data at that N.

    `well_represented` : at least `thin_fraction` of molecules at that N lie within
                         +-`neighbourhood` of the request
    `thin_tail`        : some do, but fewer than that
    `outside_support`  : none do
    `n_absent`         : no training molecule has that N at all
    """
    rows = []
    for n, target in requests:
        values = np.asarray(grouped.get(int(n), []), dtype=np.float64)
        if values.size == 0:
            rows.append({"n": int(n), "requested_qed": float(target), "verdict": "n_absent",
                         "molecules_at_n": 0, "molecules_in_window": 0,
                         "fraction_in_window": float("nan")})
            continue
        window = np.abs(values - float(target)) <= neighbourhood
        fraction = float(window.mean())
        if window.sum() == 0:
            verdict = "outside_support"
        elif fraction < thin_fraction:
            verdict = "thin_tail"
        else:
            verdict = "well_represented"
        rows.append({
            "n": int(n), "requested_qed": float(target), "verdict": verdict,
            "molecules_at_n": int(values.size),
            "molecules_in_window": int(window.sum()),
            "fraction_in_window": fraction,
            "window": [float(target) - neighbourhood, float(target) + neighbourhood],
            "observed_range": [float(values.min()), float(values.max())],
            "conditional_mean": float(values.mean()),
            "conditional_q05": float(np.quantile(values, 0.05)),
            "conditional_q95": float(np.quantile(values, 0.95)),
            # How far the request is from the achievable mass, in the data's own units.
            "target_z_at_this_n": float((float(target) - values.mean())
                                        / max(values.std(), 1e-9)),
        })
    return rows


def verify_labels(records: Sequence[dict[str, Any]], *, scaler: ConditionScaler | None = None,
                  sample: int = 2000, seed: int = 0,
                  tolerance: float = 1e-3) -> dict[str, Any]:
    """Recompute QED with RDKit from the STORED GRAPH and compare with the stored label.

    The comparison deliberately rebuilds the molecule from `atom_types` + `bond_matrix` -
    the representation the model is actually trained on - rather than re-parsing the
    original SMILES. A label that matches the SMILES but not the stored graph would be a
    silent mismatch exactly where it matters. The standardised condition vector is
    re-derived from the scaler and compared too.
    """
    rng = np.random.default_rng(int(seed))
    chosen = (rng.choice(len(records), size=min(sample, len(records)), replace=False)
              if sample and sample < len(records) else np.arange(len(records)))
    differences: list[float] = []
    condition_differences: list[float] = []
    unbuildable = 0
    for index in chosen:
        record = records[int(index)]
        atoms = np.asarray(record["atom_types"], dtype=np.int64)
        bonds = np.asarray(record["bond_matrix"], dtype=np.int64)
        try:
            with rdBase.BlockLogs():
                molecule = graph_to_mol_charge_aware(atoms, bonds)
                recomputed = float(QED.qed(molecule))
        except Exception:  # noqa: BLE001 - an unbuildable graph is counted, not fatal
            unbuildable += 1
            continue
        differences.append(abs(recomputed - float(record["qed"])))
        if scaler is not None and record.get("condition") is not None:
            expected = scaler.transform(float(record["qed"]), int(record["n_node"]))
            stored = np.asarray(record["condition"], dtype=np.float64)
            condition_differences.append(float(np.abs(expected - stored).max()))
    array = np.asarray(differences, dtype=np.float64)
    result = {
        "checked": int(len(chosen)),
        "unbuildable_graphs": unbuildable,
        "qed_label_max_abs_difference": float(array.max()) if array.size else float("nan"),
        "qed_label_mean_abs_difference": float(array.mean()) if array.size else float("nan"),
        "qed_labels_matching_within_tolerance":
            float((array <= tolerance).mean()) if array.size else float("nan"),
        "tolerance": float(tolerance),
    }
    if condition_differences:
        condition = np.asarray(condition_differences, dtype=np.float64)
        result.update({
            "condition_max_abs_difference": float(condition.max()),
            "condition_matching_within_tolerance": float((condition <= 1e-4).mean()),
        })
    result["verdict"] = (
        "labels and scaler agree with RDKit on the stored graph"
        if array.size and float(array.max()) <= tolerance
        else "MISMATCH: the stored QED labels do not reproduce from the stored graph")
    return result


def read_generated_qed(path: str | Path, *, decoder: str = "degree") -> list[float]:
    """QED of the VALID molecules in a saved `samples.jsonl`, recomputed if absent."""
    values: list[float] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = (json.loads(line).get(decoder) or {})
            if not entry.get("valid") or not entry.get("smiles"):
                continue
            value = entry.get("qed")
            if value is None:
                with rdBase.BlockLogs():
                    molecule = Chem.MolFromSmiles(entry["smiles"])
                if molecule is None:
                    continue
                value = float(QED.qed(molecule))
            values.append(float(value))
    return values


def compare_distributions(generated: Sequence[float], real: Sequence[float], *,
                          requested_qed: float | None = None,
                          bins: np.ndarray = QED_BINS) -> dict[str, Any]:
    """Generated vs real QED as DISTRIBUTIONS: quantiles, histogram overlap, Wasserstein."""
    g = np.asarray([v for v in generated if np.isfinite(v)], dtype=np.float64)
    r = np.asarray([v for v in real if np.isfinite(v)], dtype=np.float64)
    result: dict[str, Any] = {
        "generated_molecules": int(g.size), "real_molecules": int(r.size),
        "requested_qed": None if requested_qed is None else float(requested_qed),
    }
    if g.size == 0 or r.size == 0:
        result["note"] = "one population is empty; no comparison possible"
        return result
    gh = np.histogram(g, bins=bins, density=False)[0] / g.size
    rh = np.histogram(r, bins=bins, density=False)[0] / r.size
    try:
        from scipy.stats import wasserstein_distance
        earth_mover = float(wasserstein_distance(g, r))
    except Exception:  # noqa: BLE001 - scipy is present on the cluster; stay usable without
        earth_mover = float("nan")
    result.update({
        "generated": {"mean": float(g.mean()), "std": float(g.std()),
                      "quantiles": {str(q): float(np.quantile(g, q)) for q in QUANTILES}},
        "real": {"mean": float(r.mean()), "std": float(r.std()),
                 "quantiles": {str(q): float(np.quantile(r, q)) for q in QUANTILES}},
        # 1.0 = identical histograms, 0.0 = disjoint.
        "histogram_overlap": float(np.minimum(gh, rh).sum()),
        "wasserstein_distance": earth_mover,
        "generated_std_over_real_std": float(g.std() / max(r.std(), 1e-9)),
        "mean_shift": float(g.mean() - r.mean()),
    })
    if requested_qed is not None:
        result["distance_of_generated_mean_from_request"] = float(g.mean() - requested_qed)
        result["real_fraction_within_0p05_of_request"] = float(
            (np.abs(r - requested_qed) <= 0.05).mean())
    return result


def format_support_report(overall: dict[str, dict[str, Any]], rows: Sequence[dict[str, Any]],
                          feasible: Sequence[dict[str, Any]],
                          labels: dict[str, Any]) -> str:
    """The Part A verdict in readable form."""
    lines = ["ZINC250K QED support analysis", "=============================", ""]
    for name, histogram in overall.items():
        lines.append(f"{name}: {histogram['molecules']:,} molecules, QED mean "
                     f"{histogram['mean']:.4f} +- {histogram['std']:.4f}, "
                     f">=0.7 {histogram['fraction_at_least_0p7'] * 100:.1f}%, "
                     f">=0.8 {histogram['fraction_at_least_0p8'] * 100:.1f}%")
    lines += ["", "Per-N support (training split)", ""]
    lines.append("|   N | molecules | QED mean | q05 | q50 | q95 | >=0.7 | >=0.8 |")
    lines.append("|----:|----------:|---------:|----:|----:|----:|------:|------:|")
    for row in rows:
        lines.append(f"| {row['n']:>3} | {row['molecules']:>9,} | {row['qed_mean']:.4f} | "
                     f"{row['q5']:.3f} | {row['q50']:.3f} | {row['q95']:.3f} | "
                     f"{row['fraction_ge_0p7'] * 100:>5.1f}% | "
                     f"{row['fraction_ge_0p8'] * 100:>5.1f}% |")
    lines += ["", "Requested (N, QED) feasibility against the training data", ""]
    lines.append("|   N | requested | verdict | molecules within +-0.05 | share at that N | "
                 "z of target | conditional q05-q95 |")
    lines.append("|----:|----------:|---------|------------------------:|----------------:|"
                 "------------:|--------------------:|")
    for row in feasible:
        if row["verdict"] == "n_absent":
            lines.append(f"| {row['n']:>3} | {row['requested_qed']:.2f} | n_absent | 0 | 0 | "
                         f"N/A | N/A |")
            continue
        lines.append(f"| {row['n']:>3} | {row['requested_qed']:.2f} | {row['verdict']} | "
                     f"{row['molecules_in_window']:,} | "
                     f"{row['fraction_in_window'] * 100:.2f}% | "
                     f"{row['target_z_at_this_n']:+.2f} | "
                     f"{row['conditional_q05']:.3f}-{row['conditional_q95']:.3f} |")
    lines += ["", f"Label check: {labels['verdict']} "
              f"(max |RDKit - stored| = {labels['qed_label_max_abs_difference']:.2e} on "
              f"{labels['checked']:,} molecules rebuilt from their stored graphs)"]
    return "\n".join(lines)


__all__ = [
    "HIGH_QED_THRESHOLDS",
    "QED_BINS",
    "QUANTILES",
    "by_n",
    "compare_distributions",
    "feasibility",
    "format_support_report",
    "n_qed_heatmap",
    "qed_histogram",
    "read_generated_qed",
    "support_table",
    "verify_labels",
]
