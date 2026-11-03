"""Did the model hit the QED it was asked for, and is the decode path the bottleneck?

Two independent things, kept apart on purpose.

TARGET ACHIEVEMENT (`qed_target_metrics`, `qed_achievement_table`)
    For each requested (N, QED) cell: the realised QED of the VALID molecules against the
    requested value - MAE, bias, RMSE, the hit rate within a tolerance (default +-0.1) and
    the correlation between requested and realised QED. The correlation is computed
    ACROSS cells, never within one cell: inside a single cell the request is constant, so
    a within-cell correlation is undefined, and reporting one would be meaningless. It is
    also computed per N, because N and QED are correlated in ZINC and a model that only
    learned N would still show a pooled correlation (bigger molecules have different QED).
    `qed_independent_of_n` is the number that answers that: the correlation of realised
    with requested QED *within* each N, averaged over N.

DECODE CEILING (`decode_ceiling`)
    The same three decoders run on REAL validation embeddings from the frozen
    autoencoder. This is the best any sampler could do through this decode path, so it
    says whether 0% raw validity is a decoding defect or a modelling one: if real
    embeddings decode to valid, aromatic molecules with the raw argmax, the raw bond
    prediction works and the flow's embeddings are what is wrong. Cheap (one encode +
    decode pass over a validation subset) and reported beside every generated row, so a
    generated raw validity is never read against an imaginary 100%.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from zinc_gnf.dataset import MolecularGraphDataset
from zinc_gnf.embedding_populations import decode_population, real_population
from zinc_gnf.molecular_profile import profile_molecules

DEFAULT_QED_TOLERANCE = 0.1


def _correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation, NaN when it is undefined (fewer than two distinct values)."""
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 2 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def qed_target_metrics(records: Sequence[dict[str, Any]], *, requested_qed: float,
                       decoder: str = "degree",
                       tolerance: float = DEFAULT_QED_TOLERANCE) -> dict[str, Any]:
    """Realised QED of one cell's VALID molecules against the requested value.

    Invalid samples have no QED, so they cannot enter an error metric; their count is
    reported instead (`valid_molecules` / `samples`) so the achievement numbers are never
    read as if they described the whole sample.
    """
    values: list[float] = []
    for record in records:
        entry = record.get(decoder) or {}
        if entry.get("valid") and entry.get("qed") is not None:
            values.append(float(entry["qed"]))
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"requested_qed": float(requested_qed), "samples": len(records),
                "valid_molecules": 0, "realised_qed_mean": float("nan"),
                "mae": float("nan"), "bias": float("nan"), "rmse": float("nan"),
                "hit_rate": float("nan"), "tolerance": float(tolerance),
                "note": "no valid molecule in this cell; target achievement is undefined"}
    error = array - float(requested_qed)
    return {
        "requested_qed": float(requested_qed),
        "samples": len(records),
        "valid_molecules": int(array.size),
        "realised_qed_mean": float(array.mean()),
        "realised_qed_median": float(np.median(array)),
        "realised_qed_std": float(array.std()),
        "mae": float(np.abs(error).mean()),
        "bias": float(error.mean()),                      # signed: + means too drug-like
        "rmse": float(np.sqrt((error ** 2).mean())),
        "hit_rate": float((np.abs(error) <= float(tolerance)).mean()),
        "tolerance": float(tolerance),
    }


def qed_achievement_table(cells: Sequence[dict[str, Any]], *,
                          tolerance: float = DEFAULT_QED_TOLERANCE) -> dict[str, Any]:
    """Aggregate per-cell achievement into the numbers that answer "is QED controlled?".

    Each cell must carry `n`, `requested_qed` and the `qed_target_metrics` result.
    """
    rows = [c for c in cells if c["metrics"]["valid_molecules"] > 0]
    if not rows:
        return {"cells": list(cells), "overall": None,
                "note": "no cell produced a valid molecule; QED control cannot be assessed"}
    requested = [c["requested_qed"] for c in rows]
    realised = [c["metrics"]["realised_qed_mean"] for c in rows]
    weights = np.asarray([c["metrics"]["valid_molecules"] for c in rows], dtype=np.float64)
    mae = np.asarray([c["metrics"]["mae"] for c in rows], dtype=np.float64)
    hit = np.asarray([c["metrics"]["hit_rate"] for c in rows], dtype=np.float64)

    by_n: dict[int, dict[str, Any]] = {}
    for n in sorted({int(c["n"]) for c in rows}):
        same = [c for c in rows if int(c["n"]) == n]
        by_n[n] = {
            "cells": len(same),
            "correlation_requested_vs_realised": _correlation(
                [c["requested_qed"] for c in same],
                [c["metrics"]["realised_qed_mean"] for c in same]),
            # Slope of realised on requested: 1.0 would be perfect following, 0.0 none.
            "response_slope": _slope([c["requested_qed"] for c in same],
                                     [c["metrics"]["realised_qed_mean"] for c in same]),
            "mae": float(np.mean([c["metrics"]["mae"] for c in same])),
            "hit_rate": float(np.mean([c["metrics"]["hit_rate"] for c in same])),
            "realised_range": [float(min(c["metrics"]["realised_qed_mean"] for c in same)),
                               float(max(c["metrics"]["realised_qed_mean"] for c in same))],
        }
    within_n = [v["correlation_requested_vs_realised"] for v in by_n.values()
                if np.isfinite(v["correlation_requested_vs_realised"])]
    return {
        "cells": list(cells),
        "by_n": by_n,
        "overall": {
            "cells": len(rows),
            "valid_molecules": int(weights.sum()),
            "mae_unweighted": float(mae.mean()),
            "mae_weighted": float((mae * weights).sum() / weights.sum()),
            "hit_rate_unweighted": float(hit.mean()),
            "hit_rate_weighted": float((hit * weights).sum() / weights.sum()),
            "correlation_pooled": _correlation(requested, realised),
            "response_slope_pooled": _slope(requested, realised),
            # The honest headline: QED following measured WITHIN each N, so it cannot be
            # explained by N alone.
            "qed_independent_of_n": float(np.mean(within_n)) if within_n else float("nan"),
            "tolerance": float(tolerance),
        },
        "note": ("correlations are across requested values, never within a cell (the "
                 "request is constant there). 'qed_independent_of_n' averages the "
                 "within-N correlations, which is the number that survives the N-QED "
                 "correlation present in ZINC itself."),
    }


def _slope(x: Sequence[float], y: Sequence[float]) -> float:
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 2 or np.allclose(a, a[0]):
        return float("nan")
    return float(np.polyfit(a, b, 1)[0])


def decode_ceiling(autoencoder, dataset: MolecularGraphDataset, *, indices: Sequence[int],
                   decoders: Sequence[str] = ("raw", "constrained", "degree"),
                   device: str = "cpu", batch_size: int = 128) -> dict[str, Any]:
    """Run the SAME decoders on REAL validation embeddings: the decode path's own ceiling.

    If raw validity is high here and ~0 for generated embeddings, the raw bond prediction
    is not broken and the gap is a property of the sampled embeddings - i.e. a modelling
    problem, not a decoding bug. If it is low here too, the raw argmax itself is the
    bottleneck and no amount of flow training would fix raw validity.
    """
    population = real_population(autoencoder, dataset, indices, device=device,
                                 batch_size=batch_size, name="real_validation")
    records = decode_population(autoencoder, population, device=device,
                                batch_size=batch_size, decoders=tuple(decoders))
    result: dict[str, Any] = {"molecules": len(records), "decoders": {}}
    for name in decoders:
        entries = [r[name] for r in records if name in r]
        if not entries:
            continue
        valid = [e for e in entries if e.get("valid") and e.get("smiles")]
        profile = profile_molecules([e["smiles"] for e in valid], name=name, scaffolds=False)
        fallback_entries = [e for e in entries if "fallback" in e]
        result["decoders"][name] = {
            "strict_validity": len(valid) / len(entries),
            "fallback_rate": (sum(1 for e in fallback_entries if e.get("fallback"))
                              / len(fallback_entries)) if fallback_entries else float("nan"),
            "qed_mean": profile["qed"]["mean"],
            "aromatic_rings_mean": profile["aromatic_rings"]["mean"],
            "aromatic_ring_fraction": profile["aromatic_ring_fraction"],
            "rings_mean": profile["rings"]["mean"],
        }
    result["interpretation"] = (
        "ceiling of the decode path on real embeddings. Generated raw validity must be "
        "read against the 'raw' row here, not against 100%.")
    return result


def format_qed_achievement(table: dict[str, Any], *, tolerance: float = DEFAULT_QED_TOLERANCE
                           ) -> str:
    """Readable QED target-achievement report."""
    lines = ["QED target achievement", "======================", ""]
    if table.get("overall") is None:
        return "\n".join(lines + [str(table.get("note"))])
    overall = table["overall"]
    lines.append(f"cells with valid molecules: {overall['cells']} "
                 f"({overall['valid_molecules']:,} molecules)")
    lines.append(f"MAE                       {overall['mae_weighted']:.4f} "
                 f"(unweighted {overall['mae_unweighted']:.4f})")
    lines.append(f"hit rate within +-{tolerance:g}      "
                 f"{overall['hit_rate_weighted'] * 100:.1f}% "
                 f"(unweighted {overall['hit_rate_unweighted'] * 100:.1f}%)")
    lines.append(f"correlation (pooled)      {overall['correlation_pooled']:.4f}")
    lines.append(f"response slope (pooled)   {overall['response_slope_pooled']:.4f}  "
                 f"(1.0 = perfect following, 0.0 = none)")
    lines.append(f"within-N correlation      {overall['qed_independent_of_n']:.4f}  "
                 f"<- QED control that N alone cannot explain")
    lines.append("")
    lines.append("| N | cells | within-N correlation | slope | MAE | hit rate | realised range |")
    lines.append("|---:|------:|---------------------:|------:|----:|---------:|---------------:|")
    for n, row in table["by_n"].items():
        lines.append(f"| {n} | {row['cells']} | "
                     f"{row['correlation_requested_vs_realised']:.4f} | "
                     f"{row['response_slope']:.4f} | {row['mae']:.4f} | "
                     f"{row['hit_rate'] * 100:.1f}% | "
                     f"{row['realised_range'][0]:.3f}-{row['realised_range'][1]:.3f} |")
    lines.append("")
    lines.append("Per-cell errors are computed over VALID molecules only (an invalid sample "
                 "has no QED), so each cell's valid count is reported beside it.")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_QED_TOLERANCE",
    "decode_ceiling",
    "format_qed_achievement",
    "qed_achievement_table",
    "qed_target_metrics",
]
