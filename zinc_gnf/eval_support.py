"""Shared evaluation helpers for the cluster pipeline (scripts 46-49).

Deliberately free of top-level `numpy`, `torch` and `rdkit` imports: the aggregation and
metric functions here are pure Python over SAVED RECORDS, so the summariser runs on a
login node without a GPU environment and the logic is unit-testable without the heavy
stack. Functions that genuinely need numpy import it locally.

Record contract. `scripts/48` writes one JSON object per ATTEMPT, including failures:

    {"index": int, "n_node": int,              # the REQUESTED atom count
     "requested_qed": float | None,
     "model": str, "cell": str, "repetition": int, "sampling_seed": int,
     "degree": {"valid": bool, "smiles": str|None, "qed": float|None,
                "rings": int|None, "aromatic_rings": int|None,
                "heavy_atoms": int|None, "molecular_weight": float|None,
                "logp": float|None},
     "raw": {...}, "constrained": {...}}

`heavy_atoms` and the chemistry fields are computed ONCE at generation time so no
downstream step needs RDKit. An invalid attempt keeps its row with `valid: false`, which
is what makes the per-attempt denominators honest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

# MoFlow paper section 5.1, verified against the paper and mflow/generate.py.
MOFLOW_SETTINGS = {"count": 10000, "runs": 5, "temperature": 0.85}

MOFLOW_DIFFERENCES = [
    "data representation: we model the density of a FROZEN autoencoder's per-node "
    "embeddings; MoFlow models one-shot atom and bond tensors directly",
    "decoder: ours is a learned decoder plus a bond assembler; MoFlow takes argmax over "
    "its own bond tensor",
    "correction: our 'degree' assembler is an exact degree-constrained MILP that requires "
    "a single fragment and DELETES NOTHING; MoFlow reduces bond orders on over-valent "
    "atoms and then keeps the largest fragment, which can drop atoms and change N",
    "prior: ours is a conditional per-dimension sigma(c) over node embeddings; MoFlow "
    "uses N(0, (t*sigma)^2 I) with a single learned scalar sigma, so the same numeric "
    "temperature does NOT describe the same sampling distribution",
    "atom vocabulary: 17 charge-aware (element, formal charge) types vs MoFlow's 9",
    "novelty reference: we use the training split's stored SMILES; MoFlow rebuilds the "
    "reference by round-tripping its own dataset tensors through construct_mol",
    "runs: 5 runs is SAMPLING repetition from one checkpoint in both cases, so the "
    "reported sd contains no training-seed variability",
]

STRUCTURAL_SIZE_NOTE = (
    "Atom count is STRUCTURAL IN BOTH ARMS: the requested N sets how many node vectors "
    "are drawn and the decoder emits exactly one atom per vector, so size agreement is "
    "plumbing in both models and is NOT evidence that FiLM or the conditional prior "
    "works. The evidence for learned conditioning is the permuted-condition likelihood "
    "gap and the achieved-vs-requested QED response, not size agreement.")

DEFAULT_TOLERANCES = (0.1, 0.01)


# ------------------------------------------------------------------ sizes ---
class FixedCounts:
    """A `NodeCountSampler`-shaped object returning a PRE-DRAWN size sequence.

    Lets both arms receive byte-identical size vectors, so nothing in a comparison can
    differ because the two models drew different molecule sizes.
    """

    def __init__(self, counts) -> None:
        self._counts = [int(c) for c in counts]
        self.values = sorted(set(self._counts))
        self.position = 0

    def __len__(self) -> int:
        return len(self._counts)

    def sample(self, size: int, rng=None):      # noqa: ARG002 - rng deliberately unused
        if self.position + size > len(self._counts):
            raise ValueError(f"requested {size} sizes from position {self.position} but "
                             f"only {len(self._counts)} were pre-drawn")
        chunk = self._counts[self.position:self.position + size]
        self.position += size
        return chunk


# ------------------------------------------------------- run aggregation ---
def run_level_summary(values: Sequence[float]) -> dict[str, Any]:
    """Mean and POPULATION sd across repetitions, matching `np.std`'s default."""
    values = [float(v) for v in values]
    if not values:
        return {"mean": float("nan"), "sd": float("nan"), "runs": 0, "values": []}
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {"mean": mean, "sd": 0.0, "runs": 1, "values": values}
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return {"mean": mean, "sd": variance ** 0.5, "runs": len(values), "values": values}


def aggregate_runs(runs: Sequence[dict[str, Any]], headline: str) -> dict[str, Any]:
    """Run-level mean and sd of the generation-quality metrics."""
    def collect(path):
        out = []
        for entry in runs:
            node = entry["metrics"]["decoders"]
            for key in path:
                node = node[key]
            out.append(float(node))
        return out

    summary: dict[str, Any] = {}
    for label, path in (("raw_validity", ("raw", "strict_validity")),
                        (f"{headline}_validity", (headline, "strict_validity")),
                        ("uniqueness", (headline, "uniqueness")),
                        ("novelty_vs_train", (headline, "novelty_vs_train")),
                        ("nuv", (headline, "nuv"))):
        try:
            summary[label] = run_level_summary(collect(path))
        except KeyError:
            continue
    try:
        summary["atom_count_match_over_all_attempts"] = run_level_summary(
            [float(e["sizes"]["atom_count_match_over_all_attempts"]) for e in runs])
    except KeyError:
        pass
    summary["note"] = ("sd is across SAMPLING repetitions of one trained checkpoint; it "
                       "contains no training-seed variability")
    return summary


# --------------------------------------------------- per-cell QED metrics ---
def qed_cell_metrics(records: Sequence[dict[str, Any]], *, requested_qed: float | None,
                     requested_n: int | None, decoder: str = "degree",
                     tolerances: Sequence[float] = DEFAULT_TOLERANCES,
                     train_smiles: set[str] | None = None) -> dict[str, Any]:
    """Every metric for ONE (model, cell, repetition), with explicit denominators.

    Duplicates are PRESERVED: every per-attempt accuracy number below is computed over
    all scored attempts, not over the deduplicated set. Uniqueness and novelty are
    reported separately and are the only deduplicated quantities.

    Invalid attempts never disappear: `attempted` is the denominator for validity and for
    every `*_yield_over_all_attempts`.
    """
    attempted = len(records)
    entries = [r.get(decoder) or {} for r in records]
    valid = [e for e in entries if e.get("valid") and e.get("smiles")]
    scored = [e for e in valid if e.get("qed") is not None]
    qeds = [float(e["qed"]) for e in scored]

    out: dict[str, Any] = {
        "requested_qed": requested_qed,
        "requested_n": requested_n,
        "decoder": decoder,
        "attempted": attempted,
        "valid": len(valid),
        "scored": len(scored),
        "validity_over_attempts": len(valid) / attempted if attempted else float("nan"),
        "duplicates_preserved": True,
    }
    if qeds:
        mean = sum(qeds) / len(qeds)
        out["achieved_qed_mean"] = mean
        out["achieved_qed_sd"] = (
            (sum((q - mean) ** 2 for q in qeds) / len(qeds)) ** 0.5
            if len(qeds) > 1 else 0.0)
        out["achieved_qed_min"] = min(qeds)
        out["achieved_qed_max"] = max(qeds)
        if requested_qed is not None:
            errors = [abs(q - requested_qed) for q in qeds]
            # MAE is the mean of INDIVIDUAL absolute errors, not |mean - target|
            out["qed_mae"] = sum(errors) / len(errors)
            out["qed_bias"] = mean - requested_qed
            out["qed_rmse"] = (sum(e * e for e in errors) / len(errors)) ** 0.5
            out["cell_mean_abs_error"] = abs(mean - requested_qed)
            for tol in tolerances:
                key = f"{tol:g}".replace(".", "p")
                hits = sum(1 for e in errors if e <= tol)
                out[f"hit_rate_within_{key}_among_scored"] = hits / len(errors)
                out[f"valid_hit_yield_within_{key}_over_all_attempts"] = (
                    hits / attempted if attempted else float("nan"))
    else:
        out["achieved_qed_mean"] = None
        out["achieved_qed_sd"] = None
        if requested_qed is not None:
            out["qed_mae"] = None
            out["qed_bias"] = None
            for tol in tolerances:
                key = f"{tol:g}".replace(".", "p")
                out[f"hit_rate_within_{key}_among_scored"] = None
                out[f"valid_hit_yield_within_{key}_over_all_attempts"] = 0.0

    sizes = [int(e["heavy_atoms"]) for e in valid if e.get("heavy_atoms") is not None]
    if sizes:
        size_mean = sum(sizes) / len(sizes)
        out["achieved_atoms_mean"] = size_mean
        out["achieved_atoms_sd"] = (
            (sum((s - size_mean) ** 2 for s in sizes) / len(sizes)) ** 0.5
            if len(sizes) > 1 else 0.0)
    if requested_n is not None:
        matched = sum(1 for e in valid if e.get("heavy_atoms") == requested_n)
        out["size_match_among_valid"] = matched / len(valid) if valid else None
        out["size_match_over_all_attempts"] = matched / attempted if attempted else None
        out["size_is_structural"] = True

    smiles = [str(e["smiles"]) for e in valid]
    unique = set(smiles)
    out["unique_valid"] = len(unique)
    out["uniqueness_over_valid"] = len(unique) / len(valid) if valid else float("nan")
    if train_smiles is not None:
        novel = len(unique - train_smiles)
        out["novel_unique_valid"] = novel
        out["novelty_over_unique_valid"] = novel / len(unique) if unique else float("nan")
        out["nuv_over_attempts"] = novel / attempted if attempted else float("nan")
    return out


def macro_and_pooled(cells: Sequence[dict[str, Any]], *,
                     tolerances: Sequence[float] = DEFAULT_TOLERANCES) -> dict[str, Any]:
    """MACRO = equal weight per cell. POOLED = every scored attempt weighted equally.

    Kept separate and labelled, because validity collapses with N: pooling silently
    reweights the result toward the small-molecule cells where most attempts survive.
    """
    usable = [c for c in cells if c.get("qed_mae") is not None]
    macro: dict[str, Any] = {"cells": len(cells), "cells_with_scored_molecules": len(usable)}
    pooled: dict[str, Any] = dict(macro)
    if usable:
        macro["qed_mae"] = sum(c["qed_mae"] for c in usable) / len(usable)
        macro["qed_bias"] = sum(c["qed_bias"] for c in usable) / len(usable)
        weights = [c["scored"] for c in usable]
        total = sum(weights)
        pooled["qed_mae"] = (sum(c["qed_mae"] * w for c, w in zip(usable, weights)) / total
                             if total else None)
        pooled["qed_bias"] = (sum(c["qed_bias"] * w for c, w in zip(usable, weights)) / total
                              if total else None)
        pooled["scored_total"] = total
        for tol in tolerances:
            key = f"{tol:g}".replace(".", "p")
            field = f"hit_rate_within_{key}_among_scored"
            yield_field = f"valid_hit_yield_within_{key}_over_all_attempts"
            macro[field] = sum(c[field] for c in usable) / len(usable)
            pooled[field] = (sum(c[field] * w for c, w in zip(usable, weights)) / total
                             if total else None)
            macro[yield_field] = sum(c.get(yield_field, 0.0) for c in cells) / len(cells)
    attempted = sum(c["attempted"] for c in cells)
    for tol in tolerances:
        key = f"{tol:g}".replace(".", "p")
        yield_field = f"valid_hit_yield_within_{key}_over_all_attempts"
        hits = sum(round(c.get(yield_field, 0.0) * c["attempted"]) for c in cells)
        pooled[yield_field] = hits / attempted if attempted else None
    macro["weighting"] = "equal weight per requested cell"
    pooled["weighting"] = ("weighted by scored molecules per cell; shifts toward cells "
                           "with higher validity, which are the small-N cells")
    return {"macro": macro, "pooled": pooled}


# ----------------------------------------------------------- target sets ---
GRID_TARGETS = {"n": (15, 23, 30), "qed": (0.4, 0.6, 0.8)}


def percentile_targets(n_node, qed, *, n_values=GRID_TARGETS["n"],
                       percentiles=(10, 30, 50, 70, 90), min_support=50,
                       support_window=0.05, atom_window=1) -> dict[str, Any]:
    """QED targets at TRAINING percentiles, computed WITHIN each atom-count group.

    Within-group, not global: QED and molecular size are associated (molecular weight is
    one of QED's eight components), so a global percentile would place targets outside the
    support of a given N. Every target carries its own support count and an
    `extrapolation` flag.
    """
    import numpy as np

    counts = np.asarray([int(v) for v in n_node])
    scores = np.asarray([float(v) for v in qed], dtype=float)
    keep = np.isfinite(scores)
    counts, scores = counts[keep], scores[keep]
    rows = []
    for n_target in n_values:
        near = np.abs(counts - int(n_target)) <= int(atom_window)
        group = scores[near]
        for pct in percentiles:
            if group.size == 0:
                rows.append({"n": int(n_target), "percentile": pct, "qed": None,
                             "support": 0, "extrapolation": True,
                             "reason": f"no training molecule within "
                                       f"+/-{atom_window} atoms of N={n_target}"})
                continue
            value = float(np.quantile(group, pct / 100.0))
            support = int(np.sum(np.abs(group - value) <= support_window))
            rows.append({
                "n": int(n_target), "percentile": pct, "qed": round(value, 4),
                "group_molecules": int(group.size), "support": support,
                "extrapolation": bool(support < int(min_support)),
                "reason": (None if support >= int(min_support) else
                           f"only {support} training molecules within "
                           f"+/-{support_window} QED at this N"),
            })
    return {
        "targets": rows,
        "definition": (f"QED quantiles computed WITHIN each atom-count group "
                       f"(+/-{atom_window} atoms); support counted within "
                       f"+/-{support_window} QED; extrapolation when support < "
                       f"{min_support}"),
        "n_values": [int(v) for v in n_values],
        "percentiles": list(percentiles),
    }


def grid_target_support(n_node, qed, *, n_values=GRID_TARGETS["n"],
                        qed_values=GRID_TARGETS["qed"], support_window=0.05,
                        atom_window=1, min_support=50) -> list[dict[str, Any]]:
    """Training support for the fixed 3x3 grid, so thin corners are marked, not assumed."""
    import numpy as np

    counts = np.asarray([int(v) for v in n_node])
    scores = np.asarray([float(v) for v in qed], dtype=float)
    keep = np.isfinite(scores)
    counts, scores = counts[keep], scores[keep]
    rows = []
    for n_target in n_values:
        near = np.abs(counts - int(n_target)) <= int(atom_window)
        group = scores[near]
        for q_target in qed_values:
            support = int(np.sum(np.abs(group - float(q_target)) <= support_window)) \
                if group.size else 0
            rows.append({
                "n": int(n_target), "qed": float(q_target),
                "group_molecules": int(group.size), "support": support,
                "fraction_of_that_n": (support / group.size) if group.size else 0.0,
                "extrapolation": bool(support < int(min_support)),
            })
    return rows


def joint_association(n_node, qed) -> dict[str, Any]:
    """Spearman and Pearson association between N and QED. Independence is NOT assumed."""
    import numpy as np

    counts = np.asarray([int(v) for v in n_node], dtype=float)
    scores = np.asarray([float(v) for v in qed], dtype=float)
    keep = np.isfinite(scores)
    counts, scores = counts[keep], scores[keep]

    def ranks(values):
        order = np.argsort(values, kind="stable")
        out = np.empty_like(order, dtype=float)
        out[order] = np.arange(values.shape[0], dtype=float)
        return out

    rows = []
    for value in sorted(set(int(c) for c in counts)):
        sub = scores[counts == value]
        if sub.size < 20:
            continue
        rows.append({"n": value, "molecules": int(sub.size),
                     "qed_mean": float(sub.mean()), "qed_sd": float(sub.std()),
                     "qed_q05": float(np.quantile(sub, 0.05)),
                     "qed_q50": float(np.quantile(sub, 0.50)),
                     "qed_q95": float(np.quantile(sub, 0.95))})
    return {
        "spearman_n_qed": float(np.corrcoef(ranks(counts), ranks(scores))[0, 1]),
        "pearson_n_qed": float(np.corrcoef(counts, scores)[0, 1]),
        "by_n": rows,
        "note": ("a nonzero association means QED cannot be requested independently of N "
                 "without leaving the training support; molecular weight is one of QED's "
                 "eight components"),
    }


# ------------------------------------------------- diagnostic references ---
def constant_reference(targets: Sequence[float], *, weights: Sequence[float] | None = None,
                       sigma: float = 0.0) -> dict[str, Any]:
    """MAE of a hypothetical request-IGNORING generator. A DIAGNOSTIC, not a model.

    Not a lower bound: a perfect model scores 0.0. The MAE-optimal constant is the
    (weighted) MEDIAN of the targets, not the mean; with `sigma > 0` the reference is a
    constant-MEAN generator whose achieved values are Gaussian, which raises the MAE.
    """
    import math

    targets = [float(t) for t in targets]
    if not targets:
        raise ValueError("no targets")
    weights = ([1.0 / len(targets)] * len(targets) if weights is None
               else [float(w) for w in weights])
    total = sum(weights)
    weights = [w / total for w in weights]

    def expected_abs(centre, target):
        d = abs(centre - target)
        if sigma <= 0:
            return d
        phi = math.exp(-d * d / (2 * sigma * sigma)) / math.sqrt(2 * math.pi)
        cdf = 0.5 * (1 + math.erf((d / sigma) / math.sqrt(2)))
        return 2 * sigma * phi + d * (2 * cdf - 1)

    def score(centre):
        return sum(w * expected_abs(centre, t) for w, t in zip(weights, targets))

    grid = [i / 2000.0 for i in range(2001)]
    best = min(grid, key=score)
    # weighted median, reported so the median-vs-mean distinction is explicit
    order = sorted(zip(targets, weights))
    cumulative, median = 0.0, order[-1][0]
    for value, weight in order:
        cumulative += weight
        if cumulative >= 0.5:
            median = value
            break
    return {
        "kind": "DIAGNOSTIC REFERENCE, not a measured generator and not a lower bound",
        "targets": targets, "weights": weights, "sigma": sigma,
        "optimal_constant": best, "mae_at_optimal_constant": score(best),
        "weighted_median_of_targets": median,
        "mae_at_weighted_median": score(median),
        "weighted_mean_of_targets": sum(w * t for w, t in zip(weights, targets)),
        "mae_at_weighted_mean": score(sum(w * t for w, t in zip(weights, targets))),
        "note": ("a perfect model scores 0.0; this row describes 'ignore the request and "
                 "always answer the same value' and must never appear as a model row"),
    }


def response_fit(requested: Sequence[float], achieved: Sequence[float]) -> dict[str, Any]:
    """Least-squares slope and intercept of achieved on requested, with standard errors.

    Slope near 1 is desirable but not sufficient: it says nothing about bias or spread, so
    `intercept` and the per-sample MAE must be read beside it.
    """
    xs = [float(v) for v in requested]
    ys = [float(v) for v in achieved]
    n = len(xs)
    if n < 3 or len(set(xs)) < 2:
        return {"slope": None, "intercept": None, "n": n,
                "reason": "need >= 3 points at >= 2 distinct requests"}
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    if n > 2:
        s2 = sum(r * r for r in residuals) / (n - 2)
        slope_se = (s2 / sxx) ** 0.5
        intercept_se = (s2 * (1.0 / n + mx * mx / sxx)) ** 0.5
    else:
        slope_se = intercept_se = float("nan")
    syy = sum((y - my) ** 2 for y in ys)
    return {"slope": slope, "slope_se": slope_se, "intercept": intercept,
            "intercept_se": intercept_se, "n": n,
            "r_squared": (sxy * sxy / (sxx * syy)) if syy > 0 else None,
            "note": ("slope near 1 does not prove low error; read it with intercept and "
                     "the per-sample MAE")}


# --------------------------------------------------------------- storage ---
def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Atomic append-safe write of one JSON object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".partial")
    written = 0
    with staging.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")
            written += 1
    import os

    os.replace(staging, path)
    return written


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def already_done(path: str | Path, *, force: bool = False) -> bool:
    """Resumability: a completed artefact is never regenerated unless forced."""
    path = Path(path)
    return bool(path.is_file() and path.stat().st_size > 0 and not force)


def manifest_entry(**fields: Any) -> dict[str, Any]:
    """The provenance block every artefact carries."""
    required = ("flow_checkpoint", "flow_sha256", "decoder_checkpoint", "decoder_sha256",
                "temperature", "sampling_seed", "attempts", "repetitions", "decoder")
    missing = [key for key in required if key not in fields]
    if missing:
        raise ValueError(f"manifest is missing {missing}")
    fields.setdefault("training_seed", None)
    fields.setdefault("training_seed_note",
                      "read from the checkpoint's train_config when present; null means "
                      "the checkpoint did not record it")
    return dict(fields)


__all__ = [
    "DEFAULT_TOLERANCES", "GRID_TARGETS", "MOFLOW_DIFFERENCES", "MOFLOW_SETTINGS",
    "STRUCTURAL_SIZE_NOTE", "FixedCounts", "aggregate_runs", "already_done",
    "constant_reference", "grid_target_support", "joint_association", "macro_and_pooled",
    "manifest_entry", "percentile_targets", "qed_cell_metrics", "read_jsonl",
    "response_fit", "run_level_summary", "write_jsonl",
]
