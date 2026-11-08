#!/usr/bin/env python3
"""QED support, cross-decoder label agreement, and best-of-k reranking.

READ-ONLY AND NO MODEL IS LOADED. This reads `samples.jsonl` files that already exist,
recomputes every chemistry number from the stored SMILES with RDKit, and writes two files
into --output-dir. No generation, no training, no checkpoint, no flow, no autoencoder --
torch is never imported. Nothing outside --output-dir is written.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
Best-of-k reranking is INFERENCE-TIME POSTPROCESSING. It is not intrinsic conditional
generation and must never be reported as such: the model is unchanged, and the QED
improvement is produced by a selector that reads RDKit QED after the fact.

It is also NOT a universal upper bound on what a fine-tuned conditional generator could
reach. It is a FINITE-CANDIDATE PERFORMANCE REFERENCE: it says what is achievable by
selecting among the candidates this checkpoint actually produced, at a stated budget. A
retrained model can place mass where the saved pool happens to be thin, so a weak rerank
result constrains -- but does not by itself refute -- a relabeling proposal.

The three questions it answers, each of which was previously an assumption:

  SUPPORT      what range of ACTUAL decoded QED does each checkpoint already produce at
               each N? Quantiles, IQR and the mass above 0.75, stratified by decoder,
               because the achievable range is what any relabeling scheme would have to
               train on.
  LABEL        does the SAME generated embedding get substantially different QED under
               `raw`, `constrained` and `degree`? If it does, "the QED of a generated
               embedding" is not a well-defined label but a decoder-indexed one. Measured
               per embedding on jointly valid, jointly scorable cases, so the comparison
               is within-sample and not a difference of population means.
  RERANK       what does best-of-k buy, at what selection cost, and what does it cost in
               validity, aromaticity, uniqueness, novelty and diversity?

SELECTION EFFECTS, STATED PLAINLY
---------------------------------
Reranking selects on the very quantity it then reports, so the post-rerank QED mean and
MAE are optimistically biased BY CONSTRUCTION. That bias is the mechanism, not an error.
The honest reading is therefore: a QED gain is expected and uninformative on its own;
what is informative is (a) the selection cost in candidates per accepted molecule, and
(b) what happens to the metrics rerank does NOT optimise -- validity, aromaticity,
uniqueness, novelty, diversity -- because those cannot be bought by selection.

THE PARTITION PROTOCOL
----------------------
Candidates within one requested condition are shuffled and split into DISJOINT groups of
k; each group contributes its own best candidate. A pool of 100 therefore yields
floor(100/k) accepted molecules: 100 at k=1, and only 5 at k=20. Candidates are never
reused across groups, and candidates from other requested conditions are never eligible.

Repeating over several random permutations gives a PROTOCOL SPREAD, reported as min/max
across partitions. That is NOT a confidence interval: every permutation reuses the same
saved candidates, so the trials are not independent experiments. The independent-molecule
ceiling, floor(pool / k), is reported alongside every k so the thinning is visible.

k = 1 is exactly the native, unreranked baseline (every scorable candidate is its own
group), so the native comparison needs no separate code path and cannot drift from it.

Exit codes: 0 ok; 2 missing input; 3 provenance mismatch; 4 bad arguments; 1 unexpected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from rdkit import Chem, rdBase  # noqa: E402
from rdkit.Chem import QED, rdMolDescriptors  # noqa: E402

from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402

DECODERS = ("raw", "constrained", "degree")
QUANTILES = (5, 10, 25, 50, 75, 90, 95)


class ProvenanceError(ValueError):
    """A saved cell disagrees with the condition its directory name claims.

    Its own class, not a bare ValueError, because `json.JSONDecodeError` is also a
    ValueError and a corrupt sample file must not be reported as a provenance mismatch.
    """

HIGH_QED_THRESHOLDS = (0.7, 0.75, 0.8)
DISAGREEMENT_THRESHOLDS = (0.05, 0.10, 0.20)
# Tolerance for the independence check between the stored per-sample QED and our own
# recomputation from the stored SMILES. A difference above this is reported, not smoothed.
RECOMPUTE_TOLERANCE = 1e-4

# Precommitted numbers for the relabeling reassessment. These are EVIDENCE INPUTS to a
# judgement, not an automatic verdict: failing one does not refute relabeling on its own,
# and the report says which of them a retrained model could plausibly escape.
SUPPORT_MIN_FRACTION_HIGH = 0.05      # fraction of valid samples with QED >= 0.75
SUPPORT_MIN_IQR = 0.15                # within-N interquartile range of decoded QED
LABEL_MAX_MEDIAN_DISAGREEMENT = 0.05  # median per-embedding |dQED| across decoders
RERANK_TARGET_MAE = 0.05              # |mean decoded QED - request| at k = 10
DIVERSITY_RETENTION = 0.90            # fraction of native internal diversity to keep


# ------------------------------------------------------------------- statistics ---
def summarise_values(values) -> dict[str, Any]:
    """Quantiles, IQR and high-QED mass. No distributional assumption anywhere."""
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                       dtype=np.float64)
    if array.size == 0:
        return {"molecules": 0}
    percentiles = np.percentile(array, QUANTILES)
    block: dict[str, Any] = {
        "molecules": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "quantiles": {f"p{q}": float(v) for q, v in zip(QUANTILES, percentiles)},
        "iqr": float(np.percentile(array, 75) - np.percentile(array, 25)),
    }
    for threshold in HIGH_QED_THRESHOLDS:
        key = f"fraction_at_least_{str(threshold).replace('.', 'p')}"
        block[key] = float((array >= threshold).mean())
    return block


def mean_or_none(values):
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                       dtype=np.float64)
    return float(array.mean()) if array.size else None


def internal_diversity_of(smiles) -> float | None:
    """`1 - mean pairwise Tanimoto`, via the repository's existing definition.

    Imported lazily because `zinc_gnf.metrics_generation` pulls in `zinc_gnf.generation`
    and therefore torch, which this script otherwise does not need.
    """
    unique = sorted({s for s in smiles if s})
    if len(unique) < 2:
        return None
    from zinc_gnf.metrics_generation import internal_diversity
    return float(internal_diversity(unique)["internal_diversity"])


# -------------------------------------------------------------------- ingestion ---
def cell_directory(root: Path, *, temperature: float, n: int, qed: float) -> Path:
    """The layout `scripts/13_generate_nqed.py` writes: `T{temp}_N{n}_QED{qed}`."""
    return root / f"T{temperature:g}_N{n}_QED{qed:g}"


def score_entry(entry: dict[str, Any] | None) -> dict[str, Any]:
    """Per-decoder chemistry, RECOMPUTED from the stored SMILES rather than trusted.

    `samples.jsonl` already carries `qed`, but recomputing keeps this analysis
    independent of the generating run (the same choice `11_final_uncond_report.py`
    makes) and lets the two be compared. Aromatic ring count is not stored at all.
    """
    entry = entry or {}
    row: dict[str, Any] = {
        "valid": bool(entry.get("valid")),
        "sanitizable": bool(entry.get("sanitizable")),
        "connected": bool(entry.get("connected")),
        "fallback": bool(entry.get("fallback", False)),
        "smiles": entry.get("smiles"),
        "qed_stored": entry.get("qed"),
        "qed": None, "aromatic_rings": None, "rings": None, "heavy_atoms": None,
        "scorable": False,
    }
    if not (row["valid"] and row["smiles"]):
        return row
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(str(row["smiles"]))
    if molecule is None:
        return row
    row["qed"] = float(QED.qed(molecule))
    row["aromatic_rings"] = float(rdMolDescriptors.CalcNumAromaticRings(molecule))
    row["rings"] = float(molecule.GetRingInfo().NumRings())
    row["heavy_atoms"] = float(molecule.GetNumHeavyAtoms())
    row["scorable"] = True
    return row


def load_cell(directory: Path, *, n: int, qed: float, decoders) -> dict[str, Any]:
    """One requested condition: scored candidates per decoder, plus provenance."""
    samples_path = directory / "samples.jsonl"
    if not samples_path.is_file():
        raise FileNotFoundError(
            f"{samples_path} not found. The saved grid must have been written with "
            f"--save-samples; this script never regenerates samples.")
    metrics_path = directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
    requested = metrics.get("requested") or {}
    if requested:
        # provenance: the directory name and the recorded request must agree, otherwise
        # the cells could be silently mislabelled and every per-condition number is wrong
        if int(requested.get("n", n)) != int(n) or not np.isclose(
                float(requested.get("qed", qed)), float(qed), atol=1e-9):
            raise ProvenanceError(
                f"{directory.name} records requested n={requested.get('n')} "
                f"qed={requested.get('qed')} but is named for n={n} qed={qed}")

    candidates: dict[str, list] = {d: [] for d in decoders}
    requested_sizes: list[int] = []
    rows = 0
    with samples_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            rows += 1
            requested_sizes.append(int(record.get("n_node", n)))
            for decoder in decoders:
                candidates[decoder].append(score_entry(record.get(decoder)))
    if rows == 0:
        raise FileNotFoundError(f"{samples_path} is empty")

    recompute_gap = 0.0
    for decoder in decoders:
        for row in candidates[decoder]:
            if row["scorable"] and row["qed_stored"] is not None:
                recompute_gap = max(recompute_gap,
                                    abs(float(row["qed_stored"]) - row["qed"]))
    return {
        "directory": str(directory),
        "requested_n": int(n),
        "requested_qed": float(qed),
        "candidates_in_pool": rows,
        "requested_n_consistent": bool(set(requested_sizes) == {int(n)}),
        "flow_checkpoint": metrics.get("flow_checkpoint"),
        "flow_step": metrics.get("flow_step"),
        "autoencoder_sha256": str((metrics.get("autoencoder") or {}).get("sha256", ""))[:16],
        "max_stored_vs_recomputed_qed_gap": float(recompute_gap),
        # an independence check on the generating run, not a correctness requirement of
        # this script: our recomputation is the source of truth either way
        "stored_qed_matches_recomputation": bool(recompute_gap <= RECOMPUTE_TOLERANCE),
        "candidates": candidates,
    }


# ---------------------------------------------------------------------- support ---
def support_block(rows) -> dict[str, Any]:
    """Validity-stratified QED support for one (checkpoint, decoder, N) group."""
    total = len(rows)
    valid = [r for r in rows if r["valid"]]
    scorable = [r for r in rows if r["scorable"]]
    block: dict[str, Any] = {
        "candidates": total,
        "valid": len(valid),
        "scorable": len(scorable),
        "validity": len(valid) / max(1, total),
        "scorable_fraction": len(scorable) / max(1, total),
        "fallback_rate": sum(1 for r in rows if r["fallback"]) / max(1, total),
        "qed": summarise_values([r["qed"] for r in scorable]),
        "aromatic_rings_mean": mean_or_none([r["aromatic_rings"] for r in scorable]),
        "heavy_atoms_mean": mean_or_none([r["heavy_atoms"] for r in scorable]),
    }
    smiles = [r["smiles"] for r in scorable if r["smiles"]]
    block["unique_molecules"] = len(set(smiles))
    block["uniqueness_over_scorable"] = (len(set(smiles)) / len(smiles)) if smiles else None
    return block


# --------------------------------------------------------- decoder disagreement ---
def disagreement_block(cells, decoders) -> dict[str, Any]:
    """Per-EMBEDDING QED difference between decoders, on jointly scorable cases.

    Within-sample by construction: index i is the same generated embedding in every
    decoder's list, because `decode_embeddings` writes one record per sample with one
    entry per decoder. Two separate effects are reported, because they are different
    problems: whether a label EXISTS under both decoders, and whether its VALUE agrees.
    """
    result: dict[str, Any] = {"pairs": {}}
    for i, left in enumerate(decoders):
        for right in decoders[i + 1:]:
            deltas, identical, both, only_left, only_right, neither = [], 0, 0, 0, 0, 0
            for cell in cells:
                a_rows, b_rows = cell["candidates"][left], cell["candidates"][right]
                for a, b in zip(a_rows, b_rows):
                    if a["scorable"] and b["scorable"]:
                        both += 1
                        deltas.append(abs(a["qed"] - b["qed"]))
                        identical += int(a["smiles"] == b["smiles"])
                    elif a["scorable"]:
                        only_left += 1
                    elif b["scorable"]:
                        only_right += 1
                    else:
                        neither += 1
            array = np.asarray(deltas, dtype=np.float64)
            entry: dict[str, Any] = {
                "jointly_scorable": both,
                "scorable_only_under_first": only_left,
                "scorable_only_under_second": only_right,
                "scorable_under_neither": neither,
                "label_exists_under_exactly_one": only_left + only_right,
                "identical_molecule_fraction": (identical / both) if both else None,
            }
            if array.size:
                entry["absolute_qed_difference"] = {
                    "mean": float(array.mean()),
                    "median": float(np.median(array)),
                    "quantiles": {f"p{q}": float(v) for q, v in
                                  zip(QUANTILES, np.percentile(array, QUANTILES))},
                    "max": float(array.max()),
                }
                for threshold in DISAGREEMENT_THRESHOLDS:
                    key = f"fraction_above_{str(threshold).replace('.', 'p')}"
                    entry["absolute_qed_difference"][key] = float((array > threshold).mean())
            result["pairs"][f"{left}_vs_{right}"] = entry
    result["note"] = (
        "Computed per generated embedding, not between population means. "
        "`label_exists_under_exactly_one` counts embeddings that one decoder scores and "
        "the other does not: relabeling would have to drop or arbitrate those.")
    return result


# ----------------------------------------------------------------------- rerank ---
def rerank_once(rows, *, target: float, k: int, order) -> dict[str, Any]:
    """One DISJOINT partition into groups of k; each group yields its own best candidate.

    Groups are consumed whole, so every accepted molecule costs k candidates whether or
    not the group contained a scorable one. A group with no scorable candidate yields
    nothing and is counted: that is part of the selection cost, not a free retry.
    """
    groups = len(order) // k
    accepted, empty = [], 0
    for index in range(groups):
        chunk = [rows[j] for j in order[index * k:(index + 1) * k]]
        scorable = [r for r in chunk if r["scorable"]]
        if not scorable:
            empty += 1
            continue
        accepted.append(min(scorable, key=lambda r: abs(r["qed"] - target)))
    consumed = groups * k
    smiles = [r["smiles"] for r in accepted if r["smiles"]]
    unique = sorted(set(smiles))
    block: dict[str, Any] = {
        "groups_formed": groups,
        "candidates_consumed": consumed,
        "candidates_discarded_as_remainder": len(order) - consumed,
        "empty_groups": empty,
        "accepted": len(accepted),
        "candidates_per_accepted": (consumed / len(accepted)) if accepted else None,
        "qed_mean": mean_or_none([r["qed"] for r in accepted]),
        "qed_mae_vs_target": mean_or_none([abs(r["qed"] - target) for r in accepted]),
        "aromatic_rings_mean": mean_or_none([r["aromatic_rings"] for r in accepted]),
        "heavy_atoms_mean": mean_or_none([r["heavy_atoms"] for r in accepted]),
        "unique_molecules": len(unique),
        "uniqueness": (len(unique) / len(accepted)) if accepted else None,
        "unique_smiles": unique,
    }
    return block


def rerank_cell(rows, *, target: float, k: int, trials: int, seed: int,
                train_smiles: set[str] | None) -> dict[str, Any]:
    """Best-of-k over several disjoint partitions. Spread is PROTOCOL, not sampling."""
    pool = len(rows)
    if k > pool:
        return {"k": k, "skipped": f"k={k} exceeds the {pool}-candidate pool"}
    # k = 1 makes every candidate its own group, so the partition is irrelevant and one
    # trial is exact. Running more would only duplicate identical numbers.
    effective_trials = 1 if k == 1 else max(1, int(trials))
    per_trial = []
    for trial in range(effective_trials):
        generator = np.random.default_rng(seed + 10_000 * trial + k)
        order = generator.permutation(pool).tolist() if k > 1 else list(range(pool))
        per_trial.append(rerank_once(rows, target=target, k=k, order=order))

    def across(key):
        values = [t[key] for t in per_trial if t.get(key) is not None]
        if not values:
            return None
        return {"mean": float(np.mean(values)), "min": float(np.min(values)),
                "max": float(np.max(values))}

    pooled_unique = sorted({s for t in per_trial for s in t["unique_smiles"]})
    novelty = None
    if train_smiles is not None and pooled_unique:
        novelty = float(sum(1 for s in pooled_unique if s not in train_smiles)
                        / len(pooled_unique))
    block: dict[str, Any] = {
        "k": k,
        "partitions": effective_trials,
        "is_native_baseline": k == 1,
        "independent_accepted_ceiling": pool // k,
        "accepted_per_partition": across("accepted"),
        "candidates_per_accepted": across("candidates_per_accepted"),
        "empty_groups": across("empty_groups"),
        "qed_mean": across("qed_mean"),
        "qed_mae_vs_target": across("qed_mae_vs_target"),
        "aromatic_rings_mean": across("aromatic_rings_mean"),
        "heavy_atoms_mean": across("heavy_atoms_mean"),
        "uniqueness": across("uniqueness"),
        "novelty_vs_train_over_pooled_unique": novelty,
        "internal_diversity_over_pooled_unique": internal_diversity_of(pooled_unique),
        "pooled_unique_molecules": len(pooled_unique),
        "spread_note": ("min/max are across disjoint partitions of THE SAME saved "
                        "candidates. They are a protocol spread, not a confidence "
                        "interval over independent samples."),
    }
    return block


def responsiveness(cells_by_target, *, k: int) -> dict[str, Any]:
    """QED response at FIXED N: highest request minus lowest, plus monotonicity."""
    targets = sorted(cells_by_target)
    means = []
    for target in targets:
        entry = (cells_by_target[target].get("rerank") or {}).get(str(k)) or {}
        means.append((entry.get("qed_mean") or {}).get("mean"))
    if any(v is None for v in means) or len(means) < 2:
        return {"k": k, "note": "not every requested QED has a scorable result"}
    deltas = [means[i + 1] - means[i] for i in range(len(means) - 1)]
    return {
        "k": k,
        "requested": targets,
        "mean_qed_per_request": means,
        "delta_low_to_high": means[-1] - means[0],
        "monotonic_increasing": all(d > 0 for d in deltas),
        "step_deltas": deltas,
        "note": ("difference of CELL MEANS at fixed N. Not a per-molecule paired "
                 "statistic: the saved cells are separate generation calls, so a given "
                 "molecule does not appear under two requests."),
    }


# ----------------------------------------------------------------------- verdict ---
def verdict(payload, *, decoder: str) -> dict[str, Any]:
    """Mechanical evaluation of the precommitted numbers. Evidence, not a decision."""
    lines: dict[str, Any] = {"decoder": decoder, "thresholds": {
        "support_min_fraction_at_least_0p75": SUPPORT_MIN_FRACTION_HIGH,
        "support_min_iqr": SUPPORT_MIN_IQR,
        "label_max_median_disagreement": LABEL_MAX_MEDIAN_DISAGREEMENT,
        "rerank_target_mae_at_k10": RERANK_TARGET_MAE,
        "diversity_retention": DIVERSITY_RETENTION,
    }, "by_checkpoint": {}}
    for label, run in payload["runs"].items():
        checks: dict[str, Any] = {"by_n": {}}
        for n, block in run["support_by_n"].items():
            entry = block.get(decoder) or {}
            qed = entry.get("qed") or {}
            checks["by_n"][n] = {
                "fraction_at_least_0p75": qed.get("fraction_at_least_0p75"),
                "iqr": qed.get("iqr"),
                "support_sufficient": bool(
                    (qed.get("fraction_at_least_0p75") or 0.0) >= SUPPORT_MIN_FRACTION_HIGH
                    and (qed.get("iqr") or 0.0) >= SUPPORT_MIN_IQR),
                "median": (qed.get("quantiles") or {}).get("p50"),
                "molecules": qed.get("molecules"),
            }
        pair = (run.get("decoder_disagreement", {}).get("pairs", {})
                .get("constrained_vs_degree") or {})
        median = (pair.get("absolute_qed_difference") or {}).get("median")
        checks["label_median_disagreement_constrained_vs_degree"] = median
        checks["label_well_defined"] = (
            bool(median is not None and median < LABEL_MAX_MEDIAN_DISAGREEMENT))
        mae_at_10, diversity_ratio = [], []
        for n, by_target in run["cells"].items():
            for target, cell in by_target.items():
                native = (cell.get("rerank") or {}).get("1") or {}
                ten = (cell.get("rerank") or {}).get("10") or {}
                value = (ten.get("qed_mae_vs_target") or {}).get("mean")
                if value is not None:
                    mae_at_10.append(value)
                base = native.get("internal_diversity_over_pooled_unique")
                after = ten.get("internal_diversity_over_pooled_unique")
                if base and after:
                    diversity_ratio.append(after / base)
        checks["rerank_k10_mean_mae"] = mean_or_none(mae_at_10)
        checks["rerank_k10_hits_target_mae"] = bool(
            mae_at_10 and max(mae_at_10) <= RERANK_TARGET_MAE)
        checks["rerank_k10_diversity_retention"] = mean_or_none(diversity_ratio)
        checks["rerank_k10_holds_diversity"] = bool(
            diversity_ratio and min(diversity_ratio) >= DIVERSITY_RETENTION)
        lines["by_checkpoint"][label] = checks
    lines["interpretation"] = (
        "These are EVIDENCE INPUTS, not an automatic GO/NO-GO. `support_sufficient` says "
        "whether this checkpoint's own output already spans enough QED at that N for "
        "relabeling to have training signal there; a failure means relabeling would have "
        "little to learn from in the HIGH-QED region, which a retrained model could in "
        "principle escape and this measurement cannot rule out. `label_well_defined` is "
        "the stronger constraint: if the same embedding gets materially different QED "
        "from different decoders, the relabeling target itself is decoder-indexed, and "
        "no amount of retraining fixes that. The rerank numbers are a finite-candidate "
        "reference at a stated budget, and the QED gain there is selection, by design.")
    return lines


# -------------------------------------------------------------------------- CLI ---
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True,
                        help="label=path to a saved (N,QED) grid, repeatable, e.g. "
                             "early=outputs/samples/qedcontrast_seed3_early_paired")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir",
                        default="data/processed/zinc250k_charge_aware_moflow_fixed",
                        help="only used for novelty; omit with --skip-novelty")
    parser.add_argument("--skip-novelty", action="store_true")
    parser.add_argument("--target-n", default="15,23,30")
    parser.add_argument("--qed-values", default="0.4,0.6,0.8")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--decoders", default=",".join(DECODERS))
    parser.add_argument("--headline", default="degree")
    parser.add_argument("--k-values", default="1,2,5,10,20")
    parser.add_argument("--partitions", type=int, default=5,
                        help="random disjoint partitions per (cell, k); a protocol "
                             "spread, not independent repetitions")
    parser.add_argument("--seed", type=int, default=20261108)
    return parser.parse_args(argv)


def load_train_smiles(data_dir: str) -> set[str]:
    """Lazy, because `zinc_gnf.dataset` imports torch and nothing else here needs it."""
    from zinc_gnf.dataset import load_training_splits
    train_records, _, _ = load_training_splits(data_dir)
    return {str(r["smiles"]) for r in train_records}


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        targets_n = [int(v) for v in args.target_n.split(",") if v.strip()]
        qed_values = [float(v) for v in args.qed_values.split(",") if v.strip()]
        decoders = tuple(d.strip() for d in args.decoders.split(",") if d.strip())
        k_values = sorted({int(v) for v in args.k_values.split(",") if v.strip()})
        runs = {}
        for item in args.run:
            if "=" not in item:
                raise ValueError(f"--run must be label=path, got {item!r}")
            label, path = item.split("=", 1)
            if not label.strip() or not path.strip():
                raise ValueError(f"--run must be label=path, got {item!r}")
            runs[label.strip()] = Path(path.strip())
        unknown = set(decoders) - set(DECODERS)
        if unknown:
            raise ValueError(f"unknown decoders {sorted(unknown)}; known {list(DECODERS)}")
        if args.headline not in decoders:
            raise ValueError(f"--headline {args.headline} is not among --decoders")
        if not targets_n or len(qed_values) < 2:
            raise ValueError("need at least one N and two requested QED values")
        if min(k_values) < 1:
            raise ValueError("every k must be >= 1")
        if 1 not in k_values:
            raise ValueError("k=1 is the native baseline and must be included")
        if args.partitions < 1:
            raise ValueError("--partitions must be >= 1")
        if not runs:
            raise ValueError("at least one --run is required")
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4

    started = time.perf_counter()
    try:
        train_smiles = None
        if not args.skip_novelty:
            train_smiles = load_train_smiles(args.data_dir)
            print(f"novelty reference: {len(train_smiles):,} training molecules",
                  flush=True)

        payload: dict[str, Any] = {
            "framing": ("READ-ONLY analysis of ALREADY SAVED samples. Best-of-k rerank "
                        "is INFERENCE-TIME POSTPROCESSING, not intrinsic conditional "
                        "generation, and it is a finite-candidate performance reference "
                        "rather than an upper bound on a retrained model."),
            "settings": vars(args) | {"k_values": k_values, "decoders": list(decoders)},
            "runs": {},
        }

        for label, root in runs.items():
            if not root.is_dir():
                raise FileNotFoundError(f"saved grid not found: {root}")
            print(f"\n=== {label}: {root} ===", flush=True)
            run: dict[str, Any] = {"root": str(root), "cells": {}, "support_by_n": {},
                                   "provenance": {}}
            all_cells = []
            for n in targets_n:
                by_target: dict[str, Any] = {}
                pooled: dict[str, list] = {d: [] for d in decoders}
                for qed in qed_values:
                    cell = load_cell(cell_directory(root, temperature=args.temperature,
                                                    n=n, qed=qed),
                                     n=n, qed=qed, decoders=decoders)
                    all_cells.append(cell)
                    entry: dict[str, Any] = {
                        "candidates_in_pool": cell["candidates_in_pool"],
                        "requested_n_consistent": cell["requested_n_consistent"],
                        "max_stored_vs_recomputed_qed_gap":
                            cell["max_stored_vs_recomputed_qed_gap"],
                        "stored_qed_matches_recomputation":
                            cell["stored_qed_matches_recomputation"],
                        "support": {d: support_block(cell["candidates"][d])
                                    for d in decoders},
                        "rerank": {},
                    }
                    rows = cell["candidates"][args.headline]
                    for k in k_values:
                        entry["rerank"][str(k)] = rerank_cell(
                            rows, target=qed, k=k, trials=args.partitions,
                            seed=args.seed + 101 * n + int(round(1000 * qed)),
                            train_smiles=train_smiles)
                    by_target[f"{qed:.2f}"] = entry
                    for d in decoders:
                        pooled[d].extend(cell["candidates"][d])
                    native = entry["support"][args.headline]
                    print(f"  N={n} q={qed}: pool={cell['candidates_in_pool']} "
                          f"valid={native['validity']:.3f} "
                          f"native QED={(native['qed'] or {}).get('mean')} "
                          f"IQR={(native['qed'] or {}).get('iqr')} "
                          f"p75={(native['qed'] or {}).get('quantiles', {}).get('p75')}",
                          flush=True)
                    run["provenance"].setdefault(
                        "flow_checkpoint", cell["flow_checkpoint"])
                    run["provenance"].setdefault("flow_step", cell["flow_step"])
                    run["provenance"].setdefault(
                        "autoencoder_sha256", cell["autoencoder_sha256"])
                run["cells"][str(n)] = by_target
                run["support_by_n"][str(n)] = {d: support_block(pooled[d])
                                               for d in decoders}
                run["support_by_n"][str(n)]["note"] = (
                    "pooled over the requested QED values at this N: the range this "
                    "checkpoint actually produces, which is what relabeling would train "
                    "on. Per-condition numbers are under `cells`.")
            run["decoder_disagreement"] = disagreement_block(all_cells, decoders)
            run["responsiveness_by_n"] = {
                str(n): {str(k): responsiveness(run["cells"][str(n)], k=k)
                         for k in k_values}
                for n in targets_n}
            payload["runs"][label] = run

        payload["verdict"] = verdict(payload, decoder=args.headline)
        payload["elapsed_seconds"] = time.perf_counter() - started
        atomic_write_json(payload, out_dir / "qed_support_rerank.json")
        (out_dir / "qed_support_rerank.txt").write_text(
            format_report(payload, decoders=decoders, headline=args.headline,
                          k_values=k_values))
        print(f"\nwrote {out_dir / 'qed_support_rerank.json'}")
        print(f"wrote {out_dir / 'qed_support_rerank.txt'}")
        return 0
    except FileNotFoundError as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 2
    except ProvenanceError as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 3
    except Exception:  # noqa: BLE001
        print("FATAL: unexpected\n" + traceback.format_exc(), file=sys.stderr, flush=True)
        return 1


def _fmt(value, spec=".4f"):
    return format(value, spec) if isinstance(value, (int, float)) else str(value)


def format_report(payload, *, decoders, headline, k_values) -> str:
    lines = ["QED support, decoder agreement, and best-of-k reranking", "=" * 78, "",
             payload["framing"], "", f"headline decoder: {headline}", ""]
    for label, run in payload["runs"].items():
        provenance = run["provenance"]
        lines += [f"### {label}  ({run['root']})",
                  f"    flow step {provenance.get('flow_step')}  "
                  f"AE {provenance.get('autoencoder_sha256')}", "",
                  "-- QED SUPPORT, pooled over requests at each N " + "-" * 29]
        for n, block in run["support_by_n"].items():
            lines.append(f"  N={n}")
            for decoder in decoders:
                entry = block[decoder]
                qed = entry.get("qed") or {}
                quantiles = qed.get("quantiles") or {}
                lines.append(
                    f"    {decoder:<12} validity={entry['validity']:.3f} "
                    f"scorable={entry['scorable']:<4} "
                    f"QED mean={_fmt(qed.get('mean'))} "
                    f"median={_fmt(quantiles.get('p50'))} "
                    f"IQR={_fmt(qed.get('iqr'))}")
                lines.append(
                    f"    {'':<12} p5={_fmt(quantiles.get('p5'))} "
                    f"p25={_fmt(quantiles.get('p25'))} "
                    f"p75={_fmt(quantiles.get('p75'))} "
                    f"p95={_fmt(quantiles.get('p95'))} "
                    f">=0.75: {_fmt(qed.get('fraction_at_least_0p75'))} "
                    f">=0.8: {_fmt(qed.get('fraction_at_least_0p8'))}")
        lines += ["", "-- CROSS-DECODER LABEL AGREEMENT, per embedding " + "-" * 28]
        for pair, entry in run["decoder_disagreement"]["pairs"].items():
            difference = entry.get("absolute_qed_difference") or {}
            lines.append(
                f"  {pair:<28} jointly scorable={entry['jointly_scorable']:<5} "
                f"same molecule={_fmt(entry.get('identical_molecule_fraction'))}")
            lines.append(
                f"  {'':<28} |dQED| median={_fmt(difference.get('median'))} "
                f"mean={_fmt(difference.get('mean'))} "
                f"p95={_fmt((difference.get('quantiles') or {}).get('p95'))} "
                f">0.1: {_fmt(difference.get('fraction_above_0p1'))}")
            lines.append(
                f"  {'':<28} label exists under exactly one decoder: "
                f"{entry['label_exists_under_exactly_one']}")
        lines += ["", "-- BEST-OF-k RERANK (POSTPROCESSING) " + "-" * 39]
        for n, by_target in run["cells"].items():
            for target, cell in by_target.items():
                lines.append(f"  N={n} requested QED {target}")
                for k in k_values:
                    entry = cell["rerank"][str(k)]
                    if entry.get("skipped"):
                        lines.append(f"    k={k:<3} SKIPPED ({entry['skipped']})")
                        continue
                    qed_mean = (entry.get("qed_mean") or {})
                    mae = (entry.get("qed_mae_vs_target") or {})
                    tag = "  <- native" if entry["is_native_baseline"] else ""
                    lines.append(
                        f"    k={k:<3} accepted="
                        f"{_fmt((entry.get('accepted_per_partition') or {}).get('mean'), '.1f')}"
                        f"/{entry['independent_accepted_ceiling']} "
                        f"QED={_fmt(qed_mean.get('mean'))} "
                        f"[{_fmt(qed_mean.get('min'))},{_fmt(qed_mean.get('max'))}] "
                        f"MAE={_fmt(mae.get('mean'))} "
                        f"cost={_fmt((entry.get('candidates_per_accepted') or {}).get('mean'), '.1f')}"
                        f"/mol{tag}")
                    lines.append(
                        f"    {'':<5} arom="
                        f"{_fmt((entry.get('aromatic_rings_mean') or {}).get('mean'))} "
                        f"atoms={_fmt((entry.get('heavy_atoms_mean') or {}).get('mean'), '.1f')} "
                        f"uniq={_fmt((entry.get('uniqueness') or {}).get('mean'))} "
                        f"novel={_fmt(entry.get('novelty_vs_train_over_pooled_unique'))} "
                        f"intdiv={_fmt(entry.get('internal_diversity_over_pooled_unique'))}")
        lines += ["", "-- FIXED-N RESPONSIVENESS (difference of cell means) " + "-" * 23]
        for n, by_k in run["responsiveness_by_n"].items():
            for k, entry in by_k.items():
                if entry.get("note") and "not every" in entry["note"]:
                    lines.append(f"  N={n} k={k}: {entry['note']}")
                    continue
                means = " -> ".join(_fmt(v) for v in entry["mean_qed_per_request"])
                lines.append(
                    f"  N={n:<3} k={k:<3} {means}   delta="
                    f"{entry['delta_low_to_high']:+.4f}  "
                    f"monotonic={entry['monotonic_increasing']}")
        lines.append("")
    lines += ["-- PRECOMMITTED EVIDENCE CHECKS " + "-" * 44]
    for label, checks in payload["verdict"]["by_checkpoint"].items():
        lines.append(f"  {label}")
        for n, entry in checks["by_n"].items():
            lines.append(
                f"    N={n:<3} >=0.75: {_fmt(entry['fraction_at_least_0p75'])} "
                f"IQR={_fmt(entry['iqr'])} median={_fmt(entry['median'])} "
                f"-> support_sufficient={entry['support_sufficient']}")
        lines += [
            f"    label median |dQED| (constrained vs degree): "
            f"{_fmt(checks['label_median_disagreement_constrained_vs_degree'])} "
            f"-> well_defined={checks['label_well_defined']}",
            f"    rerank k=10 mean MAE={_fmt(checks['rerank_k10_mean_mae'])} "
            f"-> hits_target={checks['rerank_k10_hits_target_mae']}",
            f"    rerank k=10 diversity retention="
            f"{_fmt(checks['rerank_k10_diversity_retention'])} "
            f"-> holds={checks['rerank_k10_holds_diversity']}"]
    lines += ["", payload["verdict"]["interpretation"], "",
              "HOW TO READ THE RERANK BLOCK", "-" * 78,
              "k=1 IS the native model: every candidate is its own group. The QED gain at",
              "larger k is SELECTION on the reported quantity and is expected by",
              "construction; it is not evidence about the model. What is evidence is the",
              "selection cost per accepted molecule and the behaviour of the metrics",
              "rerank does not optimise -- validity, aromaticity, uniqueness, novelty and",
              "diversity -- together with the accepted/ceiling column, which shows how",
              "few independent molecules survive at large k."]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
