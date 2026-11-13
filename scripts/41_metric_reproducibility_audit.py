#!/usr/bin/env python3
"""Recompute the reported metrics from saved samples. READ-ONLY, NO MODEL, NO GENERATION.

Answers two of the audit questions directly from `samples.jsonl` files that already
exist. Torch is never imported except lazily for the optional real-ZINC reference.

QUESTION 6 - THE CONSTANT-PREDICTOR FLOOR, PROPERLY WEIGHTED
------------------------------------------------------------
My earlier floor of 0.1333 assumed the three QED targets carry EQUAL weight. The GNF
evaluation does not: it averages |QED_i - target_i| over scored SAMPLES, and the number
of scored samples per cell varies enormously because validity collapses with N. The
pooled target distribution is therefore not uniform, and the minimiser of

    MAE(c) = (1/M) * sum_i |c - target_i|

is the WEIGHTED median of the per-sample targets, not the median of {0.4, 0.6, 0.8}.

This script recomputes both: the sample-weighted floor using the evaluation's own
aggregation, and the uniform-weight floor, so the difference between them is visible
rather than assumed away. If the weighting matters, the earlier 0.1333 was wrong and
this supersedes it.

QUESTION 7 - PER-CONDITION AROMATICITY
--------------------------------------
Full aromatic-ring-count distributions for each of the nine (N, request) cells, not just
the pooled mean of 1.15. A pooled mean cannot distinguish "every molecule is slightly
under-aromatic" from "small molecules are fine and large ones have no rings at all", and
those imply different causes.

Exit codes: 0 ok; 2 missing input; 4 bad arguments; 1 unexpected.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AROMATIC_BINS = (0, 1, 2, 3, 4, 5)
CONSTANT_GRID = [round(0.01 * i, 2) for i in range(20, 96)]


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def stdev(values):
    values = [v for v in values if v is not None]
    return statistics.pstdev(values) if len(values) > 1 else None


def weighted_median(values) -> float | None:
    """Minimiser of sum_i |c - v_i|. With repeats this is the ordinary median of the
    expanded sample, which is exactly the sample-weighted median."""
    ordered = sorted(v for v in values if v is not None)
    return statistics.median(ordered) if ordered else None


def cell_directory(root: Path, *, temperature: float, n: int, qed: float) -> Path:
    return root / f"T{temperature:g}_N{n}_QED{qed:g}"


def score(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    """Recompute chemistry from the stored SMILES rather than trusting the file.

    RDKit is imported here rather than at module scope so the constant-predictor
    arithmetic stays importable, and therefore testable, without a chemistry stack.
    """
    from rdkit import Chem, rdBase
    from rdkit.Chem import QED, rdMolDescriptors

    entry = entry or {}
    if not (entry.get("valid") and entry.get("smiles")):
        return None
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(str(entry["smiles"]))
    if molecule is None:
        return None
    aromatic_atoms = sum(1 for atom in molecule.GetAtoms() if atom.GetIsAromatic())
    heavy = molecule.GetNumHeavyAtoms()
    return {
        "smiles": str(entry["smiles"]),
        "qed": float(QED.qed(molecule)),
        "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(molecule)),
        "rings": int(molecule.GetRingInfo().NumRings()),
        "heavy_atoms": int(heavy),
        "aromatic_atom_fraction": (aromatic_atoms / heavy) if heavy else None,
        "fallback": bool(entry.get("fallback", False)),
    }


def load_cells(root: Path, *, targets_n, qed_values, temperature, decoder):
    """Per-cell scored samples, plus the count that did not decode."""
    cells: dict[str, Any] = {}
    for n in targets_n:
        for qed in qed_values:
            directory = cell_directory(root, temperature=temperature, n=n, qed=qed)
            path = directory / "samples.jsonl"
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path} not found. This script never generates samples; the grid "
                    f"must already have been written with --save-samples.")
            scored, total = [], 0
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    total += 1
                    row = score(json.loads(line).get(decoder))
                    if row is not None:
                        scored.append(row)
            cells[f"N{n}_QED{qed:g}"] = {
                "n_node": n, "target": float(qed),
                "candidates": total, "scored": len(scored),
                "scored_fraction": len(scored) / max(1, total),
                "samples": scored,
            }
    return cells


def constant_predictor_audit(cells) -> dict[str, Any]:
    """The floor under the evaluation's OWN sample-level aggregation."""
    targets = [row_target for cell in cells.values()
               for row_target in [cell["target"]] * cell["scored"]]
    achieved = [row["qed"] for cell in cells.values() for row in cell["samples"]]
    if not targets:
        return {"samples": 0, "note": "no scored samples"}

    model_mae = mean([abs(a - t) for a, t in zip(achieved, targets)])

    def constant_mae(c):
        return mean([abs(c - t) for t in targets])

    grid = {f"{c:.2f}": constant_mae(c) for c in CONSTANT_GRID}
    best_constant = min(grid, key=lambda key: grid[key])
    exact = weighted_median(targets)

    unique_targets = sorted(set(targets))
    uniform_best = statistics.median(unique_targets)
    uniform_floor = mean([abs(uniform_best - t) for t in unique_targets])

    weights = {f"{t:g}": targets.count(t) / len(targets) for t in unique_targets}
    return {
        "samples": len(targets),
        "aggregation": "mean over SCORED SAMPLES, matching the GNF evaluation",
        "model_mae_recomputed": model_mae,
        "sample_weighted_floor": {
            "best_constant_on_grid": float(best_constant),
            "mae": grid[best_constant],
            "exact_weighted_median": exact,
            "mae_at_exact": constant_mae(exact) if exact is not None else None,
        },
        "uniform_weight_floor": {
            "best_constant": uniform_best, "mae": uniform_floor,
            "note": ("the earlier 0.1333 figure; assumes the three targets carry equal "
                     "weight, which the sample-level aggregation does not"),
        },
        "target_weights_from_scored_samples": weights,
        "weighting_matters": bool(
            exact is not None and uniform_floor is not None
            and abs(constant_mae(exact) - uniform_floor) > 0.005),
        "model_beats_sample_weighted_floor": bool(
            model_mae is not None and model_mae < grid[best_constant]),
        "mae_by_constant": grid,
        "note": ("Validity collapses with N, so cells contribute unequal numbers of "
                 "scored samples and the pooled target distribution is not uniform. The "
                 "minimiser of a sample-level L1 is the WEIGHTED median of the "
                 "per-sample targets."),
    }


def aromaticity_audit(cells, real_reference=None) -> dict[str, Any]:
    """Full per-cell aromatic-ring distributions, not a pooled mean."""
    per_cell: dict[str, Any] = {}
    pooled_counts: list[int] = []
    for key, cell in cells.items():
        counts = [row["aromatic_rings"] for row in cell["samples"]]
        pooled_counts.extend(counts)
        histogram = Counter(counts)
        per_cell[key] = {
            "n_node": cell["n_node"], "target": cell["target"],
            "scored": cell["scored"],
            "aromatic_rings_mean": mean(counts),
            "aromatic_rings_std": stdev(counts),
            "histogram": {str(b): histogram.get(b, 0) for b in AROMATIC_BINS},
            "fraction_with_no_aromatic_ring": (
                sum(1 for c in counts if c == 0) / len(counts)) if counts else None,
            "aromatic_atom_fraction_mean": mean(
                [row["aromatic_atom_fraction"] for row in cell["samples"]]),
            "rings_mean": mean([row["rings"] for row in cell["samples"]]),
            "heavy_atoms_mean": mean([row["heavy_atoms"] for row in cell["samples"]]),
        }
    by_n: dict[str, Any] = {}
    for cell in cells.values():
        bucket = by_n.setdefault(str(cell["n_node"]), [])
        bucket.extend(row["aromatic_rings"] for row in cell["samples"])
    result = {
        "pooled_mean": mean(pooled_counts),
        "by_cell": per_cell,
        "by_n": {n: {"molecules": len(values), "mean": mean(values),
                     "std": stdev(values),
                     "fraction_with_no_aromatic_ring": (
                         sum(1 for v in values if v == 0) / len(values))
                     if values else None}
                 for n, values in by_n.items()},
        "note": ("A pooled mean cannot distinguish a uniform small deficit from a "
                 "size-dependent collapse. The per-N block is what separates them."),
    }
    if real_reference:
        result["real_reference_by_n"] = real_reference
    return result


def real_aromaticity(data_dir: str, targets_n) -> dict[str, Any]:
    """Real ZINC validation molecules at matched N. Lazy import: pulls in torch."""
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdMolDescriptors
    from zinc_gnf.dataset import load_training_splits

    _, val_records, _ = load_training_splits(data_dir)
    buckets: dict[str, list[int]] = {str(n): [] for n in targets_n}
    for record in val_records:
        count = int(record.get("n_node", 0) or len(record.get("atom_types", [])))
        key = str(count)
        if key not in buckets:
            continue
        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(str(record["smiles"]))
        if molecule is not None:
            buckets[key].append(int(rdMolDescriptors.CalcNumAromaticRings(molecule)))
    return {n: {"molecules": len(values), "mean": mean(values), "std": stdev(values),
                "fraction_with_no_aromatic_ring": (
                    sum(1 for v in values if v == 0) / len(values)) if values else None}
            for n, values in buckets.items()}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir",
                        default="data/processed/zinc250k_charge_aware_moflow_fixed")
    parser.add_argument("--skip-real-reference", action="store_true")
    parser.add_argument("--target-n", default="15,23,30")
    parser.add_argument("--qed-values", default="0.4,0.6,0.8")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--decoder", default="degree",
                        choices=("raw", "constrained", "degree"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        targets_n = [int(v) for v in args.target_n.split(",") if v.strip()]
        qed_values = [float(v) for v in args.qed_values.split(",") if v.strip()]
        if not targets_n or len(qed_values) < 2:
            raise ValueError("need at least one N and two QED targets")
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 4
    try:
        cells = load_cells(Path(args.samples_root), targets_n=targets_n,
                           qed_values=qed_values, temperature=args.temperature,
                           decoder=args.decoder)
        reference = None
        if not args.skip_real_reference:
            try:
                reference = real_aromaticity(args.data_dir, targets_n)
            except Exception as error:  # noqa: BLE001
                print(f"WARNING: real reference unavailable ({error}); continuing",
                      file=sys.stderr)
        payload = {
            "framing": ("READ-ONLY recomputation from saved samples. No model, no "
                        "generation. Chemistry is recomputed from the stored SMILES "
                        "with RDKit rather than trusted."),
            "settings": vars(args),
            "constant_predictor_audit": constant_predictor_audit(cells),
            "aromaticity_audit": aromaticity_audit(cells, reference),
            "per_cell_counts": {key: {"candidates": cell["candidates"],
                                       "scored": cell["scored"],
                                       "scored_fraction": cell["scored_fraction"]}
                                 for key, cell in cells.items()},
        }
        (out_dir / "metric_audit.json").write_text(json.dumps(payload, indent=2))
        report = format_report(payload)
        (out_dir / "metric_audit.txt").write_text(report)
        print(report)
        print(f"\nwrote {out_dir / 'metric_audit.json'}")
        print(f"wrote {out_dir / 'metric_audit.txt'}")
        return 0
    except FileNotFoundError as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2
    except Exception:  # noqa: BLE001
        print("FATAL: unexpected\n" + traceback.format_exc(), file=sys.stderr)
        return 1


def _f(value, spec=".4f"):
    return format(value, spec) if isinstance(value, (int, float)) else "n/a"


def format_report(payload) -> str:
    constant = payload["constant_predictor_audit"]
    aromatic = payload["aromaticity_audit"]
    lines = ["Metric reproducibility audit (recomputed from saved samples)",
             "=" * 78, "", payload["framing"], "",
             f"decoder: {payload['settings']['decoder']}", "",
             "-- QUESTION 6: CONSTANT-PREDICTOR FLOOR, SAMPLE-WEIGHTED " + "-" * 20,
             f"  scored samples pooled:      {constant.get('samples')}",
             f"  model MAE (recomputed):     {_f(constant.get('model_mae_recomputed'))}"]
    weighted = constant.get("sample_weighted_floor") or {}
    uniform = constant.get("uniform_weight_floor") or {}
    lines += [
        f"  sample-weighted floor:      {_f(weighted.get('mae'))} "
        f"at constant {_f(weighted.get('best_constant_on_grid'), '.2f')}",
        f"  exact weighted median:      {_f(weighted.get('exact_weighted_median'), '.3f')}"
        f"  -> MAE {_f(weighted.get('mae_at_exact'))}",
        f"  uniform-weight floor:       {_f(uniform.get('mae'))} "
        f"at constant {_f(uniform.get('best_constant'), '.2f')}  (the earlier 0.1333)",
        f"  weighting changes the floor: {constant.get('weighting_matters')}",
        f"  model beats the floor:       {constant.get('model_beats_sample_weighted_floor')}",
        "", "  target weights from scored samples:"]
    for target, weight in (constant.get("target_weights_from_scored_samples") or {}).items():
        lines.append(f"    request {target}: {weight:.3f} of scored samples")
    lines += ["", f"  {constant.get('note', '')}", "",
              "-- QUESTION 7: AROMATICITY BY CONDITION " + "-" * 38,
              f"  pooled mean aromatic rings: {_f(aromatic.get('pooled_mean'), '.3f')}",
              "", "  by N (generated vs real):"]
    reference = aromatic.get("real_reference_by_n") or {}
    for n, block in (aromatic.get("by_n") or {}).items():
        real = reference.get(n) or {}
        lines.append(
            f"    N={n:<3} generated {_f(block['mean'], '.3f')} "
            f"(sd {_f(block['std'], '.3f')}, n={block['molecules']}, "
            f"no-ring {_f(block['fraction_with_no_aromatic_ring'], '.3f')})"
            f"   real {_f(real.get('mean'), '.3f')} "
            f"(n={real.get('molecules', 'n/a')})")
    lines += ["", "  per cell:"]
    for key, block in (aromatic.get("by_cell") or {}).items():
        histogram = " ".join(f"{b}:{c}" for b, c in block["histogram"].items())
        lines.append(
            f"    {key:<14} scored={block['scored']:<4} "
            f"mean={_f(block['aromatic_rings_mean'], '.3f')} "
            f"no-ring={_f(block['fraction_with_no_aromatic_ring'], '.3f')} "
            f"arom-atom-frac={_f(block['aromatic_atom_fraction_mean'], '.3f')} "
            f"[{histogram}]")
    lines += ["", f"  {aromatic.get('note', '')}", "",
              "-- SCORED-SAMPLE COUNTS (these drive the weighting above) " + "-" * 19]
    for key, block in payload["per_cell_counts"].items():
        lines.append(f"  {key:<14} {block['scored']:>4}/{block['candidates']:<4} "
                     f"({block['scored_fraction']:.3f})")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
