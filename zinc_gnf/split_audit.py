"""Duplicate, split-overlap and MoFlow-split audits for ZINC250K.

Everything here REPORTS; nothing deduplicates, drops, or reorders molecules.

* `duplicate_audit`: groups accepted records by non-stereochemical canonical
  SMILES and classifies each duplicate group (exact repeated row, same
  molecule written differently, or stereoisomers that only collapse because
  stereochemistry is stripped by our identity definition).
* `split_overlap_audit`: after splitting, counts canonical SMILES that occur
  in more than one split and how many validation/test records have a
  training-set twin (a leakage upper bound for novelty/reconstruction
  claims).
* `verify_moflow_row_identity`: before MoFlow's fixed held-out indices may be
  applied to our rows, checks that MoFlow's `zinc250k.csv` is row-for-row the
  same data as ours. MoFlow's index file is applied through its own code,
  `val_ids = [idx - 1 for idx in data]` (`data/transform_zinc250k.py`), i.e.
  the stored indices are 1-based; the raw data alone cannot confirm the base
  (both bases are in range), so the base is an explicit, reported assumption.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from rdkit import Chem, rdBase

from zinc_gnf.preprocessing import hash_file

MOFLOW_INDEX_BASE = 1          # MoFlow subtracts 1 from every stored index.
NUMERIC_TOLERANCE = 1e-9


class MoflowVerificationError(RuntimeError):
    """MoFlow's data does not match ours row-for-row; its split cannot be used."""


# ---------------------------------------------------------------------------
# Duplicates among accepted records.
# ---------------------------------------------------------------------------
def _isomeric_smiles(smiles: str) -> str | None:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def classify_duplicate_group(members: Sequence[Mapping[str, Any]]) -> str:
    """exact_repeat | same_molecule_rewritten | stereoisomers."""
    originals = {str(m["original_smiles"]).strip() for m in members}
    if len(originals) == 1:
        return "exact_repeat"
    isomeric = {_isomeric_smiles(o) for o in originals}
    if len(isomeric) == 1:
        return "same_molecule_rewritten"
    return "stereoisomers"


def duplicate_audit(
    records: Iterable[Mapping[str, Any]],
    *,
    max_groups_listed: int = 200,
) -> dict[str, Any]:
    """Group accepted records by canonical SMILES; classify the duplicates."""
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    total = 0
    for record in records:
        groups[str(record["smiles"])].append(record)
        total += 1

    duplicate_groups = {k: v for k, v in groups.items() if len(v) > 1}
    classes: Counter = Counter()
    size_histogram: Counter = Counter()
    spreads: list[float] = []
    listed = []

    ordered = sorted(
        duplicate_groups.items(),
        key=lambda kv: (-len(kv[1]), min(int(m["row_id"]) for m in kv[1])),
    )
    for smiles, members in ordered:
        kind = classify_duplicate_group(members)
        classes[kind] += 1
        size_histogram[len(members)] += 1
        qed_values = [float(m["qed_csv"]) for m in members
                      if m.get("qed_csv") is not None
                      and np.isfinite(float(m["qed_csv"]))]
        spread = (max(qed_values) - min(qed_values)) if len(qed_values) > 1 else 0.0
        spreads.append(spread)
        if len(listed) < max_groups_listed:
            listed.append({
                "canonical_smiles": smiles,
                "kind": kind,
                "row_ids": sorted(int(m["row_id"]) for m in members),
                "original_smiles": [str(m["original_smiles"]).strip()
                                    for m in sorted(members, key=lambda m: int(m["row_id"]))],
                "csv_qed_spread": spread,
            })

    excess = sum(len(v) - 1 for v in duplicate_groups.values())
    return {
        "records": total,
        "unique_canonical_smiles": len(groups),
        "duplicate_groups": len(duplicate_groups),
        "records_in_duplicate_groups": sum(len(v) for v in duplicate_groups.values()),
        "excess_records": excess,
        "excess_fraction": (excess / total) if total else None,
        "group_size_histogram": {str(k): v for k, v in sorted(size_histogram.items())},
        "group_kinds": dict(classes),
        "csv_qed_spread_max": max(spreads) if spreads else 0.0,
        "csv_qed_spread_mean": float(np.mean(spreads)) if spreads else 0.0,
        "listed_groups": listed,
        "listed_groups_truncated": len(ordered) > len(listed),
        "policy": (
            "REPORT ONLY: no record is removed or merged. Identity is the "
            "non-stereochemical canonical SMILES, so stereoisomers share an "
            "identity and appear here as duplicates."
        ),
    }


# ---------------------------------------------------------------------------
# Cross-split overlap.
# ---------------------------------------------------------------------------
def split_overlap_audit(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    max_examples: int = 25,
) -> dict[str, Any]:
    """Which canonical SMILES occur in more than one split, and how many
    validation/test records have a training-set twin."""
    names = list(splits)
    smiles_sets = {n: {str(r["smiles"]) for r in splits[n]} for n in names}

    pairwise = {}
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            pairwise[f"{left}&{right}"] = len(smiles_sets[left] & smiles_sets[right])

    in_train = {}
    if "train" in smiles_sets:
        for name in names:
            if name == "train":
                continue
            twins = [r for r in splits[name] if str(r["smiles"]) in smiles_sets["train"]]
            in_train[name] = {
                "records": len(splits[name]),
                "records_with_train_twin": len(twins),
                "fraction": (len(twins) / len(splits[name])) if splits[name] else None,
            }

    membership: dict[str, set[str]] = defaultdict(set)
    for name in names:
        for smiles in smiles_sets[name]:
            membership[smiles].add(name)
    spanning = {s: sorted(m) for s, m in membership.items() if len(m) > 1}

    examples = []
    for smiles in sorted(spanning)[:max_examples]:
        examples.append({
            "canonical_smiles": smiles,
            "splits": spanning[smiles],
            "row_ids": sorted(
                int(r["row_id"]) for n in names for r in splits[n]
                if str(r["smiles"]) == smiles
            ),
        })

    return {
        "split_sizes": {n: len(splits[n]) for n in names},
        "unique_smiles_per_split": {n: len(smiles_sets[n]) for n in names},
        "shared_unique_smiles_pairwise": pairwise,
        "canonical_smiles_spanning_multiple_splits": len(spanning),
        "records_with_train_twin": in_train,
        "examples": examples,
        "policy": (
            "REPORT ONLY: the split is not changed to remove overlap. Treat "
            "records_with_train_twin as an upper bound on leakage when "
            "interpreting novelty, uniqueness and reconstruction results."
        ),
    }


# ---------------------------------------------------------------------------
# MoFlow row-identity verification and fixed held-out positions.
# ---------------------------------------------------------------------------
def read_moflow_zinc_csv(path: str | Path) -> dict[str, list]:
    """Read MoFlow's `data/zinc250k.csv` (pandas `index_col=0` layout)."""
    columns: dict[str, list] = {
        "index": [], "smiles": [], "logP": [], "qed": [], "SAS": [],
    }
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        index_name = fieldnames[0] if fieldnames else None
        required = {"smiles", "logP", "qed", "SAS"}
        if index_name is None or not required.issubset(fieldnames):
            raise ValueError(
                f"{path}: unexpected columns {fieldnames}; need an index "
                f"column plus {sorted(required)}"
            )
        for row in reader:
            columns["index"].append(row[index_name])
            columns["smiles"].append(str(row["smiles"]).strip())
            for key in ("logP", "qed", "SAS"):
                try:
                    columns[key].append(float(row[key]))
                except (TypeError, ValueError):
                    columns[key].append(float("nan"))
    return columns


def load_valid_indices(path: str | Path) -> list[int]:
    with Path(path).open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, list) or not all(isinstance(v, int) for v in values):
        raise ValueError(f"{path}: expected a JSON list of integers")
    return values


def moflow_heldout_positions(
    valid_indices: Sequence[int],
    n_rows: int,
    *,
    index_base: int = MOFLOW_INDEX_BASE,
) -> list[int]:
    """0-based CSV row positions of MoFlow's held-out molecules."""
    if index_base not in (0, 1):
        raise ValueError("index_base must be 0 or 1")
    positions = [int(i) - index_base for i in valid_indices]
    if len(set(positions)) != len(positions):
        raise ValueError("held-out indices contain duplicates")
    if positions and (min(positions) < 0 or max(positions) >= n_rows):
        raise ValueError(
            f"held-out indices fall outside 0..{n_rows - 1} with "
            f"index_base={index_base}"
        )
    return sorted(positions)


def verify_moflow_row_identity(
    our_smiles: Sequence[str],
    our_numeric: Mapping[str, Sequence[float]],
    moflow_csv: str | Path,
    valid_idx_json: str | Path,
    *,
    index_base: int = MOFLOW_INDEX_BASE,
    tolerance: float = NUMERIC_TOLERANCE,
    max_mismatches_listed: int = 10,
) -> dict[str, Any]:
    """Compare MoFlow's rows with ours; report whether its split is usable.

    `our_numeric` maps "logP"/"qed"/"SAS" to our parsed CSV columns.
    The result never raises on a mismatch; `verified` says whether MoFlow's
    fixed indices may be applied to our row order. Use
    `require_verified` to turn a failed verification into an error.
    """
    theirs = read_moflow_zinc_csv(moflow_csv)
    indices = load_valid_indices(valid_idx_json)
    n_ours, n_theirs = len(our_smiles), len(theirs["smiles"])

    result: dict[str, Any] = {
        "moflow_csv": str(moflow_csv),
        "moflow_csv_sha256": hash_file(moflow_csv),
        "valid_idx_json": str(valid_idx_json),
        "valid_idx_json_sha256": hash_file(valid_idx_json),
        "rows_ours": n_ours,
        "rows_moflow": n_theirs,
        "row_count_equal": n_ours == n_theirs,
        "index_base_assumed": index_base,
    }

    compared = min(n_ours, n_theirs)
    mismatches = [
        i for i in range(compared)
        if str(our_smiles[i]).strip() != theirs["smiles"][i]
    ]
    result["smiles_compared"] = compared
    result["smiles_mismatches"] = len(mismatches)
    result["first_smiles_mismatches"] = [
        {"row": i, "ours": str(our_smiles[i]).strip(), "moflow": theirs["smiles"][i]}
        for i in mismatches[:max_mismatches_listed]
    ]

    try:
        index_is_positional = all(
            int(theirs["index"][i]) == i for i in range(n_theirs)
        )
    except ValueError:
        index_is_positional = False
    result["moflow_index_column_is_0_to_n_minus_1"] = index_is_positional

    numeric = {}
    for key in ("logP", "qed", "SAS"):
        ours = np.asarray(our_numeric[key][:compared], dtype=np.float64)
        other = np.asarray(theirs[key][:compared], dtype=np.float64)
        both = np.isfinite(ours) & np.isfinite(other)
        difference = np.abs(ours[both] - other[both])
        numeric[key] = {
            "max_abs_difference": float(difference.max()) if difference.size else None,
            "rows_exceeding_tolerance": int((difference > tolerance).sum()),
            "rows_nonfinite_in_one_file": int((~both).sum()),
        }
    result["numeric_columns"] = numeric
    result["numeric_tolerance"] = tolerance
    numeric_ok = all(v["rows_exceeding_tolerance"] == 0 for v in numeric.values())

    indices_ok = True
    index_problem = None
    try:
        positions = moflow_heldout_positions(indices, n_ours, index_base=index_base)
    except ValueError as error:
        indices_ok, positions, index_problem = False, [], str(error)
    result["heldout_count"] = len(indices)
    result["heldout_positions_valid"] = indices_ok
    result["heldout_positions_problem"] = index_problem
    result["index_min"] = min(indices) if indices else None
    result["index_max"] = max(indices) if indices else None
    in_range_if_0 = bool(indices) and min(indices) >= 0 and max(indices) < n_ours
    in_range_if_1 = bool(indices) and min(indices) >= 1 and max(indices) <= n_ours
    result["index_base_in_range"] = {"0": in_range_if_0, "1": in_range_if_1}
    result["index_base_confirmed_by_data"] = in_range_if_0 != in_range_if_1
    result["index_base_note"] = (
        "Both index bases are in range, so the data cannot confirm the base. "
        "The 1-based reading follows MoFlow's own code (idx - 1 in "
        "data/transform_zinc250k.py::get_val_ids)."
        if in_range_if_0 and in_range_if_1
        else "The index range is consistent with only one base."
    )

    result["row_identity_verified"] = bool(
        result["row_count_equal"] and not mismatches and numeric_ok
        and index_is_positional
    )
    result["verified"] = bool(result["row_identity_verified"] and indices_ok)
    result["train_count_if_applied"] = n_ours - len(indices) if indices_ok else None
    return result


def require_verified(verification: Mapping[str, Any]) -> None:
    if not verification.get("verified"):
        raise MoflowVerificationError(
            "MoFlow row identity NOT verified; refusing to apply its fixed "
            f"split indices. smiles_mismatches="
            f"{verification.get('smiles_mismatches')}, "
            f"row_count_equal={verification.get('row_count_equal')}, "
            f"heldout_positions_problem="
            f"{verification.get('heldout_positions_problem')}"
        )


__all__ = [
    "MOFLOW_INDEX_BASE",
    "MoflowVerificationError",
    "classify_duplicate_group",
    "duplicate_audit",
    "load_valid_indices",
    "moflow_heldout_positions",
    "read_moflow_zinc_csv",
    "require_verified",
    "split_overlap_audit",
    "verify_moflow_row_identity",
]
