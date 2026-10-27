"""Derive the MoFlow-fixed ZINC250K dataset from existing charge-aware records.

Primary experiment protocol (project decision): MoFlow's official held-out
molecules are the TEST set; a deterministic validation subset is carved ONLY
from the remaining training pool; the condition scaler is refit on the new
training subset only.

This module does not re-encode anything. It reads the already-encoded
records of the completed random-split dataset (`splits.pkl`), re-assigns them
to splits by their original CSV `row_id`, and writes a SEPARATE dataset
directory. The source directory and the neutral-only baseline are never
written (collision guards below). The official held-out index set is used
exactly as published; nothing is deduplicated or reordered, and canonical
molecule overlaps are reported separately in `split_overlap.json`.

Every consequential assumption is verified before any file is written, and
any failed check aborts the run with the list of failures:

* the source records match the current CSV row by row (SMILES) and the CSV
  SHA-256 matches the one the records were built from;
* MoFlow's `zinc250k.csv` equals ours row for row (`split_audit`);
* the held-out index base (MoFlow subtracts 1; see `split_audit`) is applied
  and the effect of the alternative base is reported;
* exact membership: test == held-out intersect accepted, no held-out row in
  train/val, train+val == accepted minus held-out, splits disjoint and
  covering every accepted record;
* all split counts and the assignment of every rejected molecule.
"""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import pickle
import time
from typing import Any

import numpy as np

from zinc_gnf.constants_charge_aware import (
    ATOM_VOCAB,
    REPRESENTATION_VERSION_CHARGE_AWARE,
    vocabulary_fingerprint,
)
from zinc_gnf.diagnostics import read_zinc_table
from zinc_gnf.preprocessing import ConditionScaler, hash_file
from zinc_gnf.preprocessing_charge_aware import (
    SPLIT_MOFLOW,
    _atomic_csv,
    _atomic_pickle,
    assert_not_baseline_dir,
    fixed_heldout_split,
    software_versions,
)
from zinc_gnf.split_audit import (
    load_valid_indices,
    moflow_heldout_positions,
    require_verified,
    split_overlap_audit,
    verify_moflow_row_identity,
)
from zinc_gnf.utils.runtime import atomic_write_json, format_duration

SCHEMA_VERSION = 1
MOFLOW_README_TRAIN_COUNT = 224_568        # MoFlow README: zinc250k train rows
MOFLOW_README_HELDOUT_COUNT = 24_887


class DerivationError(RuntimeError):
    """One or more verification checks failed; nothing was written."""


class SourceCollisionError(ValueError):
    """The output directory would overwrite the source (random-split) dataset."""


def _same_dir(a: str | Path, b: str | Path) -> bool:
    return Path(a).resolve() == Path(b).resolve()


def _read_row_index(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {int(row["row_id"]): row for row in csv.DictReader(handle)}


def _unique_smiles(records: list[dict[str, Any]]) -> set[str]:
    return {str(r["smiles"]) for r in records}


def derive_moflow_fixed_dataset(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    metrics_dir: str | Path,
    csv_path: str | Path,
    moflow_csv: str | Path,
    moflow_valid_idx: str | Path,
    baseline_processed_dir: str | Path,
    logger: Any,
    seed: int = 42,
    validation_fraction_of_pool: float = 0.10,
    expected_sha256: str | None = None,
    expected_rows: int | None = None,
    moflow_reference_train_count: int | None = MOFLOW_README_TRAIN_COUNT,
    moflow_reference_heldout_count: int | None = MOFLOW_README_HELDOUT_COUNT,
) -> dict[str, Any]:
    """Derive, verify, and write the MoFlow-fixed dataset. Returns the report."""
    source_dir, output_dir, metrics_dir = Path(source_dir), Path(output_dir), Path(metrics_dir)
    started = time.perf_counter()

    # ---- collision guards (before reading anything large) ------------------
    assert_not_baseline_dir(output_dir, baseline_processed_dir)
    assert_not_baseline_dir(metrics_dir, baseline_processed_dir)
    for target in (output_dir, metrics_dir):
        if _same_dir(target, source_dir) or Path(source_dir).resolve() in Path(target).resolve().parents:
            raise SourceCollisionError(
                f"output {target} would overwrite or sit inside the source "
                f"random-split dataset {source_dir}; choose a separate directory"
            )
    if _same_dir(output_dir, metrics_dir):
        raise SourceCollisionError("output_dir and metrics_dir must differ")

    logger.header("Derive MoFlow-fixed ZINC250K dataset (no re-encoding)")
    source_pickle = source_dir / "splits.pkl"
    source_index = source_dir / "row_index.csv"
    for required in (source_pickle, source_index, Path(csv_path), Path(moflow_csv), Path(moflow_valid_idx)):
        if not Path(required).exists():
            raise FileNotFoundError(f"required input not found: {required}")

    # ---- load source records -------------------------------------------------
    source_sha = hash_file(source_pickle)
    logger.info(f"source {source_pickle} bytes={source_pickle.stat().st_size:,} sha256={source_sha[:16]}…")
    with source_pickle.open("rb") as handle:
        payload = pickle.load(handle)
    records = sorted(
        list(payload["train"]) + list(payload["val"]) + list(payload["test"]),
        key=lambda r: int(r["row_id"]),
    )
    logger.info(f"loaded {len(records):,} encoded records "
                f"(source split {payload.get('split_protocol')})")

    failures: list[str] = []
    checks: dict[str, Any] = {}

    def check(name: str, ok: bool, detail: Any = None) -> None:
        checks[name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            failures.append(f"{name}: {detail}")

    # ---- source provenance ----------------------------------------------------
    check("source_representation",
          payload.get("representation") == REPRESENTATION_VERSION_CHARGE_AWARE,
          payload.get("representation"))
    check("source_vocabulary_fingerprint",
          payload.get("vocabulary_fingerprint") == vocabulary_fingerprint(),
          payload.get("vocabulary_fingerprint"))
    check("source_atom_vocab", list(payload.get("atom_vocab", [])) == list(ATOM_VOCAB))

    csv_sha = hash_file(csv_path)
    check("csv_sha256_matches_source", csv_sha == payload.get("csv_sha256"), csv_sha)
    if expected_sha256 is not None:
        check("csv_sha256_matches_pinned", csv_sha == expected_sha256, csv_sha)

    table = read_zinc_table(csv_path)
    n_rows = len(table)
    if expected_rows is not None:
        check("csv_row_count", n_rows == expected_rows, n_rows)

    # ---- row identity: source records vs CSV, and MoFlow vs CSV ----------------
    ids = [int(r["row_id"]) for r in records]
    check("source_row_ids_unique", len(set(ids)) == len(ids), len(ids) - len(set(ids)))
    check("source_row_ids_in_range", bool(ids) and ids[0] >= 0 and ids[-1] < n_rows, (ids[:1], ids[-1:]))
    bad = [int(r["row_id"]) for r in records
           if not (0 <= int(r["row_id"]) < n_rows)
           or str(r["original_smiles"]).strip() != table.smiles[int(r["row_id"])]]
    check("source_records_match_csv_smiles", not bad,
          f"{len(bad)} mismatching rows; first {bad[:5]}")

    row_index = _read_row_index(source_index)
    accepted_ids = set(ids)
    rejected_ids = {i for i, row in row_index.items() if row["status"] == "rejected"}
    check("row_index_covers_every_csv_row",
          set(row_index) == set(range(n_rows)), f"{len(row_index)} rows vs {n_rows}")
    check("accepted_and_rejected_partition_all_rows",
          accepted_ids.isdisjoint(rejected_ids) and len(accepted_ids) + len(rejected_ids) == n_rows,
          {"accepted": len(accepted_ids), "rejected": len(rejected_ids), "rows": n_rows})

    verification = verify_moflow_row_identity(
        table.smiles, {"logP": table.logp, "qed": table.qed, "SAS": table.sas},
        moflow_csv, moflow_valid_idx,
    )
    check("moflow_row_identity_verified", verification["verified"],
          {k: verification[k] for k in ("row_count_equal", "smiles_mismatches", "heldout_positions_valid")})
    require_verified(verification)       # hard stop: never apply unverified indices

    # ---- held-out positions and index-base handling -----------------------------
    raw_indices = load_valid_indices(moflow_valid_idx)
    heldout = set(moflow_heldout_positions(raw_indices, n_rows, index_base=1))
    alt_ok = True
    try:
        alternative = set(moflow_heldout_positions(raw_indices, n_rows, index_base=0))
    except ValueError:
        alt_ok, alternative = False, set()
    index_base = {
        "applied_base": 1,
        "basis": "MoFlow data/transform_zinc250k.py::get_val_ids computes idx - 1",
        "confirmed_by_data": verification["index_base_confirmed_by_data"],
        "alternative_base_0_in_range": alt_ok,
        "rows_whose_membership_would_change_under_base_0": (
            len(heldout ^ alternative) if alt_ok else None
        ),
        "min_index": min(raw_indices), "max_index": max(raw_indices),
        "positions_min": min(heldout), "positions_max": max(heldout),
    }
    check("heldout_indices_unique", len(heldout) == len(raw_indices), len(raw_indices) - len(heldout))
    if moflow_reference_heldout_count is not None:
        check("heldout_count_matches_moflow_reference",
              len(heldout) == moflow_reference_heldout_count, len(heldout))
    if moflow_reference_train_count is not None:
        check("pool_rows_equal_moflow_train_count",
              n_rows - len(heldout) == moflow_reference_train_count, n_rows - len(heldout))

    # ---- split --------------------------------------------------------------------
    splits = fixed_heldout_split(
        records, heldout, validation_fraction_of_pool=validation_fraction_of_pool, seed=seed,
    )
    set_of = {name: {int(r["row_id"]) for r in rs} for name, rs in splits.items()}

    rejected_in_heldout = sorted(rejected_ids & heldout)
    rejected_in_pool = sorted(rejected_ids - heldout)
    pool_accepted = accepted_ids - heldout
    n_val_expected = int(validation_fraction_of_pool * len(pool_accepted))

    check("splits_pairwise_disjoint",
          not (set_of["train"] & set_of["val"] or set_of["train"] & set_of["test"]
               or set_of["val"] & set_of["test"]))
    check("splits_cover_every_accepted_record",
          set_of["train"] | set_of["val"] | set_of["test"] == accepted_ids)
    check("test_equals_heldout_intersect_accepted", set_of["test"] == (heldout & accepted_ids),
          {"test": len(set_of["test"]), "heldout": len(heldout), "rejected_in_heldout": len(rejected_in_heldout)})
    check("no_heldout_row_in_train_or_val", not (heldout & (set_of["train"] | set_of["val"])))
    check("train_plus_val_equals_accepted_minus_heldout",
          set_of["train"] | set_of["val"] == pool_accepted)
    check("split_counts", (len(splits["test"]) == len(heldout) - len(rejected_in_heldout)
                           and len(splits["train"]) + len(splits["val"]) == len(pool_accepted)
                           and len(splits["val"]) == n_val_expected),
          {k: len(v) for k, v in splits.items()})

    rejected_assignment = [
        {
            "row_id": i,
            "reason": row_index[i]["reason"],
            "original_smiles": table.smiles[i],
            "moflow_heldout": i in heldout,
            "assignment": ("excluded: would have been MoFlow TEST" if i in heldout
                           else "excluded: would have been in the training pool"),
        }
        for i in sorted(rejected_ids)
    ]

    # ---- train-only scaler refit ------------------------------------------------------
    old_scaler = payload.get("condition_scaler")
    scaler = ConditionScaler.fit(splits["train"])
    for subset in splits.values():
        for record in subset:
            record["condition"] = scaler.transform(record["qed"], record["n_node"])
    scaler_dict = scaler.to_dict()
    check("scaler_fit_on_train_only", scaler_dict["fit_on"] == "train_only")
    refit_all = ConditionScaler.fit(splits["train"] + splits["val"] + splits["test"])
    check("scaler_differs_from_all_data_fit", tuple(refit_all.mean) != tuple(scaler.mean))

    overlap = split_overlap_audit(splits)
    train_val = _unique_smiles(splits["train"]) | _unique_smiles(splits["val"])
    train_only = _unique_smiles(splits["train"])
    test_smiles = [str(r["smiles"]) for r in splits["test"]]
    novelty_reference = {
        "main_reference": "train",
        "train": {"records": len(splits["train"]), "unique_canonical_smiles": len(train_only)},
        "train_plus_val": {"records": len(splits["train"]) + len(splits["val"]),
                           "unique_canonical_smiles": len(train_val)},
        "moflow_training_rows": n_rows - len(heldout),
        "test_records_with_twin_in_train": sum(s in train_only for s in test_smiles),
        "test_records_with_twin_in_train_or_val": sum(s in train_val for s in test_smiles),
        "note": ("Novelty must be reported against an explicit reference set; "
                 "the full dataset including test molecules is never the main reference."),
    }

    if failures:
        for line in failures:
            logger.error(f"CHECK FAILED {line}")
        raise DerivationError(f"{len(failures)} verification check(s) failed: " + "; ".join(failures))
    logger.info(f"all {len(checks)} verification checks passed")
    logger.info("split sizes: " + " ".join(f"{k}={len(v):,}" for k, v in splits.items()))
    logger.info(f"rejected rows: in held-out={len(rejected_in_heldout)} in pool={len(rejected_in_pool)}")

    # ---- report -------------------------------------------------------------------------
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "moflow_fixed_derivation",
        "representation_version": REPRESENTATION_VERSION_CHARGE_AWARE,
        "vocabulary_fingerprint": vocabulary_fingerprint(),
        "split_protocol": SPLIT_MOFLOW,
        "seed": seed,
        "validation_fraction_of_pool": validation_fraction_of_pool,
        "derived_from": {
            "source_dir": str(source_dir), "source_splits_sha256": source_sha,
            "source_split_protocol": payload.get("split_protocol"),
            "source_seed": payload.get("seed"),
            "note": "no molecule was re-encoded; records are copied by row_id",
        },
        "csv_sha256": csv_sha,
        "candidate_rows": n_rows,
        "accepted_molecules": len(records),
        "rejected_molecules": len(rejected_ids),
        "train_molecules": len(splits["train"]),
        "validation_molecules": len(splits["val"]),
        "test_molecules": len(splits["test"]),
        "moflow_heldout_rows": len(heldout),
        "moflow_training_rows": n_rows - len(heldout),
        "index_base": index_base,
        "rejected_row_assignment": rejected_assignment,
        "condition_scaler": scaler_dict,
        "previous_condition_scaler_random_split": old_scaler,
        "novelty_reference": novelty_reference,
        "split_overlap_summary": {k: v for k, v in overlap.items() if k != "examples"},
        "deduplication": "none: the official held-out set is used as published",
        "checks": checks,
        "software": software_versions(),
        "wall_seconds_before_write": time.perf_counter() - started,
    }

    # ---- write (only after everything verified) -------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "representation": REPRESENTATION_VERSION_CHARGE_AWARE,
        "vocabulary_fingerprint": vocabulary_fingerprint(),
        "atom_vocab": list(ATOM_VOCAB),
        "seed": seed,
        "split_protocol": SPLIT_MOFLOW,
        "csv_sha256": csv_sha,
        "software": report["software"],
        "condition_scaler": scaler_dict,
        "derived_from": report["derived_from"],
        "train": splits["train"], "val": splits["val"], "test": splits["test"],
    }
    _atomic_pickle(out_payload, output_dir / "splits.pkl")
    atomic_write_json(scaler_dict, output_dir / "scaler.json")

    split_of = {int(r["row_id"]): name for name, rs in splits.items() for r in rs}
    rows = []
    for i in range(n_rows):
        if i in split_of:
            rows.append((i, "accepted", "", split_of[i], int(i in heldout)))
        else:
            rows.append((i, "rejected", row_index[i]["reason"], "", int(i in heldout)))
    _atomic_csv(output_dir / "row_index.csv",
                ("row_id", "status", "reason", "split", "moflow_heldout"), rows)
    atomic_write_json(report, output_dir / "report.json")

    atomic_write_json(overlap, metrics_dir / "split_overlap.json")
    atomic_write_json(verification, metrics_dir / "moflow_split_verification.json")
    atomic_write_json({"checks": checks, "index_base": index_base,
                       "rejected_row_assignment": rejected_assignment,
                       "novelty_reference": novelty_reference},
                      metrics_dir / "split_derivation_checks.json")
    logger.info(f"wrote {output_dir} and {metrics_dir} in {format_duration(time.perf_counter() - started)}")
    return report


__all__ = [
    "DerivationError",
    "MOFLOW_README_HELDOUT_COUNT",
    "MOFLOW_README_TRAIN_COUNT",
    "SourceCollisionError",
    "derive_moflow_fixed_dataset",
]
