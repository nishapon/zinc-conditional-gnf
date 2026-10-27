"""Charge-aware ZINC250K preprocessing (representation v2).

A separate path from the neutral-only baseline (`zinc_gnf.preprocessing`,
representation v1), which is unchanged. This path keeps charged molecules
by encoding every atom as an (element, formal charge) type.

Guarantees
----------
* Every CSV row is accounted for: each row is either an accepted record or
  has an explicit rejection reason. `row_index.csv` lists every row id with
  its status, reason and split. Row ids are 0-based positions in the CSV
  (header excluded) and never change.
* Original SMILES and the CSV QED are retained on every record.
* Deterministic: encoding is order-preserving and the split depends only on
  the seed (and, for the optional fixed protocol, on verified MoFlow indices).
* Condition normalization (log1p(N), QED) is fit on the TRAINING split only.
* Splitting protocols are explicit and recorded in every output:
    - `random_80_10_10` (default): the same seeded shuffle algorithm as the
      baseline (`zinc_gnf.preprocessing.random_split`).
    - `moflow_fixed`: MoFlow's held-out molecules are the test set. Allowed
      only after `verify_moflow_row_identity` passes on the actual files; the
      validation set is a seeded fraction of the remaining pool.
* No deduplication: duplicates and cross-split overlaps are REPORTED
  (`duplicates.json`, `split_overlap.json`), never silently removed.
* Outputs go to a directory that must differ from the neutral-only baseline's.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import random
import time
from typing import Any, Sequence

import numpy as np

from zinc_gnf.chemistry_charge_aware import (
    REJECTION_REASONS,
    encode_zinc_smiles_charge_aware,
    unsupported_charge_types,
)
from zinc_gnf.constants_charge_aware import (
    ATOM_VOCAB,
    MAX_NODES,
    MIN_NODES,
    NUM_ATOM_TYPES,
    NUM_BOND_TYPES,
    REPRESENTATION_VERSION_CHARGE_AWARE,
    vocabulary_fingerprint,
)
from zinc_gnf.diagnostics import ZincTable, read_zinc_table
from zinc_gnf.preprocessing import ConditionScaler, hash_file, random_split
from zinc_gnf.split_audit import (
    duplicate_audit,
    moflow_heldout_positions,
    require_verified,
    split_overlap_audit,
    verify_moflow_row_identity,
    load_valid_indices,
)
from zinc_gnf.utils.runtime import (
    StopFlag,
    atomic_write_json,
    current_rss_mb,
    format_duration,
    json_safe,
    peak_rss_mb,
)

SCHEMA_VERSION = 1
SPLIT_RANDOM = "random_80_10_10"
SPLIT_MOFLOW = "moflow_fixed"
SPLIT_PROTOCOLS = (SPLIT_RANDOM, SPLIT_MOFLOW)


class PreprocessingInterrupted(RuntimeError):
    """SIGTERM/SIGINT arrived; no processed dataset was written."""


class OutputCollisionError(ValueError):
    """The output directory would overwrite the neutral-only baseline."""


def assert_not_baseline_dir(
    output_dir: str | Path, baseline_dir: str | Path,
) -> None:
    """Refuse to write into (or above/below) the neutral-only baseline data."""
    out = Path(output_dir).resolve()
    base = Path(baseline_dir).resolve()
    if out == base or base in out.parents or out in base.parents:
        raise OutputCollisionError(
            f"charge-aware output {out} collides with the neutral-only "
            f"baseline directory {base}; choose a separate directory"
        )


# ---------------------------------------------------------------------------
# Splitting.
# ---------------------------------------------------------------------------
def fixed_heldout_split(
    records: Sequence[dict[str, Any]],
    heldout_row_ids: set[int],
    *,
    validation_fraction_of_pool: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Test = records whose row id is in `heldout_row_ids`; the rest is
    shuffled with a seeded RNG and its first fraction becomes validation."""
    if not 0.0 < validation_fraction_of_pool < 1.0:
        raise ValueError("validation_fraction_of_pool must be in (0, 1)")
    test = [r for r in records if int(r["row_id"]) in heldout_row_ids]
    pool = [r for r in records if int(r["row_id"]) not in heldout_row_ids]
    if not test or not pool:
        raise RuntimeError("fixed held-out split produced an empty split")

    rng = random.Random(seed)
    order = list(range(len(pool)))
    rng.shuffle(order)
    n_val = int(validation_fraction_of_pool * len(pool))
    if n_val < 1:
        raise RuntimeError("validation split would be empty")
    return {
        "train": [pool[i] for i in order[n_val:]],
        "val": [pool[i] for i in order[:n_val]],
        "test": test,
    }


# ---------------------------------------------------------------------------
# Encoding loop with explicit accounting.
# ---------------------------------------------------------------------------
@dataclass
class EncodeResult:
    accepted: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    reason_counts: Counter
    unsupported_types: Counter
    seconds: float
    interrupted: bool


def encode_all(
    table: ZincTable,
    *,
    min_nodes: int,
    max_nodes: int,
    logger: Any,
    progress_every: int = 10_000,
    heartbeat_seconds: float = 30.0,
    stop_flag: StopFlag | None = None,
    row_ids: Sequence[int] | None = None,
) -> EncodeResult:
    """Encode rows in file order; every row ends up accepted or rejected."""
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reasons: Counter = Counter()
    unsupported: Counter = Counter()
    ids = list(range(len(table))) if row_ids is None else list(row_ids)
    total = len(ids)
    stop_flag = stop_flag or StopFlag()

    started = time.perf_counter()
    last = started
    interrupted = False
    for position, row_id in enumerate(ids, start=1):
        smiles = table.smiles[row_id]
        record, reason = encode_zinc_smiles_charge_aware(
            smiles, min_nodes=min_nodes, max_nodes=max_nodes, row_id=row_id,
        )
        if record is None:
            reason = reason or "unknown"
            reasons[reason] += 1
            detail = ""
            if reason == "unsupported_charge_type":
                types = unsupported_charge_types(smiles)
                unsupported.update(types)
                detail = ";".join(types)
            rejected.append({
                "row_id": row_id, "reason": reason, "detail": detail,
                "original_smiles": smiles,
            })
        else:
            csv_qed = float(table.qed[row_id])
            record["qed_csv"] = csv_qed if np.isfinite(csv_qed) else None
            accepted.append(record)

        now = time.perf_counter()
        if position % progress_every == 0 or now - last >= heartbeat_seconds or position == total:
            elapsed = now - started
            rate = position / elapsed if elapsed > 0 else float("nan")
            remaining = (total - position) / rate if rate and rate > 0 else None
            logger.info(
                f"encode {position:,}/{total:,} ({100.0 * position / max(1, total):.1f}%) "
                f"rate={rate:,.0f} mol/s elapsed={format_duration(elapsed)} "
                f"eta={format_duration(remaining)} accepted={len(accepted):,} "
                f"rejected={len(rejected):,} rss_mb={current_rss_mb()}"
            )
            last = now
        if stop_flag.requested:
            interrupted = True
            logger.warn(f"stop requested ({stop_flag.signal_name}) at {position:,}/{total:,}")
            break

    return EncodeResult(
        accepted=accepted, rejected=rejected, reason_counts=reasons,
        unsupported_types=unsupported,
        seconds=time.perf_counter() - started, interrupted=interrupted,
    )


def software_versions() -> dict[str, str]:
    """Versions that can change labels (RDKit QED) or serialization."""
    import platform

    import rdkit

    return {
        "rdkit": rdkit.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
    }


# ---------------------------------------------------------------------------
# Statistics.
# ---------------------------------------------------------------------------
def distribution(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return {"n": 0}
    return {
        "n": int(data.size), "min": float(data.min()), "max": float(data.max()),
        "mean": float(data.mean()), "std": float(data.std()),
        "median": float(np.median(data)),
    }


def atom_type_composition(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = np.zeros(NUM_ATOM_TYPES, dtype=np.int64)
    molecules = np.zeros(NUM_ATOM_TYPES, dtype=np.int64)
    for record in records:
        types = np.asarray(record["atom_types"], dtype=np.int64)
        np.add.at(counts, types, 1)
        molecules[np.unique(types)] += 1
    total = int(counts.sum())
    return {
        "counts": {ATOM_VOCAB[i]: int(counts[i]) for i in range(NUM_ATOM_TYPES)},
        "fractions": {
            ATOM_VOCAB[i]: (float(counts[i] / total) if total else 0.0)
            for i in range(NUM_ATOM_TYPES)
        },
        "molecules_containing": {
            ATOM_VOCAB[i]: int(molecules[i]) for i in range(NUM_ATOM_TYPES)
        },
        "total_atoms": total,
    }


def bond_composition(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = np.zeros(NUM_BOND_TYPES, dtype=np.int64)
    for record in records:
        upper = np.triu(np.asarray(record["bond_matrix"], dtype=np.int64), k=1)
        values, bincount = np.unique(upper, return_counts=True)
        for value, count in zip(values, bincount):
            if 0 <= int(value) < NUM_BOND_TYPES:
                counts[int(value)] += int(count)
    labels = {0: "no_bond", 1: "single", 2: "double", 3: "triple"}
    real = int(counts[1:].sum())
    return {
        "counts": {labels[i]: int(counts[i]) for i in range(NUM_BOND_TYPES)},
        "real_bond_fractions": {
            labels[i]: (float(counts[i] / real) if real else 0.0)
            for i in range(1, NUM_BOND_TYPES)
        },
        "total_real_bonds": real,
    }


def charge_composition(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    net: Counter = Counter()
    n_charged_atoms: Counter = Counter()
    charged_molecules = 0
    for record in records:
        net[int(record["net_charge"])] += 1
        n_charged_atoms[int(record["n_charged_atoms"])] += 1
        charged_molecules += int(record["n_charged_atoms"] > 0)
    total = len(records)
    return {
        "molecules": total,
        "molecules_with_charged_atoms": charged_molecules,
        "fraction_with_charged_atoms": (charged_molecules / total) if total else None,
        "net_charge_distribution": {str(k): v for k, v in sorted(net.items())},
        "charged_atoms_per_molecule": {
            str(k): v for k, v in sorted(n_charged_atoms.items())
        },
    }


def qed_discrepancy(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """CSV QED vs the QED RDKit gives the reconstructed (stored) molecule."""
    pairs = [
        (float(r["qed_csv"]), float(r["qed"])) for r in records
        if r.get("qed_csv") is not None
    ]
    if not pairs:
        return {"molecules_compared": 0}
    csv_values = np.asarray([p[0] for p in pairs])
    rdkit_values = np.asarray([p[1] for p in pairs])
    absolute = np.abs(rdkit_values - csv_values)
    out = {
        "molecules_compared": int(absolute.size),
        "mean_signed_delta": float((rdkit_values - csv_values).mean()),
        "mean_abs_delta": float(absolute.mean()),
        "median_abs_delta": float(np.median(absolute)),
        "p99_abs_delta": float(np.quantile(absolute, 0.99)),
        "max_abs_delta": float(absolute.max()),
        "above_1e-3": int((absolute > 1e-3).sum()),
        "above_1e-2": int((absolute > 1e-2).sum()),
        "above_1e-1": int((absolute > 1e-1).sum()),
        "note": "delta = RDKit(reconstructed molecule) - CSV; RDKit QED is the training label.",
    }
    for threshold in (1e-4, 1e-3, 1e-2):
        out[f"within_{threshold:.0e}".replace("e-0", "e-")] = int((absolute <= threshold).sum())
    return out


# ---------------------------------------------------------------------------
# I/O helpers (atomic, deterministic).
# ---------------------------------------------------------------------------
def _atomic_pickle(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    temporary.replace(path)


def _atomic_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    temporary.replace(path)


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------
def prepare_zinc_dataset_charge_aware(
    *,
    csv_path: str | Path,
    processed_dir: str | Path,
    metrics_dir: str | Path,
    baseline_processed_dir: str | Path,
    seed: int,
    logger: Any,
    train_fraction: float = 0.80,
    validation_fraction: float = 0.10,
    min_nodes: int = MIN_NODES,
    max_nodes: int = MAX_NODES,
    expected_sha256: str | None = None,
    expected_rows: int | None = None,
    split_protocol: str = SPLIT_RANDOM,
    moflow_csv: str | Path | None = None,
    moflow_valid_idx: str | Path | None = None,
    max_molecules: int | None = None,
    progress_every: int = 10_000,
    heartbeat_seconds: float = 30.0,
    stop_flag: StopFlag | None = None,
) -> dict[str, Any]:
    """Read -> encode (charge-aware) -> audit -> split -> scaler -> write."""
    csv_path = Path(csv_path)
    processed_dir = Path(processed_dir)
    metrics_dir = Path(metrics_dir)
    assert_not_baseline_dir(processed_dir, baseline_processed_dir)
    assert_not_baseline_dir(metrics_dir, baseline_processed_dir)
    if split_protocol not in SPLIT_PROTOCOLS:
        raise ValueError(f"split_protocol must be one of {SPLIT_PROTOCOLS}")
    if split_protocol == SPLIT_MOFLOW and not (moflow_csv and moflow_valid_idx):
        raise ValueError(
            f"split_protocol={SPLIT_MOFLOW!r} requires both --moflow-csv and "
            "--moflow-valid-idx (provided on the login node; compute nodes "
            "have no internet)"
        )
    if max_molecules is not None and split_protocol == SPLIT_MOFLOW:
        raise ValueError("--max-molecules cannot be combined with the fixed MoFlow split")

    run_started = time.perf_counter()
    logger.header("ZINC250K charge-aware preprocessing (representation v2)")
    logger.info(f"representation={REPRESENTATION_VERSION_CHARGE_AWARE} "
                f"vocabulary_fingerprint={vocabulary_fingerprint()} "
                f"atom_types={NUM_ATOM_TYPES}")
    logger.info(f"atom vocabulary: {list(ATOM_VOCAB)}")
    logger.info(f"split_protocol={split_protocol} seed={seed}")

    # ---- dataset identity ---------------------------------------------------
    started = time.perf_counter()
    csv_sha256 = hash_file(csv_path)
    if expected_sha256 is not None and csv_sha256 != expected_sha256:
        raise RuntimeError(
            f"CSV SHA-256 mismatch:\n  expected: {expected_sha256}\n"
            f"  actual:   {csv_sha256}\n  file:     {csv_path}"
        )
    logger.info(f"sha256={csv_sha256} verified={expected_sha256 is not None} "
                f"({format_duration(time.perf_counter() - started)})")

    table = read_zinc_table(csv_path)
    logger.info(f"read {len(table):,} CSV records "
                f"(empty_smiles={table.empty_smiles_rows}, "
                f"nonnumeric_qed={table.nonnumeric_qed_rows}) rss_mb={current_rss_mb()}")
    if expected_rows is not None and len(table) != expected_rows:
        raise RuntimeError(
            f"CSV has {len(table):,} records, expected {expected_rows:,}"
        )

    row_ids: list[int] | None = None
    if max_molecules is not None and max_molecules < len(table):
        from zinc_gnf.diagnostics import deterministic_sample_indices

        row_ids = deterministic_sample_indices(len(table), max_molecules, seed)
        logger.warn(f"SUBSAMPLE of {len(row_ids):,} rows for a smoke test; "
                    "the output is NOT the full dataset")

    # ---- MoFlow verification (before any use of its indices) ---------------
    moflow_verification: dict[str, Any] | None = None
    heldout_row_ids: set[int] | None = None
    if moflow_csv and moflow_valid_idx:
        moflow_verification = verify_moflow_row_identity(
            table.smiles,
            {"logP": table.logp, "qed": table.qed, "SAS": table.sas},
            moflow_csv, moflow_valid_idx,
        )
        logger.info(
            f"MoFlow verification: verified={moflow_verification['verified']} "
            f"rows={moflow_verification['rows_moflow']:,} "
            f"smiles_mismatches={moflow_verification['smiles_mismatches']} "
            f"heldout={moflow_verification['heldout_count']:,} "
            f"index_base_assumed={moflow_verification['index_base_assumed']} "
            f"(base confirmed by data: {moflow_verification['index_base_confirmed_by_data']})"
        )
        if split_protocol == SPLIT_MOFLOW:
            require_verified(moflow_verification)
            heldout_row_ids = set(moflow_heldout_positions(
                load_valid_indices(moflow_valid_idx), len(table),
            ))

    # ---- encode -------------------------------------------------------------
    result = encode_all(
        table, min_nodes=min_nodes, max_nodes=max_nodes, logger=logger,
        progress_every=progress_every, heartbeat_seconds=heartbeat_seconds,
        stop_flag=stop_flag, row_ids=row_ids,
    )
    if result.interrupted:
        raise PreprocessingInterrupted(
            "interrupted during encoding; nothing was written to the "
            "processed-data directory"
        )
    accepted, rejected = result.accepted, result.rejected
    n_input = len(row_ids) if row_ids is not None else len(table)
    if len(accepted) + len(rejected) != n_input:
        raise RuntimeError("row accounting failed: accepted + rejected != input rows")
    if not accepted:
        raise RuntimeError("no molecule was accepted; refusing to write outputs")
    logger.info(
        f"encoded: accepted={len(accepted):,} rejected={len(rejected):,} "
        f"({100.0 * len(accepted) / n_input:.3f}% accepted) in "
        f"{format_duration(result.seconds)}"
    )
    for reason, count in result.reason_counts.most_common():
        logger.info(f"  rejection {reason:<26} {count:>8,}")
    if result.unsupported_types:
        logger.warn(f"unsupported charge types found (NOT silently dropped): "
                    f"{dict(result.unsupported_types)}")

    # ---- audits (report only) ----------------------------------------------
    duplicates = duplicate_audit(accepted)
    logger.info(f"duplicates: groups={duplicates['duplicate_groups']:,} "
                f"excess_records={duplicates['excess_records']:,} "
                f"kinds={duplicates['group_kinds']}")

    # ---- split --------------------------------------------------------------
    if split_protocol == SPLIT_RANDOM:
        splits = random_split(
            accepted, train_fraction=train_fraction,
            validation_fraction=validation_fraction, seed=seed,
        )
        protocol_detail = {
            "algorithm": "zinc_gnf.preprocessing.random_split (seeded shuffle)",
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "test_fraction": 1.0 - train_fraction - validation_fraction,
        }
    else:
        splits = fixed_heldout_split(
            accepted, heldout_row_ids or set(),
            validation_fraction_of_pool=validation_fraction, seed=seed,
        )
        protocol_detail = {
            "algorithm": "MoFlow held-out rows = test; seeded validation carve-out from the rest",
            "validation_fraction_of_pool": validation_fraction,
            "moflow_index_base_assumed": moflow_verification["index_base_assumed"],
            "moflow_heldout_rows": len(heldout_row_ids or ()),
            "rejected_rows_inside_heldout": sum(
                1 for r in rejected if int(r["row_id"]) in (heldout_row_ids or set())
            ),
        }
    logger.info("split sizes: " + " ".join(f"{k}={len(v):,}" for k, v in splits.items()))

    overlap = split_overlap_audit(splits)
    logger.info(
        f"split overlap: smiles spanning splits="
        f"{overlap['canonical_smiles_spanning_multiple_splits']:,} "
        f"val/test records with train twin="
        f"{ {k: v['records_with_train_twin'] for k, v in overlap['records_with_train_twin'].items()} }"
    )

    # ---- train-only scaler, then condition vectors ---------------------------
    scaler = ConditionScaler.fit(splits["train"])
    for subset in splits.values():
        for record in subset:
            record["condition"] = scaler.transform(record["qed"], record["n_node"])
    scaler_dict = scaler.to_dict()

    # ---- statistics ------------------------------------------------------------
    split_of_row = {
        int(r["row_id"]): name for name, rs in splits.items() for r in rs
    }
    reason_of_row = {int(r["row_id"]): r["reason"] for r in rejected}
    all_ids = row_ids if row_ids is not None else range(len(table))

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "charge_aware_preprocessing",
        "representation_version": REPRESENTATION_VERSION_CHARGE_AWARE,
        "vocabulary_fingerprint": vocabulary_fingerprint(),
        "atom_vocab": list(ATOM_VOCAB),
        "num_atom_types": NUM_ATOM_TYPES,
        "num_bond_types": NUM_BOND_TYPES,
        "baseline_representation": "zinc250k_neutral_kekule_v1 (unchanged, separate outputs)",
        "csv_path": str(csv_path),
        "csv_sha256": csv_sha256,
        "sha256_verified": expected_sha256 is not None,
        "subsampled": row_ids is not None,
        "seed": seed,
        "split_protocol": split_protocol,
        "split_protocol_detail": protocol_detail,
        "min_nodes": min_nodes,
        "max_nodes": max_nodes,
        "candidate_rows": n_input,
        "accepted_molecules": len(accepted),
        "rejected_molecules": len(rejected),
        "accepted_fraction": len(accepted) / n_input,
        "rejection_reasons": {
            reason: result.reason_counts.get(reason, 0) for reason in REJECTION_REASONS
        },
        "unsupported_charge_types": dict(result.unsupported_types),
        "train_molecules": len(splits["train"]),
        "validation_molecules": len(splits["val"]),
        "test_molecules": len(splits["test"]),
        "condition_scaler": scaler_dict,
        "node_count": distribution([r["n_node"] for r in accepted]),
        "qed_reconstructed": distribution([r["qed"] for r in accepted]),
        "qed_csv_vs_rdkit": qed_discrepancy(accepted),
        "atom_composition": atom_type_composition(accepted),
        "bond_composition": bond_composition(accepted),
        "charge_composition": charge_composition(accepted),
        "duplicates_summary": {k: v for k, v in duplicates.items() if k != "listed_groups"},
        "split_overlap_summary": {k: v for k, v in overlap.items() if k != "examples"},
        "moflow_verification": moflow_verification,
        "stereochemistry_handling": (
            "stripped; molecular identity is the non-isomeric canonical SMILES "
            "(formal charges retained)"
        ),
        "deduplication": "none: duplicates and split overlap are reported, never removed",
        "software": software_versions(),
        "qed_label_note": (
            "Training labels are RDKit QED of the reconstructed molecule and "
            "depend on the RDKit version recorded in 'software'; the CSV QED "
            "was computed elsewhere and differs systematically "
            "(see qed_csv_vs_rdkit). Regenerate labels if the RDKit version "
            "changes."
        ),
        "wall_seconds_before_write": time.perf_counter() - run_started,
        "peak_rss_mb": peak_rss_mb(),
    }

    # ---- write (only after everything succeeded) ----------------------------------
    processed_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    payload = {
        "representation": REPRESENTATION_VERSION_CHARGE_AWARE,
        "vocabulary_fingerprint": vocabulary_fingerprint(),
        "atom_vocab": list(ATOM_VOCAB),
        "seed": seed,
        "split_protocol": split_protocol,
        "csv_sha256": csv_sha256,
        "software": report["software"],
        "condition_scaler": scaler_dict,
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
    }
    _atomic_pickle(payload, processed_dir / "splits.pkl")
    atomic_write_json(scaler_dict, processed_dir / "scaler.json")

    row_rows = []
    for row_id in all_ids:
        if row_id in split_of_row:
            row_rows.append((row_id, "accepted", "", split_of_row[row_id]))
        else:
            row_rows.append((row_id, "rejected", reason_of_row.get(row_id, "unknown"), ""))
    _atomic_csv(processed_dir / "row_index.csv",
                ("row_id", "status", "reason", "split"), row_rows)

    atomic_write_json(report, processed_dir / "report.json")

    _atomic_csv(
        metrics_dir / "rejected_rows.csv",
        ("row_id", "reason", "detail", "original_smiles"),
        [(r["row_id"], r["reason"], r["detail"], str(r["original_smiles"]).strip())
         for r in rejected],
    )
    atomic_write_json(report["rejection_reasons"], metrics_dir / "rejection_reasons.json")
    atomic_write_json(report["atom_composition"], metrics_dir / "atom_composition.json")
    atomic_write_json(report["bond_composition"], metrics_dir / "bond_composition.json")
    atomic_write_json(report["charge_composition"], metrics_dir / "charge_composition.json")
    atomic_write_json(report["qed_csv_vs_rdkit"], metrics_dir / "qed_discrepancy.json")
    atomic_write_json(duplicates, metrics_dir / "duplicates.json")
    atomic_write_json(overlap, metrics_dir / "split_overlap.json")
    if moflow_verification is not None:
        atomic_write_json(moflow_verification, metrics_dir / "moflow_split_verification.json")

    report["files_written"] = sorted(
        str(p) for p in list(processed_dir.glob("*")) + list(metrics_dir.glob("*"))
        if p.is_file() and not p.name.endswith(".tmp")
    )
    report["write_seconds"] = time.perf_counter() - started
    atomic_write_json(report, processed_dir / "report.json")
    logger.info(f"wrote outputs to {processed_dir} and {metrics_dir} "
                f"in {format_duration(report['write_seconds'])}")
    logger.info(f"done in {format_duration(time.perf_counter() - run_started)}")
    return report


__all__ = [
    "EncodeResult",
    "OutputCollisionError",
    "PreprocessingInterrupted",
    "SPLIT_MOFLOW",
    "SPLIT_PROTOCOLS",
    "SPLIT_RANDOM",
    "assert_not_baseline_dir",
    "atom_type_composition",
    "bond_composition",
    "charge_composition",
    "encode_all",
    "fixed_heldout_split",
    "prepare_zinc_dataset_charge_aware",
    "qed_discrepancy",
]
