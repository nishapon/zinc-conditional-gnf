"""ZINC250K preprocessing pipeline.

Reads the ZINC250K CSV, encodes molecules via `zinc_gnf.chemistry`,
cross-checks CSV QED against RDKit-recomputed QED, performs a seeded random
80/10/10 split, fits a train-only condition scaler for `(log1p(N), QED)`,
and writes four artifacts the rest of the pipeline consumes:

    data/processed/zinc250k/splits.pkl     - train/val/test records + meta
    data/processed/zinc250k/scaler.json    - condition scaler statistics
    data/processed/zinc250k/report.json    - full dataset statistics
    outputs/metrics/preprocessing/*.json   - per-category summary files

The script is CPU-only, single-process, deterministic given the config seed,
and never calls the network. The CSV must exist locally (ISML compute nodes
have no internet); missing-CSV is a hard error with a clear message.

Why `csv.DictReader` and not `pandas.read_csv`: the ZINC250K CSV contains
embedded newlines inside quoted SMILES fields (physical line count is
~498,911 but logical record count is 249,455). Line-based parsers or any
reader that doesn't honor RFC 4180 quoting will mis-align rows. The stdlib
csv module handles this correctly.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import pickle
import random
from typing import Any

import numpy as np
from tqdm import tqdm

from zinc_gnf.chemistry import encode_zinc_smiles
from zinc_gnf.constants import (
    ATOM_VOCAB,
    MAX_NODES,
    MIN_NODES,
    NUM_ATOM_TYPES,
    NUM_BOND_TYPES,
    REPRESENTATION_VERSION,
)


# Names of the two conditioning dimensions, in the order the scaler stores
# them. The flow trainer and generator both reference these names when
# standardizing conditions.
CONDITION_NAMES: tuple[str, str] = ("log1p_n_node", "qed")


# ---------------------------------------------------------------------------
# Condition scaler.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConditionScaler:
    """Standardize `(log1p(N), QED)` using train-only statistics.

    `fit` must be given training records only; validation/test records
    receive the same mean/std at transform time so no leakage occurs.
    """

    mean: tuple[float, float]
    std: tuple[float, float]
    names: tuple[str, str] = CONDITION_NAMES

    @classmethod
    def fit(cls, records: list[dict[str, Any]]) -> "ConditionScaler":
        if not records:
            raise ValueError("cannot fit scaler on an empty record list")

        values = np.asarray(
            [
                [np.log1p(int(record["n_node"])), float(record["qed"])]
                for record in records
            ],
            dtype=np.float64,
        )
        mean = values.mean(axis=0)
        std = values.std(axis=0)

        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError("non-finite condition statistics")

        # Floor the std so a near-constant feature does not create a /0 at
        # transform time. 1e-8 is well below any real scale here (both
        # log1p(N) and QED have real stds ~0.1-0.4).
        std = np.maximum(std, 1e-8)

        return cls(
            mean=(float(mean[0]), float(mean[1])),
            std=(float(std[0]), float(std[1])),
        )

    def transform(self, qed: float, n_node: int) -> np.ndarray:
        if n_node <= 0:
            raise ValueError("n_node must be positive")
        qed = float(qed)
        if not np.isfinite(qed):
            raise ValueError("qed must be finite")

        raw = np.asarray(
            [np.log1p(int(n_node)), qed], dtype=np.float32,
        )
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        return (raw - mean) / std

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "mean": list(self.mean),
            "std": list(self.std),
            "fit_on": "train_only",
        }


# ---------------------------------------------------------------------------
# Full-pipeline report.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PreprocessingReport:
    representation_version: str
    csv_path: str
    csv_sha256: str
    candidate_rows: int
    accepted_molecules: int
    accepted_fraction: float
    reference_moflow_count: int
    train_molecules: int
    validation_molecules: int
    test_molecules: int
    node_count: dict[str, float]
    qed_rdkit: dict[str, float]
    condition_scaler: dict[str, Any]
    rejection_reasons: dict[str, int]
    qed_csv_vs_rdkit: dict[str, Any]
    atom_composition: dict[str, Any]
    bond_composition: dict[str, Any]
    stereochemistry_handling: str
    seed: int


# ---------------------------------------------------------------------------
# CSV reader. Uses csv.DictReader so embedded newlines in quoted SMILES
# fields are handled correctly (the ZINC250K CSV has ~498K physical lines
# but ~249K logical records; see module docstring).
# ---------------------------------------------------------------------------
EXPECTED_CSV_COLUMNS = {"smiles", "logP", "qed", "SAS"}


def read_zinc_csv(csv_path: str | Path) -> list[dict[str, Any]]:
    """Read ZINC250K rows and return a list of `{smiles, csv_qed}` dicts.

    Raises `FileNotFoundError` with a specific message when the file is
    missing — compute nodes have no internet on this cluster, so the file
    must have been downloaded on the login node first.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"ZINC250K CSV not found at {csv_path}. "
            "Download it on the login node (compute nodes have no "
            "internet) via:\n"
            "    mkdir -p data/raw\n"
            "    wget -O data/raw/zinc250k.csv "
            "https://raw.githubusercontent.com/aspuru-guzik-group/"
            "chemical_vae/main/models/zinc_properties/"
            "250k_rndm_zinc_drugs_clean_3.csv"
        )

    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(
                f"CSV at {csv_path} is empty or malformed"
            )

        header = set(reader.fieldnames)
        if not EXPECTED_CSV_COLUMNS.issubset(header):
            missing = EXPECTED_CSV_COLUMNS - header
            raise ValueError(
                f"CSV at {csv_path} is missing columns: {sorted(missing)}. "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            smiles = str(row["smiles"]).strip()
            if not smiles:
                continue
            try:
                csv_qed = float(row["qed"])
            except (TypeError, ValueError):
                # Row has a non-numeric QED; treat as missing and let the
                # encoder recompute from RDKit. The CSV-vs-RDKit cross-check
                # will skip this row.
                csv_qed = float("nan")
            rows.append({"smiles": smiles, "csv_qed": csv_qed})

    return rows


def hash_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, chunked to avoid loading the whole thing in RAM."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Core encode-all-rows loop.
# ---------------------------------------------------------------------------
def encode_molecules(
    rows: list[dict[str, Any]],
    *,
    min_nodes: int = MIN_NODES,
    max_nodes: int = MAX_NODES,
    progress: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, float]]]:
    """Encode every CSV row and return (accepted, rejection_counts, qed_pairs).

    The third element is a list of `{csv_qed, rdkit_qed}` dicts, one per
    accepted molecule, used downstream for the CSV-vs-RDKit QED cross-check.
    """
    accepted: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}
    qed_pairs: list[dict[str, float]] = []

    iterator = tqdm(rows, desc="Encoding", unit="mol") if progress else rows

    for row in iterator:
        record, reason = encode_zinc_smiles(
            row["smiles"], min_nodes=min_nodes, max_nodes=max_nodes,
        )
        if record is None:
            rejection_counts[reason or "unknown"] = (
                rejection_counts.get(reason or "unknown", 0) + 1
            )
            continue

        csv_qed = float(row["csv_qed"])
        rdkit_qed = float(record["qed"])
        if np.isfinite(csv_qed):
            qed_pairs.append({"csv_qed": csv_qed, "rdkit_qed": rdkit_qed})

        accepted.append(record)

    return accepted, rejection_counts, qed_pairs


# ---------------------------------------------------------------------------
# Seeded random 80/10/10 split.
# ---------------------------------------------------------------------------
def random_split(
    records: list[dict[str, Any]],
    *,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Shuffle the record list with a seeded RNG and split 80/10/10.

    train_fraction + validation_fraction < 1.0 is required; the remainder
    becomes the test split.
    """
    if not records:
        raise ValueError("cannot split an empty record list")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError(
            "train + validation fractions leave no test split"
        )

    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)

    n_train = int(train_fraction * len(records))
    n_val = int(validation_fraction * len(records))

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        if not idx:
            raise RuntimeError(
                f"split {split_name!r} is empty after shuffling"
            )

    return {
        "train": [records[i] for i in train_idx],
        "val": [records[i] for i in val_idx],
        "test": [records[i] for i in test_idx],
    }


# ---------------------------------------------------------------------------
# Statistics helpers used in the report.
# ---------------------------------------------------------------------------
def _distribution_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def _qed_discrepancy_stats(
    qed_pairs: list[dict[str, float]],
) -> dict[str, Any]:
    if not qed_pairs:
        return {
            "molecules_compared": 0,
            "max_abs_delta": None,
            "mean_abs_delta": None,
            "p99_abs_delta": None,
            "within_1e_4": 0,
            "within_1e_3": 0,
            "within_1e_2": 0,
            "above_1e_1": 0,
        }

    deltas = np.abs(
        np.asarray([p["csv_qed"] - p["rdkit_qed"] for p in qed_pairs])
    )
    return {
        "molecules_compared": int(deltas.size),
        "max_abs_delta": float(deltas.max()),
        "mean_abs_delta": float(deltas.mean()),
        "p99_abs_delta": float(np.quantile(deltas, 0.99)),
        "within_1e_4": int((deltas <= 1e-4).sum()),
        "within_1e_3": int((deltas <= 1e-3).sum()),
        "within_1e_2": int((deltas <= 1e-2).sum()),
        "above_1e_1": int((deltas > 1e-1).sum()),
    }


def _atom_composition(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = np.zeros(NUM_ATOM_TYPES, dtype=np.int64)
    for record in records:
        atom_types = np.asarray(record["atom_types"], dtype=np.int64)
        np.add.at(counts, atom_types, 1)
    total = int(counts.sum())
    return {
        "counts": {
            ATOM_VOCAB[i]: int(counts[i]) for i in range(NUM_ATOM_TYPES)
        },
        "fractions": {
            ATOM_VOCAB[i]: float(counts[i] / total) if total else 0.0
            for i in range(NUM_ATOM_TYPES)
        },
        "total_atoms": total,
    }


def _bond_composition(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = np.zeros(NUM_BOND_TYPES, dtype=np.int64)
    for record in records:
        bond_matrix = np.asarray(record["bond_matrix"], dtype=np.int64)
        # Only upper triangle (undirected bonds counted once).
        upper = np.triu(bond_matrix, k=1)
        values, bincount = np.unique(upper, return_counts=True)
        for v, c in zip(values, bincount):
            if 0 <= int(v) < NUM_BOND_TYPES:
                counts[int(v)] += int(c)

    # Report only real bonds (classes 1-3), plus the raw counts.
    real_bond_total = int(counts[1:].sum())
    labels = {0: "no_bond", 1: "single", 2: "double", 3: "triple"}
    return {
        "counts": {labels[i]: int(counts[i]) for i in range(NUM_BOND_TYPES)},
        "real_bond_fractions": {
            labels[i]: float(counts[i] / real_bond_total) if real_bond_total else 0.0
            for i in range(1, NUM_BOND_TYPES)
        },
        "total_real_bonds": real_bond_total,
    }


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------
REFERENCE_MOFLOW_COUNT = 249_455


def prepare_zinc_dataset(
    *,
    csv_path: str | Path,
    processed_dir: str | Path,
    metrics_dir: str | Path,
    seed: int,
    train_fraction: float = 0.80,
    validation_fraction: float = 0.10,
    min_nodes: int = MIN_NODES,
    max_nodes: int = MAX_NODES,
    expected_sha256: str | None = None,
    progress: bool = True,
) -> PreprocessingReport:
    """Full preprocessing: read -> encode -> cross-check QED -> split ->
    fit scaler -> write artifacts -> return a report.
    """
    csv_path = Path(csv_path)
    processed_dir = Path(processed_dir)
    metrics_dir = Path(metrics_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Fingerprint the CSV so re-runs can verify dataset identity.
    csv_sha256 = hash_file(csv_path)
    if expected_sha256 is not None and csv_sha256 != expected_sha256:
        raise RuntimeError(
            f"CSV SHA-256 mismatch:\n"
            f"  expected: {expected_sha256}\n"
            f"  actual:   {csv_sha256}\n"
            f"The CSV at {csv_path} is not the file this pipeline was "
            "designed around. Re-download or override --expected-sha256."
        )

    rows = read_zinc_csv(csv_path)

    accepted, rejection_counts, qed_pairs = encode_molecules(
        rows,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        progress=progress,
    )

    splits = random_split(
        accepted,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    scaler = ConditionScaler.fit(splits["train"])

    # Attach standardized condition vector to every record.
    for subset in splits.values():
        for record in subset:
            record["condition"] = scaler.transform(
                record["qed"], record["n_node"],
            )

    # Build the report.
    all_qed = np.asarray(
        [float(r["qed"]) for r in accepted], dtype=np.float64,
    )
    all_n = np.asarray(
        [int(r["n_node"]) for r in accepted], dtype=np.int64,
    )
    report = PreprocessingReport(
        representation_version=REPRESENTATION_VERSION,
        csv_path=str(csv_path),
        csv_sha256=csv_sha256,
        candidate_rows=len(rows),
        accepted_molecules=len(accepted),
        accepted_fraction=(len(accepted) / len(rows)) if rows else 0.0,
        reference_moflow_count=REFERENCE_MOFLOW_COUNT,
        train_molecules=len(splits["train"]),
        validation_molecules=len(splits["val"]),
        test_molecules=len(splits["test"]),
        node_count=_distribution_stats(all_n.astype(np.float64)),
        qed_rdkit=_distribution_stats(all_qed),
        condition_scaler=scaler.to_dict(),
        rejection_reasons=dict(sorted(rejection_counts.items())),
        qed_csv_vs_rdkit=_qed_discrepancy_stats(qed_pairs),
        atom_composition=_atom_composition(accepted),
        bond_composition=_bond_composition(accepted),
        stereochemistry_handling=(
            "stripped; non-isomeric canonical SMILES used throughout"
        ),
        seed=seed,
    )

    # Write artifacts.
    splits_path = processed_dir / "splits.pkl"
    scaler_path = processed_dir / "scaler.json"
    report_path = processed_dir / "report.json"

    payload = {
        "representation": REPRESENTATION_VERSION,
        "seed": seed,
        "csv_sha256": csv_sha256,
        "condition_scaler": scaler.to_dict(),
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
    }
    _atomic_pickle(payload, splits_path)
    _atomic_write_json(scaler.to_dict(), scaler_path)
    _atomic_write_json(asdict(report), report_path)

    # Also split out two of the heavier report sections into their own
    # files under outputs/metrics/preprocessing — easier for a slide.
    _atomic_write_json(dict(sorted(rejection_counts.items())),
                       metrics_dir / "rejection_reasons.json")
    _atomic_write_json(_atom_composition(accepted),
                       metrics_dir / "atom_composition.json")
    _atomic_write_json(_bond_composition(accepted),
                       metrics_dir / "bond_composition.json")
    _atomic_write_json(_qed_discrepancy_stats(qed_pairs),
                       metrics_dir / "qed_discrepancy.json")

    # The raw qed pairs are useful for the CSV-vs-RDKit scatter plot; store
    # them separately so the plotting script doesn't have to re-encode.
    qed_pairs_path = metrics_dir / "qed_pairs.pkl"
    _atomic_pickle(qed_pairs, qed_pairs_path)

    return report


# ---------------------------------------------------------------------------
# Atomic write helpers so an interrupted job never leaves a half-written
# file where the next stage might read it.
# ---------------------------------------------------------------------------
def _atomic_pickle(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _atomic_write_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False, sort_keys=False)
        handle.write("\n")
    tmp.replace(path)


__all__ = [
    "CONDITION_NAMES",
    "ConditionScaler",
    "PreprocessingReport",
    "REFERENCE_MOFLOW_COUNT",
    "encode_molecules",
    "hash_file",
    "prepare_zinc_dataset",
    "random_split",
    "read_zinc_csv",
]
