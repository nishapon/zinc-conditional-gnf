"""ZINC preprocessing and scaffold-disjoint dataset splitting."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import pickle
import random
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from zinc_gnf.chemistry import encode_zinc_smiles
from zinc_gnf.constants import MAX_NODES, REPRESENTATION_VERSION


CONDITION_NAMES = ("qed", "log1p_n_node")


@dataclass(frozen=True)
class ConditionScaler:
    """Standardization statistics fitted only on training molecules."""

    mean: tuple[float, float]
    std: tuple[float, float]
    names: tuple[str, str] = CONDITION_NAMES

    @classmethod
    def fit(cls, records: list[dict[str, Any]]) -> "ConditionScaler":
        if not records:
            raise ValueError("Cannot fit the condition scaler on no records.")

        values = np.asarray(
            [
                [
                    float(record["qed"]),
                    np.log1p(int(record["n_node"])),
                ]
                for record in records
            ],
            dtype=np.float64,
        )

        mean = values.mean(axis=0)
        std = values.std(axis=0)

        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError("Condition statistics contain non-finite values.")

        std = np.maximum(std, 1e-8)

        return cls(
            mean=(float(mean[0]), float(mean[1])),
            std=(float(std[0]), float(std[1])),
        )

    def transform(self, qed: float, n_node: int) -> np.ndarray:
        qed = float(qed)
        n_node = int(n_node)

        if not np.isfinite(qed):
            raise ValueError("QED must be finite.")
        if n_node <= 0:
            raise ValueError("n_node must be positive.")

        raw = np.asarray(
            [qed, np.log1p(n_node)],
            dtype=np.float32,
        )
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)

        return (raw - mean) / std

    def inverse_transform(
        self,
        condition: np.ndarray,
    ) -> tuple[float, float]:
        condition = np.asarray(condition, dtype=np.float64)

        if condition.shape != (2,):
            raise ValueError(
                f"Expected condition shape (2,), got {condition.shape}."
            )

        raw = (
            condition * np.asarray(self.std)
            + np.asarray(self.mean)
        )

        qed = float(raw[0])
        n_node = float(np.expm1(raw[1]))
        return qed, n_node

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "mean": list(self.mean),
            "std": list(self.std),
            "fit_on": "train_only",
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> "ConditionScaler":
        return cls(
            mean=tuple(float(value) for value in payload["mean"]),
            std=tuple(float(value) for value in payload["std"]),
            names=tuple(payload.get("names", CONDITION_NAMES)),
        )


@dataclass(frozen=True)
class PreprocessingReport:
    candidate_rows: int
    eligible_molecules: int
    selected_molecules: int
    train_molecules: int
    validation_molecules: int
    test_molecules: int
    train_scaffolds: int
    validation_scaffolds: int
    test_scaffolds: int
    largest_scaffold_group: int
    acyclic_molecules: int
    rejections: dict[str, int]


def read_zinc_csv(csv_path: str | Path) -> pd.DataFrame:
    """Read a ZINC CSV and identify its SMILES column."""
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    frame = pd.read_csv(csv_path)

    aliases = (
        "smiles",
        "SMILES",
        "canonical_smiles",
        "Canonical_SMILES",
    )

    smiles_column = next(
        (name for name in aliases if name in frame.columns),
        None,
    )

    if smiles_column is None:
        raise KeyError(
            "Could not find a SMILES column. "
            f"Available columns: {list(frame.columns)}"
        )

    result = frame[[smiles_column]].copy()
    result.columns = ["smiles"]
    result["smiles"] = result["smiles"].astype(str)
    result = result[result["smiles"].str.len() > 0]

    return result.reset_index(drop=True)



def encode_unique_molecules(
    smiles_values: Iterable[str],
    *,
    target_molecules: int,
    max_nodes: int = MAX_NODES,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Encode and connectivity-deduplicate candidate molecules."""
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    rejections: Counter[str] = Counter()

    for smiles in smiles_values:
        record, reason = encode_zinc_smiles(
            smiles,
            max_nodes=max_nodes,
        )

        if record is None:
            rejections[reason or "encoder_rejected"] += 1
            continue

        canonical_smiles = record["smiles"]

        if canonical_smiles in seen:
            rejections["duplicate_connectivity"] += 1
            continue

        seen.add(canonical_smiles)
        records.append(record)

        if len(records) == target_molecules:
            break

    if len(records) < target_molecules:
        raise RuntimeError(
            f"Only {len(records)} eligible molecules were found; "
            f"{target_molecules} were requested."
        )

    return records, rejections


def murcko_scaffold(smiles: str) -> str:
    """Return the non-stereochemical Bemis–Murcko scaffold."""
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError(
            f"Invalid processed SMILES encountered: {smiles}"
        )

    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=molecule,
        includeChirality=False,
    )


def scaffold_split(
    records: list[dict[str, Any]],
    *,
    seed: int,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
]:
    """Split complete scaffold groups without scaffold leakage."""
    if not records:
        raise ValueError(
            "Cannot split an empty molecular collection."
        )

    if not 0.0 < train_fraction < 1.0:
        raise ValueError(
            "train_fraction must be between zero and one."
        )

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be between zero and one."
        )

    if train_fraction + validation_fraction >= 1.0:
        raise ValueError(
            "Train and validation fractions leave no test data."
        )

    scaffold_groups: dict[str, list[int]] = defaultdict(list)

    for index, record in enumerate(records):
        scaffold = murcko_scaffold(record["smiles"])
        scaffold_groups[scaffold].append(index)

    groups = list(scaffold_groups.values())

    # Shuffle before sorting so equal-sized scaffold groups receive a
    # deterministic but seed-dependent order.
    generator = random.Random(seed)
    generator.shuffle(groups)
    groups.sort(key=len, reverse=True)

    train_target = int(train_fraction * len(records))
    validation_target = int(
        validation_fraction * len(records)
    )

    indices: dict[str, list[int]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for group in groups:
        if (
            len(indices["train"]) + len(group)
            <= train_target
        ):
            indices["train"].extend(group)

        elif (
            len(indices["val"]) + len(group)
            <= validation_target
        ):
            indices["val"].extend(group)

        else:
            indices["test"].extend(group)

    for split_indices in indices.values():
        generator.shuffle(split_indices)

    if any(
        not split_indices
        for split_indices in indices.values()
    ):
        raise RuntimeError(
            "Scaffold grouping produced an empty split. "
            "Use more molecules or different split fractions."
        )

    flattened = [
        index
        for split_indices in indices.values()
        for index in split_indices
    ]

    if sorted(flattened) != list(range(len(records))):
        raise AssertionError(
            "Split indices do not cover every molecule exactly once."
        )

    index_to_scaffold = {
        index: scaffold
        for scaffold, group in scaffold_groups.items()
        for index in group
    }

    scaffold_sets = {
        split_name: {
            index_to_scaffold[index]
            for index in split_indices
        }
        for split_name, split_indices in indices.items()
    }

    split_pairs = (
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    )

    for left, right in split_pairs:
        if not scaffold_sets[left].isdisjoint(
            scaffold_sets[right]
        ):
            raise AssertionError(
                f"Scaffold leakage between {left} and {right}."
            )

    splits = {
        split_name: [
            records[index]
            for index in split_indices
        ]
        for split_name, split_indices in indices.items()
    }

    metadata = {
        "indices": indices,
        "scaffold_counts": {
            split_name: len(scaffolds)
            for split_name, scaffolds
            in scaffold_sets.items()
        },
        "largest_scaffold_group": max(
            map(len, groups)
        ),
        "acyclic_molecules": len(
            scaffold_groups.get("", [])
        ),
    }

    return splits, metadata



def prepare_zinc_dataset(
    csv_path: str | Path,
    output_dir: str | Path,
    *,
    target_molecules: int = 10_000,
    candidate_count: int = 12_000,
    seed: int = 20_260_928,
    max_nodes: int = MAX_NODES,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> tuple[Path, Path, PreprocessingReport]:
    """Create scaffold splits and a training-only condition scaler."""
    if target_molecules <= 0:
        raise ValueError(
            "target_molecules must be positive."
        )

    if candidate_count < target_molecules:
        raise ValueError(
            "candidate_count must be at least target_molecules."
        )

    frame = read_zinc_csv(csv_path)

    if len(frame) < candidate_count:
        raise ValueError(
            f"The CSV has {len(frame)} usable rows, fewer than "
            f"the requested {candidate_count} candidates."
        )

    # Candidate selection is reproducible and independent of CSV order.
    candidates = frame.sample(
        n=candidate_count,
        random_state=seed,
    )

    records, rejections = encode_unique_molecules(
        candidates["smiles"],
        target_molecules=target_molecules,
        max_nodes=max_nodes,
    )

    splits, metadata = scaffold_split(
        records,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    # Prevent validation/test information from entering normalization.
    scaler = ConditionScaler.fit(splits["train"])

    for subset in splits.values():
        for record in subset:
            record["condition"] = scaler.transform(
                record["qed"],
                record["n_node"],
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_path = (
        output_dir
        / "zinc_medium_10k_splits.pkl"
    )
    scaler_path = (
        output_dir
        / "condition_scaler.json"
    )
    report_path = (
        output_dir
        / "preprocessing_report.json"
    )

    payload = {
        "seed": seed,
        "split_method": (
            "Murcko scaffold, largest groups first"
        ),
        "representation": REPRESENTATION_VERSION,
        "stereochemistry": False,
        "condition_names": list(CONDITION_NAMES),
        "condition_mean": np.asarray(
            scaler.mean,
            dtype=np.float32,
        ),
        "condition_std": np.asarray(
            scaler.std,
            dtype=np.float32,
        ),
        "split_indices": metadata["indices"],
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
    }

    with split_path.open("wb") as handle:
        pickle.dump(
            payload,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    scaler_path.write_text(
        json.dumps(
            scaler.to_dict(),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    report = PreprocessingReport(
        candidate_rows=candidate_count,
        eligible_molecules=len(records),
        selected_molecules=target_molecules,
        train_molecules=len(splits["train"]),
        validation_molecules=len(splits["val"]),
        test_molecules=len(splits["test"]),
        train_scaffolds=metadata[
            "scaffold_counts"
        ]["train"],
        validation_scaffolds=metadata[
            "scaffold_counts"
        ]["val"],
        test_scaffolds=metadata[
            "scaffold_counts"
        ]["test"],
        largest_scaffold_group=metadata[
            "largest_scaffold_group"
        ],
        acyclic_molecules=metadata[
            "acyclic_molecules"
        ],
        rejections=dict(rejections),
    )

    report_path.write_text(
        json.dumps(
            asdict(report),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return split_path, scaler_path, report


__all__ = [
    "CONDITION_NAMES",
    "ConditionScaler",
    "PreprocessingReport",
    "encode_unique_molecules",
    "murcko_scaffold",
    "prepare_zinc_dataset",
    "read_zinc_csv",
    "scaffold_split",
]
