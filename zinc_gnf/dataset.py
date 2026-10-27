"""PyTorch molecular graph dataset, collate function and deterministic batching.

Consumes the charge-aware records produced by `scripts/01b_*` /
`scripts/01d_*` (representation `zinc250k_charge_aware_kekule_v2`):

    atom_types  int64 [n]      index into the 17 (element, charge) types
    bond_matrix int64 [n, n]   0 none / 1 single / 2 double / 3 triple, symmetric
    condition   float32 [2]    standardized (log1p(N), QED); train-only scaler
    qed, n_node, row_id, smiles

A batch is padded to its largest molecule and carries explicit masks:

    atom_types [B, N]     padded with 0 (always read through node_mask)
    bond_types [B, N, N]  padded with 0 (always read through pair_mask)
    node_mask  [B, N]     True for real atoms
    pair_mask  [B, N, N]  True for real ordered pairs i != j
    n_node [B], condition [B, 2], qed [B], row_id [B]

One-hot atomic identity is produced from `atom_types` by `atom_one_hot`, so the
model input is the actual (element, charge) identity, never a noise stand-in.

The test split is deliberately not loadable through `load_training_splits`:
training and model selection only ever see train and validation records.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import pickle
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from zinc_gnf.constants_charge_aware import (
    ATOM_VOCAB,
    MAX_NODES,
    NUM_ATOM_TYPES,
    NUM_BOND_TYPES,
    REPRESENTATION_VERSION_CHARGE_AWARE,
    vocabulary_fingerprint,
)


class DatasetFormatError(ValueError):
    """The records are not a valid charge-aware dataset."""


class VocabularyMismatchError(DatasetFormatError):
    """The data was built with a different atom vocabulary than this code."""


# ---------------------------------------------------------------------------
# Dataset.
# ---------------------------------------------------------------------------
class MolecularGraphDataset(Dataset):
    """In-memory dataset of molecular graphs stored compactly (int8)."""

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        *,
        max_nodes: int = MAX_NODES,
        num_atom_types: int = NUM_ATOM_TYPES,
        num_bond_types: int = NUM_BOND_TYPES,
        validate: bool = True,
    ) -> None:
        if len(records) == 0:
            raise DatasetFormatError("dataset needs at least one record")
        self.max_nodes = int(max_nodes)
        self.num_atom_types = int(num_atom_types)
        self.num_bond_types = int(num_bond_types)

        self._atom_types: list[np.ndarray] = []
        self._bond_matrix: list[np.ndarray] = []
        smiles: list[str] = []
        conditions: list[np.ndarray] = []
        qed: list[float] = []
        row_ids: list[int] = []

        for position, record in enumerate(records):
            atoms = np.asarray(record["atom_types"])
            bonds = np.asarray(record["bond_matrix"])
            n = int(atoms.shape[0])
            if validate:
                if atoms.ndim != 1 or n < 1 or n > self.max_nodes:
                    raise DatasetFormatError(f"record {position}: bad atom vector shape {atoms.shape}")
                if bonds.shape != (n, n):
                    raise DatasetFormatError(f"record {position}: bond matrix {bonds.shape} != ({n}, {n})")
                if atoms.min() < 0 or atoms.max() >= self.num_atom_types:
                    raise DatasetFormatError(f"record {position}: atom type outside 0..{self.num_atom_types - 1}")
                if bonds.min() < 0 or bonds.max() >= self.num_bond_types:
                    raise DatasetFormatError(f"record {position}: bond class outside 0..{self.num_bond_types - 1}")
                if not np.array_equal(bonds, bonds.T) or np.any(np.diag(bonds) != 0):
                    raise DatasetFormatError(f"record {position}: bond matrix must be symmetric with zero diagonal")
            self._atom_types.append(atoms.astype(np.int8))
            self._bond_matrix.append(bonds.astype(np.int8))
            smiles.append(str(record.get("smiles", "")))
            qed.append(float(record.get("qed", float("nan"))))
            row_ids.append(int(record["row_id"]) if record.get("row_id") is not None else -1)
            if record.get("condition") is not None:
                conditions.append(np.asarray(record["condition"], dtype=np.float32))

        self.smiles = smiles
        self.qed = np.asarray(qed, dtype=np.float32)
        self.row_id = np.asarray(row_ids, dtype=np.int64)
        self.n_node = np.asarray([a.shape[0] for a in self._atom_types], dtype=np.int64)
        self.condition = (
            np.stack(conditions).astype(np.float32) if len(conditions) == len(records) else None
        )

    def __len__(self) -> int:
        return len(self._atom_types)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "atom_types": self._atom_types[index],
            "bond_matrix": self._bond_matrix[index],
            "n_node": int(self.n_node[index]),
            "qed": float(self.qed[index]),
            "row_id": int(self.row_id[index]),
            "condition": None if self.condition is None else self.condition[index],
        }

    def subset(self, indices: Sequence[int]) -> "MolecularGraphDataset":
        """A new dataset holding only `indices` (order preserved), without re-validating."""
        clone = object.__new__(MolecularGraphDataset)
        clone.max_nodes, clone.num_atom_types = self.max_nodes, self.num_atom_types
        clone.num_bond_types = self.num_bond_types
        idx = [int(i) for i in indices]
        clone._atom_types = [self._atom_types[i] for i in idx]
        clone._bond_matrix = [self._bond_matrix[i] for i in idx]
        clone.smiles = [self.smiles[i] for i in idx]
        clone.qed, clone.row_id, clone.n_node = self.qed[idx], self.row_id[idx], self.n_node[idx]
        clone.condition = None if self.condition is None else self.condition[idx]
        return clone

    def true_graph(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return self._atom_types[index].astype(np.int64), self._bond_matrix[index].astype(np.int64)


# ---------------------------------------------------------------------------
# Batch + collate.
# ---------------------------------------------------------------------------
@dataclass
class MolecularBatch:
    atom_types: torch.Tensor      # [B, N] long
    bond_types: torch.Tensor      # [B, N, N] long
    node_mask: torch.Tensor       # [B, N] bool
    pair_mask: torch.Tensor       # [B, N, N] bool, True for real i != j
    n_node: torch.Tensor          # [B] long
    qed: torch.Tensor             # [B] float
    row_id: torch.Tensor          # [B] long
    condition: torch.Tensor | None = None   # [B, 2] float

    @property
    def batch_size(self) -> int:
        return int(self.atom_types.shape[0])

    @property
    def max_nodes(self) -> int:
        return int(self.atom_types.shape[1])

    def to(self, device: str | torch.device) -> "MolecularBatch":
        return MolecularBatch(
            atom_types=self.atom_types.to(device), bond_types=self.bond_types.to(device),
            node_mask=self.node_mask.to(device), pair_mask=self.pair_mask.to(device),
            n_node=self.n_node.to(device), qed=self.qed.to(device),
            row_id=self.row_id.to(device),
            condition=None if self.condition is None else self.condition.to(device),
        )


def collate_molecules(items: Sequence[dict[str, Any]]) -> MolecularBatch:
    """Pad a list of dataset items into one `MolecularBatch`."""
    if len(items) == 0:
        raise ValueError("cannot collate an empty batch")
    batch = len(items)
    width = max(int(item["n_node"]) for item in items)

    atoms = np.zeros((batch, width), dtype=np.int64)
    bonds = np.zeros((batch, width, width), dtype=np.int64)
    node_mask = np.zeros((batch, width), dtype=bool)
    for b, item in enumerate(items):
        n = int(item["n_node"])
        atoms[b, :n] = item["atom_types"]
        bonds[b, :n, :n] = item["bond_matrix"]
        node_mask[b, :n] = True

    pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
    pair_mask &= ~np.eye(width, dtype=bool)[None]

    have_condition = all(item.get("condition") is not None for item in items)
    return MolecularBatch(
        atom_types=torch.from_numpy(atoms), bond_types=torch.from_numpy(bonds),
        node_mask=torch.from_numpy(node_mask), pair_mask=torch.from_numpy(pair_mask),
        n_node=torch.tensor([int(i["n_node"]) for i in items], dtype=torch.long),
        qed=torch.tensor([float(i["qed"]) for i in items], dtype=torch.float32),
        row_id=torch.tensor([int(i["row_id"]) for i in items], dtype=torch.long),
        condition=(torch.from_numpy(np.stack([i["condition"] for i in items]).astype(np.float32))
                   if have_condition else None),
    )


def make_batch(dataset: MolecularGraphDataset, indices: Sequence[int]) -> MolecularBatch:
    return collate_molecules([dataset[int(i)] for i in indices])


def atom_one_hot(atom_types: torch.Tensor, node_mask: torch.Tensor,
                 num_atom_types: int = NUM_ATOM_TYPES) -> torch.Tensor:
    """[B, N, num_atom_types] one-hot atomic identity, zero on padding rows."""
    one_hot = torch.nn.functional.one_hot(atom_types.clamp(min=0), num_atom_types).to(torch.float32)
    return one_hot * node_mask.unsqueeze(-1).to(one_hot.dtype)


def iter_eval_batches(dataset: MolecularGraphDataset, batch_size: int,
                      indices: Sequence[int] | None = None) -> Iterator[MolecularBatch]:
    """Sequential, deterministic batches over `indices` (default: all)."""
    order = list(range(len(dataset))) if indices is None else [int(i) for i in indices]
    for start in range(0, len(order), batch_size):
        yield make_batch(dataset, order[start:start + batch_size])


# ---------------------------------------------------------------------------
# Deterministic, resumable training batcher.
# ---------------------------------------------------------------------------
class DeterministicBatcher:
    """Shuffled epochs whose order depends only on (seed, epoch).

    State is just (epoch, position), so training can resume at the exact next
    batch. Incomplete final batches are dropped, so every batch has the same
    size.
    """

    def __init__(self, size: int, batch_size: int, seed: int) -> None:
        if batch_size < 1 or size < batch_size:
            raise ValueError(f"need size >= batch_size >= 1 (size={size}, batch_size={batch_size})")
        self.size, self.batch_size, self.seed = int(size), int(batch_size), int(seed)
        self.epoch = 0
        self.position = 0
        self._order = self._permutation(0)

    def _permutation(self, epoch: int) -> np.ndarray:
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + int(epoch))
        return torch.randperm(self.size, generator=generator).numpy()

    @property
    def batches_per_epoch(self) -> int:
        return self.size // self.batch_size

    def next_indices(self) -> np.ndarray:
        if self.position + self.batch_size > self.size:
            self.epoch += 1
            self.position = 0
            self._order = self._permutation(self.epoch)
        indices = self._order[self.position:self.position + self.batch_size]
        self.position += self.batch_size
        return indices

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "position": self.position,
                "size": self.size, "batch_size": self.batch_size, "seed": self.seed}

    def load_state_dict(self, state: dict[str, int]) -> None:
        for key in ("size", "batch_size", "seed"):
            if int(state[key]) != getattr(self, key):
                raise ValueError(f"batcher {key} mismatch: checkpoint {state[key]} vs now {getattr(self, key)}")
        self.epoch, self.position = int(state["epoch"]), int(state["position"])
        self._order = self._permutation(self.epoch)


# ---------------------------------------------------------------------------
# Loading from disk.
# ---------------------------------------------------------------------------
def dataset_fingerprint(payload: dict[str, Any], counts: dict[str, int]) -> dict[str, Any]:
    """Identity of the data a model was trained on (stored in checkpoints)."""
    digest = hashlib.sha256(
        "|".join(f"{k}={counts[k]}" for k in sorted(counts)).encode()
    ).hexdigest()[:16]
    return {
        "representation": payload.get("representation"),
        "vocabulary_fingerprint": payload.get("vocabulary_fingerprint"),
        "split_protocol": payload.get("split_protocol"),
        "csv_sha256": payload.get("csv_sha256"),
        "seed": payload.get("seed"),
        "counts": dict(counts),
        "counts_digest": digest,
    }


def load_training_splits(data_dir: str | Path) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """Load ONLY the train and validation records from `<data_dir>/splits.pkl`.

    The test records are dropped from memory immediately and are never
    returned; use `load_test_split` (explicit, separately named) for final
    evaluation. Verifies representation and vocabulary.
    """
    path = Path(data_dir) / "splits.pkl"
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    _verify_payload(payload, path)
    counts = {name: len(payload[name]) for name in ("train", "val", "test")}
    fingerprint = dataset_fingerprint(payload, counts)
    train, val = payload["train"], payload["val"]
    del payload["test"]
    return train, val, fingerprint


def load_test_split(data_dir: str | Path) -> tuple[list[dict], dict[str, Any]]:
    """Load the held-out TEST records. For final evaluation only."""
    path = Path(data_dir) / "splits.pkl"
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    _verify_payload(payload, path)
    counts = {name: len(payload[name]) for name in ("train", "val", "test")}
    return payload["test"], dataset_fingerprint(payload, counts)


def _verify_payload(payload: dict[str, Any], path: Path) -> None:
    if payload.get("representation") != REPRESENTATION_VERSION_CHARGE_AWARE:
        raise DatasetFormatError(
            f"{path}: representation {payload.get('representation')!r}, "
            f"expected {REPRESENTATION_VERSION_CHARGE_AWARE!r}"
        )
    if payload.get("vocabulary_fingerprint") != vocabulary_fingerprint() \
            or list(payload.get("atom_vocab", [])) != list(ATOM_VOCAB):
        raise VocabularyMismatchError(
            f"{path}: atom vocabulary differs from this code's "
            f"(data {payload.get('vocabulary_fingerprint')} vs code {vocabulary_fingerprint()})"
        )
    for name in ("train", "val", "test"):
        if name not in payload or len(payload[name]) == 0:
            raise DatasetFormatError(f"{path}: split {name!r} missing or empty")


def seeded_subset_indices(size: int, count: int | None, seed: int) -> list[int]:
    """First `count` of a seeded permutation (sorted), or everything if None/too large."""
    if count is None or count >= size:
        return list(range(size))
    generator = torch.Generator().manual_seed(int(seed))
    return sorted(torch.randperm(size, generator=generator)[:count].tolist())


__all__ = [
    "DatasetFormatError",
    "DeterministicBatcher",
    "MolecularBatch",
    "MolecularGraphDataset",
    "VocabularyMismatchError",
    "atom_one_hot",
    "collate_molecules",
    "dataset_fingerprint",
    "iter_eval_batches",
    "load_test_split",
    "load_training_splits",
    "make_batch",
    "seeded_subset_indices",
]
