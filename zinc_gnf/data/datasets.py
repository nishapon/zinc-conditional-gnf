"""PyTorch datasets and padded molecular-graph batching."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from zinc_gnf.constants import (
    CHARGE_TO_INDEX,
    HYDROGEN_TO_INDEX,
    NODE_FEATURE_DIM,
    NUM_ATOM_TYPES,
    NUM_CHARGE_TYPES,
    NUM_HYDROGEN_TYPES,
)


class ZincDataset(Dataset):
    """Immutable view over preprocessed ZINC records."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> None:
        if not records:
            raise ValueError(
                "A ZincDataset requires at least one record."
            )

        self._records = tuple(
            dict(record)
            for record in records
        )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        return self._records[index]


def _class_indices(
    values: Any,
    mapping: Mapping[int, int],
    *,
    field: str,
) -> torch.Tensor:
    """Map chemical values to classification indices."""
    try:
        indices = [
            mapping[int(value)]
            for value in values
        ]
    except KeyError as error:
        raise ValueError(
            f"Unsupported {field} value: {error.args[0]}"
        ) from error

    return torch.tensor(
        indices,
        dtype=torch.long,
    )


def collate_molecules(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Pad molecules and create reconstruction/attention masks."""
    if not records:
        raise ValueError(
            "Cannot collate an empty batch."
        )

    batch_size = len(records)
    max_nodes = max(
        int(record["n_node"])
        for record in records
    )

    atom_targets = torch.full(
        (batch_size, max_nodes),
        -100,
        dtype=torch.long,
    )
    charge_targets = torch.full_like(
        atom_targets,
        -100,
    )
    h_targets = torch.full_like(
        atom_targets,
        -100,
    )

    bond_targets = torch.zeros(
        batch_size,
        max_nodes,
        max_nodes,
        dtype=torch.long,
    )
    node_mask = torch.zeros(
        batch_size,
        max_nodes,
        dtype=torch.bool,
    )
    node_features = torch.zeros(
        batch_size,
        max_nodes,
        NODE_FEATURE_DIM,
        dtype=torch.float32,
    )

    for batch_index, record in enumerate(records):
        n_node = int(record["n_node"])

        atoms = torch.as_tensor(
            record["atom_types"],
            dtype=torch.long,
        )
        charges = _class_indices(
            record["charges"],
            CHARGE_TO_INDEX,
            field="formal charge",
        )
        hydrogens = _class_indices(
            record["hydrogen_counts"],
            HYDROGEN_TO_INDEX,
            field="hydrogen count",
        )
        bonds = torch.as_tensor(
            record["bond_matrix"],
            dtype=torch.long,
        )

        expected_node_shape = (n_node,)

        if any(
            tuple(tensor.shape) != expected_node_shape
            for tensor in (
                atoms,
                charges,
                hydrogens,
            )
        ):
            raise ValueError(
                "A node field does not match n_node."
            )

        if tuple(bonds.shape) != (n_node, n_node):
            raise ValueError(
                "bond_matrix does not match n_node."
            )

        atom_targets[
            batch_index, :n_node
        ] = atoms
        charge_targets[
            batch_index, :n_node
        ] = charges
        h_targets[
            batch_index, :n_node
        ] = hydrogens
        bond_targets[
            batch_index, :n_node, :n_node
        ] = bonds
        node_mask[
            batch_index, :n_node
        ] = True

        node_features[
            batch_index, :n_node
        ] = torch.cat(
            [
                F.one_hot(
                    atoms,
                    NUM_ATOM_TYPES,
                ),
                F.one_hot(
                    charges,
                    NUM_CHARGE_TYPES,
                ),
                F.one_hot(
                    hydrogens,
                    NUM_HYDROGEN_TYPES,
                ),
            ],
            dim=-1,
        ).float()

    # Valid pairs are always contained within one molecule.
    real_pairs = (
        node_mask.unsqueeze(2)
        & node_mask.unsqueeze(1)
    )

    diagonal = torch.eye(
        max_nodes,
        dtype=torch.bool,
    ).unsqueeze(0)

    # All real off-diagonal atom pairs.
    pair_mask = real_pairs & ~diagonal

    # Count each undirected pair once in reconstruction loss.
    upper_triangle = torch.triu(
        torch.ones(
            max_nodes,
            max_nodes,
            dtype=torch.bool,
        ),
        diagonal=1,
    ).unsqueeze(0)

    bond_loss_mask = (
        pair_mask
        & upper_triangle
    )

    # Autoencoder attention: real bonds plus self-loops.
    encoder_attention_mask = (
        ((bond_targets > 0) | diagonal)
        & real_pairs
    )

    # Flow attention: all real atoms, without bond information.
    flow_attention_mask = real_pairs.clone()

    conditions = np.stack(
        [
            np.asarray(
                record["condition"],
                dtype=np.float32,
            )
            for record in records
        ]
    )

    if conditions.shape != (batch_size, 2):
        raise ValueError(
            "Expected condition shape "
            f"{(batch_size, 2)}, "
            f"received {conditions.shape}."
        )

    return {
        "node_features": node_features,
        "atom_targets": atom_targets,
        "charge_targets": charge_targets,
        "h_targets": h_targets,
        "bond_targets": bond_targets,
        "node_mask": node_mask,
        "pair_mask": pair_mask,
        "bond_loss_mask": bond_loss_mask,
        "encoder_attention_mask": (
            encoder_attention_mask
        ),
        "flow_attention_mask": (
            flow_attention_mask
        ),
        "n_node": torch.tensor(
            [
                int(record["n_node"])
                for record in records
            ],
            dtype=torch.long,
        ),
        "condition": torch.from_numpy(
            conditions
        ),
        "qed": torch.tensor(
            [
                float(record["qed"])
                for record in records
            ],
            dtype=torch.float32,
        ),
        "smiles": [
            str(record["smiles"])
            for record in records
        ],
    }


def batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device | str,
) -> dict[str, Any]:
    """Move tensor fields while preserving SMILES/metadata."""
    return {
        key: (
            value.to(device)
            if torch.is_tensor(value)
            else value
        )
        for key, value in batch.items()
    }


def load_preprocessed_splits(
    path: str | Path,
    *,
    include_test: bool = False,
) -> dict[str, Any]:
    """Load train/validation; test requires explicit opt-in."""
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)

    required = {
        "train",
        "val",
        "test",
    }
    missing = required.difference(payload)

    if missing:
        raise KeyError(
            "Preprocessed payload is missing: "
            + ", ".join(sorted(missing))
        )

    result = {
        "train": payload["train"],
        "val": payload["val"],
        "metadata": {
            key: value
            for key, value in payload.items()
            if key not in required
        },
    }

    if include_test:
        result["test"] = payload["test"]

    return result


def make_dataloaders(
    split_path: str | Path,
    *,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    include_test: bool = False,
    pin_memory: bool = True,
) -> dict[str, DataLoader]:
    """Create reproducible train/validation/test loaders."""
    if batch_size < 1:
        raise ValueError(
            "batch_size must be positive."
        )

    splits = load_preprocessed_splits(
        split_path,
        include_test=include_test,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    common_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_molecules,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    loaders = {
        "train": DataLoader(
            ZincDataset(splits["train"]),
            shuffle=True,
            generator=generator,
            **common_options,
        ),
        "val": DataLoader(
            ZincDataset(splits["val"]),
            shuffle=False,
            **common_options,
        ),
    }

    if include_test:
        loaders["test"] = DataLoader(
            ZincDataset(splits["test"]),
            shuffle=False,
            **common_options,
        )

    return loaders


__all__ = [
    "ZincDataset",
    "batch_to_device",
    "collate_molecules",
    "load_preprocessed_splits",
    "make_dataloaders",
]
