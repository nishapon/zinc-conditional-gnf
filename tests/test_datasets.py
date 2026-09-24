"""Tests for molecular graph batching and split loading."""

from __future__ import annotations

import pickle

import numpy as np
import torch

from zinc_gnf.chemistry import encode_zinc_smiles
from zinc_gnf.constants import NODE_FEATURE_DIM
from zinc_gnf.data.datasets import (
    batch_to_device,
    collate_molecules,
    load_preprocessed_splits,
    make_dataloaders,
)


def _record(
    smiles: str,
    condition: tuple[float, float],
) -> dict:
    record, reason = encode_zinc_smiles(smiles)

    assert reason is None
    assert record is not None

    record["condition"] = np.asarray(
        condition,
        dtype=np.float32,
    )
    return record


def _records() -> list[dict]:
    return [
        _record("CC", (-1.0, -0.5)),
        _record("CCO", (0.0, 0.0)),
        _record("c1ccccc1", (1.0, 0.5)),
    ]


def test_collate_shapes_and_padding():
    records = _records()
    batch = collate_molecules(records)

    assert batch["node_features"].shape == (
        3,
        6,
        NODE_FEATURE_DIM,
    )
    assert batch["bond_targets"].shape == (3, 6, 6)
    assert batch["condition"].shape == (3, 2)

    assert batch["node_mask"].sum(dim=1).tolist() == [
        2,
        3,
        6,
    ]

    assert torch.all(batch["atom_targets"][0, 2:] == -100)
    assert torch.all(batch["node_features"][0, 2:] == 0)


def test_masks_have_intended_semantics():
    batch = collate_molecules(_records())
    n_node = batch["n_node"]

    # Ordered atom pairs excluding diagonal.
    expected_pair_count = n_node * (n_node - 1)
    assert (
        batch["pair_mask"].sum(dim=(1, 2)).tolist()
        == expected_pair_count.tolist()
    )

    # Unique undirected pairs used for bond loss.
    expected_bond_loss_count = n_node * (n_node - 1) // 2
    assert (
        batch["bond_loss_mask"].sum(dim=(1, 2)).tolist()
        == expected_bond_loss_count.tolist()
    )

    # Flow attention permits all pairs of real nodes.
    assert (
        batch["flow_attention_mask"].sum(dim=(1, 2)).tolist()
        == n_node.square().tolist()
    )

    # Encoder attention contains a self-loop for every real node.
    encoder_diagonal = batch[
        "encoder_attention_mask"
    ].diagonal(dim1=1, dim2=2)

    assert torch.equal(
        encoder_diagonal,
        batch["node_mask"],
    )


def test_node_features_are_one_hot_triplets():
    batch = collate_molecules(_records())

    active_features = batch["node_features"][
        batch["node_mask"]
    ]

    # One atom class, one charge class, and one H-count class.
    assert torch.all(active_features.sum(dim=1) == 3)


def test_batch_to_device_preserves_non_tensor_fields():
    batch = collate_molecules(_records())
    moved = batch_to_device(batch, "cpu")

    assert moved["smiles"] == batch["smiles"]

    for value in moved.values():
        if torch.is_tensor(value):
            assert value.device.type == "cpu"


def test_test_split_requires_explicit_opt_in(tmp_path):
    records = _records()
    split_path = tmp_path / "splits.pkl"

    with split_path.open("wb") as handle:
        pickle.dump(
            {
                "train": records,
                "val": records[:1],
                "test": records[1:],
            },
            handle,
        )

    normal = load_preprocessed_splits(split_path)

    assert set(normal) == {
        "train",
        "val",
        "metadata",
    }

    explicit = load_preprocessed_splits(
        split_path,
        include_test=True,
    )

    assert "test" in explicit
    assert len(explicit["test"]) == 2


def test_make_dataloaders_excludes_test_by_default(tmp_path):
    records = _records()
    split_path = tmp_path / "splits.pkl"

    with split_path.open("wb") as handle:
        pickle.dump(
            {
                "train": records,
                "val": records[:1],
                "test": records[1:],
            },
            handle,
        )

    loaders = make_dataloaders(
        split_path,
        batch_size=2,
        seed=42,
        pin_memory=False,
    )

    assert set(loaders) == {"train", "val"}

    batch = next(iter(loaders["train"]))

    assert batch["condition"].shape[-1] == 2
    assert batch["node_features"].shape[-1] == NODE_FEATURE_DIM
