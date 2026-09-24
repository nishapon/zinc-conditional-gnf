"""Tests for embedding extraction and sharding."""

import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from zinc_gnf.chemistry import encode_zinc_smiles
from zinc_gnf.data.datasets import (
    ZincDataset,
    collate_molecules,
)
from zinc_gnf.data.embeddings import (
    RunningFeatureMoments,
    extract_embedding_split,
    pack_embedding_records,
    unpack_molecule_embedding,
)
from zinc_gnf.models.autoencoder import (
    MolecularAutoencoder,
    MolecularEmbeddingQEDHead,
)


def _record(smiles, condition):
    record, reason = encode_zinc_smiles(smiles)

    assert reason is None
    assert record is not None

    record["condition"] = np.asarray(
        condition,
        dtype=np.float32,
    )
    return record


def _loader():
    records = [
        _record("CC", (-1.0, -0.5)),
        _record("CCO", (-0.5, -0.2)),
        _record("CCN", (0.5, 0.2)),
        _record("c1ccccc1", (1.0, 0.5)),
    ]

    return DataLoader(
        ZincDataset(records),
        batch_size=2,
        shuffle=False,
        collate_fn=collate_molecules,
    )


def _models():
    torch.manual_seed(42)

    autoencoder = MolecularAutoencoder(
        noise_dim=4,
        embedding_dim=16,
        hidden_dim=32,
        message_passing_steps=1,
        num_heads=4,
    )

    qed_head = MolecularEmbeddingQEDHead(
        embedding_dim=16,
        hidden_dim=32,
        dropout=0.0,
    )

    return autoencoder, qed_head


def _load(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def test_running_moments_matches_direct_calculation():
    values = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 8.0],
        ]
    )

    moments = RunningFeatureMoments(2)
    moments.update(values[:1])
    moments.update(values[1:])

    mean, std = moments.finalize()

    assert moments.count == 3
    assert torch.allclose(
        mean,
        values.mean(dim=0),
    )
    assert torch.allclose(
        std,
        values.std(
            dim=0,
            unbiased=False,
        ),
    )


def test_pack_unpack_round_trip():
    first = torch.randn(2, 4)
    second = torch.randn(3, 4)

    shard = pack_embedding_records(
        [first, second],
        [
            torch.tensor([-1.0, 0.0]),
            torch.tensor([1.0, 0.5]),
        ],
        [0.4, 0.8],
        ["CC", "CCO"],
    )

    assert shard["offsets"].tolist() == [
        0,
        2,
        5,
    ]
    assert shard["n_node"].tolist() == [
        2,
        3,
    ]
    assert torch.equal(
        unpack_molecule_embedding(shard, 0),
        first,
    )
    assert torch.equal(
        unpack_molecule_embedding(shard, 1),
        second,
    )


def test_deterministic_sharded_extraction(tmp_path):
    autoencoder, qed_head = _models()

    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"

    first_moments = RunningFeatureMoments(16)
    second_moments = RunningFeatureMoments(16)

    first = extract_embedding_split(
        autoencoder,
        qed_head,
        _loader(),
        output_directory=first_directory,
        split_name="train",
        device="cpu",
        noise_dim=4,
        noise_seed=123,
        shard_size=2,
        moments=first_moments,
    )

    second = extract_embedding_split(
        autoencoder,
        qed_head,
        _loader(),
        output_directory=second_directory,
        split_name="train",
        device="cpu",
        noise_dim=4,
        noise_seed=123,
        shard_size=2,
        moments=second_moments,
    )

    assert first["molecules"] == 4
    assert first["nodes"] == first_moments.count
    assert len(first["shards"]) == 2

    for metric in (
        "qed_mae",
        "qed_rmse",
        "qed_correlation",
    ):
        assert math.isfinite(first[metric])

    first_mean, first_std = (
        first_moments.finalize()
    )
    second_mean, second_std = (
        second_moments.finalize()
    )

    assert torch.equal(
        first_mean,
        second_mean,
    )
    assert torch.equal(
        first_std,
        second_std,
    )
    assert (first_std > 0).all()

    for first_info, second_info in zip(
        first["shards"],
        second["shards"],
    ):
        first_shard = _load(
            first_directory
            / first_info["path"]
        )
        second_shard = _load(
            second_directory
            / second_info["path"]
        )

        for key in (
            "node_embeddings",
            "offsets",
            "condition",
            "n_node",
            "qed",
        ):
            assert torch.equal(
                first_shard[key],
                second_shard[key],
            )

        assert (
            first_shard["smiles"]
            == second_shard["smiles"]
        )
