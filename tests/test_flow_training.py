"""Tests for sharded flow data and likelihood training."""

from __future__ import annotations

import json
import math

import torch

from zinc_gnf.data.embeddings import (
    atomic_torch_save,
    pack_embedding_records,
)
from zinc_gnf.data.flow_dataset import (
    EmbeddingShardDataset,
    make_flow_dataloaders,
)
from zinc_gnf.models.flow import ConditionalNodeGNF
from zinc_gnf.training.flow import (
    evaluate_flow,
    flow_loss,
    flow_metric_summary,
    train_flow_epoch,
)


def _molecule(
    n_node,
    offset,
):
    values = torch.arange(
        n_node * 4,
        dtype=torch.float32,
    ).reshape(n_node, 4)

    return values / 10.0 + offset


def _write_shard(
    root,
    relative_path,
    embeddings,
    conditions,
    qeds,
    smiles,
):
    payload = pack_embedding_records(
        embeddings,
        [
            torch.tensor(
                condition,
                dtype=torch.float32,
            )
            for condition in conditions
        ],
        qeds,
        smiles,
    )

    atomic_torch_save(
        payload,
        root / relative_path,
    )

    return {
        "path": str(relative_path),
        "molecules": len(embeddings),
        "nodes": int(
            payload[
                "node_embeddings"
            ].shape[0]
        ),
    }


def _embedding_root(tmp_path):
    root = tmp_path / "embeddings"
    root.mkdir()

    mean = torch.tensor(
        [1.0, 2.0, 3.0, 4.0]
    )
    std = torch.tensor(
        [2.0, 2.0, 4.0, 4.0]
    )

    atomic_torch_save(
        {
            "embedding_mean": mean,
            "embedding_std": std,
            "training_node_count": 14,
            "embedding_dim": 4,
        },
        root / "normalization.pt",
    )

    train_first = _write_shard(
        root,
        "train/train-00000.pt",
        [
            _molecule(2, 1.0),
            _molecule(3, 2.0),
            _molecule(4, 3.0),
        ],
        [
            (-1.0, -0.5),
            (-0.5, -0.2),
            (0.0, 0.0),
        ],
        [0.3, 0.5, 0.7],
        ["CC", "CCO", "CCCC"],
    )

    train_second = _write_shard(
        root,
        "train/train-00001.pt",
        [
            _molecule(2, 4.0),
            _molecule(3, 5.0),
        ],
        [
            (0.5, 0.2),
            (1.0, 0.5),
        ],
        [0.8, 0.9],
        ["CN", "CCN"],
    )

    validation = _write_shard(
        root,
        "val/val-00000.pt",
        [
            _molecule(2, 1.5),
            _molecule(3, 2.5),
        ],
        [
            (-0.2, -0.1),
            (0.2, 0.1),
        ],
        [0.55, 0.65],
        ["CO", "CNO"],
    )

    manifest = {
        "embedding_dim": 4,
        "normalization": "normalization.pt",
        "training": {
            "molecules": 5,
            "nodes": 14,
            "shards": [
                train_first,
                train_second,
            ],
        },
        "validation": {
            "molecules": 2,
            "nodes": 5,
            "shards": [validation],
        },
        "test_split_accessed": False,
    }

    (root / "manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    return root, mean, std


def _flow():
    torch.manual_seed(42)

    return ConditionalNodeGNF(
        embedding_dim=4,
        condition_dim=2,
        hidden_dim=16,
        context_dim=8,
        num_heads=2,
        num_blocks=2,
    )


def test_embedding_shard_dataset_normalizes_and_indexes(
    tmp_path,
):
    root, mean, std = _embedding_root(
        tmp_path
    )

    dataset = EmbeddingShardDataset(
        root,
        split="train",
        cache_size=1,
    )

    assert len(dataset) == 5
    assert dataset.embedding_dim == 4

    first = dataset[0]
    fourth = dataset[3]
    last = dataset[-1]

    torch.testing.assert_close(
        first["h"],
        (
            _molecule(2, 1.0)
            - mean
        ) / std,
    )
    torch.testing.assert_close(
        fourth["h"],
        (
            _molecule(2, 4.0)
            - mean
        ) / std,
    )

    assert first["n_node"] == 2
    assert fourth["smiles"] == "CN"
    assert last["smiles"] == "CCN"


def test_flow_dataloaders_create_correct_masks(
    tmp_path,
):
    root, _, _ = _embedding_root(
        tmp_path
    )

    loaders = make_flow_dataloaders(
        root,
        batch_size=3,
        seed=42,
        num_workers=0,
        pin_memory=False,
    )

    assert set(loaders) == {
        "train",
        "val",
    }
    assert len(
        loaders["train"].dataset
    ) == 5
    assert len(
        loaders["val"].dataset
    ) == 2

    validation_batch = next(
        iter(loaders["val"])
    )

    assert validation_batch["h"].shape == (
        2,
        3,
        4,
    )
    assert validation_batch[
        "condition"
    ].shape == (2, 2)
    assert validation_batch[
        "mask"
    ].dtype == torch.bool
    assert validation_batch[
        "mask"
    ].sum(dim=1).tolist() == [
        2,
        3,
    ]


def test_flow_training_and_evaluation(
    tmp_path,
):
    root, _, _ = _embedding_root(
        tmp_path
    )

    loaders = make_flow_dataloaders(
        root,
        batch_size=2,
        seed=42,
    )

    model = _flow()

    before = {
        name: value.detach().clone()
        for name, value
        in model.state_dict().items()
    }

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    training = train_flow_epoch(
        model,
        loaders["train"],
        optimizer,
        device="cpu",
        gradient_clip=5.0,
    )

    changed = any(
        not torch.equal(
            before[name],
            model.state_dict()[name],
        )
        for name in before
    )

    assert changed

    required = {
        "loss",
        "nll_per_node",
        "nll_per_dimension",
        "mean_log_probability",
        "mean_prior_log_probability",
        "mean_logdet",
        "molecules",
        "nodes",
        "prior_sigma_min",
        "prior_sigma_max",
    }

    assert set(training) == required

    for value in training.values():
        assert math.isfinite(value)

    assert training["molecules"] == 5.0
    assert training["nodes"] == 14.0
    assert training[
        "prior_sigma_min"
    ] > 0

    first_validation = evaluate_flow(
        model,
        loaders["val"],
        device="cpu",
    )
    second_validation = evaluate_flow(
        model,
        loaders["val"],
        device="cpu",
    )

    assert first_validation == second_validation
    assert first_validation[
        "molecules"
    ] == 2.0

    summary = flow_metric_summary(
        first_validation
    )

    assert "NLL/node=" in summary
    assert "NLL/dim=" in summary
    assert "sigma=" in summary

    batch = next(
        iter(loaders["val"])
    )
    losses = flow_loss(
        model,
        batch,
    )

    assert torch.isfinite(
        losses["loss"]
    )
