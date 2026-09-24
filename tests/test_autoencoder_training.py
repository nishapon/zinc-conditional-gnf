"""Tests for autoencoder training and validation."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from zinc_gnf.chemistry import encode_zinc_smiles
from zinc_gnf.data.datasets import (
    ZincDataset,
    collate_molecules,
)
from zinc_gnf.models.autoencoder import MolecularAutoencoder
from zinc_gnf.training.autoencoder import (
    evaluate_autoencoder,
    metric_summary,
    train_autoencoder_epoch,
)


EXPECTED_METRICS = {
    "total_loss",
    "atom_loss",
    "charge_loss",
    "h_loss",
    "bond_loss",
    "atom_accuracy",
    "charge_accuracy",
    "hydrogen_accuracy",
    "bond_accuracy",
    "real_bond_accuracy",
    "exact_graph_accuracy",
    "molecules",
}


def _record(smiles: str) -> dict:
    record, reason = encode_zinc_smiles(smiles)

    assert reason is None
    assert record is not None

    record["condition"] = np.asarray(
        [0.0, 0.0],
        dtype=np.float32,
    )
    return record


def _loader() -> DataLoader:
    records = [
        _record("CC"),
        _record("CCO"),
        _record("CC(=O)O"),
        _record("CCN"),
    ]

    return DataLoader(
        ZincDataset(records),
        batch_size=2,
        shuffle=False,
        collate_fn=collate_molecules,
    )


def _model() -> MolecularAutoencoder:
    torch.manual_seed(42)

    return MolecularAutoencoder(
        noise_dim=4,
        embedding_dim=16,
        hidden_dim=32,
        message_passing_steps=1,
        num_heads=4,
    )


def _assert_complete_finite_metrics(
    metrics: dict[str, float],
) -> None:
    assert set(metrics) == EXPECTED_METRICS

    for name, value in metrics.items():
        assert math.isfinite(value), (
            name,
            value,
        )


def test_training_epoch_updates_model_and_returns_metrics():
    model = _model()
    loader = _loader()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    generator = torch.Generator().manual_seed(100)

    metrics = train_autoencoder_epoch(
        model,
        loader,
        optimizer,
        device="cpu",
        noise_dim=4,
        bond_class_weight=5.0,
        gradient_clip=5.0,
        noise_generator=generator,
    )

    _assert_complete_finite_metrics(metrics)

    assert metrics["molecules"] == 4.0

    changed = any(
        not torch.equal(
            before[name],
            parameter.detach(),
        )
        for name, parameter in model.named_parameters()
    )

    assert changed


def test_validation_is_deterministic_for_fixed_noise_seed():
    model = _model()
    loader = _loader()

    first = evaluate_autoencoder(
        model,
        loader,
        device="cpu",
        noise_dim=4,
        bond_class_weight=5.0,
        noise_seed=200,
    )
    second = evaluate_autoencoder(
        model,
        loader,
        device="cpu",
        noise_dim=4,
        bond_class_weight=5.0,
        noise_seed=200,
    )

    _assert_complete_finite_metrics(first)
    _assert_complete_finite_metrics(second)

    assert first == second


def test_validation_does_not_update_parameters():
    model = _model()
    loader = _loader()

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    evaluate_autoencoder(
        model,
        loader,
        device="cpu",
        noise_dim=4,
        bond_class_weight=5.0,
        noise_seed=300,
    )

    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter.detach(),
            before[name],
            atol=0,
            rtol=0,
        )


def test_metric_summary_contains_key_results():
    model = _model()
    metrics = evaluate_autoencoder(
        model,
        _loader(),
        device="cpu",
        noise_dim=4,
        bond_class_weight=5.0,
        noise_seed=400,
    )

    summary = metric_summary(metrics)

    assert "loss=" in summary
    assert "atom=" in summary
    assert "real-bond=" in summary
    assert "exact=" in summary
