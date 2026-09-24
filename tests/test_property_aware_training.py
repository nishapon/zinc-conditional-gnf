"""Tests for property-aware autoencoder training."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from zinc_gnf.chemistry import encode_zinc_smiles
from zinc_gnf.data.datasets import (
    ZincDataset,
    collate_molecules,
)
from zinc_gnf.models.autoencoder import (
    MolecularAutoencoder,
    MolecularEmbeddingQEDHead,
)
from zinc_gnf.training.property_aware import (
    evaluate_property_aware,
    evaluate_qed_head,
    property_metric_summary,
    train_property_aware_epoch,
    train_qed_head_epoch,
)


def _record(smiles: str) -> dict:
    record, reason = encode_zinc_smiles(smiles)

    assert reason is None
    assert record is not None

    # Conditions are included because the standard molecular
    # collator requires [standardized QED, standardized log1p(N)].
    record["condition"] = np.asarray(
        [0.0, 0.0],
        dtype=np.float32,
    )

    return record


def _loader() -> DataLoader:
    records = [
        _record("CC"),
        _record("CCO"),
        _record("CCN"),
        _record("CC(=O)O"),
        _record("c1ccccc1"),
        _record("CCCl"),
    ]

    return DataLoader(
        ZincDataset(records),
        batch_size=3,
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


def _snapshot(module: torch.nn.Module):
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
    }


def _parameters_changed(
    before: dict[str, torch.Tensor],
    module: torch.nn.Module,
) -> bool:
    after = module.state_dict()

    return any(
        not torch.equal(
            before[name],
            after[name],
        )
        for name in before
    )


def _parameters_unchanged(
    before: dict[str, torch.Tensor],
    module: torch.nn.Module,
) -> bool:
    after = module.state_dict()

    return all(
        torch.equal(
            before[name],
            after[name],
        )
        for name in before
    )


def _assert_finite_metrics(metrics):
    assert metrics

    for name, value in metrics.items():
        if name == "qed_correlation":
            # Correlation can only be undefined for a constant vector.
            assert math.isfinite(value) or math.isnan(value)
        else:
            assert math.isfinite(value), (
                name,
                value,
            )


def test_qed_head_warmup_keeps_autoencoder_frozen():
    loader = _loader()
    autoencoder, qed_head = _models()

    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)

    autoencoder_before = _snapshot(autoencoder)
    qed_head_before = _snapshot(qed_head)

    optimizer = torch.optim.Adam(
        qed_head.parameters(),
        lr=1e-2,
    )

    generator = torch.Generator().manual_seed(101)

    metrics = train_qed_head_epoch(
        autoencoder,
        qed_head,
        loader,
        optimizer,
        device="cpu",
        noise_dim=4,
        noise_generator=generator,
        gradient_clip=5.0,
    )

    _assert_finite_metrics(metrics)

    assert _parameters_unchanged(
        autoencoder_before,
        autoencoder,
    )
    assert _parameters_changed(
        qed_head_before,
        qed_head,
    )

    assert metrics["molecules"] == 6.0
    assert 0.0 <= metrics["qed_mae"] <= 1.0


def test_qed_validation_is_deterministic():
    loader = _loader()
    autoencoder, qed_head = _models()

    first = evaluate_qed_head(
        autoencoder,
        qed_head,
        loader,
        device="cpu",
        noise_dim=4,
        noise_seed=123,
    )
    second = evaluate_qed_head(
        autoencoder,
        qed_head,
        loader,
        device="cpu",
        noise_dim=4,
        noise_seed=123,
    )

    assert first == second
    _assert_finite_metrics(first)


def test_joint_training_updates_both_models():
    loader = _loader()
    autoencoder, qed_head = _models()

    autoencoder_before = _snapshot(autoencoder)
    qed_head_before = _snapshot(qed_head)

    optimizer = torch.optim.Adam(
        list(autoencoder.parameters())
        + list(qed_head.parameters()),
        lr=1e-3,
    )

    generator = torch.Generator().manual_seed(202)

    metrics = train_property_aware_epoch(
        autoencoder,
        qed_head,
        loader,
        optimizer,
        device="cpu",
        noise_dim=4,
        bond_class_weight=5.0,
        qed_loss_weight=2.0,
        noise_generator=generator,
        gradient_clip=5.0,
    )

    _assert_finite_metrics(metrics)

    assert _parameters_changed(
        autoencoder_before,
        autoencoder,
    )
    assert _parameters_changed(
        qed_head_before,
        qed_head,
    )

    required_metrics = {
        "total_loss",
        "reconstruction_total_loss",
        "qed_loss",
        "qed_mae",
        "qed_rmse",
        "qed_correlation",
        "qed_r_squared",
        "atom_accuracy",
        "real_bond_accuracy",
        "exact_graph_accuracy",
        "molecules",
    }

    assert required_metrics.issubset(metrics)
    assert metrics["molecules"] == 6.0


def test_property_validation_is_deterministic():
    loader = _loader()
    autoencoder, qed_head = _models()

    first = evaluate_property_aware(
        autoencoder,
        qed_head,
        loader,
        device="cpu",
        noise_dim=4,
        bond_class_weight=5.0,
        qed_loss_weight=2.0,
        noise_seed=456,
    )
    second = evaluate_property_aware(
        autoencoder,
        qed_head,
        loader,
        device="cpu",
        noise_dim=4,
        bond_class_weight=5.0,
        qed_loss_weight=2.0,
        noise_seed=456,
    )

    assert first == second
    _assert_finite_metrics(first)

    summary = property_metric_summary(first)

    assert "QED MAE=" in summary
    assert "real-bond=" in summary
    assert "exact=" in summary


def test_nonpositive_qed_weight_is_rejected():
    loader = _loader()
    autoencoder, qed_head = _models()

    optimizer = torch.optim.Adam(
        list(autoencoder.parameters())
        + list(qed_head.parameters()),
        lr=1e-3,
    )

    with pytest.raises(
        ValueError,
        match="qed_loss_weight must be positive",
    ):
        train_property_aware_epoch(
            autoencoder,
            qed_head,
            loader,
            optimizer,
            device="cpu",
            noise_dim=4,
            bond_class_weight=5.0,
            qed_loss_weight=0.0,
        )
