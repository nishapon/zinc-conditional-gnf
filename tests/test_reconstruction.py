"""Tests for reconstruction losses and metrics."""

from __future__ import annotations

import numpy as np
import torch

from zinc_gnf.chemistry import encode_zinc_smiles
from zinc_gnf.constants import (
    NUM_ATOM_TYPES,
    NUM_BOND_TYPES,
    NUM_CHARGE_TYPES,
    NUM_HYDROGEN_TYPES,
)
from zinc_gnf.data.datasets import collate_molecules
from zinc_gnf.training.reconstruction import (
    masked_graph_mean,
    merge_reconstruction_counts,
    reconstruction_counts,
    reconstruction_loss,
    reconstruction_metrics,
)


def _record(smiles: str) -> dict:
    record, reason = encode_zinc_smiles(smiles)

    assert reason is None
    assert record is not None

    record["condition"] = np.asarray(
        [0.0, 0.0],
        dtype=np.float32,
    )
    return record


def _batch() -> dict:
    return collate_molecules(
        [
            _record("CCO"),
            _record("CC(=O)O"),
        ]
    )


def _logits_from_targets(
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    logits = torch.full(
        (*targets.shape, num_classes),
        -20.0,
    )

    safe_targets = targets.clamp_min(0)

    logits.scatter_(
        dim=-1,
        index=safe_targets.unsqueeze(-1),
        value=20.0,
    )

    return logits


def _perfect_predictions(
    batch: dict,
) -> dict[str, torch.Tensor]:
    return {
        "atom_logits": _logits_from_targets(
            batch["atom_targets"],
            NUM_ATOM_TYPES,
        ),
        "charge_logits": _logits_from_targets(
            batch["charge_targets"],
            NUM_CHARGE_TYPES,
        ),
        "h_logits": _logits_from_targets(
            batch["h_targets"],
            NUM_HYDROGEN_TYPES,
        ),
        "bond_logits": _logits_from_targets(
            batch["bond_targets"],
            NUM_BOND_TYPES,
        ),
    }


def _clone_predictions(
    predictions: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key: value.clone()
        for key, value in predictions.items()
    }


def test_masked_graph_mean_balances_graphs():
    values = torch.tensor(
        [
            [2.0, 2.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )
    mask = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
        ]
    )

    result = masked_graph_mean(values, mask)

    # Graph means are 2 and 4, so their balanced mean is 3.
    torch.testing.assert_close(
        result,
        torch.tensor(3.0),
    )


def test_perfect_predictions_have_perfect_metrics():
    batch = _batch()
    predictions = _perfect_predictions(batch)

    losses = reconstruction_loss(
        predictions,
        batch,
        bond_class_weight=5.0,
    )
    counts = reconstruction_counts(
        predictions,
        batch,
    )
    metrics = reconstruction_metrics(counts)

    assert losses["total"].item() < 1e-5
    assert losses["atom"].item() < 1e-5
    assert losses["charge"].item() < 1e-5
    assert losses["h"].item() < 1e-5
    assert losses["bond"].item() < 1e-5

    assert counts["exact_graphs"] == counts["molecules"]

    for value in metrics.values():
        assert value == 1.0


def test_padding_does_not_affect_counts():
    batch = _batch()
    perfect = _perfect_predictions(batch)
    modified = _clone_predictions(perfect)

    padded_nodes = ~batch["node_mask"]

    for key in (
        "atom_logits",
        "charge_logits",
        "h_logits",
    ):
        modified[key][padded_nodes] = 1000.0

    padded_pairs = ~(
        batch["node_mask"].unsqueeze(2)
        & batch["node_mask"].unsqueeze(1)
    )
    modified["bond_logits"][padded_pairs] = 1000.0

    perfect_counts = reconstruction_counts(
        perfect,
        batch,
    )
    modified_counts = reconstruction_counts(
        modified,
        batch,
    )

    assert modified_counts == perfect_counts


def test_real_errors_are_detected_and_weighted():
    batch = _batch()
    perfect = _perfect_predictions(batch)
    incorrect = _clone_predictions(perfect)

    # Force one real atom prediction to the wrong class.
    true_atom = int(
        batch["atom_targets"][0, 0].item()
    )
    wrong_atom = (
        true_atom + 1
    ) % NUM_ATOM_TYPES

    incorrect["atom_logits"][0, 0] = -20.0
    incorrect["atom_logits"][
        0,
        0,
        wrong_atom,
    ] = 20.0

    # Force one real bond to no-bond.
    real_bond_positions = torch.nonzero(
        batch["bond_loss_mask"]
        & (batch["bond_targets"] > 0),
        as_tuple=False,
    )

    batch_index, atom_i, atom_j = (
        real_bond_positions[0].tolist()
    )

    incorrect["bond_logits"][
        batch_index,
        atom_i,
        atom_j,
    ] = -20.0

    incorrect["bond_logits"][
        batch_index,
        atom_i,
        atom_j,
        0,
    ] = 20.0

    counts = reconstruction_counts(
        incorrect,
        batch,
    )
    metrics = reconstruction_metrics(counts)

    assert counts["atom_correct"] == (
        counts["nodes"] - 1
    )
    assert counts["real_bond_correct"] == (
        counts["real_bonds"] - 1
    )
    assert metrics["exact_graph_accuracy"] < 1.0

    unweighted_loss = reconstruction_loss(
        incorrect,
        batch,
        bond_class_weight=1.0,
    )
    weighted_loss = reconstruction_loss(
        incorrect,
        batch,
        bond_class_weight=10.0,
    )

    assert (
        weighted_loss["bond"]
        > unweighted_loss["bond"]
    )


def test_reconstruction_loss_is_differentiable():
    torch.manual_seed(42)
    batch = _batch()

    batch_size, padded_nodes = (
        batch["node_mask"].shape
    )

    predictions = {
        "atom_logits": torch.randn(
            batch_size,
            padded_nodes,
            NUM_ATOM_TYPES,
            requires_grad=True,
        ),
        "charge_logits": torch.randn(
            batch_size,
            padded_nodes,
            NUM_CHARGE_TYPES,
            requires_grad=True,
        ),
        "h_logits": torch.randn(
            batch_size,
            padded_nodes,
            NUM_HYDROGEN_TYPES,
            requires_grad=True,
        ),
        "bond_logits": torch.randn(
            batch_size,
            padded_nodes,
            padded_nodes,
            NUM_BOND_TYPES,
            requires_grad=True,
        ),
    }

    losses = reconstruction_loss(
        predictions,
        batch,
        bond_class_weight=5.0,
    )

    assert torch.isfinite(losses["total"])

    losses["total"].backward()

    for logits in predictions.values():
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert logits.grad.abs().sum() > 0


def test_counts_merge_across_batches():
    batch = _batch()
    predictions = _perfect_predictions(batch)

    counts = reconstruction_counts(
        predictions,
        batch,
    )

    merged = merge_reconstruction_counts(
        [counts, counts]
    )

    for key, value in counts.items():
        assert merged[key] == 2 * value

    metrics = reconstruction_metrics(merged)

    for value in metrics.values():
        assert value == 1.0
