"""Losses and metrics for molecular graph reconstruction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from zinc_gnf.constants import NUM_BOND_TYPES


def masked_graph_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Average within each graph and then across graphs."""
    if values.shape != mask.shape:
        raise ValueError(
            f"Value shape {tuple(values.shape)} does not match "
            f"mask shape {tuple(mask.shape)}."
        )

    reduction_axes = tuple(range(1, values.ndim))

    counts = mask.sum(
        dim=reduction_axes
    ).clamp_min(1)

    sums = (
        values * mask.to(values.dtype)
    ).sum(dim=reduction_axes)

    return (sums / counts).mean()


def reconstruction_loss(
    predictions: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    *,
    bond_class_weight: float = 5.0,
) -> dict[str, torch.Tensor]:
    """Compute balanced node and weighted bond reconstruction losses."""
    if bond_class_weight <= 0:
        raise ValueError(
            "bond_class_weight must be positive."
        )

    node_mask = batch["node_mask"]
    pair_mask = batch["bond_loss_mask"]

    losses: dict[str, torch.Tensor] = {}

    for prediction_name, target_name in (
        ("atom", "atom_targets"),
        ("charge", "charge_targets"),
        ("h", "h_targets"),
    ):
        logits = predictions[
            f"{prediction_name}_logits"
        ]
        targets = batch[target_name]

        cross_entropy = F.cross_entropy(
            logits.transpose(1, 2),
            targets,
            ignore_index=-100,
            reduction="none",
        )

        losses[prediction_name] = masked_graph_mean(
            cross_entropy,
            node_mask,
        )

    bond_logits = predictions["bond_logits"]
    bond_targets = batch["bond_targets"]

    bond_cross_entropy = F.cross_entropy(
        bond_logits.permute(0, 3, 1, 2),
        bond_targets,
        reduction="none",
    )

    class_weights = torch.ones(
        NUM_BOND_TYPES,
        dtype=bond_logits.dtype,
        device=bond_logits.device,
    )
    class_weights[1:] = float(bond_class_weight)

    element_weights = class_weights[bond_targets]
    effective_weights = (
        element_weights
        * pair_mask.to(element_weights.dtype)
    )

    # Weighted mean inside each molecule prevents large molecules
    # from dominating the minibatch objective.
    per_graph_bond_loss = (
        (
            bond_cross_entropy
            * effective_weights
        ).sum(dim=(1, 2))
        / effective_weights.sum(
            dim=(1, 2)
        ).clamp_min(1.0)
    )

    losses["bond"] = per_graph_bond_loss.mean()

    losses["total"] = (
        losses["atom"]
        + losses["charge"]
        + losses["h"]
        + losses["bond"]
    )

    return losses


def reconstruction_counts(
    predictions: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
) -> dict[str, int]:
    """Return additive counts for reconstruction evaluation."""
    node_mask = batch["node_mask"]
    pair_mask = batch["bond_loss_mask"]

    counts: dict[str, int] = {
        "molecules": int(node_mask.shape[0]),
    }

    all_nodes_correct = torch.ones_like(
        node_mask,
        dtype=torch.bool,
    )

    for prediction_name, target_name in (
        ("atom", "atom_targets"),
        ("charge", "charge_targets"),
        ("h", "h_targets"),
    ):
        predicted_classes = predictions[
            f"{prediction_name}_logits"
        ].argmax(dim=-1)

        correct = (
            predicted_classes
            == batch[target_name]
        )

        counts[
            f"{prediction_name}_correct"
        ] = int(
            (correct & node_mask).sum().item()
        )

        # Padded positions do not affect exact reconstruction.
        all_nodes_correct &= (
            correct | ~node_mask
        )

    predicted_bonds = predictions[
        "bond_logits"
    ].argmax(dim=-1)

    bond_correct = (
        predicted_bonds
        == batch["bond_targets"]
    )

    real_bonds = (
        pair_mask
        & (batch["bond_targets"] > 0)
    )

    counts["nodes"] = int(
        node_mask.sum().item()
    )
    counts["pairs"] = int(
        pair_mask.sum().item()
    )
    counts["real_bonds"] = int(
        real_bonds.sum().item()
    )

    counts["bond_correct"] = int(
        (bond_correct & pair_mask).sum().item()
    )
    counts["real_bond_correct"] = int(
        (bond_correct & real_bonds).sum().item()
    )

    graph_exact = (
        all_nodes_correct.all(dim=1)
        & (
            bond_correct | ~pair_mask
        ).all(dim=2).all(dim=1)
    )

    counts["exact_graphs"] = int(
        graph_exact.sum().item()
    )

    return counts


def merge_reconstruction_counts(
    count_groups: Sequence[Mapping[str, int]],
) -> dict[str, int]:
    """Combine additive reconstruction counts across batches."""
    merged: dict[str, int] = {}

    for group in count_groups:
        for name, value in group.items():
            merged[name] = (
                merged.get(name, 0)
                + int(value)
            )

    return merged


def _safe_ratio(
    numerator: int,
    denominator: int,
) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def reconstruction_metrics(
    counts: Mapping[str, int],
) -> dict[str, float]:
    """Convert additive counts to reportable reconstruction rates."""
    required = {
        "molecules",
        "nodes",
        "pairs",
        "real_bonds",
        "atom_correct",
        "charge_correct",
        "h_correct",
        "bond_correct",
        "real_bond_correct",
        "exact_graphs",
    }

    missing = required.difference(counts)

    if missing:
        raise KeyError(
            f"Missing reconstruction counts: "
            f"{sorted(missing)}."
        )

    return {
        "atom_accuracy": _safe_ratio(
            counts["atom_correct"],
            counts["nodes"],
        ),
        "charge_accuracy": _safe_ratio(
            counts["charge_correct"],
            counts["nodes"],
        ),
        "hydrogen_accuracy": _safe_ratio(
            counts["h_correct"],
            counts["nodes"],
        ),
        "bond_accuracy": _safe_ratio(
            counts["bond_correct"],
            counts["pairs"],
        ),
        "real_bond_accuracy": _safe_ratio(
            counts["real_bond_correct"],
            counts["real_bonds"],
        ),
        "exact_graph_accuracy": _safe_ratio(
            counts["exact_graphs"],
            counts["molecules"],
        ),
    }


__all__ = [
    "masked_graph_mean",
    "merge_reconstruction_counts",
    "reconstruction_counts",
    "reconstruction_loss",
    "reconstruction_metrics",
]
