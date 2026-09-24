"""Training and validation engine for the molecular autoencoder."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from zinc_gnf.data.datasets import batch_to_device
from zinc_gnf.models.autoencoder import make_node_noise
from zinc_gnf.training.reconstruction import (
    merge_reconstruction_counts,
    reconstruction_counts,
    reconstruction_loss,
    reconstruction_metrics,
)


LOSS_NAMES = (
    "total",
    "atom",
    "charge",
    "h",
    "bond",
)


def _new_loss_totals() -> dict[str, float]:
    return {
        name: 0.0
        for name in LOSS_NAMES
    }


def _finalize_epoch(
    *,
    loss_totals: Mapping[str, float],
    count_groups: list[Mapping[str, int]],
    molecules: int,
) -> dict[str, float]:
    if molecules < 1:
        raise RuntimeError(
            "The data loader produced no molecules."
        )

    result = {
        f"{name}_loss": (
            float(loss_totals[name]) / molecules
        )
        for name in LOSS_NAMES
    }

    merged_counts = merge_reconstruction_counts(
        count_groups
    )
    result.update(
        reconstruction_metrics(merged_counts)
    )
    result["molecules"] = float(molecules)

    return result


def train_autoencoder_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str,
    noise_dim: int,
    bond_class_weight: float,
    gradient_clip: float | None = 5.0,
    noise_generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Train for one epoch and return aggregate graph metrics."""
    model.train()

    loss_totals = _new_loss_totals()
    count_groups: list[Mapping[str, int]] = []
    molecules = 0

    for raw_batch in loader:
        batch = batch_to_device(
            raw_batch,
            device,
        )
        batch_size = int(
            batch["node_features"].shape[0]
        )

        noise = make_node_noise(
            batch,
            noise_dim=noise_dim,
            generator=noise_generator,
        )

        optimizer.zero_grad(set_to_none=True)

        _, predictions = model(
            batch,
            noise,
        )

        losses = reconstruction_loss(
            predictions,
            batch,
            bond_class_weight=bond_class_weight,
        )

        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(
                "Non-finite autoencoder training loss."
            )

        losses["total"].backward()

        if gradient_clip is not None:
            if gradient_clip <= 0:
                raise ValueError(
                    "gradient_clip must be positive or None."
                )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(gradient_clip),
            )

        optimizer.step()

        for name in LOSS_NAMES:
            loss_totals[name] += (
                float(losses[name].detach().item())
                * batch_size
            )

        count_groups.append(
            reconstruction_counts(
                predictions,
                batch,
            )
        )
        molecules += batch_size

    return _finalize_epoch(
        loss_totals=loss_totals,
        count_groups=count_groups,
        molecules=molecules,
    )


@torch.inference_mode()
def evaluate_autoencoder(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device | str,
    noise_dim: int,
    bond_class_weight: float,
    noise_seed: int,
) -> dict[str, float]:
    """Evaluate with fixed validation noise for stable model selection."""
    model.eval()

    resolved_device = torch.device(device)
    generator_device = (
        resolved_device.type
        if resolved_device.type == "cuda"
        else "cpu"
    )

    noise_generator = torch.Generator(
        device=generator_device
    )
    noise_generator.manual_seed(
        int(noise_seed)
    )

    loss_totals = _new_loss_totals()
    count_groups: list[Mapping[str, int]] = []
    molecules = 0

    for raw_batch in loader:
        batch = batch_to_device(
            raw_batch,
            resolved_device,
        )
        batch_size = int(
            batch["node_features"].shape[0]
        )

        noise = make_node_noise(
            batch,
            noise_dim=noise_dim,
            generator=noise_generator,
        )

        _, predictions = model(
            batch,
            noise,
        )

        losses = reconstruction_loss(
            predictions,
            batch,
            bond_class_weight=bond_class_weight,
        )

        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(
                "Non-finite autoencoder validation loss."
            )

        for name in LOSS_NAMES:
            loss_totals[name] += (
                float(losses[name].item())
                * batch_size
            )

        count_groups.append(
            reconstruction_counts(
                predictions,
                batch,
            )
        )
        molecules += batch_size

    return _finalize_epoch(
        loss_totals=loss_totals,
        count_groups=count_groups,
        molecules=molecules,
    )


def metric_summary(
    metrics: Mapping[str, float],
) -> str:
    """Create a concise human-readable epoch report."""
    return (
        f"loss={metrics['total_loss']:.5f} | "
        f"atom={100.0 * metrics['atom_accuracy']:.2f}% | "
        f"charge={100.0 * metrics['charge_accuracy']:.2f}% | "
        f"H={100.0 * metrics['hydrogen_accuracy']:.2f}% | "
        f"bond={100.0 * metrics['bond_accuracy']:.2f}% | "
        f"real-bond="
        f"{100.0 * metrics['real_bond_accuracy']:.2f}% | "
        f"exact="
        f"{100.0 * metrics['exact_graph_accuracy']:.2f}%"
    )


__all__ = [
    "LOSS_NAMES",
    "evaluate_autoencoder",
    "metric_summary",
    "train_autoencoder_epoch",
]
