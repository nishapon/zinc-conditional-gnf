"""Property-aware molecular-autoencoder training.

Stage A trains only the QED prediction head while keeping the
autoencoder frozen.

Stage B jointly fine-tunes the autoencoder and QED head using:

    reconstruction_loss + qed_loss_weight * QED_MSE

Only training and validation loaders are accepted by the calling CLI.
This module never loads the test split itself.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from zinc_gnf.data.datasets import batch_to_device
from zinc_gnf.models.autoencoder import make_node_noise
from zinc_gnf.training.reconstruction import (
    merge_reconstruction_counts,
    reconstruction_counts,
    reconstruction_loss,
    reconstruction_metrics,
)


RECONSTRUCTION_LOSS_NAMES = (
    "atom",
    "charge",
    "h",
    "bond",
    "total",
)


def _validate_positive(
    value: float,
    *,
    name: str,
) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _new_generator(
    device: torch.device,
    seed: int,
) -> torch.Generator:
    generator_device = (
        "cuda"
        if device.type == "cuda"
        else "cpu"
    )

    generator = torch.Generator(
        device=generator_device
    )
    generator.manual_seed(int(seed))

    return generator


def _qed_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    """Return molecular-level QED regression metrics."""
    predictions = predictions.detach().float().cpu().reshape(-1)
    targets = targets.detach().float().cpu().reshape(-1)

    if predictions.shape != targets.shape:
        raise ValueError(
            "QED predictions and targets have different shapes: "
            f"{tuple(predictions.shape)} and {tuple(targets.shape)}."
        )

    if predictions.numel() == 0:
        raise ValueError(
            "Cannot calculate QED metrics for an empty set."
        )

    residual = predictions - targets

    mse = float(
        residual.square().mean().item()
    )
    mae = float(
        residual.abs().mean().item()
    )
    rmse = math.sqrt(mse)

    centered_prediction = (
        predictions - predictions.mean()
    )
    centered_target = targets - targets.mean()

    correlation_denominator = torch.sqrt(
        centered_prediction.square().sum()
        * centered_target.square().sum()
    )

    if correlation_denominator.item() > 0:
        correlation = float(
            (
                centered_prediction
                * centered_target
            ).sum().item()
            / correlation_denominator.item()
        )
    else:
        correlation = float("nan")

    target_variance = (
        centered_target.square().sum()
    )

    if target_variance.item() > 0:
        r_squared = float(
            1.0
            - residual.square().sum().item()
            / target_variance.item()
        )
    else:
        r_squared = float("nan")

    return {
        "qed_mse": mse,
        "qed_mae": mae,
        "qed_rmse": rmse,
        "qed_correlation": correlation,
        "qed_r_squared": r_squared,
        "qed_prediction_mean": float(
            predictions.mean().item()
        ),
        "qed_target_mean": float(
            targets.mean().item()
        ),
    }


def _finish_qed_epoch(
    *,
    prediction_groups: list[torch.Tensor],
    target_groups: list[torch.Tensor],
    weighted_loss_sum: float,
    molecules: int,
) -> dict[str, float]:
    if molecules <= 0:
        raise ValueError(
            "The data loader produced no molecules."
        )

    predictions = torch.cat(
        prediction_groups,
        dim=0,
    )
    targets = torch.cat(
        target_groups,
        dim=0,
    )

    metrics = _qed_metrics(
        predictions,
        targets,
    )
    metrics["loss"] = (
        weighted_loss_sum / molecules
    )
    metrics["molecules"] = float(molecules)

    return metrics


def train_qed_head_epoch(
    autoencoder: nn.Module,
    qed_head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str,
    noise_dim: int,
    noise_generator: torch.Generator | None = None,
    gradient_clip: float | None = 5.0,
) -> dict[str, float]:
    """Warm up the QED head with a frozen autoencoder."""
    resolved_device = torch.device(device)

    autoencoder.eval()
    qed_head.train()

    weighted_loss_sum = 0.0
    molecules = 0
    prediction_groups: list[torch.Tensor] = []
    target_groups: list[torch.Tensor] = []

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

        # The warm-up stage must not update the autoencoder.
        with torch.no_grad():
            embeddings = autoencoder.encoder(
                batch,
                noise,
            )

        optimizer.zero_grad(set_to_none=True)

        qed_prediction = qed_head(
            embeddings,
            batch["node_mask"],
        )
        qed_target = batch["qed"]

        qed_loss = F.mse_loss(
            qed_prediction,
            qed_target,
        )

        if not torch.isfinite(qed_loss):
            raise FloatingPointError(
                "Non-finite QED-head training loss."
            )

        qed_loss.backward()

        if gradient_clip is not None:
            _validate_positive(
                float(gradient_clip),
                name="gradient_clip",
            )
            torch.nn.utils.clip_grad_norm_(
                qed_head.parameters(),
                max_norm=float(gradient_clip),
            )

        optimizer.step()

        weighted_loss_sum += (
            float(qed_loss.detach().item())
            * batch_size
        )
        molecules += batch_size

        prediction_groups.append(
            qed_prediction.detach().cpu()
        )
        target_groups.append(
            qed_target.detach().cpu()
        )

    return _finish_qed_epoch(
        prediction_groups=prediction_groups,
        target_groups=target_groups,
        weighted_loss_sum=weighted_loss_sum,
        molecules=molecules,
    )


@torch.inference_mode()
def evaluate_qed_head(
    autoencoder: nn.Module,
    qed_head: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device | str,
    noise_dim: int,
    noise_seed: int,
) -> dict[str, float]:
    """Evaluate QED prediction with deterministic node noise."""
    resolved_device = torch.device(device)
    noise_generator = _new_generator(
        resolved_device,
        noise_seed,
    )

    autoencoder.eval()
    qed_head.eval()

    weighted_loss_sum = 0.0
    molecules = 0
    prediction_groups: list[torch.Tensor] = []
    target_groups: list[torch.Tensor] = []

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

        embeddings = autoencoder.encoder(
            batch,
            noise,
        )
        qed_prediction = qed_head(
            embeddings,
            batch["node_mask"],
        )
        qed_target = batch["qed"]

        qed_loss = F.mse_loss(
            qed_prediction,
            qed_target,
        )

        if not torch.isfinite(qed_loss):
            raise FloatingPointError(
                "Non-finite QED-head validation loss."
            )

        weighted_loss_sum += (
            float(qed_loss.item())
            * batch_size
        )
        molecules += batch_size

        prediction_groups.append(
            qed_prediction.cpu()
        )
        target_groups.append(
            qed_target.cpu()
        )

    return _finish_qed_epoch(
        prediction_groups=prediction_groups,
        target_groups=target_groups,
        weighted_loss_sum=weighted_loss_sum,
        molecules=molecules,
    )


def train_property_aware_epoch(
    autoencoder: nn.Module,
    qed_head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str,
    noise_dim: int,
    bond_class_weight: float,
    qed_loss_weight: float,
    noise_generator: torch.Generator | None = None,
    gradient_clip: float | None = 5.0,
) -> dict[str, float]:
    """Jointly fine-tune reconstruction and QED supervision."""
    _validate_positive(
        float(qed_loss_weight),
        name="qed_loss_weight",
    )

    resolved_device = torch.device(device)

    autoencoder.train()
    qed_head.train()

    reconstruction_sums = {
        name: 0.0
        for name in RECONSTRUCTION_LOSS_NAMES
    }
    total_loss_sum = 0.0
    qed_loss_sum = 0.0
    molecules = 0

    count_groups: list[Mapping[str, int]] = []
    prediction_groups: list[torch.Tensor] = []
    target_groups: list[torch.Tensor] = []

    trainable_parameters = [
        parameter
        for module in (
            autoencoder,
            qed_head,
        )
        for parameter in module.parameters()
        if parameter.requires_grad
    ]

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

        optimizer.zero_grad(set_to_none=True)

        embeddings, predictions = autoencoder(
            batch,
            noise,
        )

        reconstruction_losses = reconstruction_loss(
            predictions,
            batch,
            bond_class_weight=bond_class_weight,
        )

        qed_prediction = qed_head(
            embeddings,
            batch["node_mask"],
        )
        qed_target = batch["qed"]

        qed_loss = F.mse_loss(
            qed_prediction,
            qed_target,
        )

        total_loss = (
            reconstruction_losses["total"]
            + float(qed_loss_weight) * qed_loss
        )

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                "Non-finite property-aware training loss."
            )

        total_loss.backward()

        if gradient_clip is not None:
            _validate_positive(
                float(gradient_clip),
                name="gradient_clip",
            )
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=float(gradient_clip),
            )

        optimizer.step()

        for name in RECONSTRUCTION_LOSS_NAMES:
            reconstruction_sums[name] += (
                float(
                    reconstruction_losses[
                        name
                    ].detach().item()
                )
                * batch_size
            )

        total_loss_sum += (
            float(total_loss.detach().item())
            * batch_size
        )
        qed_loss_sum += (
            float(qed_loss.detach().item())
            * batch_size
        )
        molecules += batch_size

        count_groups.append(
            reconstruction_counts(
                predictions,
                batch,
            )
        )
        prediction_groups.append(
            qed_prediction.detach().cpu()
        )
        target_groups.append(
            qed_target.detach().cpu()
        )

    if molecules <= 0:
        raise ValueError(
            "The training loader produced no molecules."
        )

    metrics = {
        f"reconstruction_{name}_loss": (
            reconstruction_sums[name]
            / molecules
        )
        for name in RECONSTRUCTION_LOSS_NAMES
    }
    metrics["total_loss"] = (
        total_loss_sum / molecules
    )

    qed_metrics = _qed_metrics(
        torch.cat(prediction_groups),
        torch.cat(target_groups),
    )
    metrics.update(qed_metrics)
    metrics["qed_loss"] = (
        qed_loss_sum / molecules
    )

    merged_counts = merge_reconstruction_counts(
        count_groups
    )
    metrics.update(
        reconstruction_metrics(
            merged_counts
        )
    )
    metrics["molecules"] = float(molecules)

    return metrics


@torch.inference_mode()
def evaluate_property_aware(
    autoencoder: nn.Module,
    qed_head: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device | str,
    noise_dim: int,
    bond_class_weight: float,
    qed_loss_weight: float,
    noise_seed: int,
) -> dict[str, float]:
    """Evaluate reconstruction and QED prediction together."""
    _validate_positive(
        float(qed_loss_weight),
        name="qed_loss_weight",
    )

    resolved_device = torch.device(device)
    noise_generator = _new_generator(
        resolved_device,
        noise_seed,
    )

    autoencoder.eval()
    qed_head.eval()

    reconstruction_sums = {
        name: 0.0
        for name in RECONSTRUCTION_LOSS_NAMES
    }
    total_loss_sum = 0.0
    qed_loss_sum = 0.0
    molecules = 0

    count_groups: list[Mapping[str, int]] = []
    prediction_groups: list[torch.Tensor] = []
    target_groups: list[torch.Tensor] = []

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

        embeddings, predictions = autoencoder(
            batch,
            noise,
        )

        reconstruction_losses = reconstruction_loss(
            predictions,
            batch,
            bond_class_weight=bond_class_weight,
        )

        qed_prediction = qed_head(
            embeddings,
            batch["node_mask"],
        )
        qed_target = batch["qed"]

        qed_loss = F.mse_loss(
            qed_prediction,
            qed_target,
        )
        total_loss = (
            reconstruction_losses["total"]
            + float(qed_loss_weight) * qed_loss
        )

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                "Non-finite property-aware validation loss."
            )

        for name in RECONSTRUCTION_LOSS_NAMES:
            reconstruction_sums[name] += (
                float(
                    reconstruction_losses[
                        name
                    ].item()
                )
                * batch_size
            )

        total_loss_sum += (
            float(total_loss.item())
            * batch_size
        )
        qed_loss_sum += (
            float(qed_loss.item())
            * batch_size
        )
        molecules += batch_size

        count_groups.append(
            reconstruction_counts(
                predictions,
                batch,
            )
        )
        prediction_groups.append(
            qed_prediction.cpu()
        )
        target_groups.append(
            qed_target.cpu()
        )

    if molecules <= 0:
        raise ValueError(
            "The validation loader produced no molecules."
        )

    metrics = {
        f"reconstruction_{name}_loss": (
            reconstruction_sums[name]
            / molecules
        )
        for name in RECONSTRUCTION_LOSS_NAMES
    }
    metrics["total_loss"] = (
        total_loss_sum / molecules
    )

    qed_metrics = _qed_metrics(
        torch.cat(prediction_groups),
        torch.cat(target_groups),
    )
    metrics.update(qed_metrics)
    metrics["qed_loss"] = (
        qed_loss_sum / molecules
    )

    merged_counts = merge_reconstruction_counts(
        count_groups
    )
    metrics.update(
        reconstruction_metrics(
            merged_counts
        )
    )
    metrics["molecules"] = float(molecules)

    return metrics


def property_metric_summary(
    metrics: Mapping[str, Any],
) -> str:
    """Compact property-aware epoch summary."""
    return (
        f"total={float(metrics['total_loss']):.5f} | "
        f"recon={float(metrics['reconstruction_total_loss']):.5f} | "
        f"QED MAE={float(metrics['qed_mae']):.5f} | "
        f"corr={float(metrics['qed_correlation']):.4f} | "
        f"atom={100.0 * float(metrics['atom_accuracy']):.2f}% | "
        f"real-bond="
        f"{100.0 * float(metrics['real_bond_accuracy']):.2f}% | "
        f"exact={100.0 * float(metrics['exact_graph_accuracy']):.2f}%"
    )


__all__ = [
    "evaluate_property_aware",
    "evaluate_qed_head",
    "property_metric_summary",
    "train_property_aware_epoch",
    "train_qed_head_epoch",
]
