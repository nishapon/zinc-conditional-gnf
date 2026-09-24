"""Maximum-likelihood training for the conditional node GNF."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from zinc_gnf.data.flow_dataset import (
    flow_batch_to_device,
)


def flow_loss(
    model: nn.Module,
    batch: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Conditional likelihood objective for one padded batch."""
    output = model.log_prob(
        batch["h"],
        batch["condition"],
        batch["mask"],
    )

    node_count = batch["mask"].sum(
        dim=1
    ).to(batch["h"].dtype).clamp_min(1.0)

    nll_per_node = (
        -output["log_prob"] / node_count
    )

    # Dividing by the fixed embedding dimension keeps gradients
    # numerically stable. It does not change the optimum.
    nll_per_dimension = (
        nll_per_node
        / batch["h"].shape[-1]
    )

    loss = nll_per_dimension.mean()

    if not torch.isfinite(loss):
        raise FloatingPointError(
            "Non-finite conditional flow loss."
        )

    return {
        "loss": loss,
        "nll_per_node": (
            nll_per_node.mean()
        ),
        "nll_per_dimension": (
            nll_per_dimension.mean()
        ),
        "mean_log_probability": (
            output["log_prob"].mean()
        ),
        "mean_prior_log_probability": (
            output[
                "prior_log_prob"
            ].mean()
        ),
        "mean_logdet": (
            output["logdet"].mean()
        ),
    }


def _new_totals() -> dict[str, float]:
    return {
        "loss": 0.0,
        "nll_per_node": 0.0,
        "nll_per_dimension": 0.0,
        "mean_log_probability": 0.0,
        "mean_prior_log_probability": 0.0,
        "mean_logdet": 0.0,
    }


def _finish_metrics(
    totals: Mapping[str, float],
    *,
    molecules: int,
    nodes: int,
    sigma_minimum: float,
    sigma_maximum: float,
) -> dict[str, float]:
    if molecules <= 0:
        raise ValueError(
            "The flow loader produced no molecules."
        )

    metrics = {
        name: float(value) / molecules
        for name, value in totals.items()
    }
    metrics.update(
        {
            "molecules": float(molecules),
            "nodes": float(nodes),
            "prior_sigma_min": float(
                sigma_minimum
            ),
            "prior_sigma_max": float(
                sigma_maximum
            ),
        }
    )

    return metrics


def train_flow_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str,
    gradient_clip: float | None = 5.0,
) -> dict[str, float]:
    """Train the conditional GNF for one epoch."""
    model.train()
    resolved_device = torch.device(device)

    totals = _new_totals()
    molecules = 0
    nodes = 0
    sigma_minimum = float("inf")
    sigma_maximum = float("-inf")

    for raw_batch in loader:
        batch = flow_batch_to_device(
            raw_batch,
            resolved_device,
        )
        batch_size = int(
            batch["h"].shape[0]
        )

        optimizer.zero_grad(set_to_none=True)

        losses = flow_loss(
            model,
            batch,
        )
        losses["loss"].backward()

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

        with torch.no_grad():
            _, sigma = model.prior_parameters(
                batch["condition"]
            )

            sigma_minimum = min(
                sigma_minimum,
                float(sigma.min().item()),
            )
            sigma_maximum = max(
                sigma_maximum,
                float(sigma.max().item()),
            )

        for name in totals:
            totals[name] += (
                float(
                    losses[name].detach().item()
                )
                * batch_size
            )

        molecules += batch_size
        nodes += int(
            batch["mask"].sum().item()
        )

    return _finish_metrics(
        totals,
        molecules=molecules,
        nodes=nodes,
        sigma_minimum=sigma_minimum,
        sigma_maximum=sigma_maximum,
    )


@torch.inference_mode()
def evaluate_flow(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device | str,
) -> dict[str, float]:
    """Evaluate conditional likelihood without changing weights."""
    model.eval()
    resolved_device = torch.device(device)

    totals = _new_totals()
    molecules = 0
    nodes = 0
    sigma_minimum = float("inf")
    sigma_maximum = float("-inf")

    for raw_batch in loader:
        batch = flow_batch_to_device(
            raw_batch,
            resolved_device,
        )
        batch_size = int(
            batch["h"].shape[0]
        )

        losses = flow_loss(
            model,
            batch,
        )
        _, sigma = model.prior_parameters(
            batch["condition"]
        )

        sigma_minimum = min(
            sigma_minimum,
            float(sigma.min().item()),
        )
        sigma_maximum = max(
            sigma_maximum,
            float(sigma.max().item()),
        )

        for name in totals:
            totals[name] += (
                float(losses[name].item())
                * batch_size
            )

        molecules += batch_size
        nodes += int(
            batch["mask"].sum().item()
        )

    return _finish_metrics(
        totals,
        molecules=molecules,
        nodes=nodes,
        sigma_minimum=sigma_minimum,
        sigma_maximum=sigma_maximum,
    )


def flow_metric_summary(
    metrics: Mapping[str, float],
) -> str:
    return (
        f"NLL/node="
        f"{float(metrics['nll_per_node']):.5f} | "
        f"NLL/dim="
        f"{float(metrics['nll_per_dimension']):.6f} | "
        f"logdet="
        f"{float(metrics['mean_logdet']):.3f} | "
        f"sigma="
        f"{float(metrics['prior_sigma_min']):.3f}–"
        f"{float(metrics['prior_sigma_max']):.3f}"
    )


__all__ = [
    "evaluate_flow",
    "flow_loss",
    "flow_metric_summary",
    "train_flow_epoch",
]
