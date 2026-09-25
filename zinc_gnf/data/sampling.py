"""Capped condition-aware sampling utilities."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import WeightedRandomSampler


def capped_qed_sample_weights(
    qed_values: Sequence[float] | torch.Tensor,
    *,
    bins: int = 20,
    power: float = 0.5,
    max_weight: float = 4.0,
) -> torch.Tensor:
    """Compute capped inverse-frequency QED weights."""
    values = torch.as_tensor(
        qed_values,
        dtype=torch.float64,
    ).flatten()

    if values.numel() == 0:
        raise ValueError("qed_values cannot be empty.")
    if bins < 2:
        raise ValueError("bins must be at least 2.")
    if not 0.0 < power <= 1.0:
        raise ValueError("power must be in (0, 1].")
    if max_weight < 1.0:
        raise ValueError(
            "max_weight must be at least 1."
        )
    if not torch.isfinite(values).all():
        raise ValueError("QED values must be finite.")
    if (values < 0.0).any() or (values > 1.0).any():
        raise ValueError(
            "QED values must be between 0 and 1."
        )

    bin_indices = torch.floor(
        values * bins
    ).long().clamp(
        min=0,
        max=bins - 1,
    )

    bin_counts = torch.bincount(
        bin_indices,
        minlength=bins,
    ).to(torch.float64)

    sample_counts = bin_counts[bin_indices]
    largest_count = bin_counts.max()

    weights = (
        largest_count / sample_counts
    ).pow(power)

    return weights.clamp(
        min=1.0,
        max=float(max_weight),
    )


def make_capped_qed_sampler(
    qed_values: Sequence[float] | torch.Tensor,
    *,
    bins: int = 20,
    power: float = 0.5,
    max_weight: float = 4.0,
    generator: torch.Generator | None = None,
) -> WeightedRandomSampler:
    """Sample rare QED regions more often without exceeding the cap."""
    weights = capped_qed_sample_weights(
        qed_values,
        bins=bins,
        power=power,
        max_weight=max_weight,
    )

    return WeightedRandomSampler(
        weights=weights,
        num_samples=int(weights.numel()),
        replacement=True,
        generator=generator,
    )


__all__ = [
    "capped_qed_sample_weights",
    "make_capped_qed_sampler",
]
