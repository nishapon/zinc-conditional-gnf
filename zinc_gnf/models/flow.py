"""Conditional node-level graph normalizing flow.

The flow models normalized molecular node embeddings conditioned on:

    condition[:, 0] = standardized QED
    condition[:, 1] = standardized log1p(heavy-atom count)

The transformation is permutation equivariant across molecular nodes,
padding independent, and exactly invertible.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalAttentionST(nn.Module):
    """Predict affine-coupling scale and translation."""

    def __init__(
        self,
        *,
        half_dim: int,
        hidden_dim: int,
        context_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        if half_dim <= 0:
            raise ValueError(
                "half_dim must be positive."
            )
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads."
            )

        self.half_dim = int(half_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = (
            self.hidden_dim // self.num_heads
        )

        self.input_projection = nn.Linear(
            self.half_dim,
            self.hidden_dim,
        )

        # Feature-wise linear modulation from QED and node count.
        self.film = nn.Linear(
            context_dim,
            2 * self.hidden_dim,
        )
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        self.qkv_projection = nn.Linear(
            self.hidden_dim,
            3 * self.hidden_dim,
            bias=False,
        )
        self.attention_output = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
        )

        self.normalization = nn.LayerNorm(
            self.hidden_dim
        )
        self.hidden_network = nn.Sequential(
            nn.Linear(
                self.hidden_dim,
                self.hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                self.hidden_dim,
                self.hidden_dim,
            ),
            nn.SiLU(),
        )
        self.output_projection = nn.Linear(
            self.hidden_dim,
            2 * self.half_dim,
        )

        # Every coupling transformation starts as identity.
        nn.init.zeros_(
            self.output_projection.weight
        )
        nn.init.zeros_(
            self.output_projection.bias
        )

    def _split_heads(
        self,
        value: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, node_count, _ = value.shape

        return value.reshape(
            batch_size,
            node_count,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def forward(
        self,
        values: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 3:
            raise ValueError(
                "values must have shape [B, N, D]."
            )
        if mask.shape != values.shape[:2]:
            raise ValueError(
                "mask shape does not match values."
            )

        node_mask = mask.unsqueeze(-1).to(
            values.dtype
        )

        hidden = self.input_projection(
            values
        )

        gamma, beta = self.film(
            context
        ).chunk(2, dim=-1)

        hidden = (
            hidden
            * (1.0 + gamma.unsqueeze(1))
            + beta.unsqueeze(1)
        )
        hidden = hidden * node_mask

        query, key, value = (
            self.qkv_projection(hidden).chunk(
                3,
                dim=-1,
            )
        )

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        logits = (
            query @ key.transpose(-1, -2)
        ) / math.sqrt(self.head_dim)

        allowed = (
            mask[:, None, :, None]
            & mask[:, None, None, :]
        )

        logits = logits.masked_fill(
            ~allowed,
            torch.finfo(logits.dtype).min,
        )

        weights = torch.softmax(
            logits,
            dim=-1,
        )
        weights = weights.masked_fill(
            ~allowed,
            0.0,
        )
        weights = weights / weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-12)

        messages = (
            weights @ value
        ).transpose(1, 2).reshape(
            values.shape[0],
            values.shape[1],
            self.hidden_dim,
        )

        hidden = self.normalization(
            hidden
            + self.attention_output(messages)
        )
        hidden = self.hidden_network(hidden)

        scale_translation = (
            self.output_projection(hidden)
            * node_mask
        )

        return scale_translation.chunk(
            2,
            dim=-1,
        )


class FlowActNorm(nn.Module):
    """Learned invertible per-feature affine transformation."""

    def __init__(
        self,
        dimension: int,
    ) -> None:
        super().__init__()

        self.log_scale = nn.Parameter(
            torch.zeros(dimension)
        )
        self.shift = nn.Parameter(
            torch.zeros(dimension)
        )

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        float_mask = mask.unsqueeze(-1).to(
            values.dtype
        )

        transformed = (
            values * self.log_scale.exp()
            + self.shift
        ) * float_mask

        log_determinant = (
            mask.sum(dim=1).to(values.dtype)
            * self.log_scale.sum()
        )

        return transformed, log_determinant

    def inverse(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        float_mask = mask.unsqueeze(-1).to(
            values.dtype
        )

        recovered = (
            (values - self.shift)
            * (-self.log_scale).exp()
        ) * float_mask

        inverse_log_determinant = (
            -mask.sum(dim=1).to(values.dtype)
            * self.log_scale.sum()
        )

        return recovered, inverse_log_determinant


class ConditionalCouplingBlock(nn.Module):
    """Two-sided conditional affine coupling block."""

    def __init__(
        self,
        *,
        dimension: int,
        hidden_dim: int,
        context_dim: int,
        num_heads: int,
        max_log_scale: float,
    ) -> None:
        super().__init__()

        if dimension % 2 != 0:
            raise ValueError(
                "Flow dimension must be even."
            )
        if max_log_scale <= 0:
            raise ValueError(
                "max_log_scale must be positive."
            )

        half_dim = dimension // 2

        self.max_log_scale = float(
            max_log_scale
        )
        self.actnorm = FlowActNorm(
            dimension
        )

        self.first_coupling = (
            ConditionalAttentionST(
                half_dim=half_dim,
                hidden_dim=hidden_dim,
                context_dim=context_dim,
                num_heads=num_heads,
            )
        )
        self.second_coupling = (
            ConditionalAttentionST(
                half_dim=half_dim,
                hidden_dim=hidden_dim,
                context_dim=context_dim,
                num_heads=num_heads,
            )
        )

    def bounded_log_scale(
        self,
        raw_scale: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.max_log_scale
            * torch.tanh(
                raw_scale
                / self.max_log_scale
            )
        )

    def forward(
        self,
        values: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values, log_determinant = (
            self.actnorm(
                values,
                mask,
            )
        )

        first, second = values.chunk(
            2,
            dim=-1,
        )

        scale, translation = (
            self.first_coupling(
                first,
                context,
                mask,
            )
        )
        scale = self.bounded_log_scale(
            scale
        )

        second = (
            second * scale.exp()
            + translation
        )
        log_determinant = (
            log_determinant
            + scale.sum(dim=(1, 2))
        )

        scale, translation = (
            self.second_coupling(
                second,
                context,
                mask,
            )
        )
        scale = self.bounded_log_scale(
            scale
        )

        first = (
            first * scale.exp()
            + translation
        )
        log_determinant = (
            log_determinant
            + scale.sum(dim=(1, 2))
        )

        transformed = torch.cat(
            [first, second],
            dim=-1,
        )

        return transformed, log_determinant

    def inverse(
        self,
        values: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        first, second = values.chunk(
            2,
            dim=-1,
        )
        inverse_log_determinant = (
            values.new_zeros(values.shape[0])
        )

        scale, translation = (
            self.second_coupling(
                second,
                context,
                mask,
            )
        )
        scale = self.bounded_log_scale(
            scale
        )
        first = (
            first - translation
        ) * (-scale).exp()
        inverse_log_determinant = (
            inverse_log_determinant
            - scale.sum(dim=(1, 2))
        )

        scale, translation = (
            self.first_coupling(
                first,
                context,
                mask,
            )
        )
        scale = self.bounded_log_scale(
            scale
        )
        second = (
            second - translation
        ) * (-scale).exp()
        inverse_log_determinant = (
            inverse_log_determinant
            - scale.sum(dim=(1, 2))
        )

        combined = torch.cat(
            [first, second],
            dim=-1,
        )

        recovered, actnorm_log_determinant = (
            self.actnorm.inverse(
                combined,
                mask,
            )
        )
        inverse_log_determinant = (
            inverse_log_determinant
            + actnorm_log_determinant
        )

        return recovered, inverse_log_determinant


class ConditionalNodeGNF(nn.Module):
    """Conditional invertible flow over molecular node embeddings."""

    def __init__(
        self,
        *,
        embedding_dim: int = 64,
        condition_dim: int = 2,
        hidden_dim: int = 128,
        context_dim: int = 64,
        num_heads: int = 4,
        num_blocks: int = 12,
        max_log_scale: float = 0.25,
        minimum_sigma: float = 0.1,
    ) -> None:
        super().__init__()

        if embedding_dim % 2 != 0:
            raise ValueError(
                "embedding_dim must be even."
            )
        if num_blocks <= 0:
            raise ValueError(
                "num_blocks must be positive."
            )
        if condition_dim <= 0:
            raise ValueError(
                "condition_dim must be positive."
            )
        if not 0 < minimum_sigma < 1:
            raise ValueError(
                "minimum_sigma must lie in (0, 1)."
            )

        self.embedding_dim = int(
            embedding_dim
        )
        self.condition_dim = int(
            condition_dim
        )
        self.hidden_dim = int(hidden_dim)
        self.context_dim = int(context_dim)
        self.num_heads = int(num_heads)
        self.num_blocks = int(num_blocks)
        self.max_log_scale = float(
            max_log_scale
        )
        self.minimum_sigma = float(
            minimum_sigma
        )

        self.context_network = nn.Sequential(
            nn.Linear(
                self.condition_dim,
                self.context_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                self.context_dim,
                self.context_dim,
            ),
        )

        self.blocks = nn.ModuleList(
            [
                ConditionalCouplingBlock(
                    dimension=self.embedding_dim,
                    hidden_dim=self.hidden_dim,
                    context_dim=self.context_dim,
                    num_heads=self.num_heads,
                    max_log_scale=(
                        self.max_log_scale
                    ),
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.prior_head = nn.Linear(
            self.context_dim,
            2 * self.embedding_dim,
        )

        # Conditional prior begins as N(0, I).
        nn.init.zeros_(
            self.prior_head.weight
        )
        nn.init.zeros_(
            self.prior_head.bias
        )

        self.sigma_offset = math.log(
            math.expm1(
                1.0 - self.minimum_sigma
            )
        )

    def _validate_inputs(
        self,
        values: torch.Tensor,
        condition: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if values.ndim != 3:
            raise ValueError(
                "values must have shape [B, N, D]."
            )
        if values.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.embedding_dim}, received "
                f"{values.shape[-1]}."
            )
        if condition.shape != (
            values.shape[0],
            self.condition_dim,
        ):
            raise ValueError(
                "Condition shape does not match flow input."
            )
        if mask.shape != values.shape[:2]:
            raise ValueError(
                "Mask shape does not match flow input."
            )
        if mask.dtype != torch.bool:
            raise TypeError(
                "Flow mask must be boolean."
            )

    def prior_parameters(
        self,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            condition.ndim != 2
            or condition.shape[-1]
            != self.condition_dim
        ):
            raise ValueError(
                "Invalid condition shape."
            )

        context = self.context_network(
            condition
        )
        mean, raw_sigma = self.prior_head(
            context
        ).chunk(2, dim=-1)

        sigma = (
            F.softplus(
                raw_sigma
                + self.sigma_offset
            )
            + self.minimum_sigma
        )

        return mean, sigma

    def forward(
        self,
        embeddings: torch.Tensor,
        condition: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(
            embeddings,
            condition,
            mask,
        )

        context = self.context_network(
            condition
        )
        float_mask = mask.unsqueeze(-1).to(
            embeddings.dtype
        )

        latent = embeddings * float_mask
        log_determinant = embeddings.new_zeros(
            embeddings.shape[0]
        )

        for block in self.blocks:
            latent, block_log_determinant = (
                block(
                    latent,
                    context,
                    mask,
                )
            )
            log_determinant = (
                log_determinant
                + block_log_determinant
            )

        return latent, log_determinant

    def inverse(
        self,
        latent: torch.Tensor,
        condition: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(
            latent,
            condition,
            mask,
        )

        context = self.context_network(
            condition
        )
        float_mask = mask.unsqueeze(-1).to(
            latent.dtype
        )

        embeddings = latent * float_mask
        inverse_log_determinant = (
            latent.new_zeros(latent.shape[0])
        )

        for block in reversed(self.blocks):
            (
                embeddings,
                block_log_determinant,
            ) = block.inverse(
                embeddings,
                context,
                mask,
            )
            inverse_log_determinant = (
                inverse_log_determinant
                + block_log_determinant
            )

        return (
            embeddings * float_mask,
            inverse_log_determinant,
        )

    def log_prob(
        self,
        embeddings: torch.Tensor,
        condition: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        latent, log_determinant = self(
            embeddings,
            condition,
            mask,
        )

        mean, sigma = self.prior_parameters(
            condition
        )

        standardized = (
            latent - mean.unsqueeze(1)
        ) / sigma.unsqueeze(1)

        per_feature_log_probability = (
            -0.5 * standardized.square()
            - sigma.unsqueeze(1).log()
            - 0.5 * math.log(
                2.0 * math.pi
            )
        )

        prior_log_probability = (
            per_feature_log_probability
            * mask.unsqueeze(-1).to(
                embeddings.dtype
            )
        ).sum(dim=(1, 2))

        log_probability = (
            prior_log_probability
            + log_determinant
        )

        valid_dimensions = (
            mask.sum(dim=1).to(
                embeddings.dtype
            )
            * self.embedding_dim
        ).clamp_min(1.0)

        return {
            "log_prob": log_probability,
            "prior_log_prob": (
                prior_log_probability
            ),
            "logdet": log_determinant,
            "nll_per_dimension": (
                -log_probability
                / valid_dimensions
            ),
            "latent": latent,
        }

    def sample_latent(
        self,
        condition: torch.Tensor,
        n_node: torch.Tensor,
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if temperature <= 0:
            raise ValueError(
                "temperature must be positive."
            )

        n_node = n_node.to(
            device=condition.device,
            dtype=torch.long,
        )

        if n_node.ndim != 1:
            raise ValueError(
                "n_node must have shape [B]."
            )
        if n_node.shape[0] != condition.shape[0]:
            raise ValueError(
                "n_node and condition batch sizes differ."
            )
        if (n_node <= 0).any():
            raise ValueError(
                "Every graph must contain at least one node."
            )

        maximum_nodes = int(
            n_node.max().item()
        )
        mask = (
            torch.arange(
                maximum_nodes,
                device=condition.device,
            ).unsqueeze(0)
            < n_node.unsqueeze(1)
        )

        mean, sigma = self.prior_parameters(
            condition
        )

        epsilon = torch.randn(
            condition.shape[0],
            maximum_nodes,
            self.embedding_dim,
            device=condition.device,
            dtype=condition.dtype,
            generator=generator,
        )

        latent = (
            mean.unsqueeze(1)
            + float(temperature)
            * sigma.unsqueeze(1)
            * epsilon
        )
        latent = (
            latent
            * mask.unsqueeze(-1).to(
                latent.dtype
            )
        )

        return latent, mask

    def sample(
        self,
        condition: torch.Tensor,
        n_node: torch.Tensor,
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent, mask = self.sample_latent(
            condition,
            n_node,
            temperature=temperature,
            generator=generator,
        )

        embeddings, _ = self.inverse(
            latent,
            condition,
            mask,
        )

        return embeddings, mask


def build_flow_from_config(
    configuration: Mapping[str, Any],
) -> ConditionalNodeGNF:
    """Construct the conditional flow from the YAML flow section."""
    return ConditionalNodeGNF(
        embedding_dim=int(
            configuration.get(
                "embedding_dim",
                64,
            )
        ),
        condition_dim=int(
            configuration.get(
                "condition_dim",
                2,
            )
        ),
        hidden_dim=int(
            configuration.get(
                "hidden_dim",
                128,
            )
        ),
        context_dim=int(
            configuration.get(
                "context_dim",
                64,
            )
        ),
        num_heads=int(
            configuration.get(
                "attention_heads",
                4,
            )
        ),
        num_blocks=int(
            configuration.get(
                "coupling_blocks",
                12,
            )
        ),
        max_log_scale=float(
            configuration.get(
                "max_log_scale",
                0.25,
            )
        ),
        minimum_sigma=float(
            configuration.get(
                "minimum_sigma",
                0.1,
            )
        ),
    )


__all__ = [
    "ConditionalAttentionST",
    "ConditionalCouplingBlock",
    "ConditionalNodeGNF",
    "FlowActNorm",
    "build_flow_from_config",
]
