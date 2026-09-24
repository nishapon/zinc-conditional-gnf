"""Bond-aware molecular autoencoder and embedding-level QED head."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn

from zinc_gnf.constants import (
    NODE_FEATURE_DIM,
    NUM_ATOM_TYPES,
    NUM_BOND_TYPES,
    NUM_CHARGE_TYPES,
    NUM_HYDROGEN_TYPES,
)


class BondAwareAttentionBlock(nn.Module):
    """Self-attention over bonded neighbours with bond-dependent messages."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_bond_types: int = NUM_BOND_TYPES,
    ) -> None:
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads."
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.attention_norm = nn.LayerNorm(embedding_dim)

        self.qkv_projection = nn.Linear(
            embedding_dim,
            3 * embedding_dim,
            bias=False,
        )

        # Bond class zero is also used for real-node self loops.
        self.bond_logit_bias = nn.Embedding(
            num_bond_types,
            num_heads,
        )
        self.bond_value_bias = nn.Embedding(
            num_bond_types,
            embedding_dim,
        )

        # Starting from zero makes the initial attention path stable.
        nn.init.zeros_(self.bond_logit_bias.weight)
        nn.init.zeros_(self.bond_value_bias.weight)

        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.feed_forward_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, n_node, _ = tensor.shape

        return tensor.reshape(
            batch_size,
            n_node,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def forward(
        self,
        nodes: torch.Tensor,
        bond_types: torch.Tensor,
        attention_mask: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_node, embedding_dim = nodes.shape

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"received {embedding_dim}."
            )

        normalized = self.attention_norm(nodes)

        query, key, value = self.qkv_projection(
            normalized
        ).chunk(3, dim=-1)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        # [batch, head, receiver, sender]
        attention_logits = torch.matmul(
            query,
            key.transpose(-1, -2),
        )
        attention_logits = (
            attention_logits / math.sqrt(self.head_dim)
        )

        bond_bias = self.bond_logit_bias(bond_types)
        attention_logits = (
            attention_logits
            + bond_bias.permute(0, 3, 1, 2)
        )

        allowed = attention_mask.unsqueeze(1)

        # Finite masking avoids NaNs for fully padded query rows.
        attention_logits = attention_logits.masked_fill(
            ~allowed,
            torch.finfo(attention_logits.dtype).min,
        )

        attention_weights = torch.softmax(
            attention_logits,
            dim=-1,
        )
        attention_weights = attention_weights.masked_fill(
            ~allowed,
            0.0,
        )
        attention_weights = (
            attention_weights
            / attention_weights.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)
        )

        attended = torch.matmul(
            attention_weights,
            value,
        )

        bond_values = self.bond_value_bias(
            bond_types
        ).reshape(
            batch_size,
            n_node,
            n_node,
            self.num_heads,
            self.head_dim,
        )

        attended = attended + torch.einsum(
            "bhij,bijhd->bhid",
            attention_weights,
            bond_values,
        )

        attended = attended.transpose(1, 2).reshape(
            batch_size,
            n_node,
            embedding_dim,
        )

        float_mask = node_mask.unsqueeze(-1).to(nodes.dtype)

        nodes = (
            nodes + self.output_projection(attended)
        ) * float_mask

        nodes = (
            nodes
            + self.feed_forward(
                self.feed_forward_norm(nodes)
            )
        ) * float_mask

        return nodes


class MolecularEncoder(nn.Module):
    """Encode discrete molecular graphs into node embeddings."""

    def __init__(
        self,
        *,
        node_feature_dim: int = NODE_FEATURE_DIM,
        noise_dim: int = 8,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        message_passing_steps: int = 6,
        num_heads: int = 4,
    ) -> None:
        super().__init__()

        if noise_dim < 0:
            raise ValueError("noise_dim cannot be negative.")
        if message_passing_steps < 1:
            raise ValueError(
                "message_passing_steps must be positive."
            )

        self.node_feature_dim = node_feature_dim
        self.noise_dim = noise_dim
        self.embedding_dim = embedding_dim

        self.input_projection = nn.Sequential(
            nn.Linear(
                node_feature_dim + noise_dim,
                hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self.blocks = nn.ModuleList(
            [
                BondAwareAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                )
                for _ in range(message_passing_steps)
            ]
        )

        # Identity intentionally matches the successfully trained
        # property-aware ZINC autoencoder from the notebook pipeline.
        self.output_norm = nn.Identity()

    def forward(
        self,
        batch: Mapping[str, Any],
        noise: torch.Tensor,
    ) -> torch.Tensor:
        features = batch["node_features"]
        node_mask = batch["node_mask"]

        expected_noise_shape = (
            features.shape[0],
            features.shape[1],
            self.noise_dim,
        )

        if tuple(noise.shape) != expected_noise_shape:
            raise ValueError(
                f"Expected noise shape {expected_noise_shape}, "
                f"received {tuple(noise.shape)}."
            )

        if features.shape[-1] != self.node_feature_dim:
            raise ValueError(
                f"Expected node feature dimension "
                f"{self.node_feature_dim}, "
                f"received {features.shape[-1]}."
            )

        float_mask = node_mask.unsqueeze(-1).to(
            features.dtype
        )

        encoder_input = torch.cat(
            [
                features,
                noise * float_mask,
            ],
            dim=-1,
        )

        nodes = (
            self.input_projection(encoder_input)
            * float_mask
        )

        for block in self.blocks:
            nodes = block(
                nodes=nodes,
                bond_types=batch["bond_targets"],
                attention_mask=batch[
                    "encoder_attention_mask"
                ],
                node_mask=node_mask,
            )

        return self.output_norm(nodes) * float_mask


class MolecularDecoder(nn.Module):
    """Decode node embeddings into atom and pairwise bond logits."""

    def __init__(
        self,
        *,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim

        def node_head(output_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        self.atom_head = node_head(NUM_ATOM_TYPES)
        self.charge_head = node_head(NUM_CHARGE_TYPES)
        self.h_head = node_head(NUM_HYDROGEN_TYPES)

        self.bond_head = nn.Sequential(
            nn.Linear(3 * embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, NUM_BOND_TYPES),
        )

    def forward(
        self,
        nodes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if nodes.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected node embeddings with dimension "
                f"{self.embedding_dim}, received {nodes.shape[-1]}."
            )

        receiver = nodes.unsqueeze(2)
        sender = nodes.unsqueeze(1)

        # These operations are symmetric under exchanging i and j.
        pair_features = torch.cat(
            [
                receiver + sender,
                torch.abs(receiver - sender),
                receiver * sender,
            ],
            dim=-1,
        )

        return {
            "atom_logits": self.atom_head(nodes),
            "charge_logits": self.charge_head(nodes),
            "h_logits": self.h_head(nodes),
            "bond_logits": self.bond_head(pair_features),
        }


class MolecularAutoencoder(nn.Module):
    """Bond-aware molecular graph autoencoder."""

    def __init__(
        self,
        *,
        node_feature_dim: int = NODE_FEATURE_DIM,
        noise_dim: int = 8,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        message_passing_steps: int = 6,
        num_heads: int = 4,
    ) -> None:
        super().__init__()

        self.noise_dim = noise_dim
        self.embedding_dim = embedding_dim

        self.encoder = MolecularEncoder(
            node_feature_dim=node_feature_dim,
            noise_dim=noise_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            message_passing_steps=message_passing_steps,
            num_heads=num_heads,
        )

        self.decoder = MolecularDecoder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        batch: Mapping[str, Any],
        noise: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        embeddings = self.encoder(batch, noise)
        predictions = self.decoder(embeddings)

        return embeddings, predictions


class MolecularEmbeddingQEDHead(nn.Module):
    """Permutation-invariant QED predictor over node embeddings."""

    def __init__(
        self,
        *,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        pooled_dim = 3 * embedding_dim

        self.network = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def pool(
        self,
        node_embeddings: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        if node_embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.embedding_dim}, "
                f"received {node_embeddings.shape[-1]}."
            )

        mask = node_mask.unsqueeze(-1)
        float_mask = mask.to(node_embeddings.dtype)

        node_count = float_mask.sum(
            dim=1
        ).clamp_min(1.0)

        mean_pool = (
            (node_embeddings * float_mask).sum(dim=1)
            / node_count
        )

        centered = (
            node_embeddings
            - mean_pool.unsqueeze(1)
        )

        variance_pool = (
            (centered.square() * float_mask).sum(dim=1)
            / node_count
        )
        standard_deviation_pool = torch.sqrt(
            variance_pool.clamp_min(1e-8)
        )

        max_input = node_embeddings.masked_fill(
            ~mask,
            torch.finfo(node_embeddings.dtype).min,
        )
        max_pool = max_input.max(dim=1).values

        return torch.cat(
            [
                mean_pool,
                max_pool,
                standard_deviation_pool,
            ],
            dim=-1,
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        pooled = self.pool(
            node_embeddings,
            node_mask,
        )

        # RDKit QED is bounded to [0, 1].
        return torch.sigmoid(
            self.network(pooled).squeeze(-1)
        )


def make_node_noise(
    batch: Mapping[str, Any],
    *,
    noise_dim: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate masked Gaussian noise for an autoencoder batch."""
    features = batch["node_features"]
    node_mask = batch["node_mask"]

    noise = torch.randn(
        features.shape[0],
        features.shape[1],
        noise_dim,
        dtype=features.dtype,
        device=features.device,
        generator=generator,
    )

    return (
        noise
        * node_mask.unsqueeze(-1).to(noise.dtype)
    )


def build_autoencoder_from_config(
    configuration: Mapping[str, Any],
) -> MolecularAutoencoder:
    """Build the autoencoder from the project YAML section."""
    return MolecularAutoencoder(
        noise_dim=int(configuration["node_noise_dim"]),
        embedding_dim=int(configuration["embedding_dim"]),
        hidden_dim=int(configuration["hidden_dim"]),
        message_passing_steps=int(
            configuration["message_passing_steps"]
        ),
        num_heads=int(configuration["attention_heads"]),
    )


__all__ = [
    "BondAwareAttentionBlock",
    "MolecularAutoencoder",
    "MolecularDecoder",
    "MolecularEmbeddingQEDHead",
    "MolecularEncoder",
    "build_autoencoder_from_config",
    "make_node_noise",
]
