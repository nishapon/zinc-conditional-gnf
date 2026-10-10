"""Bond-aware graph encoder (PyTorch port of the teammates' v5 encoder).

Reference: `molecular_generation/bond_aware_attention.py`
(`BondAwareSelfAttentionMLP`) and `train_molecular_autoencoder.py`
(`TimestepGNN`, `embed_atom_features`) in
https://github.com/sridharmatta1/graph-normalising-flow.

Faithful to that design:
  * one-hot atomic identity -> MLP -> `node_embedding_dim` node features;
  * each step is multi-head self-attention in which a node attends only to
    itself and its bonded neighbours (self-loops carry bond type 0, as in the
    teammates' graphs);
  * a learned per-(bond type, head) bias is added to the attention logits and
    a learned per-(bond type, head, value-dim) bias to the attended values,
    so single / double / triple bonds shape message passing;
  * the value projection is shared by all heads (the teammates `repeat` it),
    the per-head outputs are concatenated, projected, concatenated with the
    node state and passed through an MLP;
  * `num_processing_steps` steps (6), weights shared across steps, no
    residual connection, no normalization (the v5 run settings).

Differences forced by the setting (documented, not behavioural):
  * dense padded tensors with masks replace graph_nets' flat edge lists;
  * the input is the 17-type charge-aware one-hot, plus OPTIONAL
    symmetry-breaking noise channels (`symmetry_noise_dim`, off by default;
    never replaces atomic identity).
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def make_mlp(in_dim: int, latent_dim: int, out_dim: int, num_layers: int) -> nn.Sequential:
    """The teammates' `make_mlp_model`: hidden width `latent_dim`, ReLU,
    no activation on the output, Glorot-normal weights and truncated-normal
    (std 0.1) biases."""
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    widths = [in_dim] + [latent_dim] * (num_layers - 1) + [out_dim]
    layers: list[nn.Module] = []
    for index in range(num_layers):
        linear = nn.Linear(widths[index], widths[index + 1])
        nn.init.xavier_normal_(linear.weight)
        nn.init.trunc_normal_(linear.bias, std=0.1, a=-0.2, b=0.2)
        layers.append(linear)
        if index < num_layers - 1:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class BondAwareAttentionBlock(nn.Module):
    """One bond-aware self-attention step on dense padded graphs."""

    def __init__(
        self,
        node_dim: int,
        *,
        kq_dim: int,
        v_dim: int,
        num_heads: int,
        concat_heads_output_dim: int,
        num_bond_types: int,
        latent_dim: int,
        num_mlp_layers: int,
    ) -> None:
        super().__init__()
        self.num_heads, self.kq_dim, self.v_dim = num_heads, kq_dim, v_dim
        self.num_bond_types = num_bond_types
        self.query = nn.Linear(node_dim, num_heads * kq_dim, bias=False)
        self.key = nn.Linear(node_dim, num_heads * kq_dim, bias=False)
        self.value = nn.Linear(node_dim, v_dim, bias=False)
        for layer in (self.query, self.key, self.value):
            nn.init.xavier_uniform_(layer.weight)
        self.logit_bias = nn.Parameter(torch.zeros(num_bond_types, num_heads))
        self.value_bias = nn.Parameter(torch.zeros(num_bond_types, num_heads, v_dim))
        self.output_projection = nn.Linear(num_heads * v_dim, concat_heads_output_dim, bias=False)
        self.mlp = make_mlp(node_dim + concat_heads_output_dim, latent_dim, node_dim, num_mlp_layers)

    def forward(self, nodes: torch.Tensor, edge_type: torch.Tensor,
                edge_mask: torch.Tensor) -> torch.Tensor:
        """nodes [B,N,D]; edge_type [B,N,N] long; edge_mask [B,N,N] bool.

        `edge_mask[b,i,j]` says node i may receive from node j (bond or
        self-loop). Every row must contain at least the self-loop.
        """
        batch, width, _ = nodes.shape
        heads = self.num_heads
        queries = self.query(nodes).view(batch, width, heads, self.kq_dim)
        keys = self.key(nodes).view(batch, width, heads, self.kq_dim)
        values = self.value(nodes)                                           # [B,N,v]

        logits = torch.einsum("bihd,bjhd->bhij", queries, keys) / math.sqrt(self.kq_dim)
        logits = logits + self.logit_bias[edge_type].permute(0, 3, 1, 2)     # [B,H,N,N]
        logits = logits.masked_fill(~edge_mask.unsqueeze(1), float("-inf"))
        weights = torch.softmax(logits, dim=-1)

        attended = torch.einsum("bhij,bjv->bhiv", weights, values)
        type_one_hot = F.one_hot(edge_type, self.num_bond_types).to(weights.dtype)
        weight_per_type = torch.einsum("bhij,bijt->bhit", weights, type_one_hot)
        attended = attended + torch.einsum("bhit,thv->bhiv", weight_per_type, self.value_bias)

        merged = attended.permute(0, 2, 1, 3).reshape(batch, width, heads * self.v_dim)
        projected = self.output_projection(merged)
        return self.mlp(torch.cat([nodes, projected], dim=-1))


class BondAwareEncoder(nn.Module):
    """One-hot atom identity + bond graph -> node embeddings `[B, N, D]`."""

    def __init__(
        self,
        *,
        num_atom_types: int,
        num_bond_types: int,
        node_embedding_dim: int,
        latent_dim: int,
        num_mlp_layers: int,
        num_processing_steps: int,
        attn_kq_dim: int,
        attn_v_dim: int,
        attn_num_heads: int,
        attn_concat_heads_output_dim: int,
        weight_sharing: bool = True,
        residual: bool = False,
        symmetry_noise_dim: int = 0,
    ) -> None:
        super().__init__()
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.symmetry_noise_dim = int(symmetry_noise_dim)
        self.residual = bool(residual)
        self.num_processing_steps = int(num_processing_steps)
        self.embed = make_mlp(num_atom_types + self.symmetry_noise_dim, latent_dim,
                              node_embedding_dim, num_mlp_layers)

        def block() -> BondAwareAttentionBlock:
            return BondAwareAttentionBlock(
                node_embedding_dim, kq_dim=attn_kq_dim, v_dim=attn_v_dim,
                num_heads=attn_num_heads,
                concat_heads_output_dim=attn_concat_heads_output_dim,
                num_bond_types=num_bond_types, latent_dim=latent_dim,
                num_mlp_layers=num_mlp_layers,
            )

        self.weight_sharing = bool(weight_sharing)
        self.blocks = nn.ModuleList([block()] if weight_sharing
                                    else [block() for _ in range(num_processing_steps)])

    def forward(
        self,
        atom_types: torch.Tensor,
        bond_types: torch.Tensor,
        node_mask: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        batch, width = atom_types.shape
        one_hot = F.one_hot(atom_types.clamp(min=0), self.num_atom_types).to(torch.float32)
        features = one_hot * node_mask.unsqueeze(-1).to(one_hot.dtype)
        if self.symmetry_noise_dim > 0:
            noise = torch.randn(batch, width, self.symmetry_noise_dim, device=features.device,
                                generator=generator)
            features = torch.cat([features, noise * node_mask.unsqueeze(-1)], dim=-1)

        # Attention graph: bonds plus a self-loop on every node (padding
        # included, so no softmax row is empty). Self-loops carry type 0.
        identity = torch.eye(width, dtype=torch.bool, device=atom_types.device).unsqueeze(0)
        pair_valid = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        edge_mask = ((bond_types > 0) & pair_valid) | identity
        edge_type = torch.where(pair_valid & ~identity, bond_types, torch.zeros_like(bond_types))

        mask = node_mask.unsqueeze(-1).to(features.dtype)
        nodes = self.embed(features) * mask
        initial = nodes
        for step in range(self.num_processing_steps):
            block = self.blocks[0] if self.weight_sharing else self.blocks[step]
            nodes = block(nodes, edge_type, edge_mask) * mask
        if self.residual:
            nodes = nodes + initial
        return nodes


__all__ = ["BondAwareAttentionBlock", "BondAwareEncoder", "make_mlp"]
