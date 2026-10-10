"""Atom and bond decoder heads (PyTorch port of the teammates' decoder).

Reference: `molecular_generation/molecular_gnn.py`
(`atom_type_logits`, `bond_type_logits`, `refine_bond_logits`).

* Atom head: MLP on each node embedding -> logits over the 17 charge-aware
  (element, formal charge) types.
* Bond head: a symmetric function of every node pair, `[x_i + x_j,
  |x_i - x_j|, x_i * x_j]` -> MLP -> 4 classes (none / single / double /
  triple), so bond(i, j) and bond(j, i) always get identical logits.
* Optional iterative refinement (`num_bond_refine_steps > 1`): each extra
  round feeds the atom's soft remaining valence, computed from the current
  bond probabilities and the TYPED valence cap of the atom's type, back into
  the node embeddings. The teammates' v4 run found this did not help; the
  default is 1 (single shot), as in their final v5 run.

Teacher forcing: like the teammates' training loss, refinement uses the TRUE
atom types during training. For reconstruction and generation it uses the
atom types the model itself predicts (`atom_types=None`).
"""

from __future__ import annotations

import torch
from torch import nn

from zinc_gnf.constants_charge_aware import ATOM_VALENCE
from zinc_gnf.encoder import make_mlp

BOND_ORDER = (0.0, 1.0, 2.0, 3.0)      # class index -> valence consumed


class BondAwareDecoder(nn.Module):
    def __init__(
        self,
        *,
        num_atom_types: int,
        num_bond_types: int,
        node_embedding_dim: int,
        latent_dim: int,
        num_mlp_layers: int,
        num_bond_refine_steps: int = 1,
        valence_caps: tuple[int, ...] = tuple(ATOM_VALENCE),
    ) -> None:
        super().__init__()
        if len(valence_caps) != num_atom_types:
            raise ValueError("valence_caps must have one entry per atom type")
        if num_bond_types != len(BOND_ORDER):
            raise ValueError(f"expected {len(BOND_ORDER)} bond classes, got {num_bond_types}")
        self.num_bond_refine_steps = int(num_bond_refine_steps)
        self.atom_head = make_mlp(node_embedding_dim, latent_dim, num_atom_types, num_mlp_layers)
        self.bond_head = make_mlp(3 * node_embedding_dim, latent_dim, num_bond_types, num_mlp_layers)
        self.update_mlps = nn.ModuleList([
            make_mlp(node_embedding_dim + 1, latent_dim, node_embedding_dim, num_mlp_layers)
            for _ in range(max(0, self.num_bond_refine_steps - 1))
        ])
        self.register_buffer("valence", torch.tensor(valence_caps, dtype=torch.float32))
        self.register_buffer("bond_order", torch.tensor(BOND_ORDER, dtype=torch.float32))

    def atom_logits(self, nodes: torch.Tensor) -> torch.Tensor:
        return self.atom_head(nodes)

    def bond_logits(self, nodes: torch.Tensor) -> torch.Tensor:
        left, right = nodes.unsqueeze(2), nodes.unsqueeze(1)             # [B,N,1,D], [B,1,N,D]
        left, right = left.expand(-1, -1, nodes.shape[1], -1), right.expand(-1, nodes.shape[1], -1, -1)
        pair = torch.cat([left + right, (left - right).abs(), left * right], dim=-1)
        return self.bond_head(pair)                                      # [B,N,N,4]

    def forward(
        self,
        nodes: torch.Tensor,
        node_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        atom_types: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        atom_logits = self.atom_logits(nodes)
        atoms_used = atom_logits.argmax(dim=-1) if atom_types is None else atom_types
        mask = node_mask.unsqueeze(-1).to(nodes.dtype)

        bond_logits = self.bond_logits(nodes)
        current = nodes
        for update in self.update_mlps:
            probabilities = torch.softmax(bond_logits, dim=-1)
            expected_order = (probabilities * self.bond_order).sum(dim=-1)
            expected_used = (expected_order * pair_mask.to(expected_order.dtype)).sum(dim=-1)
            remaining = (self.valence[atoms_used.clamp(min=0)] - expected_used).unsqueeze(-1)
            current = current + update(torch.cat([current, remaining], dim=-1)) * mask
            bond_logits = self.bond_logits(current)

        return {"atom_logits": atom_logits, "bond_logits": bond_logits, "atoms_used": atoms_used}


__all__ = ["BOND_ORDER", "BondAwareDecoder"]
