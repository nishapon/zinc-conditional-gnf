"""Charge-aware bond-aware graph autoencoder: model, loss and reconstruction metrics.

Architecture = the teammates' v5 molecular autoencoder (see `encoder.py`,
`decoder.py`) adapted to the 17 (element, formal charge) atom types:

    one-hot atoms + bond graph --BondAwareEncoder--> node embeddings [B, N, 64]
    node embeddings --BondAwareDecoder--> atom-type logits [B, N, 17]
                                          bond logits       [B, N, N, 4]

The node embeddings are what the Conditional GNF models later; the decoder
turns generated embeddings back into molecules.

Loss (as in the teammates' `molecular_reconstruction_loss`): summed atom
cross-entropy over real nodes plus summed bond cross-entropy over real
ordered pairs (i != j), the three real-bond classes up-weighted by
`bond_class_weight` (5.0). `mean_loss = total / n_real_nodes` is what the
optimizer minimises; the summed `total_loss` is also reported.

Reconstruction evaluation reports, SEPARATELY:
  * raw decoding (argmax atoms and argmax bonds, no chemical constraint);
  * constrained decoding (argmax atoms, valence-aware bond assembly with a
    connectivity pass, `decoding_charge_aware`);
  * exact graph match, exact molecule (canonical SMILES) match, and RDKit
    validity (sanitizable and single-fragment) for each;
  * the exact-reconstruction ceiling of a deterministic one-hot equivariant
    encoder (`symmetry.py`), so a low exact-match rate can be read against
    what is achievable at all.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from typing import Any, Sequence

import numpy as np
from rdkit import Chem, rdBase
import torch
from torch import nn
import torch.nn.functional as F

from zinc_gnf.chemistry import canonical_connectivity_smiles
from zinc_gnf.chemistry_charge_aware import graph_to_mol_charge_aware
from zinc_gnf.constants_charge_aware import (
    ATOM_VALENCE,
    NUM_ATOM_TYPES,
    NUM_BOND_TYPES,
)
from zinc_gnf.dataset import MolecularBatch, MolecularGraphDataset, iter_eval_batches
from zinc_gnf.decoder import BondAwareDecoder
from zinc_gnf.decoding_charge_aware import (
    degree_constrained_bond_decode,
    valence_aware_bond_decode_typed,
)
from zinc_gnf.encoder import BondAwareEncoder
from zinc_gnf.symmetry import is_decodable


@dataclass
class AutoencoderConfig:
    """Teammates' v5 settings (QM9) adapted to the 17-type ZINC representation."""

    num_atom_types: int = NUM_ATOM_TYPES            # 17 (QM9 had 4)
    num_bond_types: int = NUM_BOND_TYPES            # 4
    node_embedding_dim: int = 64
    latent_dim: int = 256
    num_mlp_layers: int = 2
    num_processing_steps: int = 6
    attn_kq_dim: int = 32
    attn_v_dim: int = 32
    attn_num_heads: int = 2
    attn_concat_heads_output_dim: int = 32
    weight_sharing: bool = True
    residual: bool = False
    bond_class_weight: float = 5.0
    num_bond_refine_steps: int = 1
    symmetry_noise_dim: int = 0                     # 0 = pure one-hot identity (decision 2)
    bond_head_layers: int | None = None             # None = num_mlp_layers (teammates' value)
    position_features: bool = False                 # EXPERIMENTAL diagnostic, see encoder.py
    degree_head: bool = False                       # per-atom heavy-degree head (symmetry tie-breaker)
    degree_loss_weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "AutoencoderConfig":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown autoencoder settings: {sorted(unknown)}")
        return cls(**values)


class GraphAutoencoder(nn.Module):
    def __init__(self, config: AutoencoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or AutoencoderConfig()
        c = self.config
        self.encoder = BondAwareEncoder(
            num_atom_types=c.num_atom_types, num_bond_types=c.num_bond_types,
            node_embedding_dim=c.node_embedding_dim, latent_dim=c.latent_dim,
            num_mlp_layers=c.num_mlp_layers, num_processing_steps=c.num_processing_steps,
            attn_kq_dim=c.attn_kq_dim, attn_v_dim=c.attn_v_dim,
            attn_num_heads=c.attn_num_heads,
            attn_concat_heads_output_dim=c.attn_concat_heads_output_dim,
            weight_sharing=c.weight_sharing, residual=c.residual,
            symmetry_noise_dim=c.symmetry_noise_dim,
            position_features=c.position_features,
        )
        self.decoder = BondAwareDecoder(
            num_atom_types=c.num_atom_types, num_bond_types=c.num_bond_types,
            node_embedding_dim=c.node_embedding_dim, latent_dim=c.latent_dim,
            num_mlp_layers=c.num_mlp_layers, num_bond_refine_steps=c.num_bond_refine_steps,
            bond_head_layers=c.bond_head_layers, degree_head=c.degree_head,
            valence_caps=tuple(ATOM_VALENCE),
        )

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode(self, batch: MolecularBatch, generator: torch.Generator | None = None) -> torch.Tensor:
        return self.encoder(batch.atom_types, batch.bond_types, batch.node_mask, generator=generator)

    def decode(self, nodes: torch.Tensor, batch: MolecularBatch, *,
               teacher_forcing: bool) -> dict[str, torch.Tensor]:
        return self.decoder(nodes, batch.node_mask, batch.pair_mask,
                            atom_types=batch.atom_types if teacher_forcing else None)

    def forward(self, batch: MolecularBatch, *, teacher_forcing: bool = True) -> dict[str, torch.Tensor]:
        nodes = self.encode(batch)
        outputs = self.decode(nodes, batch, teacher_forcing=teacher_forcing)
        outputs["nodes"] = nodes
        return outputs

    def reconstruction_loss(self, batch: MolecularBatch, outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return reconstruction_loss(batch, outputs, bond_class_weight=self.config.bond_class_weight,
                                   degree_loss_weight=self.config.degree_loss_weight)


def true_degrees(batch: MolecularBatch) -> torch.Tensor:
    """Heavy-atom degree of every atom (number of bonded neighbours), clipped to the head's range."""
    from zinc_gnf.decoder import MAX_DEGREE_CLASS
    degree = ((batch.bond_types > 0) & batch.pair_mask).sum(-1)
    return degree.clamp(max=MAX_DEGREE_CLASS)


def reconstruction_loss(batch: MolecularBatch, outputs: dict[str, torch.Tensor], *,
                        bond_class_weight: float, degree_loss_weight: float = 1.0) -> dict[str, torch.Tensor]:
    atom_logits, bond_logits = outputs["atom_logits"], outputs["bond_logits"]
    node_mask, pair_mask = batch.node_mask, batch.pair_mask
    n_real = node_mask.sum().clamp(min=1).to(atom_logits.dtype)

    atom_ce = F.cross_entropy(atom_logits.reshape(-1, atom_logits.shape[-1]),
                              batch.atom_types.reshape(-1), reduction="none").view_as(batch.atom_types)
    atom_loss = (atom_ce * node_mask).sum()

    class_weight = torch.tensor([1.0] + [float(bond_class_weight)] * (bond_logits.shape[-1] - 1),
                                device=bond_logits.device, dtype=bond_logits.dtype)
    bond_ce = F.cross_entropy(bond_logits.reshape(-1, bond_logits.shape[-1]),
                              batch.bond_types.reshape(-1), reduction="none").view_as(batch.bond_types)
    bond_loss = (bond_ce * class_weight[batch.bond_types] * pair_mask).sum()

    with torch.no_grad():
        atom_pred = atom_logits.argmax(-1)
        atom_correct = ((atom_pred == batch.atom_types) & node_mask).sum()
        bond_pred = bond_logits.argmax(-1)
        bond_correct_mask = (bond_pred == batch.bond_types) & pair_mask
        real_bond = (batch.bond_types > 0) & pair_mask
        pair_total = pair_mask.sum().clamp(min=1)
        real_total = real_bond.sum().clamp(min=1)

    degree_loss = torch.zeros((), device=atom_logits.device, dtype=atom_logits.dtype)
    degree_accuracy = torch.full((), float("nan"), device=atom_logits.device, dtype=atom_logits.dtype)
    if "degree_logits" in outputs:
        target = true_degrees(batch)
        degree_ce = F.cross_entropy(outputs["degree_logits"].reshape(-1, outputs["degree_logits"].shape[-1]),
                                    target.reshape(-1), reduction="none").view_as(target)
        degree_loss = float(degree_loss_weight) * (degree_ce * node_mask).sum()
        with torch.no_grad():
            degree_accuracy = ((outputs["degree_logits"].argmax(-1) == target) & node_mask).sum().to(
                atom_logits.dtype) / n_real

    total = atom_loss + bond_loss + degree_loss
    return {
        "total_loss": total,
        "mean_loss": total / n_real,
        "atom_loss": atom_loss,
        "bond_loss": bond_loss,
        "degree_loss": degree_loss,
        "degree_accuracy": degree_accuracy,
        "atom_accuracy": atom_correct.to(total.dtype) / n_real,
        "bond_accuracy": bond_correct_mask.sum().to(total.dtype) / pair_total,
        "real_bond_accuracy": (bond_correct_mask & real_bond).sum().to(total.dtype) / real_total,
        "n_nodes": n_real,
    }


# ---------------------------------------------------------------------------
# Reconstruction evaluation.
# ---------------------------------------------------------------------------
def _to_mol(atoms: np.ndarray, bonds: np.ndarray):
    """(mol, sanitizable, single_fragment) for a typed graph."""
    try:
        with rdBase.BlockLogs():
            mol = graph_to_mol_charge_aware(atoms, bonds)
    except Exception:  # noqa: BLE001 - any RDKit/valence failure = invalid
        return None, False, False
    return mol, True, len(Chem.GetMolFrags(mol)) == 1


def _canonical(mol) -> str | None:
    try:
        return canonical_connectivity_smiles(mol)
    except Exception:  # noqa: BLE001
        return None


@torch.no_grad()
def evaluate_reconstruction(
    model: GraphAutoencoder,
    dataset: MolecularGraphDataset,
    *,
    indices: Sequence[int] | None = None,
    device: str | torch.device = "cpu",
    batch_size: int = 128,
    compute_ceiling: bool = True,
    decode_molecules: bool = True,
    eval_seed: int | None = 0,
) -> dict[str, Any]:
    """Teacher-forced loss plus free-running reconstruction metrics.

    Atom types are predicted (not given) for the decoding metrics. Raw and
    constrained decoding are reported as separate metric groups.
    """
    was_training = model.training
    model.eval()
    started = time.perf_counter()
    order = list(range(len(dataset))) if indices is None else [int(i) for i in indices]
    # Models with symmetry-noise channels are stochastic; a seeded generator on
    # the model's device makes evaluation reproducible.
    generator = None
    if eval_seed is not None and model.config.symmetry_noise_dim > 0:
        generator = torch.Generator(device=device).manual_seed(int(eval_seed))

    sums = {"total_loss": 0.0, "atom_loss": 0.0, "bond_loss": 0.0, "n_nodes": 0.0,
            "atom_correct": 0.0, "bond_correct": 0.0, "bond_total": 0.0,
            "real_bond_correct": 0.0, "real_bond_total": 0.0}
    has_degree = model.config.degree_head
    decoders = ("raw", "constrained") + (("degree",) if has_degree else ())
    groups = {name: {"valid": 0, "sanitizable": 0, "graph_exact": 0, "molecule_exact": 0}
              for name in decoders}
    degree_correct = degree_total = degree_fallbacks = 0
    strata_keys = ("n", "constrained", "raw", "degree")
    constrained_connected = 0
    atom_sequence_exact = 0
    ceiling_flags: list[bool] = []
    exact_given_decodable = [0, 0]
    strata = {"decodable": {k: 0 for k in strata_keys},
              "non_decodable": {k: 0 for k in strata_keys}}
    n_molecules = 0

    for batch in iter_eval_batches(dataset, batch_size, order):
        batch_indices = order[n_molecules:n_molecules + batch.batch_size]
        batch = batch.to(device)

        nodes = model.encode(batch, generator=generator)
        outputs = model.decode(nodes, batch, teacher_forcing=True)
        outputs["nodes"] = nodes
        losses = model.reconstruction_loss(batch, outputs)
        n = float(losses["n_nodes"])
        sums["total_loss"] += float(losses["total_loss"])
        sums["atom_loss"] += float(losses["atom_loss"])
        sums["bond_loss"] += float(losses["bond_loss"])
        sums["n_nodes"] += n

        free = outputs if model.config.num_bond_refine_steps <= 1 else model.decode(
            outputs["nodes"], batch, teacher_forcing=False)
        atom_pred = free["atom_logits"].argmax(-1)
        bond_probs = torch.softmax(free["bond_logits"], dim=-1)
        bond_pred = free["bond_logits"].argmax(-1)

        real_bond = (batch.bond_types > 0) & batch.pair_mask
        correct = (bond_pred == batch.bond_types) & batch.pair_mask
        sums["atom_correct"] += float(((atom_pred == batch.atom_types) & batch.node_mask).sum())
        sums["bond_correct"] += float(correct.sum())
        sums["bond_total"] += float(batch.pair_mask.sum())
        sums["real_bond_correct"] += float((correct & real_bond).sum())
        sums["real_bond_total"] += float(real_bond.sum())

        if decode_molecules or compute_ceiling:
            atoms_np = atom_pred.cpu().numpy()
            raw_np = bond_pred.cpu().numpy()
            probs_np = bond_probs.cpu().numpy()
            if has_degree:
                degree_np = free["degree_logits"].argmax(-1).cpu().numpy()
            for b, dataset_index in enumerate(batch_indices):
                n_b = int(batch.n_node[b])
                true_atoms, true_bonds = dataset.true_graph(dataset_index)
                if compute_ceiling:
                    flag = is_decodable(true_atoms, true_bonds)
                    ceiling_flags.append(flag)
                if not decode_molecules:
                    continue
                atoms = atoms_np[b, :n_b]
                atoms_ok = np.array_equal(atoms, true_atoms)
                atom_sequence_exact += int(atoms_ok)
                reference = dataset.smiles[dataset_index]

                raw_bonds = raw_np[b, :n_b, :n_b].copy()
                np.fill_diagonal(raw_bonds, 0)
                constrained, info = valence_aware_bond_decode_typed(atoms, probs_np[b, :n_b, :n_b])
                constrained_connected += int(info["connected"])

                stratum = None
                if compute_ceiling:
                    stratum = strata["decodable" if ceiling_flags[-1] else "non_decodable"]
                    stratum["n"] += 1
                candidates = [("raw", raw_bonds), ("constrained", constrained)]
                if has_degree:
                    predicted_degree = degree_np[b, :n_b]
                    degree_bonds, degree_info = degree_constrained_bond_decode(
                        atoms, probs_np[b, :n_b, :n_b], predicted_degree)
                    degree_fallbacks += int(degree_info.get("fallback", False))
                    degree_correct += int((predicted_degree == (true_bonds > 0).sum(1).clip(max=6)).sum())
                    degree_total += n_b
                    candidates.append(("degree", degree_bonds))
                for name, bonds in candidates:
                    mol, sanitizable, single = _to_mol(atoms, bonds)
                    g = groups[name]
                    g["sanitizable"] += int(sanitizable)
                    g["valid"] += int(sanitizable and single)
                    graph_exact = bool(atoms_ok and np.array_equal(bonds, true_bonds))
                    g["graph_exact"] += int(graph_exact)
                    molecule_exact = bool(sanitizable and single and _canonical(mol) == reference)
                    g["molecule_exact"] += int(molecule_exact)
                    if stratum is not None:
                        stratum[name] += int(molecule_exact)
                    if name == "constrained" and compute_ceiling:
                        exact_given_decodable[1] += int(ceiling_flags[-1])
                        exact_given_decodable[0] += int(ceiling_flags[-1] and molecule_exact)
        n_molecules += batch.batch_size

    if was_training:
        model.train()

    def rate(num: float, den: float) -> float:
        return float(num / den) if den else float("nan")

    result: dict[str, Any] = {
        "molecules": n_molecules,
        "mean_loss": rate(sums["total_loss"], sums["n_nodes"]),
        "total_loss_per_molecule": rate(sums["total_loss"], n_molecules),
        "atom_loss_per_node": rate(sums["atom_loss"], sums["n_nodes"]),
        "bond_loss_per_node": rate(sums["bond_loss"], sums["n_nodes"]),
        "atom_accuracy": rate(sums["atom_correct"], sums["n_nodes"]),
        "bond_accuracy": rate(sums["bond_correct"], sums["bond_total"]),
        "real_bond_accuracy": rate(sums["real_bond_correct"], sums["real_bond_total"]),
        "seconds": time.perf_counter() - started,
    }
    if decode_molecules:
        result["atom_sequence_exact"] = rate(atom_sequence_exact, n_molecules)
        if has_degree:
            result["degree_accuracy"] = rate(degree_correct, degree_total)
            result["degree_decode_fallback_rate"] = rate(degree_fallbacks, n_molecules)
        result["constrained_connected"] = rate(constrained_connected, n_molecules)
        for name, g in groups.items():
            result[name] = {
                "validity": rate(g["valid"], n_molecules),
                "sanitizable": rate(g["sanitizable"], n_molecules),
                "exact_graph_match": rate(g["graph_exact"], n_molecules),
                "exact_molecule_match": rate(g["molecule_exact"], n_molecules),
            }
    if compute_ceiling and ceiling_flags:
        result["equivariant_exact_ceiling"] = float(np.mean(ceiling_flags))
        if decode_molecules:
            result["constrained_exact_molecule_given_decodable"] = rate(
                exact_given_decodable[0], exact_given_decodable[1])
            result["exact_molecule_by_symmetry"] = {
                name: {"molecules": v["n"],
                       "constrained_exact": rate(v["constrained"], v["n"]),
                       "raw_exact": rate(v["raw"], v["n"])}
                for name, v in strata.items()
            }
            if has_degree:
                for name, v in strata.items():
                    result["exact_molecule_by_symmetry"][name]["degree_exact"] = rate(v["degree"], v["n"])
    return result


__all__ = [
    "AutoencoderConfig",
    "GraphAutoencoder",
    "evaluate_reconstruction",
    "reconstruction_loss",
]
