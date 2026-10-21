"""A decoder-aligned structural objective: the term our likelihood never contained.

THE DIAGNOSIS THIS IMPLEMENTS A FIX FOR
---------------------------------------
Our flow maximises the likelihood of SMOOTHED PER-NODE embeddings. The decoder, however,
reads bonds out of PAIRS of node embeddings (its pair features are
`[h_i + h_j, |h_i - h_j|, h_i * h_j]`). Those two things are not the same objective:

* the likelihood has 64*N dimensions per molecule, all per-node, and is dominated by
  getting the per-node marginals right;
* the chemistry lives in O(N^2) pairwise relations, which contribute almost nothing to
  the NLL - a sample can have a near-perfect per-node marginal fit and still carry no
  usable bond information.

The shuffled-node control measured exactly that: real node embeddings, regrouped into
molecules at random - identical marginals, destroyed pairing - reproduce the flow's
failure (P(no bond) 0.9983 shuffled / 0.9940 flow vs 0.9039 real; aromatic rings 0.027 /
0.012 vs 1.830). So the flow is behaving like a correct per-node marginal model with the
joint structure missing, and no amount of further MLE training re-weights the objective
towards the part that matters.

WHAT THIS MODULE ADDS
---------------------
A distribution-level penalty on what the FROZEN decoder says about the flow's OWN
samples, computed from differentiable class probabilities (never argmax, never RDKit):

  bonds_per_atom          expected bond count / N        attacks P(no bond) directly
  cycle_rank              E - N + 1 (per molecule)       rings are cycles; no cycles, no
                                                         aromatic rings
  double_bond_fraction    share of bond mass in class 2  in a Kekule representation an
                                                         aromatic ring is half double
                                                         bonds, so this is the closest
                                                         differentiable aromaticity proxy
  isolated_atom_rate      prod_j P(no bond | i, j)       attacks fragmentation, which is
                                                         what makes a sample invalid
  valence_overflow        relu(expected degree - cap)    attacks the other failure mode

Each is compared with the SAME statistic measured on TRAINING molecules (`StructureTargets`),
and only as a MOMENT: the loss is a weighted squared error between the mean over a batch
of samples and the training mean. No per-sample supervision, no RDKit in the loop, no
test or validation data.

HONEST ACCOUNTING
-----------------
* Invertibility is untouched: a loss cannot make a bijection non-bijective, and the flow
  is the same architecture, parameterisation and checkpoint lineage.
* The objective is NO LONGER pure maximum likelihood. It is
  `NLL(data) + lambda * moment-mismatch(samples)` - a hybrid of forward KL on the data
  and a reverse-direction moment match on samples. Both terms must be reported
  separately, always, and the validation NLL tells us what the structural term cost in
  likelihood terms. A result is only meaningful with both numbers in view.
* It optimises decoder-level STRUCTURE, not QED and not validity as scored by RDKit, so
  improvements in QED or validity are consequences, not targets.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Sequence

import numpy as np
import torch

from zinc_gnf.constants_charge_aware import ATOM_VALENCE
from zinc_gnf.decoder import BOND_ORDER

# Weights are 1/scale^2-ish so that each term contributes comparably at the start; they
# are deliberately simple and are recorded in every checkpoint.
DEFAULT_WEIGHTS: dict[str, float] = {
    "bonds_per_atom": 1.0,
    "cycle_rank": 0.1,
    "double_bond_fraction": 4.0,
    "isolated_atom_rate": 4.0,
    "valence_overflow": 1.0,
}
TERMS = tuple(DEFAULT_WEIGHTS)


@dataclass
class StructureTargets:
    """The same statistics, measured on TRAINING graphs (never validation or test)."""

    bonds_per_atom: float
    cycle_rank: float
    double_bond_fraction: float
    isolated_atom_rate: float
    valence_overflow: float
    molecules: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "StructureTargets":
        return cls(**{k: values[k] for k in cls.__dataclass_fields__})

    @classmethod
    def from_records(cls, records: Sequence[dict[str, Any]], *, limit: int | None = None
                     ) -> "StructureTargets":
        """Measured from the stored graphs, i.e. the representation the model learns."""
        chosen = records if limit is None else records[:limit]
        if not chosen:
            raise ValueError("no training records to measure structure targets from")
        bonds_per_atom, cycles, double_fraction, isolated = [], [], [], []
        for record in chosen:
            bonds = np.asarray(record["bond_matrix"], dtype=np.int64)
            n = int(bonds.shape[0])
            upper = np.triu(bonds, k=1)
            edges = int((upper > 0).sum())
            bonds_per_atom.append(edges / max(n, 1))
            cycles.append(edges - n + 1)
            total = max(edges, 1)
            double_fraction.append(int((upper == 2).sum()) / total)
            degree = (bonds > 0).sum(axis=1)
            isolated.append(float((degree == 0).mean()))
        return cls(
            bonds_per_atom=float(np.mean(bonds_per_atom)),
            cycle_rank=float(np.mean(cycles)),
            double_bond_fraction=float(np.mean(double_fraction)),
            isolated_atom_rate=float(np.mean(isolated)),
            valence_overflow=0.0,                      # real molecules never overflow
            molecules=len(chosen),
        )


def structure_statistics(bond_logits: torch.Tensor, atom_logits: torch.Tensor,
                         mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Differentiable decoder-level structure statistics for one batch of samples.

    Everything is built from class PROBABILITIES, so the result is differentiable with
    respect to the embeddings the flow produced. The bond matrix is symmetrised by
    averaging the two directions (the decoder's pair features are symmetric, so this is
    exact up to float noise) and the diagonal and padding are removed.
    """
    probabilities = torch.softmax(bond_logits, dim=-1)
    probabilities = 0.5 * (probabilities + probabilities.transpose(1, 2))
    pair = (mask.unsqueeze(1) & mask.unsqueeze(2)).to(probabilities.dtype)
    eye = torch.eye(mask.shape[1], device=mask.device, dtype=pair.dtype).unsqueeze(0)
    pair = pair * (1.0 - eye)

    p_bond = (1.0 - probabilities[..., 0]) * pair                      # [B, N, N]
    counts = mask.sum(-1).clamp(min=1).to(p_bond.dtype)                # [B]
    edges = 0.5 * p_bond.sum((1, 2))                                   # expected bonds
    bonds_per_atom = edges / counts
    cycle_rank = edges - counts + 1.0

    order = torch.as_tensor(BOND_ORDER, device=bond_logits.device, dtype=p_bond.dtype)
    bond_mass = 0.5 * (probabilities * pair.unsqueeze(-1)).sum((1, 2))  # [B, 4]
    total_bond_mass = bond_mass[:, 1:].sum(-1).clamp(min=1e-6)
    double_bond_fraction = bond_mass[:, 2] / total_bond_mass

    # P(node i has no bond at all) = prod_j P(no bond | i, j) over real j != i.
    log_no_bond = torch.log(probabilities[..., 0].clamp(min=1e-6, max=1.0)) * pair
    isolated = torch.exp(log_no_bond.sum(-1)) * mask.to(p_bond.dtype)
    isolated_atom_rate = isolated.sum(-1) / counts

    expected_degree = (p_bond * order[1:].sum() / 3.0).sum(-1)          # placeholder shape
    # Expected consumed valence per atom, using the per-class orders.
    consumed = (probabilities * pair.unsqueeze(-1) * order).sum((-1, -2))
    caps = torch.as_tensor(ATOM_VALENCE, device=bond_logits.device, dtype=p_bond.dtype)
    atom_probabilities = torch.softmax(atom_logits, dim=-1)
    expected_cap = (atom_probabilities * caps).sum(-1)
    overflow = torch.relu(consumed - expected_cap) * mask.to(p_bond.dtype)
    valence_overflow = overflow.sum(-1) / counts
    del expected_degree

    return {
        "bonds_per_atom": bonds_per_atom,
        "cycle_rank": cycle_rank,
        "double_bond_fraction": double_bond_fraction,
        "isolated_atom_rate": isolated_atom_rate,
        "valence_overflow": valence_overflow,
    }


def structure_loss(statistics: dict[str, torch.Tensor], targets: StructureTargets, *,
                   weights: dict[str, float] | None = None
                   ) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted squared error between batch MEANS and the training means.

    Moments only: the model is never told what any individual molecule should look like,
    so this cannot memorise targets or leak a label.
    """
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    unknown = set(weights) - set(TERMS)
    if unknown:
        raise ValueError(f"unknown structure terms {sorted(unknown)}")
    total = None
    report: dict[str, float] = {}
    for name in TERMS:
        weight = float(weights.get(name, 0.0))
        value = statistics[name].mean()
        target = float(getattr(targets, name))
        term = (value - target) ** 2
        report[f"{name}_value"] = float(value.detach())
        report[f"{name}_target"] = target
        report[f"{name}_squared_error"] = float(term.detach())
        if weight:
            total = term * weight if total is None else total + term * weight
    if total is None:
        raise ValueError("every structure weight is zero; nothing would be optimised")
    report["structure_loss"] = float(total.detach())
    return total, report


@torch.no_grad()
def structure_report(flow, autoencoder, *, counts: Sequence[int],
                     condition: torch.Tensor | None, targets: StructureTargets,
                     temperature: float = 1.0, device: torch.device | str = "cpu",
                     generator: torch.Generator | None = None) -> dict[str, float]:
    """The same statistics on fresh samples, for logging (no gradient)."""
    from zinc_gnf.flow import node_mask_from_counts
    from zinc_gnf.generation import generation_batch

    mask = node_mask_from_counts(counts, device=device)
    z = flow.prior.sample(mask, condition, temperature=temperature, generator=generator)
    embeddings = flow.inverse(z, mask, condition)
    batch = generation_batch(counts, device=device)
    outputs = autoencoder.decode(embeddings, batch, teacher_forcing=False)
    statistics = structure_statistics(outputs["bond_logits"], outputs["atom_logits"], mask)
    _, report = structure_loss(statistics, targets)
    return report


__all__ = [
    "DEFAULT_WEIGHTS",
    "TERMS",
    "StructureTargets",
    "structure_loss",
    "structure_report",
    "structure_statistics",
]
