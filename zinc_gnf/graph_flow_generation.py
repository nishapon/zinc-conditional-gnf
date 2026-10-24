"""Stage 22: native conditional sampling from the direct graph flow, and its decoding.

There is no decoder here, and that is the point of the stage. The flow emits dequantized
atom and bond tensors; `argmax` reads the discrete graph straight off them, and
`graph_to_mol_charge_aware` - the same function the rest of the project uses - turns that
graph into a molecule. Nothing is assembled, nothing is repaired by an integer program,
and no property model is consulted.

Two numbers are reported and never merged:

  RAW        the argmax graph, no chemistry at all. THIS is the architectural number, and
             it is the one comparable with "validity without check" in the literature
             (MoFlow reported 82% and GraphAF 68% on ZINC250K) and with our embedding
             pipeline's ~14% (unconditional arm B) and ~9% (conditional).
  CORRECTED  after MoFlow's post-hoc valency correction. Reported separately because
             correction changes the molecule AFTER sampling, so it is not the model's
             validity. It also drops atoms - it keeps the largest connected component -
             so exact-N must be measured before and after, which it is.

Sampling is native: one pass from p(G | N, QED). No guidance, no reranking, no
generate-many-and-keep-the-best, and the requested QED is never compared against an RDKit
oracle to choose a sample. The only place RDKit appears is sanitisation and measurement.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from rdkit import Chem, rdBase
import torch

from zinc_gnf.constants_charge_aware import ATOM_VALENCE
from zinc_gnf.generation import _molecule_record

MAX_CORRECTION_STEPS_PER_ATOM = 12


def _components(bond: np.ndarray) -> list[list[int]]:
    n = bond.shape[0]
    seen = [False] * n
    found: list[list[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        stack, group = [start], []
        seen[start] = True
        while stack:
            node = stack.pop()
            group.append(node)
            for other in range(n):
                if not seen[other] and bond[node, other] > 0:
                    seen[other] = True
                    stack.append(other)
        found.append(sorted(group))
    return found


def valency_correction(atom_types: np.ndarray, bond_matrix: np.ndarray, *,
                       valence_caps: Sequence[int] = ATOM_VALENCE
                       ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """MoFlow's post-hoc validity correction, applied to our charge-aware atom types.

    As described in the MoFlow paper: while some atom exceeds its valence, sort its bonds
    by order and remove one order from the largest; then keep the largest connected
    component. Our `(element, charge)` atom vocabulary already carries the charge-adjusted
    cap, so MoFlow's special case for N+, S+ and O+ is unnecessary here.

    Returns `(atom_types, bond_matrix, info)` for the corrected molecule. Dropping atoms
    changes N, which is why `info` records how many were kept.
    """
    atoms = np.asarray(atom_types, dtype=np.int64).reshape(-1)
    bonds = np.asarray(bond_matrix, dtype=np.int64).copy()
    caps = np.asarray([int(valence_caps[int(a)]) for a in atoms], dtype=np.int64)
    removed = 0
    budget = MAX_CORRECTION_STEPS_PER_ATOM * max(atoms.size, 1)
    while removed < budget:
        over = np.flatnonzero(bonds.sum(axis=1) > caps)
        if over.size == 0:
            break
        atom = int(over[0])
        partners = np.flatnonzero(bonds[atom] > 0)
        if partners.size == 0:
            break
        target = int(partners[int(np.argmax(bonds[atom, partners]))])
        bonds[atom, target] = bonds[target, atom] = bonds[atom, target] - 1
        removed += 1
    groups = _components(bonds) if atoms.size else [[]]
    keep = max(groups, key=lambda group: (len(group), -group[0] if group else 0))
    index = np.asarray(keep, dtype=np.int64)
    info = {
        "bond_orders_removed": int(removed),
        "atoms_total": int(atoms.size),
        "atoms_kept": int(index.size),
        "components_before_pruning": len(groups),
        "changed": bool(removed > 0 or index.size != atoms.size),
        "note": ("correction happens AFTER sampling and keeps the largest connected "
                 "component, so it can change N; raw validity is reported separately"),
    }
    if index.size == 0:
        return atoms[:0], bonds[:0, :0], info
    return atoms[index], bonds[np.ix_(index, index)], info


def _heavy_atoms(smiles: str | None) -> int:
    if not smiles:
        return -1
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    return int(molecule.GetNumHeavyAtoms()) if molecule is not None else -1


def _ring_statistics(smiles: str | None) -> dict[str, Any]:
    """Rings and aromatic rings, counted exactly as `molecular_profile` counts them."""
    if not smiles:
        return {"rings": None, "aromatic_rings": None}
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return {"rings": None, "aromatic_rings": None}
    info = molecule.GetRingInfo()
    aromatic = sum(1 for ring in info.AtomRings()
                   if all(molecule.GetAtomWithIdx(i).GetIsAromatic() for i in ring))
    return {"rings": int(info.NumRings()), "aromatic_rings": int(aromatic)}


def _entry(atom_types: np.ndarray, bond_matrix: np.ndarray, requested_n: int,
           extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """One decoding attempt, scored with the project's own validity definition."""
    if atom_types.size == 0:
        record: dict[str, Any] = {"sanitizable": False, "connected": False, "valid": False,
                                  "smiles": None, "fragments": None, "qed": None,
                                  "rings": None, "aromatic_rings": None}
    else:
        record = _molecule_record(atom_types, bond_matrix)
        record.update(_ring_statistics(record.get("smiles")))
    record["bonds"] = int((np.asarray(bond_matrix) > 0).sum() // 2)
    record["heavy_atoms"] = _heavy_atoms(record.get("smiles"))
    record["exact_n"] = bool(record["heavy_atoms"] == int(requested_n))
    # these two keys exist so `metrics_generation.decoder_metrics` can score these
    # records unchanged; this model has no fallback mechanism at all
    record["fallback"] = bool((extra or {}).get("changed", False))
    record["fallback_reason"] = (extra or {}).get("reason")
    if extra:
        record["correction"] = extra
    return record


def graphs_to_records(graphs: Sequence[dict[str, Any]], *, requested_n: Sequence[int],
                      correct: bool = True) -> list[dict[str, Any]]:
    """Discrete graphs -> records carrying a `raw` and (optionally) a `corrected` entry."""
    records: list[dict[str, Any]] = []
    for graph, wanted in zip(graphs, requested_n):
        atoms = np.asarray(graph["atom_types"], dtype=np.int64)
        bonds = np.asarray(graph["bond_matrix"], dtype=np.int64)
        record: dict[str, Any] = {"n_node": int(wanted),
                                  "requested_n": int(wanted),
                                  "atom_types": atoms.tolist(),
                                  "nonfinite": not bool(graph.get("finite", True))}
        if record["nonfinite"]:
            empty = np.zeros(0, dtype=np.int64)
            record["raw"] = _entry(empty, np.zeros((0, 0), dtype=np.int64), wanted,
                                   {"reason": "nonfinite_sample"})
            if correct:
                record["corrected"] = dict(record["raw"])
            records.append(record)
            continue
        record["raw"] = _entry(atoms, bonds, wanted)
        if correct:
            fixed_atoms, fixed_bonds, info = valency_correction(atoms, bonds)
            info["reason"] = "valency_correction" if info["changed"] else None
            record["corrected"] = _entry(fixed_atoms, fixed_bonds, wanted, info)
        records.append(record)
    return records


@torch.no_grad()
def generate_conditional(flow, *, scaler, n: int, qed: float, count: int,
                         temperature: float = 1.0, seed: int = 0,
                         device: torch.device | str = "cpu",
                         unconditional: bool = False,
                         correct: bool = True) -> dict[str, Any]:
    """Sample `count` molecules from p(G | N, QED). Native: one pass, no selection.

    `unconditional=True` uses the model's learned null condition instead, which is the
    within-model reference the conditioning-dropout training provides. It is a REFERENCE,
    not a guidance signal: the two are never mixed or extrapolated between.
    """
    from zinc_gnf.conditioning import standardise_condition

    flow.eval()
    device = torch.device(device)
    counts = [int(n)] * int(count)
    if unconditional:
        context = flow.null_context(int(count), device)
    else:
        values = torch.from_numpy(standardise_condition(
            flow.config.condition_features, scaler=scaler, n=counts,
            qed=[float(qed)] * int(count))).to(device)
        context = flow.conditioning(values)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    sampled = flow.sample(counts, context, temperature=float(temperature),
                          generator=generator, device=device)
    graphs = flow.discrete_graphs(sampled)
    records = graphs_to_records(graphs, requested_n=counts, correct=correct)
    for position, record in enumerate(records):
        record["index"] = position
        record["requested_qed"] = float(qed)
        record["temperature"] = float(temperature)
        record["unconditional"] = bool(unconditional)
    nonfinite = sum(1 for r in records if r["nonfinite"])
    return {
        "records": records,
        "settings": {"requested_n": int(n), "requested_qed": float(qed),
                     "count": int(count), "temperature": float(temperature),
                     "seed": int(seed), "unconditional": bool(unconditional),
                     "nonfinite_samples": nonfinite,
                     "nonfinite_rate": nonfinite / max(len(records), 1),
                     "latent_round_trip_relative_error": sampled.get(
                         "latent_round_trip_relative_error"),
                     "round_trip_note": ("sampling is the INVERSE of the flow; if this "
                                         "error is not small the molecules are not drawn "
                                         "from the trained density"),
                     "decoders": ["raw"] + (["corrected"] if correct else []),
                     "note": ("native conditional sampling: no guidance, no reranking, "
                              "no candidate selection, and the target QED is never "
                              "compared with an oracle to choose a sample")},
    }


__all__ = [
    "generate_conditional",
    "graphs_to_records",
    "valency_correction",
]
