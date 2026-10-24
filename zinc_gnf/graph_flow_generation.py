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

from collections import Counter
from typing import Any, Sequence

import numpy as np
from rdkit import Chem, rdBase
import torch

from zinc_gnf.chemistry_charge_aware import graph_to_mol_charge_aware
from zinc_gnf.constants_charge_aware import ATOM_VALENCE, ATOM_VOCAB
from zinc_gnf.generation import _molecule_record

MAX_CORRECTION_STEPS_PER_ATOM = 12
ATOM_BOND_LABELS = {0: "none", 1: "single", 2: "double", 3: "triple"}


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


def graph_structure_statistics(atom_types: np.ndarray, bond_matrix: np.ndarray, *,
                               valence_caps: Sequence[int] = ATOM_VALENCE
                               ) -> dict[str, Any]:
    """Structural facts about ONE raw graph, and the single reason it is not a molecule.

    Stage 22b reported raw validity 0.0012 without saying why; stage 22c showed the
    answer is "both at once" for 74.2% of samples at N=23. Every graph is therefore
    attributed to exactly one cause so the two failure modes can never be conflated:
    too few or badly placed bonds (disconnection) and too many on one atom (over-valence)
    have opposite fixes.
    """
    atoms = np.asarray(atom_types, dtype=np.int64).reshape(-1)
    bonds = np.asarray(bond_matrix, dtype=np.int64)
    caps = np.asarray([int(valence_caps[int(a)]) for a in atoms], dtype=np.int64)
    totals = bonds.sum(axis=1)
    over = int((totals > caps).sum())
    groups = _components(bonds) if atoms.size else []
    upper = bonds[np.triu_indices(atoms.size, 1)] if atoms.size > 1 else np.zeros(0, int)
    sanitizable = True
    if atoms.size:
        try:
            with rdBase.BlockLogs():
                graph_to_mol_charge_aware(atoms, bonds)
        except Exception:  # noqa: BLE001
            sanitizable = False
    disconnected = len(groups) > 1
    if over and disconnected:
        cause = "over_valent_and_disconnected"
    elif over:
        cause = "over_valent_only"
    elif disconnected:
        cause = "disconnected_only"
    elif not sanitizable:
        cause = "other_sanitisation_failure"
    else:
        cause = "valid"
    return {
        "cause": cause,
        "atoms": int(atoms.size),
        "bonds": int((upper > 0).sum()),
        "bond_density": float((upper > 0).mean()) if upper.size else float("nan"),
        "components": len(groups),
        "largest_component": max((len(g) for g in groups), default=0),
        "isolated_atoms": int(sum(1 for g in groups if len(g) == 1)),
        "over_valent_atoms": over,
        "excess_bond_order": int(np.clip(totals - caps, 0, None).sum()),
        "bond_type_counts": {int(k): int((upper == k).sum()) for k in range(4)},
        "atom_type_counts": {int(k): int((atoms == k).sum())
                             for k in np.unique(atoms)} if atoms.size else {},
    }


def population_structure(graphs: Sequence[dict[str, Any]], *, requested_n: int
                         ) -> dict[str, Any]:
    """Aggregate the per-graph statistics over a sampled population."""
    rows = [graph_structure_statistics(g["atom_types"], g["bond_matrix"]) for g in graphs]
    if not rows:
        return {"samples": 0}
    causes = Counter(r["cause"] for r in rows)
    bond_types: Counter = Counter()
    atom_types: Counter = Counter()
    for row in rows:
        bond_types.update(row["bond_type_counts"])
        atom_types.update(row["atom_type_counts"])
    total_bond_slots = sum(bond_types.values()) or 1
    total_atoms = sum(atom_types.values()) or 1
    lost, removed, exact = [], [], []
    for graph in graphs:
        _, _, info = valency_correction(graph["atom_types"], graph["bond_matrix"])
        lost.append(info["atoms_total"] - info["atoms_kept"])
        removed.append(info["bond_orders_removed"])
        exact.append(int(info["atoms_kept"] == int(requested_n)))
    mean = lambda key: float(np.mean([r[key] for r in rows]))  # noqa: E731
    return {
        "samples": len(rows),
        "requested_n": int(requested_n),
        "bonds_mean": mean("bonds"),
        "bond_density_mean": mean("bond_density"),
        "components_mean": mean("components"),
        "largest_component_mean": mean("largest_component"),
        "largest_component_fraction_of_n": mean("largest_component") / max(requested_n, 1),
        "isolated_atoms_mean": mean("isolated_atoms"),
        "over_valent_atoms_mean": mean("over_valent_atoms"),
        "excess_bond_order_mean": mean("excess_bond_order"),
        "causes": dict(causes),
        "cause_fractions": {k: v / len(rows) for k, v in causes.items()},
        "raw_valid_rate": causes.get("valid", 0) / len(rows),
        "bond_type_fraction": {ATOM_BOND_LABELS[k]: v / total_bond_slots
                               for k, v in sorted(bond_types.items())},
        "atom_type_fraction": {ATOM_VOCAB[k]: v / total_atoms
                               for k, v in sorted(atom_types.items())},
        "correction_atoms_lost_mean": float(np.mean(lost)),
        "correction_orders_removed_mean": float(np.mean(removed)),
        "correction_exact_n_rate": float(np.mean(exact)),
    }


def real_structure_reference(records: Sequence[dict[str, Any]], *, n: int,
                            limit: int = 512) -> dict[str, Any] | None:
    """The same statistics computed on REAL training molecules at this N.

    Without this the generated numbers have no scale: stage 22c only became readable once
    17.4 generated bonds could be set beside 24.7 real ones at N=23.
    """
    rows = [r for r in records if int(r["n_node"]) == int(n)][:int(limit)]
    if not rows:
        return None
    graphs = [{"atom_types": np.asarray(r["atom_types"], dtype=np.int64),
               "bond_matrix": np.asarray(r["bond_matrix"], dtype=np.int64)}
              for r in rows]
    reference = population_structure(graphs, requested_n=int(n))
    reference["source"] = "train_split"
    return reference


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
    if np.asarray(atom_types).size == 0:
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
    structure = population_structure(graphs, requested_n=int(n))
    records = graphs_to_records(graphs, requested_n=counts, correct=correct)
    for position, record in enumerate(records):
        record["index"] = position
        record["requested_qed"] = float(qed)
        record["temperature"] = float(temperature)
        record["unconditional"] = bool(unconditional)
    nonfinite = sum(1 for r in records if r["nonfinite"])
    return {
        "records": records,
        "structure": structure,
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
    "ATOM_BOND_LABELS",
    "generate_conditional",
    "graph_structure_statistics",
    "graphs_to_records",
    "population_structure",
    "real_structure_reference",
    "valency_correction",
]
