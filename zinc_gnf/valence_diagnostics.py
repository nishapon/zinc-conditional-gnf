"""Stage 24: why are the bond COUNTS right and the molecules still invalid?

Stage 23 moved the direct graph flow a long way: at N=23 the generated bond count is 24.45
against 24.65 real, and connectivity is nearly solved (1.20 components). But 89.5% of
graphs still contain an over-valent atom, raw validity is 7.81%, and aromatic rings sit at
0.008 per corrected molecule. Correct global marginals, wrong local correlations.

THE TWO HYPOTHESES THIS MODULE TESTS
------------------------------------
H-VALENCE. The flow factorises as `log p(G|c) = log p_bond(x_b|c) + log p_atom(x_a|A,c)`:
bonds are generated FIRST and atoms are conditioned on them. Valence is a constraint
between an atom's TYPE and the sum of its incident bond orders, so the bond block
structurally cannot respect it - it does not know the atom types yet. The atom block is
the only stage that could, and it is choosing against a marginal that is 73% carbon. If
that is the mechanism, then for most graphs there EXISTS an atom assignment that makes
every atom valence-feasible, and the bond pattern is not the problem.
`best_case_atom_assignment` measures exactly that, and the answer decides whether the
remedy is the factorisation order or the bond distribution.

H-AROMATIC. `valency_correction` removes one order from the LARGEST-order bond of an
over-valent atom. In a Kekule aromatic ring the largest-order bonds are precisely the
DOUBLE bonds, so the corrector preferentially destroys the alternation that aromaticity is
perceived from - and with 89.5% over-valence nearly every molecule passes through it.
`correction_damage` measures aromaticity before and after correction on the same graphs,
which separates "the model never made an aromatic ring" from "the repair destroyed it".

Everything here is read-only measurement. No checkpoint is modified, no molecule is
selected by property, and the oracle structures used by the counterfactuals are real
TRAINING graphs used as a reference - never as a generation aid - and are labelled as such
in every result.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np
from rdkit import Chem, rdBase

from zinc_gnf.chemistry_charge_aware import graph_to_mol_charge_aware
from zinc_gnf.constants_charge_aware import ATOM_VALENCE, ATOM_VOCAB

MAX_VALENCE = max(ATOM_VALENCE)
RING_SIZES_OF_INTEREST = (5, 6)


def _element_and_charge(atom_type: int) -> tuple[str, int]:
    """'N+1' -> ('N', +1); 'C' -> ('C', 0)."""
    label = ATOM_VOCAB[int(atom_type)]
    for marker in ("+", "-"):
        if marker in label:
            element, charge = label.split(marker, 1)
            return element, int(charge) * (1 if marker == "+" else -1)
    return label, 0


def _bond_order_sums(bond_matrix: np.ndarray) -> np.ndarray:
    return np.asarray(bond_matrix, dtype=np.int64).sum(axis=1)


# ------------------------------------------------------------------- valence ---
def valence_profile(graphs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Which atoms are over-valent, by how much, and could any atom type carry the load?

    `infeasible_for_every_type` is the number that separates the two explanations: an atom
    whose bond-order sum exceeds the largest cap in the vocabulary (S, 6) cannot be made
    valid by ANY relabelling, so its BONDS are wrong. An atom that merely exceeds its own
    cap could have been fixed by a different atom type.
    """
    per_type: dict[int, dict[str, Any]] = {}
    per_element: dict[str, dict[str, Any]] = {}
    excess_histogram: Counter = Counter()
    sums_by_type: dict[int, Counter] = {}
    atoms_total = over_total = infeasible_total = 0
    for graph in graphs:
        atoms = np.asarray(graph["atom_types"], dtype=np.int64)
        totals = _bond_order_sums(graph["bond_matrix"])
        for atom_type, total in zip(atoms, totals):
            cap = ATOM_VALENCE[int(atom_type)]
            excess = int(total) - int(cap)
            element, charge = _element_and_charge(int(atom_type))
            atoms_total += 1
            entry = per_type.setdefault(int(atom_type), {
                "label": ATOM_VOCAB[int(atom_type)], "element": element,
                "formal_charge": charge, "allowed_valence": int(cap),
                "atoms": 0, "over_valent": 0, "excess_total": 0,
                "bond_order_sum_total": 0})
            entry["atoms"] += 1
            entry["bond_order_sum_total"] += int(total)
            group = per_element.setdefault(element, {
                "atoms": 0, "over_valent": 0, "excess_total": 0})
            group["atoms"] += 1
            sums_by_type.setdefault(int(atom_type), Counter())[int(total)] += 1
            if excess > 0:
                over_total += 1
                entry["over_valent"] += 1
                entry["excess_total"] += excess
                group["over_valent"] += 1
                group["excess_total"] += excess
                excess_histogram[excess] += 1
            if int(total) > MAX_VALENCE:
                infeasible_total += 1
    for entry in per_type.values():
        entry["over_valent_rate"] = entry["over_valent"] / max(entry["atoms"], 1)
        entry["mean_excess_when_over"] = (entry["excess_total"]
                                          / max(entry["over_valent"], 1))
        entry["mean_bond_order_sum"] = (entry["bond_order_sum_total"]
                                        / max(entry["atoms"], 1))
    for group in per_element.values():
        group["over_valent_rate"] = group["over_valent"] / max(group["atoms"], 1)
    return {
        "atoms": atoms_total,
        "over_valent_atoms": over_total,
        "over_valent_atom_rate": over_total / max(atoms_total, 1),
        "infeasible_for_every_type": infeasible_total,
        "infeasible_rate": infeasible_total / max(atoms_total, 1),
        "max_valence_in_vocabulary": int(MAX_VALENCE),
        "by_atom_type": {ATOM_VOCAB[k]: v for k, v in sorted(per_type.items())},
        "by_element": dict(sorted(per_element.items())),
        "excess_histogram": dict(sorted(excess_histogram.items())),
        "bond_order_sum_histogram": {
            ATOM_VOCAB[k]: dict(sorted(v.items())) for k, v in sorted(sums_by_type.items())},
        "note": ("infeasible_for_every_type counts atoms whose bond-order sum exceeds the "
                 "largest cap in the vocabulary; those cannot be repaired by relabelling "
                 "and indicate the BONDS are wrong rather than the atom choice"),
    }


def joint_atom_bond_distribution(graphs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """P(atom type, bond-order sum), the joint the factorisation has to get right."""
    joint: Counter = Counter()
    for graph in graphs:
        atoms = np.asarray(graph["atom_types"], dtype=np.int64)
        totals = _bond_order_sums(graph["bond_matrix"])
        for atom_type, total in zip(atoms, totals):
            joint[(ATOM_VOCAB[int(atom_type)], int(total))] += 1
    denominator = sum(joint.values()) or 1
    return {
        "pairs": {f"{label}|{total}": count / denominator
                  for (label, total), count in sorted(joint.items())},
        "atoms": denominator,
        "note": "keys are 'atom label|total bond order'",
    }


def best_case_atom_assignment(graphs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """COUNTERFACTUAL: keep the generated bonds, relabel atoms as favourably as possible.

    For each atom, pick the atom type with the smallest cap that still covers its
    bond-order sum. This is not a generation method - it is the measurement that decides
    the remedy:

      * if most graphs become valence-feasible, the generated BOND PATTERN is chemically
        realisable and the failure is the ATOM CHOICE, which is what a factorisation that
        conditions bonds on atoms (or an atom block that sees valence) would fix;
      * if they do not, the bond distribution itself puts more order on an atom than any
        element in the vocabulary can carry, and the bond block is the problem.

    The relabelled atom distribution is reported too, because a "fix" that turns most of
    the molecule into sulphur is not a fix.
    """
    feasible = 0
    changed_atoms = total_atoms = 0
    replacement: Counter = Counter()
    original: Counter = Counter()
    per_graph: list[dict[str, Any]] = []
    order = sorted(range(len(ATOM_VALENCE)), key=lambda i: (ATOM_VALENCE[i], i))
    for graph in graphs:
        atoms = np.asarray(graph["atom_types"], dtype=np.int64)
        totals = _bond_order_sums(graph["bond_matrix"])
        graph_feasible = True
        graph_changed = 0
        for atom_type, total in zip(atoms, totals):
            total_atoms += 1
            original[ATOM_VOCAB[int(atom_type)]] += 1
            choice = next((i for i in order if ATOM_VALENCE[i] >= int(total)), None)
            if choice is None:
                graph_feasible = False
                replacement["<infeasible>"] += 1
                continue
            # keep the original type when it already works, so the count is honest
            if ATOM_VALENCE[int(atom_type)] >= int(total):
                choice = int(atom_type)
            else:
                graph_changed += 1
                changed_atoms += 1
            replacement[ATOM_VOCAB[int(choice)]] += 1
        feasible += int(graph_feasible)
        per_graph.append({"valence_feasible": graph_feasible,
                          "atoms_relabelled": graph_changed})
    denominator = max(total_atoms, 1)
    return {
        "graphs": len(graphs),
        "valence_feasible_graphs": feasible,
        "valence_feasible_rate": feasible / max(len(graphs), 1),
        "atoms_relabelled": changed_atoms,
        "atoms_relabelled_rate": changed_atoms / denominator,
        "original_atom_distribution": {k: v / denominator
                                       for k, v in sorted(original.items())},
        "best_case_atom_distribution": {k: v / denominator
                                        for k, v in sorted(replacement.items())},
        "per_graph": per_graph,
        "note": ("a DIAGNOSTIC relabelling, never a generation method. A high feasible "
                 "rate means the bonds are realisable and the atom choice is the failure"),
    }


def argmax_shift(continuous: np.ndarray, mask: np.ndarray, *,
                 labels: Sequence[str]) -> dict[str, Any]:
    """Does discretisation itself move the marginal, or only sharpen it?

    Compares the argmax histogram with the histogram implied by treating the dequantized
    channel values as unnormalised scores. If argmax concentrates on the majority class
    far more than the soft scores do, the DISCRETISATION is part of the marginal error
    rather than the density being wrong.
    """
    values = np.asarray(continuous, dtype=np.float64)
    keep = np.asarray(mask, dtype=bool)
    rows = values[keep]
    if rows.size == 0:
        return {"rows": 0}
    hard = Counter(int(i) for i in rows.argmax(axis=-1))
    shifted = rows - rows.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    soft = (weights / weights.sum(axis=-1, keepdims=True)).mean(axis=0)
    total = sum(hard.values()) or 1
    return {
        "rows": int(rows.shape[0]),
        "argmax_distribution": {labels[k]: v / total for k, v in sorted(hard.items())},
        "soft_distribution": {labels[k]: float(soft[k]) for k in range(len(labels))},
        "max_absolute_difference": float(max(
            abs(hard.get(k, 0) / total - float(soft[k])) for k in range(len(labels)))),
        "note": ("the soft column treats dequantized values as scores; it is a reference "
                 "for how much argmax sharpens, not a second model"),
    }


# --------------------------------------------------------------- aromaticity ---
def _rings(bond_matrix: np.ndarray) -> list[list[int]]:
    """Smallest-set-of-smallest-rings style cycle basis, computed on the GRAPH.

    Computed without RDKit so that rings can be counted in graphs RDKit refuses to
    sanitize - which is 92% of them at N=23 and exactly the population in question.
    """
    import networkx as nx

    matrix = np.asarray(bond_matrix, dtype=np.int64)
    graph = nx.Graph()
    graph.add_nodes_from(range(matrix.shape[0]))
    for i, j in zip(*np.nonzero(np.triu(matrix, 1))):
        graph.add_edge(int(i), int(j))
    try:
        return [list(cycle) for cycle in nx.minimum_cycle_basis(graph)]
    except Exception:  # noqa: BLE001  - disconnected or degenerate graphs
        return []


def _ordered_cycle(bond_matrix: np.ndarray, ring: Sequence[int]) -> list[int] | None:
    """Put a ring's atoms into adjacency order so bond orders can be read round it."""
    members = set(int(x) for x in ring)
    if len(members) < 3:
        return None
    matrix = np.asarray(bond_matrix, dtype=np.int64)
    start = min(members)
    path = [start]
    seen = {start}
    while len(path) < len(members):
        current = path[-1]
        nxt = next((other for other in sorted(members)
                    if other not in seen and matrix[current, other] > 0), None)
        if nxt is None:
            return None
        path.append(nxt)
        seen.add(nxt)
    if matrix[path[-1], path[0]] <= 0:
        return None
    return path


def ring_bond_pattern(bond_matrix: np.ndarray, ring: Sequence[int]) -> dict[str, Any]:
    """The cyclic sequence of bond orders round a ring, and what kind of pattern it is."""
    cycle = _ordered_cycle(bond_matrix, ring)
    if cycle is None:
        return {"size": len(set(int(x) for x in ring)), "pattern": "not_a_simple_cycle"}
    matrix = np.asarray(bond_matrix, dtype=np.int64)
    orders = [int(matrix[cycle[i], cycle[(i + 1) % len(cycle)]])
              for i in range(len(cycle))]
    distinct = set(orders)
    if distinct == {1}:
        pattern = "all_single"
    elif len(cycle) % 2 == 0 and all(orders[i] != orders[i - 1] for i in range(len(orders))) \
            and distinct == {1, 2}:
        pattern = "alternating"
    elif 2 in distinct:
        pattern = "mixed_with_double"
    else:
        pattern = "other"
    return {"size": len(cycle), "orders": orders, "pattern": pattern,
            "atoms": cycle}


def ring_profile(graphs: Sequence[dict[str, Any]], *, label: str = "generated"
                 ) -> dict[str, Any]:
    """Ring sizes, and - for 5- and 6-rings - what bond-order pattern they carry.

    This is the question "does the model build rings but miss the alternation?". A
    Kekule representation only becomes aromatic to RDKit when the alternation is exactly
    right, so `all_single` rings are cyclohexanes, not benzenes.
    """
    sizes: Counter = Counter()
    patterns: Counter = Counter()
    pattern_by_size: dict[int, Counter] = {}
    ring_atoms: Counter = Counter()
    graphs_with_ring = 0
    for graph in graphs:
        rings = _rings(graph["bond_matrix"])
        graphs_with_ring += int(bool(rings))
        atoms = np.asarray(graph["atom_types"], dtype=np.int64)
        for ring in rings:
            report = ring_bond_pattern(graph["bond_matrix"], ring)
            sizes[report["size"]] += 1
            if report["size"] in RING_SIZES_OF_INTEREST:
                patterns[report["pattern"]] += 1
                pattern_by_size.setdefault(report["size"], Counter())[report["pattern"]] += 1
                for index in report.get("atoms", []):
                    ring_atoms[ATOM_VOCAB[int(atoms[index])]] += 1
    total_rings = sum(sizes.values()) or 1
    total_patterns = sum(patterns.values()) or 1
    total_ring_atoms = sum(ring_atoms.values()) or 1
    return {
        "population": label,
        "graphs": len(graphs),
        "graphs_with_a_ring": graphs_with_ring,
        "graphs_with_a_ring_rate": graphs_with_ring / max(len(graphs), 1),
        "rings_total": sum(sizes.values()),
        "rings_per_graph": sum(sizes.values()) / max(len(graphs), 1),
        "ring_size_distribution": {k: v / total_rings for k, v in sorted(sizes.items())},
        "ring_size_counts": dict(sorted(sizes.items())),
        "pattern_distribution_5_and_6": {k: v / total_patterns
                                         for k, v in sorted(patterns.items())},
        "pattern_by_size": {size: dict(sorted(counter.items()))
                            for size, counter in sorted(pattern_by_size.items())},
        "ring_atom_distribution": {k: v / total_ring_atoms
                                   for k, v in sorted(ring_atoms.items())},
        "note": ("a Kekule 6-ring is only perceived aromatic when the single/double "
                 "alternation is exactly right, so 'all_single' rings are saturated "
                 "rings and not failed benzenes"),
    }


def aromatic_rings_of(atom_types: np.ndarray, bond_matrix: np.ndarray) -> int | None:
    """RDKit's aromatic ring count after sanitisation, or None if it refuses the graph."""
    try:
        with rdBase.BlockLogs():
            molecule = graph_to_mol_charge_aware(atom_types, bond_matrix)
    except Exception:  # noqa: BLE001
        return None
    info = molecule.GetRingInfo()
    return sum(1 for ring in info.AtomRings()
               if all(molecule.GetAtomWithIdx(i).GetIsAromatic() for i in ring))


def correction_damage(graphs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Does the corrector destroy aromaticity that the raw graph had?

    `valency_correction` removes one order from the LARGEST-order bond of an over-valent
    atom, and in a Kekule aromatic ring the largest-order bonds are the DOUBLE bonds. With
    89.5% of graphs over-valent at N=23, almost every molecule passes through that rule.
    Measured on the same graphs before and after, so the two explanations - "the model
    never made an aromatic ring" and "the repair destroyed it" - are separated.
    """
    from zinc_gnf.graph_flow_generation import valency_correction

    before_total = after_total = 0
    sanitizable_before = sanitizable_after = 0
    had_aromatic = kept_aromatic = lost_aromatic = gained_aromatic = 0
    alternating_before = alternating_after = 0
    atoms_lost: list[int] = []
    for graph in graphs:
        atoms = np.asarray(graph["atom_types"], dtype=np.int64)
        bonds = np.asarray(graph["bond_matrix"], dtype=np.int64)
        before = aromatic_rings_of(atoms, bonds)
        fixed_atoms, fixed_bonds, info = valency_correction(atoms, bonds)
        after = (aromatic_rings_of(fixed_atoms, fixed_bonds)
                 if fixed_atoms.size else None)
        atoms_lost.append(info["atoms_total"] - info["atoms_kept"])
        sanitizable_before += int(before is not None)
        sanitizable_after += int(after is not None)
        before_total += before or 0
        after_total += after or 0
        alternating_before += sum(
            1 for ring in _rings(bonds)
            if ring_bond_pattern(bonds, ring).get("pattern") == "alternating")
        if fixed_atoms.size:
            alternating_after += sum(
                1 for ring in _rings(fixed_bonds)
                if ring_bond_pattern(fixed_bonds, ring).get("pattern") == "alternating")
        if before:
            had_aromatic += 1
            if after:
                kept_aromatic += 1
            else:
                lost_aromatic += 1
        elif after:
            gained_aromatic += 1
    count = max(len(graphs), 1)
    return {
        "graphs": len(graphs),
        "sanitizable_before": sanitizable_before,
        "sanitizable_after": sanitizable_after,
        "aromatic_rings_before_total": before_total,
        "aromatic_rings_after_total": after_total,
        "aromatic_rings_before_per_graph": before_total / count,
        "aromatic_rings_after_per_graph": after_total / count,
        "graphs_with_aromatic_before": had_aromatic,
        "aromatic_kept": kept_aromatic,
        "aromatic_lost_to_correction": lost_aromatic,
        "aromatic_gained_by_correction": gained_aromatic,
        "alternating_rings_before": alternating_before,
        "alternating_rings_after": alternating_after,
        "alternating_rings_destroyed": alternating_before - alternating_after,
        "atoms_lost_mean": float(np.mean(atoms_lost)) if atoms_lost else float("nan"),
        "note": ("`aromatic_lost_to_correction` above zero means the repair is destroying "
                 "aromaticity the raw graph already had; `alternating_rings_destroyed` is "
                 "the mechanism, since the rule deletes the largest-order bond"),
    }


def real_reference_profile(records: Sequence[dict[str, Any]], *, n: int | None = None,
                           limit: int = 512) -> dict[str, Any]:
    """The ceiling: the same statistics on REAL training molecules through our own path.

    Answers "do real molecules even retain aromaticity after our serialisation,
    Kekulisation and reconstruction?" - if they do not, no model could.
    """
    rows = [r for r in records if n is None or int(r["n_node"]) == int(n)][:int(limit)]
    if not rows:
        return {"molecules": 0}
    graphs = [{"atom_types": np.asarray(r["atom_types"], dtype=np.int64),
               "bond_matrix": np.asarray(r["bond_matrix"], dtype=np.int64)}
              for r in rows]
    aromatic = [aromatic_rings_of(g["atom_types"], g["bond_matrix"]) for g in graphs]
    resolved = [a for a in aromatic if a is not None]
    atoms: Counter = Counter()
    for graph in graphs:
        atoms.update(ATOM_VOCAB[int(a)] for a in graph["atom_types"])
    denominator = sum(atoms.values()) or 1
    return {
        "molecules": len(graphs),
        "n": n,
        "reconstructable": len(resolved),
        "reconstructable_rate": len(resolved) / max(len(graphs), 1),
        "aromatic_rings_per_molecule": (float(np.mean(resolved)) if resolved
                                        else float("nan")),
        "molecules_with_an_aromatic_ring": sum(1 for a in resolved if a),
        "atom_distribution": {k: v / denominator for k, v in sorted(atoms.items())},
        "valence": valence_profile(graphs),
        "rings": ring_profile(graphs, label="real_train"),
        "source": "train_split",
        "note": ("the ceiling for every generated number here; computed through "
                 "graph_to_mol_charge_aware, the same function generation uses"),
    }


__all__ = [
    "MAX_VALENCE",
    "argmax_shift",
    "aromatic_rings_of",
    "best_case_atom_assignment",
    "correction_damage",
    "joint_atom_bond_distribution",
    "real_reference_profile",
    "ring_bond_pattern",
    "ring_profile",
    "valence_profile",
]
