"""Chemically constrained bond decoding for the charge-aware representation.

A port of the teammates' `decode_bonds_valence_aware` (molecular_generation/
qm9_chem.py) to the 17 (element, charge) atom types and their typed valence
caps. It is used to turn decoder bond probabilities into a bond matrix that
respects each atom's valence and is connected whenever the caps allow:

1. candidate bonds are processed in decreasing confidence; each is accepted
   at its predicted order, downgraded (e.g. double -> single) when the
   remaining valence of either atom is smaller;
2. a connectivity pass repeatedly joins the two components linked by the
   highest-probability bond that still fits both atoms' remaining valence.

Raw (argmax) decoding and constrained decoding are reported SEPARATELY by the
evaluation code; this module is only the constrained path.

Limitation: the caps bound total bond order, not formal-charge consistency.
An atom type is taken as given (the decoder predicts it), and hydrogens are
inferred later by RDKit from valence.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from zinc_gnf.constants_charge_aware import ATOM_VALENCE, NUM_BOND_TYPES


def valence_aware_bond_decode_typed(
    atom_types: Sequence[int] | np.ndarray,
    bond_probabilities: np.ndarray,
    valence_caps: Sequence[int] = ATOM_VALENCE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a valence-respecting bond matrix from `[n, n, 4]` class probabilities.

    Returns `(bond_matrix, info)`; `info["connected"]` says whether a single
    component was reached, `info["rescue_edges"]` counts bonds added only by
    the connectivity pass.
    """
    atoms = np.asarray(atom_types, dtype=np.int64).reshape(-1)
    probs = np.asarray(bond_probabilities, dtype=np.float64)
    n = int(atoms.shape[0])
    if probs.shape != (n, n, NUM_BOND_TYPES):
        raise ValueError(f"bond_probabilities shape {probs.shape} != ({n}, {n}, {NUM_BOND_TYPES})")
    if not np.isfinite(probs).all():
        raise ValueError("bond probabilities contain non-finite values")
    probs = 0.5 * (probs + probs.transpose(1, 0, 2))      # bonds are undirected

    remaining = np.asarray([valence_caps[int(a)] for a in atoms], dtype=np.int64)
    bond = np.zeros((n, n), dtype=np.int64)

    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            predicted = int(np.argmax(probs[i, j]))
            if predicted == 0:
                continue
            candidates.append((float(probs[i, j, predicted]), i, j, predicted))
    candidates.sort(key=lambda c: c[0], reverse=True)

    for _, i, j, predicted in candidates:
        order = min(predicted, int(remaining[i]), int(remaining[j]))
        if order >= 1:
            bond[i, j] = bond[j, i] = order
            remaining[i] -= order
            remaining[j] -= order

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if bond[i, j] != 0:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    max_order = NUM_BOND_TYPES - 1
    rescue_edges = 0
    while len({find(i) for i in range(n)}) > 1:
        best: tuple[float, int, int, int] | None = None
        for i in range(n):
            if remaining[i] <= 0:
                continue
            ri = find(i)
            for j in range(i + 1, n):
                if remaining[j] <= 0 or find(j) == ri:
                    continue
                for order in range(1, min(int(remaining[i]), int(remaining[j]), max_order) + 1):
                    p = float(probs[i, j, order])
                    if best is None or p > best[0]:
                        best = (p, i, j, order)
        if best is None:
            break                                   # valence exhausted: stays disconnected
        _, i, j, order = best
        bond[i, j] = bond[j, i] = order
        remaining[i] -= order
        remaining[j] -= order
        parent[find(i)] = find(j)
        rescue_edges += 1

    return bond, {
        "connected": len({find(i) for i in range(n)}) <= 1,
        "rescue_edges": rescue_edges,
        "valence_used": int(bond.sum()),
    }


__all__ = ["valence_aware_bond_decode_typed"]
