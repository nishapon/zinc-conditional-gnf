"""Exact reconstruction ceiling of a permutation-equivariant encoder.

The autoencoder (as in the teammates' design) maps one-hot atom identity plus
the bond graph to node embeddings with a permutation-equivariant network, and
decodes every atom pair from the two embeddings alone. Two atoms that
colour refinement (1-WL, with bond orders as edge labels) cannot tell apart
receive identical embeddings from ANY such encoder, and then the decoder
cannot tell which of them is bonded to a third atom. A molecule is therefore
exactly decodable only if every pair of atoms with the same unordered colour
pair carries the same bond order.

`is_decodable` checks that exact condition. It is an upper bound on exact
reconstruction for a deterministic one-hot encoder with a pairwise decoder,
independent of model size or training. Measured on the real data
(sandbox): ~74% of ZINC250K test molecules against ~93% of QM9, which matches
the teammates' ~90% QM9 reconstruction. Breaking ties needs extra node
features (e.g. random symmetry-breaking noise); see
`AutoencoderConfig.symmetry_noise_dim`, which is OFF by default.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def refine_colors(atom_types: Sequence[int] | np.ndarray, bond_matrix: np.ndarray) -> list[int]:
    """1-WL colour refinement to convergence (bond orders label the edges)."""
    atoms = np.asarray(atom_types, dtype=np.int64).reshape(-1)
    bonds = np.asarray(bond_matrix, dtype=np.int64)
    n = int(atoms.shape[0])
    neighbours = [
        [(j, int(bonds[i, j])) for j in range(n) if bonds[i, j] > 0] for i in range(n)
    ]
    colors = [int(a) for a in atoms]
    for _ in range(n + 1):
        signatures = [
            (colors[i], tuple(sorted((colors[j], b) for j, b in neighbours[i])))
            for i in range(n)
        ]
        ids = {s: k for k, s in enumerate(sorted(set(signatures)))}
        refined = [ids[s] for s in signatures]
        converged = len(set(refined)) == len(set(colors))
        colors = refined
        if converged:
            break
    return colors


def has_indistinguishable_atoms(atom_types, bond_matrix) -> bool:
    return len(set(refine_colors(atom_types, bond_matrix))) < len(np.asarray(atom_types).reshape(-1))


def is_decodable(atom_types, bond_matrix) -> bool:
    """True iff some pairwise decoder on equivariant embeddings could be exact."""
    atoms = np.asarray(atom_types, dtype=np.int64).reshape(-1)
    bonds = np.asarray(bond_matrix, dtype=np.int64)
    n = int(atoms.shape[0])
    colors = refine_colors(atoms, bonds)
    seen: dict[tuple[int, int], int] = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = (min(colors[i], colors[j]), max(colors[i], colors[j]))
            order = int(bonds[i, j])
            if seen.setdefault(key, order) != order:
                return False
    return True


def reconstruction_ceiling(graphs: Sequence[tuple[np.ndarray, np.ndarray]]) -> float:
    """Fraction of `(atom_types, bond_matrix)` graphs that are exactly decodable."""
    if len(graphs) == 0:
        return float("nan")
    return float(np.mean([is_decodable(a, b) for a, b in graphs]))


__all__ = [
    "has_indistinguishable_atoms",
    "is_decodable",
    "reconstruction_ceiling",
    "refine_colors",
]
