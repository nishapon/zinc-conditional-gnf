"""Shared molecular representation constants for ZINC250K.

Matches MoFlow's ZINC250K representation (atom types = 9 heavy elements, bond
classes = {no bond, single, double, triple}, kekulized, hydrogens implicit,
neutral molecules only) so our evaluation numbers are directly comparable to
MoFlow's published results.

Why a separate constants module (not inlined into chemistry.py): several
other modules (dataset collation, decoder valence checks, generation
sanitation) need these constants but should NOT depend on RDKit at import
time. Keeping constants free of heavy imports means importing `zinc_gnf`
doesn't drag RDKit into every unit test.
"""

from __future__ import annotations

from rdkit import Chem


# ---------------------------------------------------------------------------
# Representation version string. Bump this when any part of the atom / bond /
# kekulization pipeline changes in a way that breaks checkpoint compatibility.
# ---------------------------------------------------------------------------
REPRESENTATION_VERSION = "zinc250k_neutral_kekule_v1"


# ---------------------------------------------------------------------------
# Atom vocabulary — the 9 heavy elements present in ZINC250K.
#
# MoFlow uses 9+1 (9 real elements + 1 virtual padding atom). We handle
# padding via a boolean node mask instead, so our vocabulary stays exactly 9
# and the "virtual atom" never leaks into decoding or metrics.
# ---------------------------------------------------------------------------
ATOM_VOCAB: tuple[str, ...] = (
    "C", "N", "O", "F", "P", "S", "Cl", "Br", "I",
)

ATOM_TO_INDEX: dict[str, int] = {
    symbol: index for index, symbol in enumerate(ATOM_VOCAB)
}

NUM_ATOM_TYPES: int = len(ATOM_VOCAB)


# ---------------------------------------------------------------------------
# Bond classes:
#   0 = no bond (implicit between any two atoms with no explicit bond)
#   1 = single
#   2 = double
#   3 = triple
#
# Aromatic bonds are kekulized to single/double BEFORE encoding, so no
# separate aromatic class is needed. If RDKit ever hands us an aromatic
# bond at encode time, we treat the molecule as unprocessable and drop it.
# ---------------------------------------------------------------------------
BOND_TO_INDEX: dict[Chem.BondType, int] = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
}

INDEX_TO_BOND: dict[int, Chem.BondType] = {
    index: bond_type for bond_type, index in BOND_TO_INDEX.items()
}

NUM_BOND_TYPES: int = 1 + len(BOND_TO_INDEX)  # = 4


# ---------------------------------------------------------------------------
# Molecule size bounds for ZINC250K.
#
# MIN_NODES = 2 rules out single-atom "molecules" (ZINC250K has none, but we
# guard against degenerate inputs anyway). MAX_NODES = 38 matches MoFlow
# Table 1 exactly.
# ---------------------------------------------------------------------------
MIN_NODES: int = 2
MAX_NODES: int = 38


# ---------------------------------------------------------------------------
# Node feature dimension.
#
# For ZINC250K neutral molecules, each atom contributes exactly its one-hot
# atom type. No separate charge dimensions (all neutral), no separate
# hydrogen-count dimension (RDKit infers hydrogens from valence at decode
# time). This keeps the encoder input dimension at 9 and leaves charge /
# hydrogen prediction out of the architecture entirely.
# ---------------------------------------------------------------------------
NODE_FEATURE_DIM: int = NUM_ATOM_TYPES  # = 9


# ---------------------------------------------------------------------------
# Standard valence (maximum total bond order) for a neutral atom of each
# element in our vocab. Used by:
#   * dataset preprocessing — to drop any molecule whose kekulized valence
#     already exceeds this (shouldn't happen for well-formed ZINC entries
#     but we check);
#   * the decoder's iterative bond refinement — soft "remaining valence"
#     feature fed back into node embeddings;
#   * the generation-time valence-aware greedy bond decoder — hard cap on
#     total predicted bond order per atom.
#
# Values are the standard chemistry textbook defaults:
#   C: 4   N: 3   O: 2   F: 1
#   P: 5   S: 6   Cl: 1  Br: 1  I: 1
# ---------------------------------------------------------------------------
VALENCE: dict[str, int] = {
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "P": 5,
    "S": 6,
    "Cl": 1,
    "Br": 1,
    "I": 1,
}

# Parallel array indexed by ATOM_VOCAB position — convenient for tensorized
# valence lookup via `torch.gather` or NumPy indexing.
ATOM_VALENCE: tuple[int, ...] = tuple(VALENCE[symbol] for symbol in ATOM_VOCAB)


__all__ = [
    "REPRESENTATION_VERSION",
    "ATOM_VOCAB",
    "ATOM_TO_INDEX",
    "NUM_ATOM_TYPES",
    "BOND_TO_INDEX",
    "INDEX_TO_BOND",
    "NUM_BOND_TYPES",
    "MIN_NODES",
    "MAX_NODES",
    "NODE_FEATURE_DIM",
    "VALENCE",
    "ATOM_VALENCE",
]
