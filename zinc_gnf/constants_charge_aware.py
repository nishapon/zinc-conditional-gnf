"""Charge-aware ZINC250K representation constants (representation v2).

The neutral-only baseline (`zinc_gnf.constants`, representation
`zinc250k_neutral_kekule_v1`) is NOT modified by this module and remains the
documented baseline. This module defines a separate, versioned representation
in which every atom is an (element, formal charge) type, so charged molecules
(protonated amines, carboxylates, nitro groups, zwitterions, ...) are encoded
losslessly instead of being rejected.

Vocabulary provenance
---------------------
The charged atom types are the ones MEASURED on all 249,455 ZINC250K
molecules by the Phase 0 chemistry diagnostics (ISML job 323813), in order of
decreasing atom count:

    N+1  76,787     O-1  24,276     N-1  1,539     S-1  446
    O+1      18     S+1       6     C-1      3     P+1    2

No frequency threshold is applied and no type is discarded. If a molecule
contains an (element, charge) pair outside this table the encoder rejects it
with the explicit reason `unsupported_charge_type` and the preprocessing report
lists the offending types; it is never dropped silently.

Index compatibility
-------------------
Indices 0..8 are exactly the nine neutral elements of representation v1
(`C N O F P S Cl Br I`), so a neutral molecule has identical `atom_types` in
both representations. Charged types occupy indices 9..16.

Valence caps
------------
For each type the cap is the largest total bond order (sum of bond orders to
heavy atoms) for which RDKit sanitization succeeds, measured with RDKit and
pinned by `tests/test_charge_aware_chemistry.py::test_valence_caps_match_rdkit`.
Hydrogens are implicit and inferred by RDKit from this valence. Neutral caps
are kept identical to v1 so neutral molecules encode identically. v1's cap of
1 for neutral iodine is deliberately conservative (RDKit accepts hypervalent
iodine up to 5); it is the only cap below RDKit's maximum, and the full
ZINC250K run produced zero `valence_overflow` rejections.
"""

from __future__ import annotations

import hashlib

from zinc_gnf.constants import (
    ATOM_VOCAB as NEUTRAL_ELEMENT_VOCAB,
    ATOM_VALENCE as NEUTRAL_ATOM_VALENCE,
    BOND_TO_INDEX,
    INDEX_TO_BOND,
    MAX_NODES,
    MIN_NODES,
    NUM_BOND_TYPES,
)

REPRESENTATION_VERSION_CHARGE_AWARE = "zinc250k_charge_aware_kekule_v2"

# (element, formal charge, valence cap), in fixed order = type index - 9.
_CHARGED_TYPES: tuple[tuple[str, int, int], ...] = (
    ("N", +1, 4),
    ("O", -1, 1),
    ("N", -1, 2),
    ("S", -1, 5),
    ("O", +1, 3),
    ("S", +1, 5),
    ("C", -1, 3),
    ("P", +1, 4),
)

CHARGED_ATOM_TYPES: tuple[tuple[str, int], ...] = tuple(
    (symbol, charge) for symbol, charge, _ in _CHARGED_TYPES
)

# Full ordered list of (element, charge) pairs.
ATOM_TYPES: tuple[tuple[str, int], ...] = (
    tuple((symbol, 0) for symbol in NEUTRAL_ELEMENT_VOCAB) + CHARGED_ATOM_TYPES
)


def atom_type_label(symbol: str, charge: int) -> str:
    """`C`, `N+1`, `O-1` (matches the Phase 0 diagnostics labels)."""
    return symbol if charge == 0 else f"{symbol}{charge:+d}"


ATOM_VOCAB: tuple[str, ...] = tuple(
    atom_type_label(symbol, charge) for symbol, charge in ATOM_TYPES
)
NUM_ATOM_TYPES: int = len(ATOM_VOCAB)
NODE_FEATURE_DIM: int = NUM_ATOM_TYPES  # one-hot over (element, charge) types

ATOM_TYPE_TO_INDEX: dict[tuple[str, int], int] = {
    atom_type: index for index, atom_type in enumerate(ATOM_TYPES)
}
LABEL_TO_INDEX: dict[str, int] = {
    label: index for index, label in enumerate(ATOM_VOCAB)
}
ELEMENT_OF_TYPE: tuple[str, ...] = tuple(symbol for symbol, _ in ATOM_TYPES)
CHARGE_OF_TYPE: tuple[int, ...] = tuple(charge for _, charge in ATOM_TYPES)
SUPPORTED_ELEMENTS: frozenset[str] = frozenset(NEUTRAL_ELEMENT_VOCAB)

ATOM_VALENCE: tuple[int, ...] = (
    tuple(NEUTRAL_ATOM_VALENCE) + tuple(cap for _, _, cap in _CHARGED_TYPES)
)
VALENCE: dict[str, int] = dict(zip(ATOM_VOCAB, ATOM_VALENCE))

NEUTRAL_PREFIX_LENGTH: int = len(NEUTRAL_ELEMENT_VOCAB)


def vocabulary_fingerprint() -> str:
    """Short hash of (version, labels, caps) to detect vocabulary drift.

    Stored in preprocessing outputs so a checkpoint trained on one
    vocabulary cannot be silently paired with data from another.
    """
    text = "|".join(
        [REPRESENTATION_VERSION_CHARGE_AWARE]
        + [f"{label}:{cap}" for label, cap in zip(ATOM_VOCAB, ATOM_VALENCE)]
    )
    return hashlib.sha256(text.encode("ascii")).hexdigest()[:16]


assert len(ATOM_VOCAB) == len(set(ATOM_VOCAB)) == len(ATOM_VALENCE)
assert tuple(ATOM_VOCAB[:NEUTRAL_PREFIX_LENGTH]) == tuple(NEUTRAL_ELEMENT_VOCAB)


__all__ = [
    "ATOM_TYPES",
    "ATOM_TYPE_TO_INDEX",
    "ATOM_VALENCE",
    "ATOM_VOCAB",
    "BOND_TO_INDEX",
    "CHARGED_ATOM_TYPES",
    "CHARGE_OF_TYPE",
    "ELEMENT_OF_TYPE",
    "INDEX_TO_BOND",
    "LABEL_TO_INDEX",
    "MAX_NODES",
    "MIN_NODES",
    "NEUTRAL_PREFIX_LENGTH",
    "NODE_FEATURE_DIM",
    "NUM_ATOM_TYPES",
    "NUM_BOND_TYPES",
    "REPRESENTATION_VERSION_CHARGE_AWARE",
    "SUPPORTED_ELEMENTS",
    "VALENCE",
    "atom_type_label",
    "vocabulary_fingerprint",
]
