"""Charge-aware SMILES <-> graph conversion (representation v2).

Each atom is an (element, formal charge) type from
`zinc_gnf.constants_charge_aware`. Hydrogens stay implicit and are inferred by
RDKit from the typed valence, so `[NH3+]`, `C(=O)[O-]`, `[N+](=O)[O-]`,
`c1cc[nH+]cc1` and the other charge patterns of ZINC250K round-trip exactly.

This module is separate from `zinc_gnf.chemistry` (the neutral-only v1
baseline), which is unchanged. It reuses v1's `canonical_connectivity_smiles`
so "molecular identity" is defined identically in both representations:
non-stereochemical canonical SMILES, which retains formal charges.

Policy (same as v1): the encoder rejects what it cannot represent EXACTLY and
returns `(None, reason)`; it never approximates. Rejections are counted per
reason by the caller. In particular a charge pattern outside the measured
vocabulary is rejected as `unsupported_charge_type`, never dropped silently.

QED is computed on the molecule reconstructed from the stored graph, i.e. the
same path generated molecules take, and it is therefore a property of the
representation. Because charges are preserved this equals the QED of the
originally parsed molecule (verified by tests).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import QED

from zinc_gnf.chemistry import canonical_connectivity_smiles
from zinc_gnf.constants_charge_aware import (
    ATOM_TYPE_TO_INDEX,
    ATOM_VALENCE,
    ATOM_TYPES,
    BOND_TO_INDEX,
    CHARGE_OF_TYPE,
    ELEMENT_OF_TYPE,
    INDEX_TO_BOND,
    MAX_NODES,
    MIN_NODES,
    NUM_ATOM_TYPES,
    NUM_BOND_TYPES,
    REPRESENTATION_VERSION_CHARGE_AWARE,
    SUPPORTED_ELEMENTS,
)

# Every reason encode_zinc_smiles_charge_aware can return, so reports can
# list zeros and tests can assert the set is closed.
REJECTION_REASONS: tuple[str, ...] = (
    "empty_smiles",
    "parse_failure",
    "multiple_fragments",
    "isotope",
    "radical",
    "unsupported_element",
    "unsupported_charge_type",
    "node_count_below_min",
    "node_count_above_max",
    "canonical_parse_failure",
    "kekulization_failure",
    "unsupported_bond",
    "valence_overflow",
    "reconstruction_failure",
    "roundtrip_mismatch",
    "qed_computation_failure",
    "qed_out_of_range",
)


def atom_type_pairs(atom_types: Sequence[int] | np.ndarray) -> list[tuple[str, int]]:
    """`[(element, formal_charge), ...]` for a vector of type indices."""
    return [ATOM_TYPES[int(index)] for index in np.asarray(atom_types).reshape(-1)]


def formal_charges(atom_types: Sequence[int] | np.ndarray) -> np.ndarray:
    """Per-atom formal charges (int64) for a vector of type indices."""
    return np.asarray(
        [CHARGE_OF_TYPE[int(i)] for i in np.asarray(atom_types).reshape(-1)],
        dtype=np.int64,
    )


def net_charge(atom_types: Sequence[int] | np.ndarray) -> int:
    return int(formal_charges(atom_types).sum())


def unsupported_atom_label(symbol: str, charge: int) -> str:
    return symbol if charge == 0 else f"{symbol}{charge:+d}"


def encode_zinc_smiles_charge_aware(
    smiles: str,
    *,
    min_nodes: int = MIN_NODES,
    max_nodes: int = MAX_NODES,
    row_id: int | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Encode one ZINC SMILES. Returns `(record, None)` or `(None, reason)`.

    On an `unsupported_charge_type` rejection the offending types can be
    obtained with `unsupported_charge_types(smiles)`.
    """
    if not isinstance(smiles, str) or not smiles:
        return None, "empty_smiles"

    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None, "parse_failure"

    if len(Chem.GetMolFrags(molecule)) != 1:
        return None, "multiple_fragments"

    # Atom-level checks, in a fixed order so each rejected molecule gets one
    # deterministic reason. Chemical well-formedness first, then vocabulary.
    unsupported_charge = False
    for atom in molecule.GetAtoms():
        if atom.GetIsotope() != 0:
            return None, "isotope"
        if atom.GetNumRadicalElectrons() != 0:
            return None, "radical"
        if atom.GetSymbol() not in SUPPORTED_ELEMENTS:
            return None, "unsupported_element"
        if (atom.GetSymbol(), atom.GetFormalCharge()) not in ATOM_TYPE_TO_INDEX:
            unsupported_charge = True
    if unsupported_charge:
        return None, "unsupported_charge_type"

    n_node = molecule.GetNumAtoms()
    if n_node < min_nodes:
        return None, "node_count_below_min"
    if n_node > max_nodes:
        return None, "node_count_above_max"

    canonical = canonical_connectivity_smiles(molecule)
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(canonical)
    if molecule is None:
        return None, "canonical_parse_failure"

    # Atom order = canonical SMILES order (identical convention to v1).
    atom_types = np.asarray(
        [
            ATOM_TYPE_TO_INDEX[(atom.GetSymbol(), atom.GetFormalCharge())]
            for atom in molecule.GetAtoms()
        ],
        dtype=np.int64,
    )

    kekule = Chem.Mol(molecule)
    try:
        with rdBase.BlockLogs():
            Chem.Kekulize(kekule, clearAromaticFlags=True)
    except Exception:  # noqa: BLE001 - RDKit raises various C++ exceptions.
        return None, "kekulization_failure"

    bond_matrix = np.zeros((n_node, n_node), dtype=np.int64)
    for bond in kekule.GetBonds():
        order = BOND_TO_INDEX.get(bond.GetBondType())
        if order is None:
            return None, "unsupported_bond"
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_matrix[i, j] = order
        bond_matrix[j, i] = order

    capacity = np.asarray(
        [ATOM_VALENCE[int(index)] for index in atom_types], dtype=np.int64,
    )
    if np.any(bond_matrix.sum(axis=1) > capacity):
        return None, "valence_overflow"

    try:
        reconstructed = graph_to_mol_charge_aware(atom_types, bond_matrix)
    except Exception:  # noqa: BLE001
        return None, "reconstruction_failure"

    # Exact identity: the reconstructed molecule must have the same
    # non-stereochemical canonical SMILES, which includes every formal charge.
    if canonical_connectivity_smiles(reconstructed) != canonical:
        return None, "roundtrip_mismatch"

    # QED of the reconstructed molecule (the representation's own label).
    try:
        qed_value = float(QED.qed(reconstructed))
    except Exception:  # noqa: BLE001
        return None, "qed_computation_failure"
    if not np.isfinite(qed_value) or not 0.0 <= qed_value <= 1.0:
        return None, "qed_out_of_range"

    charges = formal_charges(atom_types)
    record: dict[str, Any] = {
        "representation": REPRESENTATION_VERSION_CHARGE_AWARE,
        "row_id": row_id,
        "original_smiles": smiles,
        "smiles": canonical,
        "atom_types": atom_types,
        "bond_matrix": bond_matrix,
        "n_node": int(n_node),
        "qed": qed_value,
        "net_charge": int(charges.sum()),
        "n_charged_atoms": int(np.count_nonzero(charges)),
    }
    return record, None


def unsupported_charge_types(smiles: str) -> list[str]:
    """Labels of (element, charge) pairs in `smiles` outside the vocabulary."""
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles) if smiles else None
    if molecule is None:
        return []
    return sorted({
        unsupported_atom_label(atom.GetSymbol(), atom.GetFormalCharge())
        for atom in molecule.GetAtoms()
        if atom.GetSymbol() in SUPPORTED_ELEMENTS
        and (atom.GetSymbol(), atom.GetFormalCharge()) not in ATOM_TYPE_TO_INDEX
    })


def graph_to_mol_charge_aware(
    atom_types: np.ndarray,
    bond_matrix: np.ndarray,
) -> Chem.Mol:
    """Reconstruct a sanitized RDKit molecule from a typed graph.

    Element and formal charge come from the atom type; hydrogens are left
    implicit and RDKit infers them from valence at sanitization. Raises on
    any input RDKit cannot sanitize (callers treat that as a failed sample).
    """
    atom_types = np.asarray(atom_types, dtype=np.int64).reshape(-1)
    bond_matrix = np.asarray(bond_matrix, dtype=np.int64)

    n_node = atom_types.shape[0]
    if n_node == 0:
        raise ValueError("empty graph")
    if np.any(atom_types < 0) or np.any(atom_types >= NUM_ATOM_TYPES):
        raise ValueError("atom type index out of range")
    if bond_matrix.shape != (n_node, n_node):
        raise ValueError(
            f"bond_matrix shape {bond_matrix.shape} does not match "
            f"atom count {n_node}"
        )
    if not np.array_equal(bond_matrix, bond_matrix.T):
        raise ValueError("bond_matrix must be symmetric")
    if np.any(np.diag(bond_matrix) != 0):
        raise ValueError("bond_matrix diagonal must be zero")
    if np.any(bond_matrix < 0) or np.any(bond_matrix >= NUM_BOND_TYPES):
        raise ValueError("bond class out of range")

    editable = Chem.RWMol()
    for index in atom_types:
        atom = Chem.Atom(ELEMENT_OF_TYPE[int(index)])
        atom.SetFormalCharge(CHARGE_OF_TYPE[int(index)])
        atom.SetNoImplicit(False)       # RDKit infers hydrogens from valence
        editable.AddAtom(atom)

    for i in range(n_node):
        for j in range(i + 1, n_node):
            order = int(bond_matrix[i, j])
            if order == 0:
                continue
            editable.AddBond(i, j, INDEX_TO_BOND[order])

    molecule = editable.GetMol()
    with rdBase.BlockLogs():
        Chem.SanitizeMol(molecule)
    return molecule


__all__ = [
    "REJECTION_REASONS",
    "atom_type_pairs",
    "encode_zinc_smiles_charge_aware",
    "formal_charges",
    "graph_to_mol_charge_aware",
    "net_charge",
    "unsupported_charge_types",
]
