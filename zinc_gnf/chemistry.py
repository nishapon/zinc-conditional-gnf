"""RDKit interactions for ZINC250K: SMILES encoding, graph reconstruction,
and chemistry-aware bond decoding for generation (Slice 5).

Design principles:
  * This module is PURE: no I/O, no network calls, no multiprocessing. It
    just encodes/decodes molecules via RDKit, so each function can be tested
    in isolation on a laptop with a one-second roundtrip test.
  * The representation is explicitly NON-STEREOCHEMICAL (see
    canonical_connectivity_smiles below). Our (atom_types, bond_matrix)
    representation does not carry stereo information, so comparing
    reconstructed vs. original SMILES must likewise ignore stereo.
  * Encoder rejects molecules it cannot represent EXACTLY. "Reject" means
    returning (None, reason) rather than silently approximating. The caller
    (preprocessing.py) counts rejections per reason and reports them, so no
    data loss is hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import QED

from zinc_gnf.constants import (
    ATOM_TO_INDEX,
    ATOM_VALENCE,
    ATOM_VOCAB,
    BOND_TO_INDEX,
    INDEX_TO_BOND,
    MAX_NODES,
    MIN_NODES,
    NUM_BOND_TYPES,
    REPRESENTATION_VERSION,
)


# ---------------------------------------------------------------------------
# Canonical SMILES — non-stereochemical form.
#
# ZINC250K SMILES carry stereo markers (@, /, \). Our (atom_types, bond_matrix)
# representation does not store stereo, so an honest roundtrip check must
# first strip stereo from the input before comparing against the
# reconstruction. This function is the single place that decides what
# "canonical identity" means for us — downstream code must call it rather than
# writing ad-hoc Chem.MolToSmiles(...) invocations.
# ---------------------------------------------------------------------------
def canonical_connectivity_smiles(molecule: Chem.Mol) -> str:
    """Non-stereochemical canonical SMILES.

    We clone the input to avoid mutating the caller's molecule, then strip
    stereochemistry and atom-map numbers, then canonicalize with
    `isomericSmiles=False` so no stereo markers leak in via a late
    re-perception. The resulting string is the ground-truth identity of
    the molecule in our pipeline.
    """
    molecule = Chem.Mol(molecule)
    Chem.RemoveStereochemistry(molecule)

    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)

    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=False,
    )


# ---------------------------------------------------------------------------
# Encoding: SMILES → (atom_types, bond_matrix, qed, n_node).
# ---------------------------------------------------------------------------
def encode_zinc_smiles(
    smiles: str,
    *,
    min_nodes: int = MIN_NODES,
    max_nodes: int = MAX_NODES,
) -> tuple[dict[str, Any] | None, str | None]:
    """Encode a single ZINC SMILES into our graph representation.

    Returns `(record, None)` on success, or `(None, reason)` on rejection.

    The pipeline is strict: a molecule survives only if it is parseable,
    a single fragment, chemically well-formed (neutral, no isotopes, no
    radicals), in-vocabulary, kekulizable, and roundtrips exactly through
    `graph_to_mol` under the non-stereochemical canonical form.
    """
    # Guard against unexpected input types from the CSV (pandas sometimes
    # returns float NaN for empty cells; defensive cast first).
    if not isinstance(smiles, str) or not smiles:
        return None, "empty_smiles"

    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        return None, "parse_failure"

    if len(Chem.GetMolFrags(molecule)) != 1:
        return None, "multiple_fragments"

    # Single early pass over atoms covering every atom-level reason. This
    # order matters: we check chemical validity (isotope, radical, charge,
    # vocab) BEFORE the size check so e.g. [NH4+] is reported as
    # non_neutral rather than as a 1-atom size reject, which is more
    # informative when the per-reason counts show up in the report.
    for atom in molecule.GetAtoms():
        if atom.GetIsotope() != 0:
            return None, "isotope"
        if atom.GetNumRadicalElectrons() != 0:
            return None, "radical"
        if atom.GetFormalCharge() != 0:
            return None, "non_neutral"
        if atom.GetSymbol() not in ATOM_TO_INDEX:
            return None, "unsupported_element"

    n_node = molecule.GetNumAtoms()
    if n_node < min_nodes:
        return None, "node_count_below_min"
    if n_node > max_nodes:
        return None, "node_count_above_max"

    # Canonicalize before further processing so we have a single stable
    # identity string to roundtrip against at the end.
    canonical = canonical_connectivity_smiles(molecule)

    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(canonical)

    if molecule is None:
        return None, "canonical_parse_failure"

    # Rebuild atom_types from the canonical form. The vocabulary + neutrality
    # checks already passed on the pre-canonical molecule, which has the same
    # atoms, so these lookups are guaranteed safe.
    atom_types = [
        ATOM_TO_INDEX[atom.GetSymbol()] for atom in molecule.GetAtoms()
    ]

    # Kekulize so aromatic bonds collapse to single/double. If kekulization
    # fails, the molecule has an aromatic system we can't represent cleanly.
    kekule = Chem.Mol(molecule)
    try:
        with rdBase.BlockLogs():
            Chem.Kekulize(kekule, clearAromaticFlags=True)
    except Exception:  # noqa: BLE001 — RDKit raises various C++ exceptions.
        return None, "kekulization_failure"

    # Build the bond matrix. Any bond type not in BOND_TO_INDEX (e.g. a
    # bond that remained aromatic post-kekulize, or an unusual order) is a
    # hard reject.
    bond_matrix = np.zeros((n_node, n_node), dtype=np.int64)
    for bond in kekule.GetBonds():
        order = BOND_TO_INDEX.get(bond.GetBondType())
        if order is None:
            return None, "unsupported_bond"
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_matrix[i, j] = order
        bond_matrix[j, i] = order

    atom_types = np.asarray(atom_types, dtype=np.int64)

    # Pre-check valence: an in-vocab, neutral, kekulized molecule should
    # always satisfy per-atom standard valence. Any exception here points
    # to a dataset entry outside what our vocabulary models.
    per_atom_bond_order = bond_matrix.sum(axis=1)
    atom_valence_cap = np.asarray(
        [ATOM_VALENCE[int(idx)] for idx in atom_types],
        dtype=np.int64,
    )
    if np.any(per_atom_bond_order > atom_valence_cap):
        return None, "valence_overflow"

    # Round-trip: reconstruct a mol from our representation and verify the
    # non-stereochemical canonical SMILES is identical. This catches any
    # subtle encoding issue (hydrogen counts, kekulization asymmetry, etc.)
    # before the record ever reaches training.
    try:
        reconstructed = graph_to_mol(atom_types, bond_matrix)
    except Exception:  # noqa: BLE001
        return None, "reconstruction_failure"

    reconstructed_canonical = canonical_connectivity_smiles(reconstructed)
    if reconstructed_canonical != canonical:
        return None, "roundtrip_mismatch"

    # Compute QED on the reconstructed mol (not the original parse) so the
    # value stored for training is produced by the EXACT same path that
    # Slice 5 generation will use when scoring generated molecules.
    try:
        rdkit_qed = float(QED.qed(reconstructed))
    except Exception:  # noqa: BLE001
        return None, "qed_computation_failure"

    if not np.isfinite(rdkit_qed) or not 0.0 <= rdkit_qed <= 1.0:
        return None, "qed_out_of_range"

    record = {
        "representation": REPRESENTATION_VERSION,
        "original_smiles": smiles,
        "smiles": canonical,            # non-isomeric canonical form
        "atom_types": atom_types,       # np.int64, shape [n_node]
        "bond_matrix": bond_matrix,     # np.int64, shape [n_node, n_node]
        "n_node": int(n_node),
        "qed": rdkit_qed,               # RDKit-recomputed; stored for training
    }
    return record, None


# ---------------------------------------------------------------------------
# Decoding: (atom_types, bond_matrix) → RDKit mol.
# ---------------------------------------------------------------------------
def graph_to_mol(
    atom_types: np.ndarray,
    bond_matrix: np.ndarray,
) -> Chem.Mol:
    """Reconstruct an RDKit mol from our neutral, implicit-H representation.

    Hydrogens are left implicit — RDKit will compute them from valence at
    sanitize time. Any input that cannot be sanitized raises; the caller is
    expected to catch (encode_zinc_smiles treats such failures as rejections,
    constrained_bond_decode treats them as sample-level failures).
    """
    atom_types = np.asarray(atom_types, dtype=np.int64).reshape(-1)
    bond_matrix = np.asarray(bond_matrix, dtype=np.int64)

    n_node = atom_types.shape[0]
    if bond_matrix.shape != (n_node, n_node):
        raise ValueError(
            f"bond_matrix shape {bond_matrix.shape} "
            f"does not match atom count {n_node}"
        )
    if not np.array_equal(bond_matrix, bond_matrix.T):
        raise ValueError("bond_matrix must be symmetric")
    if np.any(np.diag(bond_matrix) != 0):
        raise ValueError("bond_matrix diagonal must be zero")

    editable = Chem.RWMol()
    for atom_index in atom_types:
        atom = Chem.Atom(ATOM_VOCAB[int(atom_index)])
        atom.SetFormalCharge(0)         # we only model neutral atoms
        atom.SetNoImplicit(False)       # let RDKit infer hydrogens
        editable.AddAtom(atom)

    for i in range(n_node):
        for j in range(i + 1, n_node):
            order = int(bond_matrix[i, j])
            if order == 0:
                continue
            if order not in INDEX_TO_BOND:
                raise ValueError(f"unsupported bond class: {order}")
            editable.AddBond(i, j, INDEX_TO_BOND[order])

    molecule = editable.GetMol()
    Chem.SanitizeMol(molecule)
    return molecule


# ---------------------------------------------------------------------------
# Valence-aware bond decoding for generation (used in Slice 5).
#
# Lives in this module because it's part of the chemistry layer and shares
# the vocabulary / valence tables. The generator (Slice 5) imports this
# function; preprocessing never uses it.
# ---------------------------------------------------------------------------
@dataclass
class UnionFind:
    """Disjoint-set structure used to enforce connectivity in decoding."""

    size: int

    def __post_init__(self) -> None:
        self.parent = list(range(self.size))
        self.rank = [0] * self.size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True

    def connected(self) -> bool:
        if self.size <= 1:
            return True
        root = self.find(0)
        return all(self.find(i) == root for i in range(1, self.size))


def valence_aware_bond_decode(
    atom_types: np.ndarray,
    bond_probabilities: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Build a connected, valence-respecting bond matrix from decoder output.

    Processes candidate (i, j) pairs in descending log-odds order (real-bond
    vs. no-bond). Each candidate is accepted at the highest supported bond
    order that fits within both endpoints' remaining valence. Then a
    connectivity-rescue pass links any disconnected components using the
    highest-probability inter-component bond that still has valence room.

    Returns (bond_matrix, info) on success, or (None, info) when connectivity
    cannot be achieved within valence constraints. `info` always records
    `tree_edges`, `extra_edges`, `connected`, and `reason` so the generator
    can report per-sample decode quality.

    NOT USED in Slice 1 (preprocessing). Co-located here because it shares
    the vocabulary and valence tables.
    """
    atom_types = np.asarray(atom_types, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(bond_probabilities, dtype=np.float64)

    n_node = int(atom_types.shape[0])
    expected = (n_node, n_node, NUM_BOND_TYPES)
    if probabilities.shape != expected:
        raise ValueError(
            f"bond_probabilities shape {probabilities.shape} "
            f"does not match expected {expected}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("bond probabilities contain non-finite values")

    # Enforce symmetry: molecular bonds are undirected.
    probabilities = 0.5 * (probabilities + probabilities.transpose(1, 0, 2))

    capacity = np.asarray(
        [ATOM_VALENCE[int(idx)] for idx in atom_types], dtype=np.int64,
    )
    used = np.zeros(n_node, dtype=np.int64)
    bond_matrix = np.zeros((n_node, n_node), dtype=np.int64)

    # Score each (i, j) pair by log-odds of "any real bond" vs. "no bond".
    candidates: list[dict[str, Any]] = []
    for i in range(n_node):
        for j in range(i + 1, n_node):
            best_order = int(np.argmax(probabilities[i, j, 1:]) + 1)
            real_prob = float(probabilities[i, j, best_order])
            no_bond_prob = float(probabilities[i, j, 0])
            log_odds = float(
                np.log(real_prob + 1e-12) - np.log(no_bond_prob + 1e-12)
            )
            candidates.append(
                {
                    "i": i,
                    "j": j,
                    "best_order": best_order,
                    "real_prob": real_prob,
                    "log_odds": log_odds,
                }
            )
    candidates.sort(key=lambda c: (c["log_odds"], c["real_prob"]), reverse=True)

    components = UnionFind(n_node)
    tree_edges = 0

    # Pass 1: lay down a spanning backbone of single bonds. We walk candidates
    # top-down and accept any that joins two currently-disjoint components
    # (so the graph becomes connected ASAP).
    for cand in candidates:
        i, j = cand["i"], cand["j"]
        if components.find(i) == components.find(j):
            continue
        if used[i] >= capacity[i] or used[j] >= capacity[j]:
            continue
        bond_matrix[i, j] = 1
        bond_matrix[j, i] = 1
        used[i] += 1
        used[j] += 1
        components.union(i, j)
        tree_edges += 1
        if components.connected():
            break

    if not components.connected():
        return None, {
            "connected": False,
            "tree_edges": tree_edges,
            "extra_edges": 0,
            "upgraded_edges": 0,
            "reason": "insufficient_valence_for_connectivity",
        }

    # Pass 2: upgrade backbone edges to their model-preferred bond order
    # when valence allows and the model actually wanted a multiple bond.
    upgraded_edges = 0
    for cand in candidates:
        i, j = cand["i"], cand["j"]
        if bond_matrix[i, j] == 0:
            continue
        if cand["log_odds"] <= 0 or cand["best_order"] <= 1:
            continue
        want_extra = cand["best_order"] - int(bond_matrix[i, j])
        allowed = min(want_extra, capacity[i] - used[i], capacity[j] - used[j])
        if allowed > 0:
            new_order = int(bond_matrix[i, j]) + allowed
            bond_matrix[i, j] = new_order
            bond_matrix[j, i] = new_order
            used[i] += allowed
            used[j] += allowed
            upgraded_edges += 1

    # Pass 3: add additional model-supported edges beyond the spanning tree.
    extra_edges = 0
    for cand in candidates:
        i, j = cand["i"], cand["j"]
        if bond_matrix[i, j] != 0:
            continue
        if cand["log_odds"] <= 0:
            continue
        headroom = min(capacity[i] - used[i], capacity[j] - used[j])
        if headroom < 1:
            continue
        order = min(cand["best_order"], headroom)
        bond_matrix[i, j] = order
        bond_matrix[j, i] = order
        used[i] += order
        used[j] += order
        extra_edges += 1

    return bond_matrix, {
        "connected": True,
        "tree_edges": tree_edges,
        "extra_edges": extra_edges,
        "upgraded_edges": upgraded_edges,
        "reason": "",
    }


__all__ = [
    "canonical_connectivity_smiles",
    "encode_zinc_smiles",
    "graph_to_mol",
    "UnionFind",
    "valence_aware_bond_decode",
]
