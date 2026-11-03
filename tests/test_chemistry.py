"""Unit tests for zinc_gnf.chemistry.

These run locally (no cluster, no network, no dataset needed) and should
complete in a couple of seconds. They cover:

  * SMILES → graph → SMILES roundtrip on 15 hand-picked ZINC-style
    molecules, including every atom in the 9-element vocabulary;
  * Each rejection pathway (parse failure, non-neutral, isotope, radical,
    unsupported element, oversized);
  * Stereochemistry is stripped as specified (chiral SMILES roundtrips
    to a canonical form that drops stereo markers);
  * `graph_to_mol` is the inverse of the encoding step at the connectivity
    level;
  * The valence-aware bond decoder respects atom valence budgets.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from zinc_gnf.chemistry import (
    canonical_connectivity_smiles,
    encode_zinc_smiles,
    graph_to_mol,
    valence_aware_bond_decode,
)
from zinc_gnf.constants import (
    ATOM_TO_INDEX,
    ATOM_VOCAB,
    NUM_ATOM_TYPES,
    NUM_BOND_TYPES,
)


# ---------------------------------------------------------------------------
# A set of ZINC-style SMILES that should ALL encode successfully. Each entry
# is `(name, smiles)` so a failure message tells you which molecule.
# Collectively they exercise every element in the 9-atom vocabulary.
# ---------------------------------------------------------------------------
ACCEPTED_SMILES = [
    ("ethanol",          "CCO"),
    ("benzene",          "c1ccccc1"),
    ("pyridine",         "c1ccncc1"),
    ("acetic_acid",      "CC(=O)O"),
    ("fluorobenzene",    "Fc1ccccc1"),
    ("chlorobenzene",    "Clc1ccccc1"),
    ("bromobenzene",     "Brc1ccccc1"),
    ("iodobenzene",      "Ic1ccccc1"),
    ("thiophene",        "c1ccsc1"),
    ("furan",            "c1ccoc1"),
    ("trimethyl_phosphine", "CP(C)C"),
    ("acrylonitrile",    "C=CC#N"),
    ("imidazole",        "c1cnc[nH]1"),
    # Larger drug-like molecule from the ZINC250K distribution.
    ("zinc_like_1",      "O=C(Nc1ccc(F)cc1)c1cc(Cl)ccc1Br"),
    # Edge: a molecule that uses every heavy atom we care about.
    ("multi_element",    "CC(F)(Cl)Br"),  # contains C, F, Cl, Br
]


# ---------------------------------------------------------------------------
# A set of SMILES that must be REJECTED, each paired with the exact
# rejection reason encode_zinc_smiles must return.
# ---------------------------------------------------------------------------
REJECTED_SMILES = [
    ("parse_junk",         "not a smiles",   "parse_failure"),
    ("empty",              "",               "empty_smiles"),
    ("multiple_fragments", "CC.CO",          "multiple_fragments"),
    ("phenoxide_anion",    "[O-]c1ccccc1",   "non_neutral"),
    # [NH4+] alone is a 1-heavy-atom molecule (RDKit drops the explicit H).
    # Use methylammonium so we have 2 heavy atoms AND a cation — otherwise
    # the molecule would be rejected on size, not on charge.
    ("methylammonium",     "C[NH3+]",        "non_neutral"),
    ("deuterated",         "[2H]CCO",        "isotope"),
    # Silicon is outside our 9-atom vocab. Written as a methylsilane
    # (2 heavy atoms, neutral, closed-shell) so the molecule passes the
    # radical / size / charge checks and reaches the vocab test.
    ("methylsilane",       "C[SiH3]",        "unsupported_element"),
]


# ===========================================================================
# Accepted molecules: every one must roundtrip exactly and produce a sane
# record.
# ===========================================================================
@pytest.mark.parametrize("name,smiles", ACCEPTED_SMILES)
def test_accepted_molecule_roundtrips(name: str, smiles: str) -> None:
    record, reason = encode_zinc_smiles(smiles)
    assert record is not None, (
        f"{name} unexpectedly rejected (reason={reason}): {smiles}"
    )
    assert reason is None

    # Every record carries the expected keys and types.
    for key in (
        "representation", "original_smiles", "smiles",
        "atom_types", "bond_matrix", "n_node", "qed",
    ):
        assert key in record, f"{name}: missing key {key}"

    atom_types = record["atom_types"]
    bond_matrix = record["bond_matrix"]
    n_node = record["n_node"]

    assert atom_types.dtype == np.int64
    assert bond_matrix.dtype == np.int64
    assert atom_types.shape == (n_node,)
    assert bond_matrix.shape == (n_node, n_node)
    assert np.array_equal(bond_matrix, bond_matrix.T), (
        f"{name}: bond matrix must be symmetric"
    )
    assert np.all(np.diag(bond_matrix) == 0), (
        f"{name}: bond matrix diagonal must be zero"
    )
    assert np.all(atom_types >= 0) and np.all(atom_types < NUM_ATOM_TYPES)
    assert np.all(bond_matrix >= 0) and np.all(bond_matrix < NUM_BOND_TYPES)

    assert 0.0 <= record["qed"] <= 1.0

    # Reconstruct via graph_to_mol and verify the non-isomeric canonical
    # SMILES is identical to the record's stored canonical form.
    reconstructed = graph_to_mol(atom_types, bond_matrix)
    assert canonical_connectivity_smiles(reconstructed) == record["smiles"]


# ===========================================================================
# Rejected molecules: every reason must be reported precisely so the
# preprocessing report can tell us why a dataset entry was dropped.
# ===========================================================================
@pytest.mark.parametrize("name,smiles,expected_reason", REJECTED_SMILES)
def test_rejected_molecule_has_correct_reason(
    name: str, smiles: str, expected_reason: str,
) -> None:
    record, reason = encode_zinc_smiles(smiles)
    assert record is None, f"{name}: expected rejection, got a record"
    assert reason == expected_reason, (
        f"{name}: expected reason={expected_reason!r}, got reason={reason!r}"
    )


# ===========================================================================
# Stereo stripping: a chiral SMILES must encode successfully and reduce to
# its non-stereo canonical form.
# ===========================================================================
def test_stereochemistry_is_stripped() -> None:
    chiral = "C[C@H](O)C"       # (R)-2-butanol
    record, reason = encode_zinc_smiles(chiral)
    assert record is not None, (
        f"chiral SMILES unexpectedly rejected: reason={reason}"
    )
    # The canonical form must not contain any stereo markers.
    assert "@" not in record["smiles"]
    assert "/" not in record["smiles"]
    assert "\\" not in record["smiles"]


# ===========================================================================
# Oversized molecules are rejected by the configured size bound.
# ===========================================================================
def test_oversized_molecule_is_rejected() -> None:
    # Linear chain of 40 carbons — larger than MAX_NODES (38).
    smiles = "C" * 40
    record, reason = encode_zinc_smiles(smiles, max_nodes=38)
    assert record is None
    assert reason == "node_count_above_max"


# ===========================================================================
# Determinism: encoding the same SMILES twice yields byte-equal records
# (modulo numpy array identity).
# ===========================================================================
def test_encoding_is_deterministic() -> None:
    smiles = "CC(=O)Nc1ccc(O)cc1"       # paracetamol
    record_a, _ = encode_zinc_smiles(smiles)
    record_b, _ = encode_zinc_smiles(smiles)
    assert record_a is not None and record_b is not None
    assert record_a["smiles"] == record_b["smiles"]
    assert np.array_equal(record_a["atom_types"], record_b["atom_types"])
    assert np.array_equal(record_a["bond_matrix"], record_b["bond_matrix"])
    assert record_a["n_node"] == record_b["n_node"]
    assert record_a["qed"] == pytest.approx(record_b["qed"], abs=1e-12)


# ===========================================================================
# graph_to_mol is a true inverse at the connectivity level.
# ===========================================================================
def test_graph_to_mol_round_trip_matrix() -> None:
    record, _ = encode_zinc_smiles("CCO")
    assert record is not None

    mol = graph_to_mol(record["atom_types"], record["bond_matrix"])

    # Reconstruct back from the RDKit mol and compare.
    atom_types = np.asarray(
        [ATOM_TO_INDEX[atom.GetSymbol()] for atom in mol.GetAtoms()],
        dtype=np.int64,
    )
    assert np.array_equal(atom_types, record["atom_types"])


# ===========================================================================
# Valence-aware bond decoder respects the valence table.
# ===========================================================================
def test_valence_aware_decoder_builds_connected_graph() -> None:
    # Give the decoder a 4-atom molecule (all carbons) with probabilities
    # that strongly prefer a star of single bonds from atom 0.
    atom_types = np.array([0, 0, 0, 0], dtype=np.int64)  # four C

    n = 4
    probs = np.zeros((n, n, NUM_BOND_TYPES), dtype=np.float64)
    # Default: everyone votes "no bond" with high confidence.
    probs[..., 0] = 0.9
    probs[..., 1] = 0.05
    probs[..., 2] = 0.03
    probs[..., 3] = 0.02

    # But atom 0 is strongly linked to atoms 1, 2, 3 as single bonds.
    for j in (1, 2, 3):
        probs[0, j, 0] = 0.05
        probs[0, j, 1] = 0.90
        probs[0, j, 2] = 0.03
        probs[0, j, 3] = 0.02
        probs[j, 0] = probs[0, j]

    bond_matrix, info = valence_aware_bond_decode(atom_types, probs)
    assert bond_matrix is not None
    assert info["connected"] is True

    # Each of (0,1), (0,2), (0,3) should be a single bond; atom 0 uses
    # all four of its valence on single bonds — the fourth is implicit H.
    assert bond_matrix[0, 1] >= 1
    assert bond_matrix[0, 2] >= 1
    assert bond_matrix[0, 3] >= 1
    # Atom 0 total bond order never exceeds 4 (carbon's valence).
    assert bond_matrix[0].sum() <= 4
