"""Round-trip and invariant tests for the charge-aware representation (v2).

These need no dataset, network or cluster. They check that charged molecules
survive SMILES -> typed graph -> molecule EXACTLY: node count, element
identity, formal charges, bond orders (via the non-stereochemical canonical
SMILES) and implicit-hydrogen counts.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from rdkit import Chem, rdBase
from rdkit.Chem import QED

from zinc_gnf import constants as v1_constants
from zinc_gnf import constants_charge_aware as v2
from zinc_gnf.chemistry import canonical_connectivity_smiles, encode_zinc_smiles
from zinc_gnf.chemistry_charge_aware import (
    REJECTION_REASONS,
    atom_type_pairs,
    encode_zinc_smiles_charge_aware,
    formal_charges,
    graph_to_mol_charge_aware,
    net_charge,
    unsupported_charge_types,
)
from zinc_gnf.representations import (
    CHARGE_AWARE_V2,
    NEUTRAL_V1,
    get_representation,
)

# (name, SMILES, expected charge-typed atoms present in the encoding)
CHARGED_MOLECULES = [
    # --- protonated amines -------------------------------------------------
    ("methylammonium",        "C[NH3+]",                       {"N+1": 1}),
    ("dimethylammonium",      "C[NH2+]C",                      {"N+1": 1}),
    ("trimethylammonium",     "C[NH+](C)C",                    {"N+1": 1}),
    ("tetramethylammonium",   "C[N+](C)(C)C",                  {"N+1": 1}),
    ("protonated_piperidine", "C1CC[NH2+]CC1",                 {"N+1": 1}),
    ("zinc_like_ammonium",    "CC(C)(C)c1ccc(C[NH+](C)C2CC2)cc1", {"N+1": 1}),
    # --- anions ------------------------------------------------------------
    ("acetate",               "CC(=O)[O-]",                    {"O-1": 1}),
    ("phenoxide",             "[O-]c1ccccc1",                  {"O-1": 1}),
    ("benzoate",              "[O-]C(=O)c1ccccc1",             {"O-1": 1}),
    ("thiolate",              "c1ccccc1[S-]",                  {"S-1": 1}),
    ("azide",                 "CN=[N+]=[N-]",                  {"N+1": 1, "N-1": 1}),
    ("amide_anion",           "CS(=O)(=O)[N-]c1ccccc1",        {"N-1": 1}),
    # --- zwitterions / net-neutral charge separation ---------------------------
    ("glycine_zwitterion",    "[NH3+]CC(=O)[O-]",              {"N+1": 1, "O-1": 1}),
    ("betaine",               "C[N+](C)(C)CC(=O)[O-]",         {"N+1": 1, "O-1": 1}),
    # --- nitro / N-oxide -------------------------------------------------------
    ("nitromethane",          "C[N+](=O)[O-]",                 {"N+1": 1, "O-1": 1}),
    ("nitrobenzene",          "[O-][N+](=O)c1ccccc1",          {"N+1": 1, "O-1": 1}),
    ("dinitro",               "Cc1ccc([N+](=O)[O-])cc1[N+](=O)[O-]", {"N+1": 2, "O-1": 2}),
    ("pyridine_n_oxide",      "[O-][n+]1ccccc1",               {"N+1": 1, "O-1": 1}),
    # --- charged aromatic structures -----------------------------------------------
    ("pyridinium",            "C[n+]1ccccc1",                  {"N+1": 1}),
    ("protonated_pyridine",   "c1cc[nH+]cc1",                  {"N+1": 1}),
    ("imidazolium",           "Cn1cc[nH+]c1",                  {"N+1": 1}),
    ("methylimidazolium",     "C[n+]1ccn(C)c1",                {"N+1": 1}),
    ("pyrylium",              "c1cc[o+]cc1",                   {"O+1": 1}),
    ("quinolinium",           "c1ccc2[nH+]cccc2c1",            {"N+1": 1}),
    # --- rare charge types ----------------------------------------------------------
    ("oxonium",               "C[O+](C)C",                     {"O+1": 1}),
    ("sulfonium",             "C[S+](C)C",                     {"S+1": 1}),
    ("isocyanide",            "[C-]#[N+]C",                    {"C-1": 1, "N+1": 1}),
    ("phosphonium",           "C[P+](C)(C)C",                  {"P+1": 1}),
]

NEUTRAL_MOLECULES = [
    "CCO", "c1ccccc1", "CC(=O)Nc1ccc(O)cc1", "c1cnc[nH]1", "Fc1ccccc1",
    "Clc1ccccc1", "Brc1ccccc1", "Ic1ccccc1", "c1ccsc1", "CP(C)C",
    "O=C(Nc1ccc(F)cc1)c1cc(Cl)ccc1Br", "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "C[C@H](N)C(=O)O",
]


def parse(smiles: str) -> Chem.Mol:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None, smiles
    return molecule


def type_counter(record) -> Counter:
    return Counter(
        v2.atom_type_label(symbol, charge)
        for symbol, charge in atom_type_pairs(record["atom_types"])
        if charge != 0
    )


# ===========================================================================
# Vocabulary and representation interface.
# ===========================================================================
def test_vocabulary_matches_phase0_measurements():
    assert v2.NUM_ATOM_TYPES == 17
    assert v2.ATOM_VOCAB == (
        "C", "N", "O", "F", "P", "S", "Cl", "Br", "I",
        "N+1", "O-1", "N-1", "S-1", "O+1", "S+1", "C-1", "P+1",
    )
    assert len(set(v2.ATOM_VOCAB)) == 17
    assert v2.NODE_FEATURE_DIM == 17
    assert v2.REPRESENTATION_VERSION_CHARGE_AWARE == "zinc250k_charge_aware_kekule_v2"


def test_neutral_indices_are_identical_to_v1():
    assert v2.ATOM_VOCAB[:9] == tuple(v1_constants.ATOM_VOCAB)
    assert v2.ATOM_VALENCE[:9] == tuple(v1_constants.ATOM_VALENCE)
    for index, symbol in enumerate(v1_constants.ATOM_VOCAB):
        assert v2.ATOM_TYPE_TO_INDEX[(symbol, 0)] == index
        assert v2.CHARGE_OF_TYPE[index] == 0


def test_baseline_v1_is_unchanged():
    assert v1_constants.REPRESENTATION_VERSION == "zinc250k_neutral_kekule_v1"
    assert len(v1_constants.ATOM_VOCAB) == 9
    assert v1_constants.NODE_FEATURE_DIM == 9
    record, reason = encode_zinc_smiles("C[NH3+]")
    assert record is None and reason == "non_neutral"      # still neutral-only
    record, _ = encode_zinc_smiles("CCO")
    assert record["representation"] == "zinc250k_neutral_kekule_v1"


def test_vocabulary_fingerprint_is_pinned():
    # Changing the vocabulary, order or caps must be a deliberate, versioned
    # change: update the version string and this pin together.
    assert v2.vocabulary_fingerprint() == "ef8f94033dbbc9d2"


def _max_valence_that_sanitizes(symbol: str, charge: int) -> int:
    best = -1
    for neighbours in range(0, 9):
        editable = Chem.RWMol()
        centre = editable.AddAtom(Chem.Atom(symbol))
        editable.GetAtomWithIdx(centre).SetFormalCharge(charge)
        for _ in range(neighbours):
            other = editable.AddAtom(Chem.Atom("C"))
            editable.AddBond(centre, other, Chem.BondType.SINGLE)
        try:
            with rdBase.BlockLogs():
                Chem.SanitizeMol(editable.GetMol())
            best = neighbours
        except Exception:  # noqa: BLE001
            pass
    return best


# v1 deliberately caps neutral iodine at 1 although RDKit accepts hypervalent
# iodine (up to 5). v2 keeps every neutral cap identical to v1 so neutral
# encodings are identical; the real ZINC250K run had 0 valence_overflow
# rejections, so no molecule is affected. This is the ONLY allowed difference.
CONSERVATIVE_NEUTRAL_CAPS = {"I": 1}


@pytest.mark.parametrize("index", range(v2.NUM_ATOM_TYPES))
def test_valence_caps_match_rdkit(index):
    symbol, charge = v2.ATOM_TYPES[index]
    cap = v2.ATOM_VALENCE[index]
    rdkit_max = _max_valence_that_sanitizes(symbol, charge)
    label = v2.ATOM_VOCAB[index]

    assert cap <= rdkit_max, f"{label}: cap {cap} exceeds RDKit's maximum {rdkit_max}"
    if charge == 0 and symbol in CONSERVATIVE_NEUTRAL_CAPS:
        assert cap == CONSERVATIVE_NEUTRAL_CAPS[symbol] < rdkit_max
    else:
        assert cap == rdkit_max, (
            f"valence cap for {label} disagrees with the installed RDKit"
        )


def test_representation_registry_is_interface_compatible():
    assert get_representation(NEUTRAL_V1.version) is NEUTRAL_V1
    assert get_representation(CHARGE_AWARE_V2.version) is CHARGE_AWARE_V2
    assert NEUTRAL_V1.node_feature_dim == 9 and not NEUTRAL_V1.charge_aware
    assert CHARGE_AWARE_V2.node_feature_dim == 17 and CHARGE_AWARE_V2.charge_aware
    assert NEUTRAL_V1.num_bond_types == CHARGE_AWARE_V2.num_bond_types == 4
    assert NEUTRAL_V1.max_nodes == CHARGE_AWARE_V2.max_nodes == 38
    assert len(CHARGE_AWARE_V2.valence) == len(CHARGE_AWARE_V2.atom_vocab)
    assert CHARGE_AWARE_V2.charges[v2.LABEL_TO_INDEX["O-1"]] == -1
    assert CHARGE_AWARE_V2.elements[v2.LABEL_TO_INDEX["N+1"]] == "N"
    with pytest.raises(KeyError, match="unknown representation"):
        get_representation("nope")


# ===========================================================================
# Exact round trip for charged molecules.
# ===========================================================================
@pytest.mark.parametrize("name,smiles,expected_types", CHARGED_MOLECULES)
def test_charged_molecule_roundtrips_exactly(name, smiles, expected_types):
    original = parse(smiles)
    record, reason = encode_zinc_smiles_charge_aware(smiles, row_id=7)
    assert record is not None, f"{name} rejected: {reason}"
    assert reason is None

    # Node count and element/charge identity are preserved.
    assert record["n_node"] == original.GetNumAtoms()
    assert record["atom_types"].shape == (original.GetNumAtoms(),)
    original_pairs = Counter(
        (a.GetSymbol(), a.GetFormalCharge()) for a in original.GetAtoms()
    )
    assert Counter(atom_type_pairs(record["atom_types"])) == original_pairs
    assert type_counter(record) == Counter(expected_types)

    # Net charge and the charged-atom count.
    assert record["net_charge"] == sum(a.GetFormalCharge() for a in original.GetAtoms())
    assert net_charge(record["atom_types"]) == record["net_charge"]
    assert record["n_charged_atoms"] == sum(expected_types.values())

    # Reconstruction reproduces the molecule exactly (identity = canonical
    # non-stereo SMILES, which includes every formal charge).
    rebuilt = graph_to_mol_charge_aware(record["atom_types"], record["bond_matrix"])
    assert canonical_connectivity_smiles(rebuilt) == record["smiles"]
    assert record["smiles"] == canonical_connectivity_smiles(original)
    assert sorted(a.GetFormalCharge() for a in rebuilt.GetAtoms()) == sorted(
        a.GetFormalCharge() for a in original.GetAtoms()
    )

    # Chemically valid implicit-hydrogen behaviour: same total hydrogens.
    assert Chem.AddHs(rebuilt).GetNumAtoms() == Chem.AddHs(original).GetNumAtoms()
    assert sorted(a.GetTotalNumHs() for a in rebuilt.GetAtoms()) == sorted(
        a.GetTotalNumHs() for a in original.GetAtoms()
    )

    # QED is a property of the reconstructed molecule and equals the QED of
    # the molecule as written in the source.
    assert record["qed"] == pytest.approx(QED.qed(rebuilt), abs=1e-12)
    assert record["qed"] == pytest.approx(QED.qed(original), abs=1e-9)

    assert record["row_id"] == 7
    assert record["original_smiles"] == smiles
    assert record["representation"] == v2.REPRESENTATION_VERSION_CHARGE_AWARE


@pytest.mark.parametrize("name,smiles,expected_types", CHARGED_MOLECULES)
def test_v1_baseline_rejects_what_v2_accepts(name, smiles, expected_types):
    record, reason = encode_zinc_smiles(smiles)
    assert record is None and reason == "non_neutral"


@pytest.mark.parametrize("smiles", NEUTRAL_MOLECULES)
def test_neutral_molecules_encode_identically_to_v1(smiles):
    old, _ = encode_zinc_smiles(smiles)
    new, _ = encode_zinc_smiles_charge_aware(smiles)
    assert old is not None and new is not None
    assert np.array_equal(old["atom_types"], new["atom_types"])
    assert np.array_equal(old["bond_matrix"], new["bond_matrix"])
    assert old["smiles"] == new["smiles"]
    assert old["qed"] == new["qed"]
    assert new["net_charge"] == 0 and new["n_charged_atoms"] == 0


def test_every_charge_type_in_the_vocabulary_is_exercised():
    seen = set()
    for _, smiles, _ in CHARGED_MOLECULES:
        record, _ = encode_zinc_smiles_charge_aware(smiles)
        seen |= set(type_counter(record))
    assert seen == set(v2.ATOM_VOCAB[9:])


def test_record_has_baseline_keys_plus_charge_fields():
    old, _ = encode_zinc_smiles("CCO")
    new, _ = encode_zinc_smiles_charge_aware("CCO", row_id=3)
    assert set(old) <= set(new)
    assert {"row_id", "net_charge", "n_charged_atoms"} <= set(new)
    for key in ("atom_types", "bond_matrix"):
        assert new[key].dtype == np.int64
    assert np.array_equal(new["bond_matrix"], new["bond_matrix"].T)
    assert np.all(np.diag(new["bond_matrix"]) == 0)
    assert new["atom_types"].max() < v2.NUM_ATOM_TYPES


def test_stereochemistry_stripped_and_charges_kept():
    record, _ = encode_zinc_smiles_charge_aware("C[C@H]([NH3+])C(=O)[O-]")
    assert "@" not in record["smiles"] and "/" not in record["smiles"]
    assert type_counter(record) == Counter({"N+1": 1, "O-1": 1})


def test_stereoisomers_share_one_identity():
    first, _ = encode_zinc_smiles_charge_aware("C[C@H]([NH3+])C(=O)[O-]")
    second, _ = encode_zinc_smiles_charge_aware("C[C@@H]([NH3+])C(=O)[O-]")
    assert first["smiles"] == second["smiles"]
    assert np.array_equal(first["atom_types"], second["atom_types"])


def test_encoding_is_deterministic():
    a, _ = encode_zinc_smiles_charge_aware("Cc1ccc([N+](=O)[O-])cc1[N+](=O)[O-]")
    b, _ = encode_zinc_smiles_charge_aware("Cc1ccc([N+](=O)[O-])cc1[N+](=O)[O-]")
    assert np.array_equal(a["atom_types"], b["atom_types"])
    assert np.array_equal(a["bond_matrix"], b["bond_matrix"])
    assert a["smiles"] == b["smiles"] and a["qed"] == b["qed"]


def test_different_smiles_for_one_molecule_give_one_graph():
    first, _ = encode_zinc_smiles_charge_aware("C[N+](=O)[O-]")
    second, _ = encode_zinc_smiles_charge_aware("[O-][N+](=O)C")
    assert first["smiles"] == second["smiles"]
    assert np.array_equal(first["atom_types"], second["atom_types"])
    assert np.array_equal(first["bond_matrix"], second["bond_matrix"])


# ===========================================================================
# Hydrogen inference from typed valence (reconstruction side).
# ===========================================================================
@pytest.mark.parametrize("label,bonds,hydrogens", [
    ("N+1", 1, 3), ("N+1", 2, 2), ("N+1", 3, 1), ("N+1", 4, 0),
    ("O-1", 1, 0), ("N-1", 2, 0), ("N-1", 1, 1),
    ("O+1", 3, 0), ("O+1", 2, 1), ("S+1", 3, 0), ("C-1", 3, 0),
    ("P+1", 4, 0), ("S-1", 1, 0),
])
def test_typed_valence_determines_implicit_hydrogens(label, bonds, hydrogens):
    atom_types = [v2.LABEL_TO_INDEX[label]] + [v2.LABEL_TO_INDEX["C"]] * bonds
    matrix = np.zeros((bonds + 1, bonds + 1), dtype=np.int64)
    for k in range(1, bonds + 1):
        matrix[0, k] = matrix[k, 0] = 1
    molecule = graph_to_mol_charge_aware(np.array(atom_types), matrix)
    centre = molecule.GetAtomWithIdx(0)
    assert centre.GetFormalCharge() == v2.CHARGE_OF_TYPE[v2.LABEL_TO_INDEX[label]]
    assert centre.GetTotalNumHs() == hydrogens


def test_formal_charge_helpers():
    atom_types = np.array([v2.LABEL_TO_INDEX[l] for l in ("N+1", "O-1", "C", "O-1")])
    assert formal_charges(atom_types).tolist() == [1, -1, 0, -1]
    assert net_charge(atom_types) == -1
    assert atom_type_pairs(atom_types)[0] == ("N", 1)


# ===========================================================================
# Rejections are explicit, from a closed set, and never silent.
# ===========================================================================
REJECTED = [
    ("empty", "", "empty_smiles"),
    ("junk", "not a smiles", "parse_failure"),
    ("fragments", "CC.[Na+].[Cl-]", "multiple_fragments"),
    ("isotope", "[13CH3]CO", "isotope"),
    ("radical", "C[CH2]", "radical"),
    ("silicon", "C[SiH3]", "unsupported_element"),
    ("sodium_ion_pair", "C[Na]", "unsupported_element"),
    ("chloronium", "C[Cl+]C", "unsupported_charge_type"),
    ("dication_carbon", "C[C+2]C", "unsupported_charge_type"),
    ("single_atom", "C", "node_count_below_min"),
    ("phosphine_oxide_PH2", "O=[PH2]C(Cc1ccccc1)c1ccccc1", "roundtrip_mismatch"),
]


@pytest.mark.parametrize("name,smiles,expected", REJECTED)
def test_rejection_reasons(name, smiles, expected):
    record, reason = encode_zinc_smiles_charge_aware(smiles)
    assert record is None
    assert reason == expected


def test_oversized_molecule_rejected():
    record, reason = encode_zinc_smiles_charge_aware("CCCCCC", max_nodes=5)
    assert record is None and reason == "node_count_above_max"
    record, reason = encode_zinc_smiles_charge_aware("CC", min_nodes=3)
    assert record is None and reason == "node_count_below_min"


def test_all_reasons_belong_to_the_declared_set():
    for _, smiles, _ in REJECTED:
        _, reason = encode_zinc_smiles_charge_aware(smiles)
        assert reason in REJECTION_REASONS
    assert len(REJECTION_REASONS) == len(set(REJECTION_REASONS))


def test_unsupported_charge_types_are_named_not_hidden():
    assert unsupported_charge_types("C[Cl+]C") == ["Cl+1"]
    assert unsupported_charge_types("C[C+2]C") == ["C+2"]
    assert unsupported_charge_types("C[NH3+]") == []
    assert unsupported_charge_types("not a smiles") == []
    assert unsupported_charge_types("") == []


# ===========================================================================
# graph_to_mol input validation.
# ===========================================================================
def test_graph_to_mol_validation():
    types = np.array([0, 0])
    good = np.array([[0, 1], [1, 0]])
    assert graph_to_mol_charge_aware(types, good).GetNumAtoms() == 2
    with pytest.raises(ValueError, match="empty"):
        graph_to_mol_charge_aware(np.array([], dtype=np.int64), np.zeros((0, 0)))
    with pytest.raises(ValueError, match="out of range"):
        graph_to_mol_charge_aware(np.array([0, 17]), good)
    with pytest.raises(ValueError, match="out of range"):
        graph_to_mol_charge_aware(np.array([0, -1]), good)
    with pytest.raises(ValueError, match="shape"):
        graph_to_mol_charge_aware(types, np.zeros((3, 3), dtype=np.int64))
    with pytest.raises(ValueError, match="symmetric"):
        graph_to_mol_charge_aware(types, np.array([[0, 1], [0, 0]]))
    with pytest.raises(ValueError, match="diagonal"):
        graph_to_mol_charge_aware(types, np.array([[1, 0], [0, 0]]))
    with pytest.raises(ValueError, match="bond class"):
        graph_to_mol_charge_aware(types, np.array([[0, 4], [4, 0]]))


def test_valence_violation_raises_on_reconstruction():
    # An [O-] with two bonds exceeds its typed valence of 1.
    types = np.array([v2.LABEL_TO_INDEX["O-1"], 0, 0])
    matrix = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    with pytest.raises(Exception):
        graph_to_mol_charge_aware(types, matrix)
