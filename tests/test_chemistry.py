"""Tests for ZINC graph conversion and constrained decoding."""

import numpy as np
import pytest

from zinc_gnf.chemistry import (
    canonical_connectivity_smiles,
    constrained_bond_decode,
    decode_constrained_molecule,
    encode_zinc_smiles,
    graph_to_mol,
)
from zinc_gnf.constants import (
    ATOM_TO_INDEX,
    NUM_BOND_TYPES,
)


@pytest.mark.parametrize(
    "smiles",
    [
        "CCO",
        "c1ccccc1",
        "CC(=O)O",
        "C[NH3+]",
        "CCCl",
    ],
)
def test_encode_round_trip(smiles):
    record, reason = encode_zinc_smiles(smiles)

    assert reason is None
    assert record is not None

    molecule = graph_to_mol(
        record["atom_types"],
        record["charges"],
        record["hydrogen_counts"],
        record["bond_matrix"],
    )

    assert (
        canonical_connectivity_smiles(molecule)
        == record["smiles"]
    )
    assert 0.0 <= record["qed"] <= 1.0


def test_rejects_unsupported_element():
    record, reason = encode_zinc_smiles(
        "C[SiH2]C"
    )

    assert record is None
    assert reason == "unsupported_element"


def test_graph_to_mol_rejects_self_bond():
    with pytest.raises(
        ValueError,
        match="Self-bonds",
    ):
        graph_to_mol(
            [
                ATOM_TO_INDEX["C"],
                ATOM_TO_INDEX["C"],
            ],
            [0, 0],
            [3, 3],
            np.asarray(
                [
                    [1, 1],
                    [1, 0],
                ],
                dtype=np.int64,
            ),
        )


def test_constrained_decoder_ethane():
    atoms = np.asarray(
        [
            ATOM_TO_INDEX["C"],
            ATOM_TO_INDEX["C"],
        ],
        dtype=np.int64,
    )
    charges = np.asarray(
        [0, 0],
        dtype=np.int64,
    )

    probabilities = np.zeros(
        (2, 2, NUM_BOND_TYPES),
        dtype=np.float64,
    )
    probabilities[..., 0] = 1.0
    probabilities[0, 1] = [
        0.02,
        0.95,
        0.02,
        0.01,
    ]
    probabilities[1, 0] = [
        0.02,
        0.95,
        0.02,
        0.01,
    ]

    result = decode_constrained_molecule(
        atoms,
        charges,
        probabilities,
    )

    assert result["connected_by_decoder"]
    assert result["domain_valid"]
    assert result["smiles"] == "CC"


def test_constrained_decoder_limits_carbon_valence():
    n_node = 6

    atoms = np.full(
        n_node,
        ATOM_TO_INDEX["C"],
        dtype=np.int64,
    )
    charges = np.zeros(
        n_node,
        dtype=np.int64,
    )

    probabilities = np.zeros(
        (
            n_node,
            n_node,
            NUM_BOND_TYPES,
        ),
        dtype=np.float64,
    )
    probabilities[..., 0] = 0.01
    probabilities[..., 3] = 0.99

    bonds, information = constrained_bond_decode(
        atoms,
        charges,
        probabilities,
    )

    assert bonds is not None
    assert information["connected_by_decoder"]
    assert np.all(bonds.sum(axis=1) <= 4)
    assert np.array_equal(bonds, bonds.T)
    assert np.all(np.diag(bonds) == 0)
