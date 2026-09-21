"""Shared molecular representation constants for ZINC."""

from rdkit import Chem


REPRESENTATION_VERSION = "connectivity_charge_totalH_kekule_v1"

ATOM_VOCAB = ("C", "N", "O", "F", "P", "S", "Cl", "Br", "I")
ATOM_TO_INDEX = {
    symbol: index
    for index, symbol in enumerate(ATOM_VOCAB)
}

CHARGE_VALUES = (-1, 0, 1)
CHARGE_TO_INDEX = {
    charge: index
    for index, charge in enumerate(CHARGE_VALUES)
}

HYDROGEN_VALUES = (0, 1, 2, 3, 4)
HYDROGEN_TO_INDEX = {
    count: index
    for index, count in enumerate(HYDROGEN_VALUES)
}

BOND_TO_INDEX = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
}
INDEX_TO_BOND = {
    index: bond_type
    for bond_type, index in BOND_TO_INDEX.items()
}

NUM_ATOM_TYPES = len(ATOM_VOCAB)
NUM_CHARGE_TYPES = len(CHARGE_VALUES)
NUM_HYDROGEN_TYPES = len(HYDROGEN_VALUES)
NUM_BOND_TYPES = 1 + len(BOND_TO_INDEX)

NODE_FEATURE_DIM = (
    NUM_ATOM_TYPES
    + NUM_CHARGE_TYPES
    + NUM_HYDROGEN_TYPES
)

MIN_NODES = 2
MAX_NODES = 38

VALENCE_LIMITS = {
    ("C", -1): 3, ("C", 0): 4, ("C", 1): 3,
    ("N", -1): 2, ("N", 0): 3, ("N", 1): 4,
    ("O", -1): 1, ("O", 0): 2, ("O", 1): 3,
    ("F", -1): 0, ("F", 0): 1, ("F", 1): 2,
    ("P", -1): 4, ("P", 0): 5, ("P", 1): 4,
    ("S", -1): 5, ("S", 0): 6, ("S", 1): 5,
    ("Cl", -1): 0, ("Cl", 0): 1, ("Cl", 1): 2,
    ("Br", -1): 0, ("Br", 0): 1, ("Br", 1): 2,
    ("I", -1): 0, ("I", 0): 1, ("I", 1): 2,
}
