"""RDKit conversion and chemistry-aware decoding for ZINC."""

from dataclasses import dataclass

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, QED

from zinc_gnf.constants import (
    ATOM_TO_INDEX,
    ATOM_VOCAB,
    BOND_TO_INDEX,
    CHARGE_VALUES,
    HYDROGEN_VALUES,
    INDEX_TO_BOND,
    MAX_NODES,
    MIN_NODES,
    NUM_BOND_TYPES,
    REPRESENTATION_VERSION,
    VALENCE_LIMITS,
)


def canonical_connectivity_smiles(molecule):
    """Canonical molecular identity without stereochemistry."""
    molecule = Chem.Mol(molecule)
    Chem.RemoveStereochemistry(molecule)

    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)

    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=False,
    )


def validate_graph_arrays(
    atom_types,
    charges,
    bond_matrix,
    hydrogen_counts=None,
):
    atom_types = np.asarray(atom_types, dtype=np.int64)
    charges = np.asarray(charges, dtype=np.int64)
    bonds = np.asarray(bond_matrix, dtype=np.int64)

    hydrogens = (
        None
        if hydrogen_counts is None
        else np.asarray(hydrogen_counts, dtype=np.int64)
    )

    n_node = len(atom_types)

    if charges.shape != (n_node,):
        raise ValueError("Atom and charge lengths differ.")

    if hydrogens is not None and hydrogens.shape != (n_node,):
        raise ValueError(
            "Atom and hydrogen-count lengths differ."
        )

    if bonds.shape != (n_node, n_node):
        raise ValueError(
            f"Expected bond matrix {(n_node, n_node)}, "
            f"received {bonds.shape}."
        )

    if np.any(
        (atom_types < 0)
        | (atom_types >= len(ATOM_VOCAB))
    ):
        raise ValueError("Unsupported atom-type index.")

    if not np.isin(charges, CHARGE_VALUES).all():
        raise ValueError("Unsupported formal charge.")

    if (
        hydrogens is not None
        and not np.isin(
            hydrogens,
            HYDROGEN_VALUES,
        ).all()
    ):
        raise ValueError("Unsupported hydrogen count.")

    if not np.array_equal(bonds, bonds.T):
        raise ValueError("Bond matrix must be symmetric.")

    if np.any(np.diag(bonds) != 0):
        raise ValueError("Self-bonds are not allowed.")

    if np.any(
        (bonds < 0)
        | (bonds >= NUM_BOND_TYPES)
    ):
        raise ValueError("Unsupported bond class.")

    return atom_types, charges, bonds, hydrogens


def graph_to_mol(
    atom_types,
    charges,
    hydrogen_counts,
    bond_matrix,
):
    """Exact reconstruction using stored hydrogen counts."""
    atom_types, charges, bonds, hydrogens = (
        validate_graph_arrays(
            atom_types,
            charges,
            bond_matrix,
            hydrogen_counts,
        )
    )

    editable = Chem.RWMol()

    for atom_index, charge, hydrogen_count in zip(
        atom_types,
        charges,
        hydrogens,
    ):
        atom = Chem.Atom(
            ATOM_VOCAB[int(atom_index)]
        )
        atom.SetFormalCharge(int(charge))
        atom.SetNumExplicitHs(int(hydrogen_count))
        atom.SetNoImplicit(True)
        editable.AddAtom(atom)

    for i in range(len(atom_types)):
        for j in range(i + 1, len(atom_types)):
            order = int(bonds[i, j])

            if order:
                editable.AddBond(
                    i,
                    j,
                    INDEX_TO_BOND[order],
                )

    molecule = editable.GetMol()
    Chem.SanitizeMol(molecule)
    return molecule


def graph_to_mol_with_implicit_hydrogens(
    atom_types,
    charges,
    bond_matrix,
):
    """Generated-graph conversion with RDKit-inferred hydrogens."""
    atom_types, charges, bonds, _ = (
        validate_graph_arrays(
            atom_types,
            charges,
            bond_matrix,
        )
    )

    editable = Chem.RWMol()

    for atom_index, charge in zip(
        atom_types,
        charges,
    ):
        atom = Chem.Atom(
            ATOM_VOCAB[int(atom_index)]
        )
        atom.SetFormalCharge(int(charge))
        atom.SetNoImplicit(False)
        editable.AddAtom(atom)

    for i in range(len(atom_types)):
        for j in range(i + 1, len(atom_types)):
            order = int(bonds[i, j])

            if order:
                editable.AddBond(
                    i,
                    j,
                    INDEX_TO_BOND[order],
                )

    molecule = editable.GetMol()
    Chem.SanitizeMol(molecule)
    return molecule


def encode_zinc_smiles(
    smiles,
    min_nodes=MIN_NODES,
    max_nodes=MAX_NODES,
):
    """Return (encoded_record, rejection_reason)."""
    molecule = Chem.MolFromSmiles(str(smiles))

    if molecule is None:
        return None, "parse_failure"

    if len(Chem.GetMolFrags(molecule)) != 1:
        return None, "multiple_fragments"

    if any(
        atom.GetIsotope() != 0
        for atom in molecule.GetAtoms()
    ):
        return None, "isotope"

    if any(
        atom.GetNumRadicalElectrons() != 0
        for atom in molecule.GetAtoms()
    ):
        return None, "radical"

    canonical = canonical_connectivity_smiles(
        molecule
    )
    molecule = Chem.MolFromSmiles(canonical)

    if molecule is None:
        return None, "canonical_parse_failure"

    n_node = molecule.GetNumAtoms()

    if not min_nodes <= n_node <= max_nodes:
        return None, "node_count"

    symbols = [
        atom.GetSymbol()
        for atom in molecule.GetAtoms()
    ]
    charges = [
        atom.GetFormalCharge()
        for atom in molecule.GetAtoms()
    ]
    hydrogen_counts = [
        int(atom.GetTotalNumHs())
        for atom in molecule.GetAtoms()
    ]

    if any(
        symbol not in ATOM_TO_INDEX
        for symbol in symbols
    ):
        return None, "unsupported_element"

    if any(
        charge not in CHARGE_VALUES
        for charge in charges
    ):
        return None, "unsupported_charge"

    if any(
        count not in HYDROGEN_VALUES
        for count in hydrogen_counts
    ):
        return None, "unsupported_hydrogen_count"

    qed_value = float(QED.qed(molecule))
    logp_value = float(Crippen.MolLogP(molecule))

    kekule = Chem.Mol(molecule)

    try:
        Chem.Kekulize(
            kekule,
            clearAromaticFlags=True,
        )
    except Exception:
        return None, "kekulization_failure"

    bonds = np.zeros(
        (n_node, n_node),
        dtype=np.int64,
    )

    for bond in kekule.GetBonds():
        order = BOND_TO_INDEX.get(
            bond.GetBondType()
        )

        if order is None:
            return None, "unsupported_bond"

        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bonds[i, j] = order
        bonds[j, i] = order

    atom_types = np.asarray(
        [
            ATOM_TO_INDEX[symbol]
            for symbol in symbols
        ],
        dtype=np.int64,
    )
    charges = np.asarray(
        charges,
        dtype=np.int64,
    )
    hydrogen_counts = np.asarray(
        hydrogen_counts,
        dtype=np.int64,
    )

    try:
        reconstructed = graph_to_mol(
            atom_types,
            charges,
            hydrogen_counts,
            bonds,
        )
        reconstructed_smiles = (
            canonical_connectivity_smiles(
                reconstructed
            )
        )
    except Exception:
        return None, "reconstruction_failure"

    if reconstructed_smiles != canonical:
        return None, "reconstruction_mismatch"

    return {
        "representation": REPRESENTATION_VERSION,
        "original_smiles": str(smiles),
        "smiles": canonical,
        "atom_types": atom_types,
        "charges": charges,
        "hydrogen_counts": hydrogen_counts,
        "bond_matrix": bonds,
        "n_node": n_node,
        "qed": qed_value,
        "logp": logp_value,
    }, None


def atom_valence_limit(symbol, charge):
    key = (symbol, int(charge))

    if key not in VALENCE_LIMITS:
        raise ValueError(
            f"No valence rule for "
            f"element={symbol}, charge={charge}."
        )

    return VALENCE_LIMITS[key]


@dataclass
class UnionFind:
    """Disjoint-set structure used to enforce connectivity."""

    size: int

    def __post_init__(self):
        self.parent = list(range(self.size))
        self.rank = [0] * self.size

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[
                self.parent[item]
            ]
            item = self.parent[item]

        return item

    def union(self, left, right):
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

    def connected(self):
        if self.size <= 1:
            return True

        roots = {
            self.find(index)
            for index in range(self.size)
        }
        return len(roots) == 1


def constrained_bond_decode(
    atom_types,
    charges,
    bond_probabilities,
):
    """Construct a connected, valence-limited bond matrix.

    Candidate bonds come only from decoder probabilities.
    Ground-truth bonds are never used.
    """
    atom_types = np.asarray(
        atom_types,
        dtype=np.int64,
    )
    charges = np.asarray(
        charges,
        dtype=np.int64,
    )
    probabilities = np.asarray(
        bond_probabilities,
        dtype=np.float64,
    )

    n_node = len(atom_types)

    if charges.shape != (n_node,):
        raise ValueError(
            "Atom and charge lengths differ."
        )

    expected_shape = (
        n_node,
        n_node,
        NUM_BOND_TYPES,
    )

    if probabilities.shape != expected_shape:
        raise ValueError(
            f"Expected bond probabilities {expected_shape}, "
            f"received {probabilities.shape}."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Bond probabilities contain non-finite values."
        )

    # Molecular bonds are undirected.
    probabilities = 0.5 * (
        probabilities
        + probabilities.transpose(1, 0, 2)
    )

    symbols = [
        ATOM_VOCAB[int(index)]
        for index in atom_types
    ]

    capacities = np.asarray(
        [
            atom_valence_limit(symbol, charge)
            for symbol, charge in zip(
                symbols,
                charges,
            )
        ],
        dtype=np.int64,
    )

    used_valence = np.zeros(
        n_node,
        dtype=np.int64,
    )
    bonds = np.zeros(
        (n_node, n_node),
        dtype=np.int64,
    )

    candidates = []

    for i in range(n_node):
        for j in range(i + 1, n_node):
            best_order = int(
                np.argmax(
                    probabilities[i, j, 1:]
                )
                + 1
            )

            real_probability = float(
                probabilities[i, j, best_order]
            )
            no_bond_probability = float(
                probabilities[i, j, 0]
            )

            log_odds = float(
                np.log(real_probability + 1e-12)
                - np.log(no_bond_probability + 1e-12)
            )

            candidates.append({
                "i": i,
                "j": j,
                "best_order": best_order,
                "real_probability": real_probability,
                "log_odds": log_odds,
            })

    candidates.sort(
        key=lambda candidate: (
            candidate["log_odds"],
            candidate["real_probability"],
        ),
        reverse=True,
    )

    components = UnionFind(n_node)
    tree_edges = set()

    # Pass 1: create a connected single-bond spanning structure.
    for candidate in candidates:
        i = candidate["i"]
        j = candidate["j"]

        if components.find(i) == components.find(j):
            continue

        if used_valence[i] >= capacities[i]:
            continue

        if used_valence[j] >= capacities[j]:
            continue

        bonds[i, j] = 1
        bonds[j, i] = 1

        used_valence[i] += 1
        used_valence[j] += 1

        components.union(i, j)
        tree_edges.add((i, j))

        if components.connected():
            break

    if not components.connected():
        return None, {
            "connected_by_decoder": False,
            "tree_edges": len(tree_edges),
            "extra_edges": 0,
            "upgraded_edges": 0,
            "reason": (
                "insufficient_valence_for_connectivity"
            ),
        }

    candidate_lookup = {
        (
            candidate["i"],
            candidate["j"],
        ): candidate
        for candidate in candidates
    }

    # Pass 2: upgrade tree bonds when supported by the model.
    upgraded_edges = 0

    for i, j in tree_edges:
        candidate = candidate_lookup[(i, j)]

        if candidate["log_odds"] <= 0:
            continue

        requested_addition = (
            candidate["best_order"] - 1
        )

        allowed_addition = min(
            requested_addition,
            capacities[i] - used_valence[i],
            capacities[j] - used_valence[j],
        )

        if allowed_addition > 0:
            new_order = 1 + allowed_addition

            bonds[i, j] = new_order
            bonds[j, i] = new_order

            used_valence[i] += allowed_addition
            used_valence[j] += allowed_addition
            upgraded_edges += 1

    # Pass 3: add additional model-supported bonds.
    extra_edges = 0

    for candidate in candidates:
        i = candidate["i"]
        j = candidate["j"]

        if bonds[i, j] != 0:
            continue

        if candidate["log_odds"] <= 0:
            continue

        remaining = min(
            capacities[i] - used_valence[i],
            capacities[j] - used_valence[j],
        )

        if remaining < 1:
            continue

        order = min(
            candidate["best_order"],
            remaining,
        )

        bonds[i, j] = order
        bonds[j, i] = order

        used_valence[i] += order
        used_valence[j] += order
        extra_edges += 1

    return bonds, {
        "connected_by_decoder": True,
        "tree_edges": len(tree_edges),
        "extra_edges": extra_edges,
        "upgraded_edges": upgraded_edges,
        "reason": "",
    }


def inspect_raw_molecule(
    atom_types,
    charges,
    hydrogen_counts,
    bond_matrix,
    original_smiles=None,
):
    """Inspect direct decoder output without repairing predictions."""
    result = {
        "sanitizable": False,
        "connected": False,
        "radical_free": False,
        "domain_valid": False,
        "smiles": None,
        "exact_molecule": False,
        "error": "",
    }

    with rdBase.BlockLogs():
        try:
            molecule = graph_to_mol(
                atom_types,
                charges,
                hydrogen_counts,
                bond_matrix,
            )

            smiles = canonical_connectivity_smiles(
                molecule
            )
            connected = (
                len(Chem.GetMolFrags(molecule)) == 1
            )
            radical_free = all(
                atom.GetNumRadicalElectrons() == 0
                for atom in molecule.GetAtoms()
            )
            domain_valid = connected and radical_free

            result.update({
                "sanitizable": True,
                "connected": connected,
                "radical_free": radical_free,
                "domain_valid": domain_valid,
                "smiles": smiles,
                "exact_molecule": bool(
                    domain_valid
                    and original_smiles is not None
                    and smiles == original_smiles
                ),
            })

        except Exception as error:
            result["error"] = (
                f"{type(error).__name__}: {error}"
            )

    return result


def decode_constrained_molecule(
    atom_types,
    charges,
    bond_probabilities,
    original_smiles=None,
):
    """Decode and inspect one chemistry-constrained sample."""
    bonds, decoder_information = (
        constrained_bond_decode(
            atom_types,
            charges,
            bond_probabilities,
        )
    )

    result = {
        **decoder_information,
        "sanitizable": False,
        "connected": False,
        "radical_free": False,
        "domain_valid": False,
        "smiles": None,
        "exact_molecule": False,
        "error": "",
    }

    if bonds is None:
        return result

    with rdBase.BlockLogs():
        try:
            molecule = (
                graph_to_mol_with_implicit_hydrogens(
                    atom_types,
                    charges,
                    bonds,
                )
            )

            smiles = canonical_connectivity_smiles(
                molecule
            )
            connected = (
                len(Chem.GetMolFrags(molecule)) == 1
            )
            radical_free = all(
                atom.GetNumRadicalElectrons() == 0
                for atom in molecule.GetAtoms()
            )
            domain_valid = connected and radical_free

            result.update({
                "sanitizable": True,
                "connected": connected,
                "radical_free": radical_free,
                "domain_valid": domain_valid,
                "smiles": smiles,
                "exact_molecule": bool(
                    domain_valid
                    and original_smiles is not None
                    and smiles == original_smiles
                ),
            })

        except Exception as error:
            result["error"] = (
                f"{type(error).__name__}: {error}"
            )

    return result
