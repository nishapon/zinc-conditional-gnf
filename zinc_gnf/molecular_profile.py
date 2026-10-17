"""Chemical profile of a set of molecules, for like-for-like comparison (diagnostic S3).

The same statistics are computed for three populations, so the question "where is the
molecular quality lost" has an answer rather than a guess:

  real            ZINC250K VALIDATION molecules (the reference; test is never read)
  reconstruction  the frozen autoencoder's reconstructions of those same molecules
  generated       molecules already sampled from the flow (read from samples.jsonl)

If `real` and `reconstruction` agree but `generated` does not, the loss is in the flow or
in the sampling; if `reconstruction` already disagrees with `real`, the autoencoder is
implicated. This module only measures - it draws no conclusion on its own.

Reported per population: molecule counts and structural validity, QED (mean, median,
quartiles, and the fraction at or above 0.7 and 0.5, which is the threshold the later
QED-conditioned model has to hit), ring counts, aromaticity, atom and formal-charge
composition, bond-order composition, heavy-atom counts and scaffold diversity.

`aromatic_ring_fraction` (aromatic rings / all rings) and `ring_bond_fractions` separate
the two ways aromaticity can disappear: the ring never formed (a connectivity failure),
or the ring formed but its bond orders are not an alternating kekule pattern (a
bond-order failure). Our representation has no aromatic atom type, so aromaticity is
perceived by RDKit from the kekule orders and these two failures need telling apart.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import QED
from rdkit.Chem.Scaffolds import MurckoScaffold

QED_THRESHOLDS = (0.5, 0.7)
BOND_NAMES = {Chem.BondType.SINGLE: "single", Chem.BondType.DOUBLE: "double",
              Chem.BondType.TRIPLE: "triple", Chem.BondType.AROMATIC: "aromatic"}


def _fractions(counter: Counter, total: int) -> dict[str, float]:
    return {key: counter[key] / total for key in sorted(counter, key=lambda k: -counter[k])} \
        if total else {}


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan"),
                "q25": float("nan"), "q75": float("nan"), "min": float("nan"), "max": float("nan")}
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "median": float(np.median(array)),
            "std": float(array.std()), "q25": float(np.quantile(array, 0.25)),
            "q75": float(np.quantile(array, 0.75)), "min": float(array.min()),
            "max": float(array.max())}


def profile_molecules(smiles: Iterable[str | None], *, name: str = "population",
                      scaffolds: bool = True) -> dict[str, Any]:
    """Chemical statistics of a SMILES population.

    `None` entries count as structurally invalid (a sample that produced no molecule),
    which is what keeps the generated population comparable to the real one: the
    denominators are stated explicitly in the result.
    """
    entries = list(smiles)
    submitted = len(entries)
    parsed = []
    unparsable = 0
    disconnected = 0
    for item in entries:
        if item is None:
            unparsable += 1
            continue
        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(item)
        if molecule is None:
            unparsable += 1
            continue
        if len(Chem.GetMolFrags(molecule)) != 1:
            disconnected += 1
            continue
        parsed.append((item, molecule))

    qed_values: list[float] = []
    rings: list[int] = []
    aromatic_rings: list[int] = []
    aromatic_fraction: list[float] = []
    heavy_atoms: list[int] = []
    atoms: Counter = Counter()
    charges: Counter = Counter()
    bonds: Counter = Counter()
    molecules_with_aromatic_ring = 0
    molecules_with_ring = 0
    molecules_with_charge = 0
    ring_bonds: Counter = Counter()
    scaffold_smiles: list[str] = []

    for item, molecule in parsed:
        try:
            qed_values.append(float(QED.qed(molecule)))
        except Exception:  # noqa: BLE001
            pass
        ring_info = molecule.GetRingInfo()
        rings.append(int(ring_info.NumRings()))
        aromatic_ring_count = sum(
            1 for ring in ring_info.AtomRings()
            if all(molecule.GetAtomWithIdx(i).GetIsAromatic() for i in ring))
        aromatic_rings.append(aromatic_ring_count)
        molecules_with_aromatic_ring += int(aromatic_ring_count > 0)
        molecules_with_ring += int(ring_info.NumRings() > 0)
        heavy = molecule.GetNumHeavyAtoms()
        heavy_atoms.append(int(heavy))
        aromatic_atoms = sum(1 for atom in molecule.GetAtoms() if atom.GetIsAromatic())
        aromatic_fraction.append(aromatic_atoms / heavy if heavy else 0.0)
        charged = 0
        for atom in molecule.GetAtoms():
            atoms[atom.GetSymbol()] += 1
            charge = int(atom.GetFormalCharge())
            charges[f"{atom.GetSymbol()}{charge:+d}" if charge else atom.GetSymbol()] += 1
            charged += int(charge != 0)
        molecules_with_charge += int(charged > 0)
        for bond in molecule.GetBonds():
            bond_name = BOND_NAMES.get(bond.GetBondType(), str(bond.GetBondType()))
            bonds[bond_name] += 1
            if bond.IsInRing():
                ring_bonds[bond_name] += 1
        if scaffolds:
            try:
                with rdBase.BlockLogs():
                    scaffold_smiles.append(MurckoScaffold.MurckoScaffoldSmiles(mol=molecule))
            except Exception:  # noqa: BLE001
                pass

    total_atoms = sum(atoms.values())
    total_bonds = sum(bonds.values())
    unique = {s for s, _ in parsed}
    profile: dict[str, Any] = {
        "name": name,
        "submitted": submitted,
        "structurally_valid": len(parsed),
        "structural_validity": (len(parsed) / submitted) if submitted else float("nan"),
        "unparsable": unparsable,
        "disconnected": disconnected,
        "unique_molecules": len(unique),
        "uniqueness": (len(unique) / len(parsed)) if parsed else float("nan"),
        "qed": _quantiles(qed_values),
        "qed_molecules": len(qed_values),
        "rings": _quantiles(rings),
        "aromatic_rings": _quantiles(aromatic_rings),
        "aromatic_atom_fraction": _quantiles(aromatic_fraction),
        "molecules_with_aromatic_ring": (molecules_with_aromatic_ring / len(parsed)) if parsed else float("nan"),
        "molecules_with_ring": (molecules_with_ring / len(parsed)) if parsed else float("nan"),
        # Separates the two ways aromaticity can be lost: no ring at all (connectivity),
        # or a ring whose bond orders do not form an alternating kekule pattern (orders).
        "aromatic_ring_fraction": (sum(aromatic_rings) / sum(rings)) if sum(rings) else float("nan"),
        "ring_bond_fractions": _fractions(ring_bonds, sum(ring_bonds.values())),
        "molecules_with_formal_charge": (molecules_with_charge / len(parsed)) if parsed else float("nan"),
        "heavy_atoms": _quantiles(heavy_atoms),
        "atom_fractions": _fractions(atoms, total_atoms),
        "charge_type_fractions": _fractions(charges, total_atoms),
        "bond_fractions": _fractions(bonds, total_bonds),
        "bonds_per_atom": (total_bonds / total_atoms) if total_atoms else float("nan"),
    }
    for threshold in QED_THRESHOLDS:
        above = sum(1 for value in qed_values if value >= threshold)
        profile[f"qed_at_least_{str(threshold).replace('.', 'p')}"] = (
            above / len(qed_values)) if qed_values else float("nan")
    if scaffolds and scaffold_smiles:
        distinct = len(set(scaffold_smiles))
        profile["scaffolds"] = {"molecules": len(scaffold_smiles), "distinct": distinct,
                                "per_molecule": distinct / len(scaffold_smiles)}
    return profile


def compare_profiles(profiles: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Side-by-side view plus differences against the FIRST profile (the reference)."""
    if not profiles:
        raise ValueError("need at least one profile")
    reference = profiles[0]
    comparison: dict[str, Any] = {"reference": reference["name"],
                                  "populations": [p["name"] for p in profiles],
                                  "profiles": {p["name"]: p for p in profiles},
                                  "differences_vs_reference": {}}
    keys = [("qed_mean", lambda p: p["qed"]["mean"]),
            ("qed_median", lambda p: p["qed"]["median"]),
            ("qed_at_least_0p7", lambda p: p["qed_at_least_0p7"]),
            ("rings_mean", lambda p: p["rings"]["mean"]),
            ("aromatic_rings_mean", lambda p: p["aromatic_rings"]["mean"]),
            ("aromatic_ring_fraction", lambda p: p["aromatic_ring_fraction"]),
            ("aromatic_atom_fraction_mean", lambda p: p["aromatic_atom_fraction"]["mean"]),
            ("heavy_atoms_mean", lambda p: p["heavy_atoms"]["mean"]),
            ("bonds_per_atom", lambda p: p["bonds_per_atom"]),
            ("structural_validity", lambda p: p["structural_validity"])]
    for profile in profiles[1:]:
        comparison["differences_vs_reference"][profile["name"]] = {
            label: float(getter(profile) - getter(reference)) for label, getter in keys}
    return comparison


def format_comparison(comparison: dict[str, Any]) -> str:
    profiles = [comparison["profiles"][name] for name in comparison["populations"]]
    width = max(18, *(len(p["name"]) + 2 for p in profiles))
    rows: list[tuple[str, list[str]]] = [
        ("molecules submitted", [f"{p['submitted']:,}" for p in profiles]),
        ("structurally valid", [f"{p['structural_validity']:.4f}" for p in profiles]),
        ("unique (of valid)", [f"{p['uniqueness']:.4f}" for p in profiles]),
        ("QED mean", [f"{p['qed']['mean']:.4f}" for p in profiles]),
        ("QED median", [f"{p['qed']['median']:.4f}" for p in profiles]),
        ("QED q25-q75", [f"{p['qed']['q25']:.2f}-{p['qed']['q75']:.2f}" for p in profiles]),
        ("QED >= 0.7", [f"{p['qed_at_least_0p7']:.4f}" for p in profiles]),
        ("QED >= 0.5", [f"{p['qed_at_least_0p5']:.4f}" for p in profiles]),
        ("rings mean", [f"{p['rings']['mean']:.3f}" for p in profiles]),
        ("aromatic rings mean", [f"{p['aromatic_rings']['mean']:.3f}" for p in profiles]),
        ("aromatic / all rings", [f"{p['aromatic_ring_fraction']:.4f}" for p in profiles]),
        ("with any ring", [f"{p['molecules_with_ring']:.4f}" for p in profiles]),
        ("aromatic atom frac", [f"{p['aromatic_atom_fraction']['mean']:.4f}" for p in profiles]),
        ("with aromatic ring", [f"{p['molecules_with_aromatic_ring']:.4f}" for p in profiles]),
        ("with formal charge", [f"{p['molecules_with_formal_charge']:.4f}" for p in profiles]),
        ("heavy atoms mean", [f"{p['heavy_atoms']['mean']:.2f}" for p in profiles]),
        ("bonds per atom", [f"{p['bonds_per_atom']:.4f}" for p in profiles]),
    ]
    for key in ("single", "double", "triple", "aromatic"):
        rows.append((f"bond frac {key}",
                     [f"{p['bond_fractions'].get(key, 0.0):.4f}" for p in profiles]))
    for key in ("single", "double", "aromatic"):
        rows.append((f"RING bond frac {key}",
                     [f"{p['ring_bond_fractions'].get(key, 0.0):.4f}" for p in profiles]))
    for element in ("C", "N", "O", "S", "F", "Cl"):
        rows.append((f"atom frac {element}",
                     [f"{p['atom_fractions'].get(element, 0.0):.4f}" for p in profiles]))
    rows.append(("scaffolds/molecule",
                 [f"{p.get('scaffolds', {}).get('per_molecule', float('nan')):.4f}" for p in profiles]))

    header = f"{'metric':<24}" + "".join(f"{p['name']:>{width}}" for p in profiles)
    lines = ["Molecular profile comparison (reference: " + comparison["reference"] + ")",
             header, "-" * len(header)]
    for label, values in rows:
        lines.append(f"{label:<24}" + "".join(f"{value:>{width}}" for value in values))
    return "\n".join(lines)


__all__ = [
    "QED_THRESHOLDS",
    "compare_profiles",
    "format_comparison",
    "profile_molecules",
]
