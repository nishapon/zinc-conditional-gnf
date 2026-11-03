"""Chemical profiling and population comparison (diagnostic S3).

Every expected value below is either hand-countable (atoms, bonds, rings) or a
fixed RDKit output verified against the library (QED of the specific molecules
used here), so a change in behaviour shows up as a failing number rather than a
plausible-looking one.
"""

from __future__ import annotations

import math

import pytest

from zinc_gnf.molecular_profile import (
    QED_THRESHOLDS,
    compare_profiles,
    format_comparison,
    profile_molecules,
)

# RDKit QED of these molecules (checked against the library):
#   c1ccccc1 0.4426   CCO 0.4068   CC(=O)Nc1ccccc1 0.6228   CCN(CC)CC 0.5184
#   CC(C)Cc1ccc(C(C)C(=O)O)cc1 0.8216   O=C(Nc1ccc(F)cc1)c1cc(Cl)ccc1Br 0.8642
DRUG_LIKE = ["CC(C)Cc1ccc(C(C)C(=O)O)cc1", "O=C(Nc1ccc(F)cc1)c1cc(Cl)ccc1Br"]
SIMPLE = ["c1ccccc1", "CCO"]


def test_counts_separate_invalid_unparsable_and_disconnected():
    profile = profile_molecules(["c1ccccc1", "CCO", None, "CC.O", "not a smiles"],
                                name="mixed")
    assert profile["name"] == "mixed"
    assert profile["submitted"] == 5
    assert profile["structurally_valid"] == 2
    assert profile["structural_validity"] == pytest.approx(2 / 5)
    assert profile["unparsable"] == 2            # the None and the unparsable string
    assert profile["disconnected"] == 1          # CC.O parses but is two fragments
    assert profile["unique_molecules"] == 2 and profile["uniqueness"] == pytest.approx(1.0)


def test_atom_bond_and_ring_statistics_are_hand_countable():
    # benzene: 6 aromatic C, 6 aromatic bonds, 1 ring; ethanol: 2 C + 1 O, 2 single bonds
    profile = profile_molecules(SIMPLE, name="simple")
    assert profile["atom_fractions"]["C"] == pytest.approx(8 / 9)
    assert profile["atom_fractions"]["O"] == pytest.approx(1 / 9)
    assert profile["bond_fractions"]["aromatic"] == pytest.approx(6 / 8)
    assert profile["bond_fractions"]["single"] == pytest.approx(2 / 8)
    assert profile["bonds_per_atom"] == pytest.approx(8 / 9)
    assert profile["rings"]["mean"] == pytest.approx(0.5)
    assert profile["aromatic_rings"]["mean"] == pytest.approx(0.5)
    assert profile["molecules_with_aromatic_ring"] == pytest.approx(0.5)
    assert profile["aromatic_atom_fraction"]["mean"] == pytest.approx(0.5)   # 1.0 and 0.0
    assert profile["heavy_atoms"]["mean"] == pytest.approx(4.5)
    assert profile["molecules_with_formal_charge"] == pytest.approx(0.0)


def test_aromatic_ring_fraction_separates_connectivity_from_bond_orders():
    """A ring that exists but is not aromatic is a bond-order failure, not a missing ring."""
    profile = profile_molecules(["c1ccccc1", "C1CCCCC1"], name="rings")
    assert profile["rings"]["mean"] == pytest.approx(1.0)          # both have a ring
    assert profile["molecules_with_ring"] == pytest.approx(1.0)
    assert profile["aromatic_rings"]["mean"] == pytest.approx(0.5)  # only benzene's is aromatic
    assert profile["aromatic_ring_fraction"] == pytest.approx(0.5)
    # benzene's six ring bonds are aromatic, cyclohexane's six are single
    assert profile["ring_bond_fractions"]["aromatic"] == pytest.approx(0.5)
    assert profile["ring_bond_fractions"]["single"] == pytest.approx(0.5)

    acyclic = profile_molecules(["CCO", "CCN(CC)CC"], name="acyclic")
    assert acyclic["rings"]["mean"] == pytest.approx(0.0)
    assert acyclic["molecules_with_ring"] == pytest.approx(0.0)
    assert math.isnan(acyclic["aromatic_ring_fraction"])           # no rings to divide by
    assert acyclic["ring_bond_fractions"] == {}


def test_comparison_reports_the_ring_and_aromatic_differences():
    reference = profile_molecules(["c1ccccc1", "CC(=O)Nc1ccccc1"], name="real_validation")
    rings_but_not_aromatic = profile_molecules(["C1CCCCC1", "C1CCCCC1C"], name="generated")
    comparison = compare_profiles([reference, rings_but_not_aromatic])
    difference = comparison["differences_vs_reference"]["generated"]
    assert difference["aromatic_rings_mean"] < 0                   # aromaticity lost
    assert difference["aromatic_ring_fraction"] == pytest.approx(-1.0)
    assert difference["rings_mean"] == pytest.approx(0.0)          # but the rings are still there


def test_qed_statistics_and_the_zero_point_seven_threshold():
    profile = profile_molecules(SIMPLE + DRUG_LIKE, name="four")
    assert profile["qed_molecules"] == 4
    assert profile["qed"]["mean"] == pytest.approx((0.4426 + 0.4068 + 0.8216 + 0.8642) / 4, abs=1e-3)
    assert profile["qed"]["median"] == pytest.approx((0.4426 + 0.8216) / 2, abs=1e-3)
    assert profile["qed"]["min"] == pytest.approx(0.4068, abs=1e-3)
    assert profile["qed"]["max"] == pytest.approx(0.8642, abs=1e-3)
    assert profile["qed_at_least_0p7"] == pytest.approx(0.5)      # the two drug-like ones
    assert profile["qed_at_least_0p5"] == pytest.approx(0.5)
    low = profile_molecules(SIMPLE, name="low")
    assert low["qed_at_least_0p7"] == pytest.approx(0.0)
    assert low["qed_at_least_0p5"] == pytest.approx(0.0)
    assert set(QED_THRESHOLDS) == {0.5, 0.7}


def test_charged_molecules_are_counted_and_typed():
    profile = profile_molecules(["CC(=O)[O-]", "C[NH3+]", "CCO"], name="charged")
    assert profile["structurally_valid"] == 3
    assert profile["molecules_with_formal_charge"] == pytest.approx(2 / 3)
    assert "O-1" in profile["charge_type_fractions"]
    assert "N+1" in profile["charge_type_fractions"]
    assert profile["charge_type_fractions"]["C"] > 0              # neutral carbons keep their symbol


def test_scaffold_diversity_is_reported():
    profile = profile_molecules(["c1ccccc1", "CC(=O)Nc1ccccc1", "CCO"], name="scaffolds")
    assert profile["scaffolds"]["molecules"] == 3
    assert profile["scaffolds"]["distinct"] == 2                  # benzene twice, empty once
    assert profile["scaffolds"]["per_molecule"] == pytest.approx(2 / 3)
    without = profile_molecules(["c1ccccc1"], name="off", scaffolds=False)
    assert "scaffolds" not in without


def test_an_empty_or_all_invalid_population_does_not_raise():
    empty = profile_molecules([], name="empty")
    assert empty["submitted"] == 0 and empty["structurally_valid"] == 0
    assert math.isnan(empty["structural_validity"]) and math.isnan(empty["qed"]["mean"])
    assert math.isnan(empty["qed_at_least_0p7"])
    invalid = profile_molecules([None, None], name="none")
    assert invalid["structural_validity"] == 0.0 and invalid["unparsable"] == 2
    assert math.isnan(invalid["qed"]["median"])


def test_comparison_differences_are_against_the_first_population():
    real = profile_molecules(DRUG_LIKE, name="real_validation")
    poor = profile_molecules(SIMPLE, name="generated")
    comparison = compare_profiles([real, poor])
    assert comparison["reference"] == "real_validation"
    assert comparison["populations"] == ["real_validation", "generated"]
    difference = comparison["differences_vs_reference"]["generated"]
    assert difference["qed_mean"] == pytest.approx(
        poor["qed"]["mean"] - real["qed"]["mean"], abs=1e-9)
    assert difference["qed_mean"] < -0.3                      # the gap this diagnostic exists to size
    assert difference["qed_at_least_0p7"] == pytest.approx(-1.0)
    assert difference["heavy_atoms_mean"] < 0
    assert "real_validation" not in comparison["differences_vs_reference"]
    with pytest.raises(ValueError):
        compare_profiles([])


def test_formatted_comparison_lists_the_requested_metrics():
    comparison = compare_profiles([profile_molecules(DRUG_LIKE, name="real_validation"),
                                   profile_molecules(SIMPLE, name="ae_reconstruction"),
                                   profile_molecules(SIMPLE, name="flow_generated")])
    text = format_comparison(comparison)
    for fragment in ("QED mean", "QED median", "QED >= 0.7", "rings mean", "aromatic atom frac",
                     "bond frac single", "bond frac aromatic", "atom frac C",
                     "structurally valid", "scaffolds/molecule"):
        assert fragment in text, fragment
    for name in ("real_validation", "ae_reconstruction", "flow_generated"):
        assert name in text
