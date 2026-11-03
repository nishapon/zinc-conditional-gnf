"""Tests for the reconstruction-failure analysis and the experiment hooks. CPU only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.test_autoencoder_model import SMILES, make_records
from zinc_gnf.autoencoder import AutoencoderConfig, GraphAutoencoder, evaluate_reconstruction
from zinc_gnf.chemistry_charge_aware import encode_zinc_smiles_charge_aware
from zinc_gnf.dataset import MolecularGraphDataset, make_batch
from zinc_gnf.reconstruction_analysis import (
    MOTIFS,
    analyze_reconstructions,
    aromatic_bucket,
    bond_metric_diagnostics,
    ceiling_assumption_report,
    charge_bucket,
    classify_failure,
    describe_molecule,
    embedding_class_check,
    embedding_robustness,
    embedding_statistics,
    mixed_class_pairs,
    motif_census,
    noise_draw_spread,
    rdkit_symmetry_classes,
    ring_bucket,
    size_bucket,
    summarize_failures,
    symmetry_bucket,
    symmetry_source,
    symmetry_source_census,
    uniform_graph_rescue,
    wilson_interval,
)

TINY = dict(node_embedding_dim=16, latent_dim=32, num_processing_steps=2,
            attn_kq_dim=8, attn_v_dim=8, attn_concat_heads_output_dim=8)
RING_SMILES = ["C1CCNCC1", "c1ccccc1", "Fc1ccc(F)cc1", "CCN(CC)CC", "O1CCNCC1", "CCO", "C[NH3+]",
               "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "c1cc[nH+]cc1", "CC(=O)[O-]", "N1CCNCC1", "CC(C)(C)c1ccccc1"]


@pytest.fixture(scope="module")
def ring_dataset():
    return MolecularGraphDataset(make_records(RING_SMILES))


def graph(smiles):
    record, _ = encode_zinc_smiles_charge_aware(smiles)
    return record["smiles"], record["atom_types"], record["bond_matrix"]


# ===========================================================================
# Symmetry sources and the identity-level check.
# ===========================================================================
@pytest.mark.parametrize("smiles,expected", [
    ("CCO", "none"),
    ("c1ccccc1C", "none"),                   # mono-substituted phenyl: orbit-mates never bonded to each other
    ("CC(C)(C)c1ccccc1", "none"),            # equivalent leaves on one centre are not ambiguous
    ("C1CCNCC1", "ring_positions"),          # piperidine
    ("N1CCNCC1", "ring_positions"),          # piperazine
    ("O1CCNCC1", "ring_positions"),          # morpholine
    ("c1ccccc1", "ring_positions"),          # benzene
    ("Fc1ccc(F)cc1", "ring_positions"),      # para-disubstituted
    ("CCN(CC)CC", "acyclic_chains"),         # three equivalent ethyl groups
])
def test_symmetry_source(smiles, expected):
    canonical, atoms, bonds = graph(smiles)
    assert symmetry_source(canonical, atoms, bonds) == expected


def test_mixed_class_pairs_empty_exactly_when_decodable():
    for smiles in RING_SMILES:
        canonical, atoms, bonds = graph(smiles)
        _, mixed = mixed_class_pairs(atoms, bonds)
        from zinc_gnf.symmetry import is_decodable
        assert (not mixed) == is_decodable(atoms, bonds), smiles


def test_symmetry_census_over_a_dataset(ring_dataset):
    census = symmetry_source_census(ring_dataset)
    assert census["molecules"] == len(RING_SMILES)
    assert census["decodable_fraction"] + census["unreachable_fraction"] == pytest.approx(1.0)
    assert census["unreachable_by_source"]["ring_positions"] >= 5
    assert census["unreachable_by_source"].get("acyclic_chains", 0) == 1
    assert sum(census["share_of_unreachable"].values()) == pytest.approx(1.0)


def test_identity_level_rescue_cannot_fix_symmetric_rings():
    """A colour-pair-uniform graph can never equal piperidine or benzene as a molecule."""
    for smiles in ("C1CCNCC1", "c1ccccc1", "Fc1ccc(F)cc1", "O1CCNCC1"):
        canonical, atoms, bonds = graph(smiles)
        assert uniform_graph_rescue(atoms, bonds, canonical) == "not_rescued", smiles


def test_identity_level_rescue_is_not_applicable_to_decodable_molecules():
    for smiles in ("CCO", "O=C(Nc1ccc(F)cc1)c1cc(Cl)ccc1Br"):
        canonical, atoms, bonds = graph(smiles)
        assert uniform_graph_rescue(atoms, bonds, canonical) == "not_applicable"


def test_identity_level_rescue_search_limit():
    canonical, atoms, bonds = graph("Fc1ccc(F)cc1")
    assert uniform_graph_rescue(atoms, bonds, canonical, max_mixed_classes=0) == "undetermined"
    assert uniform_graph_rescue(atoms, bonds, canonical) == "not_rescued"


def test_wl_vs_rdkit_symmetry_classes():
    from zinc_gnf.symmetry import refine_colors
    # Non-aromatic molecules: 1-WL with bond orders and RDKit's classes coincide.
    for smiles in ("C1CCNCC1", "CCN(CC)CC", "O1CCNCC1"):
        canonical, atoms, bonds = graph(smiles)
        reference = rdkit_symmetry_classes(canonical, len(atoms))
        assert len(set(refine_colors(atoms, bonds))) == len(set(reference)), smiles
    # Aromatic rings: 1-WL sees the (arbitrary) Kekule bond-order pattern that RDKit's
    # aromatic perception ignores, so it can have EXTRA classes. This is conservative for the
    # ceiling only if the Kekule pattern is a fixed function of the graph, which it is here.
    canonical, atoms, bonds = graph("Fc1ccc(F)cc1")
    assert len(set(refine_colors(atoms, bonds))) >= len(set(rdkit_symmetry_classes(canonical, len(atoms))))
    assert rdkit_symmetry_classes("not a smiles", 3) is None
    assert rdkit_symmetry_classes("CCO", 7) is None


def test_ceiling_assumption_report_fields(ring_dataset):
    report = ceiling_assumption_report(ring_dataset, rescue_sample=50)
    assert report["molecules"] == len(RING_SMILES)
    low, high = report["ceiling_ci95"]
    assert low <= report["ceiling"] <= high
    assert 0.0 <= report["wl_partition_equals_rdkit_symmetry_classes"] <= 1.0
    assert report["identity_level_rescue"].get("rescued", 0) == 0
    assert report["identity_level_rescue"].get("not_applicable", 0) == 0
    assert report["identity_level_rescue_tested"] == report["non_decodable_molecules"]
    assert len(report["assumptions"]) == 5 and "unreachable_by_source" in report


def test_wilson_interval():
    low, high = wilson_interval(50, 100)
    assert low < 0.5 < high and 0.39 < low < 0.41 and 0.59 < high < 0.61
    assert wilson_interval(0, 10)[0] == 0.0 and wilson_interval(10, 10)[1] == 1.0
    assert np.isnan(wilson_interval(0, 0)[0])


# ===========================================================================
# Descriptors, strata and failure taxonomy.
# ===========================================================================
def test_describe_molecule():
    canonical, atoms, bonds = graph("c1ccc2[nH+]cccc2c1")
    d = describe_molecule(canonical, atoms, bonds)
    assert d["n_node"] == 10 and d["n_rings"] == 2 and d["fused_rings"] is True
    assert d["aromatic_fraction"] == 1.0 and d["n_charged_atoms"] == 1 and d["net_charge"] == 1
    assert d["largest_ring"] == 6
    acyclic = describe_molecule(*graph("CCO")[:1], *graph("CCO")[1:])
    assert acyclic["n_rings"] == 0 and acyclic["aromatic_fraction"] == 0.0 and acyclic["fused_rings"] is False


def test_buckets():
    assert size_bucket(10) == "1-14" and size_bucket(20) == "20-24" and size_bucket(38) == "35-38"
    assert ring_bucket(0) == "0" and ring_bucket(3) == "3" and ring_bucket(7) == "4+"
    assert aromatic_bucket(0) == "none" and aromatic_bucket(0.4) == "partial (<=50%)" \
        and aromatic_bucket(0.9) == "mostly (>50%)"
    assert charge_bucket({"n_charged_atoms": 0, "net_charge": 0}) == "neutral"
    assert charge_bucket({"n_charged_atoms": 2, "net_charge": 0}) == "charged, net neutral"
    assert charge_bucket({"n_charged_atoms": 1, "net_charge": 1}) == "charged, net ion"
    assert symmetry_bucket({"decodable": True, "symmetric_atoms": 0}) == "decodable"
    assert symmetry_bucket({"decodable": True, "symmetric_atoms": 4}) == "symmetric but decodable"
    assert symmetry_bucket({"decodable": False, "symmetric_atoms": 4}) == "symmetry-unreachable"


@pytest.mark.parametrize("row,expected", [
    ({"constrained_exact": True}, "exact"),
    ({"constrained_exact": False, "decodable": False}, "symmetry_unreachable"),
    ({"constrained_exact": False, "decodable": True, "atoms_exact": False}, "decodable_atom_error"),
    ({"constrained_exact": False, "decodable": True, "atoms_exact": True, "constrained_valid": False},
     "decodable_invalid_assembly"),
    ({"constrained_exact": False, "decodable": True, "atoms_exact": True, "constrained_valid": True,
      "constrained_wrong_order": 2, "constrained_false_bonds": 0, "constrained_missed_bonds": 0},
     "decodable_wrong_bond_order_only"),
    ({"constrained_exact": False, "decodable": True, "atoms_exact": True, "constrained_valid": True,
      "constrained_wrong_order": 0, "constrained_false_bonds": 1, "constrained_missed_bonds": 0},
     "decodable_wrong_connectivity"),
])
def test_failure_taxonomy(row, expected):
    assert classify_failure(row) == expected


def test_analysis_on_a_model_has_consistent_totals(ring_dataset):
    torch.manual_seed(0)
    model = GraphAutoencoder(AutoencoderConfig(**TINY))
    analysis = analyze_reconstructions(model, ring_dataset, batch_size=5)
    rows = analysis["rows"]
    assert len(rows) == len(RING_SMILES)
    summary = summarize_failures(analysis)
    assert summary["molecules"] == len(rows)
    assert sum(summary["failure_taxonomy_counts"].values()) == len(rows)
    assert sum(summary["failure_taxonomy"].values()) == pytest.approx(1.0)
    assert summary["ceiling_decodable_fraction"] == pytest.approx(np.mean([r["decodable"] for r in rows]))
    # Every stratification partitions the molecules.
    for key in ("by_size", "by_rings", "by_fused", "by_aromaticity", "by_charge", "by_symmetry"):
        assert sum(group["molecules"] for group in summary[key]) == len(rows), key
    # Unreachable molecules cannot be exact for a deterministic model.
    assert all(not r["constrained_exact"] for r in rows if not r["decodable"])
    assert summary["exact_among_symmetry_unreachable"] == 0.0


def test_analysis_agrees_with_evaluate_reconstruction(ring_dataset):
    torch.manual_seed(1)
    model = GraphAutoencoder(AutoencoderConfig(**TINY))
    analysis = analyze_reconstructions(model, ring_dataset, batch_size=4)
    reference = evaluate_reconstruction(model, ring_dataset, batch_size=4)
    rows = analysis["rows"]
    assert np.mean([r["constrained_exact"] for r in rows]) == pytest.approx(reference["constrained"]["exact_molecule_match"])
    assert np.mean([r["raw_exact"] for r in rows]) == pytest.approx(reference["raw"]["exact_molecule_match"])
    assert np.mean([r["constrained_valid"] for r in rows]) == pytest.approx(reference["constrained"]["validity"])
    assert np.mean([r["atoms_exact"] for r in rows]) == pytest.approx(reference["atom_sequence_exact"])


def test_bond_metrics_expose_inflation_by_the_no_bond_class(ring_dataset):
    torch.manual_seed(2)
    model = GraphAutoencoder(AutoencoderConfig(**TINY))
    bonds = bond_metric_diagnostics(analyze_reconstructions(model, ring_dataset))
    for decoding in ("raw", "constrained"):
        block = bonds[decoding]
        confusion = np.asarray(block["confusion_true_by_predicted"])
        assert confusion.shape == (4, 4) and confusion.sum() == block["undirected_pairs"]
        assert block["pair_accuracy"] == pytest.approx(np.trace(confusion) / confusion.sum())
        assert block["majority_class_baseline_accuracy"] == pytest.approx(confusion[0].sum() / confusion.sum())
        assert 0.5 < block["majority_class_baseline_accuracy"] < 1.0   # no-bond is the majority class
        assert block["accuracy_gain_over_baseline"] == pytest.approx(
            block["pair_accuracy"] - block["majority_class_baseline_accuracy"])
        assert set(block["per_class"]) == {"no_bond", "single", "double", "triple"}
    assert "exact_graph_rate_if_pair_errors_were_independent" in bonds["per_molecule"]


def test_perfect_prediction_has_no_inflation():
    """An oracle model on the no-bond baseline: accuracy 1.0 but baseline stays ~0.9."""
    records = make_records(["CCO", "CC(=O)[O-]", "C[NH3+]"])
    dataset = MolecularGraphDataset(records)

    class Oracle(GraphAutoencoder):
        def decode(self, nodes, batch, *, teacher_forcing):
            big = 30.0
            return {"atom_logits": torch.nn.functional.one_hot(batch.atom_types, 17).float() * big,
                    "bond_logits": torch.nn.functional.one_hot(batch.bond_types, 4).float() * big,
                    "atoms_used": batch.atom_types}
    oracle = Oracle(AutoencoderConfig(**TINY))
    analysis = analyze_reconstructions(oracle, dataset)
    summary = summarize_failures(analysis)
    assert summary["constrained_exact"] == 1.0 and summary["raw_exact"] == 1.0
    bonds = bond_metric_diagnostics(analysis)
    assert bonds["constrained"]["pair_accuracy"] == 1.0
    assert bonds["constrained"]["balanced_accuracy"] == 1.0
    assert bonds["per_molecule"]["molecules_with_zero_bond_errors"] == 1.0


# ===========================================================================
# Experiment hooks added to the model.
# ===========================================================================
def test_bond_head_layers_and_defaults_keep_checkpoint_compatibility():
    default = GraphAutoencoder(AutoencoderConfig(**TINY))
    deeper = GraphAutoencoder(AutoencoderConfig(**TINY, bond_head_layers=3))
    assert len([m for m in default.decoder.bond_head if isinstance(m, torch.nn.Linear)]) == 2
    assert len([m for m in deeper.decoder.bond_head if isinstance(m, torch.nn.Linear)]) == 3
    # An old checkpoint's config dict (no new keys) still builds the identical model.
    old = {k: v for k, v in AutoencoderConfig(**TINY).to_dict().items()
           if k not in ("bond_head_layers", "position_features")}
    assert AutoencoderConfig.from_dict(old) == AutoencoderConfig(**TINY)
    assert default.state_dict().keys() == GraphAutoencoder(AutoencoderConfig.from_dict(old)).state_dict().keys()


def test_position_features_are_deterministic_and_not_equivariant(ring_dataset):
    config = AutoencoderConfig(**TINY, position_features=True)
    model = GraphAutoencoder(config).eval()
    assert model.encoder.embed[0].in_features == 17 + 38
    batch = make_batch(ring_dataset, [1])               # benzene
    nodes = model.encode(batch)[0, :6]
    assert not torch.allclose(nodes, nodes[0].expand_as(nodes), atol=1e-4)    # positions break the tie
    assert torch.equal(model.encode(batch), model.encode(batch))              # but deterministic


def test_evaluation_with_noise_is_reproducible_and_stratified(ring_dataset):
    torch.manual_seed(3)
    model = GraphAutoencoder(AutoencoderConfig(**TINY, symmetry_noise_dim=4))
    a = evaluate_reconstruction(model, ring_dataset, eval_seed=7)
    b = evaluate_reconstruction(model, ring_dataset, eval_seed=7)
    assert a["mean_loss"] == b["mean_loss"] and a["constrained"] == b["constrained"]
    strata = a["exact_molecule_by_symmetry"]
    assert strata["decodable"]["molecules"] + strata["non_decodable"]["molecules"] == len(RING_SMILES)
    spread = noise_draw_spread(model, ring_dataset, draws=3)
    assert len(spread["exact_values"]) == 3 and spread["min"] <= spread["mean"] <= spread["max"]
    deterministic = GraphAutoencoder(AutoencoderConfig(**TINY))
    assert noise_draw_spread(deterministic, ring_dataset, draws=3)["std"] == 0.0


# ===========================================================================
# Flow-relevant embedding measurements.
# ===========================================================================
def test_class_check_separates_deterministic_from_noise_models(ring_dataset):
    torch.manual_seed(4)
    deterministic = GraphAutoencoder(AutoencoderConfig(**TINY))
    check = embedding_class_check(deterministic, ring_dataset)
    assert check["within_class_embeddings_identical"] == 1.0         # equivariance: the ceiling's premise
    assert check["within_class_distance_max_relative"] < 1e-4
    noisy = GraphAutoencoder(AutoencoderConfig(**TINY, symmetry_noise_dim=4))
    noisy_check = embedding_class_check(noisy, ring_dataset)
    assert noisy_check["within_class_embeddings_identical"] < 0.5      # noise breaks symmetry
    assert noisy_check["within_class_distance_max_relative"] > 1e-3


def test_embedding_statistics_noise_fraction(ring_dataset):
    torch.manual_seed(5)
    plain = embedding_statistics(GraphAutoencoder(AutoencoderConfig(**TINY)), ring_dataset)
    noisy = embedding_statistics(GraphAutoencoder(AutoencoderConfig(**TINY, symmetry_noise_dim=4)), ring_dataset)
    assert plain["noise_variance_fraction"] == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < noisy["noise_variance_fraction"] <= 1.5
    for stats in (plain, noisy):
        assert stats["dimension"] == 16 and stats["nodes"] == sum(len(encode_zinc_smiles_charge_aware(s)[0]["atom_types"])
                                                                   for s in RING_SMILES)
        assert 1.0 <= stats["participation_ratio"] <= 16.0


def test_embedding_robustness_is_monotone_for_an_oracle_like_model(ring_dataset):
    torch.manual_seed(6)
    model = GraphAutoencoder(AutoencoderConfig(**TINY))
    results = embedding_robustness(model, ring_dataset, sigmas=(0.0, 0.5, 5.0))
    assert [r["sigma"] for r in results] == [0.0, 0.5, 5.0]
    assert all(0.0 <= r["exact"] <= r["validity"] <= 1.0 for r in results)
    # The zero-noise entry equals plain free-running reconstruction.
    reference = evaluate_reconstruction(model, ring_dataset, compute_ceiling=False)["constrained"]
    assert results[0]["exact"] == pytest.approx(reference["exact_molecule_match"])
    assert results[0]["validity"] == pytest.approx(reference["validity"])


def test_motif_census(ring_dataset):
    census = motif_census(ring_dataset)
    assert census["decodable_molecules"] + census["unreachable_molecules"] == len(RING_SMILES)
    assert set(census["motifs"]) == set(MOTIFS)
    assert census["motifs"]["piperidine"]["unreachable"] > 0
    assert census["any_motif"]["unreachable"] >= census["motifs"]["piperidine"]["unreachable"]
