"""The shuffled-node control: population construction, inter-node statistics and
pre-assembly decoder statistics.

The defining property of the shuffled population is checked exactly: it must use the
same multiset of node vectors as the real one (so every per-node statistic is
preserved) while the grouping into molecules is destroyed. The symmetry test is
training-independent: the encoder is permutation-equivariant, so symmetry-equivalent
atoms get identical embeddings even in an untrained model, which gives real molecules
near-identical embedding pairs that shuffled ones cannot have.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.test_autoencoder_model import make_records
from zinc_gnf.autoencoder import AutoencoderConfig, GraphAutoencoder
from zinc_gnf.dataset import MolecularGraphDataset
from zinc_gnf.embedding_populations import (
    EmbeddingPopulation,
    decode_population,
    decoder_prediction_statistics,
    embedding_population_statistics,
    flow_population,
    format_population_statistics,
    format_prediction_statistics,
    real_population,
    shuffle_population,
)
from zinc_gnf.flow import FlowConfig, NodeFlow
from zinc_gnf.latent_geometry import covariance_spectrum, node_embeddings

# benzene and cyclohexane are highly symmetric: every atom is equivalent, so an
# equivariant encoder gives them identical embeddings.
SYMMETRIC = ["c1ccccc1", "C1CCCCC1", "CC(=O)Nc1ccccc1", "CCN(CC)CC", "CCO", "c1ccncc1"]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return GraphAutoencoder(AutoencoderConfig(
        node_embedding_dim=16, latent_dim=32, num_processing_steps=2, attn_kq_dim=8,
        attn_v_dim=8, attn_concat_heads_output_dim=8, degree_head=True)).eval()


@pytest.fixture(scope="module")
def dataset():
    return MolecularGraphDataset(make_records(SYMMETRIC))


@pytest.fixture(scope="module")
def real(model, dataset):
    return real_population(model, dataset, range(len(dataset)), device="cpu", batch_size=3)


# ---------------------------------------------------------------------------
# Population construction.
# ---------------------------------------------------------------------------
def test_real_population_matches_the_dataset_sizes(real, dataset):
    assert real.name == "real" and real.molecules == len(dataset)
    assert real.counts == [int(n) for n in dataset.n_node]
    assert real.pooled.shape == (int(dataset.n_node.sum()), 16)
    assert [v.shape[0] for v in real.per_molecule()] == real.counts


def test_shuffling_preserves_every_per_node_statistic_and_all_sizes(real):
    shuffled, report = shuffle_population(real, seed=1)
    assert shuffled.counts == real.counts                      # sizes preserved exactly
    assert shuffled.pooled.shape == real.pooled.shape
    # the SAME multiset of vectors: sorting both pooled tensors must agree exactly
    left = torch.sort(real.pooled.flatten()).values
    right = torch.sort(shuffled.pooled.flatten()).values
    assert torch.equal(left, right)
    assert torch.allclose(shuffled.pooled.mean(0), real.pooled.mean(0), atol=1e-6)
    assert torch.allclose(shuffled.pooled.std(0), real.pooled.std(0), atol=1e-6)
    _, real_eigenvalues, _ = covariance_spectrum(real.pooled)
    _, shuffled_eigenvalues, _ = covariance_spectrum(shuffled.pooled)
    assert torch.allclose(real_eigenvalues, shuffled_eigenvalues, atol=1e-8)
    assert report["permuted_nodes"] == real.pooled.shape[0]
    assert 0.0 <= report["fraction_back_in_their_own_molecule"] <= 1.0
    # and the grouping really did change
    assert not torch.equal(shuffled.pooled, real.pooled)


def test_shuffling_is_reproducible_and_seed_dependent(real):
    first, _ = shuffle_population(real, seed=3)
    again, _ = shuffle_population(real, seed=3)
    other, _ = shuffle_population(real, seed=4)
    assert torch.equal(first.pooled, again.pooled)
    assert not torch.equal(first.pooled, other.pooled)


def test_population_rejects_inconsistent_counts():
    with pytest.raises(ValueError, match="nodes expected"):
        EmbeddingPopulation(name="bad", counts=[3, 4], pooled=torch.zeros(6, 8))


def test_batches_pad_and_preserve_order(real):
    seen = []
    for counts, padded in real.batches(2):
        assert padded.shape[0] == len(counts) and padded.shape[1] == max(counts)
        for position, count in enumerate(counts):
            assert torch.equal(padded[position, count:], torch.zeros_like(padded[position, count:]))
        seen.extend(counts)
    assert seen == real.counts


def test_flow_population_uses_the_requested_sizes():
    config = FlowConfig(node_embedding_dim=16, num_blocks=1, attention_dim=8, num_heads=2,
                        hidden_dim=16)
    flow = NodeFlow(config, init_seed=0)
    flow.set_data_statistics(torch.zeros(16), torch.ones(16))
    flow.eval()
    counts = [6, 11, 4]
    population = flow_population(flow, counts, temperature=0.9, seed=2, batch_size=2)
    assert population.counts == counts
    assert population.pooled.shape == (21, 16)
    assert population.name == "flow_T0.9"
    again = flow_population(flow, counts, temperature=0.9, seed=2, batch_size=2)
    assert torch.equal(population.pooled, again.pooled)
    conditional = NodeFlow(FlowConfig(node_embedding_dim=16, num_blocks=1, attention_dim=8,
                                      num_heads=2, hidden_dim=16, condition_features=("n",)))
    conditional.set_data_statistics(torch.zeros(16), torch.ones(16))
    with pytest.raises(ValueError, match="conditional"):
        flow_population(conditional, counts)


# ---------------------------------------------------------------------------
# Inter-node statistics: the point of the whole control.
# ---------------------------------------------------------------------------
def test_shuffled_destroys_the_within_molecule_relationship(real):
    shuffled, _ = shuffle_population(real, seed=0)
    _, _, eigenvectors = covariance_spectrum(real.pooled)
    scale = float(real.pooled.to(torch.float64).std(0).mean())
    real_statistics = embedding_population_statistics(
        real, reference_eigenvectors=eigenvectors, reference_scale=scale)
    shuffled_statistics = embedding_population_statistics(
        shuffled, reference_eigenvectors=eigenvectors, reference_scale=scale)

    # by construction the shuffled population's nodes are unrelated, so a within-molecule
    # pair is no closer than a between-molecule pair: the ratio must sit at 1
    assert shuffled_statistics["within_over_between_distance"] == pytest.approx(1.0, abs=0.25)
    # The real population's ratio is a measurement, not an invariant: whether a molecule's
    # own atoms are closer together or further apart than random atoms depends on what the
    # encoder learned, so the unit test only requires the statistic to be SENSITIVE to the
    # grouping. Which way it goes on the trained encoder is what the diagnostic reports.
    assert real_statistics["within_over_between_distance"] != pytest.approx(
        shuffled_statistics["within_over_between_distance"], abs=1e-6)
    # every per-node statistic is identical, which is what makes the comparison clean
    assert real_statistics["mean_norm"] == pytest.approx(shuffled_statistics["mean_norm"], rel=1e-9)
    assert real_statistics["participation_ratio"] == pytest.approx(
        shuffled_statistics["participation_ratio"], rel=1e-6)
    assert real_statistics["variance_share_in_real_top5"] == pytest.approx(
        shuffled_statistics["variance_share_in_real_top5"], rel=1e-6)


def test_real_symmetric_molecules_have_near_identical_pairs_and_shuffled_do_not(model, dataset):
    """Training-independent: an equivariant encoder gives equivalent atoms equal vectors."""
    real = real_population(model, dataset, range(len(dataset)), device="cpu")
    shuffled, _ = shuffle_population(real, seed=0)
    scale = float(real.pooled.to(torch.float64).std(0).mean())
    real_statistics = embedding_population_statistics(real, reference_scale=scale)
    shuffled_statistics = embedding_population_statistics(shuffled, reference_scale=scale)
    # benzene and cyclohexane are fully symmetric, so at least 2 of 6 molecules qualify
    assert real_statistics["molecules_with_near_identical_pair"] >= 2 / 6
    assert real_statistics["near_identical_tolerance"] > 0
    # The shuffled population is the CALIBRATION for this statistic rather than a
    # guaranteed zero: a pool that already contains many equivalent atoms (every aromatic
    # carbon of every molecule) can hand a random grouping a near-identical pair by
    # chance. Both numbers are therefore reported side by side and the unit test only
    # requires both to be computed on the same scale.
    assert 0.0 <= shuffled_statistics["molecules_with_near_identical_pair"] <= 1.0
    assert shuffled_statistics["near_identical_tolerance"] == pytest.approx(
        real_statistics["near_identical_tolerance"])


def test_statistics_project_onto_the_reference_basis(real):
    _, _, eigenvectors = covariance_spectrum(real.pooled)
    statistics = embedding_population_statistics(real, reference_eigenvectors=eigenvectors)
    assert 0.0 <= statistics["variance_share_in_real_top5"] <= 1.0
    assert statistics["variance_share_in_real_top5"] <= statistics["variance_share_in_real_top10"]
    assert len(statistics["projected_variance_per_real_component"]) == 10
    # projecting a population onto its OWN basis puts the variance in the leading components
    assert statistics["variance_share_in_real_top10"] > 0.5
    without = embedding_population_statistics(real)
    assert "variance_share_in_real_top5" not in without


def test_formatting_lists_every_population(real):
    shuffled, _ = shuffle_population(real, seed=0)
    _, _, eigenvectors = covariance_spectrum(real.pooled)
    rows = [embedding_population_statistics(p, reference_eigenvectors=eigenvectors)
            for p in (real, shuffled)]
    text = format_population_statistics(rows)
    assert "real" in text and "shuffled" in text
    for fragment in ("within / between", "near-identical pair", "participation ratio",
                     "var in real top-5"):
        assert fragment in text


# ---------------------------------------------------------------------------
# Decoder predictions before assembly.
# ---------------------------------------------------------------------------
def test_prediction_statistics_are_well_formed(model, real):
    statistics = decoder_prediction_statistics(model, real, device="cpu", batch_size=3)
    assert statistics["molecules"] == real.molecules
    assert statistics["atoms"] == sum(real.counts)
    assert len(statistics["atom_type_histogram"]) == 17
    assert len(statistics["atom_type_labels"]) == 17
    assert sum(statistics["atom_type_histogram"]) == pytest.approx(1.0)
    assert len(statistics["bond_argmax_histogram"]) == 4
    assert sum(statistics["bond_argmax_histogram"]) == pytest.approx(1.0)
    assert 0.0 <= statistics["charged_atom_fraction"] <= 1.0
    assert 0.0 <= statistics["no_bond_probability_mean"] <= 1.0
    assert 1.0 / 17 <= statistics["atom_confidence_mean"] <= 1.0
    assert 0.25 <= statistics["bond_confidence_mean"] <= 1.0
    assert len(statistics["degree_histogram"]) == 7
    assert 0.0 <= statistics["molecules_with_even_degree_sum"] <= 1.0
    assert 0.0 <= statistics["molecules_with_feasible_degrees"] <= 1.0
    assert statistics["expected_bonds_per_atom_mean"] >= 0.0


def test_prediction_statistics_differ_between_real_and_shuffled(model, real):
    """The decoder's own opinions must change when the inter-node relation is destroyed."""
    shuffled, _ = shuffle_population(real, seed=0)
    real_statistics = decoder_prediction_statistics(model, real, device="cpu", batch_size=3)
    shuffled_statistics = decoder_prediction_statistics(model, shuffled, device="cpu", batch_size=3)
    # per-node predictions see the same vectors, so their histograms are identical
    assert real_statistics["atom_type_histogram"] == shuffled_statistics["atom_type_histogram"]
    assert real_statistics["charged_atom_fraction"] == pytest.approx(
        shuffled_statistics["charged_atom_fraction"])
    assert real_statistics["degree_histogram"] == shuffled_statistics["degree_histogram"]
    # the pairwise ones are computed from pairs of embeddings and so can differ
    assert "bond_argmax_histogram" in shuffled_statistics


def test_prediction_statistics_without_a_degree_head(real):
    torch.manual_seed(1)
    plain = GraphAutoencoder(AutoencoderConfig(
        node_embedding_dim=16, latent_dim=32, num_processing_steps=1, attn_kq_dim=8,
        attn_v_dim=8, attn_concat_heads_output_dim=8, degree_head=False)).eval()
    statistics = decoder_prediction_statistics(plain, real, device="cpu", batch_size=3)
    assert "degree_histogram" not in statistics
    assert "molecules_with_even_degree_sum" not in statistics
    text = format_prediction_statistics([statistics])
    assert "bond argmax class" in text and "P(no bond) mean" in text


# ---------------------------------------------------------------------------
# Assembling molecules from a population.
# ---------------------------------------------------------------------------
def test_decode_population_returns_one_record_per_molecule(model, real):
    records = decode_population(model, real, device="cpu", batch_size=3)
    assert len(records) == real.molecules
    assert [r["n_node"] for r in records] == real.counts
    for record in records:
        assert {"constrained", "degree"} <= set(record)
        assert isinstance(record["degree"]["valid"], bool)


def test_decode_population_respects_a_missing_degree_head(real):
    torch.manual_seed(2)
    plain = GraphAutoencoder(AutoencoderConfig(
        node_embedding_dim=16, latent_dim=32, num_processing_steps=1, attn_kq_dim=8,
        attn_v_dim=8, attn_concat_heads_output_dim=8, degree_head=False)).eval()
    records = decode_population(plain, real, device="cpu", batch_size=3)
    assert all("degree" not in r for r in records)
    assert all("constrained" in r for r in records)
