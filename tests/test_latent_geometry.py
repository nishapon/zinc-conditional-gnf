"""Covariance spectrum and direction-resolved perturbation (diagnostic S2).

The spectrum is checked against synthetic data whose eigenvalues are known by
construction, and the perturbation against the geometry it claims to respect
(which subspace it lives in, and how its length scales).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tests.test_autoencoder_model import make_records
from zinc_gnf.autoencoder import AutoencoderConfig, GraphAutoencoder
from zinc_gnf.dataset import MolecularGraphDataset
from zinc_gnf.latent_geometry import (
    DIRECTION_GROUPS,
    anisotropic_robustness,
    covariance_spectrum,
    direction_indices,
    format_robustness,
    format_spectrum,
    node_embeddings,
    perturbation,
)

SMILES = ["CCO", "c1ccccc1", "CC(=O)Nc1ccccc1", "CCN(CC)CC", "c1ccncc1", "CC(C)C"]


def synthetic_embeddings(variances, nodes=4000, seed=0) -> torch.Tensor:
    """Independent Gaussian columns with the requested variances (so the spectrum is known)."""
    generator = torch.Generator().manual_seed(seed)
    standard = torch.randn((nodes, len(variances)), generator=generator)
    return standard * torch.tensor(variances, dtype=torch.float32).sqrt()


# ---------------------------------------------------------------------------
# Spectrum.
# ---------------------------------------------------------------------------
def test_spectrum_recovers_known_eigenvalues_in_descending_order():
    variances = [9.0, 4.0, 1.0] + [1e-6] * 5
    summary, eigenvalues, eigenvectors = covariance_spectrum(synthetic_embeddings(variances))
    assert summary["dimension"] == 8 and summary["nodes"] == 4000
    assert eigenvalues.tolist() == sorted(eigenvalues.tolist(), reverse=True)
    assert float(eigenvalues[0]) == pytest.approx(9.0, rel=0.1)
    assert float(eigenvalues[1]) == pytest.approx(4.0, rel=0.1)
    assert float(eigenvalues[2]) == pytest.approx(1.0, rel=0.1)
    assert float(eigenvalues[3]) < 1e-4
    assert summary["total_variance"] == pytest.approx(14.0, rel=0.1)
    # eigenvectors are orthonormal and paired with their eigenvalue
    assert torch.allclose(eigenvectors.T @ eigenvectors, torch.eye(8, dtype=eigenvectors.dtype),
                          atol=1e-6)


def test_participation_ratio_and_effective_rank_match_their_definitions():
    variances = [9.0, 4.0, 1.0] + [1e-6] * 5
    summary, eigenvalues, _ = covariance_spectrum(synthetic_embeddings(variances))
    expected = float(eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum())
    assert summary["participation_ratio"] == pytest.approx(expected, rel=1e-6)
    assert 1.0 < summary["participation_ratio"] < 3.0            # three unequal directions
    assert summary["effective_rank_90"] == 2                     # 9+4 of 14 is 93%
    assert summary["effective_rank_99"] == 3                     # the 1.0 direction is needed
    assert summary["effective_rank_999"] == 3
    assert summary["variance_share_top1"] == pytest.approx(9 / 14, rel=0.1)


def test_spectrum_reports_the_correlations_a_diagonal_whitening_cannot_remove():
    generator = torch.Generator().manual_seed(1)
    shared = torch.randn((3000, 1), generator=generator)
    independent = torch.randn((3000, 2), generator=generator)
    # dimensions 0 and 1 are almost the same variable; dimension 2 is independent
    data = torch.cat([shared, shared + 0.05 * independent[:, :1], independent[:, 1:]], dim=1)
    summary, eigenvalues, _ = covariance_spectrum(data, correlation_pairs=3)
    assert summary["max_absolute_correlation"] > 0.9
    strongest = summary["strongest_correlations"][0]
    assert sorted(strongest["dimensions"]) == [0, 1]
    assert summary["effective_rank_99"] <= 2                     # really a 2-dimensional cloud
    # and the marginals are all O(1), which is exactly why diagonal scaling cannot fix it
    assert 0.5 < summary["per_dimension_std_min"] < 2.0


def test_spectrum_flags_dimensions_below_the_whitening_floor():
    summary, _, _ = covariance_spectrum(synthetic_embeddings([4.0, 1e-8, 1e-8, 1.0]))
    assert summary["dimensions_below_whitening_floor_0p01"] == 2
    assert summary["condition_number_max_over_min"] > 1e6


def test_spectrum_rejects_degenerate_input():
    with pytest.raises(ValueError):
        covariance_spectrum(torch.zeros(1, 4))
    with pytest.raises(ValueError):
        covariance_spectrum(torch.zeros(4))


def test_format_spectrum_mentions_the_decisive_numbers():
    summary, _, _ = covariance_spectrum(synthetic_embeddings([4.0, 1.0, 1e-8]))
    text = format_spectrum(summary)
    for fragment in ("participation ratio", "effective rank", "correlation",
                     "dimensions below 0.01 std"):
        assert fragment in text


# ---------------------------------------------------------------------------
# Direction groups and perturbation geometry.
# ---------------------------------------------------------------------------
def test_direction_indices_pick_the_intended_eigenvalues():
    assert direction_indices("top", 10, 3) == [0, 1, 2]
    assert direction_indices("bottom", 10, 3) == [7, 8, 9]
    assert direction_indices("isotropic", 10, 3) == list(range(10))
    with pytest.raises(ValueError):
        direction_indices("sideways", 10, 3)
    with pytest.raises(ValueError):
        direction_indices("top", 10, 0)
    with pytest.raises(ValueError):
        direction_indices("top", 10, 11)


def test_perturbation_lives_in_the_requested_subspace():
    variances = [16.0, 8.0, 4.0, 1.0, 0.25, 0.01]
    _, eigenvalues, eigenvectors = covariance_spectrum(synthetic_embeddings(variances))
    embeddings = synthetic_embeddings(variances, nodes=200, seed=3)
    generator = torch.Generator().manual_seed(0)
    bottom = perturbation(embeddings, eigenvalues, eigenvectors, group="bottom",
                          magnitude=0.3, mode="absolute", k=2, generator=generator, scale_unit=1.0)
    top_basis = eigenvectors[:, :4].to(bottom.dtype)
    # a bottom-group displacement has no component in the dominant directions
    assert float((bottom @ top_basis).abs().max()) < 1e-4
    bottom_basis = eigenvectors[:, 4:].to(bottom.dtype)
    assert float((bottom @ bottom_basis).abs().max()) > 1e-3


def test_absolute_mode_matches_displacement_length_across_groups():
    variances = [16.0, 8.0, 4.0, 1.0, 0.25, 0.01]
    _, eigenvalues, eigenvectors = covariance_spectrum(synthetic_embeddings(variances))
    embeddings = synthetic_embeddings(variances, nodes=4000, seed=5)
    lengths = {}
    for group in ("top", "bottom"):
        generator = torch.Generator().manual_seed(7)
        displacement = perturbation(embeddings, eigenvalues, eigenvectors, group=group,
                                    magnitude=0.2, mode="absolute", k=2, generator=generator,
                                    scale_unit=1.0)
        lengths[group] = float(displacement.norm(dim=1).mean())
    # same k, same coefficient scale -> same expected length, so any difference in the
    # decoder's response is about direction, not about how far we moved
    assert lengths["top"] == pytest.approx(lengths["bottom"], rel=1e-6)
    # a 2-direction isotropic Gaussian displacement of coefficient std s has
    # E||d|| = s * sqrt(pi / 2)
    assert lengths["top"] == pytest.approx(0.2 * math.sqrt(math.pi / 2), rel=0.05)


def test_relative_mode_scales_with_each_directions_own_spread():
    variances = [16.0, 8.0, 4.0, 1.0, 0.25, 0.01]
    _, eigenvalues, eigenvectors = covariance_spectrum(synthetic_embeddings(variances))
    embeddings = synthetic_embeddings(variances, nodes=4000, seed=5)
    lengths = {}
    for group in ("top", "bottom"):
        generator = torch.Generator().manual_seed(7)
        displacement = perturbation(embeddings, eigenvalues, eigenvectors, group=group,
                                    magnitude=0.2, mode="relative", k=2, generator=generator)
        lengths[group] = float(displacement.norm(dim=1).mean())
    # top directions have far more spread, so a relative perturbation there is much longer
    assert lengths["top"] > 10 * lengths["bottom"]


def test_zero_magnitude_is_exactly_no_perturbation():
    _, eigenvalues, eigenvectors = covariance_spectrum(synthetic_embeddings([4.0, 1.0, 0.1]))
    embeddings = synthetic_embeddings([4.0, 1.0, 0.1], nodes=50, seed=2)
    for group in DIRECTION_GROUPS:
        displacement = perturbation(embeddings, eigenvalues, eigenvectors, group=group,
                                    magnitude=0.0, mode="absolute", k=1)
        assert torch.equal(displacement, torch.zeros_like(embeddings))


def test_perturbation_is_reproducible_and_validates_its_mode():
    _, eigenvalues, eigenvectors = covariance_spectrum(synthetic_embeddings([4.0, 1.0, 0.1]))
    embeddings = synthetic_embeddings([4.0, 1.0, 0.1], nodes=20, seed=4)
    kwargs = dict(group="top", magnitude=0.2, mode="absolute", k=2, scale_unit=1.0)
    first = perturbation(embeddings, eigenvalues, eigenvectors,
                         generator=torch.Generator().manual_seed(11), **kwargs)
    second = perturbation(embeddings, eigenvalues, eigenvectors,
                          generator=torch.Generator().manual_seed(11), **kwargs)
    assert torch.equal(first, second)
    with pytest.raises(ValueError):
        perturbation(embeddings, eigenvalues, eigenvectors, group="top", magnitude=0.2,
                     mode="sideways", k=2)


# ---------------------------------------------------------------------------
# Robustness measurement, through the real autoencoder.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_model_and_data():
    torch.manual_seed(0)
    model = GraphAutoencoder(AutoencoderConfig(
        node_embedding_dim=16, latent_dim=32, num_processing_steps=1, attn_kq_dim=8,
        attn_v_dim=8, attn_concat_heads_output_dim=8, degree_head=True)).eval()
    dataset = MolecularGraphDataset(make_records(SMILES))
    return model, dataset


def test_robustness_rows_cover_every_setting_with_one_shared_baseline(tiny_model_and_data):
    model, dataset = tiny_model_and_data
    embeddings = node_embeddings(model, dataset, device="cpu")
    _, eigenvalues, eigenvectors = covariance_spectrum(embeddings)
    rows = anisotropic_robustness(model, dataset, eigenvalues=eigenvalues,
                                  eigenvectors=eigenvectors, device="cpu",
                                  magnitudes=(0.0, 0.1), k=3, batch_size=4)
    baseline = [r for r in rows if r["mode"] == "baseline"]
    assert len(baseline) == 1                                    # exactly one shared baseline
    assert baseline[0]["magnitude"] == 0.0 and baseline[0]["mean_displacement_norm"] == 0.0
    perturbed = [r for r in rows if r["mode"] != "baseline"]
    assert len(perturbed) == 2 * 3                               # 2 modes x 3 groups x 1 magnitude
    assert {r["group"] for r in perturbed} == set(DIRECTION_GROUPS)
    assert {r["mode"] for r in perturbed} == {"absolute", "relative"}
    for row in rows:
        assert row["molecules"] == len(dataset)
        assert 0.0 <= row["degree"]["exact_molecule_match"] <= 1.0
        assert 0.0 <= row["degree"]["strict_validity"] <= 1.0
        assert 0.0 <= row["degree_accuracy"] <= 1.0
        assert 0.0 <= row["degree_fallback_rate"] <= 1.0
        assert "constrained" in row and "qed_mean" in row["degree"]
        assert row["scale_unit_per_dimension_std"] > 0


def test_larger_magnitudes_move_the_embeddings_further(tiny_model_and_data):
    model, dataset = tiny_model_and_data
    embeddings = node_embeddings(model, dataset, device="cpu")
    _, eigenvalues, eigenvectors = covariance_spectrum(embeddings)
    rows = anisotropic_robustness(model, dataset, eigenvalues=eigenvalues,
                                  eigenvectors=eigenvectors, device="cpu",
                                  magnitudes=(0.0, 0.1, 0.4), groups=("isotropic",),
                                  modes=("absolute",), k=3, batch_size=4)
    norms = [r["mean_displacement_norm"] for r in rows]
    assert norms == sorted(norms)
    assert norms[-1] > 3 * norms[1] > 0


def test_baseline_reconstruction_is_measured_through_the_same_code_path(tiny_model_and_data):
    """The magnitude-0 row must equal a direct evaluation of the unperturbed model."""
    from zinc_gnf.autoencoder import evaluate_reconstruction

    model, dataset = tiny_model_and_data
    embeddings = node_embeddings(model, dataset, device="cpu")
    _, eigenvalues, eigenvectors = covariance_spectrum(embeddings)
    rows = anisotropic_robustness(model, dataset, eigenvalues=eigenvalues,
                                  eigenvectors=eigenvectors, device="cpu",
                                  magnitudes=(0.0,), k=3, batch_size=4)
    direct = evaluate_reconstruction(model, dataset, device="cpu", compute_ceiling=False,
                                     batch_size=4)
    assert rows[0]["degree"]["exact_molecule_match"] == pytest.approx(
        direct["degree"]["exact_molecule_match"], abs=1e-9)
    assert rows[0]["degree"]["strict_validity"] == pytest.approx(
        direct["degree"]["validity"], abs=1e-9)
    assert rows[0]["degree_accuracy"] == pytest.approx(direct["degree_accuracy"], abs=1e-9)
    assert rows[0]["degree_fallback_rate"] == pytest.approx(
        direct["degree_decode_fallback_rate"], abs=1e-9)


def test_robustness_works_without_a_degree_head():
    torch.manual_seed(1)
    model = GraphAutoencoder(AutoencoderConfig(
        node_embedding_dim=16, latent_dim=32, num_processing_steps=1, attn_kq_dim=8,
        attn_v_dim=8, attn_concat_heads_output_dim=8, degree_head=False)).eval()
    dataset = MolecularGraphDataset(make_records(SMILES[:3]))
    embeddings = node_embeddings(model, dataset, device="cpu")
    _, eigenvalues, eigenvectors = covariance_spectrum(embeddings)
    rows = anisotropic_robustness(model, dataset, eigenvalues=eigenvalues,
                                  eigenvectors=eigenvectors, device="cpu",
                                  magnitudes=(0.0, 0.2), groups=("top",), modes=("absolute",),
                                  k=2, batch_size=4)
    assert all("degree" not in r for r in rows)
    assert all("constrained" in r for r in rows)
    assert "degree_accuracy" not in rows[0]
    assert "decoder: constrained" in format_robustness(rows)


def test_node_embeddings_returns_only_real_nodes(tiny_model_and_data):
    model, dataset = tiny_model_and_data
    embeddings = node_embeddings(model, dataset, device="cpu", batch_size=2)
    assert embeddings.shape == (int(dataset.n_node.sum()), model.config.node_embedding_dim)
    subset = node_embeddings(model, dataset, indices=[0, 1], device="cpu")
    assert subset.shape[0] == int(dataset.n_node[0] + dataset.n_node[1])


def test_format_robustness_tabulates_every_row(tiny_model_and_data):
    model, dataset = tiny_model_and_data
    embeddings = node_embeddings(model, dataset, device="cpu")
    _, eigenvalues, eigenvectors = covariance_spectrum(embeddings)
    rows = anisotropic_robustness(model, dataset, eigenvalues=eigenvalues,
                                  eigenvectors=eigenvectors, device="cpu",
                                  magnitudes=(0.0, 0.1), k=2, batch_size=4)
    text = format_robustness(rows)
    assert "basis from TRAIN" in text and "decoder: degree" in text
    assert len(text.splitlines()) == len(rows) + 3               # title, header, rule, rows
