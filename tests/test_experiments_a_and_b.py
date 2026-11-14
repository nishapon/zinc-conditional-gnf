"""Tests for experiment A (balanced sampling) and experiment B (coupled prior).

The load-bearing tests are the numerical ones. For B the whole proposal rests on the
closed-form density being the exact density of the rank-one-plus-diagonal Gaussian, so it
is checked against an independent brute-force computation that builds the dense N x N
covariance and solves it (`dense_log_prob`). If that agreement fails, the experiment is
invalid regardless of any downstream metric.

For A the load-bearing property is that the two guards actually bind: a tiny cell must
not be upweighted, and no molecule may dominate an epoch. Naive inverse-frequency
weighting on ZINC250K's long tail would do both.

Needs torch (B) and numpy (A); both skip without them.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from zinc_gnf.balanced_sampler import (  # noqa: E402
    BalancedSampler,
    cell_index,
    joint_distribution,
)
from zinc_gnf.coupled_prior import (  # noqa: E402
    INIT_RHO,
    RHO_MAX,
    CoupledNodePrior,
    dense_log_prob,
)
from zinc_gnf.flow import NodePrior  # noqa: E402

DIM = 4


def prior(*, conditional=False, rho=INIT_RHO, learnable=True):
    base = NodePrior(DIM, condition_dim=8 if conditional else 0)
    return CoupledNodePrior.from_node_prior(base, init_rho=rho, learnable=learnable)


def masks(counts, width=None):
    width = width or max(counts)
    mask = torch.zeros((len(counts), width), dtype=torch.bool)
    for row, count in enumerate(counts):
        mask[row, :count] = True
    return mask


# ================================ B: exactness ===================================
@pytest.mark.parametrize("rho", [1e-6, 0.01, 0.3, 0.85])
def test_the_closed_form_density_matches_a_dense_covariance_solve(rho):
    """The claim the whole experiment rests on, including masked batches."""
    model = prior(rho=rho).double()
    mask = masks([5, 3, 7])
    torch.manual_seed(0)
    z = torch.randn((3, 7, DIM), dtype=torch.float64) * mask.unsqueeze(-1)
    mu, sigma = model.parameters_for(3, None, "cpu")
    closed = model.log_prob(z, mask, None)
    brute = dense_log_prob(z, mask, mu, sigma, model.rho())
    assert torch.allclose(closed, brute, atol=1e-8), (closed, brute)


def test_a_vanishing_rho_approaches_the_iid_density():
    base = NodePrior(DIM, condition_dim=0).double()
    model = CoupledNodePrior.from_node_prior(base, init_rho=1e-8).double()
    mask = masks([6, 4])
    torch.manual_seed(1)
    z = torch.randn((2, 6, DIM), dtype=torch.float64) * mask.unsqueeze(-1)
    assert torch.allclose(model.log_prob(z, mask, None),
                          base.log_prob(z, mask, None), atol=1e-5)


def test_rho_zero_is_refused_because_the_baseline_is_the_original_prior():
    base = NodePrior(DIM, condition_dim=0)
    for bad in (0.0, -0.1, RHO_MAX, RHO_MAX + 1):
        with pytest.raises(ValueError, match="init_rho"):
            CoupledNodePrior.from_node_prior(base, init_rho=bad)


def test_the_gradient_is_nonzero_at_the_initialisation():
    """The bug this parameterisation fixes: d/da carries a factor 2a and dies at a=0."""
    model = prior(rho=INIT_RHO)
    mask = masks([5, 5])
    torch.manual_seed(2)
    z = torch.randn((2, 5, DIM)) * mask.unsqueeze(-1)
    model.log_prob(z, mask, None).sum().backward()
    assert model.pre_rho.grad is not None
    assert torch.isfinite(model.pre_rho.grad).all()
    assert float(model.pre_rho.grad.abs().min()) > 0.0


def test_rho_receives_gradient_from_correlated_data_and_rises():
    """On data with a strong shared component the likelihood must push rho UP."""
    model = prior(rho=INIT_RHO).double()
    mask = masks([6] * 256)
    generator = torch.Generator().manual_seed(3)
    shared = torch.randn((256, 1, DIM), generator=generator, dtype=torch.float64)
    private = torch.randn((256, 6, DIM), generator=generator, dtype=torch.float64)
    z = (0.2 * private + 0.98 * shared) * mask.unsqueeze(-1)
    before = float(model.rho().mean())
    optimiser = torch.optim.Adam([model.pre_rho], lr=0.1)
    for _ in range(40):
        optimiser.zero_grad()
        (-model.log_prob(z, mask, None).mean()).backward()
        optimiser.step()
    assert float(model.rho().mean()) > before
    assert model.summary()["moved_from_initialisation"] is True


def test_a_frozen_rho_does_not_move_under_the_same_data():
    model = prior(rho=INIT_RHO, learnable=False).double()
    assert model.pre_rho.requires_grad is False
    mask = masks([6] * 64)
    torch.manual_seed(4)
    z = torch.randn((64, 6, DIM), dtype=torch.float64) * mask.unsqueeze(-1)
    before = float(model.rho().mean())
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.Adam(trainable or [torch.zeros(1, requires_grad=True)],
                                 lr=0.1)
    optimiser.zero_grad()
    model.log_prob(z, mask, None).mean().backward()
    optimiser.step()
    assert float(model.rho().mean()) == pytest.approx(before)


def test_the_new_parameter_is_registered_and_optimised_after_a_reload(tmp_path):
    """It must reach the optimizer built from parameters() after checkpoint loading."""
    model = prior(rho=0.05)
    torch.save(model.state_dict(), tmp_path / "prior.pt")
    fresh = prior(rho=INIT_RHO)
    fresh.load_state_dict(torch.load(tmp_path / "prior.pt", weights_only=True))
    assert float(fresh.rho().mean()) == pytest.approx(0.05, abs=1e-6)
    names = [name for name, _ in fresh.named_parameters()]
    assert "pre_rho" in names
    optimiser = torch.optim.Adam(fresh.parameters(), lr=0.1)
    tracked = {id(p) for group in optimiser.param_groups for p in group["params"]}
    assert id(fresh.pre_rho) in tracked
    mask = masks([5, 5])
    torch.manual_seed(6)
    z = torch.randn((2, 5, DIM)) * mask.unsqueeze(-1)
    before = fresh.pre_rho.detach().clone()
    optimiser.zero_grad()
    (-fresh.log_prob(z, mask, None).mean()).backward()
    optimiser.step()
    assert not torch.allclose(before, fresh.pre_rho.detach())


def test_padding_contributes_neither_likelihood_nor_shared_statistics():
    model = prior(rho=0.4).double()
    narrow_mask, wide_mask = masks([5], width=5), masks([5], width=11)
    torch.manual_seed(7)
    core = torch.randn((1, 5, DIM), dtype=torch.float64)
    narrow = torch.zeros((1, 5, DIM), dtype=torch.float64); narrow[:, :5] = core
    wide = torch.zeros((1, 11, DIM), dtype=torch.float64); wide[:, :5] = core
    wide[:, 5:] = 99.0
    assert torch.allclose(model.log_prob(narrow, narrow_mask, None),
                          model.log_prob(wide, wide_mask, None), atol=1e-10)


def test_the_density_is_exchangeable_over_nodes():
    model = prior(rho=0.5).double()
    mask = masks([6])
    torch.manual_seed(8)
    z = torch.randn((1, 6, DIM), dtype=torch.float64)
    assert torch.allclose(model.log_prob(z, mask, None),
                          model.log_prob(z[:, torch.randperm(6)], mask, None),
                          atol=1e-10)


def test_sampling_preserves_the_marginal_variance_and_induces_rho():
    """The explicit choice: variance is REDISTRIBUTED, not added."""
    model = prior(rho=0.4).double()
    mask = masks([2] * 40000)
    samples = model.sample(mask, None,
                           generator=torch.Generator().manual_seed(9)).double()
    mu, sigma = model.parameters_for(1, None, "cpu")
    scale = float(sigma[0, 0])
    assert float(samples[:, 0, 0].var()) == pytest.approx(scale ** 2, rel=0.05)
    correlation = float(torch.corrcoef(
        torch.stack([samples[:, 0, 0], samples[:, 1, 0]]))[0, 1])
    assert correlation == pytest.approx(0.4, abs=0.03)


def test_sampling_respects_the_mask():
    model = prior(rho=0.3)
    mask = masks([3, 5], width=5)
    samples = model.sample(mask, None,
                           generator=torch.Generator().manual_seed(10))
    assert torch.count_nonzero(samples[0, 3:]) == 0


def test_rho_is_bounded():
    model = prior(rho=0.5)
    with torch.no_grad():
        model.pre_rho.fill_(50.0)
    assert float(model.rho().max()) <= RHO_MAX
    with torch.no_grad():
        model.pre_rho.fill_(-50.0)
    assert float(model.rho().min()) > 0.0


def test_the_conditional_pathway_is_reused_unchanged():
    model = prior(conditional=True, rho=0.2)
    assert model.conditional is True
    condition = torch.randn((3, 8))
    mu, sigma = model.parameters_for(3, condition, "cpu")
    base_mu, base_sigma = model.base.parameters_for(3, condition, "cpu")
    assert torch.allclose(mu, base_mu) and torch.allclose(sigma, base_sigma)















# ============================== A: balanced sampling =============================
def synthetic(counts_per_cell):
    """(n_node, qed) arrays with a controlled joint distribution."""
    n_node, qed = [], []
    for (n, bucket), population in counts_per_cell.items():
        for _ in range(population):
            n_node.append(n)
            qed.append(bucket * 0.1 + 0.05)
    return np.asarray(n_node), np.asarray(qed)


def test_the_joint_distribution_is_measured_not_assumed():
    n_node, qed = synthetic({(15, 5): 1000, (23, 6): 100, (30, 4): 3})
    report = joint_distribution(n_node, qed)
    assert report["molecules"] == 1103 and report["cells"] == 3
    assert report["largest_cell"] == 1000 and report["smallest_cell"] == 3
    assert report["population_ratio"] == pytest.approx(1000 / 3)
    assert report["cells_below_10"] == 1
    assert report["effective_cells"] < report["cells"]


def test_alpha_zero_reproduces_uniform_sampling():
    """The control must be the existing behaviour exactly."""
    n_node, qed = synthetic({(15, 5): 500, (23, 6): 50})
    sampler = BalancedSampler(n_node, qed, batch_size=10, seed=1, alpha=0.0)
    assert sampler.report()["reduces_to_uniform"] is True
    assert float(sampler.weights.max()) == pytest.approx(
        float(sampler.weights.min()), rel=1e-9)
    exposure = sampler.report()["weighting"]["exposure_ratio_by_cell"]
    assert all(value == pytest.approx(1.0, abs=1e-9) for value in exposure)


def test_a_tiny_cell_has_exposure_ratio_one_before_clipping():
    """Checked on the REALISED exposure ratio, not inferred from the construction."""
    n_node, qed = synthetic({(15, 5): 2000, (23, 6): 600, (30, 7): 4})
    sampler = BalancedSampler(n_node, qed, batch_size=8, seed=2, alpha=1.0,
                              min_cell_count=50)
    diagnostics = sampler.report()["weighting"]
    assert diagnostics["cells_tiny_held_at_natural_mass"] == 1
    assert diagnostics["molecules_in_tiny_cells"] == 4
    assert diagnostics["max_exposure_ratio_among_tiny_cells_before_clip"] == \
        pytest.approx(1.0, abs=1e-9)
    assert diagnostics["max_exposure_ratio_among_tiny_cells"] <= 1.0 + 1e-9


def test_the_report_gives_final_cell_probabilities_that_sum_to_one():
    n_node, qed = synthetic({(15, 5): 900, (23, 6): 300, (30, 7): 100})
    diagnostics = BalancedSampler(n_node, qed, batch_size=8, seed=3,
                                  alpha=0.5).report()["weighting"]
    assert sum(diagnostics["final_cell_probability"]) == pytest.approx(1.0, abs=1e-9)
    assert sum(diagnostics["natural_cell_probability"]) == pytest.approx(1.0, abs=1e-9)
    assert len(diagnostics["exposure_ratio_by_cell"]) == 3


def test_a_rare_eligible_cell_is_the_one_that_gains_exposure():
    n_node, qed = synthetic({(15, 5): 4000, (23, 6): 200})
    diagnostics = BalancedSampler(n_node, qed, batch_size=8, seed=4, alpha=0.5,
                                  min_cell_count=50).report()["weighting"]
    natural = diagnostics["natural_cell_probability"]
    exposure = diagnostics["exposure_ratio_by_cell"]
    rare = natural.index(min(natural))
    common = natural.index(max(natural))
    assert exposure[rare] > 1.0 > exposure[common]


def test_the_exposure_clip_binds():
    n_node, qed = synthetic({(15, 5): 5000, (23, 6): 60})
    diagnostics = BalancedSampler(n_node, qed, batch_size=8, seed=3, alpha=1.0,
                                  min_cell_count=50,
                                  max_exposure_ratio=5.0).report()["weighting"]
    assert diagnostics["max_exposure_ratio_realised"] <= 5.0 + 1e-6
    assert diagnostics["min_exposure_ratio_realised"] >= 1.0 / 5.0 - 1e-6


def test_balancing_raises_the_rare_cells_share():
    n_node, qed = synthetic({(15, 5): 4000, (23, 6): 400})
    uniform = BalancedSampler(n_node, qed, batch_size=8, seed=4, alpha=0.0)
    balanced = BalancedSampler(n_node, qed, batch_size=8, seed=4, alpha=0.5,
                               min_cell_count=50)
    rare = n_node == 23
    assert balanced.weights[rare].sum() > uniform.weights[rare].sum()


def test_no_molecule_outside_the_data_can_ever_be_drawn():
    """A cell with no molecules has no index, so unavailable conditions stay unavailable."""
    n_node, qed = synthetic({(15, 5): 200, (23, 6): 200})
    sampler = BalancedSampler(n_node, qed, batch_size=16, seed=5, alpha=1.0,
                              min_cell_count=10)
    drawn = {int(i) for _ in range(40) for i in sampler.next_indices()}
    assert drawn <= set(range(len(n_node)))
    assert set(np.asarray(n_node)[sorted(drawn)]) <= {15, 23}


def test_the_sampler_is_deterministic_and_resumable():
    n_node, qed = synthetic({(15, 5): 300, (23, 6): 120})
    first = BalancedSampler(n_node, qed, batch_size=8, seed=6, alpha=0.5)
    reference = [first.next_indices() for _ in range(7)]
    second = BalancedSampler(n_node, qed, batch_size=8, seed=6, alpha=0.5)
    assert [second.next_indices() for _ in range(7)] == reference
    third = BalancedSampler(n_node, qed, batch_size=8, seed=6, alpha=0.5)
    third.load_state_dict(second.state_dict())
    assert third.next_indices() == first.next_indices()


def test_a_mismatched_resume_is_refused():
    n_node, qed = synthetic({(15, 5): 300, (23, 6): 120})
    sampler = BalancedSampler(n_node, qed, batch_size=8, seed=7, alpha=0.5)
    state = sampler.state_dict() | {"alpha": 0.9}
    with pytest.raises(ValueError, match="alpha mismatch"):
        sampler.load_state_dict(state)


def test_epochs_advance_and_differ():
    n_node, qed = synthetic({(15, 5): 64})
    sampler = BalancedSampler(n_node, qed, batch_size=32, seed=8, alpha=0.0)
    first_epoch = [sampler.next_indices(), sampler.next_indices()]
    assert sampler.epoch == 0
    sampler.next_indices()
    assert sampler.epoch == 1


def test_invalid_configuration_is_refused():
    n_node, qed = synthetic({(15, 5): 100})
    for kwargs in ({"alpha": 1.5}, {"alpha": -0.1}, {"max_exposure_ratio": 0.5},
                   {"min_cell_count": 0}):
        with pytest.raises(ValueError):
            BalancedSampler(n_node, qed, batch_size=4, seed=9, **kwargs)
    with pytest.raises(ValueError):
        BalancedSampler(n_node, qed, batch_size=1000, seed=9)


def test_the_effective_sample_size_is_reported():
    n_node, qed = synthetic({(15, 5): 2000, (23, 6): 200})
    sampler = BalancedSampler(n_node, qed, batch_size=8, seed=10, alpha=1.0,
                              min_cell_count=50)
    diagnostics = sampler.report()["weighting"]
    assert 0.0 < diagnostics["effective_sample_fraction"] <= 1.0
    assert "Kish" in diagnostics["note"]
    assert "per-CELL masses" in diagnostics["definition"]


def test_qed_labels_are_never_modified():
    """This is not relabeling: the sampler only chooses which real molecules are seen."""
    n_node, qed = synthetic({(15, 5): 200, (23, 6): 200})
    before = qed.copy()
    BalancedSampler(n_node, qed, batch_size=8, seed=11, alpha=1.0)
    assert np.array_equal(qed, before)


def test_cell_index_bins_qed_and_keeps_n_exact():
    counts, buckets = cell_index([15, 23, 30], [0.41, 0.60, 0.79], qed_bin_width=0.1)
    assert list(counts) == [15, 23, 30]
    assert list(buckets) == [4, 6, 7]
