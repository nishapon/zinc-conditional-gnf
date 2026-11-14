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
    MAX_COUPLING,
    CoupledNodePrior,
    dense_log_prob,
)
from zinc_gnf.flow import NodePrior  # noqa: E402

DIM = 4


def prior(*, conditional=False, coupling=0.0):
    base = NodePrior(DIM, condition_dim=8 if conditional else 0)
    return CoupledNodePrior.from_node_prior(base, init_coupling=coupling)


def masks(counts, width=None):
    width = width or max(counts)
    mask = torch.zeros((len(counts), width), dtype=torch.bool)
    for row, count in enumerate(counts):
        mask[row, :count] = True
    return mask


# ================================ B: exactness ===================================
@pytest.mark.parametrize("coupling", [0.0, 0.25, 0.7, 1.3])
def test_the_closed_form_density_matches_a_dense_covariance_solve(coupling):
    """The claim the whole experiment rests on."""
    model = prior(coupling=coupling)
    mask = masks([5, 3, 7])
    torch.manual_seed(0)
    z = torch.randn((3, 7, DIM), dtype=torch.float64)
    model = model.double()
    mu, sigma = model.parameters_for(3, None, "cpu")
    closed = model.log_prob(z, mask, None)
    brute = dense_log_prob(z, mask, mu, sigma, model.coupling())
    assert torch.allclose(closed, brute, atol=1e-8), (closed, brute)


def test_zero_coupling_reproduces_the_iid_prior_exactly():
    """Required for the warm start and for the unchanged-training control to be fair."""
    base = NodePrior(DIM, condition_dim=0).double()
    model = CoupledNodePrior.from_node_prior(base, init_coupling=0.0).double()
    mask = masks([6, 4])
    torch.manual_seed(1)
    z = torch.randn((2, 6, DIM), dtype=torch.float64)
    assert torch.allclose(model.log_prob(z, mask, None),
                          base.log_prob(z, mask, None), atol=1e-12)
    assert model.summary()["reduces_to_iid_prior"] is True


def test_the_density_ignores_the_padding_width():
    """A padded column must not change the density, or N would be confounded."""
    model = prior(coupling=0.6).double()
    narrow_mask, wide_mask = masks([5], width=5), masks([5], width=11)
    torch.manual_seed(2)
    core = torch.randn((1, 5, DIM), dtype=torch.float64)
    narrow = torch.zeros((1, 5, DIM), dtype=torch.float64)
    narrow[:, :5] = core
    wide = torch.zeros((1, 11, DIM), dtype=torch.float64)
    wide[:, :5] = core
    wide[:, 5:] = 99.0                      # garbage in the padding
    assert torch.allclose(model.log_prob(narrow, narrow_mask, None),
                          model.log_prob(wide, wide_mask, None), atol=1e-10)


def test_the_density_is_exchangeable_over_nodes():
    model = prior(coupling=0.8).double()
    mask = masks([6])
    torch.manual_seed(3)
    z = torch.randn((1, 6, DIM), dtype=torch.float64)
    permuted = z[:, torch.randperm(6)]
    assert torch.allclose(model.log_prob(z, mask, None),
                          model.log_prob(permuted, mask, None), atol=1e-10)


def test_the_density_depends_on_the_node_count_not_only_the_sample_count():
    """The N-dependent determinant term is what makes this prior size-aware."""
    model = prior(coupling=0.9).double()
    value = torch.full((1, 8, DIM), 0.4, dtype=torch.float64)
    small = model.log_prob(value * masks([3], width=8).unsqueeze(-1),
                           masks([3], width=8), None)
    large = model.log_prob(value * masks([8], width=8).unsqueeze(-1),
                           masks([8], width=8), None)
    assert not torch.allclose(small, large)


def test_the_density_normalises_by_monte_carlo_integration():
    """An independent check that it is a density, not just a formula: sampling from the
    stated generative process and scoring gives a mean log-prob matching the analytic
    entropy of the same Gaussian."""
    model = prior(coupling=0.5).double()
    mask = masks([4] * 4096)
    samples = model.sample(mask, None, temperature=1.0,
                           generator=torch.Generator().manual_seed(5))
    scored = model.log_prob(samples.double(), mask, None)
    mu, sigma = model.parameters_for(1, None, "cpu")
    coupling = model.coupling()
    count = 4
    entropy = 0.0
    for dimension in range(DIM):
        a2 = float(coupling[dimension]) ** 2
        log_det = count * math.log(float(sigma[0, dimension]) ** 2) \
            + math.log(1.0 + count * a2)
        entropy += 0.5 * (count * math.log(2 * math.pi) + log_det + count)
    assert float(scored.mean()) == pytest.approx(-entropy, rel=0.02)


def test_sampling_induces_the_stated_cross_node_correlation():
    """`a` must create correlation between nodes, not just extra variance."""
    model = prior(coupling=1.0).double()
    mask = masks([2] * 20000)
    samples = model.sample(mask, None,
                           generator=torch.Generator().manual_seed(7)).double()
    first, second = samples[:, 0, 0], samples[:, 1, 0]
    correlation = float(torch.corrcoef(torch.stack([first, second]))[0, 1])
    a2 = float(model.coupling()[0]) ** 2
    assert correlation == pytest.approx(a2 / (1.0 + a2), abs=0.03)


def test_zero_coupling_leaves_nodes_uncorrelated():
    model = prior(coupling=0.0).double()
    mask = masks([2] * 20000)
    samples = model.sample(mask, None,
                           generator=torch.Generator().manual_seed(8)).double()
    correlation = float(torch.corrcoef(
        torch.stack([samples[:, 0, 0], samples[:, 1, 0]]))[0, 1])
    assert abs(correlation) < 0.03


def test_local_diversity_is_preserved():
    """Per-node variance must stay sigma^2 * (1 + a^2), not collapse to a shared value."""
    model = prior(coupling=0.8).double()
    mask = masks([3] * 20000)
    samples = model.sample(mask, None,
                           generator=torch.Generator().manual_seed(9)).double()
    within = (samples[:, 0, 0] - samples[:, 1, 0]).var()
    mu, sigma = model.parameters_for(1, None, "cpu")
    assert float(within) == pytest.approx(2.0 * float(sigma[0, 0]) ** 2, rel=0.06)


def test_the_coupling_is_bounded_and_non_negative():
    model = prior(coupling=0.0)
    with torch.no_grad():
        model.pre_coupling.fill_(50.0)
    assert float(model.coupling().max()) <= MAX_COUPLING
    with torch.no_grad():
        model.pre_coupling.fill_(-50.0)
    assert float(model.coupling().min()) >= 0.0


def test_an_out_of_range_initial_coupling_is_refused():
    base = NodePrior(DIM, condition_dim=0)
    for bad in (-0.1, MAX_COUPLING, MAX_COUPLING + 1):
        with pytest.raises(ValueError):
            CoupledNodePrior.from_node_prior(base, init_coupling=bad)


def test_the_density_is_differentiable_in_the_coupling():
    model = prior(coupling=0.3)
    mask = masks([5, 5])
    torch.manual_seed(11)
    z = torch.randn((2, 5, DIM))
    model.log_prob(z, mask, None).sum().backward()
    assert model.pre_coupling.grad is not None
    assert torch.isfinite(model.pre_coupling.grad).all()
    assert float(model.pre_coupling.grad.abs().sum()) > 0.0


def test_the_conditional_pathway_is_reused_unchanged():
    model = prior(conditional=True, coupling=0.4)
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


def test_a_tiny_cell_is_not_upweighted():
    """Upweighting three molecules memorises them; it does not cover their region."""
    n_node, qed = synthetic({(15, 5): 2000, (30, 7): 4})
    sampler = BalancedSampler(n_node, qed, batch_size=8, seed=2, alpha=1.0,
                              min_cell_count=50)
    diagnostics = sampler.report()["weighting"]
    assert diagnostics["cells_held_at_natural_weight"] == 1
    assert diagnostics["molecules_in_held_cells"] == 4
    tiny = sampler.weights[n_node == 30]
    big = sampler.weights[n_node == 15]
    assert float(tiny.max()) <= float(big.max()) * 1.0001


def test_the_weight_ratio_cap_binds():
    n_node, qed = synthetic({(15, 5): 5000, (23, 6): 60})
    sampler = BalancedSampler(n_node, qed, batch_size=8, seed=3, alpha=1.0,
                              min_cell_count=50, max_weight_ratio=5.0)
    assert sampler.report()["weighting"]["realised_weight_ratio"] <= 5.0 + 1e-6


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
    for kwargs in ({"alpha": 1.5}, {"alpha": -0.1}, {"max_weight_ratio": 0.5},
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
