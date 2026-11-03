"""Correctness of the normalizing flow itself: invertibility, log-determinants,
masking, permutation equivariance, prior and conditioning hooks.

These are the properties that make the model a valid density, so they are
checked analytically (against autograd Jacobians) rather than by training.
"""

from __future__ import annotations

import math

import pytest
import torch

from zinc_gnf.flow import (
    ActNorm,
    ConditionEmbedding,
    FlowConfig,
    InvertiblePointwiseLinear,
    NodeFlow,
    NodePrior,
    node_mask_from_counts,
)

TINY = dict(node_embedding_dim=8, num_blocks=2, attention_dim=8, num_heads=2, hidden_dim=16)


def make_flow(**overrides) -> NodeFlow:
    config = FlowConfig(**{**TINY, **overrides})
    flow = NodeFlow(config, init_seed=0)
    dim = config.node_embedding_dim
    generator = torch.Generator().manual_seed(5)
    mean = torch.randn(dim, generator=generator)
    std = torch.rand(dim, generator=generator) + 0.5
    flow.set_data_statistics(mean, std)
    return flow


def make_inputs(counts=(4, 3), dim=8, seed=0):
    mask = node_mask_from_counts(counts)
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((len(counts), mask.shape[1], dim), generator=generator)
    return x * mask.unsqueeze(-1), mask


def initialize(flow: NodeFlow, x: torch.Tensor, mask: torch.Tensor,
               condition: torch.Tensor | None = None) -> None:
    """Run one forward pass in training mode so data-dependent ActNorm initializes."""
    flow.train()
    with torch.no_grad():
        flow(x, mask, condition)
    flow.eval()


# ---------------------------------------------------------------------------
# Invertibility.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("config", [
    {}, {"actnorm": False}, {"invertible_linear": False}, {"whiten": False},
    {"num_blocks": 4}, {"max_log_scale": 0.25}, {"condition_features": ("n",)},
])
def test_forward_then_inverse_returns_the_input(config):
    flow = make_flow(**config)
    x, mask = make_inputs()
    condition = None
    if flow.config.conditional:
        condition = flow.conditioning(torch.tensor([[0.5], [-1.0]]), None, 2, "cpu")
    initialize(flow, x, mask, condition)
    # Break the identity initialization: an untrained flow is the identity map,
    # which would make invertibility trivially true.
    with torch.no_grad():
        for parameter in flow.parameters():
            parameter.add_(0.05 * torch.randn(parameter.shape))
    with torch.no_grad():
        z, _ = flow(x, mask, condition)
        recovered = flow.inverse(z, mask, condition)
    assert torch.allclose(recovered, x, atol=1e-5), float((recovered - x).abs().max())


def test_sample_is_inside_the_models_own_support():
    """`sample` must be exactly `inverse(prior sample)`, with padding left at zero."""
    flow = make_flow()
    mask = node_mask_from_counts([5, 2])
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        embeddings = flow.sample(mask, temperature=1.0, generator=generator)
        z, _ = flow(embeddings, mask)
        back = flow.inverse(z, mask)
    assert torch.allclose(back, embeddings, atol=1e-5)
    assert torch.equal(embeddings[1, 2:], torch.zeros_like(embeddings[1, 2:]))


def test_actnorm_and_invertible_linear_round_trip():
    act_norm, linear = ActNorm(6), InvertiblePointwiseLinear(6, torch.Generator().manual_seed(1))
    x, mask = make_inputs(counts=(3, 2), dim=6, seed=1)
    act_norm.train()
    y, _ = act_norm(x, mask)
    assert torch.allclose(act_norm.inverse(y, mask), x, atol=1e-6)
    with torch.no_grad():
        linear.log_s.add_(0.1 * torch.randn(6))
        linear.lower.add_(0.1 * torch.randn(6, 6))
        linear.upper.add_(0.1 * torch.randn(6, 6))
    y, _ = linear(x, mask)
    assert torch.allclose(linear.inverse(y, mask), x, atol=1e-5)


# ---------------------------------------------------------------------------
# Log-determinants, against autograd.
# ---------------------------------------------------------------------------
def analytic_and_numeric_log_det(flow: NodeFlow, x: torch.Tensor, mask: torch.Tensor,
                                 graph: int) -> tuple[float, float]:
    """(reported log-det, log|det| of the autograd Jacobian) for one graph."""
    nodes = int(mask[graph].sum())
    dim = x.shape[-1]
    single_mask = mask[graph: graph + 1, :nodes]
    single_x = x[graph: graph + 1, :nodes]

    def transform(flat: torch.Tensor) -> torch.Tensor:
        z, _ = flow(flat.view(1, nodes, dim), single_mask)
        return z.reshape(-1)

    jacobian = torch.autograd.functional.jacobian(transform, single_x.reshape(-1), vectorize=True)
    with torch.no_grad():
        _, log_det = flow(single_x, single_mask)
    sign, log_absolute = torch.linalg.slogdet(jacobian)
    assert float(sign) != 0.0, "the transform must be non-degenerate"
    return float(log_det[0]), float(log_absolute)


@pytest.mark.parametrize("config", [
    {}, {"actnorm": False}, {"invertible_linear": False}, {"whiten": False}, {"num_blocks": 3},
])
def test_log_determinant_matches_the_autograd_jacobian(config):
    flow = make_flow(**config)
    x, mask = make_inputs(counts=(4, 2))
    initialize(flow, x, mask)
    with torch.no_grad():                   # make the flow a non-trivial map
        for parameter in flow.parameters():
            parameter.add_(0.1 * torch.randn(parameter.shape))
    for graph in (0, 1):
        reported, numeric = analytic_and_numeric_log_det(flow, x, mask, graph)
        assert reported == pytest.approx(numeric, abs=1e-3), (config, graph, reported, numeric)


def test_whitening_contributes_exactly_minus_log_std_per_node():
    flow = make_flow()
    x, mask = make_inputs()
    initialize(flow, x, mask)
    expected = -torch.log(flow.effective_std).sum() * mask.sum(-1)
    # The blocks see different inputs with and without whitening, so compare only
    # the whitening term itself.
    with torch.no_grad():
        whitened, whitening_log_det = flow.whiten(x, mask)
        _, with_whitening = flow(x, mask)
        flow.config.whiten = False
        _, without = flow(x, mask)
    assert torch.allclose(whitening_log_det, expected, atol=1e-6)
    assert torch.allclose(whitened[mask], ((x - flow.data_mean) / flow.effective_std)[mask], atol=1e-6)
    assert not torch.allclose(with_whitening, without)


def test_log_probability_is_a_normalized_density_in_one_dimension():
    """Numerically integrate exp(log p) for a 1-node, 2-dimensional flow."""
    flow = NodeFlow(FlowConfig(node_embedding_dim=2, num_blocks=1, attention_dim=4,
                               num_heads=1, hidden_dim=8, whiten=False), init_seed=1)
    flow.set_data_statistics(torch.zeros(2), torch.ones(2))
    mask = node_mask_from_counts([1])
    with torch.no_grad():
        for parameter in flow.parameters():
            parameter.add_(0.1 * torch.randn(parameter.shape))
    flow.eval()
    grid = torch.linspace(-6, 6, 121)
    spacing = float(grid[1] - grid[0])
    points = torch.stack(torch.meshgrid(grid, grid, indexing="ij"), dim=-1).reshape(-1, 1, 2)
    with torch.no_grad():
        densities = []
        for start in range(0, points.shape[0], 2048):
            chunk = points[start:start + 2048]
            batch_mask = mask.expand(chunk.shape[0], -1)
            densities.append(flow.log_prob(chunk, batch_mask)["log_prob"].exp())
        integral = float(torch.cat(densities).sum()) * spacing * spacing
    assert integral == pytest.approx(1.0, abs=0.02), integral


# ---------------------------------------------------------------------------
# Masking.
# ---------------------------------------------------------------------------
def test_padding_cannot_influence_real_nodes():
    """A molecule's latent must not depend on what it is batched with."""
    flow = make_flow()
    x, mask = make_inputs(counts=(4, 3))
    initialize(flow, x, mask)
    alone_mask = node_mask_from_counts([3])
    alone_x = x[1:2, :3]
    with torch.no_grad():
        z_batched, log_det_batched = flow(x, mask)
        z_alone, log_det_alone = flow(alone_x, alone_mask)
    assert torch.allclose(z_batched[1, :3], z_alone[0], atol=1e-6)
    assert float(log_det_batched[1]) == pytest.approx(float(log_det_alone[0]), abs=1e-5)


def test_garbage_in_the_padding_is_ignored():
    flow = make_flow()
    x, mask = make_inputs(counts=(4, 2))
    initialize(flow, x, mask)
    polluted = x.clone()
    polluted[1, 2:] = 1e3
    with torch.no_grad():
        clean_z, clean_log_det = flow(x, mask)
        dirty_z, dirty_log_det = flow(polluted, mask)
    assert torch.allclose(clean_z * mask.unsqueeze(-1), dirty_z * mask.unsqueeze(-1), atol=1e-6)
    assert torch.allclose(clean_log_det, dirty_log_det, atol=1e-6)
    assert torch.equal(dirty_z[1, 2:], torch.zeros_like(dirty_z[1, 2:]))


def test_log_probability_scales_with_node_count_not_padded_width():
    flow = make_flow()
    x, mask = make_inputs(counts=(4, 4))
    initialize(flow, x, mask)
    wide_mask = node_mask_from_counts([4, 4], max_nodes=9)
    wide_x = torch.zeros((2, 9, x.shape[-1]))
    wide_x[:, :4] = x
    with torch.no_grad():
        narrow = flow.log_prob(x, mask)["log_prob"]
        wide = flow.log_prob(wide_x, wide_mask)["log_prob"]
    assert torch.allclose(narrow, wide, atol=1e-5)


# ---------------------------------------------------------------------------
# Permutation equivariance.
# ---------------------------------------------------------------------------
def test_permuting_atoms_permutes_the_latent_and_leaves_the_density_alone():
    flow = make_flow()
    x, mask = make_inputs(counts=(5, 5))
    initialize(flow, x, mask)
    with torch.no_grad():
        for parameter in flow.parameters():
            parameter.add_(0.1 * torch.randn(parameter.shape))
    permutation = torch.tensor([3, 0, 4, 1, 2])
    with torch.no_grad():
        z, log_det = flow(x, mask)
        z_permuted, log_det_permuted = flow(x[:, permutation], mask)
        log_probability = flow.log_prob(x, mask)["log_prob"]
        log_probability_permuted = flow.log_prob(x[:, permutation], mask)["log_prob"]
    assert torch.allclose(z_permuted, z[:, permutation], atol=1e-5)
    assert torch.allclose(log_det_permuted, log_det, atol=1e-5)
    assert torch.allclose(log_probability_permuted, log_probability, atol=1e-5)


# ---------------------------------------------------------------------------
# Initialization: the flow starts as the identity with zero log-determinant.
# ---------------------------------------------------------------------------
def test_actnorm_initializes_to_zero_mean_unit_variance_on_real_nodes():
    act_norm = ActNorm(4)
    x, mask = make_inputs(counts=(6, 5), dim=4, seed=7)
    x = x * 3.0 + 2.0
    x = x * mask.unsqueeze(-1)
    act_norm.train()
    y, _ = act_norm(x, mask)
    real = y[mask]
    assert bool(act_norm.initialized)
    assert torch.allclose(real.mean(0), torch.zeros(4), atol=1e-5)
    assert torch.allclose(real.std(0, unbiased=False), torch.ones(4), atol=1e-4)


def test_untrained_flow_is_the_identity_on_whitened_data():
    flow = make_flow()
    x, mask = make_inputs()
    flow.train()
    with torch.no_grad():
        z, log_det = flow(x, mask)
        whitened, whitening_log_det = flow.whiten(x, mask)
    # ActNorm normalizes the whitened batch and the couplings/linear start as
    # identity/permutation, so the only log-det terms are whitening and ActNorm.
    assert torch.isfinite(z).all()
    assert torch.allclose(log_det, whitening_log_det + sum(
        block.act_norm.log_scale.sum() * mask.sum(-1) for block in flow.blocks), atol=1e-5)
    assert float(z.abs().max()) < 20.0
    assert whitened.shape == z.shape


def test_couplings_start_as_the_identity():
    flow = make_flow()
    x, mask = make_inputs()
    block = flow.blocks[0]
    with torch.no_grad():
        for coupling in block.couplings:
            y, log_det = coupling(x, mask, None)
            assert torch.allclose(y, x * mask.unsqueeze(-1), atol=1e-7)
            assert torch.allclose(log_det, torch.zeros(2), atol=1e-7)


def test_invertible_linear_starts_as_a_permutation_with_zero_log_determinant():
    linear = InvertiblePointwiseLinear(6, torch.Generator().manual_seed(2))
    weight = linear.weight()
    assert torch.allclose(weight.sum(0), torch.ones(6))        # permutation matrix
    assert torch.allclose(weight.sum(1), torch.ones(6))
    assert float(linear.log_s.sum().detach()) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Prior.
# ---------------------------------------------------------------------------
def test_prior_log_probability_matches_the_gaussian_formula():
    prior = NodePrior(3, min_sigma=0.1, init_sigma=1.0)
    mask = node_mask_from_counts([2, 1])
    z = torch.tensor([[[0.0, 1.0, -1.0], [0.5, 0.5, 0.5]],
                      [[2.0, 0.0, 0.0], [9.9, 9.9, 9.9]]])     # second row of graph 1 is padding
    with torch.no_grad():
        log_probability = prior.log_prob(z, mask, None)
    mu, sigma = (t.detach() for t in prior.parameters_for(2, None, "cpu"))
    def manual(vector):
        return float(sum(-0.5 * (math.log(2 * math.pi) + 2 * math.log(float(s)) + ((v - m) / s) ** 2)
                         for v, m, s in zip(vector, mu[0], sigma[0])))
    assert float(log_probability[0]) == pytest.approx(manual([0.0, 1.0, -1.0]) + manual([0.5, 0.5, 0.5]), abs=1e-5)
    assert float(log_probability[1]) == pytest.approx(manual([2.0, 0.0, 0.0]), abs=1e-5)


def test_prior_starts_at_the_requested_sigma_and_respects_the_floor():
    for conditional in (0, 5):
        prior = NodePrior(4, condition_dim=conditional, min_sigma=0.1, init_sigma=1.0)
        condition = torch.randn(3, 5) if conditional else None
        with torch.no_grad():
            mu, sigma = prior.parameters_for(3, condition, "cpu")
        assert torch.allclose(sigma, torch.full_like(sigma, 1.0), atol=1e-5)
        assert torch.allclose(mu, torch.zeros_like(mu), atol=1e-6)
        assert float(sigma.min()) >= 0.1
    with pytest.raises(ValueError):
        NodePrior(4, min_sigma=1.0, init_sigma=0.5)


def test_prior_sampling_respects_temperature_and_masks():
    prior = NodePrior(4, min_sigma=0.1, init_sigma=1.0)
    mask = node_mask_from_counts([3, 1])
    cold = prior.sample(mask, None, temperature=0.1, generator=torch.Generator().manual_seed(0))
    hot = prior.sample(mask, None, temperature=2.0, generator=torch.Generator().manual_seed(0))
    assert float(cold[mask].std()) < float(hot[mask].std())
    assert torch.equal(hot[1, 1:], torch.zeros_like(hot[1, 1:]))
    assert torch.allclose(hot[mask], 20.0 * cold[mask], atol=1e-5)      # same noise draw


def test_conditional_prior_requires_and_uses_its_condition():
    prior = NodePrior(4, condition_dim=3, min_sigma=0.1, init_sigma=1.0)
    with pytest.raises(ValueError):
        prior.parameters_for(2, None, "cpu")
    with torch.no_grad():       # zero-initialized heads mean conditioning starts inert
        prior.mu_head.weight.add_(0.5 * torch.randn(4, prior.mu_head.in_features))
    first, _ = prior.parameters_for(1, torch.ones(1, 3), "cpu")
    second, _ = prior.parameters_for(1, -torch.ones(1, 3), "cpu")
    assert not torch.allclose(first, second)


# ---------------------------------------------------------------------------
# Conditioning hooks (built now, exercised by the later conditional stages).
# ---------------------------------------------------------------------------
def test_unconditional_flow_has_no_conditioning_parameters():
    flow = make_flow()
    assert flow.condition_embedding is None
    assert flow.prior.conditional is False
    assert flow.conditioning(None, None, 2, "cpu") is None
    for block in flow.blocks:
        for coupling in block.couplings:
            assert coupling.network.film is None


def test_conditional_flow_wires_film_and_a_conditional_prior():
    flow = make_flow(condition_features=("n", "qed"))
    assert flow.condition_embedding is not None and flow.prior.conditional
    for block in flow.blocks:
        for coupling in block.couplings:
            assert coupling.network.film is not None
    values = torch.tensor([[0.3, -0.2], [1.0, 0.8]])
    embedding = flow.conditioning(values, None, 2, "cpu")
    assert embedding.shape == (2, flow.config.condition_embedding_dim)
    x, mask = make_inputs()
    initialize(flow, x, mask, embedding)
    with torch.no_grad():       # FiLM is zero-initialized: inert until trained
        z_a, _ = flow(x, mask, embedding)
        z_b, _ = flow(x, mask, flow.conditioning(values * -1, None, 2, "cpu"))
    assert torch.allclose(z_a, z_b, atol=1e-6)
    for block in flow.blocks:   # once FiLM has weights, conditioning changes the map
        for coupling in block.couplings:
            with torch.no_grad():
                coupling.network.film.weight.add_(0.3 * torch.randn(coupling.network.film.weight.shape))
                coupling.network.scale_head.weight.add_(0.1 * torch.randn(
                    coupling.network.scale_head.weight.shape))
    with torch.no_grad():
        z_c, _ = flow(x, mask, embedding)
    assert not torch.allclose(z_a, z_c, atol=1e-4)


def test_condition_embedding_uses_a_learned_token_for_absent_features():
    embedding = ConditionEmbedding(("n", "qed"), 6, 8)
    values = torch.tensor([[0.5, 10.0], [0.5, -10.0]])
    absent_qed = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    out = embedding(values, absent_qed)
    assert torch.allclose(out[0], out[1], atol=1e-6)       # QED ignored when absent
    present = embedding(values, torch.ones_like(values))
    assert not torch.allclose(present[0], present[1], atol=1e-4)
    with pytest.raises(ValueError):
        embedding(torch.zeros(2, 1), torch.ones(2, 1))


def test_conditional_flow_refuses_to_run_without_a_condition():
    flow = make_flow(condition_features=("n",))
    x, mask = make_inputs()
    with pytest.raises(ValueError):
        flow(x, mask, None)
    with pytest.raises(ValueError):
        flow.conditioning(None, None, 2, "cpu")


# ---------------------------------------------------------------------------
# Configuration and masks.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [
    {"node_embedding_dim": 7}, {"num_blocks": 0}, {"attention_dim": 7, "num_heads": 2},
    {"condition_features": ("mass",)}, {"condition_features": ("n", "n")},
    {"max_log_scale": 0.0}, {"noise_sigma": -0.1},
])
def test_configuration_rejects_impossible_settings(bad):
    with pytest.raises(ValueError):
        FlowConfig(**bad)


def test_configuration_round_trips_through_a_dictionary():
    config = FlowConfig(condition_features=("n",), num_blocks=3)
    restored = FlowConfig.from_dict(config.to_dict())
    assert restored == config
    assert restored.condition_features == ("n",)
    with pytest.raises(ValueError):
        FlowConfig.from_dict({**config.to_dict(), "nonsense": 1})


def test_node_mask_from_counts():
    mask = node_mask_from_counts([3, 1], max_nodes=4)
    assert mask.tolist() == [[True, True, True, False], [True, False, False, False]]
    with pytest.raises(ValueError):
        node_mask_from_counts([0])
    with pytest.raises(ValueError):
        node_mask_from_counts([5], max_nodes=4)
    with pytest.raises(ValueError):
        node_mask_from_counts([])


def test_statistics_must_be_set_and_sane():
    flow = NodeFlow(FlowConfig(**TINY))
    assert bool(flow.statistics_set) is False
    with pytest.raises(ValueError):
        flow.set_data_statistics(torch.zeros(8), -torch.ones(8))           # negative std
    with pytest.raises(ValueError):
        flow.set_data_statistics(torch.zeros(8), torch.full((8,), float("nan")))
    with pytest.raises(ValueError):
        flow.set_data_statistics(torch.zeros(3), torch.ones(3))           # wrong shape
    flow.set_data_statistics(torch.zeros(8), torch.full((8,), 0.001))
    assert bool(flow.statistics_set)
    assert flow.floored_dimensions() == 8                                 # all below the floor
    assert float(flow.effective_std.min()) == pytest.approx(flow.config.whiten_std_floor)


def test_a_constant_embedding_dimension_is_handled_not_rejected():
    """A dead encoder unit (exactly zero variance) must not abort a run."""
    flow = NodeFlow(FlowConfig(**TINY))
    std = torch.zeros(8)
    std[:3] = torch.tensor([0.4, 0.2, 0.9])                    # only 3 of 8 dimensions vary
    mean = torch.arange(8, dtype=torch.float32)
    flow.set_data_statistics(mean, std)
    assert bool(flow.statistics_set) and flow.floored_dimensions() == 5
    x, mask = make_inputs(dim=8)
    x = (x * std + mean) * mask.unsqueeze(-1)                  # data constant in 5 dimensions
    flow.train()
    with torch.no_grad():
        whitened, log_det = flow.whiten(x, mask)
        parts = flow.log_prob(x, mask)
    assert torch.isfinite(whitened).all() and torch.isfinite(log_det).all()
    assert torch.isfinite(parts["log_prob"]).all() and torch.isfinite(parts["z"]).all()
    # the constant dimensions still receive smoothing noise, so the density is not a point mass
    noisy = flow.add_smoothing_noise(x, mask, generator=torch.Generator().manual_seed(0))
    constant_part = (noisy - x)[mask][:, 3:]
    assert float(constant_part.abs().min()) > 0.0
    expected = flow.config.noise_sigma * flow.config.whiten_std_floor
    assert float(constant_part.abs().max()) < 10 * expected


def test_smoothing_noise_is_masked_and_scaled_by_the_data_std():
    flow = make_flow(noise_sigma=0.5)
    x, mask = make_inputs()
    noisy = flow.add_smoothing_noise(x, mask, generator=torch.Generator().manual_seed(0))
    assert torch.equal(noisy[1, 3:], x[1, 3:])                 # padding untouched
    assert not torch.allclose(noisy[mask], x[mask])
    zero_noise = make_flow(noise_sigma=0.0)
    assert torch.equal(zero_noise.add_smoothing_noise(x, mask), x)


def test_shape_mismatch_is_reported_clearly():
    flow = make_flow()
    x, mask = make_inputs()
    with pytest.raises(ValueError, match="do not match"):
        flow(x[:, :2], mask)
