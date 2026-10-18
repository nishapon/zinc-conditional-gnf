"""Arm B: the node-autoregressive coupling.

Checks the four properties that make it a valid flow (exact invertibility through the
sequential inverse, analytic log-determinant, causality, padding invariance), the two
that make it a clean experiment (Arm B starts as exactly Arm A, and its shared weights
are identical to Arm A's for the same seed), and the consequence we must document
(the model is node-order dependent and therefore no longer exchangeable).
"""

from __future__ import annotations

import pytest
import torch

from zinc_gnf.flow import (
    FlowConfig,
    NodeAutoregressiveCoupling,
    NodeFlow,
    real_node_reversal_index,
    node_mask_from_counts,
)

SMALL = dict(node_embedding_dim=8, num_blocks=2, attention_dim=8, num_heads=2, hidden_dim=16)


def build(node_autoregressive: bool, *, seed: int = 7, **overrides) -> NodeFlow:
    torch.manual_seed(seed)
    config = FlowConfig(**{**SMALL, **overrides, "node_autoregressive": node_autoregressive})
    flow = NodeFlow(config, init_seed=0)
    flow.set_data_statistics(torch.zeros(config.node_embedding_dim),
                             torch.ones(config.node_embedding_dim))
    return flow


def inputs(counts=(5, 3), dim=8, seed=0):
    mask = node_mask_from_counts(counts)
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((len(counts), mask.shape[1], dim), generator=generator)
    return x * mask.unsqueeze(-1), mask


def activate(flow: NodeFlow, x, mask, *, noise: float = 0.05, seed: int = 0) -> None:
    """Initialize ActNorm and move the flow away from its identity initialization."""
    flow.train()
    with torch.no_grad():
        flow(x, mask)
    flow.eval()
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in flow.parameters():
            parameter.add_(noise * torch.randn(parameter.shape, generator=generator))


# ---------------------------------------------------------------------------
# Invertibility and the log-determinant.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("counts", [(5, 3), (1, 4), (7, 7), (2,)])
def test_sequential_inverse_recovers_the_input(counts):
    flow = build(True)
    x, mask = inputs(counts)
    activate(flow, x, mask)
    with torch.no_grad():
        z, _ = flow(x, mask)
        recovered = flow.inverse(z, mask)
    assert torch.allclose(recovered, x, atol=1e-5), float((recovered - x).abs().max())


def test_log_determinant_matches_the_autograd_jacobian():
    flow = build(True)
    x, mask = inputs((4, 2))
    activate(flow, x, mask)
    for graph in (0, 1):
        nodes = int(mask[graph].sum())
        single_mask, single_x = mask[graph:graph + 1, :nodes], x[graph:graph + 1, :nodes]

        def transform(flat):
            out, _ = flow(flat.view(1, nodes, 8), single_mask)
            return out.reshape(-1)

        jacobian = torch.autograd.functional.jacobian(transform, single_x.reshape(-1),
                                                      vectorize=True)
        with torch.no_grad():
            _, reported = flow(single_x, single_mask)
        sign, numeric = torch.linalg.slogdet(jacobian.double())
        assert float(sign) != 0.0
        assert float(reported[0]) == pytest.approx(float(numeric), abs=1e-3)


def test_the_coupling_alone_round_trips_and_reports_its_own_log_determinant():
    config = FlowConfig(**SMALL, node_autoregressive=True)
    coupling = NodeAutoregressiveCoupling(config, reverse=False)
    x, mask = inputs((5, 3))
    with torch.no_grad():       # zero-initialized heads: identity with zero log-det
        y, log_det = coupling(x, mask, None)
        assert torch.allclose(y, x, atol=1e-7)
        assert torch.allclose(log_det, torch.zeros(2), atol=1e-7)
        for parameter in coupling.parameters():
            parameter.add_(0.1 * torch.randn(parameter.shape))
        y, log_det = coupling(x, mask, None)
        assert torch.allclose(coupling.inverse(y, mask, None), x, atol=1e-5)
    scale, _ = coupling._scale_and_translation(x, mask, None)
    assert torch.allclose(log_det, (scale * mask.unsqueeze(-1)).sum(dim=(1, 2)), atol=1e-6)


# ---------------------------------------------------------------------------
# Causality.
# ---------------------------------------------------------------------------
def test_the_conditioner_cannot_see_the_node_it_transforms():
    """(s_i, t_i) must be a function of x_{<i} only - checked by gradient."""
    config = FlowConfig(**SMALL, node_autoregressive=True)
    coupling = NodeAutoregressiveCoupling(config, reverse=False)
    with torch.no_grad():
        for parameter in coupling.parameters():
            parameter.add_(0.2 * torch.randn(parameter.shape))
    mask = node_mask_from_counts([6])
    x = torch.randn((1, 6, 8), requires_grad=True)
    scale, translation = coupling._scale_and_translation(x, mask, None)
    for position in range(6):
        for output in (scale, translation):
            gradient = torch.autograd.grad(output[0, position].sum(), x, retain_graph=True)[0]
            # strictly earlier nodes may matter; node `position` and later must not
            assert float(gradient[0, position:].abs().max()) == 0.0, position
            if position > 0:
                assert float(gradient[0, :position].abs().max()) > 0.0, position


def test_the_transformation_is_triangular_in_the_node_index():
    config = FlowConfig(**SMALL, node_autoregressive=True)
    coupling = NodeAutoregressiveCoupling(config, reverse=False)
    with torch.no_grad():
        for parameter in coupling.parameters():
            parameter.add_(0.2 * torch.randn(parameter.shape))
    mask = node_mask_from_counts([5])
    x = torch.randn((1, 5, 8), requires_grad=True)
    y, _ = coupling(x, mask, None)
    for position in range(5):
        gradient = torch.autograd.grad(y[0, position].sum(), x, retain_graph=True)[0]
        later = gradient[0, position + 1:]
        if later.numel():
            assert float(later.abs().max()) == 0.0                       # no later node
        assert float(gradient[0, position].abs().max()) > 0.0            # its own, scaled


def test_reversed_blocks_look_the_other_way_and_still_invert():
    config = FlowConfig(**SMALL, node_autoregressive=True)
    coupling = NodeAutoregressiveCoupling(config, reverse=True)
    with torch.no_grad():
        for parameter in coupling.parameters():
            parameter.add_(0.2 * torch.randn(parameter.shape))
    mask = node_mask_from_counts([5])
    x = torch.randn((1, 5, 8), requires_grad=True)
    y, _ = coupling(x, mask, None)
    gradient = torch.autograd.grad(y[0, 0].sum(), x, retain_graph=True)[0]
    assert float(gradient[0, 1:].abs().max()) > 0.0      # the FIRST node now sees later ones
    with torch.no_grad():
        y, _ = coupling(x.detach(), mask, None)
        assert torch.allclose(coupling.inverse(y, mask, None), x.detach(), atol=1e-5)


def test_real_node_reversal_index_is_an_involution_that_spares_padding():
    mask = node_mask_from_counts([4, 2, 1], max_nodes=4)
    index = real_node_reversal_index(mask)
    assert index[0].tolist() == [3, 2, 1, 0]
    assert index[1].tolist() == [1, 0, 2, 3]            # padding positions stay put
    assert index[2].tolist() == [0, 1, 2, 3]
    twice = torch.gather(index, 1, index)
    assert twice.tolist() == [[0, 1, 2, 3]] * 3


# ---------------------------------------------------------------------------
# Padding.
# ---------------------------------------------------------------------------
def test_padding_cannot_influence_real_nodes_or_the_log_determinant():
    flow = build(True)
    x, mask = inputs((5, 2))
    activate(flow, x, mask)
    polluted = x.clone()
    polluted[1, 2:] = 1e3
    with torch.no_grad():
        clean_z, clean_log_det = flow(x, mask)
        dirty_z, dirty_log_det = flow(polluted, mask)
    assert torch.allclose(clean_z * mask.unsqueeze(-1), dirty_z * mask.unsqueeze(-1), atol=1e-6)
    assert torch.allclose(clean_log_det, dirty_log_det, atol=1e-6)
    assert torch.equal(dirty_z[1, 2:], torch.zeros_like(dirty_z[1, 2:]))


def test_a_molecule_is_unaffected_by_what_it_is_batched_with():
    flow = build(True)
    x, mask = inputs((6, 3))
    activate(flow, x, mask)
    alone_mask, alone_x = node_mask_from_counts([3]), x[1:2, :3]
    with torch.no_grad():
        batched_z, batched_log_det = flow(x, mask)
        alone_z, alone_log_det = flow(alone_x, alone_mask)
    assert torch.allclose(batched_z[1, :3], alone_z[0], atol=1e-6)
    assert float(batched_log_det[1]) == pytest.approx(float(alone_log_det[0]), abs=1e-5)


# ---------------------------------------------------------------------------
# Arm B starts as Arm A, and shares its weights.
# ---------------------------------------------------------------------------
def test_arm_b_is_arm_a_at_initialization():
    arm_a, arm_b = build(False), build(True)
    x, mask = inputs((5, 3))
    for flow in (arm_a, arm_b):
        flow.train()
        with torch.no_grad():
            flow(x, mask)
        flow.eval()
    with torch.no_grad():
        z_a, log_det_a = arm_a(x, mask)
        z_b, log_det_b = arm_b(x, mask)
        probability_a = arm_a.log_prob(x, mask)["log_prob"]
        probability_b = arm_b.log_prob(x, mask)["log_prob"]
    assert torch.allclose(z_a, z_b, atol=1e-7)
    assert torch.allclose(log_det_a, log_det_b, atol=1e-7)
    assert torch.allclose(probability_a, probability_b, atol=1e-6)


def test_shared_weights_are_identical_so_the_arms_are_comparable():
    """Random-number accounting: the new couplings must be built AFTER everything else."""
    arm_a, arm_b = build(False, seed=11), build(True, seed=11)
    state_a, state_b = arm_a.state_dict(), arm_b.state_dict()
    assert set(state_a) < set(state_b)
    for key, value in state_a.items():
        assert torch.equal(value, state_b[key]), key
    extra = sorted(set(state_b) - set(state_a))
    assert extra and all("node_coupling" in key for key in extra)
    # one coupling per block, every other one reversed
    assert [block.node_coupling.reverse for block in arm_b.blocks] == [False, True]
    assert all(block.node_coupling is None for block in arm_a.blocks)


def test_arm_b_has_more_parameters_and_reports_it():
    arm_a, arm_b = build(False), build(True)
    assert arm_b.num_parameters() > arm_a.num_parameters()
    assert arm_a.config.node_autoregressive is False
    assert arm_b.config.node_autoregressive is True


# ---------------------------------------------------------------------------
# Order dependence: the documented cost of the mechanism.
# ---------------------------------------------------------------------------
def test_arm_a_is_permutation_equivariant_and_arm_b_is_not():
    permutation = torch.tensor([3, 0, 4, 1, 2])
    x, mask = inputs((5, 5))
    arm_a, arm_b = build(False), build(True)
    for flow in (arm_a, arm_b):
        activate(flow, x, mask)
    with torch.no_grad():
        z_a, _ = arm_a(x, mask)
        z_a_permuted, _ = arm_a(x[:, permutation], mask)
        probability_b = arm_b.log_prob(x, mask)["log_prob"]
        probability_b_permuted = arm_b.log_prob(x[:, permutation], mask)["log_prob"]
    assert torch.allclose(z_a_permuted, z_a[:, permutation], atol=1e-5)
    # Arm B models ORDERED node tuples, so permuting the atoms changes the density.
    assert not torch.allclose(probability_b_permuted, probability_b, atol=1e-3)


def test_the_module_documents_its_order_dependence():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "zinc_gnf" / "flow.py").read_text()
    assert "NODE-ORDER DEPENDENT" in source
    assert "no longer exchangeable" in source


# ---------------------------------------------------------------------------
# Conditioning, sampling and checkpoints.
# ---------------------------------------------------------------------------
def test_the_conditional_film_path_works_in_the_new_coupling():
    flow = build(True, condition_features=("n", "qed"))
    values = torch.tensor([[0.3, -0.2], [1.0, 0.8]])
    embedding = flow.conditioning(values, None, 2, "cpu")
    x, mask = inputs((5, 3))
    flow.train()
    with torch.no_grad():
        flow(x, mask, embedding)
    flow.eval()
    with pytest.raises(ValueError):
        flow(x, mask, None)
    with torch.no_grad():
        before, _ = flow(x, mask, embedding)
        for block in flow.blocks:
            coupling = block.node_coupling
            coupling.film.weight.add_(0.3 * torch.randn(coupling.film.weight.shape))
            coupling.scale_head.weight.add_(0.1 * torch.randn(coupling.scale_head.weight.shape))
        after, _ = flow(x, mask, embedding)
    assert not torch.allclose(before, after, atol=1e-4)


def test_sampling_is_reproducible_and_stays_inside_the_model():
    flow = build(True)
    mask = node_mask_from_counts([5, 2])
    x, train_mask = inputs((5, 2))
    activate(flow, x, train_mask)
    with torch.no_grad():
        first, latents = flow.sample(mask, temperature=0.9,
                                     generator=torch.Generator().manual_seed(3),
                                     return_latent=True)
        again = flow.sample(mask, temperature=0.9, generator=torch.Generator().manual_seed(3))
        recovered, _ = flow(first, mask)
    assert torch.allclose(first, again)
    assert torch.allclose(recovered, latents, atol=1e-5)
    assert torch.equal(first[1, 2:], torch.zeros_like(first[1, 2:]))


def test_checkpoint_round_trip_preserves_the_new_couplings(tmp_path):
    from zinc_gnf.constants_charge_aware import vocabulary_fingerprint
    from zinc_gnf.trainer_flow import load_flow

    flow = build(True)
    x, mask = inputs((5, 3))
    activate(flow, x, mask)
    path = tmp_path / "arm_b.pt"
    torch.save({"model_state_dict": flow.state_dict(), "flow_config": flow.config.to_dict(),
                "vocabulary_fingerprint": vocabulary_fingerprint(), "step": 10}, path)
    restored, probe = load_flow(path, device="cpu")
    assert probe["flow_config"]["node_autoregressive"] is True
    assert all(block.node_coupling is not None for block in restored.blocks)
    with torch.no_grad():
        assert torch.allclose(restored.log_prob(x, mask)["log_prob"],
                              flow.log_prob(x, mask)["log_prob"], atol=1e-6)


def test_configuration_round_trip_and_default():
    assert FlowConfig().node_autoregressive is False
    config = FlowConfig(node_autoregressive=True)
    assert FlowConfig.from_dict(config.to_dict()) == config
