"""Synthetic validation of the three-way likelihood-gap decomposition.

The decomposition is an algebraic identity, so it must hold EXACTLY (to round-off) for
any flow, trained or not. These tests pin that, pin the direction of each term on a
hand-built case where the answer is known analytically, and pin the route-separation
logic of the geometry section.

Requires torch + rdkit (the cluster env).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.test_conditional_flow import conditional_flow, wake_up_conditioning
from zinc_gnf.conditioning import condition_embedding_for
from zinc_gnf.flow import node_mask_from_counts

ROOT = Path(__file__).resolve().parents[1]
DIM = 8


def _script():
    spec = importlib.util.spec_from_file_location(
        "qed_gap_decomposition", ROOT / "scripts" / "30_qed_gap_decomposition.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _setup(seed=0, n=(5, 6, 7)):
    flow = wake_up_conditioning(conditional_flow(features=("n", "qed"), dimension=DIM,
                                                 seed=seed)).double()
    flow.eval()
    counts = list(n)
    mask = node_mask_from_counts(counts, max_nodes=max(counts))
    torch.manual_seed(seed + 1)
    x = (torch.randn(len(counts), mask.shape[1], DIM, dtype=torch.float64)
         * mask.unsqueeze(-1))
    values = torch.tensor([[0.3, 0.5], [-0.2, -0.4], [0.1, 0.8]], dtype=torch.float64)
    return flow, x, mask, values


def _terms(flow, x, mask, values, alt):
    present = torch.ones_like(values)
    b = values.shape[0]
    cond_t = condition_embedding_for(flow, values, present, b, "cpu")
    cond_a = condition_embedding_for(flow, alt, present, b, "cpu")
    z_t, ld_t = flow(x, mask, cond_t)
    z_a, ld_a = flow(x, mask, cond_a)
    lp_t_zt = flow.prior.log_prob(z_t, mask, cond_t)
    lp_t_za = flow.prior.log_prob(z_a, mask, cond_t)
    lp_a_za = flow.prior.log_prob(z_a, mask, cond_a)
    return {
        "d_z": lp_t_zt - lp_t_za,
        "d_prior": lp_t_za - lp_a_za,
        "d_logdet": ld_t - ld_a,
        "d_total": (lp_t_zt + ld_t) - (lp_a_za + ld_a),
    }


# ------------------------------------------------------------------ the identity ---
def test_decomposition_is_exact_to_roundoff():
    flow, x, mask, values = _setup()
    alt = values.clone()
    alt[:, 1] = alt[:, 1] + 1.0
    t = _terms(flow, x, mask, values, alt)
    residual = (t["d_total"] - (t["d_z"] + t["d_prior"] + t["d_logdet"])).abs().max()
    assert float(residual) < 1e-9, f"identity violated by {float(residual):.3e}"


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("offset", [-2.0, -0.5, 0.5, 3.0])
def test_identity_holds_for_many_seeds_and_offsets(seed, offset):
    flow, x, mask, values = _setup(seed=seed)
    alt = values.clone()
    alt[:, 1] = alt[:, 1] + offset
    t = _terms(flow, x, mask, values, alt)
    residual = (t["d_total"] - (t["d_z"] + t["d_prior"] + t["d_logdet"])).abs().max()
    assert float(residual) < 1e-9


def test_identity_holds_under_permutation_protocol():
    flow, x, mask, values = _setup()
    alt = values.clone()
    alt[:, 1] = values[[1, 2, 0], 1]        # QED permuted, N kept
    t = _terms(flow, x, mask, values, alt)
    residual = (t["d_total"] - (t["d_z"] + t["d_prior"] + t["d_logdet"])).abs().max()
    assert float(residual) < 1e-9
    # N column must be untouched by the QED-only protocol
    assert torch.allclose(alt[:, 0], values[:, 0])


def test_zero_perturbation_gives_exactly_zero_everywhere():
    flow, x, mask, values = _setup()
    t = _terms(flow, x, mask, values, values.clone())
    for key, value in t.items():
        assert float(value.abs().max()) < 1e-12, key


# --------------------------------------------------- which term carries what ---
def test_dead_conditioning_puts_everything_at_zero():
    """A zero-initialised conditional flow is the unconditional map: all terms vanish."""
    flow = conditional_flow(features=("n", "qed"), dimension=DIM).double()
    flow.eval()
    counts = [5, 6, 7]
    mask = node_mask_from_counts(counts, max_nodes=7)
    torch.manual_seed(3)
    x = torch.randn(3, 7, DIM, dtype=torch.float64) * mask.unsqueeze(-1)
    values = torch.tensor([[0.3, 0.5], [-0.2, -0.4], [0.1, 0.8]], dtype=torch.float64)
    alt = values.clone()
    alt[:, 1] += 1.0
    t = _terms(flow, x, mask, values, alt)
    for key, value in t.items():
        assert float(value.abs().max()) < 1e-10, f"{key} moved on a dead conditioner"


def test_prior_only_conditioning_isolates_d_prior():
    """Zero the FiLM but keep the prior heads: z cannot move, so d_z = d_logdet = 0."""
    flow, x, mask, values = _setup()
    with torch.no_grad():
        for name, parameter in flow.named_parameters():
            if "film" in name:
                parameter.zero_()
    alt = values.clone()
    alt[:, 1] += 1.0
    t = _terms(flow, x, mask, values, alt)
    assert float(t["d_z"].abs().max()) < 1e-10        # z is condition-independent now
    assert float(t["d_logdet"].abs().max()) < 1e-10   # so is the volume
    assert float(t["d_prior"].abs().max()) > 0.0      # the prior still responds
    residual = (t["d_total"] - t["d_prior"]).abs().max()
    assert float(residual) < 1e-9


def test_frozen_prior_isolates_the_flow_channels():
    """Zero the prior heads but keep FiLM: d_prior must vanish, d_z/d_logdet need not."""
    flow, x, mask, values = _setup()
    with torch.no_grad():
        flow.prior.mu_head.weight.zero_()
        flow.prior.mu_head.bias.zero_()
        flow.prior.sigma_head.weight.zero_()
    alt = values.clone()
    alt[:, 1] += 1.0
    t = _terms(flow, x, mask, values, alt)
    assert float(t["d_prior"].abs().max()) < 1e-10
    residual = (t["d_total"] - (t["d_z"] + t["d_logdet"])).abs().max()
    assert float(residual) < 1e-9


def test_terms_are_signed_and_can_cancel():
    """The report must not use percentages: terms genuinely take both signs."""
    flow, x, mask, values = _setup(seed=5)
    alt = values.clone()
    alt[:, 1] += 2.0
    t = _terms(flow, x, mask, values, alt)
    absolute = sum(float(t[k].abs().mean()) for k in ("d_z", "d_prior", "d_logdet"))
    assert absolute > abs(float(t["d_total"].mean())) - 1e-12


# ------------------------------------------------------------ geometry routes ---
def test_route_separation_recovers_the_full_delta_when_the_map_is_condition_free():
    """With FiLM zeroed, f^-1 does not depend on c, so the map route must be exactly 0
    and the prior route must equal the full displacement."""
    flow, _, _, _ = _setup()
    with torch.no_grad():
        for name, parameter in flow.named_parameters():
            if "film" in name:
                parameter.zero_()
    samples, n = 4, 6
    mask = torch.ones((samples, n), dtype=torch.bool)
    generator = torch.Generator().manual_seed(11)
    eps = torch.randn((samples, n, DIM), generator=generator, dtype=torch.float64)
    low = torch.tensor([[0.2, -0.8]] * samples, dtype=torch.float64)
    high = torch.tensor([[0.2, +0.8]] * samples, dtype=torch.float64)
    c_lo = condition_embedding_for(flow, low, torch.ones_like(low), samples, "cpu")
    c_hi = condition_embedding_for(flow, high, torch.ones_like(high), samples, "cpu")
    with torch.no_grad():
        mu_l, s_l = flow.prior.parameters_for(samples, c_lo, "cpu")
        mu_h, s_h = flow.prior.parameters_for(samples, c_hi, "cpu")
        z_lo = mu_l.unsqueeze(1) + s_l.unsqueeze(1) * eps
        z_hi = mu_h.unsqueeze(1) + s_h.unsqueeze(1) * eps
        full = flow.inverse(z_hi, mask, c_hi) - flow.inverse(z_lo, mask, c_lo)
        prior_route = flow.inverse(z_hi, mask, c_lo) - flow.inverse(z_lo, mask, c_lo)
        map_route = flow.inverse(z_lo, mask, c_hi) - flow.inverse(z_lo, mask, c_lo)
    assert float(map_route.abs().max()) < 1e-10
    assert float((full - prior_route).abs().max()) < 1e-10


def test_bootstrap_ci_brackets_a_known_mean():
    module = _script()
    rng = np.random.default_rng(0)
    values = rng.normal(0.5, 0.1, size=400)
    result = module.bootstrap_ci(values)
    assert result["ci95"][0] < 0.5 < result["ci95"][1]
    assert result["excludes_zero"] is True
    null = module.bootstrap_ci(rng.normal(0.0, 1.0, size=400))
    assert null["excludes_zero"] is False


def test_bootstrap_ci_handles_degenerate_input():
    module = _script()
    assert module.bootstrap_ci(np.array([])) ["n"] == 0
    assert module.bootstrap_ci(np.array([1.0]))["n"] == 1
    mixed = module.bootstrap_ci(np.array([1.0, np.nan, 2.0, np.inf]))
    assert mixed["n"] == 2
