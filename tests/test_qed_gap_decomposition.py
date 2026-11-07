"""Synthetic validation of the three-way likelihood-gap decomposition.

The decomposition is an algebraic identity, so it must hold EXACTLY (to round-off) for
any flow, trained or not. These tests pin that, pin the direction of each term on a
hand-built case where the answer is known analytically, and pin the route-separation
logic of the geometry section.

Requires torch + rdkit (the cluster env).
"""

from __future__ import annotations

import importlib.util
import math
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


# ============================================================================
# Regression tests for job 366591: the final-offset failure
# (worst absolute residual 1.638e+04 at a fixed 1e-5 bound).
#
# Cause: the 'offset' protocol pushed standardised QED outside the train support, the
# flow extrapolated (its coupling TRANSLATIONS are unclamped), |z_a| reached ~1e9, the
# Gaussian log-densities reached ~1e19, and d_prior = log p_t(z_a) - log p_a(z_a)
# subtracted two numbers of that magnitude. float64 round-off on 1e19 is ~2e3, so a
# residual of 1.6e4 is arithmetic precision, not an algebraic error. These tests pin
# that reading and pin the corrections.
# ============================================================================


def test_identity_is_exact_at_extreme_magnitude_in_exact_scaling():
    """The algebra survives huge latents: the residual tracks the TERM scale, not a
    fixed bound. Reproduces the failure mode in miniature."""
    flow, x, mask, values = _setup()
    alt = values.clone()
    alt[:, 1] += 1.0
    # drive the latent to a huge magnitude the way an out-of-support condition does
    big = x * 1e6
    t = _terms(flow, big, mask, values, alt)
    scale = max(float(t[k].abs().max()) for k in ("d_z", "d_prior", "d_logdet"))
    residual = float((t["d_total"] - (t["d_z"] + t["d_prior"] + t["d_logdet"])).abs().max())
    # absolute residual grows with scale; RELATIVE residual stays at machine precision
    assert residual / max(scale, 1.0) < 1e-11, (
        f"relative residual {residual / max(scale, 1.0):.3e} at term scale {scale:.3e}")


def test_relative_criterion_accepts_roundoff_and_rejects_algebraic_error():
    """The backward-error bound must not be a blanket loosening."""
    module = _script()
    rel, floor = module.IDENTITY_RELATIVE_TOLERANCE, module.IDENTITY_ABSOLUTE_FLOOR
    # the four residuals actually observed in job 366591 on term scale ~1e4
    for observed in (4.55e-13, 9.09e-13, 1.82e-12):
        allowed = max(floor, 1.0e4 * rel)
        assert observed <= allowed, f"{observed:.2e} wrongly rejected"
    # the failing case at its true term scale is round-off and is accepted
    assert 1.638e4 <= max(floor, 9.2e18 * rel)
    # an algebraic error is O(1) RELATIVE and is still rejected at every scale
    for scale in (1e4, 1e12, 1e19):
        bogus = 0.01 * scale
        assert bogus > max(floor, scale * rel), f"algebraic error hidden at scale {scale:.0e}"


def test_nan_safe_max_cannot_drop_a_nan():
    """Python's max(0.0, nan) returns 0.0, which silently hid non-finite residuals."""
    module = _script()
    assert max(0.0, float("nan")) == 0.0          # the original defect
    assert math.isnan(module.nan_safe_max(0.0, float("nan")))
    assert math.isnan(module.nan_safe_max(float("nan"), 1.0))
    assert module.nan_safe_max(1.0, 2.0) == 2.0
    assert module.nan_safe_max(3.0, 2.0) == 3.0


def test_grid_protocol_stays_inside_train_support():
    """The generation-matched protocol must never leave support, by construction."""
    module = _script()
    _, _, _, values = _setup()
    support = (-5.28, 1.94)                        # ZINC standardised QED range
    grid = [-2.396, -0.952, 0.492]                 # raw 0.4 / 0.6 / 0.8 standardised
    rng = np.random.default_rng(0)
    alt, in_support, clamped = module.build_alternative(
        values, 1, "grid", rng=rng, qed_offset=0.0, grid_std=grid,
        support=support, clamp=True)
    assert bool(in_support.all()), "grid protocol left the train support"
    assert not bool(clamped.any())
    # N column untouched, and each alternative is an actual grid value
    assert torch.allclose(alt[:, 0], values[:, 0])
    for v in alt[:, 1].tolist():
        assert min(abs(v - g) for g in grid) < 1e-6
    # and it picks the FARTHEST grid target, maximising the contrast
    for i, true_q in enumerate(values[:, 1].tolist()):
        best = max(grid, key=lambda g: abs(g - true_q))
        assert abs(float(alt[i, 1]) - best) < 1e-6


def test_offset_protocol_is_flagged_out_of_support_and_clamped():
    module = _script()
    support = (-5.28, 1.94)
    # a molecule at the very top of the range, offset +1 -> 2.94, i.e. raw QED > 1
    values = torch.tensor([[0.3, 1.90]], dtype=torch.float64)
    alt, in_support, clamped = module.build_alternative(
        values, 1, "offset", rng=np.random.default_rng(0), qed_offset=1.0,
        grid_std=[], support=support, clamp=True)
    assert bool(clamped.all()), "out-of-support offset was not flagged as clamped"
    assert bool(in_support.all()), "clamping must bring it back into support"
    assert float(alt[0, 1]) == pytest.approx(1.94)
    # unclamped: stays outside, and the script must mark it so
    alt2, in_support2, clamped2 = module.build_alternative(
        values, 1, "offset", rng=np.random.default_rng(0), qed_offset=1.0,
        grid_std=[], support=support, clamp=False)
    assert float(alt2[0, 1]) == pytest.approx(2.90)
    assert not bool(in_support2.any())


def test_offset_of_one_sigma_does_not_reproduce_the_generation_grid():
    """Quantifies why the offset protocol is not a generation-matched probe."""
    qed_mean, qed_sd = 0.7318, 0.1385
    grid = [(q - qed_mean) / qed_sd for q in (0.4, 0.6, 0.8)]
    assert max(grid) - min(grid) == pytest.approx(2.888, abs=0.01)
    # one standardised sigma is 0.1385 raw QED: far smaller than the grid span, and it is
    # applied RELATIVE to each molecule, so high-QED molecules are pushed out of support
    assert 1.0 * qed_sd == pytest.approx(0.1385, abs=1e-4)
    assert qed_mean + (1.94 + 1.0) * qed_sd > 1.0        # asks for QED > 1


def test_in_support_protocols_are_classified_separately():
    module = _script()
    assert "permute-qed" in module.IN_SUPPORT_PROTOCOLS
    assert "grid" in module.IN_SUPPORT_PROTOCOLS
    assert "offset" in module.EXTRAPOLATION_PROTOCOLS
    assert set(module.IN_SUPPORT_PROTOCOLS) & set(module.EXTRAPOLATION_PROTOCOLS) == set()


def test_permutation_protocol_keeps_n_and_uses_real_qed_values():
    module = _script()
    _, _, _, values = _setup()
    alt, in_support, clamped = module.build_alternative(
        values, 1, "permute-qed", rng=np.random.default_rng(1), qed_offset=0.0,
        grid_std=[], support=(-5.28, 1.94), clamp=True)
    assert torch.allclose(alt[:, 0], values[:, 0])       # N untouched
    assert bool(in_support.all()) and not bool(clamped.any())
    pool = sorted(round(v, 9) for v in values[:, 1].tolist())
    assert sorted(round(v, 9) for v in alt[:, 1].tolist()) == pool
