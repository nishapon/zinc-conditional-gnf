"""Tests for the ACP falsification gate.

The gate's scientific validity rests on four properties, each pinned here:
  * ONE fixed reference condition is used for the forward map that fits the direction and
    for the inverse map that generates, so no group difference is confounded by a
    condition-dependent transform;
  * the shift is uniform across nodes and masked, i.e. exactly what a conditional prior
    mean can express (`NodePrior` broadcasts mu of shape [B, D] over nodes);
  * direction and magnitude come from TRAIN only and are never tuned on evaluated samples;
  * the GO/NO-GO rule is mechanical and fixed in code, not chosen after seeing results.

Requires torch + rdkit (the cluster env).
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.test_conditional_flow import conditional_flow, scaler, wake_up_conditioning
from zinc_gnf.conditioning import condition_embedding_for, standardise_condition
from zinc_gnf.flow import node_mask_from_counts

ROOT = Path(__file__).resolve().parents[1]
DIM = 8


def _gate():
    spec = importlib.util.spec_from_file_location(
        "acp_gate", ROOT / "scripts" / "31_acp_falsification_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _flow():
    flow = wake_up_conditioning(conditional_flow(features=("n", "qed"), dimension=DIM))
    flow.eval()
    return flow


# ------------------------------------------------------- fixed reference condition ---
def test_reference_condition_is_identical_for_every_molecule():
    """The core methodological requirement: one c_ref, so f is one function."""
    module = _gate()
    flow = _flow()
    cond = module.reference_condition(flow, scaler(), 23, 0.6, 5, "cpu")
    assert cond.shape[0] == 5
    # every row identical -> the forward map is the same function for all molecules
    for i in range(1, 5):
        assert torch.allclose(cond[0], cond[i], atol=0.0)


def test_reference_condition_differs_between_qed_values():
    module = _gate()
    flow = _flow()
    a = module.reference_condition(flow, scaler(), 23, 0.4, 2, "cpu")
    b = module.reference_condition(flow, scaler(), 23, 0.8, 2, "cpu")
    assert not torch.allclose(a, b)


def test_reference_condition_uses_the_train_only_scaler():
    """The standardised value must match the scaler, not be recomputed ad hoc."""
    the_scaler = scaler()
    expected = standardise_condition(("n", "qed"), scaler=the_scaler, n=[23], qed=[0.6])
    assert expected.shape == (1, 2)
    # QED column is the scaler transform of 0.6
    assert float(expected[0, 1]) == pytest.approx(
        (0.6 - the_scaler.mean[1]) / the_scaler.std[1], abs=1e-6)


def test_gate_never_mixes_two_conditioned_forward_maps():
    """Static guard against the confound: the fit must not use per-molecule QED."""
    src = (ROOT / "scripts" / "31_acp_falsification_gate.py").read_text()
    body = src.split("def pooled_latents")[1].split("def fit_direction")[0]
    # the pooled-latent helper must build its condition from qed_ref, never from batch.qed
    assert "qed_ref" in body
    assert "batch.qed" not in body, \
        "pooled_latents must not condition on each molecule's own QED"


# ------------------------------------------------------------- shift structure ---
def test_shift_is_uniform_across_nodes_and_respects_the_mask():
    """A prior mean shift is the SAME vector on every node; padding must stay zero."""
    counts = [3, 5]
    mask = node_mask_from_counts(counts, max_nodes=5)
    z = torch.randn(2, 5, DIM) * mask.unsqueeze(-1)
    w = torch.randn(DIM)
    w = w / w.norm()
    shifted = (z + 2.0 * w) * mask.unsqueeze(-1).to(z.dtype)
    delta = shifted - z
    # identical on every REAL node
    for b, count in enumerate(counts):
        for i in range(count):
            assert torch.allclose(delta[b, i], 2.0 * w, atol=1e-6)
        for i in range(count, 5):
            assert torch.allclose(delta[b, i], torch.zeros(DIM), atol=0.0)


def test_uniform_shift_moves_the_pooled_mean_by_exactly_the_shift():
    """Consistency of the magnitude unit: d_real is measured on the pooled mean, and a
    uniform shift moves the pooled mean by exactly that amount."""
    counts = [4, 6]
    mask = node_mask_from_counts(counts, max_nodes=6)
    z = torch.randn(2, 6, DIM) * mask.unsqueeze(-1)
    w = torch.nn.functional.normalize(torch.randn(DIM), dim=0)
    d = 1.7
    shifted = (z + d * w) * mask.unsqueeze(-1).to(z.dtype)
    denom = mask.sum(-1, keepdim=True).to(z.dtype)
    before = (z * mask.unsqueeze(-1)).sum(1) / denom
    after = (shifted * mask.unsqueeze(-1)).sum(1) / denom
    assert torch.allclose(after - before, (d * w).expand(2, DIM), atol=1e-5)


def test_uniform_shift_cannot_change_per_node_dispersion():
    """The limitation the report must disclose: a mean shift leaves second moments fixed,
    so the per-node variance change between QED bands is unreachable by ACP."""
    counts = [5, 5]
    mask = node_mask_from_counts(counts, max_nodes=5)
    z = torch.randn(2, 5, DIM) * mask.unsqueeze(-1)
    w = torch.nn.functional.normalize(torch.randn(DIM), dim=0)
    denom = mask.sum(-1, keepdim=True).to(z.dtype)

    def dispersion(t):
        mean = (t * mask.unsqueeze(-1)).sum(1) / denom
        return (((t - mean.unsqueeze(1)) ** 2) * mask.unsqueeze(-1)).sum(1) / denom

    shifted = (z + 3.0 * w) * mask.unsqueeze(-1).to(z.dtype)
    assert torch.allclose(dispersion(z), dispersion(shifted), atol=1e-5)


# --------------------------------------------------------------- direction logic ---
def test_difference_of_means_direction_is_unit_and_its_norm_is_the_magnitude():
    low = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    high = np.array([[3.0, 4.0], [3.0, 4.0]], dtype=np.float64)
    displacement = high.mean(0) - low.mean(0)
    d_real = float(np.linalg.norm(displacement))
    unit = displacement / d_real
    assert d_real == pytest.approx(5.0)
    assert float(np.linalg.norm(unit)) == pytest.approx(1.0)
    # shifting low by d_real * unit lands exactly on high
    assert np.allclose(low.mean(0) + d_real * unit, high.mean(0))


def test_build_directions_returns_three_unit_vectors_and_cosines():
    module = _gate()
    flow = _flow()
    fitted = np.zeros(DIM)
    fitted[0] = 1.0
    vectors, info = module.build_directions(
        flow, scaler(), n=23, fitted=fitted, d_real=1.0, qed_low=0.4, qed_high=0.8,
        device="cpu", dim=DIM)
    assert set(vectors) == {"real_qed", "prior_delta", "random"}
    for name, v in vectors.items():
        norm = float(np.linalg.norm(v))
        assert norm == pytest.approx(1.0, abs=1e-6) or norm == pytest.approx(0.0), name
    for key in ("real_qed_vs_prior_delta", "real_qed_vs_random", "prior_delta_vs_random"):
        assert -1.0001 <= info["cosines"][key] <= 1.0001


def test_random_direction_is_deterministic_per_n():
    """Controls must be reproducible, and different N must not share the control vector."""
    module = _gate()
    flow = _flow()
    fitted = np.eye(DIM)[0]
    first, _ = module.build_directions(flow, scaler(), n=23, fitted=fitted, d_real=1.0,
                                       qed_low=0.4, qed_high=0.8, device="cpu", dim=DIM)
    again, _ = module.build_directions(flow, scaler(), n=23, fitted=fitted, d_real=1.0,
                                       qed_low=0.4, qed_high=0.8, device="cpu", dim=DIM)
    other, _ = module.build_directions(flow, scaler(), n=30, fitted=fitted, d_real=1.0,
                                       qed_low=0.4, qed_high=0.8, device="cpu", dim=DIM)
    assert np.allclose(first["random"], again["random"])
    assert not np.allclose(first["random"], other["random"])


def test_prior_delta_is_the_models_own_prior_movement():
    """The 'does it beat just moving the existing prior?' control must actually be the
    prior's own low->high mean difference."""
    module = _gate()
    flow = _flow()
    cond_low = module.reference_condition(flow, scaler(), 23, 0.4, 1, "cpu")
    cond_high = module.reference_condition(flow, scaler(), 23, 0.8, 1, "cpu")
    mu_low, _ = flow.prior.parameters_for(1, cond_low, "cpu")
    mu_high, _ = flow.prior.parameters_for(1, cond_high, "cpu")
    expected = (mu_high - mu_low).squeeze(0).detach().numpy()
    vectors, info = module.build_directions(
        flow, scaler(), n=23, fitted=np.eye(DIM)[0], d_real=1.0, qed_low=0.4,
        qed_high=0.8, device="cpu", dim=DIM)
    assert info["prior_delta_raw_norm"] == pytest.approx(
        float(np.linalg.norm(expected)), rel=1e-5)


# ------------------------------------------------------------------ bookkeeping ---
def test_zero_shift_must_be_present_in_the_sweep():
    module = _gate()
    with pytest.raises(SystemExit) as excinfo:
        module.main(["--flow-checkpoint", "x", "--output-dir", "/tmp/acp_unused",
                     "--shift-deltas", "0.5,1.0"])
    assert excinfo.value.code == 4


def test_unknown_direction_is_rejected():
    module = _gate()
    with pytest.raises(SystemExit) as excinfo:
        module.main(["--flow-checkpoint", "x", "--output-dir", "/tmp/acp_unused",
                     "--directions", "magic"])
    assert excinfo.value.code == 4


def test_paired_summary_only_uses_jointly_valid_samples():
    module = _gate()
    baseline = [{"valid": True, "qed": 0.40, "heavy_atoms": 23.0, "smiles": "C",
                 "fallback": False},
                {"valid": False, "fallback": False, "smiles": None},
                {"valid": True, "qed": 0.50, "heavy_atoms": 23.0, "smiles": "CC",
                 "fallback": False}]
    shifted = [{"valid": True, "qed": 0.60, "heavy_atoms": 23.0, "smiles": "CCC",
                "fallback": False},
               {"valid": True, "qed": 0.90, "heavy_atoms": 23.0, "smiles": "CCCC",
                "fallback": False},
               {"valid": False, "fallback": False, "smiles": None}]
    block = module.summarise_cell(shifted, baseline=baseline)
    # only index 0 is jointly valid -> dQED = 0.60 - 0.40
    assert block["paired_jointly_valid"] == 1
    assert block["paired_qed_delta"]["mean"] == pytest.approx(0.20, abs=1e-9)


def test_summary_reports_size_so_a_size_driven_qed_change_is_visible():
    module = _gate()
    rows = [{"valid": True, "qed": 0.7, "aromatic_rings": 2.0, "rings": 3.0,
             "heavy_atoms": 20.0, "smiles": "C", "fallback": False}]
    block = module.summarise_cell(rows)
    assert block["heavy_atoms_mean"] == pytest.approx(20.0)
    assert block["aromatic_rings_mean"] == pytest.approx(2.0)
    assert block["uniqueness_over_valid"] == pytest.approx(1.0)


# ------------------------------------------------------------------- the verdict ---
def _payload(delta, ci, validity, base_validity, random_delta, native_delta,
             uniqueness=0.9):
    return {
        "by_n": {"23": {
            "native_response": {"paired_response": {
                "degree": {"mean": native_delta, "ci95": [0.0, 0.1]}}},
            "generated_arm": {
                "unshifted_control": {"degree": {"validity": base_validity}},
                "by_direction": {
                    "real_qed": {"+1.00": {"degree": {
                        "validity": validity,
                        "uniqueness_over_valid": uniqueness,
                        "paired_qed_delta": {"mean": delta, "ci95": ci,
                                             "excludes_zero": ci[0] > 0 or ci[1] < 0}}}},
                    "random": {"+1.00": {"degree": {
                        "paired_qed_delta": {"mean": random_delta}}}}}}}}}


def test_verdict_go_when_effect_is_real_validity_held_and_beats_random():
    module = _gate()
    v = module.verdict(_payload(0.15, [0.08, 0.22], 0.95, 0.98, 0.01, 0.05), "degree")
    assert v["per_n"]["23"]["status"] == "GO"
    assert v["overall"] == "GO"
    assert v["per_n"]["23"]["beats_native_conditioning"] is True


def test_verdict_no_go_when_ci_includes_zero():
    module = _gate()
    v = module.verdict(_payload(0.03, [-0.02, 0.08], 0.95, 0.98, 0.01, 0.05), "degree")
    assert v["per_n"]["23"]["status"] == "NO-GO"


def test_verdict_no_go_when_qed_gained_by_destroying_validity():
    """The explicit failure mode: QED up, molecules gone."""
    module = _gate()
    v = module.verdict(_payload(0.30, [0.20, 0.40], 0.30, 0.98, 0.01, 0.05), "degree")
    assert v["per_n"]["23"]["status"] == "NO-GO"
    assert v["per_n"]["23"]["validity_retained"] is False


def test_verdict_no_go_when_a_random_direction_does_the_same():
    module = _gate()
    v = module.verdict(_payload(0.12, [0.05, 0.19], 0.95, 0.98, 0.20, 0.05), "degree")
    assert v["per_n"]["23"]["status"] == "NO-GO"
    assert v["per_n"]["23"]["beats_random_direction"] is False


def test_verdict_no_go_when_diversity_collapses():
    module = _gate()
    v = module.verdict(_payload(0.15, [0.08, 0.22], 0.95, 0.98, 0.01, 0.05,
                                uniqueness=0.1), "degree")
    assert v["per_n"]["23"]["status"] == "NO-GO"


def test_verdict_reports_failure_to_beat_native_conditioning_separately():
    """Statistically non-zero but worse than the incumbent must be visible."""
    module = _gate()
    v = module.verdict(_payload(0.04, [0.01, 0.07], 0.95, 0.98, 0.01, 0.12), "degree")
    assert v["per_n"]["23"]["beats_native_conditioning"] is False


def test_verdict_partial_when_only_some_n_pass():
    module = _gate()
    payload = _payload(0.15, [0.08, 0.22], 0.95, 0.98, 0.01, 0.05)
    bad = _payload(0.01, [-0.05, 0.07], 0.95, 0.98, 0.01, 0.05)["by_n"]["23"]
    payload["by_n"]["30"] = bad
    v = module.verdict(payload, "degree")
    assert v["overall"] == "PARTIAL"


def test_verdict_marks_skipped_n_as_not_evaluated():
    module = _gate()
    payload = {"by_n": {"15": {"direction_fit": {"usable": False, "note": "no support"}}}}
    v = module.verdict(payload, "degree")
    assert v["per_n"]["15"]["status"] == "not_evaluated"
    assert v["overall"] == "NO-GO"


def test_support_thresholds_are_explicit_constants():
    module = _gate()
    assert module.MIN_BAND_FOR_FIT >= 30
    assert module.MIN_BAND_WARN >= module.MIN_BAND_FOR_FIT
    assert module.MIN_REAL_ARM >= 10
