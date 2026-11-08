"""Tests for decoder-aware guidance (Stage 1).

The experiment is only worth running if six properties hold, so each one is pinned here
rather than argued for in prose:

  1. the loss is TARGET MATCHING, so guidance moves QED DOWN for a low request and UP
     for a high one. The previous design maximised predicted QED, which is wrong for
     every condition except the highest; `test_descent_direction_*` checks the sign of
     the descent direction against the surrogate's own gradient, exactly, at B = 1.
  2. no gradient ever touches `flow.inverse` or a matrix inverse. This is the v1 failure
     mode (`flow.py:247`, `flow.py:503-520`), and it is avoided STRUCTURALLY: the AST
     tests assert the module contains no inverse call at all, and the runtime test makes
     `torch.linalg.inv` raise and still guides successfully.
  3. the decoder and the readout are FROZEN: guidance produces d/dx only, never a
     parameter gradient or a parameter update.
  4. the trust region, the padding mask and the molecular size are respected on every
     step, and a non-finite step reverts and is COUNTED rather than silently skipped.
  5. the derived structural features really are the graph identities they claim to be:
     `rings = E - N + 1` and `sum_i deg_i = 2E` are checked against hand-built logits
     with known answers, and statistics are invariant to padding width.
  6. the script writes nothing but its own report: no optimizer, no `backward()`, no
     `torch.save`, and every `flow.inverse` call site is lexically inside `no_grad`.

Requires torch + rdkit (the cluster env); the sandbox cannot execute these.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from zinc_gnf.autoencoder import AutoencoderConfig, GraphAutoencoder
from zinc_gnf import decoder_guidance as dg
from zinc_gnf.decoder_guidance import (
    FEATURE_SETS,
    MANIFOLD_WEIGHTS,
    RidgeReadout,
    decoder_statistics,
    embedding_features,
    feature_names,
    feature_vector,
    features_with_grad,
    guidance_loss,
    guide,
    manifold_loss,
    readout_quality,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "zinc_gnf" / "decoder_guidance.py"
SCRIPT_PATH = ROOT / "scripts" / "32_decoder_guidance_pilot.py"
SLURM_PATH = ROOT / "slurm" / "32_decoder_guidance_pilot.sh"

TINY = dict(node_embedding_dim=16, latent_dim=32, num_processing_steps=2,
            attn_kq_dim=8, attn_v_dim=8, attn_concat_heads_output_dim=8)
DIM = TINY["node_embedding_dim"]


# --------------------------------------------------------------------- helpers ---
def pilot():
    spec = importlib.util.spec_from_file_location("guidance_pilot", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def autoencoder():
    torch.manual_seed(0)
    model = GraphAutoencoder(AutoencoderConfig(**TINY, degree_head=True))
    model.eval().requires_grad_(False)          # frozen, as in the pilot
    return model


def masks(counts, width=None):
    width = width or max(counts)
    node = torch.zeros((len(counts), width), dtype=torch.bool)
    for row, count in enumerate(counts):
        node[row, :count] = True
    eye = torch.eye(width, dtype=torch.bool).unsqueeze(0)
    return node, (node.unsqueeze(2) & node.unsqueeze(1)) & ~eye


def embeddings_for(node_mask, *, seed=1):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((node_mask.shape[0], node_mask.shape[1], DIM), generator=generator)
    return x * node_mask.unsqueeze(-1).to(x.dtype)


def fitted_readout(autoencoder, *, feature_set="structural", seed=3):
    """A ridge readout fitted to a KNOWN linear function of the features.

    Using a synthetic but exactly-linear target means the surrogate's gradient is
    trustworthy, so a direction test measures the OBJECTIVE, not the fit quality.
    """
    node, pair = masks([7] * 64)
    x = embeddings_for(node, seed=seed)
    features = embedding_features(autoencoder, x, node, pair,
                                  feature_set=feature_set).numpy().astype(np.float64)
    truth = np.linspace(-1.0, 1.0, features.shape[1])
    targets = features @ truth
    readout, _ = RidgeReadout.fit(features, targets, target="qed")
    return readout


def logits_for_cycle(count, *, width=None, degree=2.0, strength=20.0):
    """Hand-built decoder outputs for a single `count`-cycle, so E, N and rings are known.

    `count` ring edges, every node of degree 2, therefore E = N and rings = E - N + 1 = 1.
    """
    width = width or count
    bond = torch.zeros((1, width, width, 4))
    bond[..., 0] = strength                                  # default: no bond
    for i in range(count):
        j = (i + 1) % count
        bond[0, i, j, 0] = -strength
        bond[0, i, j, 1] = strength
        bond[0, j, i, 0] = -strength
        bond[0, j, i, 1] = strength
    atom = torch.zeros((1, width, 17))
    atom[0, :, 0] = strength
    degrees = torch.zeros((1, width, 7))
    degrees[0, :, int(degree)] = strength
    return {"atom_logits": atom, "bond_logits": bond, "degree_logits": degrees}


# ====================================================================================
# 1. The objective is TARGET MATCHING, in both directions.
# ====================================================================================
def _descent_vs_surrogate_gradient(autoencoder, offset):
    """Inner product of the descent direction with grad(q_hat). B = 1, so it is exact."""
    node, pair = masks([9])
    readout = fitted_readout(autoencoder)
    x = embeddings_for(node, seed=11)
    probe = x.clone().requires_grad_(True)
    features, statistics = features_with_grad(autoencoder, probe, node, pair)
    predicted = readout.predict_torch(features)
    (surrogate_gradient,) = torch.autograd.grad(predicted.sum(), probe, retain_graph=True)
    target = float(predicted.item()) + offset
    loss, diagnostics = guidance_loss(readout, features, statistics, None,
                                      qed_target=target, lambda_qed=1.0,
                                      lambda_manifold=0.0)
    (loss_gradient,) = torch.autograd.grad(loss.sum(), probe)
    descent = -loss_gradient
    assert float(diagnostics["surrogate_qed"].item()) == pytest.approx(
        float(predicted.item()), abs=1e-6)
    return float((descent * surrogate_gradient).sum())


def test_descent_direction_lowers_predicted_qed_for_a_low_target(autoencoder):
    """A request BELOW the current prediction must push the prediction DOWN."""
    assert _descent_vs_surrogate_gradient(autoencoder, -1.0) < 0.0


def test_descent_direction_raises_predicted_qed_for_a_high_target(autoencoder):
    """A request ABOVE the current prediction must push the prediction UP."""
    assert _descent_vs_surrogate_gradient(autoencoder, +1.0) > 0.0


def test_the_loss_is_a_squared_error_not_a_maximisation(autoencoder):
    """Two targets equidistant from the prediction give the SAME loss.

    A maximise-QED objective cannot have this property, so this test fails if anyone
    reverts the correction.
    """
    node, pair = masks([9])
    readout = fitted_readout(autoencoder)
    features, statistics = features_with_grad(
        autoencoder, embeddings_for(node, seed=11).requires_grad_(True), node, pair)
    centre = float(readout.predict_torch(features).item())
    losses = []
    for target in (centre - 0.3, centre + 0.3):
        loss, _ = guidance_loss(readout, features, statistics, None, qed_target=target,
                                lambda_qed=1.0, lambda_manifold=0.0)
        losses.append(float(loss.item()))
    assert losses[0] == pytest.approx(losses[1], rel=1e-6)
    assert losses[0] == pytest.approx(0.09, rel=1e-5)


def test_loss_is_zero_exactly_at_the_requested_value(autoencoder):
    node, pair = masks([9])
    readout = fitted_readout(autoencoder)
    features, statistics = features_with_grad(
        autoencoder, embeddings_for(node, seed=11).requires_grad_(True), node, pair)
    exact = float(readout.predict_torch(features).item())
    loss, _ = guidance_loss(readout, features, statistics, None, qed_target=exact,
                            lambda_qed=1.0, lambda_manifold=0.0)
    assert float(loss.item()) == pytest.approx(0.0, abs=1e-10)


def test_guidance_moves_the_surrogate_prediction_toward_a_low_request(autoencoder):
    node, pair = masks([9])
    readout = fitted_readout(autoencoder)
    x = embeddings_for(node, seed=11)
    before = float(readout.predict_numpy(
        embedding_features(autoencoder, x, node, pair).numpy().astype(np.float64))[0])
    target = before - 0.5
    guided, _ = guide(autoencoder, x, node, pair, readout=readout, targets=None,
                      qed_target=target, lambda_qed=1.0, lambda_manifold=0.0, steps=10,
                      step_size=0.2, trust_radius=0.5)
    after = float(readout.predict_numpy(
        embedding_features(autoencoder, guided, node, pair).numpy().astype(np.float64))[0])
    assert abs(after - target) < abs(before - target)
    assert after < before


def test_guidance_moves_the_surrogate_prediction_toward_a_high_request(autoencoder):
    node, pair = masks([9])
    readout = fitted_readout(autoencoder)
    x = embeddings_for(node, seed=11)
    before = float(readout.predict_numpy(
        embedding_features(autoencoder, x, node, pair).numpy().astype(np.float64))[0])
    target = before + 0.5
    guided, _ = guide(autoencoder, x, node, pair, readout=readout, targets=None,
                      qed_target=target, lambda_qed=1.0, lambda_manifold=0.0, steps=10,
                      step_size=0.2, trust_radius=0.5)
    after = float(readout.predict_numpy(
        embedding_features(autoencoder, guided, node, pair).numpy().astype(np.float64))[0])
    assert abs(after - target) < abs(before - target)
    assert after > before


# ====================================================================================
# 2. No gradient path through any inverse. v1's failure mode, excluded structurally.
# ====================================================================================
def _calls_in(path):
    tree = ast.parse(path.read_text())
    return [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]


def test_the_guidance_module_contains_no_inverse_call_at_all():
    """AST, not grep: docstrings mention `flow.inverse`, code must never call it."""
    offenders = [call for call in _calls_in(MODULE_PATH)
                 if "inverse" in call or "linalg.inv" in call or "pinv" in call]
    assert offenders == []


def test_the_guidance_module_never_references_the_flow():
    """The flow is not even an argument, so it cannot be differentiated through."""
    tree = ast.parse(MODULE_PATH.read_text())
    for function in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        names = [a.arg for a in function.args.args] + [a.arg for a in function.args.kwonlyargs]
        assert "flow" not in names, function.name


def test_guidance_runs_even_when_every_matrix_inverse_is_poisoned(autoencoder, monkeypatch):
    """Runtime proof. `flow.py:247` is the site that went non-finite in v1."""
    def poisoned(*args, **kwargs):
        raise AssertionError("guidance must never evaluate a matrix inverse")

    monkeypatch.setattr(torch.linalg, "inv", poisoned)
    monkeypatch.setattr(torch.linalg, "pinv", poisoned)
    node, pair = masks([9])
    readout = fitted_readout(autoencoder)
    guided, report = guide(autoencoder, embeddings_for(node, seed=5), node, pair,
                           readout=readout, targets=None, qed_target=0.6,
                           lambda_qed=1.0, lambda_manifold=0.0, steps=3,
                           step_size=0.1, trust_radius=0.5)
    assert torch.isfinite(guided).all()
    assert report["nonfinite_steps"] == 0


def test_pooling_uses_no_square_root_in_a_gradient_path():
    """v1's other NaN source: d/dv sqrt(v) -> infinity for symmetric atoms (v ~ 0).

    The only norms left in the module are on DETACHED tensors (step normalisation and
    reporting), so no square root sits on the backward path.
    """
    # the only square root in the module is `math.sqrt` on a PLAIN PYTHON FLOAT, used to
    # size the float32 projection margin. No tensor square root exists anywhere.
    square_roots = [call for call in _calls_in(MODULE_PATH) if "sqrt" in call]
    assert square_roots == ["math.sqrt"] * len(square_roots)
    graph_functions = {"decoder_statistics", "feature_vector", "features_with_grad",
                       "manifold_loss", "guidance_loss", "_masked_mean"}
    tree = ast.parse(MODULE_PATH.read_text())
    for function in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        if function.name not in graph_functions:
            continue
        calls = [ast.unparse(n.func) for n in ast.walk(function) if isinstance(n, ast.Call)]
        assert not [c for c in calls if c.endswith(("sqrt", "norm", "std"))], function.name


def test_every_flow_inverse_in_the_pilot_is_lexically_inside_no_grad():
    lines = SCRIPT_PATH.read_text().splitlines()
    sites = [i for i, line in enumerate(lines) if "flow.inverse(" in line]
    assert sites, "the pilot must actually generate from the flow"
    for index in sites:
        window = "\n".join(lines[max(0, index - 6):index])
        assert "no_grad" in window, lines[index]


# ====================================================================================
# 3. Decoder and readout stay frozen. Guidance yields d/dx only.
# ====================================================================================
def test_guidance_produces_no_parameter_gradients(autoencoder):
    for parameter in autoencoder.parameters():
        parameter.grad = None
    node, pair = masks([9])
    guide(autoencoder, embeddings_for(node, seed=7), node, pair,
          readout=fitted_readout(autoencoder), targets=None, qed_target=0.6,
          lambda_qed=1.0, lambda_manifold=0.0, steps=4, step_size=0.1, trust_radius=0.5)
    assert all(not p.requires_grad for p in autoencoder.parameters())
    assert all(p.grad is None for p in autoencoder.parameters())


def test_guidance_leaves_every_decoder_weight_bit_identical(autoencoder):
    before = {k: v.clone() for k, v in autoencoder.decoder.state_dict().items()}
    node, pair = masks([9])
    guide(autoencoder, embeddings_for(node, seed=7), node, pair,
          readout=fitted_readout(autoencoder), targets=None, qed_target=0.6,
          lambda_qed=1.0, lambda_manifold=0.0, steps=4, step_size=0.1, trust_radius=0.5)
    after = autoencoder.decoder.state_dict()
    for key, value in before.items():
        assert torch.equal(value, after[key]), key


def test_the_readout_is_frozen_during_guidance(autoencoder):
    readout = fitted_readout(autoencoder)
    snapshot = readout.to_dict()
    node, pair = masks([9])
    guide(autoencoder, embeddings_for(node, seed=7), node, pair, readout=readout,
          targets=None, qed_target=0.9, lambda_qed=1.0, lambda_manifold=0.0, steps=5,
          step_size=0.2, trust_radius=1.0)
    assert readout.to_dict() == snapshot


def test_the_returned_embeddings_carry_no_graph(autoencoder):
    node, pair = masks([9])
    guided, _ = guide(autoencoder, embeddings_for(node, seed=7), node, pair,
                      readout=fitted_readout(autoencoder), targets=None, qed_target=0.6,
                      lambda_qed=1.0, lambda_manifold=0.0, steps=3, step_size=0.1,
                      trust_radius=0.5)
    assert not guided.requires_grad and guided.grad_fn is None


# ====================================================================================
# 4. Trust region, masking, size preservation, non-finite accounting.
# ====================================================================================
def test_displacement_never_exceeds_the_trust_radius(autoencoder):
    """The ENFORCED displacement is bounded outright, with no tolerance at all.

    The projection targets `trust_radius * (1 - 32*sqrt(numel)*eps)`, a shrink that
    dominates float32 round-off, so the bound is an inequality rather than an
    approximate one. The externally remeasured value is allowed only the
    `eps * ||x_0||` precision limit of recovering a small difference of large vectors,
    which the module reports itself.
    """
    node, pair = masks([9, 6])
    radius = 0.4
    guided, report = guide(autoencoder, embeddings_for(node, seed=13), node, pair,
                           readout=fitted_readout(autoencoder), targets=None,
                           qed_target=5.0, lambda_qed=1.0, lambda_manifold=0.0,
                           steps=40, step_size=0.9, trust_radius=radius)
    assert report["displacement_max"] <= radius
    floor = report["displacement_measurement_floor"]
    assert 0.0 < floor < 0.01 * radius          # the limit is real but negligible here
    per_molecule = (guided - embeddings_for(node, seed=13)).flatten(1).norm(dim=1)
    assert float(per_molecule.max()) <= radius + floor
    assert report["displacement_max_remeasured"] <= radius + floor
    # an absurd target with a long schedule should actually reach the boundary,
    # otherwise the bound above would hold trivially
    assert report["trust_region_saturated_fraction"] > 0.0
    assert report["steps_applied"] == 40


def test_a_tiny_trust_radius_confines_guidance_to_a_tiny_ball(autoencoder):
    """The case that exposed the old bug: a 1e-3 radius around a start of norm ~12.

    Enforcing the region on `(x_k - x_0)` meant measuring a 1e-3 displacement by
    cancelling two vectors of norm ~12, which in float32 cannot resolve it better than
    about 0.1%; the clip therefore overshot. Carrying the displacement as the state makes
    the enforced bound exact, and the remeasured value is still only accurate to the
    reported floor -- which is why the two are asserted at different tolerances.
    """
    node, pair = masks([9])
    guided, report = guide(autoencoder, embeddings_for(node, seed=13), node, pair,
                           readout=fitted_readout(autoencoder), targets=None,
                           qed_target=5.0, lambda_qed=1.0, lambda_manifold=0.0,
                           steps=30, step_size=0.5, trust_radius=1e-3)
    assert report["displacement_max"] <= 1e-3
    assert report["displacement_max_remeasured"] <= (
        1e-3 + report["displacement_measurement_floor"])
    # the bound is not met by refusing to move
    assert report["steps_applied"] == 30 and report["displacement_max"] > 0.0
    assert torch.isfinite(guided).all()


def test_the_trust_region_shrink_is_numerically_motivated_and_scientifically_nil():
    """The projection margin must be round-off scale, not a hidden reduction in radius."""
    epsilon = float(torch.finfo(torch.float32).eps)
    for elements in (9 * DIM, 30 * 128):
        shrink = 32.0 * (elements ** 0.5) * epsilon
        assert shrink < 1e-3, (elements, shrink)     # below any measurable effect
        assert shrink > 16.0 * epsilon, (elements, shrink)   # above float32 round-off


def test_the_enforced_radius_is_reported_next_to_the_requested_one(autoencoder):
    node, pair = masks([9])
    _, report = guide(autoencoder, embeddings_for(node, seed=13), node, pair,
                      readout=fitted_readout(autoencoder), targets=None,
                      qed_target=5.0, lambda_qed=1.0, lambda_manifold=0.0, steps=2,
                      step_size=0.5, trust_radius=0.5)
    assert report["trust_radius"] == 0.5
    assert report["trust_radius_effective"] < report["trust_radius"]
    assert report["trust_radius_effective"] > 0.9999 * report["trust_radius"]


def test_padding_stays_exactly_zero_and_the_size_is_preserved(autoencoder):
    counts = [9, 5, 3]
    node, pair = masks(counts)
    guided, _ = guide(autoencoder, embeddings_for(node, seed=17), node, pair,
                      readout=fitted_readout(autoencoder), targets=None, qed_target=0.9,
                      lambda_qed=1.0, lambda_manifold=0.0, steps=8, step_size=0.3,
                      trust_radius=1.0)
    padding = ~node
    assert torch.count_nonzero(guided[padding]) == 0
    # the mask itself is untouched, so molecular size cannot drift
    assert node.sum(-1).tolist() == counts


def test_the_mask_argument_is_not_mutated(autoencoder):
    node, pair = masks([9, 5])
    node_before, pair_before = node.clone(), pair.clone()
    guide(autoencoder, embeddings_for(node, seed=17), node, pair,
          readout=fitted_readout(autoencoder), targets=None, qed_target=0.9,
          lambda_qed=1.0, lambda_manifold=0.0, steps=5, step_size=0.3, trust_radius=1.0)
    assert torch.equal(node, node_before) and torch.equal(pair, pair_before)


def test_zero_steps_reproduces_the_native_sample_exactly(autoencoder):
    node, pair = masks([9, 5])
    x = embeddings_for(node, seed=19)
    guided, report = guide(autoencoder, x, node, pair,
                           readout=fitted_readout(autoencoder), targets=None,
                           qed_target=0.6, lambda_qed=1.0, lambda_manifold=0.0, steps=0,
                           step_size=0.3, trust_radius=1.0)
    assert torch.equal(guided, x)
    assert report["displacement_max"] == 0.0 and report["nonfinite_steps"] == 0
    assert report["steps_applied"] == 0 and report["constant_objective_steps"] == 0


def test_both_lambdas_zero_is_a_no_op_rather_than_a_crash(autoencoder):
    """The degenerate objective is CONSTANT, so there is no gradient to take.

    `allow_unused=True` does not cover this: the error is that the OUTPUT has no
    `grad_fn`, not that an input went unused. It has to be detected before
    `torch.autograd.grad` is called, counted, and reported.
    """
    node, pair = masks([9])
    x = embeddings_for(node, seed=19)
    guided, report = guide(autoencoder, x, node, pair, readout=None, targets=None,
                           qed_target=0.6, lambda_qed=0.0, lambda_manifold=0.0, steps=3,
                           step_size=0.3, trust_radius=1.0)
    assert torch.equal(guided, x)                        # EXACTLY the native sample
    assert report["displacement_max"] == 0.0
    assert report["displacement_max_remeasured"] == 0.0
    assert report["constant_objective_steps"] == 3        # counted, not silently zeroed
    assert report["nonfinite_steps"] == 0                 # it is not a numerical failure
    assert report["steps_applied"] == 0


def test_a_constant_objective_is_counted_rather_than_crashing_or_hiding(autoencoder,
                                                                       monkeypatch):
    """The same path guards a caller that wrapped `guide` in `no_grad` by mistake.

    Without the check that is a silent wrong answer: the report would read as "guidance
    had no effect" when in fact no gradient was ever available.
    """
    def detached_loss(readout, features, statistics, targets, **kwargs):
        return torch.zeros(features.shape[0], dtype=features.dtype), {}

    monkeypatch.setattr(dg, "guidance_loss", detached_loss)
    node, pair = masks([9])
    x = embeddings_for(node, seed=19)
    guided, report = guide(autoencoder, x, node, pair,
                           readout=fitted_readout(autoencoder), targets=None,
                           qed_target=0.6, lambda_qed=1.0, lambda_manifold=0.0, steps=4,
                           step_size=0.3, trust_radius=1.0)
    assert report["constant_objective_steps"] == 4
    assert report["steps_applied"] == 0 and torch.equal(guided, x)
    assert "constant_objective_steps" in report["fallback_behaviour"]


def test_negative_steps_is_rejected(autoencoder):
    node, pair = masks([9])
    with pytest.raises(ValueError):
        guide(autoencoder, embeddings_for(node), node, pair, readout=None, targets=None,
              qed_target=0.6, lambda_qed=0.0, lambda_manifold=1.0, steps=-1,
              step_size=0.1, trust_radius=1.0)


def test_a_non_finite_objective_reverts_and_is_counted(autoencoder, monkeypatch):
    def nan_loss(readout, features, statistics, targets, **kwargs):
        return features.sum(-1) * float("nan"), {}

    monkeypatch.setattr(dg, "guidance_loss", nan_loss)
    node, pair = masks([9])
    x = embeddings_for(node, seed=23)
    guided, report = guide(autoencoder, x, node, pair,
                           readout=fitted_readout(autoencoder), targets=None,
                           qed_target=0.6, lambda_qed=1.0, lambda_manifold=0.0, steps=6,
                           step_size=0.3, trust_radius=1.0)
    assert report["nonfinite_steps"] == 6
    assert report["nonfinite_rate"] == 1.0
    assert torch.equal(guided, x)                       # reverted, not corrupted
    assert "fallback_behaviour" in report


def test_a_non_finite_update_reverts_and_is_counted(autoencoder, monkeypatch):
    node, pair = masks([9])
    x = embeddings_for(node, seed=23)          # built BEFORE randn is poisoned
    monkeypatch.setattr(torch, "randn",
                        lambda *a, **k: torch.full(a[0], float("inf")))
    guided, report = guide(autoencoder, x, node, pair, readout=None, targets=None,
                           qed_target=0.6, lambda_qed=0.0, lambda_manifold=0.0, steps=4,
                           step_size=0.3, trust_radius=1.0, direction="random",
                           generator=torch.Generator().manual_seed(0))
    assert report["nonfinite_steps"] == 4 and torch.equal(guided, x)


def test_a_partially_non_finite_run_is_reported_not_hidden(autoencoder, monkeypatch):
    calls = {"n": 0}
    real = dg.guidance_loss

    def sometimes_nan(*args, **kwargs):
        calls["n"] += 1
        loss, diagnostics = real(*args, **kwargs)
        if calls["n"] % 2 == 0:
            return loss * float("nan"), diagnostics
        return loss, diagnostics

    monkeypatch.setattr(dg, "guidance_loss", sometimes_nan)
    node, pair = masks([9])
    _, report = guide(autoencoder, embeddings_for(node, seed=29), node, pair,
                      readout=fitted_readout(autoencoder), targets=None, qed_target=0.9,
                      lambda_qed=1.0, lambda_manifold=0.0, steps=6, step_size=0.2,
                      trust_radius=1.0)
    assert report["nonfinite_steps"] == 3
    assert report["nonfinite_rate"] == pytest.approx(0.5)


def test_random_arm_is_matched_norm_with_the_gradient_arm(autoencoder):
    """The control must displace by the SAME amount, or it controls for nothing."""
    node, pair = masks([9])
    x = embeddings_for(node, seed=31)
    common = dict(targets=None, qed_target=0.9, steps=1, step_size=0.25,
                  trust_radius=0.8)
    _, gradient_report = guide(autoencoder, x, node, pair,
                               readout=fitted_readout(autoencoder), lambda_qed=1.0,
                               lambda_manifold=0.0, **common)
    _, random_report = guide(autoencoder, x, node, pair, readout=None, lambda_qed=0.0,
                             lambda_manifold=0.0, direction="random",
                             generator=torch.Generator().manual_seed(0), **common)
    expected = 0.25 * 0.8
    assert gradient_report["displacement_mean"] == pytest.approx(expected, rel=1e-4)
    assert random_report["displacement_mean"] == pytest.approx(expected, rel=1e-4)


def test_random_arm_actually_moves_and_is_not_the_gradient(autoencoder):
    node, pair = masks([9])
    x = embeddings_for(node, seed=31)
    gradient_guided, _ = guide(autoencoder, x, node, pair,
                               readout=fitted_readout(autoencoder), targets=None,
                               qed_target=0.9, lambda_qed=1.0, lambda_manifold=0.0,
                               steps=5, step_size=0.2, trust_radius=0.8)
    random_guided, _ = guide(autoencoder, x, node, pair, readout=None, targets=None,
                             qed_target=0.9, lambda_qed=0.0, lambda_manifold=0.0,
                             steps=5, step_size=0.2, trust_radius=0.8,
                             direction="random",
                             generator=torch.Generator().manual_seed(0))
    assert not torch.allclose(gradient_guided, random_guided)
    assert not torch.allclose(random_guided, x)


def test_random_direction_requires_an_explicit_generator(autoencoder):
    node, pair = masks([9])
    with pytest.raises(ValueError):
        guide(autoencoder, embeddings_for(node), node, pair, readout=None, targets=None,
              qed_target=0.6, lambda_qed=0.0, lambda_manifold=0.0, steps=1,
              step_size=0.1, trust_radius=1.0, direction="random", generator=None)


def test_one_molecule_cannot_set_the_step_for_the_batch(autoencoder):
    """Per-molecule normalisation: a huge-gradient molecule must not drag the others."""
    node, pair = masks([9, 9])
    x = embeddings_for(node, seed=37)
    x = torch.stack([x[0], x[1] * 50.0])
    guided, report = guide(autoencoder, x, node, pair,
                           readout=fitted_readout(autoencoder), targets=None,
                           qed_target=0.9, lambda_qed=1.0, lambda_manifold=0.0, steps=3,
                           step_size=0.2, trust_radius=0.5)
    per_molecule = (guided - x * node.unsqueeze(-1)).flatten(1).norm(dim=1)
    assert float(per_molecule.min()) > 0.0           # BOTH molecules actually moved
    # the bound is checked on the enforced quantity: remeasuring molecule 2 against a
    # start of norm ~600 is only accurate to eps * 600, which is the reported floor
    assert report["displacement_max"] <= 0.5
    assert float(per_molecule.max()) <= 0.5 + report["displacement_measurement_floor"]


def test_the_report_states_the_fallback_behaviour_explicitly(autoencoder):
    node, pair = masks([9])
    _, report = guide(autoencoder, embeddings_for(node), node, pair, readout=None,
                      targets=None, qed_target=0.6, lambda_qed=0.0,
                      lambda_manifold=0.0, steps=2, step_size=0.1, trust_radius=1.0)
    for key in ("steps_requested", "steps_applied", "nonfinite_steps", "nonfinite_rate",
                "constant_objective_steps", "fallback_behaviour", "trust_radius",
                "trust_radius_effective", "step_size", "direction",
                "displacement_mean", "displacement_max", "displacement_max_remeasured",
                "displacement_measurement_floor", "trust_region_saturated_fraction",
                "lambda_qed", "lambda_manifold", "history"):
        assert key in report
    # the three ways a step can fail to apply are each named in the prose
    for phrase in ("nonfinite_steps", "constant_objective_steps", "steps_applied"):
        assert phrase in report["fallback_behaviour"]


# ====================================================================================
# 5. The derived features are the graph identities they claim to be.
# ====================================================================================
def test_expected_edge_count_matches_the_hand_built_cycle():
    node, pair = masks([6])
    statistics = decoder_statistics(logits_for_cycle(6), node, pair)
    assert float(statistics["edges"]) == pytest.approx(6.0, abs=1e-3)
    assert float(statistics["n_node"]) == 6.0
    assert float(statistics["edges_per_node"]) == pytest.approx(1.0, abs=1e-3)


def test_cyclomatic_identity_rings_equals_edges_minus_nodes_plus_one():
    node, pair = masks([6])
    statistics = decoder_statistics(logits_for_cycle(6), node, pair)
    expected = float(statistics["edges"]) - float(statistics["n_node"]) + 1.0
    assert float(statistics["rings_from_bonds"]) == pytest.approx(expected, abs=1e-6)
    assert float(statistics["rings_from_bonds"]) == pytest.approx(1.0, abs=1e-3)


def test_a_tree_has_zero_rings_and_two_chords_add_two_rings():
    """E = N - 1 gives rings = 0; adding two chords to a 6-cycle gives rings = 3."""
    node, pair = masks([6])
    chain = logits_for_cycle(6)
    # break the ring closure 5-0 -> a path, E = 5
    chain["bond_logits"][0, 5, 0, 0] = 20.0
    chain["bond_logits"][0, 5, 0, 1] = -20.0
    chain["bond_logits"][0, 0, 5, 0] = 20.0
    chain["bond_logits"][0, 0, 5, 1] = -20.0
    statistics = decoder_statistics(chain, node, pair)
    assert float(statistics["edges"]) == pytest.approx(5.0, abs=1e-3)
    assert float(statistics["rings_from_bonds"]) == pytest.approx(0.0, abs=1e-3)

    bicycle = logits_for_cycle(6)
    for i, j in ((0, 3), (1, 4)):
        for a, b in ((i, j), (j, i)):
            bicycle["bond_logits"][0, a, b, 0] = -20.0
            bicycle["bond_logits"][0, a, b, 1] = 20.0
    statistics = decoder_statistics(bicycle, node, pair)
    assert float(statistics["edges"]) == pytest.approx(8.0, abs=1e-3)
    assert float(statistics["rings_from_bonds"]) == pytest.approx(3.0, abs=1e-3)


def test_degree_sum_identity_sum_of_degrees_is_twice_the_edge_count():
    node, pair = masks([6])
    statistics = decoder_statistics(logits_for_cycle(6, degree=2.0), node, pair)
    assert float(statistics["edges_from_degree"]) == pytest.approx(6.0, abs=1e-3)
    assert float(statistics["head_disagreement"]) == pytest.approx(0.0, abs=1e-3)


def test_head_disagreement_is_nonzero_when_the_two_heads_contradict():
    node, pair = masks([6])
    # bond head says a 6-cycle (E = 6) while the degree head insists every atom has 4
    statistics = decoder_statistics(logits_for_cycle(6, degree=4.0), node, pair)
    assert float(statistics["edges_from_degree"]) == pytest.approx(12.0, abs=1e-2)
    assert float(statistics["head_disagreement"]) == pytest.approx(6.0 / 6.0, abs=1e-2)


def test_statistics_are_invariant_to_padding_width():
    """Molecular-size handling: the same molecule padded wider must give identical stats.

    This is what makes the per-N manifold targets comparable across the batch.
    """
    narrow_node, narrow_pair = masks([6], width=6)
    wide_node, wide_pair = masks([6], width=11)
    narrow = decoder_statistics(logits_for_cycle(6, width=6), narrow_node, narrow_pair)
    wide = decoder_statistics(logits_for_cycle(6, width=11), wide_node, wide_pair)
    for key in narrow:
        assert torch.allclose(narrow[key], wide[key], atol=1e-5), key


def test_bond_histogram_is_a_probability_distribution():
    node, pair = masks([6])
    statistics = decoder_statistics(logits_for_cycle(6), node, pair)
    assert float(statistics["bond_histogram"].sum()) == pytest.approx(1.0, abs=1e-5)
    assert float(statistics["atom_histogram"].sum()) == pytest.approx(1.0, abs=1e-5)
    assert float(statistics["degree_histogram"].sum()) == pytest.approx(1.0, abs=1e-5)


def test_structural_features_are_a_strict_superset_of_the_moments(autoencoder):
    node, pair = masks([7] * 4)
    x = embeddings_for(node, seed=41)
    moments = embedding_features(autoencoder, x, node, pair, feature_set="moments")
    structural = embedding_features(autoencoder, x, node, pair, feature_set="structural")
    assert structural.shape[1] > moments.shape[1]
    assert torch.allclose(structural[:, :moments.shape[1]], moments, atol=0.0)
    assert structural.shape[1] - moments.shape[1] == 6


def test_feature_names_match_the_feature_dimension(autoencoder):
    node, pair = masks([7] * 2)
    x = embeddings_for(node, seed=43)
    for feature_set in FEATURE_SETS:
        vector = embedding_features(autoencoder, x, node, pair, feature_set=feature_set)
        names = feature_names(autoencoder.config.num_atom_types,
                              autoencoder.config.num_bond_types, 7,
                              feature_set=feature_set)
        assert len(names) == vector.shape[1], feature_set
        assert len(set(names)) == len(names)


def test_an_unknown_feature_set_is_rejected():
    node, pair = masks([6])
    statistics = decoder_statistics(logits_for_cycle(6), node, pair)
    with pytest.raises(ValueError):
        feature_vector(statistics, feature_set="whatever")


def test_features_are_differentiable_in_the_embeddings(autoencoder):
    node, pair = masks([7])
    probe = embeddings_for(node, seed=47).requires_grad_(True)
    features, _ = features_with_grad(autoencoder, probe, node, pair)
    (gradient,) = torch.autograd.grad(features.sum(), probe)
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0.0


# ====================================================================================
# 6. The manifold term: per-N, per-node, and it penalises head disagreement.
# ====================================================================================
def manifold_target_from(statistics):
    return {"bond_histogram": statistics["bond_histogram"][0].clone(),
            "degree_histogram": statistics["degree_histogram"][0].clone(),
            "edges_per_node": statistics["edges_per_node"][0].clone(),
            "rings_per_node": (statistics["rings_from_bonds"]
                               / statistics["n_node"])[0].clone()}


def test_manifold_loss_is_zero_when_the_statistics_already_match():
    node, pair = masks([6])
    statistics = decoder_statistics(logits_for_cycle(6, degree=2.0), node, pair)
    total, parts = manifold_loss(statistics, manifold_target_from(statistics))
    assert float(total) == pytest.approx(0.0, abs=1e-4)
    assert set(parts) >= {"bond_histogram", "degree_histogram", "edges_per_node",
                          "rings_per_node", "head_consistency"}


def test_manifold_loss_penalises_head_disagreement_even_at_matching_targets():
    """The one term with no target: sum_i deg_i = 2E is a self-consistency identity."""
    node, pair = masks([6])
    consistent = decoder_statistics(logits_for_cycle(6, degree=2.0), node, pair)
    targets = manifold_target_from(consistent)
    inconsistent = decoder_statistics(logits_for_cycle(6, degree=4.0), node, pair)
    _, parts = manifold_loss(inconsistent, targets)
    assert float(parts["head_consistency"]) > 0.5


def test_manifold_loss_grows_as_the_statistics_move_away():
    node, pair = masks([6])
    targets = manifold_target_from(
        decoder_statistics(logits_for_cycle(6, degree=2.0), node, pair))
    near = decoder_statistics(logits_for_cycle(6, degree=2.0), node, pair)
    far = decoder_statistics(logits_for_cycle(3, width=6, degree=2.0), node, pair)
    assert float(manifold_loss(far, targets)[0]) > float(manifold_loss(near, targets)[0])


def test_extensive_manifold_terms_are_per_node_so_size_cannot_dominate():
    """A 6-cycle and a 12-cycle share edges_per_node = 1, so the term is size-neutral."""
    small_node, small_pair = masks([6])
    big_node, big_pair = masks([12])
    small = decoder_statistics(logits_for_cycle(6), small_node, small_pair)
    big = decoder_statistics(logits_for_cycle(12), big_node, big_pair)
    assert float(small["edges_per_node"]) == pytest.approx(
        float(big["edges_per_node"]), abs=1e-3)
    targets = {"edges_per_node": small["edges_per_node"][0].clone()}
    assert float(manifold_loss(big, targets)[0]) == pytest.approx(0.0, abs=1e-5)


def test_manifold_loss_requires_at_least_one_target():
    node, pair = masks([6])
    statistics = decoder_statistics(logits_for_cycle(6), node, pair)
    statistics.pop("head_disagreement")
    with pytest.raises(ValueError):
        manifold_loss(statistics, {})


def test_manifold_weights_are_fixed_constants_not_swept():
    assert set(MANIFOLD_WEIGHTS) == {"bond_histogram", "degree_histogram",
                                     "edges_per_node", "rings_per_node",
                                     "head_consistency"}
    assert all(value == 1.0 for value in MANIFOLD_WEIGHTS.values())


def test_manifold_arm_needs_no_readout_at_all(autoencoder):
    """The manifold arm must be runnable when Phase A rejects the surrogate."""
    node, pair = masks([7])
    statistics = decoder_statistics(
        logits_for_cycle(7), node, pair)
    targets = manifold_target_from(statistics)
    guided, report = guide(autoencoder, embeddings_for(node, seed=53), node, pair,
                           readout=None, targets=targets, qed_target=0.6,
                           lambda_qed=0.0, lambda_manifold=1.0, steps=5, step_size=0.2,
                           trust_radius=0.8)
    assert torch.isfinite(guided).all()
    assert report["nonfinite_steps"] == 0
    assert report["displacement_mean"] > 0.0


def test_qed_arm_without_a_readout_is_refused_rather_than_silently_skipped(autoencoder):
    node, pair = masks([7])
    features, statistics = features_with_grad(
        autoencoder, embeddings_for(node).requires_grad_(True), node, pair)
    with pytest.raises(ValueError):
        guidance_loss(None, features, statistics, None, qed_target=0.6, lambda_qed=1.0,
                      lambda_manifold=0.0)


def test_manifold_arm_without_targets_is_refused(autoencoder):
    node, pair = masks([7])
    features, statistics = features_with_grad(
        autoencoder, embeddings_for(node).requires_grad_(True), node, pair)
    with pytest.raises(ValueError):
        guidance_loss(None, features, statistics, None, qed_target=0.6, lambda_qed=0.0,
                      lambda_manifold=1.0)


# ====================================================================================
# 7. The ridge surrogate: deterministic, linear, honestly scored.
# ====================================================================================
def ridge_data(rows=200, columns=6, seed=5, noise=0.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, columns))
    weights = np.linspace(-1.0, 1.0, columns)
    y = x @ weights + 0.3 + noise * rng.normal(size=rows)
    return x, y, weights


def test_ridge_recovers_a_linear_target():
    x, y, _ = ridge_data()
    readout, info = RidgeReadout.fit(x, y, target="qed")
    assert float(np.abs(readout.predict_numpy(x) - y).mean()) < 1e-2
    assert info["features"] == x.shape[1] and info["train_samples"] == x.shape[0]
    assert info["alpha_selected"] in dg.RIDGE_ALPHAS


def test_ridge_is_deterministic():
    x, y, _ = ridge_data()
    first = RidgeReadout.fit(x, y, target="qed")[0].to_dict()
    second = RidgeReadout.fit(x, y, target="qed")[0].to_dict()
    assert first == second


def test_ridge_torch_and_numpy_predictions_agree():
    x, y, _ = ridge_data()
    readout, _ = RidgeReadout.fit(x, y, target="qed")
    torch_prediction = readout.predict_torch(torch.as_tensor(x, dtype=torch.float64))
    assert np.allclose(torch_prediction.numpy(), readout.predict_numpy(x), atol=1e-8)


def test_ridge_gradient_is_point_dependent_through_the_decoder(autoencoder):
    """A LINEAR readout still gives a LOCAL direction in x, because J_phi(x) varies.

    This is the property the fixed-latent-direction experiment lacked, and the reason a
    linear surrogate is not equivalent to a single global translation.
    """
    node, pair = masks([8])
    readout = fitted_readout(autoencoder)
    directions = []
    for seed in (61, 67):
        probe = embeddings_for(node, seed=seed).requires_grad_(True)
        features, _ = features_with_grad(autoencoder, probe, node, pair)
        (gradient,) = torch.autograd.grad(readout.predict_torch(features).sum(), probe)
        flat = gradient.flatten()
        directions.append(flat / flat.norm().clamp(min=1e-12))
    cosine = float((directions[0] * directions[1]).sum())
    assert abs(cosine) < 0.99, "the surrogate gradient collapsed to one fixed direction"


def test_ridge_refuses_an_underpowered_fit():
    x, y, _ = ridge_data(rows=10)
    with pytest.raises(ValueError):
        RidgeReadout.fit(x, y, target="qed")


def test_ridge_refuses_misaligned_features_and_targets():
    x, y, _ = ridge_data()
    with pytest.raises(ValueError):
        RidgeReadout.fit(x, y[:-5], target="qed")


def test_ridge_tolerates_a_constant_feature():
    x, y, _ = ridge_data()
    x[:, 0] = 1.0
    readout, _ = RidgeReadout.fit(x, y, target="qed")
    assert np.isfinite(readout.predict_numpy(x)).all()


def test_readout_quality_reports_mae_bias_and_correlation_per_n():
    x, y, _ = ridge_data(noise=0.05)
    readout, _ = RidgeReadout.fit(x, y, target="qed")
    groups = np.array([15] * 100 + [30] * 100)
    quality = readout_quality(readout, x, y, groups=groups)
    assert quality["target"] == "qed"
    assert set(quality["per_n"]) == {"15", "30"}
    for block in (quality["overall"], quality["per_n"]["15"], quality["per_n"]["30"]):
        assert block["mae"] is not None and block["correlation"] > 0.9
        assert "bias" in block and "truth_std" in block


def test_readout_quality_does_not_invent_a_correlation_from_too_few_points():
    x, y, _ = ridge_data()
    readout, _ = RidgeReadout.fit(x, y, target="qed")
    quality = readout_quality(readout, x[:2], y[:2])
    assert quality["overall"]["mae"] is None
    assert quality["overall"]["correlation"] is None


def test_readout_quality_returns_none_correlation_for_a_constant_truth():
    x, y, _ = ridge_data()
    readout, _ = RidgeReadout.fit(x, y, target="qed")
    quality = readout_quality(readout, x, np.full(x.shape[0], 0.5))
    assert quality["overall"]["correlation"] is None
    assert quality["overall"]["mae"] is not None


def test_readout_round_trips_through_its_dict():
    x, y, _ = ridge_data()
    readout, _ = RidgeReadout.fit(x, y, target="qed")
    blob = readout.to_dict()
    rebuilt = RidgeReadout(np.asarray(blob["weights"]), blob["bias"],
                           np.asarray(blob["feature_mean"]),
                           np.asarray(blob["feature_scale"]), blob["alpha"],
                           blob["target"])
    assert np.allclose(rebuilt.predict_numpy(x), readout.predict_numpy(x))


# ====================================================================================
# 8. The pilot script: hard gate, read-only, honest framing, CLI contract.
# ====================================================================================
def test_surrogate_thresholds_are_precommitted_constants():
    module = pilot()
    assert module.SURROGATE_MAX_MAE_REAL == 0.08
    assert module.SURROGATE_MIN_CORR_REAL == 0.45
    assert module.SURROGATE_MAX_MAE_GENERATED == 0.12
    assert module.SURROGATE_MIN_CORR_GENERATED == 0.30
    # the generated thresholds are the binding ones and must be no stricter than real
    assert module.SURROGATE_MAX_MAE_GENERATED >= module.SURROGATE_MAX_MAE_REAL


def test_the_five_arms_include_the_matched_norm_control_and_native():
    assert pilot().ARMS == ("native", "qed_only", "manifold_only", "combined", "random")


def test_main_returns_an_exit_code_rather_than_raising():
    module = pilot()
    assert module.main(["--flow-checkpoint", "/nope", "--output-dir", "/dev/null/x"]) == 4


def test_bad_grid_arguments_are_rejected_with_code_four(tmp_path):
    module = pilot()
    base = ["--flow-checkpoint", "/nope", "--output-dir", str(tmp_path / "out")]
    assert module.main(base + ["--qed-values", "0.6"]) == 4        # need two targets
    assert module.main(base + ["--target-n", ""]) == 4
    assert module.main(base + ["--samples-per-cell", "1"]) == 4
    assert module.main(base + ["--guidance-steps", "0"]) == 4
    assert module.main(base + ["--decoders", ""]) == 4


def test_a_valid_configuration_passes_validation_and_fails_later_on_the_checkpoint(tmp_path):
    """Exit 2 proves argument validation passed and the missing INPUT is what stopped it."""
    module = pilot()
    assert module.main(["--flow-checkpoint", str(tmp_path / "absent.pt"),
                        "--output-dir", str(tmp_path / "out"),
                        "--data-dir", str(tmp_path / "absent-data")]) in (1, 2)


def test_the_pilot_trains_nothing_and_saves_no_checkpoint():
    calls = _calls_in(SCRIPT_PATH)
    for forbidden in ("torch.save", "backward"):
        assert not any(forbidden in call for call in calls), forbidden
    source = SCRIPT_PATH.read_text()
    for forbidden in ("torch.optim", "Optimizer", "Trainer(", ".step()", "zero_grad",
                      "requires_grad_(True)"):
        assert forbidden not in source, forbidden


def test_the_pilot_freezes_both_models():
    source = SCRIPT_PATH.read_text()
    assert "flow.eval().requires_grad_(False)" in source
    assert "autoencoder.eval().requires_grad_(False)" in source


def test_the_pilot_never_loads_the_test_split():
    source = SCRIPT_PATH.read_text()
    assert "test_records" not in source and "load_test" not in source


def test_the_pilot_checks_autoencoder_provenance_and_records_the_override():
    source = SCRIPT_PATH.read_text()
    assert "ResumeMismatchError" in source
    assert "sha256_match" in source and "mismatch_override_used" in source


def test_the_report_is_labelled_as_guidance_not_conditioning():
    source = SCRIPT_PATH.read_text()
    assert "NOT intrinsic conditional generation" in source
    assert "GUIDANCE, NOT CONDITIONING" in source


def test_the_report_puts_surrogate_qed_next_to_actual_qed():
    """The exploitation detector. Without both numbers the result is uninterpretable."""
    source = SCRIPT_PATH.read_text()
    assert "surrogate_minus_actual" in source
    assert "surrogate-actual" in source


def test_raw_and_corrected_validity_are_reported_separately():
    source = SCRIPT_PATH.read_text()
    assert "degree,constrained" in source          # both decoders, both validity numbers
    assert 'entry["decoders"][d] = summarise(' in source


def test_runtime_is_measured_and_not_assumed():
    source = SCRIPT_PATH.read_text()
    assert "MEASURED, not assumed" in source
    assert "perf_counter" in source


def test_the_pilot_defaults_to_the_bounded_sample_size():
    args = pilot().parse_args(["--flow-checkpoint", "f", "--output-dir", "o"])
    assert args.samples_per_cell == 8
    assert args.device == "cpu"
    assert args.target_n == "15,23,30"
    assert args.qed_values == "0.4,0.6,0.8"


def test_qed_arms_are_skipped_when_phase_a_rejects_the_surrogate():
    """The hard gate is in code, not in the write-up."""
    source = SCRIPT_PATH.read_text()
    assert 'if arm in ("qed_only", "combined") and not qed_enabled:' in source
    assert "surrogate failed Phase A" in source


def test_the_native_arm_is_evaluated_first_so_pairing_is_possible():
    module = pilot()
    assert module.ARMS[0] == "native"


def test_summarise_pairs_only_jointly_valid_samples():
    module = pilot()
    rows = [{"valid": True, "smiles": "CCO", "qed": 0.5, "aromatic_rings": 0.0,
             "heavy_atoms": 3.0},
            {"valid": False, "smiles": None}]
    native = [{"valid": True, "smiles": "CCN", "qed": 0.4, "aromatic_rings": 0.0,
               "heavy_atoms": 3.0},
              {"valid": True, "smiles": "CCC", "qed": 0.45, "aromatic_rings": 0.0,
               "heavy_atoms": 3.0}]
    block = module.summarise(rows, qed_target=0.6, baseline=native)
    assert block["paired_jointly_valid"] == 1
    assert block["paired_qed_delta_vs_native"]["mean"] == pytest.approx(0.1)
    assert block["validity"] == 0.5


def test_summarise_reports_mae_against_the_request_not_just_the_mean():
    module = pilot()
    rows = [{"valid": True, "smiles": "CCO", "qed": 0.5, "aromatic_rings": 0.0,
             "heavy_atoms": 3.0}]
    block = module.summarise(rows, qed_target=0.8)
    assert block["qed_mae_vs_target"] == pytest.approx(0.3)
    assert block["qed_mean"] == pytest.approx(0.5)


def test_summarise_handles_an_all_invalid_cell():
    module = pilot()
    block = module.summarise([{"valid": False, "smiles": None}] * 4, qed_target=0.6)
    assert block["validity"] == 0.0 and block["qed_mae_vs_target"] is None
    assert block["unique_smiles"] == 0


def test_summarise_reports_size_so_a_size_driven_qed_change_is_visible():
    module = pilot()
    rows = [{"valid": True, "smiles": "CCO", "qed": 0.5, "aromatic_rings": 1.0,
             "heavy_atoms": 12.0}]
    block = module.summarise(rows, qed_target=0.6)
    assert block["heavy_atoms_mean"] == pytest.approx(12.0)
    assert block["aromatic_rings_mean"] == pytest.approx(1.0)


def test_bootstrap_ci_flags_an_interval_that_excludes_zero():
    module = pilot()
    assert module.bootstrap_ci([0.2] * 40)["excludes_zero"] is True
    assert module.bootstrap_ci([-0.5, 0.5] * 20)["excludes_zero"] is False
    assert module.bootstrap_ci([])["mean"] is None


def test_pair_masks_exclude_the_diagonal_and_the_padding():
    module = pilot()
    node = torch.tensor([[True, True, False]])
    pair = module.pair_masks_for(node)
    assert pair.tolist() == [[[False, True, False], [True, False, False],
                              [False, False, False]]]


# ====================================================================================
# 9. The SLURM wrapper matches the CLI and refuses the login node.
# ====================================================================================
def test_slurm_flags_are_a_subset_of_the_scripts_arguments():
    """A typo here costs a queued job, so the wrapper and the CLI are pinned together."""
    parser_flags = {line.strip().split('"')[1]
                    for line in SCRIPT_PATH.read_text().splitlines()
                    if line.strip().startswith('p.add_argument("--')}
    assert parser_flags, "could not read the argparse flags"
    used = set()
    for line in SLURM_PATH.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith("--"):
            used.add(stripped.split()[0])
    assert used, "could not read the flags the wrapper passes"
    assert used <= parser_flags, used - parser_flags
    # the flags that carry the pilot's bounds must actually be passed, not defaulted
    for required in ("--flow-checkpoint", "--ae-checkpoint", "--output-dir",
                     "--samples-per-cell", "--guidance-steps", "--target-n",
                     "--qed-values", "--device"):
        assert required in used, required


def test_slurm_refuses_to_run_outside_slurm():
    source = SLURM_PATH.read_text()
    assert 'if [[ -z "${SLURM_JOB_ID:-}" ]]' in source
    assert "do not run on the login node" in source
    assert "set -euo pipefail" in source


def test_slurm_requests_cpu_and_bounded_walltime():
    source = SLURM_PATH.read_text()
    assert "#SBATCH --partition=CPU" in source
    assert "--device cpu" in source
    assert "#SBATCH --time=04:00:00" in source


def test_slurm_defaults_to_the_bounded_pilot_not_the_full_run():
    source = SLURM_PATH.read_text()
    assert 'SAMPLES="${SAMPLES:-8}"' in source
    assert "DO NOT scale up" in source


def test_slurm_checks_provenance_rather_than_waiving_it():
    source = SLURM_PATH.read_text()
    assert "provenance CHECKED" in source
    assert "do not set ALLOW_AE_MISMATCH=1" in source


def test_slurm_writes_only_into_its_own_output_directory():
    source = SLURM_PATH.read_text()
    assert 'OUT="${OUT:-outputs/metrics/decoder_guidance_pilot}"' in source
    # no destructive command anywhere in the executable part of the wrapper
    executable = [line for line in source.splitlines()
                  if line.strip() and not line.strip().startswith("#")]
    for line in executable:
        first = line.strip().split()[0]
        assert first not in {"rm", "mv", "cp", "truncate", "dd", "chmod", "chown"}, line
    assert "mkdir -p outputs/logs" in source        # it only ever creates directories


def test_slurm_verifies_its_inputs_exist_before_launching():
    source = SLURM_PATH.read_text()
    for required in ("splits.pkl", "scaler.json"):
        assert required in source


def test_slurm_points_at_the_final_checkpoints():
    source = SLURM_PATH.read_text()
    assert "flow_nqed_ae128_qedcontrast/seed_3/best.pt" in source
    assert "ae_degree_128d/seed_3/last.pt" in source
