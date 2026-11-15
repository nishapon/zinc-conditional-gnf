"""Experiment C: the bond-aware pairwise coupling conditioner.

Four groups, in the order the claims have to hold:

  FLOW MATHEMATICS   inversion, the log-determinant against a numerical Jacobian,
                     padding, and node permutation. If any of these fail the pathway is
                     not a normalizing flow any more and nothing else matters.
  PARENT PRESERVATION the warm-started arm reproduces the parent density exactly, and the
                     only state-dict keys it adds are the new pathway's.
  THE PATHWAY IS ALIVE gradients actually reach it, and a nonzero bridge actually changes
                     the conditioner's output. Both directions matter: an identity-
                     initialised pathway that never engages looks exactly like a working
                     one in the loss curve.
  CLASS IMBALANCE    "no bond anywhere" must not look successful.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from zinc_gnf.bond_pairwise import (  # noqa: E402
    BondPairwiseConditioner,
    bond_loss_and_metrics,
    bond_pathway_report,
    bond_targets,
    clear_published_logits,
    collect_published_logits,
    pair_mask_from_node_mask,
    set_publishing,
    upper_triangle_mask,
)
from zinc_gnf.flow import FlowConfig, NodeFlow, node_mask_from_counts  # noqa: E402

DIM = 8
CONDITION = 8


def config(*, bond_aware=True, pair_dim=4, autoregressive=False):
    return FlowConfig(node_embedding_dim=DIM, num_blocks=2, hidden_dim=16,
                      attention_dim=8, num_heads=2,
                      node_autoregressive=autoregressive,
                      bond_aware_coupling=bond_aware, bond_pair_dim=pair_dim,
                      condition_features=("n", "qed"),
                      condition_embedding_dim=CONDITION, condition_hidden_dim=8)


def pretend_trained(flow, *, seed=21, std=0.05):
    """Give the zero-initialised output heads nonzero weights.

    Necessary, not cosmetic. `CouplingNetwork` zero-initialises `scale_head` and
    `translation_head`, so a FRESHLY CONSTRUCTED flow is the exact identity: its
    log-determinant is 0, and `d(scale)/d(hidden)` is 0, which means every gradient
    through the conditioner vanishes no matter what the new pathway does. Tests run on
    such a flow would pass without testing anything.

    The parent checkpoint these arms warm-start from has trained heads, so filling them in
    is what makes the tests see the situation the arms actually run in.
    """
    generator = torch.Generator().manual_seed(seed)
    for name, module in flow.named_modules():
        if not name.endswith(("scale_head", "translation_head", "film")):
            continue
        if not hasattr(module, "weight"):
            continue
        with torch.no_grad():
            module.weight.copy_(
                torch.randn(module.weight.shape, generator=generator) * std)
            if getattr(module, "bias", None) is not None:
                module.bias.copy_(
                    torch.randn(module.bias.shape, generator=generator) * std)
    return flow


def built(*, bond_aware=True, pair_dim=4, autoregressive=False, seed=0,
          trained=True):
    """A flow with statistics set, heads filled in and ActNorm initialised; EVAL mode.

    ActNorm initialises itself from the first batch it sees while training, so every test
    that compares two flows or differentiates one has to get that out of the way first.
    """
    flow = NodeFlow(config(bond_aware=bond_aware, pair_dim=pair_dim,
                           autoregressive=autoregressive), init_seed=seed)
    generator = torch.Generator().manual_seed(1)
    flow.set_data_statistics(torch.randn(DIM, generator=generator),
                             torch.rand(DIM, generator=generator) + 0.5)
    if trained:
        pretend_trained(flow)
    flow.train()
    with torch.no_grad():
        flow.log_prob(torch.randn((12, 5, DIM),
                                  generator=torch.Generator().manual_seed(2)),
                      torch.ones((12, 5), dtype=torch.bool),
                      torch.zeros((12, CONDITION)))
    flow.eval()
    return flow


def inputs(counts=(4, 3), width=None, seed=5):
    mask = node_mask_from_counts(counts, max_nodes=width)
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((len(counts), mask.shape[1], DIM), generator=generator)
    x = x * mask.unsqueeze(-1)
    condition = torch.randn((len(counts), CONDITION), generator=generator)
    return x, mask, condition


# ============================== flow mathematics ================================
@pytest.mark.parametrize("autoregressive", [False, True])
def test_the_bond_aware_flow_inverts_exactly(autoregressive):
    """The conditioner reads the pass-through half, so the inverse recomputes it."""
    flow = built(autoregressive=autoregressive)
    x, mask, condition = inputs()
    with torch.no_grad():
        z, log_det = flow.forward(x, mask, condition)
        recovered = flow.inverse(z, mask, condition)
    error = float(((recovered - x) * mask.unsqueeze(-1)).abs().max())
    assert error < 1e-4, f"round-trip error {error}"
    # the flow must not be the identity, or the round trip proves nothing
    assert float(log_det.abs().max()) > 0.0


def test_the_log_determinant_matches_a_numerical_jacobian():
    """The analytic log-det must equal log|det| of the autograd Jacobian.

    This is the test that would catch a conditioner accidentally reading the TRANSFORMED
    half: the Jacobian would stop being triangular and the closed form would be wrong,
    while inversion alone might still look fine.
    """
    flow = built(pair_dim=3).double()
    width = 4
    mask = node_mask_from_counts((width,))
    condition = torch.randn((1, CONDITION),
                            generator=torch.Generator().manual_seed(8)).double()
    base = torch.randn((1, width, DIM),
                       generator=torch.Generator().manual_seed(9)).double()

    def transform(flat):
        z, _ = flow.forward(flat.view(1, width, DIM), mask, condition)
        return z.reshape(-1)

    jacobian = torch.autograd.functional.jacobian(transform, base.reshape(-1))
    numerical = torch.linalg.slogdet(jacobian)[1]
    with torch.no_grad():
        _, analytic = flow.forward(base, mask, condition)
    assert float(analytic[0]) != 0.0, "a zero log-det would make this vacuous"
    assert float(numerical) == pytest.approx(float(analytic[0]), abs=1e-7), (
        f"numerical {float(numerical)} vs analytic {float(analytic[0])}")


def test_padding_cannot_change_a_real_molecule():
    """A padded column must contribute to neither the output nor the log-determinant."""
    flow = built()
    narrow_x, narrow_mask, condition = inputs(counts=(3,), width=3)
    wide_mask = node_mask_from_counts((3,), max_nodes=7)
    wide_x = torch.zeros((1, 7, DIM))
    wide_x[:, :3] = narrow_x
    # deliberately fill the padding with large garbage
    wide_x[:, 3:] = 99.0
    with torch.no_grad():
        narrow_z, narrow_det = flow.forward(narrow_x, narrow_mask, condition)
        wide_z, wide_det = flow.forward(wide_x, wide_mask, condition)
    assert torch.allclose(wide_z[:, :3], narrow_z, atol=1e-6)
    assert float((wide_det - narrow_det).abs().max()) < 1e-6
    assert float(wide_z[:, 3:].abs().max()) == 0.0


def test_the_pair_gate_is_zero_on_padding_and_the_diagonal():
    conditioner = BondPairwiseConditioner(DIM, pair_dim=4, condition_dim=CONDITION)
    mask = node_mask_from_counts((2, 4), max_nodes=5)
    source = torch.randn((2, 5, DIM), generator=torch.Generator().manual_seed(3))
    _, logits = conditioner(source * mask.unsqueeze(-1), mask,
                            torch.zeros((2, CONDITION)))
    # detach before any scalar conversion: the conditioner's parameters require grad, so
    # float() on a graph-attached tensor warns
    gate = torch.sigmoid(logits).detach()
    pair_mask = pair_mask_from_node_mask(mask)
    assert float(gate[~pair_mask].abs().max()) == 0.0
    assert float(torch.diagonal(gate, dim1=1, dim2=2).abs().max()) == 0.0
    # and the real block is not degenerate
    assert float(gate[pair_mask].max()) > 0.0


def test_the_pair_logits_are_exactly_symmetric():
    """A bond is a symmetric relation; Q K^T is not. This is one of the distinctions.

    The assertion is bitwise on purpose. `b_ij != b_ji` by ANY amount means one unordered
    pair contributes two different values to the two nodes' degrees, and the auxiliary
    loss scores a symmetric relation as though direction mattered.
    """
    conditioner = BondPairwiseConditioner(DIM, pair_dim=5, condition_dim=CONDITION)
    with torch.no_grad():
        conditioner.product_weight.normal_()
    mask = torch.ones((3, 6), dtype=torch.bool)
    source = torch.randn((3, 6, DIM), generator=torch.Generator().manual_seed(4))
    condition = torch.randn((3, CONDITION))
    _, logits = conditioner(source, mask, condition)
    assert torch.equal(logits, logits.transpose(1, 2))

    # and the roundoff that made this fail before is measured, not assumed: the formula
    # is symmetric, so the raw gap must sit at float32 roundoff for the logit scale
    report = conditioner.asymmetry_report(source, mask, condition)
    assert report["valid_pairs"] > 0
    assert report["symmetrised_max_abs_asymmetry"] == 0.0
    assert report["raw_nan"] == 0 and report["symmetrised_nan"] == 0
    assert report["non_finite_on_valid_pairs"] == 0
    assert report["raw_relative_asymmetry"] < 1e-5, (
        f"asymmetry too large to be roundoff, trace the bilinear term first: {report}")


def test_the_symmetrisation_survives_padding_and_a_masked_batch():
    """-inf must never enter the symmetrisation arithmetic and produce a NaN."""
    conditioner = BondPairwiseConditioner(DIM, pair_dim=4, condition_dim=CONDITION)
    with torch.no_grad():
        conditioner.product_weight.normal_()
    mask = node_mask_from_counts((1, 3, 6), max_nodes=6)
    source = torch.randn((3, 6, DIM), generator=torch.Generator().manual_seed(23))
    condition = torch.randn((3, CONDITION))
    _, logits = conditioner(source * mask.unsqueeze(-1), mask, condition)
    assert torch.equal(logits, logits.transpose(1, 2))
    assert not bool(torch.isnan(logits).any())
    report = conditioner.asymmetry_report(source * mask.unsqueeze(-1), mask, condition)
    assert report["symmetrised_nan"] == 0
    assert report["symmetrised_max_abs_asymmetry"] == 0.0
    # the single-atom molecule has no valid pair at all, and that is not an error
    assert float(torch.sigmoid(logits[0]).detach().abs().max()) == 0.0


def test_a_node_permutation_permutes_the_scores_consistently():
    """Permuting the atoms must permute the score matrix, not change its contents."""
    conditioner = BondPairwiseConditioner(DIM, pair_dim=4, condition_dim=CONDITION)
    with torch.no_grad():
        conditioner.product_weight.normal_()
    mask = torch.ones((1, 5), dtype=torch.bool)
    source = torch.randn((1, 5, DIM), generator=torch.Generator().manual_seed(6))
    condition = torch.randn((1, CONDITION))
    order = torch.tensor([3, 0, 4, 1, 2])
    features, logits = conditioner(source, mask, condition)
    permuted_features, permuted_logits = conditioner(source[:, order], mask, condition)
    assert torch.allclose(permuted_logits, logits[:, order][:, :, order], atol=1e-6)
    assert torch.allclose(permuted_features, features[:, order], atol=1e-6)


def test_the_soft_degree_counts_neighbours_rather_than_normalising():
    """The distinction from softmax attention, stated as a test.

    Softmax weights sum to 1 regardless of how many neighbours are active, so an attention
    head cannot represent absolute degree. With every gate driven to 1, the soft degree
    must equal the number of real neighbours.
    """
    conditioner = BondPairwiseConditioner(DIM, pair_dim=4, condition_dim=0)
    with torch.no_grad():
        conditioner.bias.fill_(40.0)             # sigmoid saturates at 1
        conditioner.sum_head.weight.zero_()
        conditioner.sum_head.bias.zero_()
        conditioner.absolute_head.weight.zero_()
        conditioner.product_weight.zero_()
    counts = (2, 5, 7)
    mask = node_mask_from_counts(counts, max_nodes=8)
    source = torch.randn((3, 8, DIM), generator=torch.Generator().manual_seed(7))
    features, logits = conditioner(source * mask.unsqueeze(-1), mask, None)
    gate = (torch.sigmoid(logits).detach()
            * pair_mask_from_node_mask(mask).to(torch.float32))
    degree = gate.sum(-1)
    for row, count in enumerate(counts):
        assert float(degree[row, 0]) == pytest.approx(count - 1, abs=1e-4), count
    # and the feature the conditioner consumes carries it
    expected = torch.log1p(degree)
    assert torch.allclose(features.detach()[..., -2], expected * mask, atol=1e-5)


def test_the_conditioner_never_sees_a_bond_matrix():
    """Generation time has no bonds, so the signature must not admit one."""
    import inspect

    parameters = list(inspect.signature(BondPairwiseConditioner.forward)
                      .parameters)
    assert parameters == ["self", "source", "mask", "condition"]
    flow = built()
    mask = node_mask_from_counts((4, 6))
    with torch.no_grad():
        drawn = flow.sample(mask, torch.zeros((2, CONDITION)),
                            generator=torch.Generator().manual_seed(11))
    assert drawn.shape == (2, 6, DIM) and torch.isfinite(drawn).all()


# ============================ parent preservation ===============================
def test_the_warm_started_arm_reproduces_the_parent_density_exactly():
    """Zero bridge plus the parent's own weights must give the parent's likelihood."""
    parent = built(bond_aware=False)
    state = parent.state_dict()
    arm = NodeFlow(config(bond_aware=True), init_seed=0)
    incompatible = arm.load_state_dict(state, strict=False)
    assert incompatible.unexpected_keys == []
    arm.eval()
    x, mask, condition = inputs(counts=(4, 6))
    with torch.no_grad():
        before = parent.log_prob(x, mask, condition)["log_prob"]
        after = arm.log_prob(x, mask, condition)["log_prob"]
    assert torch.equal(before, after), (
        f"the pathway is not identity-initialised: {before.tolist()} vs {after.tolist()}")


def test_the_only_unloaded_keys_are_the_new_pathway():
    """A typo in a new parameter name would otherwise hide inside `strict=False`."""
    parent = built(bond_aware=False)
    arm = NodeFlow(config(bond_aware=True), init_seed=0)
    missing = arm.load_state_dict(parent.state_dict(), strict=False).missing_keys
    assert missing, "nothing new was added"
    for key in missing:
        assert ".bond." in key or key.endswith("pair_to_hidden.weight"), key
    assert not set(missing) & set(parent.state_dict())


def test_turning_the_pathway_off_leaves_the_model_byte_identical():
    """The control arm has to be the pre-change model exactly, weights included.

    `CouplingNetwork` initialises its Linear layers from the GLOBAL rng, so if the new
    modules were constructed even when disabled, they would consume draws and every
    downstream layer would initialise differently. The control would then differ from the
    parent architecture for a reason that has nothing to do with the experiment. Both
    flows are built after the same global seed, so identical weights prove the disabled
    path consumes nothing.
    """
    legacy = {key: value for key, value in config(bond_aware=False).to_dict().items()
              if not key.startswith("bond_")}
    assert "bond_aware_coupling" not in legacy      # an old checkpoint's config
    torch.manual_seed(100)
    old_style = NodeFlow(FlowConfig.from_dict(legacy), init_seed=4)
    torch.manual_seed(100)
    new_style = NodeFlow(config(bond_aware=False), init_seed=4)
    left, right = old_style.state_dict(), new_style.state_dict()
    assert list(left) == list(right)
    for key in left:
        assert torch.equal(left[key], right[key]), key
    assert not any("bond" in key or "pair_to_hidden" in key for key in left)
    assert old_style.num_parameters() == new_style.num_parameters()


def test_the_pathway_report_counts_one_conditioner_per_affine_coupling():
    flow = built()
    report = bond_pathway_report(flow)
    assert report["conditioners"] == 2 * flow.config.num_blocks
    assert report["bridge_is_zero"] is True
    assert report["new_parameters"] > 0
    plain = bond_pathway_report(built(bond_aware=False))
    assert plain["conditioners"] == 0
    assert flow.num_parameters() - built(bond_aware=False).num_parameters() == \
        report["new_parameters"]


# ============================ the pathway is alive ==============================
def test_at_step_zero_the_bridge_has_gradient_and_the_scorer_does_not():
    """An honest consequence of zero-initialising the bridge, asserted rather than hoped.

    d(hidden)/d(bridge weight) = pair_features, which is nonzero, so the bridge moves on
    the first step. d(hidden)/d(scorer) carries a factor of the bridge weight, which is
    zero, so the scorer gets no LIKELIHOOD gradient until the bridge has moved.
    """
    flow = built()
    flow.train()
    x, mask, condition = inputs(counts=(4, 5))
    flow.negative_log_likelihood(x, mask, condition)["loss"].backward()
    network = flow.blocks[0].couplings[0].network
    bridge = network.pair_to_hidden.weight.grad
    assert bridge is not None and float(bridge.abs().max()) > 0.0
    scorer = network.bond.pair_projection.weight.grad
    assert scorer is None or float(scorer.abs().max()) == 0.0


def test_once_the_bridge_is_nonzero_the_scorer_receives_likelihood_gradient():
    flow = built()
    flow.train()
    for _, module in flow.named_modules():
        if hasattr(module, "pair_to_hidden") and module.pair_to_hidden is not None:
            with torch.no_grad():
                module.pair_to_hidden.weight.normal_(std=0.05)
    x, mask, condition = inputs(counts=(4, 5))
    flow.negative_log_likelihood(x, mask, condition)["loss"].backward()
    scorer = flow.blocks[0].couplings[0].network.bond.pair_projection.weight.grad
    assert scorer is not None
    assert float(scorer.abs().max()) > 0.0


def test_bond_supervision_reaches_the_scorer_even_with_a_zero_bridge():
    """Why the supervised arm engages at step 0 while the likelihood-only arm waits."""
    conditioner = BondPairwiseConditioner(DIM, pair_dim=4, condition_dim=0)
    mask = torch.ones((2, 5), dtype=torch.bool)
    source = torch.randn((2, 5, DIM), generator=torch.Generator().manual_seed(12))
    _, logits = conditioner(source, mask, None)
    bonds = torch.zeros((2, 5, 5), dtype=torch.long)
    bonds[:, 0, 1] = bonds[:, 1, 0] = 1
    bonds[:, 2, 3] = bonds[:, 3, 2] = 2
    result = bond_loss_and_metrics([logits], bonds, mask)
    result["loss"].backward()
    assert float(conditioner.pair_projection.weight.grad.abs().max()) > 0.0


def test_a_nonzero_bridge_changes_the_conditioner_output():
    """The pathway must be able to move scale and translation, not just exist."""
    flow = built()
    x, mask, condition = inputs(counts=(4, 5))
    coupling = flow.blocks[0].couplings[0]
    source, _ = coupling._split(x)
    with torch.no_grad():
        before = coupling._scale_and_translation(source, mask, condition)
        coupling.network.pair_to_hidden.weight.normal_(std=0.5)
        after = coupling._scale_and_translation(source, mask, condition)
    scale_change = float((after[0] - before[0]).abs().max())
    translation_change = float((after[1] - before[1]).abs().max())
    assert scale_change > 0.0 and translation_change > 0.0, (
        f"scale moved {scale_change}, translation moved {translation_change}")


def test_the_flow_still_inverts_after_the_pathway_has_weight():
    """Identity initialisation must not be what makes inversion work."""
    flow = built()
    for _, module in flow.named_modules():
        if hasattr(module, "pair_to_hidden") and module.pair_to_hidden is not None:
            with torch.no_grad():
                module.pair_to_hidden.weight.normal_(std=0.3)
    x, mask, condition = inputs(counts=(5, 3))
    with torch.no_grad():
        z, _ = flow.forward(x, mask, condition)
        recovered = flow.inverse(z, mask, condition)
    assert float(((recovered - x) * mask.unsqueeze(-1)).abs().max()) < 1e-4


# ============================== publishing / staleness ==========================
def test_logits_are_published_only_while_training():
    flow = built()
    assert set_publishing(flow, True) == 2 * flow.config.num_blocks
    x, mask, condition = inputs()
    flow.eval()
    with torch.no_grad():
        flow.log_prob(x, mask, condition)
    with pytest.raises(RuntimeError, match="published no bond logits"):
        collect_published_logits(flow, batch_size=2, width=mask.shape[1])
    flow.train()
    flow.log_prob(x, mask, condition)
    found = collect_published_logits(flow, batch_size=2, width=mask.shape[1])
    assert len(found) == 2 * flow.config.num_blocks


def test_a_stale_tensor_from_another_batch_is_refused():
    """The shape check is what makes the module-stash approach safe."""
    flow = built()
    set_publishing(flow, True)
    flow.train()
    x, mask, condition = inputs(counts=(4, 3))
    flow.log_prob(x, mask, condition)
    with pytest.raises(RuntimeError, match="stale tensor"):
        collect_published_logits(flow, batch_size=7, width=mask.shape[1])
    assert clear_published_logits(flow) == 2 * flow.config.num_blocks
    with pytest.raises(RuntimeError, match="published no bond logits"):
        collect_published_logits(flow, batch_size=2, width=mask.shape[1])


def test_an_empty_logit_list_is_refused_rather_than_scored_as_zero():
    with pytest.raises(ValueError, match="no bond logits"):
        bond_loss_and_metrics([], torch.zeros((1, 3, 3), dtype=torch.long),
                              torch.ones((1, 3), dtype=torch.bool))


# ================================ class imbalance ===============================
FIXTURE_COUNTS = (10, 12)


def molecule_batch(counts=FIXTURE_COUNTS, width=12, seed=13):
    """A chain plus a ring closure, so the positive rate resembles a real molecule.

    Every real node therefore has EXACTLY 2 bonded neighbours and `count - 1` valid
    neighbours, which is what makes the expected soft degree computable in closed form.
    """
    mask = node_mask_from_counts(counts, max_nodes=width)
    bonds = torch.zeros((len(counts), width, width), dtype=torch.long)
    for row, count in enumerate(counts):
        for i in range(count - 1):
            bonds[row, i, i + 1] = bonds[row, i + 1, i] = 1
        bonds[row, 0, count - 1] = bonds[row, count - 1, 0] = 1
    return bonds, mask


def expected_soft_degree(counts, *, bonded_logit, other_logit):
    """The soft degree `bond_loss_and_metrics` must report, derived from the fixture.

    Independent of the implementation. The documented averaging rule is: average the
    layers' logits, take the sigmoid on the counted (upper-triangle) pairs, mirror it to
    a symmetric matrix, sum over j to get a per-node SOFT degree, then average over real
    nodes weighted by node count. With a chain-plus-ring fixture each node has 2 bonded
    neighbours and `count - 1 - 2` unbonded valid neighbours, so

        soft degree(node) = 2*sigmoid(bonded) + (count - 3)*sigmoid(other)

    This is an expected neighbour count under the gate, NOT a thresholded count: a
    confident "no bond" still contributes sigmoid(-8) = 3.3535e-4 per pair, so the value
    for an all-negative predictor is small but provably nonzero.
    """
    import math

    assert all(count >= 3 for count in counts), (
        "below 3 atoms the fixture's ring closure coincides with a chain bond, so each "
        "node has 1 bonded neighbour rather than 2 and this formula does not apply")
    sigmoid = lambda z: 1.0 / (1.0 + math.exp(-z))  # noqa: E731
    nodes = sum(counts)
    total = 0.0
    for count in counts:
        per_node = (2 * sigmoid(bonded_logit)
                    + max(count - 1 - 2, 0) * sigmoid(other_logit))
        total += count * per_node
    return total / nodes


def test_predicting_no_bond_everywhere_scores_zero_f1_and_a_worse_loss():
    """The requirement stated directly: the trivial predictor must not look successful."""
    bonds, mask = molecule_batch()
    target = bond_targets(bonds)
    counted = upper_triangle_mask(pair_mask_from_node_mask(mask))
    confident = torch.where(target > 0.5, 8.0, -8.0)
    trivial = torch.full_like(confident, -8.0)

    good = bond_loss_and_metrics([confident], bonds, mask)
    bad = bond_loss_and_metrics([trivial], bonds, mask)

    # accuracy ALONE makes the trivial predictor look fine ...
    assert bad["metrics"]["bond_accuracy"] > 0.70
    # ... which is why the trivial baseline is reported next to it
    assert bad["metrics"]["bond_accuracy"] == pytest.approx(
        bad["metrics"]["bond_trivial_accuracy"], abs=1e-9)
    # F1 is zero, because recall is zero. These are the checks that matter.
    assert bad["metrics"]["bond_recall"] == 0.0
    assert bad["metrics"]["bond_f1"] == 0.0
    assert bad["metrics"]["bond_precision"] == 0.0
    # The SOFT degree is not zero and must not be asserted to be: a confident -8 still
    # carries sigmoid(-8) = 3.3535e-4 of probability per pair. The expected value is
    # computed from the fixture instead, so the metric is pinned exactly rather than
    # loosely bounded.
    assert bad["metrics"]["bond_predicted_degree"] == pytest.approx(
        expected_soft_degree(FIXTURE_COUNTS, bonded_logit=-8.0, other_logit=-8.0),
        rel=1e-5)
    # what makes it useless is its size relative to the truth, not its being zero
    assert bad["metrics"]["bond_predicted_degree"] < 0.01 * bad["metrics"][
        "bond_true_degree"]
    # and the reweighted loss is strictly worse than getting the bonds right
    assert float(bad["loss"]) > float(good["loss"])
    assert good["metrics"]["bond_f1"] == pytest.approx(1.0)
    assert good["metrics"]["bond_predicted_degree"] == pytest.approx(
        expected_soft_degree(FIXTURE_COUNTS, bonded_logit=8.0, other_logit=-8.0),
        rel=1e-5)
    # the imbalance is real in this fixture (a real N=23 molecule is near 0.10)
    positives = float((target * counted.to(target.dtype)).sum())
    assert positives / float(counted.sum()) < 0.25


@pytest.mark.parametrize("counts,width", [((10, 12), 12), ((4,), 4), ((3, 9, 12), 12),
                                          ((5, 5), 9)])
def test_the_soft_degree_matches_an_independent_calculation(counts, width):
    """Pins the metric's masking AND its denominator, not just its magnitude.

    Padded rows must be excluded from both the sum and the node count, so the varied
    widths here would catch a denominator of `batch * width` instead of real nodes, and
    the single-molecule and ragged cases would catch a per-molecule average being passed
    off as a per-node one.
    """
    bonds, mask = molecule_batch(counts=counts, width=width)
    target = bond_targets(bonds)
    for bonded, other in ((-8.0, -8.0), (8.0, -8.0), (2.0, -1.0)):
        logits = torch.where(target > 0.5, bonded, other)
        metrics = bond_loss_and_metrics([logits], bonds, mask)["metrics"]
        assert metrics["bond_predicted_degree"] == pytest.approx(
            expected_soft_degree(counts, bonded_logit=bonded, other_logit=other),
            rel=1e-5), (counts, bonded, other)
    # the true degree is 2 everywhere in this fixture, by construction
    assert bond_loss_and_metrics([torch.zeros_like(target)], bonds,
                                 mask)["metrics"]["bond_true_degree"] == \
        pytest.approx(2.0, rel=1e-6)


def test_the_soft_degree_is_unchanged_by_padding_width():
    """Same molecules, more padding: a degree that moved would prove a masking bug."""
    counts = (6, 9)
    narrow = bond_loss_and_metrics(
        [torch.zeros((2, 9, 9))], *molecule_batch(counts=counts, width=9))["metrics"]
    bonds, mask = molecule_batch(counts=counts, width=14)
    wide = bond_loss_and_metrics([torch.zeros((2, 14, 14))], bonds, mask)["metrics"]
    assert wide["bond_predicted_degree"] == pytest.approx(
        narrow["bond_predicted_degree"], rel=1e-6)
    assert wide["bond_pos_rate"] == pytest.approx(narrow["bond_pos_rate"], rel=1e-6)
    assert wide["bond_true_degree"] == pytest.approx(narrow["bond_true_degree"],
                                                     rel=1e-6)


def test_the_positive_weight_is_the_measured_ratio_not_a_constant():
    bonds, mask = molecule_batch()
    counted = upper_triangle_mask(pair_mask_from_node_mask(mask))
    target = bond_targets(bonds)
    positives = float((target * counted.to(target.dtype)).sum())
    pairs = float(counted.sum())
    metrics = bond_loss_and_metrics([torch.zeros_like(target)], bonds,
                                    mask)["metrics"]
    assert metrics["bond_pos_weight"] == pytest.approx(
        (pairs - positives) / positives, rel=1e-6)
    assert metrics["bond_pos_rate"] == pytest.approx(positives / pairs, rel=1e-6)


def test_the_positive_weight_is_capped():
    """One almost-bond-free batch must not produce an unbounded gradient."""
    mask = torch.ones((1, 20), dtype=torch.bool)
    bonds = torch.zeros((1, 20, 20), dtype=torch.long)
    bonds[0, 0, 1] = bonds[0, 1, 0] = 1
    metrics = bond_loss_and_metrics([torch.zeros((1, 20, 20))], bonds,
                                    mask)["metrics"]
    assert metrics["bond_pos_weight"] == 50.0


def test_each_unordered_pair_is_counted_once():
    bonds, mask = molecule_batch(counts=(4,), width=4)
    counted = upper_triangle_mask(pair_mask_from_node_mask(mask))
    assert int(counted.sum()) == 4 * 3 // 2
    assert bool((counted & counted.transpose(1, 2)).sum() == 0)


def test_padded_pairs_contribute_nothing_to_the_loss():
    """A -inf logit on padding must not make the loss non-finite."""
    bonds, mask = molecule_batch(counts=(3, 8), width=8)
    conditioner = BondPairwiseConditioner(DIM, pair_dim=4, condition_dim=0)
    source = torch.randn((2, 8, DIM), generator=torch.Generator().manual_seed(14))
    _, logits = conditioner(source * mask.unsqueeze(-1), mask, None)
    assert bool(torch.isinf(logits).any()), "padding should be -inf before masking"
    result = bond_loss_and_metrics([logits], bonds, mask)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert torch.isfinite(conditioner.pair_projection.weight.grad).all()


def test_the_loss_is_the_mean_over_layers_so_depth_does_not_reweight_it():
    bonds, mask = molecule_batch()
    target = bond_targets(bonds)
    single = bond_loss_and_metrics([torch.zeros_like(target)], bonds, mask)
    repeated = bond_loss_and_metrics([torch.zeros_like(target)] * 5, bonds, mask)
    assert float(repeated["loss"]) == pytest.approx(float(single["loss"]), rel=1e-6)
    assert repeated["metrics"]["bond_layers"] == 5
