"""Experiment C wiring: the three arms, the warm start, and the separated loss terms.

Separate from `test_bond_pairwise.py` (which tests the mechanism) and from
`test_arm_integration.py` / `test_arm_warm_start.py` (experiments A and B), so the
mechanism suite can be imported on a checkout whose `trainer_flow.py` has diverged.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from zinc_gnf.bond_arms import (  # noqa: E402
    BOND_ARCHITECTURE_KEYS,
    BOND_ARMS,
    BondArmConfig,
    BondArmTrainer,
    bond_flow_config,
    estimate_cost,
)
from zinc_gnf.bond_pairwise import set_publishing  # noqa: E402
from zinc_gnf.dataset import DeterministicBatcher  # noqa: E402
from zinc_gnf.experiment_arms import (  # noqa: E402
    architecture_differences,
    load_arm_flow,
    prepare_warm_start,
    verify_additive_architecture,
)
from zinc_gnf.flow import FlowConfig, NodeFlow, node_mask_from_counts  # noqa: E402
from zinc_gnf.trainer_flow import (  # noqa: E402
    FlowTrainConfig,
    FlowTrainer,
    ResumeMismatchError,
    vocabulary_fingerprint,
)

DIM = 8
CONDITION = 8
PARENT_LR = 1e-3
EXPERIMENT_LR = 1e-5


def parent_config():
    return FlowConfig(node_embedding_dim=DIM, num_blocks=2, hidden_dim=16,
                      attention_dim=8, num_heads=2, node_autoregressive=True,
                      condition_features=("n", "qed"),
                      condition_embedding_dim=CONDITION, condition_hidden_dim=8)


def parent_checkpoint(tmp_path):
    """A parent WITHOUT the pathway, trained one step, with ActNorm initialised."""
    config = parent_config()
    flow = NodeFlow(config, init_seed=0)
    generator = torch.Generator().manual_seed(1)
    flow.set_data_statistics(torch.randn(DIM, generator=generator),
                             torch.rand(DIM, generator=generator) + 0.5)
    optimizer = torch.optim.Adam(flow.parameters(), lr=PARENT_LR)
    mask = torch.ones((4, 5), dtype=torch.bool)
    optimizer.zero_grad()
    flow.negative_log_likelihood(torch.randn((4, 5, DIM)), mask,
                                 torch.zeros((4, CONDITION)))["loss"].backward()
    optimizer.step()
    path = tmp_path / "parent.pt"
    torch.save({
        "step": 50000, "epoch": 7, "best_loss": -12.5, "best_step": 49000,
        "bad_evals": 4, "history": [{"step": 1}],
        "batcher": DeterministicBatcher(500, 128, 3).state_dict(),
        "model_state_dict": flow.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "flow_config": config.to_dict(),
        "train_config": FlowTrainConfig(learning_rate=PARENT_LR, seed=3,
                                         batch_size=128).to_dict(),
        "vocabulary_fingerprint": vocabulary_fingerprint(),
        "autoencoder": {"sha256": "a" * 64}, "data_fingerprint": {"counts_digest": "d"},
        "embedding_statistics": {"molecules": 20000},
    }, path)
    return path, config


def experiment_config(tmp_path, steps=3000):
    return FlowTrainConfig(output_dir=str(tmp_path / "arm"), seed=3, batch_size=128,
                           max_steps=steps, learning_rate=EXPERIMENT_LR,
                           warmup_steps=100, early_stopping_patience=None)


# ================================ the three arms =================================
def test_the_three_arms_separate_architecture_from_training_signal():
    control = BondArmConfig(arm="control")
    pairwise = BondArmConfig(arm="pairwise", bond_loss_weight=0.0)
    supervised = BondArmConfig(arm="pairwise_supervised", bond_loss_weight=0.1)
    for arm in (control, pairwise, supervised):
        arm.validate()
    assert control.uses_pairwise is False
    # the two pairwise arms share the architecture exactly ...
    assert pairwise.uses_pairwise is supervised.uses_pairwise is True
    assert pairwise.flow_config_overrides() == supervised.flow_config_overrides()
    # ... and differ only in the training signal
    assert pairwise.bond_supervised is False
    assert supervised.bond_supervised is True


def test_a_pairwise_arm_with_a_bond_loss_is_refused_as_a_mislabelled_arm():
    """Letting `pairwise` carry a bond loss would silently destroy the comparison."""
    with pytest.raises(ValueError, match="must have bond_loss_weight exactly 0"):
        BondArmConfig(arm="pairwise", bond_loss_weight=0.1).validate()
    with pytest.raises(ValueError, match="IS arm 'pairwise'"):
        BondArmConfig(arm="pairwise_supervised", bond_loss_weight=0.0).validate()
    with pytest.raises(ValueError, match="arm must be one of"):
        BondArmConfig(arm="bonds").validate()


def test_the_control_arm_adds_no_parameter_and_the_pairwise_arms_add_the_same():
    control = bond_flow_config(parent_config().to_dict(), BondArmConfig(arm="control"))
    assert control.bond_aware_coupling is False
    counts = {}
    for name in ("pairwise", "pairwise_supervised"):
        arm = BondArmConfig(arm=name, bond_loss_weight=0.0 if name == "pairwise" else 0.1)
        config = bond_flow_config(parent_config().to_dict(), arm)
        assert config.bond_aware_coupling is True
        counts[name] = NodeFlow(config, init_seed=0).num_parameters()
    plain = NodeFlow(control, init_seed=0).num_parameters()
    assert counts["pairwise"] == counts["pairwise_supervised"]
    assert counts["pairwise"] > plain


def test_the_arm_flow_config_is_the_parents_plus_only_the_declared_keys():
    parent = parent_config().to_dict()
    arm = bond_flow_config(parent, BondArmConfig(arm="pairwise", bond_loss_weight=0.0))
    differences = architecture_differences(parent, arm.to_dict())
    assert set(differences) <= set(BOND_ARCHITECTURE_KEYS), differences
    assert differences["bond_aware_coupling"] == (False, True)


# ========================== additive-architecture gate ===========================
def test_an_additive_change_is_accepted_and_recorded():
    parent = parent_config().to_dict()
    arm = bond_flow_config(parent, BondArmConfig(arm="pairwise", bond_loss_weight=0.0))
    report = verify_additive_architecture(
        architecture_differences(parent, arm.to_dict()), BOND_ARCHITECTURE_KEYS)
    assert report["bond_aware_coupling"]["parent"] is False
    assert report["bond_aware_coupling"]["arm"] is True
    assert report["bond_aware_coupling"]["default"] is False


def test_an_undeclared_architecture_difference_is_refused():
    parent = parent_config().to_dict()
    arm = dict(parent, num_blocks=4)
    with pytest.raises(ResumeMismatchError, match="num_blocks"):
        verify_additive_architecture(architecture_differences(parent, arm),
                                     BOND_ARCHITECTURE_KEYS)


def test_removing_a_pathway_the_parent_trained_is_refused_not_treated_as_additive():
    """The parent having the feature ON means warm-starting would DROP trained weights."""
    parent = dict(parent_config().to_dict(), bond_aware_coupling=True)
    arm = dict(parent_config().to_dict(), bond_aware_coupling=False)
    with pytest.raises(ResumeMismatchError, match="not the default|not an addition"):
        verify_additive_architecture(architecture_differences(parent, arm),
                                     BOND_ARCHITECTURE_KEYS)


# =============================== the warm start ==================================
@pytest.mark.parametrize("arm_name", list(BOND_ARMS))
def test_the_warm_start_translates_the_parent_for_every_arm(tmp_path, arm_name):
    parent, _ = parent_checkpoint(tmp_path)
    blob = torch.load(parent, map_location="cpu", weights_only=False)
    arm = BondArmConfig(arm=arm_name,
                        bond_loss_weight=0.1 if arm_name == "pairwise_supervised" else 0.0)
    flow_config = bond_flow_config(blob["flow_config"], arm)
    train_config = experiment_config(tmp_path)
    target = tmp_path / "arm" / "last.pt"
    report = prepare_warm_start(
        parent, target, arm_config=arm, train_config=train_config,
        flow_config=flow_config,
        sampler_state=DeterministicBatcher(500, 128, 3).state_dict(),
        architecture_exceptions=BOND_ARCHITECTURE_KEYS)

    written = torch.load(target, map_location="cpu", weights_only=False)
    assert written["train_config"]["learning_rate"] == EXPERIMENT_LR
    assert written["step"] == 0 and written["best_loss"] is None
    assert written["parent"]["step"] == 50000
    assert written["arm"]["arm"] == arm_name
    added = report["warm_start"]["architecture_added"]
    if arm.uses_pairwise:
        assert set(added) <= set(BOND_ARCHITECTURE_KEYS) and added
    else:
        assert added == {}
    # the new tensors exist and carry no Adam moments
    mapping = report["warm_start"]["optimizer_mapping"]
    new = mapping["new_parameters_without_moments"]
    assert (len(new) > 0) is arm.uses_pairwise
    for key in new:
        assert ".bond." in key or key.endswith("pair_to_hidden.weight"), key


def test_the_warm_started_arm_scores_the_parents_likelihood(tmp_path):
    """Zero bridge plus the parent's weights: the arm starts at the parent's density."""
    parent, _ = parent_checkpoint(tmp_path)
    blob = torch.load(parent, map_location="cpu", weights_only=False)
    arm = BondArmConfig(arm="pairwise", bond_loss_weight=0.0)
    flow_config = bond_flow_config(blob["flow_config"], arm)
    target = tmp_path / "arm" / "last.pt"
    prepare_warm_start(parent, target, arm_config=arm,
                       train_config=experiment_config(tmp_path),
                       flow_config=flow_config,
                       sampler_state=DeterministicBatcher(500, 128, 3).state_dict(),
                       architecture_exceptions=BOND_ARCHITECTURE_KEYS)
    augmented, probe = load_arm_flow(target)
    assert probe["bond_aware"] is True

    original = NodeFlow(FlowConfig.from_dict(blob["flow_config"]), init_seed=0)
    original.load_state_dict(blob["model_state_dict"])
    original.eval()

    mask = node_mask_from_counts((4, 6))
    x = torch.randn((2, 6, DIM), generator=torch.Generator().manual_seed(15))
    condition = torch.zeros((2, CONDITION))
    with torch.no_grad():
        before = original.log_prob(x, mask, condition)["log_prob"]
        after = augmented.log_prob(x, mask, condition)["log_prob"]
    assert torch.equal(before, after)


def test_generation_from_an_arm_checkpoint_needs_no_bond_matrix(tmp_path):
    parent, _ = parent_checkpoint(tmp_path)
    blob = torch.load(parent, map_location="cpu", weights_only=False)
    arm = BondArmConfig(arm="pairwise_supervised", bond_loss_weight=0.1)
    target = tmp_path / "arm" / "last.pt"
    prepare_warm_start(parent, target, arm_config=arm,
                       train_config=experiment_config(tmp_path),
                       flow_config=bond_flow_config(blob["flow_config"], arm),
                       sampler_state=DeterministicBatcher(500, 128, 3).state_dict(),
                       architecture_exceptions=BOND_ARCHITECTURE_KEYS)
    flow, _ = load_arm_flow(target)
    mask = node_mask_from_counts((3, 7))
    with torch.no_grad():
        drawn = flow.sample(mask, torch.zeros((2, CONDITION)),
                            generator=torch.Generator().manual_seed(16))
    assert drawn.shape == (2, 7, DIM) and torch.isfinite(drawn).all()
    assert float(drawn[0, 3:].abs().max()) == 0.0


# ============================ the loss hook, separated ===========================
class Batch:
    """Minimal stand-in for MolecularBatch: only what `_augment_loss` reads."""

    def __init__(self, bonds, node_mask):
        self.bond_types, self.node_mask = bonds, node_mask

    @property
    def batch_size(self):
        return int(self.node_mask.shape[0])


def arm_trainer(arm_name, weight):
    trainer = BondArmTrainer.__new__(BondArmTrainer)
    trainer.arm_config = BondArmConfig(arm=arm_name, bond_loss_weight=weight)
    trainer.arm_config.validate()
    trainer.arm_report = {}
    return trainer


def small_flow(bond_aware=True):
    config = FlowConfig(node_embedding_dim=DIM, num_blocks=2, hidden_dim=16,
                        attention_dim=8, num_heads=2,
                        bond_aware_coupling=bond_aware, bond_pair_dim=4,
                        condition_features=("n", "qed"),
                        condition_embedding_dim=CONDITION, condition_hidden_dim=8)
    flow = NodeFlow(config, init_seed=0)
    generator = torch.Generator().manual_seed(1)
    flow.set_data_statistics(torch.randn(DIM, generator=generator),
                             torch.rand(DIM, generator=generator) + 0.5)
    set_publishing(flow, True)
    flow.train()
    return flow


def ring_batch(counts=(10, 12), width=12):
    mask = node_mask_from_counts(counts, max_nodes=width)
    bonds = torch.zeros((len(counts), width, width), dtype=torch.long)
    for row, count in enumerate(counts):
        for i in range(count - 1):
            bonds[row, i, i + 1] = bonds[row, i + 1, i] = 1
        bonds[row, 0, count - 1] = bonds[row, count - 1, 0] = 1
    return Batch(bonds, mask)


def run_step(flow, batch):
    x = torch.randn((batch.batch_size, batch.node_mask.shape[1], DIM),
                    generator=torch.Generator().manual_seed(17))
    condition = torch.zeros((batch.batch_size, CONDITION))
    return flow.negative_log_likelihood(x, batch.node_mask, condition)


def test_the_supervised_arm_adds_the_weighted_bond_term_to_the_loss():
    flow, batch = small_flow(), ring_batch()
    parts = run_step(flow, batch)
    trainer = arm_trainer("pairwise_supervised", 0.1)
    total, extras = trainer._augment_loss(parts["loss"], flow=flow, batch=batch,
                                          parts=parts)
    assert float(total) > float(parts["loss"])
    assert float(total) == pytest.approx(
        float(parts["loss"]) + extras["bond_loss_weighted"], rel=1e-6)
    assert extras["bond_loss_weighted"] == pytest.approx(
        0.1 * extras["bond_loss"], rel=1e-6)


def test_the_likelihood_only_arm_measures_the_bond_loss_without_optimising_it():
    """Same diagnostic, no gradient: that is what makes the two arms comparable."""
    flow, batch = small_flow(), ring_batch()
    parts = run_step(flow, batch)
    trainer = arm_trainer("pairwise", 0.0)
    total, extras = trainer._augment_loss(parts["loss"], flow=flow, batch=batch,
                                          parts=parts)
    assert total is parts["loss"]
    assert extras["bond_loss"] > 0.0          # measured
    assert extras["bond_loss_weighted"] == 0.0  # not optimised


def test_the_control_arm_adds_nothing_and_reports_nothing():
    flow, batch = small_flow(bond_aware=False), ring_batch()
    parts = run_step(flow, batch)
    trainer = arm_trainer("control", 0.0)
    total, extras = trainer._augment_loss(parts["loss"], flow=flow, batch=batch,
                                          parts=parts)
    assert total is parts["loss"] and extras == {}


def test_both_loss_terms_stay_separately_visible():
    """The likelihood and the auxiliary term must never collapse into one number."""
    flow, batch = small_flow(), ring_batch()
    parts = run_step(flow, batch)
    _, extras = arm_trainer("pairwise_supervised", 0.1)._augment_loss(
        parts["loss"], flow=flow, batch=batch, parts=parts)
    for key in ("bond_loss", "bond_loss_weighted", "bond_f1", "bond_recall",
                "bond_precision", "bond_accuracy", "bond_trivial_accuracy",
                "bond_pos_rate", "bond_pos_weight", "bond_predicted_degree",
                "bond_true_degree", "bond_bridge_norm"):
        assert key in extras, key
        assert isinstance(extras[key], (int, float)), key
    # the trainer logs `nll_per_node` next to these, so the two are never conflated
    import inspect

    source = inspect.getsource(FlowTrainer.train)
    assert '"nll_per_node": float(parts["loss"].detach())' in source, (
        "the trainer must log the pure likelihood separately from the total loss")
    assert "**auxiliary" in source, "auxiliary terms must reach the metrics"
    assert extras["bond_bridge_norm"] == 0.0   # zero-initialised at step 0


def test_the_default_loss_hook_is_a_no_op():
    """Experiment A and B arms, and plain runs, must be unaffected by the new hook."""
    sentinel = object()
    loss, extras = FlowTrainer._augment_loss(None, sentinel, flow=None, batch=None,
                                             parts=None)
    assert loss is sentinel and extras == {}


def test_the_hook_refuses_a_step_whose_forward_did_not_run():
    """No stale supervision: a missing forward raises instead of reusing old logits."""
    from zinc_gnf.bond_pairwise import clear_published_logits

    flow, batch = small_flow(), ring_batch()
    parts = run_step(flow, batch)
    clear_published_logits(flow)
    with pytest.raises(RuntimeError, match="published no bond logits"):
        arm_trainer("pairwise_supervised", 0.1)._augment_loss(
            parts["loss"], flow=flow, batch=batch, parts=parts)


def test_the_hook_clears_the_logits_so_the_next_step_cannot_reuse_them():
    flow, batch = small_flow(), ring_batch()
    parts = run_step(flow, batch)
    trainer = arm_trainer("pairwise_supervised", 0.1)
    trainer._augment_loss(parts["loss"], flow=flow, batch=batch, parts=parts)
    with pytest.raises(RuntimeError, match="published no bond logits"):
        trainer._augment_loss(parts["loss"], flow=flow, batch=batch, parts=parts)


def test_the_supervised_loss_reaches_the_scorer_parameters():
    flow, batch = small_flow(), ring_batch()
    parts = run_step(flow, batch)
    total, _ = arm_trainer("pairwise_supervised", 0.1)._augment_loss(
        parts["loss"], flow=flow, batch=batch, parts=parts)
    total.backward()
    scorer = flow.blocks[0].couplings[0].network.bond.pair_projection.weight
    assert scorer.grad is not None and float(scorer.grad.abs().max()) > 0.0


def test_a_mismatched_arm_resume_is_rejected_on_the_bond_keys():
    trainer = arm_trainer("pairwise_supervised", 0.1)
    trainer.verify_arm({"arm": BondArmConfig(
        arm="pairwise_supervised", bond_loss_weight=0.1).to_dict()})
    with pytest.raises(ResumeMismatchError, match="arm configuration differs"):
        trainer.verify_arm({"arm": BondArmConfig(
            arm="pairwise", bond_loss_weight=0.0).to_dict()})
    with pytest.raises(ResumeMismatchError, match="arm configuration differs"):
        trainer.verify_arm({"arm": BondArmConfig(
            arm="pairwise_supervised", bond_loss_weight=0.5).to_dict()})
    with pytest.raises(ResumeMismatchError, match="no arm block"):
        trainer.verify_arm({})


# ================================= cost estimate =================================
def test_the_cost_estimate_is_measured_from_the_built_flow():
    flow = small_flow()
    cost = estimate_cost(flow, batch_size=128, max_nodes=38)
    assert cost["conditioners"] == 2 * flow.config.num_blocks
    assert cost["pair_dim"] == 4
    assert cost["new_parameters"] > 0
    assert cost["activation_mb_total"] > 0
    # the dominant term must scale linearly in pair_dim and quadratically in N
    wider = estimate_cost(flow, batch_size=256, max_nodes=38)
    assert wider["activation_mb_total"] == pytest.approx(
        2 * cost["activation_mb_total"], rel=1e-9)
    assert estimate_cost(flow, batch_size=128, max_nodes=76)[
        "activation_mb_total"] == pytest.approx(4 * cost["activation_mb_total"],
                                                 rel=1e-9)


def test_the_control_arm_has_no_pathway_cost():
    cost = estimate_cost(small_flow(bond_aware=False), batch_size=128, max_nodes=38)
    assert cost["conditioners"] == 0
    assert cost["new_parameters"] == 0
    assert cost["activation_mb_total"] == 0
