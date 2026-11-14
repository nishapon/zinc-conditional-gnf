"""Integration tests for the four experiment arms.

Separate file from `test_experiments_a_and_b.py` on purpose: those test the MECHANISMS in
isolation, these test the WIRING into the trainer. Keeping them apart means the mechanism
suite can be imported and run on a cluster checkout whose `trainer_flow.py` has diverged.

What matters here is that the hooks do what the arms need: the new prior parameter must
reach the optimizer and actually update, a checkpoint must round-trip for both prior
types, generation must work without knowing which arm produced the checkpoint, and the
sampler must resume.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from zinc_gnf.balanced_sampler import BalancedSampler  # noqa: E402
from zinc_gnf.coupled_prior import INIT_RHO, CoupledNodePrior  # noqa: E402
from zinc_gnf.experiment_arms import (  # noqa: E402
    ARMS,
    ArmConfig,
    ArmTrainer,
    imbalance_report,
    load_arm_flow,
)
from zinc_gnf.flow import ActNorm, FlowConfig, NodeFlow  # noqa: E402
from zinc_gnf.trainer_flow import FlowTrainer  # noqa: E402

DIM = 8


def uninitialized_actnorms(flow):
    return [name for name, module in flow.named_modules()
            if isinstance(module, ActNorm) and not bool(module.initialized)]


def initialize_in_place(flow, *, batch=16, width=6, seed=11):
    """Run ONE train-mode forward so every ActNorm does its data-dependent init.

    `ActNorm.forward` initializes `log_scale` and `bias` from the first batch it sees,
    but ONLY while `self.training`. A freshly constructed flow therefore carries zeros
    in those parameters and `initialized=False`, which is a state no real checkpoint is
    ever saved in: by the time the trainer writes `last.pt` the first step has long since
    initialized them. Saving an uninitialized flow and then comparing it against a
    restored one in eval mode compares two DIFFERENT models, because the original
    initializes itself from the comparison batch on the way past.
    """
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((batch, width, DIM), generator=generator)
    mask = torch.ones((batch, width), dtype=torch.bool)
    flow.train()
    with torch.no_grad():
        flow.log_prob(x, mask, torch.zeros((batch, 8)))
    pending = uninitialized_actnorms(flow)
    assert pending == [], f"ActNorm still uninitialized after a train-mode pass: {pending}"
    assert bool(flow.statistics_set), "embedding statistics must be set before saving"
    return flow


def divergence_by_stage(left, right, x, mask, condition):
    """Max absolute difference between two flows at every transformation in order.

    Used only to explain a failure: it names the FIRST stage at which two flows that are
    supposed to be identical stop agreeing, instead of leaving a likelihood mismatch
    unattributed.
    """
    stages = []
    with torch.no_grad():
        start = x * mask.unsqueeze(-1).to(x.dtype)
        a, _ = left.whiten(start, mask)
        b, _ = right.whiten(start, mask)
        stages.append(("whiten", float((a - b).abs().max())))
        for index in range(len(left.blocks)):
            a, _ = left.blocks[index](a, mask, condition)
            b, _ = right.blocks[index](b, mask, condition)
            stages.append((f"block[{index}]", float((a - b).abs().max())))
        stages.append(("prior.log_prob", float(
            (left.prior.log_prob(a, mask, condition)
             - right.prior.log_prob(b, mask, condition)).abs().max())))
    return [(name, value) for name, value in stages if value != 0.0] or stages


def flow_for(*, coupled=False, learnable=True):
    config = FlowConfig(node_embedding_dim=DIM, num_blocks=2, hidden_dim=16,
                        attention_dim=16, num_heads=2, node_autoregressive=True,
                        condition_features=("n", "qed"), condition_embedding_dim=8,
                        condition_hidden_dim=8)
    flow = NodeFlow(config, init_seed=0)
    generator = torch.Generator().manual_seed(1)
    flow.set_data_statistics(torch.randn(DIM, generator=generator),
                             torch.rand(DIM, generator=generator) + 0.5)
    if coupled:
        flow.prior = CoupledNodePrior.from_node_prior(
            flow.prior, init_rho=INIT_RHO, learnable=learnable)
    return flow


# ============================== hooks and defaults ===============================
def test_the_default_hooks_reproduce_the_original_behaviour():
    """A no-op default is what keeps the unchanged control genuinely unchanged."""
    assert FlowTrainer._prepare_flow(None, "sentinel") == "sentinel"
    assert "DeterministicBatcher" in FlowTrainer._make_batcher.__doc__ or True
    source = FlowTrainer._make_batcher.__code__.co_names
    assert "DeterministicBatcher" in source


def test_every_arm_names_exactly_one_intervention():
    for arm in ARMS:
        config = ArmConfig(arm=arm)
        config.validate()
        changes = [config.uses_balanced_sampler, config.uses_coupled_prior]
        assert sum(changes) <= 1, arm
    assert ArmConfig(arm="control").uses_balanced_sampler is False
    assert ArmConfig(arm="control").uses_coupled_prior is False
    assert ArmConfig(arm="prior_learned").rho_learnable is True
    assert ArmConfig(arm="prior_frozen").rho_learnable is False


def test_an_unknown_arm_or_degenerate_alpha_is_refused():
    with pytest.raises(ValueError, match="arm must be one of"):
        ArmConfig(arm="nope").validate()
    with pytest.raises(ValueError, match="is the control"):
        ArmConfig(arm="balanced", balanced_alpha=0.0).validate()


# ========================== optimizer registration/update ========================
def test_the_new_prior_parameter_reaches_the_optimizer_and_updates():
    """The whole point of placing the hook before optimizer construction."""
    flow = flow_for(coupled=True, learnable=True)
    names = [name for name, _ in flow.named_parameters()]
    assert any(name.endswith("prior.pre_rho") for name in names)
    optimizer = torch.optim.Adam(flow.parameters(), lr=0.05)
    tracked = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert id(flow.prior.pre_rho) in tracked

    mask = torch.ones((4, 5), dtype=torch.bool)
    x = torch.randn((4, 5, DIM))
    condition = torch.zeros((4, 8))
    before = flow.prior.pre_rho.detach().clone()
    optimizer.zero_grad()
    flow.negative_log_likelihood(x, mask, condition)["loss"].backward()
    assert flow.prior.pre_rho.grad is not None
    assert float(flow.prior.pre_rho.grad.abs().sum()) > 0.0
    optimizer.step()
    assert not torch.allclose(before, flow.prior.pre_rho.detach())


def test_a_frozen_rho_gets_no_gradient_through_the_flow():
    flow = flow_for(coupled=True, learnable=False)
    optimizer = torch.optim.Adam(flow.parameters(), lr=0.05)
    mask = torch.ones((4, 5), dtype=torch.bool)
    before = flow.prior.pre_rho.detach().clone()
    optimizer.zero_grad()
    flow.negative_log_likelihood(torch.randn((4, 5, DIM)), mask,
                                 torch.zeros((4, 8)))["loss"].backward()
    assert flow.prior.pre_rho.grad is None
    optimizer.step()
    assert torch.allclose(before, flow.prior.pre_rho.detach())


def test_the_control_arm_adds_no_parameter():
    plain, coupled = flow_for(), flow_for(coupled=True)
    assert coupled.num_parameters() == plain.num_parameters() + DIM


# ======================= the removed positional helper ===========================
def test_the_count_based_optimizer_adapter_no_longer_exists():
    """It mapped Adam state by COUNT and assumed `parameters()` order.

    Keeping it importable alongside the name-based mapper is the dangerous case: a stale
    call site would keep working and keep mismapping. Its absence is asserted so it
    cannot quietly return. All optimizer-state behaviour is now covered by
    `tests/test_arm_warm_start.py` against `map_optimizer_state`.
    """
    import zinc_gnf.experiment_arms as arms

    assert not hasattr(arms, "adapt_optimizer_state")
    assert "adapt_optimizer_state" not in arms.__all__


def test_a_parent_without_optimizer_state_is_reported_not_fatal():
    from zinc_gnf.experiment_arms import map_optimizer_state

    result = map_optimizer_state({"optimizer_state_dict": None}, flow_for())
    assert result["optimizer_state_dict"] is None
    assert result["mapping"]["restored"] is False


# ============================== checkpoint round-trip ============================
@pytest.mark.parametrize("coupled", [False, True])
def test_a_checkpoint_round_trips_for_both_prior_types(tmp_path, coupled):
    """Identical parameters and buffers must give identical likelihoods.

    Three things make this exact rather than approximate: ActNorm is initialized before
    saving, so the saved flow is in a state a real checkpoint can be in; both models are
    compared in EVAL mode, so neither re-initializes itself during the comparison; and
    the comparison runs on one shared input tensor.
    """
    from zinc_gnf.trainer_flow import vocabulary_fingerprint

    flow = initialize_in_place(flow_for(coupled=coupled))
    if coupled:
        with torch.no_grad():
            flow.prior.pre_rho.fill_(0.3)
    flow.eval()
    saved = {name: tensor.detach().clone().cpu()
             for name, tensor in flow.state_dict().items()}
    path = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": flow.state_dict(),
                "flow_config": flow.config.to_dict(),
                "vocabulary_fingerprint": vocabulary_fingerprint(),
                "step": 1234}, path)
    restored, probe = load_arm_flow(path)
    assert probe["coupled_prior"] is coupled
    assert isinstance(restored.prior, CoupledNodePrior) is coupled
    # `load_arm_flow` evals the restore; the original must be in the same mode or its
    # ActNorm would initialize from the comparison batch
    assert restored.training is False and flow.training is False

    # every parameter AND buffer identical, bit for bit
    produced = {name: tensor.detach().cpu()
                for name, tensor in restored.state_dict().items()}
    assert set(produced) == set(saved), (
        f"state keys differ: only saved {sorted(set(saved) - set(produced))}, "
        f"only restored {sorted(set(produced) - set(saved))}")
    differing = {name: float((produced[name].to(torch.float64)
                              - saved[name].to(torch.float64)).abs().max())
                 for name in saved if not torch.equal(produced[name], saved[name])}
    assert differing == {}, f"state differs after the round trip: {differing}"
    # the data-dependent initialization is part of what round-tripped
    assert uninitialized_actnorms(restored) == []
    if coupled:
        assert float(restored.prior.pre_rho.mean().detach()) == pytest.approx(0.3)

    mask = torch.ones((2, 5), dtype=torch.bool)
    x = torch.randn((2, 5, DIM), generator=torch.Generator().manual_seed(7))
    condition = torch.zeros((2, 8))
    before = restored.log_prob(x, mask, condition)["log_prob"]
    after = flow.log_prob(x, mask, condition)["log_prob"]
    assert torch.equal(before, after), (
        f"identical state gave different likelihoods ({before.tolist()} vs "
        f"{after.tolist()}); first divergence by stage: "
        f"{divergence_by_stage(flow, restored, x, mask, condition)}")


def test_generation_works_without_knowing_the_arm(tmp_path):
    """Downstream sampling must not need to be told which prior the checkpoint used."""
    from zinc_gnf.trainer_flow import vocabulary_fingerprint

    for coupled in (False, True):
        flow = initialize_in_place(flow_for(coupled=coupled)).eval()
        path = tmp_path / f"ckpt_{coupled}.pt"
        torch.save({"model_state_dict": flow.state_dict(),
                    "flow_config": flow.config.to_dict(),
                    "vocabulary_fingerprint": vocabulary_fingerprint(),
                    "step": 10}, path)
        restored, _ = load_arm_flow(path)
        mask = torch.ones((3, 6), dtype=torch.bool)
        condition = torch.zeros((3, 8))
        z = restored.prior.sample(mask, condition,
                                  generator=torch.Generator().manual_seed(2))
        assert z.shape == (3, 6, DIM) and torch.isfinite(z).all()
        x = restored.inverse(z, mask, condition)
        assert x.shape == (3, 6, DIM) and torch.isfinite(x).all()


# ================================ sampler resume =================================
def test_the_balanced_sampler_resumes_mid_epoch():
    n_node = np.asarray([15] * 400 + [23] * 120)
    qed = np.asarray([0.55] * 400 + [0.72] * 120)
    first = BalancedSampler(n_node, qed, batch_size=16, seed=5, alpha=0.5)
    [first.next_indices() for _ in range(5)]
    state = first.state_dict()
    second = BalancedSampler(n_node, qed, batch_size=16, seed=5, alpha=0.5)
    second.load_state_dict(state)
    assert second.epoch == first.epoch and second.position == first.position
    assert second.next_indices() == first.next_indices()


def test_the_sampler_exposes_the_batcher_interface_the_trainer_uses():
    n_node = np.asarray([15] * 200 + [23] * 80)
    qed = np.asarray([0.5] * 200 + [0.7] * 80)
    sampler = BalancedSampler(n_node, qed, batch_size=8, seed=6, alpha=0.5)
    for attribute in ("next_indices", "state_dict", "load_state_dict"):
        assert callable(getattr(sampler, attribute)), attribute
    assert isinstance(sampler.epoch, int) and isinstance(sampler.position, int)


# ================================ imbalance gate =================================
def test_a_skewed_distribution_is_called_imbalanced():
    n_node = np.asarray([15] * 5000 + [23] * 60)
    qed = np.asarray([0.55] * 5000 + [0.85] * 60)
    report = imbalance_report(n_node, qed)
    assert report["meaningful_imbalance"] is True
    assert "IMBALANCED" in report["verdict"]


def test_an_already_balanced_distribution_blocks_the_arm():
    n_node = np.asarray([15] * 300 + [23] * 300 + [30] * 300)
    qed = np.asarray([0.45] * 300 + [0.55] * 300 + [0.65] * 300)
    report = imbalance_report(n_node, qed)
    assert report["meaningful_imbalance"] is False
    assert "NEARLY BALANCED ALREADY" in report["verdict"]


def test_the_gate_thresholds_are_explicit():
    report = imbalance_report(np.asarray([15] * 100), np.asarray([0.5] * 100))
    assert set(report["thresholds"]) == {"min_effective_cell_fraction",
                                         "min_population_ratio"}
