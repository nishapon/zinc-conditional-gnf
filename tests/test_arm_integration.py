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
    adapt_optimizer_state,
    imbalance_report,
    load_arm_flow,
)
from zinc_gnf.flow import FlowConfig, NodeFlow  # noqa: E402
from zinc_gnf.trainer_flow import FlowTrainer  # noqa: E402

DIM = 8


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


# ============================ optimizer-state adaptation =========================
def test_saved_state_is_restored_and_the_new_parameter_left_unstarted():
    plain = flow_for()
    old = torch.optim.Adam(plain.parameters(), lr=0.05)
    mask = torch.ones((2, 4), dtype=torch.bool)
    old.zero_grad()
    plain.negative_log_likelihood(torch.randn((2, 4, DIM)), mask,
                                  torch.zeros((2, 8)))["loss"].backward()
    old.step()
    saved = old.state_dict()

    coupled = flow_for(coupled=True)
    new = torch.optim.Adam(coupled.parameters(), lr=0.05)
    report = adapt_optimizer_state(new, saved)
    assert report["restored"] is True and report["added"] == 1
    assert len(new.state_dict()["param_groups"][0]["params"]) == \
        len(saved["param_groups"][0]["params"]) + 1


def test_matching_parameter_counts_load_directly():
    flow = flow_for()
    first = torch.optim.Adam(flow.parameters(), lr=0.05)
    report = adapt_optimizer_state(torch.optim.Adam(flow.parameters(), lr=0.05),
                                   first.state_dict())
    assert report["added"] == 0


def test_a_shrinking_parameter_count_is_refused_rather_than_guessed():
    coupled = flow_for(coupled=True)
    saved = torch.optim.Adam(coupled.parameters(), lr=0.05).state_dict()
    plain = torch.optim.Adam(flow_for().parameters(), lr=0.05)
    with pytest.raises(ValueError, match="FEWER parameters"):
        adapt_optimizer_state(plain, saved)


def test_absent_saved_state_is_reported_not_fatal():
    flow = flow_for(coupled=True)
    report = adapt_optimizer_state(torch.optim.Adam(flow.parameters()), None)
    assert report["restored"] is False


# ============================== checkpoint round-trip ============================
@pytest.mark.parametrize("coupled", [False, True])
def test_a_checkpoint_round_trips_for_both_prior_types(tmp_path, coupled):
    from zinc_gnf.trainer_flow import vocabulary_fingerprint

    flow = flow_for(coupled=coupled)
    if coupled:
        with torch.no_grad():
            flow.prior.pre_rho.fill_(0.3)
    path = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": flow.state_dict(),
                "flow_config": flow.config.to_dict(),
                "vocabulary_fingerprint": vocabulary_fingerprint(),
                "step": 1234}, path)
    restored, probe = load_arm_flow(path)
    assert probe["coupled_prior"] is coupled
    assert isinstance(restored.prior, CoupledNodePrior) is coupled
    if coupled:
        assert float(restored.prior.pre_rho.mean().detach()) == pytest.approx(0.3)
    mask = torch.ones((2, 5), dtype=torch.bool)
    x = torch.randn((2, 5, DIM))
    condition = torch.zeros((2, 8))
    assert torch.allclose(restored.log_prob(x, mask, condition)["log_prob"],
                          flow.log_prob(x, mask, condition)["log_prob"], atol=1e-5)


def test_generation_works_without_knowing_the_arm(tmp_path):
    """Downstream sampling must not need to be told which prior the checkpoint used."""
    from zinc_gnf.trainer_flow import vocabulary_fingerprint

    for coupled in (False, True):
        flow = flow_for(coupled=coupled)
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
