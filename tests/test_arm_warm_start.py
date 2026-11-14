"""End-to-end tests for the two warm-start paths, across all four arms.

PATH 1 translates the common parent into an arm-compatible checkpoint so that
`FlowTrainer._restore`'s strict checks pass honestly rather than being weakened. Those
checks are the thing under test here: the learning-rate comparison, the unconditional
batcher restore, and the inherited best/early-stopping counters.

PATH 2 resumes an interrupted arm run and must reject a changed arm configuration.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from zinc_gnf.balanced_sampler import BalancedSampler  # noqa: E402
from zinc_gnf.coupled_prior import CoupledNodePrior  # noqa: E402
from zinc_gnf.dataset import DeterministicBatcher  # noqa: E402
from zinc_gnf.experiment_arms import (  # noqa: E402
    ARMS,
    ArmConfig,
    draw_latents,
    load_arm_flow,
    map_optimizer_state,
    parameter_order,
    prepare_warm_start,
)
from zinc_gnf.flow import FlowConfig, NodeFlow  # noqa: E402
from zinc_gnf.trainer_flow import FlowTrainConfig, vocabulary_fingerprint  # noqa: E402

DIM = 8
PARENT_LR = 1e-3
EXPERIMENT_LR = 1e-5


def flow_config():
    return FlowConfig(node_embedding_dim=DIM, num_blocks=2, hidden_dim=16,
                      attention_dim=16, num_heads=2, node_autoregressive=True,
                      condition_features=("n", "qed"), condition_embedding_dim=8,
                      condition_hidden_dim=8)


def parent_checkpoint(tmp_path):
    """A parent that looks like the real 50k one: lr 1e-3, step 50000, a best_loss."""
    config = flow_config()
    flow = NodeFlow(config, init_seed=0)
    generator = torch.Generator().manual_seed(1)
    flow.set_data_statistics(torch.randn(DIM, generator=generator),
                             torch.rand(DIM, generator=generator) + 0.5)
    optimizer = torch.optim.Adam(flow.parameters(), lr=PARENT_LR)
    mask = torch.ones((2, 4), dtype=torch.bool)
    optimizer.zero_grad()
    flow.negative_log_likelihood(torch.randn((2, 4, DIM)), mask,
                                 torch.zeros((2, 8)))["loss"].backward()
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


def sampler_state(arm, train_config):
    if arm.uses_balanced_sampler:
        n_node = np.asarray([15] * 400 + [23] * 100)
        qed = np.asarray([0.55] * 400 + [0.82] * 100)
        return BalancedSampler(n_node, qed, batch_size=train_config.batch_size,
                               seed=train_config.seed, alpha=arm.balanced_alpha,
                               min_cell_count=arm.min_cell_count).state_dict()
    return DeterministicBatcher(500, train_config.batch_size,
                                train_config.seed).state_dict()


# ============================ PATH 1, all four arms ==============================
@pytest.mark.parametrize("arm_name", list(ARMS))
def test_path_one_produces_a_checkpoint_that_satisfies_every_restore_check(
        tmp_path, arm_name):
    parent, config = parent_checkpoint(tmp_path)
    arm = ArmConfig(arm=arm_name)
    train_config = experiment_config(tmp_path)
    target = tmp_path / "arm" / "last.pt"
    report = prepare_warm_start(parent, target, arm_config=arm,
                                train_config=train_config, flow_config=config,
                                sampler_state=sampler_state(arm, train_config))
    blob = torch.load(target, map_location="cpu", weights_only=False)

    # the three checks that would otherwise reject a warm start
    assert blob["train_config"]["learning_rate"] == EXPERIMENT_LR
    assert blob["train_config"]["seed"] == train_config.seed
    assert blob["train_config"]["batch_size"] == train_config.batch_size
    # experiment-local bookkeeping, NOT inherited from a 50k run
    assert blob["step"] == 0
    assert blob["best_loss"] is None and blob["best_step"] == -1
    assert blob["bad_evals"] == 0 and blob["history"] == []
    # compatibility fingerprints preserved so those checks stay active
    assert blob["vocabulary_fingerprint"] == vocabulary_fingerprint()
    assert blob["autoencoder"]["sha256"] == "a" * 64
    assert blob["data_fingerprint"]["counts_digest"] == "d"
    assert blob["embedding_statistics"]["molecules"] == 20000
    # parent identity kept separate
    assert blob["parent"]["step"] == 50000
    assert blob["parent"]["learning_rate"] == PARENT_LR
    assert blob["arm"]["arm"] == arm_name
    # RNG deliberately omitted so all arms start from set_seed(seed)
    assert "torch_rng_state" not in blob
    assert report["warm_start"]["learning_rate_override"]["experiment"] == EXPERIMENT_LR


@pytest.mark.parametrize("arm_name", list(ARMS))
def test_path_one_round_trips_and_samples_for_every_arm(tmp_path, arm_name):
    parent, config = parent_checkpoint(tmp_path)
    arm = ArmConfig(arm=arm_name)
    train_config = experiment_config(tmp_path)
    target = tmp_path / "arm" / "last.pt"
    prepare_warm_start(parent, target, arm_config=arm, train_config=train_config,
                       flow_config=config,
                       sampler_state=sampler_state(arm, train_config))
    flow, probe = load_arm_flow(target)
    assert probe["coupled_prior"] is arm.uses_coupled_prior
    assert isinstance(flow.prior, CoupledNodePrior) is arm.uses_coupled_prior

    mask = torch.ones((3, 5), dtype=torch.bool)
    condition = torch.zeros((3, 8))
    eps = torch.randn((3, 5, DIM), generator=torch.Generator().manual_seed(2))
    shared = torch.randn((3, 1, DIM), generator=torch.Generator().manual_seed(3))
    z = draw_latents(flow, mask, condition, eps=eps, shared=shared)
    assert z.shape == (3, 5, DIM) and torch.isfinite(z).all()
    x = flow.inverse(z, mask, condition)
    assert x.shape == (3, 5, DIM) and torch.isfinite(x).all()


@pytest.mark.parametrize("arm_name", ["prior_frozen", "prior_learned"])
def test_an_optimizer_built_after_warm_start_updates_the_prior(tmp_path, arm_name):
    parent, config = parent_checkpoint(tmp_path)
    arm = ArmConfig(arm=arm_name)
    train_config = experiment_config(tmp_path)
    target = tmp_path / "arm" / "last.pt"
    prepare_warm_start(parent, target, arm_config=arm, train_config=train_config,
                       flow_config=config,
                       sampler_state=sampler_state(arm, train_config))
    flow, _ = load_arm_flow(target)
    flow.prior.pre_rho.requires_grad_(arm.rho_learnable)
    optimizer = torch.optim.Adam(flow.parameters(), lr=EXPERIMENT_LR)
    blob = torch.load(target, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(blob["optimizer_state_dict"])
    # the experiment rate must survive loading the parent's moments
    assert all(group["lr"] == EXPERIMENT_LR for group in optimizer.param_groups)

    mask = torch.ones((2, 4), dtype=torch.bool)
    before = flow.prior.pre_rho.detach().clone()
    optimizer.zero_grad()
    flow.negative_log_likelihood(torch.randn((2, 4, DIM)), mask,
                                 torch.zeros((2, 8)))["loss"].backward()
    optimizer.step()
    moved = not torch.allclose(before, flow.prior.pre_rho.detach())
    assert moved is arm.rho_learnable


# ========================= optimizer mapping by NAME =============================
def test_the_mapping_uses_names_reconstructed_from_the_parent_config(tmp_path):
    parent, config = parent_checkpoint(tmp_path)
    blob = torch.load(parent, map_location="cpu", weights_only=False)
    order = parameter_order(blob["flow_config"])
    assert len(order) == len(blob["optimizer_state_dict"]["param_groups"][0]["params"])

    flow = NodeFlow(config, init_seed=0)
    flow.prior = CoupledNodePrior.from_node_prior(flow.prior, init_rho=0.01)
    result = map_optimizer_state(blob, flow, learning_rate=EXPERIMENT_LR)
    mapping = result["mapping"]
    assert mapping["restored"] is True
    assert mapping["arm_parameters"] == mapping["parent_parameters"] + 1
    assert mapping["new_parameters_without_moments"] == ["prior.pre_rho"]
    assert "index -> name -> arm index" in mapping["method"]
    assert result["optimizer_state_dict"]["param_groups"][0]["lr"] == EXPERIMENT_LR


def test_a_shape_mismatch_is_refused_rather_than_mapped(tmp_path):
    parent, _ = parent_checkpoint(tmp_path)
    blob = torch.load(parent, map_location="cpu", weights_only=False)
    wider = FlowConfig(node_embedding_dim=DIM * 2, num_blocks=2, hidden_dim=16,
                       attention_dim=16, num_heads=2, node_autoregressive=True,
                       condition_features=("n", "qed"), condition_embedding_dim=8,
                       condition_hidden_dim=8)
    with pytest.raises(ValueError, match="mapping failed|refusing to guess"):
        map_optimizer_state(blob, NodeFlow(wider, init_seed=0))


# =============================== PATH 2, resume ==================================
def test_the_balanced_sampler_refuses_a_deterministic_batcher_state():
    """The unconditional batcher restore must not absorb a foreign state silently."""
    n_node = np.asarray([15] * 300 + [23] * 90)
    qed = np.asarray([0.5] * 300 + [0.8] * 90)
    sampler = BalancedSampler(n_node, qed, batch_size=8, seed=3, alpha=0.5)
    with pytest.raises(ValueError, match="not written by a BalancedSampler"):
        sampler.load_state_dict(DeterministicBatcher(390, 8, 3).state_dict())


def test_a_balanced_state_round_trips():
    n_node = np.asarray([15] * 300 + [23] * 90)
    qed = np.asarray([0.5] * 300 + [0.8] * 90)
    first = BalancedSampler(n_node, qed, batch_size=8, seed=3, alpha=0.5)
    [first.next_indices() for _ in range(4)]
    second = BalancedSampler(n_node, qed, batch_size=8, seed=3, alpha=0.5)
    second.load_state_dict(first.state_dict())
    assert second.next_indices() == first.next_indices()


def test_resuming_under_a_changed_arm_is_rejected(tmp_path):
    from zinc_gnf.experiment_arms import ArmTrainer
    from zinc_gnf.trainer_flow import ResumeMismatchError

    trainer = ArmTrainer.__new__(ArmTrainer)
    trainer.arm_config = ArmConfig(arm="prior_learned")
    with pytest.raises(ResumeMismatchError, match="arm configuration differs"):
        trainer.verify_arm({"arm": ArmConfig(arm="control").to_dict()})
    with pytest.raises(ResumeMismatchError, match="no arm block"):
        trainer.verify_arm({})
    trainer.verify_arm({"arm": ArmConfig(arm="prior_learned").to_dict()})


# ====================== the evaluator samples the coupled prior ==================
def test_draw_latents_includes_the_shared_component_for_a_coupled_prior():
    """A manual mu + T*sigma*eps would omit it and make rho look like zero."""
    config = flow_config()
    plain = NodeFlow(config, init_seed=0)
    generator = torch.Generator().manual_seed(1)
    plain.set_data_statistics(torch.randn(DIM, generator=generator),
                              torch.rand(DIM, generator=generator) + 0.5)
    coupled = NodeFlow(config, init_seed=0)
    coupled.set_data_statistics(torch.randn(DIM, generator=torch.Generator().manual_seed(1)),
                                torch.rand(DIM, generator=torch.Generator().manual_seed(1)) + 0.5)
    coupled.prior = CoupledNodePrior.from_node_prior(coupled.prior, init_rho=0.5)

    mask = torch.ones((4, 5), dtype=torch.bool)
    condition = torch.zeros((4, 8))
    eps = torch.randn((4, 5, DIM), generator=torch.Generator().manual_seed(2))
    shared = torch.ones((4, 1, DIM)) * 3.0
    a = draw_latents(plain, mask, condition, eps=eps, shared=shared)
    b = draw_latents(coupled, mask, condition, eps=eps, shared=shared)
    assert not torch.allclose(a, b), "the shared component had no effect"


def test_draw_latents_is_unchanged_for_the_plain_prior():
    config = flow_config()
    flow = NodeFlow(config, init_seed=0)
    generator = torch.Generator().manual_seed(1)
    flow.set_data_statistics(torch.randn(DIM, generator=generator),
                             torch.rand(DIM, generator=generator) + 0.5)
    mask = torch.ones((2, 4), dtype=torch.bool)
    condition = torch.zeros((2, 8))
    eps = torch.randn((2, 4, DIM), generator=torch.Generator().manual_seed(5))
    mu, sigma = flow.prior.parameters_for(2, condition, "cpu")
    manual = (mu.unsqueeze(1) + 0.2 * sigma.unsqueeze(1) * eps) * mask.unsqueeze(-1)
    assert torch.allclose(
        draw_latents(flow, mask, condition, eps=eps,
                     shared=torch.randn((2, 1, DIM)), temperature=0.2),
        manual, atol=1e-6)
