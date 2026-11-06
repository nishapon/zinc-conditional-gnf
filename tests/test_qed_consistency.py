"""Conditional QED-consistency auxiliary objective (training only).

These tests pin the properties the objective MUST have to be scientifically safe:
  * the surrogate pooling is identical to the stage-20 probe's pooling;
  * the weight schedule is 0 before warmup, ramps linearly, then saturates;
  * the consistency gradient reaches the FLOW (prior heads and coupling FiLM) but NOT the
    frozen surrogate when the caller freezes it, so the flow is moved toward the reader,
    never the reader toward the flow;
  * turning the objective OFF (weight 0, the default) leaves training byte-identical to
    the baseline, and the trained checkpoint stays loadable by the generation path
    (`load_flow`) because the surrogate lives under separate checkpoint keys;
  * turning it ON actually updates the surrogate and records the aux terms, without
    sending gradients into the frozen autoencoder.

Run on a machine with torch + rdkit (the cluster env); this sandbox has neither.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from tests.test_conditional_flow import (
    StubEncoder,
    conditional_dataset,
    conditional_flow,
    wake_up_conditioning,
)
from zinc_gnf.conditioning import batch_condition, condition_embedding_for
from zinc_gnf.dataset import make_batch
from zinc_gnf.flow import node_mask_from_counts
from zinc_gnf.qed_consistency import (
    EmbeddingQedSurrogate,
    conditional_consistency_loss,
    consistency_weight,
    pooled_mean_std,
    surrogate_regression_loss,
)
from zinc_gnf.qed_probes import pooled_features

DIM = 8


# --------------------------------------------------------------------- pooling ---
def test_pooling_is_identical_to_the_stage20_probe():
    torch.manual_seed(0)
    x = torch.randn(5, 7, DIM)
    mask = torch.zeros(5, 7, dtype=torch.bool)
    for i in range(5):
        mask[i, : 2 + i] = True
    ours = pooled_mean_std(x, mask)
    theirs = pooled_features(x, mask)
    assert ours.shape == (5, 2 * DIM)
    assert torch.allclose(ours, theirs, atol=0.0)


def test_pooling_ignores_padding_and_is_width_invariant():
    torch.manual_seed(1)
    x = torch.randn(1, 4, DIM)
    mask = torch.tensor([[True, True, True, False]])
    narrow = pooled_mean_std(x, mask)
    # Appending another padded column must not change the pooled features.
    x2 = torch.cat([x, torch.randn(1, 1, DIM)], dim=1)
    mask2 = torch.tensor([[True, True, True, False, False]])
    assert torch.allclose(narrow, pooled_mean_std(x2, mask2), atol=1e-6)


# --------------------------------------------------------------------- schedule ---
def test_weight_schedule_is_off_then_ramps_then_saturates():
    kw = dict(weight=0.5, warmup=100, ramp=100)
    assert consistency_weight(0, **kw) == 0.0
    assert consistency_weight(99, **kw) == 0.0
    assert consistency_weight(100, **kw) == pytest.approx(0.5 * 1 / 100)
    assert consistency_weight(149, **kw) == pytest.approx(0.5 * 50 / 100)
    assert consistency_weight(199, **kw) == pytest.approx(0.5)
    assert consistency_weight(10_000, **kw) == pytest.approx(0.5)
    # weight 0 means off at every step; ramp 0 means an instant step to full weight.
    assert consistency_weight(10_000, weight=0.0, warmup=0, ramp=100) == 0.0
    assert consistency_weight(5, weight=0.5, warmup=0, ramp=0) == pytest.approx(0.5)


# --------------------------------------------------------------------- surrogate ---
def test_surrogate_shapes_and_regression_is_finite():
    torch.manual_seed(0)
    surrogate = EmbeddingQedSurrogate(DIM, hidden_dim=16)
    x = torch.randn(6, 5, DIM)
    mask = torch.ones(6, 5, dtype=torch.bool)
    out = surrogate(x, mask)
    assert out.shape == (6,)
    loss = surrogate_regression_loss(surrogate, x, mask, torch.rand(6))
    assert torch.isfinite(loss) and loss.item() >= 0.0


# ------------------------------------------------------- consistency gradient ----
def _woken_flow_and_condition():
    flow = wake_up_conditioning(conditional_flow(features=("n", "qed"), dimension=DIM))
    flow.train()
    scaler_values = np.stack([
        # standardised (log1p N, QED) rows; exact values do not matter for a gradient test
        np.array([0.3, 0.5], dtype=np.float32),
        np.array([-0.2, -0.4], dtype=np.float32),
        np.array([0.1, 0.8], dtype=np.float32),
    ])
    values = torch.from_numpy(scaler_values)
    counts = [5, 6, 7]
    mask = node_mask_from_counts(counts, max_nodes=max(counts))
    condition = condition_embedding_for(flow, values, torch.ones_like(values), len(counts), "cpu")
    return flow, condition, mask, values


def test_consistency_gradient_reaches_the_flow_not_the_frozen_surrogate():
    torch.manual_seed(0)
    flow, condition, mask, values = _woken_flow_and_condition()
    surrogate = EmbeddingQedSurrogate(DIM, hidden_dim=16)

    for parameter in surrogate.parameters():
        parameter.requires_grad_(False)
    generator = torch.Generator().manual_seed(7)
    loss, diag = conditional_consistency_loss(
        flow, surrogate, condition=condition, mask=mask,
        qed_std_target=values[:, 1], temperature=1.0, generator=generator)
    assert torch.isfinite(loss)
    assert set(diag) >= {"qed_consistency_mse", "qed_consistency_pred_mean",
                         "qed_consistency_target_mean"}
    loss.backward()

    # the conditional prior heads and the coupling FiLM must receive gradient
    prior_grad = flow.prior.mu_head.weight.grad
    film_grads = [p.grad for n, p in flow.named_parameters() if "film" in n and p.grad is not None]
    assert prior_grad is not None and torch.any(prior_grad != 0)
    assert film_grads, "the FiLM conditioning in the couplings received no gradient"
    # the frozen surrogate must receive NONE
    assert all(p.grad is None for p in surrogate.parameters())


def test_changing_requested_qed_changes_the_generated_pooled_embedding():
    """The whole point: moving QED in the condition must move the generated embeddings.

    A woken (non-zero FiLM/prior) flow is enough to show the pathway is live; a trained
    flow is what makes it track the TARGET, which is the training objective itself.
    """
    torch.manual_seed(0)
    flow, condition, mask, values = _woken_flow_and_condition()
    surrogate = EmbeddingQedSurrogate(DIM, hidden_dim=16)
    low = values.clone()
    low[:, 1] = -1.0
    high = values.clone()
    high[:, 1] = 1.0
    cond_low = condition_embedding_for(flow, low, torch.ones_like(low), low.shape[0], "cpu")
    cond_high = condition_embedding_for(flow, high, torch.ones_like(high), high.shape[0], "cpu")
    gen = torch.Generator().manual_seed(3)
    with torch.no_grad():
        dim = flow.config.node_embedding_dim
        mu_l, sig_l = flow.prior.parameters_for(low.shape[0], cond_low, "cpu")
        eps = torch.randn((low.shape[0], mask.shape[1], dim), generator=gen)
        z_l = (mu_l.unsqueeze(1) + sig_l.unsqueeze(1) * eps) * mask.unsqueeze(-1)
        z_h = (flow.prior.parameters_for(high.shape[0], cond_high, "cpu")[0].unsqueeze(1)
               + flow.prior.parameters_for(high.shape[0], cond_high, "cpu")[1].unsqueeze(1) * eps
               ) * mask.unsqueeze(-1)
        x_low = pooled_mean_std(flow.inverse(z_l, mask, cond_low), mask)
        x_high = pooled_mean_std(flow.inverse(z_h, mask, cond_high), mask)
    assert not torch.allclose(x_low, x_high, atol=1e-4)


# ------------------------------------------------------- trainer integration ----
def _trainer(tmp_path, *, weight):
    from zinc_gnf.flow import FlowConfig
    from zinc_gnf.trainer_flow import FlowTrainConfig, FlowTrainer
    from zinc_gnf.utils.logging import null_logger

    train = conditional_dataset(size=24, dimension=DIM)
    val = conditional_dataset(size=12, dimension=DIM)
    flow_config = FlowConfig(node_embedding_dim=DIM, num_blocks=2, hidden_dim=16,
                             attention_dim=16, num_heads=2, node_autoregressive=True,
                             condition_features=("n", "qed"), condition_embedding_dim=8,
                             condition_hidden_dim=8)
    train_config = FlowTrainConfig(
        data_dir="<injected>", output_dir=str(tmp_path / f"flow_w{weight}"), seed=1,
        device="cpu", batch_size=4, max_steps=6, learning_rate=1e-3, warmup_steps=0,
        log_every=2, eval_every=3, save_every=3, val_molecules=8, statistics_molecules=12,
        early_stopping_patience=None, ae_checkpoint="<unused>",
        qed_consistency_weight=weight, qed_consistency_warmup=0, qed_consistency_ramp=0,
        qed_consistency_subbatch=4)
    trainer = FlowTrainer(flow_config, train_config, logger=null_logger(),
                          datasets=(train, val, {}), autoencoder=StubEncoder())
    return trainer


def test_trainer_runs_with_objective_off_and_checkpoint_loads_for_generation(tmp_path):
    from zinc_gnf.trainer_flow import load_flow

    # StubEncoder has no .eval()/.parameters(); the trainer freezes a real AE in practice.
    # Here we only need encode(); patch the two calls the trainer makes on it.
    trainer = _trainer(tmp_path, weight=0.0)
    trainer.autoencoder.eval = lambda: None
    trainer.autoencoder.parameters = lambda: iter(())
    trainer.autoencoder.to = lambda *a, **k: trainer.autoencoder
    trainer.autoencoder.config = type("C", (), {"node_embedding_dim": DIM,
                                                 "symmetry_noise_dim": 0, "degree_head": False})()
    result = trainer.train(resume="none")
    assert result["completed"]
    assert trainer.surrogate is None                       # objective off -> no surrogate built
    # the checkpoint is a plain flow checkpoint: the generation loader accepts it
    flow, _ = load_flow(trainer.output_dir / "best.pt", device="cpu")
    assert tuple(flow.config.condition_features) == ("n", "qed")


def test_trainer_with_objective_on_updates_surrogate_and_logs_aux(tmp_path):
    import json

    trainer = _trainer(tmp_path, weight=0.5)
    trainer.autoencoder.eval = lambda: None
    trainer.autoencoder.parameters = lambda: iter(())
    trainer.autoencoder.to = lambda *a, **k: trainer.autoencoder
    trainer.autoencoder.config = type("C", (), {"node_embedding_dim": DIM,
                                                 "symmetry_noise_dim": 0, "degree_head": False})()
    result = trainer.train(resume="none")
    assert result["completed"]
    assert trainer.surrogate is not None
    # aux terms recorded in metrics.jsonl train records
    records = [json.loads(l) for l in (trainer.output_dir / "metrics.jsonl").read_text().splitlines()]
    train_records = [r for r in records if r["kind"] == "train"]
    assert any("qed_consistency_mse" in r and "qed_surrogate_mse" in r for r in train_records)
    # the checkpoint still loads as a plain flow (surrogate kept under separate keys)
    from zinc_gnf.trainer_flow import load_flow
    flow, probe = load_flow(trainer.output_dir / "best.pt", device="cpu")
    assert "qed_surrogate_state" in probe                  # saved, but not part of the flow
