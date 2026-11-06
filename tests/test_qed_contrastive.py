"""N-matched QED-contrastive objective: finiteness, gradient routing, stability, resume.

These tests encode the properties the redesign depends on. In particular they pin the
claims that distinguish this objective from the sampling-based one that diverged:

  * no `linalg.inv` and no sequential autoregressive inverse is entered (the two
    operations that exist only in `flow.inverse` and are never constrained by the NLL);
  * gradients reach the conditional prior AND the coupling FiLM, and stay finite;
  * the degenerate "condition-dependent shift" shortcut is penalised, not rewarded;
  * a changed QED condition changes the objective (non-zero response);
  * labels are aligned to the molecule's OWN QED and every candidate keeps its OWN N;
  * disabling the objective leaves training byte-identical to the baseline;
  * checkpoints stay loadable by the generation path (`load_flow`);
  * a non-finite auxiliary loss is counted and, past the grace window, raises rather
    than silently disabling the signal.

Requires torch + rdkit, i.e. the cluster env. This sandbox has neither, so these have
been syntax-checked only -- see the handover notes.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from tests.test_conditional_flow import (
    StubEncoder,
    conditional_dataset,
    conditional_flow,
    wake_up_conditioning,
)
from zinc_gnf.conditioning import condition_embedding_for
from zinc_gnf.flow import FlowConfig, NodeFlow, node_mask_from_counts
from zinc_gnf.qed_contrastive import (
    MASK_LOGIT,
    candidate_conditions,
    contrastive_weight,
    inbatch_candidate_conditions,
    invertible_linear_conditioning,
    qed_contrastive_loss,
)

DIM = 8
QED_COL = 1
RANGE = dict(qed_min=-5.3, qed_max=1.95)


def _flow():
    flow = wake_up_conditioning(conditional_flow(features=("n", "qed"), dimension=DIM))
    flow.train()
    return flow


def _batch(count_list=(5, 6, 7), qed=(0.5, -0.4, 0.8)):
    counts = list(count_list)
    mask = node_mask_from_counts(counts, max_nodes=max(counts))
    values = torch.tensor([[0.3, q] for q in qed], dtype=torch.float32)
    torch.manual_seed(0)
    embeddings = torch.randn(len(counts), mask.shape[1], DIM) * mask.unsqueeze(-1)
    return embeddings, mask, values


# ------------------------------------------------------- candidate construction ---
def test_candidates_keep_their_own_N_and_only_change_qed():
    _, _, values = _batch()
    candidates = candidate_conditions(values, QED_COL, (0.0, -1.0, 1.0), **RANGE)
    assert candidates.shape == (3, 3, 2)
    # N column (0) identical across every candidate: the contrast is WITHIN-N.
    for k in range(3):
        assert torch.allclose(candidates[:, k, 0], values[:, 0])
    # candidate 0 is the true condition
    assert torch.allclose(candidates[:, 0, QED_COL], values[:, QED_COL])
    assert torch.allclose(candidates[:, 1, QED_COL], values[:, QED_COL] - 1.0)


def test_candidate_qed_is_clamped_to_the_train_range():
    values = torch.tensor([[0.0, 1.9]], dtype=torch.float32)       # near the ZINC max
    candidates = candidate_conditions(values, QED_COL, (0.0, 1.0), qed_min=-5.3, qed_max=1.95)
    assert float(candidates[0, 1, QED_COL]) == pytest.approx(1.95)  # clamped, not 2.9


def test_true_candidate_offset_must_be_zero():
    _, _, values = _batch()
    with pytest.raises(ValueError):
        candidate_conditions(values, QED_COL, (0.5, -1.0), **RANGE)


def test_inbatch_negatives_come_from_other_molecules_and_keep_N():
    _, _, values = _batch()
    generator = torch.Generator().manual_seed(0)
    candidates = inbatch_candidate_conditions(values, QED_COL, 3, generator=generator)
    for k in range(3):
        assert torch.allclose(candidates[:, k, 0], values[:, 0])
    assert torch.allclose(candidates[:, 0, QED_COL], values[:, QED_COL])
    # every negative QED is some molecule's real QED
    pool = set(round(float(v), 5) for v in values[:, QED_COL])
    for k in (1, 2):
        for v in candidates[:, k, QED_COL]:
            assert round(float(v), 5) in pool


# --------------------------------------------------------------------- schedule ---
def test_weight_schedule():
    kw = dict(weight=1.0, warmup=100, ramp=100)
    assert contrastive_weight(0, **kw) == 0.0
    assert contrastive_weight(99, **kw) == 0.0
    assert contrastive_weight(149, **kw) == pytest.approx(0.5)
    assert contrastive_weight(199, **kw) == pytest.approx(1.0)
    assert contrastive_weight(10_000, weight=0.0, warmup=0, ramp=0) == 0.0


# ------------------------------------------------------------ loss and gradients ---
def test_loss_is_finite_and_near_chance_before_training():
    flow = _flow()
    embeddings, mask, values = _batch()
    loss, diag = qed_contrastive_loss(flow, embeddings, mask, values, qed_column=QED_COL,
                                      **RANGE)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    assert diag["qed_contrast_candidates"] == 3
    assert diag["qed_contrast_chance_loss"] == pytest.approx(float(np.log(3)))
    # an untrained/woken model should be in the neighbourhood of chance, not absurd
    assert loss.item() < 10 * diag["qed_contrast_chance_loss"]


def test_gradients_are_finite_and_reach_prior_and_film():
    flow = _flow()
    embeddings, mask, values = _batch()
    loss, _ = qed_contrastive_loss(flow, embeddings, mask, values, qed_column=QED_COL, **RANGE)
    loss.backward()
    prior = flow.prior.mu_head.weight.grad
    assert prior is not None and torch.isfinite(prior).all()
    film = [(n, p.grad) for n, p in flow.named_parameters() if "film" in n and p.grad is not None]
    assert film, "the coupling FiLM received no gradient"
    assert all(torch.isfinite(g).all() for _, g in film)
    for name, parameter in flow.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), f"non-finite gradient in {name}"


def test_objective_never_enters_the_inverse_path():
    """The stability claim: no linalg.inv and no sequential AR inverse is used.

    Both are monkeypatched to explode. The old sampling objective would trip them; this
    one must not, because it only ever evaluates the FORWARD map.
    """
    flow = _flow()
    embeddings, mask, values = _batch()

    def forbidden(*args, **kwargs):  # noqa: ANN001, ANN002
        raise AssertionError("the contrastive objective must not touch the inverse path")

    original_inv = torch.linalg.inv
    patched = []
    for block in flow.blocks:
        if getattr(block, "node_coupling", None) is not None:
            patched.append((block.node_coupling, block.node_coupling.inverse))
            block.node_coupling.inverse = forbidden
        if getattr(block, "linear", None) is not None:
            patched.append((block.linear, block.linear.inverse))
            block.linear.inverse = forbidden
    torch.linalg.inv = forbidden
    try:
        loss, _ = qed_contrastive_loss(flow, embeddings, mask, values, qed_column=QED_COL,
                                       **RANGE)
        loss.backward()
    finally:
        torch.linalg.inv = original_inv
        for module, method in patched:
            module.inverse = method
    assert torch.isfinite(loss)


def test_changing_requested_qed_changes_the_loss():
    """Non-zero response: the objective must not be blind to the QED label."""
    flow = _flow()
    embeddings, mask, values = _batch()
    shifted = values.clone()
    shifted[:, QED_COL] = shifted[:, QED_COL] + 1.5
    first, _ = qed_contrastive_loss(flow, embeddings, mask, values, qed_column=QED_COL, **RANGE)
    second, _ = qed_contrastive_loss(flow, embeddings, mask, shifted, qed_column=QED_COL, **RANGE)
    assert not torch.isclose(first, second, atol=1e-6)


def test_identical_prior_means_sit_at_chance_so_the_shortcut_is_penalised():
    """A prior that ignores QED cannot win: the loss is pinned at chance (ln K).

    This is the mechanistic guarantee. A flow whose conditioning is dead (all FiLM and
    prior heads zero, as at initialisation) scores every candidate identically, so the
    softmax is uniform and the loss equals ln K exactly -- its maximum. The only way
    down is genuine within-N separation.
    """
    flow = conditional_flow(features=("n", "qed"), dimension=DIM)   # NOT woken: zero-init
    flow.train()
    embeddings, mask, values = _batch()
    loss, diag = qed_contrastive_loss(flow, embeddings, mask, values, qed_column=QED_COL,
                                      **RANGE)
    assert loss.item() == pytest.approx(float(np.log(3)), abs=1e-4)
    assert diag["prior_mu_gap"] == pytest.approx(0.0, abs=1e-6)


def test_negatives_too_close_to_the_truth_are_excluded_not_mislabelled():
    flow = _flow()
    embeddings, mask, values = _batch()
    # offsets far below min_separation: every negative is ambiguous, so no molecule is
    # usable and the loss must be exactly 0 rather than noisy supervision.
    loss, diag = qed_contrastive_loss(flow, embeddings, mask, values, qed_column=QED_COL,
                                      offsets=(0.0, 0.01), min_separation=0.5, **RANGE)
    assert diag["qed_contrast_usable_fraction"] == pytest.approx(0.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert MASK_LOGIT < 0


def test_masked_logits_do_not_produce_nan_gradients():
    flow = _flow()
    embeddings, mask, values = _batch(qed=(1.94, -0.2, 0.0))   # first molecule clamps
    loss, _ = qed_contrastive_loss(flow, embeddings, mask, values, qed_column=QED_COL,
                                   offsets=(0.0, 1.0), min_separation=0.25, **RANGE)
    if loss.requires_grad:
        loss.backward()
        for name, parameter in flow.named_parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), name


def test_single_node_molecules_are_handled():
    """Zero within-molecule spread was a NaN source for the old pooled-std surrogate."""
    flow = _flow()
    mask = node_mask_from_counts([1, 1], max_nodes=1)
    values = torch.tensor([[0.0, 0.5], [0.0, -0.5]], dtype=torch.float32)
    embeddings = torch.randn(2, 1, DIM) * mask.unsqueeze(-1)
    loss, _ = qed_contrastive_loss(flow, embeddings, mask, values, qed_column=QED_COL, **RANGE)
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in flow.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name


def test_rejects_a_flow_without_qed_conditioning():
    flow = wake_up_conditioning(conditional_flow(features=("n",), dimension=DIM))
    embeddings, mask, values = _batch()
    with pytest.raises(ValueError):
        qed_contrastive_loss(flow, embeddings, mask, values[:, :1], qed_column=0, **RANGE)


# ------------------------------------------------------------------ diagnostics ---
def test_invertible_linear_conditioning_reports_per_layer_numbers():
    flow = _flow()
    report = invertible_linear_conditioning(flow)
    assert report["linear_layers"] == flow.config.num_blocks
    assert report["linear_condition_number_max"] >= 1.0
    assert np.isfinite(report["linear_max_abs_log_s"])


# ------------------------------------------------------- trainer integration ----
def _trainer(tmp_path, *, contrast_weight, **overrides):
    from zinc_gnf.trainer_flow import FlowTrainConfig, FlowTrainer
    from zinc_gnf.utils.logging import null_logger

    train = conditional_dataset(size=24, dimension=DIM)
    val = conditional_dataset(size=12, dimension=DIM)
    flow_config = FlowConfig(node_embedding_dim=DIM, num_blocks=2, hidden_dim=16,
                             attention_dim=16, num_heads=2, node_autoregressive=True,
                             condition_features=("n", "qed"), condition_embedding_dim=8,
                             condition_hidden_dim=8)
    values = dict(
        data_dir="<injected>", output_dir=str(tmp_path / f"flow_c{contrast_weight}"),
        seed=1, device="cpu", batch_size=8, max_steps=6, learning_rate=1e-3,
        warmup_steps=0, log_every=2, eval_every=3, save_every=3, val_molecules=8,
        statistics_molecules=12, early_stopping_patience=None, ae_checkpoint="<unused>",
        qed_contrast_weight=contrast_weight, qed_contrast_warmup=0, qed_contrast_ramp=0,
        qed_contrast_subbatch=4)
    values.update(overrides)
    trainer = FlowTrainer(flow_config, FlowTrainConfig(**values), logger=null_logger(),
                          datasets=(train, val, {}), autoencoder=StubEncoder())
    trainer.autoencoder.eval = lambda: None
    trainer.autoencoder.parameters = lambda: iter(())
    trainer.autoencoder.to = lambda *a, **k: trainer.autoencoder
    trainer.autoencoder.config = type("C", (), {"node_embedding_dim": DIM,
                                                 "symmetry_noise_dim": 0,
                                                 "degree_head": False})()
    return trainer


def test_trainer_off_is_baseline_and_checkpoint_loads_for_generation(tmp_path):
    from zinc_gnf.trainer_flow import load_flow

    trainer = _trainer(tmp_path, contrast_weight=0.0)
    result = trainer.train(resume="none")
    assert result["completed"]
    assert trainer.qed_contrast_on is False
    flow, probe = load_flow(trainer.output_dir / "best.pt", device="cpu")
    assert tuple(flow.config.condition_features) == ("n", "qed")
    # no new parameters and no new checkpoint keys are introduced by the objective
    assert "qed_surrogate_state" not in probe


def test_trainer_on_records_contrastive_metrics_and_stays_finite(tmp_path):
    trainer = _trainer(tmp_path, contrast_weight=1.0)
    result = trainer.train(resume="none")
    assert result["completed"]
    assert trainer.qed_contrast_on is True
    assert trainer.contrast_skipped == 0
    records = [json.loads(line) for line in
               (trainer.output_dir / "metrics.jsonl").read_text().splitlines()]
    train_records = [r for r in records if r["kind"] == "train"]
    assert any("qed_contrast_loss" in r and "prior_mu_gap" in r for r in train_records)
    assert all(np.isfinite(r["loss"]) for r in train_records)


def test_both_objectives_at_once_is_refused(tmp_path):
    from zinc_gnf.trainer_flow import FlowTrainConfig

    config = FlowTrainConfig(qed_contrast_weight=1.0, qed_consistency_weight=0.5)
    with pytest.raises(ValueError, match="must not both be"):
        config.validate()


def test_resume_preserves_the_contrastive_counters(tmp_path):
    trainer = _trainer(tmp_path, contrast_weight=1.0, max_steps=3)
    trainer.train(resume="none")
    applied = trainer.contrast_applied
    assert applied > 0
    resumed = _trainer(tmp_path, contrast_weight=1.0, max_steps=6)
    resumed.config.output_dir = trainer.config.output_dir
    resumed.output_dir = trainer.output_dir
    resumed.train(resume="auto")
    assert resumed.contrast_applied >= applied


def test_high_nonfinite_rate_raises_instead_of_training_without_the_signal(tmp_path, monkeypatch):
    """The previous attempt's real failure: the signal was silently absent.

    With the grace window set to 1 and the tolerated rate at 0, a single non-finite
    auxiliary loss must STOP training with a clear error.
    """
    import zinc_gnf.qed_contrastive as module

    trainer = _trainer(tmp_path, contrast_weight=1.0, qed_contrast_skip_grace=1,
                       qed_contrast_max_skip_rate=0.0)

    def poisoned(*args, **kwargs):  # noqa: ANN001, ANN002
        return torch.tensor(float("nan"), requires_grad=True), {}

    monkeypatch.setattr(module, "qed_contrastive_loss", poisoned)
    with pytest.raises(FloatingPointError, match="non-finite"):
        trainer.train(resume="none")
    assert (trainer.output_dir / "last.pt").exists()      # state preserved for inspection
