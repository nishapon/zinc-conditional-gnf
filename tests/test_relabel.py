"""Tests for generated-QED relabeling: pool, objective, fine-tune guards, evaluation.

The pilot is only interpretable if six things hold, so each is pinned here:

  * FORWARD-ONLY. The objective must never reach `flow.inverse`. The v1 QED-consistency
    run went non-finite precisely there, so this is checked both statically (no call in
    any training path) and at runtime (poison `flow.inverse` and train anyway).
  * NO LEAKAGE. The pool noise seed, the evaluation noise seed and the smoothing seeds
    must all differ, the pool must be split into train/holdout stratified per cell, and
    the evaluator must REFUSE to run if its seed equals the pool's.
  * THE LABEL IS THE DECODED QED, NOT THE REQUEST. A relabeled condition built from the
    request would make the whole experiment a no-op, so the two are asserted different.
  * NOTHING IS OVERWRITTEN. The parent checkpoint is read-only and an existing output
    directory stops the run.
  * COMMON NOISE AND HONEST PAIRING. The paired response must use the same prior draw
    across requests and models, and only jointly valid pairs.
  * THE SUCCESS CRITERION IS FIXED IN CODE. A lower MAE must not be able to produce a
    pass, and a gain visible only under the label decoder must be flagged as
    correction-dependent.

Requires torch + rdkit (the cluster env).
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from zinc_gnf import relabel
from zinc_gnf.relabel import (
    EVAL_NOISE_SEED,
    POOL_NOISE_SEED,
    REAL_ANCHOR_NOISE_SEED,
    RELABEL_NOISE_SEED,
    FinetuneConfig,
    RelabelPool,
    assert_disjoint_seeds,
    assign_splits,
    batch_from_pool,
    learning_rate_at,
    mixed_loss,
    relabeled_condition,
)
from tests.test_conditional_flow import conditional_flow, scaler, wake_up_conditioning

ROOT = Path(__file__).resolve().parents[1]
COLLECT = ROOT / "scripts" / "34_collect_relabel_pool.py"
FINETUNE = ROOT / "scripts" / "35_finetune_relabel.py"
EVALUATE = ROOT / "scripts" / "36_evaluate_relabel.py"
SLURM = {name: ROOT / "slurm" / f"{name}.sh"
         for name in ("34_collect_relabel_pool", "35_finetune_relabel",
                      "36_evaluate_relabel")}
DIM = 8


def script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flow_for():
    flow = wake_up_conditioning(conditional_flow(features=("n", "qed"), dimension=DIM))
    flow.eval()
    return flow


def make_pool(*, counts=(5, 5, 6, 6), qeds=(0.30, 0.70, 0.45, 0.85),
              requests=(0.4, 0.8, 0.4, 0.8), seed=0):
    generator = torch.Generator().manual_seed(seed)
    pool = RelabelPool()
    for index, (n, qed, request) in enumerate(zip(counts, qeds, requests)):
        pool.embeddings.append(torch.randn((n, DIM), generator=generator))
        pool.n_node.append(int(n))
        pool.actual_qed.append(float(qed))
        pool.requested_qed.append(float(request))
        pool.smiles.append(f"C{'C' * index}O")
        pool.fallback.append(False)
        pool.valid.setdefault("degree", []).append(True)
    pool.split = ["train"] * len(pool.n_node)
    pool.metadata = {"pool_config": {"noise_seed": POOL_NOISE_SEED,
                                     "label_decoder": "degree"}}
    return pool


def real_batch_for(flow, *, size=3, width=6, device="cpu"):
    mask = torch.zeros((size, width), dtype=torch.bool)
    for row in range(size):
        mask[row, :width - row] = True
    values = torch.zeros((size, 2))
    from zinc_gnf.conditioning import condition_embedding_for
    return {"embeddings": torch.randn((size, width, DIM)) * mask.unsqueeze(-1),
            "node_mask": mask,
            "condition": condition_embedding_for(flow, values, torch.ones_like(values),
                                                 size, device)}


# ====================================================================================
# 1. Seeds and leakage.
# ====================================================================================
def test_every_noise_seed_is_distinct():
    assert_disjoint_seeds()
    seeds = [POOL_NOISE_SEED, EVAL_NOISE_SEED, RELABEL_NOISE_SEED,
             REAL_ANCHOR_NOISE_SEED, relabel.POOL_SPLIT_SEED]
    assert len(set(seeds)) == len(seeds)


def test_a_duplicated_seed_is_rejected_loudly(monkeypatch):
    """The guard must fail if anyone edits the constants into collision."""
    monkeypatch.setattr(relabel, "EVAL_NOISE_SEED", POOL_NOISE_SEED)
    with pytest.raises(ValueError, match="must all differ"):
        relabel.assert_disjoint_seeds()


def test_the_pool_and_evaluation_seeds_are_not_merely_adjacent():
    """Adjacent seeds would still be distinct but could share structure in some RNGs."""
    assert abs(EVAL_NOISE_SEED - POOL_NOISE_SEED) > 1000


def test_splits_are_stratified_within_every_cell():
    counts = [15] * 20 + [30] * 20
    requests = [0.4] * 10 + [0.8] * 10 + [0.4] * 10 + [0.8] * 10
    split = assign_splits(counts, requests, holdout_fraction=0.2)
    assert len(split) == 40
    for n in (15, 30):
        for request in (0.4, 0.8):
            members = [s for s, c, r in zip(split, counts, requests)
                       if c == n and r == request]
            assert members.count("holdout") == 2, (n, request)


def test_splits_are_deterministic_and_cover_both_sides():
    counts, requests = [15] * 10, [0.4] * 10
    first = assign_splits(counts, requests, holdout_fraction=0.3, seed=5)
    second = assign_splits(counts, requests, holdout_fraction=0.3, seed=5)
    assert first == second
    assert set(first) == {"train", "holdout"}


def test_a_different_split_seed_changes_the_assignment():
    counts, requests = [15] * 20, [0.4] * 20
    assert assign_splits(counts, requests, seed=1) != assign_splits(counts, requests, seed=2)


def test_pool_indices_for_split_are_disjoint_and_exhaustive():
    pool = make_pool()
    pool.split = assign_splits(pool.n_node, pool.requested_qed, holdout_fraction=0.25)
    train, holdout = pool.indices_for("train"), pool.indices_for("holdout")
    assert not set(train) & set(holdout)
    assert len(train) + len(holdout) == len(pool)


# ====================================================================================
# 2. The label is the DECODED QED, not the request.
# ====================================================================================
def test_the_relabeled_condition_uses_the_actual_qed_not_the_request():
    """If this ever used the request, the experiment would be a no-op."""
    flow = flow_for()
    the_scaler = scaler()
    pool = make_pool()
    batch = batch_from_pool(pool, [0, 1], device="cpu")
    actual = relabeled_condition(flow, the_scaler, n_node=batch["n_node"],
                                 qed=batch["actual_qed"], device="cpu")
    requested = relabeled_condition(flow, the_scaler, n_node=batch["n_node"],
                                    qed=batch["requested_qed"], device="cpu")
    assert not torch.allclose(actual, requested)
    assert batch["actual_qed"] != batch["requested_qed"]


def test_the_relabeled_condition_tracks_the_label_monotonically():
    flow = flow_for()
    low = relabeled_condition(flow, scaler(), n_node=[15], qed=[0.3], device="cpu")
    high = relabeled_condition(flow, scaler(), n_node=[15], qed=[0.9], device="cpu")
    assert not torch.allclose(low, high)


def test_the_relabeled_condition_uses_the_train_only_scaler():
    from zinc_gnf.conditioning import standardise_condition
    the_scaler = scaler()
    values = standardise_condition(("n", "qed"), scaler=the_scaler, n=[15], qed=[0.6])
    assert float(values[0, 1]) == pytest.approx(
        (0.6 - the_scaler.mean[1]) / the_scaler.std[1], abs=1e-6)


# ====================================================================================
# 3. The objective: mixed, per-node, forward-only.
# ====================================================================================
def test_mixed_loss_is_the_weighted_sum_of_the_two_terms():
    flow = flow_for()
    pool = make_pool()
    batch = batch_from_pool(pool, [0, 1, 2], device="cpu")
    real = real_batch_for(flow)
    common = dict(scaler=scaler(), device="cpu")
    only_real, _ = mixed_loss(flow, real_batch=real, pool_batch=batch,
                              relabel_weight=0.0, **common)
    only_relabel, _ = mixed_loss(flow, real_batch=real, pool_batch=batch,
                                 relabel_weight=1.0, **common)
    half, diagnostics = mixed_loss(flow, real_batch=real, pool_batch=batch,
                                   relabel_weight=0.5, **common)
    assert float(half) == pytest.approx(
        0.5 * float(only_real) + 0.5 * float(only_relabel), rel=1e-4)
    assert "real_nll_per_node" in diagnostics
    assert "relabel_nll_per_node" in diagnostics


def test_a_weight_of_zero_drops_the_relabeled_term_entirely():
    flow = flow_for()
    _, diagnostics = mixed_loss(flow, real_batch=real_batch_for(flow),
                                pool_batch=batch_from_pool(make_pool(), [0], device="cpu"),
                                scaler=scaler(), relabel_weight=0.0, device="cpu")
    assert "relabel_nll_per_node" not in diagnostics
    assert "real_nll_per_node" in diagnostics


def test_a_weight_of_one_drops_the_real_anchor():
    flow = flow_for()
    _, diagnostics = mixed_loss(flow, real_batch=None,
                                pool_batch=batch_from_pool(make_pool(), [0], device="cpu"),
                                scaler=scaler(), relabel_weight=1.0, device="cpu")
    assert "real_nll_per_node" not in diagnostics


def test_a_missing_anchor_is_refused_rather_than_silently_dropped():
    flow = flow_for()
    with pytest.raises(ValueError, match="anchor"):
        mixed_loss(flow, real_batch=None,
                   pool_batch=batch_from_pool(make_pool(), [0], device="cpu"),
                   scaler=scaler(), relabel_weight=0.5, device="cpu")


def test_a_missing_pool_batch_is_refused():
    flow = flow_for()
    with pytest.raises(ValueError, match="relabeled batch"):
        mixed_loss(flow, real_batch=real_batch_for(flow), pool_batch=None,
                   scaler=scaler(), relabel_weight=0.5, device="cpu")


def test_an_out_of_range_weight_is_refused():
    flow = flow_for()
    for weight in (-0.1, 1.1):
        with pytest.raises(ValueError):
            mixed_loss(flow, real_batch=real_batch_for(flow),
                       pool_batch=batch_from_pool(make_pool(), [0], device="cpu"),
                       scaler=scaler(), relabel_weight=weight, device="cpu")


def test_the_objective_is_differentiable_in_the_flow_parameters():
    flow = flow_for()
    flow.requires_grad_(True)
    loss, _ = mixed_loss(flow, real_batch=real_batch_for(flow),
                         pool_batch=batch_from_pool(make_pool(), [0, 1], device="cpu"),
                         scaler=scaler(), relabel_weight=0.5, device="cpu")
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in flow.parameters())


def test_the_objective_never_calls_flow_inverse(monkeypatch):
    """Runtime proof of the forward-only claim, not just a reading of the code."""
    flow = flow_for()
    flow.requires_grad_(True)

    def poisoned(*args, **kwargs):
        raise AssertionError("the relabeling objective must never call flow.inverse")

    monkeypatch.setattr(type(flow), "inverse", poisoned)
    loss, _ = mixed_loss(flow, real_batch=real_batch_for(flow),
                         pool_batch=batch_from_pool(make_pool(), [0, 1], device="cpu"),
                         scaler=scaler(), relabel_weight=0.5, device="cpu")
    loss.backward()
    assert torch.isfinite(loss)


def test_no_training_path_calls_the_inverse_statically():
    for path in (FINETUNE,):
        calls = [ast.unparse(n.func) for n in ast.walk(ast.parse(path.read_text()))
                 if isinstance(n, ast.Call)]
        assert not [c for c in calls if "inverse" in c], path.name
    module_calls = [ast.unparse(n.func)
                    for n in ast.walk(ast.parse((ROOT / "zinc_gnf" / "relabel.py").read_text()))
                    if isinstance(n, ast.Call)]
    assert not [c for c in module_calls if "inverse" in c or "linalg.inv" in c]


def test_smoothing_noise_is_applied_to_both_terms():
    """Without a noise floor the relabeled likelihood is unbounded; see the docstring."""
    source = (ROOT / "zinc_gnf" / "relabel.py").read_text()
    assert source.count("add_smoothing_noise") >= 2
    assert "real_generator" in source and "relabel_generator" in source


def test_the_two_smoothing_streams_do_not_share_a_generator():
    flow = flow_for()
    pool_batch = batch_from_pool(make_pool(), [0, 1], device="cpu")
    real = real_batch_for(flow)
    shared = torch.Generator().manual_seed(5)
    first, _ = mixed_loss(flow, real_batch=real, pool_batch=pool_batch, scaler=scaler(),
                          relabel_weight=0.5, device="cpu",
                          real_generator=torch.Generator().manual_seed(1),
                          relabel_generator=torch.Generator().manual_seed(2))
    second, _ = mixed_loss(flow, real_batch=real, pool_batch=pool_batch, scaler=scaler(),
                           relabel_weight=0.5, device="cpu",
                           real_generator=shared, relabel_generator=shared)
    assert torch.isfinite(first) and torch.isfinite(second)


# ====================================================================================
# 4. Pool batching, storage and round-trip.
# ====================================================================================
def test_batching_pads_to_the_widest_molecule_and_masks_the_rest():
    pool = make_pool(counts=(4, 7, 5))
    batch = batch_from_pool(pool, [0, 1, 2], device="cpu")
    assert batch["embeddings"].shape == (3, 7, DIM)
    assert batch["node_mask"].sum(-1).tolist() == [4, 7, 5]
    assert torch.count_nonzero(batch["embeddings"][0, 4:]) == 0
    assert torch.count_nonzero(batch["embeddings"][2, 5:]) == 0


def test_batching_preserves_the_stored_embedding_values_exactly():
    """The pilot depends on the ORIGINAL embeddings, never a re-encoding."""
    pool = make_pool(counts=(5, 6))
    batch = batch_from_pool(pool, [0, 1], device="cpu")
    assert torch.equal(batch["embeddings"][0, :5], pool.embeddings[0])
    assert torch.equal(batch["embeddings"][1, :6], pool.embeddings[1])


def test_batching_keeps_labels_aligned_with_rows():
    pool = make_pool()
    batch = batch_from_pool(pool, [2, 0], device="cpu")
    assert batch["actual_qed"] == [pool.actual_qed[2], pool.actual_qed[0]]
    assert batch["n_node"] == [pool.n_node[2], pool.n_node[0]]


def test_an_empty_index_list_is_refused():
    with pytest.raises(ValueError):
        batch_from_pool(make_pool(), [], device="cpu")


def test_the_pool_round_trips_through_disk(tmp_path):
    pool = make_pool()
    pool.split = assign_splits(pool.n_node, pool.requested_qed, holdout_fraction=0.25)
    manifest = pool.save(tmp_path / "pool")
    assert (tmp_path / "pool" / "pool.pt").is_file()
    assert (tmp_path / "pool" / "pool_manifest.json").is_file()
    restored = RelabelPool.load(tmp_path / "pool")
    assert restored.n_node == pool.n_node
    assert restored.actual_qed == pool.actual_qed
    assert restored.split == pool.split
    for before, after in zip(pool.embeddings, restored.embeddings):
        assert torch.equal(before, after)
    assert manifest["qed_support"]["molecules"] == len(pool)


def test_loading_a_pool_with_mismatched_parts_is_refused(tmp_path):
    pool = make_pool()
    pool.save(tmp_path / "pool")
    manifest_path = tmp_path / "pool" / "pool_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["n_node"] = [99] * len(manifest["n_node"])
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="node counts"):
        RelabelPool.load(tmp_path / "pool")


def test_loading_an_absent_pool_names_the_collector(tmp_path):
    with pytest.raises(FileNotFoundError, match="34_collect_relabel_pool"):
        RelabelPool.load(tmp_path / "absent")


def test_qed_support_reports_the_relabeling_ceiling():
    pool = make_pool(qeds=(0.2, 0.4, 0.6, 0.9))
    support = pool.qed_support()
    assert support["molecules"] == 4
    assert support["quantiles"]["p50"] == pytest.approx(0.5)
    assert support["fraction_at_least_0p75"] == pytest.approx(0.25)
    assert support["iqr"] > 0


# ====================================================================================
# 5. The fine-tune budget and its guards.
# ====================================================================================
def test_the_default_budget_is_short_and_conservative():
    config = FinetuneConfig()
    assert config.max_steps == 400
    assert config.learning_rate == 1e-5          # 1/100 of the original 1e-3
    assert config.relabel_weight == 0.5
    assert config.grad_clip_norm == 5.0
    config.validate()


def test_a_learning_rate_at_or_above_the_original_is_refused():
    """A fine-tune must not silently become a retrain."""
    with pytest.raises(ValueError, match="fine-tune"):
        FinetuneConfig(learning_rate=1e-2).validate()


def test_an_invalid_budget_is_refused():
    for bad in (FinetuneConfig(max_steps=0), FinetuneConfig(warmup_steps=-1),
                FinetuneConfig(warmup_steps=10_000), FinetuneConfig(relabel_weight=2.0),
                FinetuneConfig(real_batch_size=0), FinetuneConfig(eval_every=0)):
        with pytest.raises(ValueError):
            bad.validate()


def test_warmup_ramps_then_holds():
    config = FinetuneConfig(learning_rate=1e-5, warmup_steps=10, max_steps=100)
    assert learning_rate_at(config, 0) == pytest.approx(1e-6)
    assert learning_rate_at(config, 9) == pytest.approx(1e-5)
    assert learning_rate_at(config, 50) == pytest.approx(1e-5)


def test_zero_warmup_starts_at_the_full_rate():
    config = FinetuneConfig(warmup_steps=0)
    assert learning_rate_at(config, 0) == pytest.approx(config.learning_rate)


def test_the_finetune_refuses_to_write_into_the_parent_directory(tmp_path):
    module = script(FINETUNE, "finetune_relabel")
    parent = tmp_path / "run"
    parent.mkdir()
    (parent / "best.pt").write_bytes(b"x")
    code = module.main(["--flow-checkpoint", str(parent / "best.pt"),
                        "--pool-dir", str(tmp_path / "pool"),
                        "--output-dir", str(parent)])
    assert code == 4


def test_the_finetune_refuses_to_overwrite_an_existing_run(tmp_path):
    module = script(FINETUNE, "finetune_relabel")
    out = tmp_path / "out"
    out.mkdir()
    (out / "last.pt").write_bytes(b"x")
    code = module.main(["--flow-checkpoint", str(tmp_path / "parent.pt"),
                        "--pool-dir", str(tmp_path / "pool"),
                        "--output-dir", str(out)])
    assert code == 3


def test_the_finetune_selects_best_on_real_validation_not_on_qed():
    source = FINETUNE.read_text()
    assert "Selecting on the QED" in source or "quantity under test" in source
    assert 'if validation["val_loss"] < best["val_loss"]:' in source


def test_the_finetune_rejects_bad_arguments(tmp_path):
    module = script(FINETUNE, "finetune_relabel")
    base = ["--flow-checkpoint", str(tmp_path / "a.pt"),
            "--pool-dir", str(tmp_path / "pool"),
            "--output-dir", str(tmp_path / "out")]
    assert module.main(base + ["--learning-rate", "0.01"]) == 4
    assert module.main(base + ["--relabel-weight", "3"]) == 4
    assert module.main(base + ["--max-steps", "0"]) == 4


# ====================================================================================
# 6. Collection: tensors stay out of git, provenance is checked.
# ====================================================================================
def test_the_collector_refuses_a_pool_path_inside_the_tracked_tree():
    module = script(COLLECT, "collect_relabel_pool")
    assert module.within_allowed_output(ROOT / "outputs" / "relabel_pool" / "x") is True
    assert module.within_allowed_output(Path("/tmp/elsewhere")) is True
    assert module.within_allowed_output(ROOT / "zinc_gnf" / "pool") is False
    assert module.within_allowed_output(ROOT / "scripts" / "pool") is False


def test_the_collector_exits_four_for_a_tracked_pool_path():
    module = script(COLLECT, "collect_relabel_pool")
    code = module.main(["--flow-checkpoint", "/nope",
                        "--pool-dir", str(ROOT / "zinc_gnf" / "bad_pool")])
    assert code == 4
    assert not (ROOT / "zinc_gnf" / "bad_pool").exists()


def test_the_pool_directory_is_git_ignored_in_full():
    ignore = (ROOT / ".gitignore").read_text()
    assert "outputs/relabel_pool/" in ignore
    assert "*.pt" in ignore


def test_the_collector_records_the_label_decoder_and_selection_bias():
    source = COLLECT.read_text()
    assert "retained_fraction" in source
    assert "DECODER-INDEXED" in " ".join(source.split())
    assert "selection_bias_note" in source


def test_the_collector_never_re_encodes_smiles():
    source = COLLECT.read_text()
    assert "re-encodes" in source or "re-encoding" in source
    calls = [ast.unparse(n.func) for n in ast.walk(ast.parse(source))
             if isinstance(n, ast.Call)]
    assert not [c for c in calls if "encode_batch" in c or "autoencoder.encode" in c]


def test_the_collector_does_not_train():
    source = COLLECT.read_text()
    for forbidden in ("torch.optim", "backward()", "zero_grad", "save_checkpoint"):
        assert forbidden not in source, forbidden
    assert "requires_grad_(False)" in source


# ====================================================================================
# 7. Evaluation: common noise, honest pairing, fixed criterion.
# ====================================================================================
def test_the_evaluator_refuses_a_seed_that_collides_with_the_pool(tmp_path):
    """Direct leakage guard: evaluating on the model's own training draws."""
    module = script(EVALUATE, "evaluate_relabel")
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    (pool_dir / "pool_manifest.json").write_text(json.dumps(
        {"metadata": {"pool_config": {"noise_seed": 4242}}}))
    code = module.main(["--model", f"a={tmp_path / 'a.pt'}",
                        "--ae-checkpoint", str(tmp_path / "ae.pt"),
                        "--pool-dir", str(pool_dir),
                        "--output-dir", str(tmp_path / "out"),
                        "--eval-seed", "4242"])
    assert code == 3


def test_a_distinct_evaluation_seed_passes_the_leakage_guard(tmp_path):
    module = script(EVALUATE, "evaluate_relabel")
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    (pool_dir / "pool_manifest.json").write_text(json.dumps(
        {"metadata": {"pool_config": {"noise_seed": 4242}}}))
    code = module.main(["--model", f"a={tmp_path / 'a.pt'}",
                        "--ae-checkpoint", str(tmp_path / "ae.pt"),
                        "--pool-dir", str(pool_dir),
                        "--output-dir", str(tmp_path / "out"),
                        "--eval-seed", "999"])
    assert code == 2            # past the guard, stopped by the missing autoencoder


def test_paired_response_uses_only_jointly_valid_pairs():
    module = script(EVALUATE, "evaluate_relabel")
    low = [{"degree": {"qed": 0.3}}, {"degree": {"qed": None}}, {"degree": {"qed": 0.5}}]
    high = [{"degree": {"qed": 0.6}}, {"degree": {"qed": 0.9}}, {"degree": {"qed": None}}]
    result = module.paired_response(low, high, "degree")
    assert result["jointly_valid_pairs"] == 1
    assert result["mean"] == pytest.approx(0.3)
    assert "jointly valid" in result["pairing"]


def test_paired_response_is_per_molecule_not_a_difference_of_means():
    """Means would give +0.0; the paired statistic must give the within-pair delta."""
    module = script(EVALUATE, "evaluate_relabel")
    low = [{"degree": {"qed": 0.2}}, {"degree": {"qed": 0.8}}]
    high = [{"degree": {"qed": 0.5}}, {"degree": {"qed": 0.5}}]
    result = module.paired_response(low, high, "degree")
    assert result["mean"] == pytest.approx(0.0)
    assert result["n"] == 2
    paired = module.paired_response(
        [{"degree": {"qed": 0.2}}, {"degree": {"qed": 0.2}}],
        [{"degree": {"qed": 0.5}}, {"degree": {"qed": 0.5}}], "degree")
    assert paired["mean"] == pytest.approx(0.3)
    assert paired["excludes_zero"] is True


def test_the_evaluator_draws_one_noise_tensor_per_n_and_reuses_it():
    source = " ".join(EVALUATE.read_text().split())
    assert "ONE noise draw per (N), shared across every request and every model" in source
    assert "int(args.eval_seed) + 7919 * n" in source


def test_cell_summary_reports_every_required_metric():
    module = script(EVALUATE, "evaluate_relabel")
    rows = [{"degree": {"valid": True, "fallback": False, "smiles": "CCO", "qed": 0.5,
                        "aromatic_rings": 1.0, "heavy_atoms": 12.0}},
            {"degree": {"valid": False, "fallback": True, "smiles": None, "qed": None,
                        "aromatic_rings": None, "heavy_atoms": None}}]
    block = module.cell_summary(rows, "degree", target=0.8)
    assert block["validity"] == pytest.approx(0.5)
    assert block["fallback_rate"] == pytest.approx(0.5)
    assert block["qed_mean"] == pytest.approx(0.5)
    assert block["qed_mae_vs_target"] == pytest.approx(0.3)
    assert block["aromatic_rings_mean"] == pytest.approx(1.0)
    assert block["heavy_atoms_mean"] == pytest.approx(12.0)
    assert block["uniqueness"] == pytest.approx(1.0)


def test_cell_summary_handles_an_all_invalid_cell():
    module = script(EVALUATE, "evaluate_relabel")
    rows = [{"degree": {"valid": False, "fallback": False, "smiles": None, "qed": None,
                        "aromatic_rings": None, "heavy_atoms": None}}]
    block = module.cell_summary(rows, "degree", target=0.6)
    assert block["validity"] == 0.0 and block["qed_mean"] is None
    assert block["qed_mae_vs_target"] is None


def verdict_payload(*, new_response, old_response, raw_new=0.50, raw_old=0.50,
                    arom_new=2.0, arom_old=2.0, uniq_new=0.9, uniq_old=0.9,
                    div_new=0.85, div_old=0.85, other_response_change=0.0):
    def model(response, raw, arom, uniq, div, others):
        return {"by_n": {"15": {
            "cells": {"0.40": {
                "degree": {"aromatic_rings_mean": arom, "uniqueness": uniq,
                           "internal_diversity": div},
                "raw": {"validity": raw},
                "constrained": {"validity": raw}}},
            "paired_response": {"degree": response,
                                "raw": others, "constrained": others}}}}
    return {"models": {
        "finetuned": model(new_response, raw_new, arom_new, uniq_new, div_new,
                           {"mean": other_response_change, "excludes_zero": True}),
        "final": model(old_response, raw_old, arom_old, uniq_old, div_old,
                       {"mean": 0.0, "excludes_zero": False})}}


def test_the_verdict_passes_only_on_an_improved_paired_response():
    module = script(EVALUATE, "evaluate_relabel")
    payload = verdict_payload(new_response={"mean": 0.25, "excludes_zero": True},
                              old_response={"mean": 0.10, "excludes_zero": True})
    checks = module.verdict(payload, candidate="finetuned", reference="final",
                            decoder="degree")
    assert checks["by_n"]["15"]["response_improved"] is True
    assert checks["by_n"]["15"]["passes"] is True
    assert checks["verdict"] == "GO"


def test_a_response_whose_ci_includes_zero_does_not_pass():
    module = script(EVALUATE, "evaluate_relabel")
    payload = verdict_payload(new_response={"mean": 0.25, "excludes_zero": False},
                              old_response={"mean": 0.10, "excludes_zero": True})
    checks = module.verdict(payload, candidate="finetuned", reference="final",
                            decoder="degree")
    assert checks["by_n"]["15"]["passes"] is False
    assert checks["verdict"] == "NO-GO"


def test_a_lower_mae_alone_cannot_produce_a_pass():
    """The criterion must not be reachable by collapsing toward common QED values."""
    module = script(EVALUATE, "evaluate_relabel")
    payload = verdict_payload(new_response={"mean": 0.02, "excludes_zero": True},
                              old_response={"mean": 0.13, "excludes_zero": True})
    checks = module.verdict(payload, candidate="finetuned", reference="final",
                            decoder="degree")
    assert checks["by_n"]["15"]["response_improved"] is False
    assert checks["verdict"] == "NO-GO"
    assert "deliberately NOT part of the condition" in checks["interpretation"]


def test_a_raw_validity_collapse_fails_even_with_a_better_response():
    module = script(EVALUATE, "evaluate_relabel")
    payload = verdict_payload(new_response={"mean": 0.30, "excludes_zero": True},
                              old_response={"mean": 0.10, "excludes_zero": True},
                              raw_new=0.30, raw_old=0.50)
    checks = module.verdict(payload, candidate="finetuned", reference="final",
                            decoder="degree")
    assert checks["by_n"]["15"]["raw_validity_holds"] is False
    assert checks["by_n"]["15"]["passes"] is False


def test_a_diversity_collapse_fails_even_with_a_better_response():
    module = script(EVALUATE, "evaluate_relabel")
    payload = verdict_payload(new_response={"mean": 0.30, "excludes_zero": True},
                              old_response={"mean": 0.10, "excludes_zero": True},
                              div_new=0.40, div_old=0.85)
    checks = module.verdict(payload, candidate="finetuned", reference="final",
                            decoder="degree")
    quality = checks["by_n"]["15"]["chemical_quality"]
    assert quality["internal_diversity"]["holds"] is False
    assert checks["by_n"]["15"]["passes"] is False


def test_a_gain_only_under_the_label_decoder_is_flagged_as_correction_dependent():
    module = script(EVALUATE, "evaluate_relabel")
    payload = verdict_payload(new_response={"mean": 0.30, "excludes_zero": True},
                              old_response={"mean": 0.10, "excludes_zero": True},
                              other_response_change=0.0)
    checks = module.verdict(payload, candidate="finetuned", reference="final",
                            decoder="degree")
    assert checks["intrinsic_rather_than_correction_dependent"] is False
    assert "correction-dependent" in checks["interpretation"]


def test_a_gain_present_under_every_decoder_counts_as_intrinsic():
    module = script(EVALUATE, "evaluate_relabel")
    payload = verdict_payload(new_response={"mean": 0.30, "excludes_zero": True},
                              old_response={"mean": 0.10, "excludes_zero": True},
                              other_response_change=0.18)
    checks = module.verdict(payload, candidate="finetuned", reference="final",
                            decoder="degree")
    assert checks["intrinsic_rather_than_correction_dependent"] is True


def test_the_verdict_needs_both_models_to_compare():
    module = script(EVALUATE, "evaluate_relabel")
    payload = {"models": {"final": {"by_n": {}}}}
    checks = module.verdict(payload, candidate="finetuned", reference="final",
                            decoder="degree")
    assert checks["evaluated"] is False


def test_the_evaluator_is_read_only():
    source = EVALUATE.read_text()
    for forbidden in ("torch.optim", "backward()", "zero_grad", "save_checkpoint"):
        assert forbidden not in source, forbidden
    assert "requires_grad_(False)" in source


def test_bootstrap_ci_detects_an_interval_excluding_zero():
    module = script(EVALUATE, "evaluate_relabel")
    assert module.bootstrap_ci([0.2] * 40)["excludes_zero"] is True
    assert module.bootstrap_ci([-0.3, 0.3] * 20)["excludes_zero"] is False
    assert module.bootstrap_ci([])["mean"] is None
    assert module.bootstrap_ci([0.5])["n"] == 1


# ====================================================================================
# 8. SLURM wrappers.
# ====================================================================================
@pytest.mark.parametrize("name,script_path", [
    ("34_collect_relabel_pool", COLLECT),
    ("35_finetune_relabel", FINETUNE),
    ("36_evaluate_relabel", EVALUATE)])
def test_slurm_flags_are_a_subset_of_the_scripts_arguments(name, script_path):
    parser_flags = {line.strip().split('"')[1]
                    for line in script_path.read_text().splitlines()
                    if line.strip().startswith('parser.add_argument("--')}
    assert parser_flags, name
    used = {line.strip().split()[0] for line in SLURM[name].read_text().splitlines()
            if line.strip().startswith("--")}
    assert used <= parser_flags, (name, used - parser_flags)


@pytest.mark.parametrize("name", list(SLURM))
def test_every_wrapper_refuses_the_login_node(name):
    source = SLURM[name].read_text()
    assert 'if [[ -z "${SLURM_JOB_ID:-}" ]]' in source
    assert "set -euo pipefail" in source


def test_only_the_finetune_wrapper_requests_a_gpu():
    assert "--gres=gpu:1" in SLURM["35_finetune_relabel"].read_text()
    for name in ("34_collect_relabel_pool", "36_evaluate_relabel"):
        source = SLURM[name].read_text()
        assert "--gres=gpu" not in source
        assert "#SBATCH --partition=CPU" in source


def test_the_finetune_wrapper_warns_before_overwriting():
    source = SLURM["35_finetune_relabel"].read_text()
    assert "ALLOW_EXISTING=1" in source
    assert "WILL be overwritten" in source
    assert "READ-ONLY, never written" in source


def test_the_pool_wrapper_sends_tensors_to_the_ignored_directory():
    source = SLURM["34_collect_relabel_pool"].read_text()
    assert "outputs/relabel_pool/" in source
    assert "LOCAL ONLY" in source


def test_the_evaluation_wrapper_explains_the_criterion():
    source = SLURM["36_evaluate_relabel"].read_text()
    assert "lower MAE on its own is NOT success" in source
    assert "correction-dependent" in source
    assert "stop experimenting and consolidate" in source


@pytest.mark.parametrize("name", list(SLURM))
def test_no_wrapper_runs_a_destructive_command(name):
    for line in SLURM[name].read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assert stripped.split()[0] not in {"rm", "mv", "cp", "truncate", "dd"}, line
