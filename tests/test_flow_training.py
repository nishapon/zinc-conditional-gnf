"""Flow training loop: frozen autoencoder, split integrity, exact resume,
embedding statistics, the non-finite-loss guard and checkpoint loading.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.test_autoencoder_training import write_dataset
from zinc_gnf.autoencoder import AutoencoderConfig, GraphAutoencoder
from zinc_gnf.dataset import MolecularGraphDataset, iter_eval_batches, load_training_splits
from zinc_gnf.flow import FlowConfig, NodeFlow
from zinc_gnf.trainer_autoencoder import ResumeMismatchError
from zinc_gnf.trainer_flow import (
    FlowTrainConfig,
    FlowTrainer,
    autoencoder_seen_indices,
    embedding_statistics_from_dataset,
    file_sha256,
    load_flow,
)
from zinc_gnf.utils.logging import null_logger

EMBEDDING_DIM = 16
FLOW = dict(node_embedding_dim=EMBEDDING_DIM, num_blocks=2, attention_dim=8, num_heads=2,
            hidden_dim=16)


def tiny_autoencoder(degree_head: bool = True) -> GraphAutoencoder:
    model = GraphAutoencoder(AutoencoderConfig(
        node_embedding_dim=EMBEDDING_DIM, latent_dim=32, num_processing_steps=1,
        attn_kq_dim=8, attn_v_dim=8, attn_concat_heads_output_dim=8, degree_head=degree_head))
    model.eval()
    return model


def train_config(tmp_path: Path, data: Path, **overrides) -> FlowTrainConfig:
    values = dict(
        data_dir=str(data), output_dir=str(tmp_path / "flow"), seed=1, device="cpu",
        batch_size=4, max_steps=6, learning_rate=1e-3, warmup_steps=0, log_every=2,
        eval_every=3, save_every=3, val_molecules=8, statistics_molecules=20,
        early_stopping_patience=None, ae_checkpoint="<unused>")
    values.update(overrides)
    return FlowTrainConfig(**values)


def run_trainer(tmp_path: Path, data: Path, *, autoencoder=None, flow_overrides=None,
                resume="none", **overrides):
    flow_config = FlowConfig(**{**FLOW, **(flow_overrides or {})})
    trainer = FlowTrainer(flow_config, train_config(tmp_path, data, **overrides),
                          logger=null_logger(), autoencoder=autoencoder or tiny_autoencoder())
    return trainer, trainer.train(resume=resume)


# ---------------------------------------------------------------------------
# End to end.
# ---------------------------------------------------------------------------
def test_training_writes_every_artifact_and_a_finite_loss(tmp_path):
    data = write_dataset(tmp_path / "data")
    # no injected logger: the trainer must create train.log in its own output directory
    trainer = FlowTrainer(FlowConfig(**FLOW), train_config(tmp_path, data),
                          autoencoder=tiny_autoencoder())
    result = trainer.train(resume="none")
    output = Path(trainer.config.output_dir)
    for name in ("config.json", "metrics.jsonl", "train.log", "last.pt", "best.pt", "summary.json"):
        assert (output / name).exists(), name
    assert result["completed"] and result["final_step"] == 6
    assert np.isfinite(result["best_val_loss"])
    assert result["best_step"] in (3, 6)
    configuration = json.loads((output / "config.json").read_text())
    assert configuration["flow"]["node_embedding_dim"] == EMBEDDING_DIM
    assert configuration["embedding_statistics"]["molecules"] == 20
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert {r["kind"] for r in records} == {"train", "val"}
    assert all(np.isfinite(r["loss"]) for r in records if r["kind"] == "train")


def test_validation_is_deterministic_and_uses_a_fixed_subset(tmp_path):
    data = write_dataset(tmp_path / "data")
    trainer, _ = run_trainer(tmp_path, data)
    flow = NodeFlow(trainer.flow_config)
    flow.set_data_statistics(torch.zeros(EMBEDDING_DIM), torch.ones(EMBEDDING_DIM))
    first = trainer.validate(flow)
    second = trainer.validate(flow)
    assert first == second                                  # fixed noise seed, fixed subset
    assert first["val_molecules"] == len(trainer.val_indices)


def test_the_test_split_is_never_touched(tmp_path):
    data = write_dataset(tmp_path / "data")
    payload_train, payload_val, _ = load_training_splits(data)
    trainer, result = run_trainer(tmp_path, data)
    assert result["data"]["used"] == {"train": len(payload_train), "val": len(payload_val)}
    assert not hasattr(trainer, "test_set")
    source = Path("zinc_gnf/trainer_flow.py").read_text()
    assert "load_test_split" not in source
    assert "load_training_splits" in source


# ---------------------------------------------------------------------------
# The autoencoder is frozen.
# ---------------------------------------------------------------------------
def test_autoencoder_parameters_are_frozen_and_unchanged(tmp_path):
    data = write_dataset(tmp_path / "data")
    autoencoder = tiny_autoencoder()
    before = copy.deepcopy(autoencoder.state_dict())
    trainer, _ = run_trainer(tmp_path, data, autoencoder=autoencoder)
    assert all(not p.requires_grad for p in autoencoder.parameters())
    assert not autoencoder.training
    after = autoencoder.state_dict()
    for key, value in before.items():
        assert torch.equal(value, after[key]), key


def test_embedding_width_must_match_the_autoencoder(tmp_path):
    data = write_dataset(tmp_path / "data")
    with pytest.raises(ResumeMismatchError, match="embedding width"):
        run_trainer(tmp_path, data, flow_overrides={"node_embedding_dim": 32})


def test_autoencoder_checkpoint_is_recorded_with_its_hash(tmp_path):
    data = write_dataset(tmp_path / "data")
    autoencoder = tiny_autoencoder()
    checkpoint = tmp_path / "ae.pt"
    torch.save({"model_state_dict": autoencoder.state_dict(),
                "model_config": autoencoder.config.to_dict(),
                "vocabulary_fingerprint": __import__(
                    "zinc_gnf.constants_charge_aware", fromlist=["x"]).vocabulary_fingerprint(),
                "step": 42}, checkpoint)
    flow_config = FlowConfig(**FLOW)
    trainer = FlowTrainer(flow_config, train_config(tmp_path, data, ae_checkpoint=str(checkpoint)),
                          logger=null_logger())
    result = trainer.train(resume="none")
    assert result["autoencoder"]["sha256"] == file_sha256(checkpoint)
    assert result["autoencoder"]["step"] == 42
    assert Path(result["autoencoder"]["checkpoint"]) == checkpoint


def test_missing_autoencoder_checkpoint_is_a_clear_error(tmp_path):
    data = write_dataset(tmp_path / "data")
    flow_config = FlowConfig(**FLOW)
    trainer = FlowTrainer(flow_config, train_config(tmp_path, data, ae_checkpoint="nope.pt"),
                          logger=null_logger())
    with pytest.raises(FileNotFoundError):
        trainer.train(resume="none")


# ---------------------------------------------------------------------------
# Embedding statistics.
# ---------------------------------------------------------------------------
def test_embedding_statistics_match_a_direct_computation(tmp_path):
    data = write_dataset(tmp_path / "data")
    records, _, _ = load_training_splits(data)
    dataset = MolecularGraphDataset(records)
    autoencoder = tiny_autoencoder()
    indices = list(range(len(dataset)))
    statistics = embedding_statistics_from_dataset(autoencoder, dataset, indices=indices,
                                                   device="cpu", batch_size=4)
    collected = []
    with torch.no_grad():
        for batch in iter_eval_batches(dataset, 4, indices):
            nodes = autoencoder.encode(batch)
            collected.append(nodes[batch.node_mask])
    stacked = torch.cat(collected)
    assert statistics["nodes"] == stacked.shape[0]
    assert torch.allclose(statistics["mean"], stacked.mean(0), atol=1e-4)
    assert torch.allclose(statistics["std"], stacked.std(0, unbiased=False), atol=1e-4)


def test_statistics_are_measured_on_train_only_and_reused_on_resume(tmp_path):
    data = write_dataset(tmp_path / "data")
    autoencoder = tiny_autoencoder()
    trainer, _ = run_trainer(tmp_path, data, autoencoder=autoencoder)
    first = torch.load(Path(trainer.config.output_dir) / "last.pt", map_location="cpu",
                       weights_only=False)
    mean = first["model_state_dict"]["data_mean"].clone()

    def explode(*args, **kwargs):                 # must not be called again on resume
        raise AssertionError("statistics were recomputed on resume")

    trainer2 = FlowTrainer(FlowConfig(**FLOW), train_config(tmp_path, data, max_steps=8),
                           logger=null_logger(), autoencoder=autoencoder)
    trainer2._fit_statistics = explode
    trainer2.train(resume="auto")
    second = torch.load(Path(trainer2.config.output_dir) / "last.pt", map_location="cpu",
                        weights_only=False)
    assert torch.equal(second["model_state_dict"]["data_mean"], mean)


def test_refuses_to_train_without_statistics(tmp_path):
    data = write_dataset(tmp_path / "data")
    flow_config = FlowConfig(**FLOW)
    trainer = FlowTrainer(flow_config, train_config(tmp_path, data), logger=null_logger(),
                          autoencoder=tiny_autoencoder())
    trainer._fit_statistics = lambda flow: None           # pretend measurement silently failed
    with pytest.raises(RuntimeError, match="statistics"):
        trainer.train(resume="none")


def test_autoencoder_seen_indices_reproduce_the_experiment_scripts_choice():
    pool, count, seed = 2000, 500, 3
    expected = sorted(int(i) for i in np.random.default_rng(seed).permutation(pool)[:count])
    assert autoencoder_seen_indices(pool, seed, count) == expected
    assert autoencoder_seen_indices(10, 3, 50) == list(range(10))
    assert len(autoencoder_seen_indices(202111, 3, 100000)) == 100000


def test_ae_seen_only_restricts_the_training_pool(tmp_path):
    data = write_dataset(tmp_path / "data")
    trainer, result = run_trainer(tmp_path, data, ae_seen_only=True, ae_subset_seed=3,
                                  ae_subset_size=12, max_steps=2, eval_every=2, save_every=2)
    assert result["data"]["used"]["train"] == 12
    assert result["data"]["ae_seen_only"] is True
    assert len(trainer.train_set) == 12


# ---------------------------------------------------------------------------
# Resume.
# ---------------------------------------------------------------------------
def test_resume_is_exact_three_plus_three_equals_six(tmp_path):
    data = write_dataset(tmp_path / "data")
    autoencoder = tiny_autoencoder()
    straight, _ = run_trainer(tmp_path / "straight", data, autoencoder=autoencoder, max_steps=6,
                              eval_every=6, save_every=6)
    split_first, _ = run_trainer(tmp_path / "split", data, autoencoder=autoencoder, max_steps=3,
                                 eval_every=3, save_every=3)
    trainer = FlowTrainer(FlowConfig(**FLOW),
                          train_config(tmp_path / "split", data, max_steps=6, eval_every=3,
                                       save_every=3),
                          logger=null_logger(), autoencoder=autoencoder)
    trainer.train(resume="auto")
    one = torch.load(Path(straight.config.output_dir) / "last.pt", map_location="cpu",
                     weights_only=False)
    two = torch.load(Path(trainer.config.output_dir) / "last.pt", map_location="cpu",
                     weights_only=False)
    assert one["step"] == two["step"] == 6
    assert one["batcher"] == two["batcher"]
    for key, value in one["model_state_dict"].items():
        assert torch.equal(value, two["model_state_dict"][key]), key


@pytest.mark.parametrize("change,message", [
    ({"flow_overrides": {"num_blocks": 3}}, "flow configuration"),
    ({"seed": 99}, "seed"),
    ({"batch_size": 2}, "batch_size"),
    ({"learning_rate": 5e-4}, "learning_rate"),
])
def test_resume_refuses_a_different_setup(tmp_path, change, message):
    data = write_dataset(tmp_path / "data")
    autoencoder = tiny_autoencoder()
    run_trainer(tmp_path, data, autoencoder=autoencoder, max_steps=3, eval_every=3, save_every=3)
    flow_overrides = change.pop("flow_overrides", None)
    with pytest.raises(ResumeMismatchError, match=message):
        run_trainer(tmp_path, data, autoencoder=autoencoder, resume="auto",
                    flow_overrides=flow_overrides, **change)


def test_resume_refuses_a_different_autoencoder(tmp_path):
    data = write_dataset(tmp_path / "data")
    from zinc_gnf.constants_charge_aware import vocabulary_fingerprint
    paths = []
    for index in (0, 1):
        torch.manual_seed(index)
        model = tiny_autoencoder()
        path = tmp_path / f"ae{index}.pt"
        torch.save({"model_state_dict": model.state_dict(),
                    "model_config": model.config.to_dict(),
                    "vocabulary_fingerprint": vocabulary_fingerprint()}, path)
        paths.append(path)
    first = FlowTrainer(FlowConfig(**FLOW),
                        train_config(tmp_path, data, ae_checkpoint=str(paths[0]), max_steps=3,
                                     eval_every=3, save_every=3), logger=null_logger())
    first.train(resume="none")
    second = FlowTrainer(FlowConfig(**FLOW),
                         train_config(tmp_path, data, ae_checkpoint=str(paths[1]), max_steps=6,
                                      eval_every=3, save_every=3), logger=null_logger())
    with pytest.raises(ResumeMismatchError, match="autoencoder checkpoint differs"):
        second.train(resume="auto")


def test_missing_resume_checkpoint_is_an_error(tmp_path):
    data = write_dataset(tmp_path / "data")
    trainer = FlowTrainer(FlowConfig(**FLOW), train_config(tmp_path, data), logger=null_logger(),
                          autoencoder=tiny_autoencoder())
    with pytest.raises(FileNotFoundError):
        trainer.train(resume=str(tmp_path / "absent.pt"))


# ---------------------------------------------------------------------------
# Guards.
# ---------------------------------------------------------------------------
def test_non_finite_loss_saves_the_state_and_raises(tmp_path, monkeypatch):
    data = write_dataset(tmp_path / "data")

    class ExplodingFlow(NodeFlow):
        def negative_log_likelihood(self, x, mask, condition=None):
            parts = super().negative_log_likelihood(x, mask, condition)
            parts["loss"] = parts["loss"] * float("nan")
            return parts

    monkeypatch.setattr("zinc_gnf.trainer_flow.NodeFlow", ExplodingFlow)
    with pytest.raises(FloatingPointError, match="non-finite loss"):
        run_trainer(tmp_path, data, autoencoder=tiny_autoencoder())
    assert (Path(tmp_path / "flow") / "last.pt").exists()


def test_learning_rate_warms_up_linearly(tmp_path):
    data = write_dataset(tmp_path / "data")
    trainer = FlowTrainer(FlowConfig(**FLOW),
                          train_config(tmp_path, data, warmup_steps=10, learning_rate=1e-3),
                          logger=null_logger(), autoencoder=tiny_autoencoder())
    assert trainer._learning_rate(0) == pytest.approx(1e-4)
    assert trainer._learning_rate(9) == pytest.approx(1e-3)
    assert trainer._learning_rate(500) == pytest.approx(1e-3)


@pytest.mark.parametrize("bad", [
    {"batch_size": 0}, {"max_steps": 0}, {"device": "tpu"}, {"warmup_steps": -1},
    {"ae_seen_only": True, "ae_subset_seed": None},
])
def test_training_configuration_validation(bad):
    with pytest.raises(ValueError):
        FlowTrainConfig(**bad).validate()


def test_unknown_training_setting_is_rejected():
    with pytest.raises(ValueError, match="unknown flow training settings"):
        FlowTrainConfig.from_dict({"nonsense": 1})


# ---------------------------------------------------------------------------
# Loading a trained flow back.
# ---------------------------------------------------------------------------
def test_load_flow_restores_the_density_exactly(tmp_path):
    data = write_dataset(tmp_path / "data")
    trainer, _ = run_trainer(tmp_path, data)
    checkpoint = Path(trainer.config.output_dir) / "best.pt"
    flow, probe = load_flow(checkpoint, device="cpu")
    assert bool(flow.statistics_set)
    assert probe["flow_config"]["num_blocks"] == 2
    records, _, _ = load_training_splits(data)
    dataset = MolecularGraphDataset(records[:4])
    batch = next(iter(iter_eval_batches(dataset, 4)))
    with torch.no_grad():
        embeddings = tiny_autoencoder().encode(batch)
        first = flow.log_prob(embeddings, batch.node_mask)["log_prob"]
        reloaded, _ = load_flow(checkpoint, device="cpu")
        second = reloaded.log_prob(embeddings, batch.node_mask)["log_prob"]
    assert torch.allclose(first, second)


def test_load_flow_refuses_a_checkpoint_without_statistics(tmp_path):
    from zinc_gnf.constants_charge_aware import vocabulary_fingerprint
    flow = NodeFlow(FlowConfig(**FLOW))
    path = tmp_path / "unfit.pt"
    torch.save({"model_state_dict": flow.state_dict(), "flow_config": flow.config.to_dict(),
                "vocabulary_fingerprint": vocabulary_fingerprint()}, path)
    with pytest.raises(ResumeMismatchError, match="no embedding statistics"):
        load_flow(path)


def test_load_flow_refuses_a_foreign_vocabulary(tmp_path):
    flow = NodeFlow(FlowConfig(**FLOW))
    path = tmp_path / "foreign.pt"
    torch.save({"model_state_dict": flow.state_dict(), "flow_config": flow.config.to_dict(),
                "vocabulary_fingerprint": "deadbeef"}, path)
    with pytest.raises(ResumeMismatchError, match="different atom vocabulary"):
        load_flow(path)


def test_file_sha256_matches_hashlib(tmp_path):
    import hashlib
    path = tmp_path / "blob.bin"
    payload = b"zinc" * 1000
    path.write_bytes(payload)
    assert file_sha256(path) == hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Compatibility with a checkpoint written by the REAL autoencoder trainer.
# ---------------------------------------------------------------------------
def real_autoencoder_checkpoint(tmp_path: Path, data: Path) -> Path:
    """Train the autoencoder for two steps through `AutoencoderTrainer`.

    This is the code path that produced `outputs/ablation/ae_degree/seed_3/last.pt`
    (via scripts/03b), so the resulting file has the real key layout: `configuration`,
    `model_config`, `data_fingerprint`, `batcher`, optimizer state, RNG states and the
    rest of `_state()`. The hand-written checkpoints used elsewhere in these tests
    only carry three keys and would not catch a layout mismatch.
    """
    from zinc_gnf.trainer_autoencoder import AutoencoderTrainer, TrainConfig

    model_config = AutoencoderConfig(
        node_embedding_dim=EMBEDDING_DIM * 4, latent_dim=32, num_processing_steps=1,
        attn_kq_dim=8, attn_v_dim=8, attn_concat_heads_output_dim=8, degree_head=True)
    trainer = AutoencoderTrainer(
        model_config,
        TrainConfig(data_dir=str(data), output_dir=str(tmp_path / "ae_run"), seed=3,
                    device="cpu", batch_size=4, max_steps=2, log_every=2, eval_every=2,
                    save_every=2, val_loss_molecules=8, val_recon_molecules=4,
                    selection_key="exact_molecule_degree"),
        logger=null_logger())
    trainer.train(resume="none")
    return tmp_path / "ae_run" / "last.pt"


def test_flow_accepts_a_checkpoint_from_the_real_autoencoder_trainer(tmp_path):
    data = write_dataset(tmp_path / "data")
    checkpoint = real_autoencoder_checkpoint(tmp_path, data)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # the layout the real seed_3/last.pt has
    assert {"model_config", "vocabulary_fingerprint", "model_state_dict", "configuration",
            "data_fingerprint", "batcher", "step"} <= set(payload)
    assert payload["model_config"]["degree_head"] is True
    assert payload["configuration"]["training"]["selection_key"] == "exact_molecule_degree"

    from zinc_gnf.trainer_autoencoder import load_autoencoder
    autoencoder, probe = load_autoencoder(checkpoint, device="cpu")
    assert autoencoder.config.degree_head and autoencoder.decoder.degree_head is not None
    assert autoencoder.config.node_embedding_dim == EMBEDDING_DIM * 4

    flow_config = FlowConfig(**{**FLOW, "node_embedding_dim": EMBEDDING_DIM * 4})
    trainer = FlowTrainer(flow_config,
                          train_config(tmp_path, data, ae_checkpoint=str(checkpoint),
                                       output_dir=str(tmp_path / "flow_real")),
                          logger=null_logger())
    result = trainer.train(resume="none")
    assert result["completed"]
    assert result["autoencoder"]["sha256"] == file_sha256(checkpoint)
    assert result["autoencoder"]["model_config"]["degree_head"] is True
    assert result["autoencoder"]["step"] == 2
    assert np.isfinite(result["best_val_loss"])

    # and the trained flow can be sampled through that same decoder
    from zinc_gnf.generation import NodeCountSampler, generate_molecules
    flow, _ = load_flow(tmp_path / "flow_real" / "best.pt", device="cpu")
    sampled = generate_molecules(flow, autoencoder, count=4,
                                 node_counts=NodeCountSampler([6, 7, 8]), seed=0, batch_size=4)
    assert len(sampled["records"]) == 4
    for record in sampled["records"]:
        assert {"raw", "constrained", "degree"} <= set(record)


# ---------------------------------------------------------------------------
# Numerical safety.
# ---------------------------------------------------------------------------
def test_runaway_log_determinant_is_warned_about_in_the_log(tmp_path):
    data = write_dataset(tmp_path / "data")
    output = tmp_path / "warned"
    trainer = FlowTrainer(FlowConfig(**FLOW),
                          train_config(tmp_path, data, output_dir=str(output),
                                       log_det_warn_threshold=0.0),
                          autoencoder=tiny_autoencoder())
    trainer.train(resume="none")
    log_text = (output / "train.log").read_text()
    assert "exceeds" in log_text and "log|det J|/node" in log_text
    assert "max_log_scale" in log_text              # the log says what to do about it


def test_a_sane_threshold_does_not_warn(tmp_path):
    data = write_dataset(tmp_path / "data")
    output = tmp_path / "quiet"
    trainer = FlowTrainer(FlowConfig(**FLOW),
                          train_config(tmp_path, data, output_dir=str(output)),
                          autoencoder=tiny_autoencoder())
    trainer.train(resume="none")
    assert trainer.config.log_det_warn_threshold == 50.0
    assert "exceeds" not in (output / "train.log").read_text()


# ---------------------------------------------------------------------------
# Pausing a run at an intermediate step, then resuming it exactly.
# ---------------------------------------------------------------------------
def test_pause_stops_the_run_without_completing_it(tmp_path):
    data = write_dataset(tmp_path / "data")
    trainer, result = run_trainer(tmp_path, data, max_steps=8, pause_at_step=4,
                                  eval_every=4, save_every=4)
    assert result["paused"] is True and result["paused_at_step"] == 4
    assert result["final_step"] == 4
    assert result["completed"] is False          # the seed must NOT look finished
    assert result["early_stopped"] is False
    assert result["interrupted_by"] is None
    assert result["max_steps"] == 8              # the target is unchanged by the pause
    summary = json.loads((Path(trainer.config.output_dir) / "summary.json").read_text())
    assert summary["paused"] and not summary["completed"] and not summary["early_stopped"]


def test_the_cluster_skip_rule_resumes_a_paused_seed(tmp_path):
    """The exact expression the SLURM script evaluates must classify a pause as resumable."""
    data = write_dataset(tmp_path / "data")
    trainer, _ = run_trainer(tmp_path, data, max_steps=8, pause_at_step=4,
                             eval_every=4, save_every=4)
    summary = json.loads((Path(trainer.config.output_dir) / "summary.json").read_text())
    state = ("done" if summary.get("completed") or summary.get("early_stopped")
             else ("paused" if summary.get("paused") else "unfinished"))
    assert state == "paused"                     # i.e. resumed, not skipped

    finished, _ = run_trainer(tmp_path / "finished", data, max_steps=4, eval_every=4, save_every=4)
    done = json.loads((Path(finished.config.output_dir) / "summary.json").read_text())
    assert (done.get("completed") or done.get("early_stopped")) and not done.get("paused")


def test_paused_checkpoint_carries_optimizer_rng_and_schedule_state(tmp_path):
    data = write_dataset(tmp_path / "data")
    trainer, _ = run_trainer(tmp_path, data, max_steps=8, pause_at_step=4,
                             eval_every=4, save_every=4)
    checkpoint = torch.load(Path(trainer.config.output_dir) / "last.pt", map_location="cpu",
                            weights_only=False)
    assert checkpoint["step"] == 4
    assert checkpoint["optimizer_state_dict"] is not None
    assert checkpoint["optimizer_state_dict"]["state"]                 # Adam moments exist
    assert checkpoint["torch_rng_state"] is not None
    assert checkpoint["numpy_rng_state"] is not None
    assert checkpoint["noise_generator_state"] is not None             # smoothing-noise stream
    assert checkpoint["batcher"]["position"] or checkpoint["batcher"]["epoch"]
    assert checkpoint["embedding_statistics"]["molecules"] == 20
    # the learning-rate schedule is a pure function of the step, so restoring the step
    # restores the schedule exactly; there is no separate scheduler object to lose
    assert trainer._learning_rate(checkpoint["step"]) == pytest.approx(
        trainer._learning_rate(4))


def test_resume_after_pause_is_bitwise_identical_to_an_uninterrupted_run(tmp_path):
    data = write_dataset(tmp_path / "data")
    autoencoder = tiny_autoencoder()
    straight, straight_result = run_trainer(tmp_path / "straight", data, autoencoder=autoencoder,
                                            max_steps=8, eval_every=4, save_every=4)
    paused, paused_result = run_trainer(tmp_path / "paused", data, autoencoder=autoencoder,
                                        max_steps=8, pause_at_step=4, eval_every=4, save_every=4)
    assert paused_result["paused"] and not straight_result["paused"]
    # resume with the pause flag STILL SET: it is behind us and must be ignored
    resumed = FlowTrainer(FlowConfig(**FLOW),
                          train_config(tmp_path / "paused", data, max_steps=8, pause_at_step=4,
                                       eval_every=4, save_every=4),
                          logger=null_logger(), autoencoder=autoencoder)
    result = resumed.train(resume="auto")
    assert result["completed"] and result["final_step"] == 8 and not result["paused"]

    one = torch.load(Path(straight.config.output_dir) / "last.pt", map_location="cpu",
                     weights_only=False)
    two = torch.load(Path(resumed.config.output_dir) / "last.pt", map_location="cpu",
                     weights_only=False)
    assert one["step"] == two["step"] == 8
    assert one["batcher"] == two["batcher"]
    for key, value in one["model_state_dict"].items():
        assert torch.equal(value, two["model_state_dict"][key]), key
    for key, state in one["optimizer_state_dict"]["state"].items():
        for name, value in state.items():
            if torch.is_tensor(value):
                assert torch.equal(value, two["optimizer_state_dict"]["state"][key][name]), (key, name)
    assert one["best_loss"] == pytest.approx(two["best_loss"])


def test_a_pause_point_already_passed_is_ignored(tmp_path):
    data = write_dataset(tmp_path / "data")
    autoencoder = tiny_autoencoder()
    run_trainer(tmp_path, data, autoencoder=autoencoder, max_steps=6, eval_every=3, save_every=3)
    # the run already reached step 6; asking to pause at 4 must not stop it again
    trainer = FlowTrainer(FlowConfig(**FLOW),
                          train_config(tmp_path, data, max_steps=10, pause_at_step=4,
                                       eval_every=3, save_every=3),
                          logger=null_logger(), autoencoder=autoencoder)
    result = trainer.train(resume="auto")
    assert result["final_step"] == 10 and result["completed"] and not result["paused"]


def test_the_paused_checkpoint_can_be_generated_from_before_resuming(tmp_path):
    from zinc_gnf.generation import NodeCountSampler, generate_molecules

    data = write_dataset(tmp_path / "data")
    autoencoder = tiny_autoencoder()
    trainer, _ = run_trainer(tmp_path, data, autoencoder=autoencoder, max_steps=8,
                             pause_at_step=4, eval_every=4, save_every=4)
    paused_checkpoint = Path(trainer.config.output_dir) / "last.pt"
    flow, probe = load_flow(paused_checkpoint, device="cpu")
    assert probe["step"] == 4 and bool(flow.statistics_set)
    sampled = generate_molecules(flow, autoencoder, count=4,
                                 node_counts=NodeCountSampler([6, 7]), seed=0, batch_size=4)
    assert len(sampled["records"]) == 4
    assert sampled["settings"]["latent_round_trip_relative_error"] < 1e-3
    # sampling must not disturb the checkpoint the resume will read
    assert torch.load(paused_checkpoint, map_location="cpu", weights_only=False)["step"] == 4


@pytest.mark.parametrize("bad", [
    {"pause_at_step": 0}, {"pause_at_step": -1}, {"pause_at_step": 11, "max_steps": 10},
])
def test_pause_configuration_is_validated(bad):
    with pytest.raises(ValueError, match="pause_at_step"):
        FlowTrainConfig(**bad).validate()


def test_pause_defaults_to_off():
    assert FlowTrainConfig().pause_at_step is None
    FlowTrainConfig(pause_at_step=10_000, max_steps=50_000).validate()
