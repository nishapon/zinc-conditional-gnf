"""Tests for autoencoder training: selection, checkpointing, exact resume, the
test-split guard and the command-line scripts. CPU only, tiny synthetic data."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import pickle

import numpy as np
import pytest
import torch

from tests.test_autoencoder_model import SMILES, make_records
from zinc_gnf.autoencoder import AutoencoderConfig, GraphAutoencoder, evaluate_reconstruction
from zinc_gnf.constants_charge_aware import (
    ATOM_VOCAB,
    REPRESENTATION_VERSION_CHARGE_AWARE,
    vocabulary_fingerprint,
)
from zinc_gnf.dataset import (
    DatasetFormatError,
    MolecularGraphDataset,
    VocabularyMismatchError,
    load_test_split,
    load_training_splits,
)
from zinc_gnf.trainer_autoencoder import (
    AutoencoderTrainer,
    ResumeMismatchError,
    TrainConfig,
    load_autoencoder,
    resolve_device,
    selection_key_value,
)
from zinc_gnf.utils.logging import JobLogger
from zinc_gnf.utils.runtime import StopFlag

TINY = dict(node_embedding_dim=16, latent_dim=32, num_processing_steps=2,
            attn_kq_dim=8, attn_v_dim=8, attn_concat_heads_output_dim=8)


def write_dataset(path: Path, *, representation=REPRESENTATION_VERSION_CHARGE_AWARE,
                  fingerprint=None) -> Path:
    records = make_records(SMILES * 2)                       # 32 molecules, repeated on purpose
    for i, r in enumerate(records):
        r["row_id"] = i
    payload = {
        "representation": representation,
        "vocabulary_fingerprint": fingerprint or vocabulary_fingerprint(),
        "atom_vocab": list(ATOM_VOCAB), "split_protocol": "moflow_fixed", "seed": 42,
        "csv_sha256": "abc", "condition_scaler": {"fit_on": "train_only"},
        "train": records[:20], "val": records[20:28], "test": records[28:],
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "splits.pkl").write_bytes(pickle.dumps(payload))
    return path


def train_config(tmp_path, **overrides) -> TrainConfig:
    values = dict(data_dir=str(tmp_path / "data"), output_dir=str(tmp_path / "run"), seed=3,
                  device="cpu", batch_size=4, max_steps=6, log_every=2, eval_every=3, save_every=3,
                  val_loss_molecules=8, val_recon_molecules=6, grad_clip_norm=5.0)
    values.update(overrides)
    return TrainConfig(**values)


@pytest.fixture
def data(tmp_path):
    return write_dataset(tmp_path / "data")


def quiet_logger(tmp_path, name="run"):
    return JobLogger("t", tmp_path / name, filename="train.log")


def run(tmp_path, *, name="run", model=None, resume="none", **overrides):
    cfg = train_config(tmp_path, output_dir=str(tmp_path / name), **overrides)
    trainer = AutoencoderTrainer(model or AutoencoderConfig(**TINY), cfg, quiet_logger(tmp_path, name))
    return trainer, trainer.train(resume=resume)


def state_of(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


# ===========================================================================
# Data loading: the test split stays out of training.
# ===========================================================================
def test_training_loader_never_returns_the_test_split(data):
    train, val, fingerprint = load_training_splits(data)
    assert len(train) == 20 and len(val) == 8
    assert fingerprint["counts"] == {"train": 20, "val": 8, "test": 4}
    assert fingerprint["split_protocol"] == "moflow_fixed"
    test, test_fingerprint = load_test_split(data)             # explicit, separately named API
    assert len(test) == 4 and test_fingerprint["counts_digest"] == fingerprint["counts_digest"]


def test_trainer_never_loads_test_records(tmp_path, data, monkeypatch):
    loaded = []
    import zinc_gnf.trainer_autoencoder as module
    original = module.load_training_splits

    def spy(directory):
        result = original(directory)
        loaded.append(result)
        return result

    monkeypatch.setattr(module, "load_training_splits", spy)
    monkeypatch.setattr(module, "load_training_splits", spy)
    forbidden = []
    monkeypatch.setattr("zinc_gnf.dataset.load_test_split", lambda *a, **k: forbidden.append(1))
    trainer, result = run(tmp_path)
    assert result["completed"] and loaded and not forbidden
    assert not hasattr(trainer, "test_set")
    train_rows = {int(i) for i in trainer.train_set.row_id}
    val_rows = {int(i) for i in trainer.val_set.row_id}
    assert train_rows.isdisjoint(val_rows) and max(train_rows | val_rows) < 28     # rows 28-31 are test


def test_loader_rejects_wrong_representation_and_vocabulary(tmp_path):
    bad = write_dataset(tmp_path / "bad", representation="zinc250k_neutral_kekule_v1")
    with pytest.raises(DatasetFormatError, match="representation"):
        load_training_splits(bad)
    drifted = write_dataset(tmp_path / "drift", fingerprint="0" * 16)
    with pytest.raises(VocabularyMismatchError):
        load_training_splits(drifted)
    with pytest.raises(FileNotFoundError):
        load_training_splits(tmp_path / "nowhere")


# ===========================================================================
# Training, selection and checkpoints.
# ===========================================================================
def test_training_runs_logs_and_writes_checkpoints(tmp_path, data):
    trainer, result = run(tmp_path)
    out = tmp_path / "run"
    assert result["completed"] and result["final_step"] == 6 and not result["interrupted_by"]
    for name in ("config.json", "metrics.jsonl", "train.log", "last.pt", "best.pt", "summary.json"):
        assert (out / name).exists(), name
    assert not list(out.glob("*.tmp"))
    lines = [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines()]
    kinds = {l["kind"] for l in lines}
    assert kinds == {"train", "val"}
    val = [l for l in lines if l["kind"] == "val"]
    assert [v["step"] for v in val] == [3, 6]
    for key in ("val_mean_loss", "val_exact_molecule_constrained", "val_validity_constrained",
                "val_validity_raw", "val_exact_molecule_raw", "val_equivariant_ceiling"):
        assert key in val[0]
    config = json.loads((out / "config.json").read_text())
    assert config["data"]["split_protocol"] == "moflow_fixed"
    assert config["data"]["used"] == {"train": 20, "val": 8}
    assert "test split NOT loaded" in (out / "train.log").read_text()


def test_best_checkpoint_follows_validation_not_training(tmp_path, data):
    trainer, result = run(tmp_path, max_steps=9, eval_every=3, selection_key="val_loss")
    metrics = [json.loads(l) for l in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    val = [m for m in metrics if m["kind"] == "val"]
    best = min(val, key=lambda m: m["val_mean_loss"])
    assert result["best_step"] == best["step"]
    checkpoint = state_of(tmp_path / "run" / "best.pt")
    assert checkpoint["step"] == best["step"]
    assert checkpoint["val_metrics"]["val_mean_loss"] == pytest.approx(best["val_mean_loss"])
    assert state_of(tmp_path / "run" / "last.pt")["step"] == 9


def test_selection_key_ordering():
    better = {"val_mean_loss": 1.0, "val_exact_molecule_constrained": 0.5}
    worse = {"val_mean_loss": 0.5, "val_exact_molecule_constrained": 0.4}
    assert selection_key_value(better, "exact_molecule_constrained") > selection_key_value(worse, "exact_molecule_constrained")
    assert selection_key_value(worse, "val_loss") > selection_key_value(better, "val_loss")
    tie_a = {"val_mean_loss": 1.0, "val_exact_molecule_constrained": 0.5}
    tie_b = {"val_mean_loss": 2.0, "val_exact_molecule_constrained": 0.5}
    assert selection_key_value(tie_a, "exact_molecule_constrained") > selection_key_value(tie_b, "exact_molecule_constrained")
    nan = {"val_mean_loss": 1.0, "val_exact_molecule_constrained": float("nan")}
    assert selection_key_value(nan, "exact_molecule_constrained")[0] == -1.0


def test_training_is_deterministic_for_a_fixed_seed(tmp_path, data):
    run(tmp_path, name="a")
    run(tmp_path, name="b")
    a, b = state_of(tmp_path / "a" / "last.pt"), state_of(tmp_path / "b" / "last.pt")
    for key in a["model_state_dict"]:
        assert torch.equal(a["model_state_dict"][key], b["model_state_dict"][key]), key
    run(tmp_path, name="c", seed=4)
    c = state_of(tmp_path / "c" / "last.pt")
    assert any(not torch.equal(a["model_state_dict"][k], c["model_state_dict"][k]) for k in a["model_state_dict"])


def test_resume_is_exact(tmp_path, data):
    """3 steps + resume + 3 steps must equal 6 uninterrupted steps, bit for bit."""
    run(tmp_path, name="straight", max_steps=6)
    run(tmp_path, name="split", max_steps=3)
    _, second = run(tmp_path, name="split", max_steps=6, resume="auto")
    assert second["completed"] and second["final_step"] == 6
    straight, split = state_of(tmp_path / "straight" / "last.pt"), state_of(tmp_path / "split" / "last.pt")
    assert split["step"] == straight["step"] == 6
    for key in straight["model_state_dict"]:
        assert torch.equal(straight["model_state_dict"][key], split["model_state_dict"][key]), key
    assert split["batcher"] == straight["batcher"]
    assert "RESUMED" in (tmp_path / "split" / "train.log").read_text()


def test_resume_restores_best_bookkeeping(tmp_path, data):
    run(tmp_path, name="r", max_steps=6, selection_key="val_loss")
    before = state_of(tmp_path / "r" / "last.pt")
    trainer, result = run(tmp_path, name="r", max_steps=9, resume="auto", selection_key="val_loss")
    after = state_of(tmp_path / "r" / "last.pt")
    assert after["step"] == 9
    assert after["best_step"] >= 0 and result["best_step"] == after["best_step"]
    assert after["best_key"] >= before["best_key"]                  # best can only improve on resume
    assert len(after["history"]) == len(before["history"]) + 1


def test_resume_mismatches_are_refused(tmp_path, data):
    run(tmp_path, name="m", max_steps=3)
    with pytest.raises(ResumeMismatchError, match="seed"):
        run(tmp_path, name="m", max_steps=6, resume="auto", seed=99)
    with pytest.raises(ResumeMismatchError, match="batch_size"):
        run(tmp_path, name="m", max_steps=6, resume="auto", batch_size=2)
    other = AutoencoderConfig(**{**TINY, "node_embedding_dim": 24})
    with pytest.raises(ResumeMismatchError, match="autoencoder configuration"):
        run(tmp_path, name="m", max_steps=6, resume="auto", model=other)
    with pytest.raises(ResumeMismatchError, match="training data"):
        run(tmp_path, name="m", max_steps=6, resume="auto", max_train_molecules=12)
    with pytest.raises(FileNotFoundError):
        run(tmp_path, name="m2", resume=str(tmp_path / "nope.pt"))


def test_auto_resume_with_no_checkpoint_starts_fresh(tmp_path, data):
    _, result = run(tmp_path, name="fresh", resume="auto")
    assert result["completed"]
    assert "RESUMED" not in (tmp_path / "fresh" / "train.log").read_text()


def test_interruption_saves_a_resumable_checkpoint(tmp_path, data):
    flag = StopFlag()
    flag.signal_name = "SIGTERM"
    cfg = train_config(tmp_path, output_dir=str(tmp_path / "int"), max_steps=6)
    trainer = AutoencoderTrainer(AutoencoderConfig(**TINY), cfg, quiet_logger(tmp_path, "int"))
    result = trainer.train(resume="none", stop_flag=flag)
    assert result["interrupted_by"] == "SIGTERM" and not result["completed"] and result["final_step"] == 1
    assert state_of(tmp_path / "int" / "last.pt")["step"] == 1
    _, finished = run(tmp_path, name="int", max_steps=6, resume="auto")
    assert finished["completed"] and finished["final_step"] == 6


def test_early_stopping_counts_evaluations_without_improvement(tmp_path, data):
    _, result = run(tmp_path, name="es", max_steps=40, eval_every=2, early_stopping_patience=2,
                    learning_rate=0.0)                   # lr 0 -> no improvement after the first eval
    assert result["early_stopped"] and result["final_step"] < 40
    assert result["best_step"] == 2


def test_validation_subsets_are_fixed_across_training_seeds(tmp_path, data):
    a, _ = run(tmp_path, name="s1", seed=1, max_steps=3)
    b, _ = run(tmp_path, name="s2", seed=2, max_steps=3)
    assert a.loss_indices == b.loss_indices and a.recon_indices == b.recon_indices


def test_nonfinite_loss_stops_training(tmp_path, data):
    cfg = train_config(tmp_path, output_dir=str(tmp_path / "nan"), learning_rate=float("inf"), max_steps=5)
    trainer = AutoencoderTrainer(AutoencoderConfig(**TINY), cfg, quiet_logger(tmp_path, "nan"))
    with pytest.raises((FloatingPointError, ValueError, RuntimeError)):
        trainer.train(resume="none")


def test_train_config_validation():
    with pytest.raises(ValueError, match="selection_key"):
        TrainConfig(selection_key="bogus").validate()
    with pytest.raises(ValueError, match="device"):
        TrainConfig(device="tpu").validate()
    with pytest.raises(ValueError, match="batch_size"):
        TrainConfig(batch_size=0).validate()
    with pytest.raises(ValueError, match="unknown training settings"):
        TrainConfig.from_dict({"nope": 1})
    assert TrainConfig.from_dict(TrainConfig().to_dict()) == TrainConfig()


def test_cuda_request_fails_loudly_without_a_gpu():
    assert resolve_device("cpu").type == "cpu"
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="cuda"):
            resolve_device("cuda")
        assert resolve_device("auto").type == "cpu"


def test_load_autoencoder_roundtrips_and_checks_vocabulary(tmp_path, data):
    run(tmp_path, name="ck", max_steps=3)
    model, checkpoint = load_autoencoder(tmp_path / "ck" / "best.pt")
    assert not model.training and checkpoint["step"] in (3,)
    reference = state_of(tmp_path / "ck" / "best.pt")["model_state_dict"]
    for key, value in model.state_dict().items():
        assert torch.equal(value, reference[key])
    tampered = state_of(tmp_path / "ck" / "best.pt")
    tampered["vocabulary_fingerprint"] = "0" * 16
    torch.save(tampered, tmp_path / "bad.pt")
    with pytest.raises(ResumeMismatchError, match="vocabulary"):
        load_autoencoder(tmp_path / "bad.pt")


def test_evaluation_of_a_loaded_checkpoint_matches_in_training_numbers(tmp_path, data):
    trainer, result = run(tmp_path, name="ev", max_steps=3, val_loss_molecules=8, val_recon_molecules=8)
    model, _ = load_autoencoder(tmp_path / "ev" / "best.pt")
    dataset = MolecularGraphDataset(load_training_splits(data)[1])
    fresh = evaluate_reconstruction(model, dataset, indices=trainer.recon_indices, batch_size=3)
    logged = result["best_validation"]
    assert fresh["constrained"]["exact_molecule_match"] == pytest.approx(logged["val_exact_molecule_constrained"])
    assert fresh["raw"]["validity"] == pytest.approx(logged["val_validity_raw"])


# ===========================================================================
# Command-line scripts.
# ===========================================================================
def load_script(repository_root: Path, name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), repository_root / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_cli_end_to_end(tmp_path, data, repository_root):
    script = load_script(repository_root, "02_train_autoencoder.py")
    out = tmp_path / "cli"
    args = ["--config", str(repository_root / "configs" / "zinc250k_charge_aware_ae.yaml"),
            "--data-dir", str(data), "--output-dir", str(out), "--device", "cpu", "--batch-size", "4",
            "--max-steps", "4", "--eval-every", "2", "--save-every", "2", "--log-every", "2",
            "--val-loss-molecules", "8", "--val-recon-molecules", "4",
            "--num-processing-steps", "2", "--resume", "none"]
    assert script.main(args) == 0
    assert (out / "best.pt").exists() and (out / "summary.json").exists()
    config = json.loads((out / "config.json").read_text())
    assert config["autoencoder"]["num_atom_types"] == 17
    assert config["autoencoder"]["symmetry_noise_dim"] == 0
    assert config["autoencoder"]["num_processing_steps"] == 2          # override applied
    assert config["autoencoder"]["bond_class_weight"] == 5.0           # yaml default kept
    assert script.main(args[:-2] + ["--resume", "auto"]) == 0           # already finished: resumes, no extra steps
    assert script.main(args[:2] + ["--data-dir", str(tmp_path / "none"), "--output-dir", str(tmp_path / "x"),
                                   "--device", "cpu"]) == 2
    assert script.main(["--config", str(tmp_path / "missing.yaml")]) == 4
    assert script.main(args + ["--batch-size", "0"]) == 4              # later flag wins
    assert script.main(args[:-2] + ["--resume", "auto", "--seed", "9", "--max-steps", "8"]) == 3


def test_train_cli_cuda_fails_loudly_on_cpu_only_machines(tmp_path, data, repository_root):
    if torch.cuda.is_available():
        pytest.skip("a GPU is visible")
    script = load_script(repository_root, "02_train_autoencoder.py")
    code = script.main(["--data-dir", str(data), "--output-dir", str(tmp_path / "gpu"), "--device", "cuda",
                        "--max-steps", "2", "--resume", "none"])
    assert code == 4
    assert not (tmp_path / "gpu" / "last.pt").exists()


def test_evaluate_cli_and_test_split_guard(tmp_path, data, repository_root):
    run(tmp_path, name="e", max_steps=3)
    script = load_script(repository_root, "02b_evaluate_autoencoder.py")
    report = tmp_path / "report.json"
    common = ["--checkpoint", str(tmp_path / "e" / "best.pt"), "--data-dir", str(data), "--device", "cpu"]
    assert script.main(common + ["--split", "val", "--output", str(report)]) == 0
    parsed = json.loads(report.read_text())
    assert parsed["split"] == "val" and parsed["molecules_evaluated"] == 8
    assert set(parsed["metrics"]) >= {"raw", "constrained", "equivariant_exact_ceiling", "atom_accuracy"}
    # The held-out test split needs an explicit opt-in.
    assert script.main(common + ["--split", "test"]) == 4
    assert script.main(common + ["--split", "test", "--allow-test", "--output", str(tmp_path / "t.json")]) == 0
    assert json.loads((tmp_path / "t.json").read_text())["split"] == "test"
    assert script.main(["--checkpoint", str(tmp_path / "none.pt"), "--data-dir", str(data), "--device", "cpu"]) == 2
