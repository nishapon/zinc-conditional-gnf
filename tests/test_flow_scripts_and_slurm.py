"""The shipped flow configuration, the three flow scripts end to end on tiny
data, and static ISML-rule checks of the flow SLURM scripts.

No job is submitted and no GPU is used; the SLURM scripts are parsed and
`bash -n` checked only.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import subprocess

import pytest
import torch

from tests.test_autoencoder_training import write_dataset
from zinc_gnf.autoencoder import AutoencoderConfig, GraphAutoencoder
from zinc_gnf.constants_charge_aware import vocabulary_fingerprint
from zinc_gnf.flow import FlowConfig
from zinc_gnf.trainer_flow import FlowTrainConfig
from zinc_gnf.utils.config import load_config

REPOSITORY = Path(__file__).resolve().parents[1]
SLURM = REPOSITORY / "slurm"
CONFIG = "configs/zinc250k_flow_uncond.yaml"
GPU_JOBS = ["04_flow_smoke_gpu.sh", "04_flow_uncond_gpu.sh"]
CPU_JOBS = ["05_generate_uncond.sh"]
SELECTED_AUTOENCODER = "outputs/ablation/ae_degree/seed_3/last.pt"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""),
                                                  REPOSITORY / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def text(name: str) -> str:
    return (SLURM / name).read_text(encoding="utf-8")


def code_lines(source: str) -> str:
    """The executable part of a script: comments and echoed help text removed."""
    kept = [line for line in source.splitlines()
            if not line.lstrip().startswith("#") and "echo " not in line]
    return "\n".join(kept)


def directives(source: str) -> dict[str, str]:
    found = {}
    for line in source.splitlines():
        match = re.match(r"#SBATCH\s+--([\w-]+)(?:=(.*))?$", line.strip())
        if match:
            found[match.group(1)] = (match.group(2) or "").strip()
    return found


def write_autoencoder(path: Path, *, degree_head: bool = True, seed: int = 0) -> Path:
    """A tiny but real autoencoder checkpoint, in the format `load_autoencoder` expects."""
    torch.manual_seed(seed)
    model = GraphAutoencoder(AutoencoderConfig(
        node_embedding_dim=64, latent_dim=32, num_processing_steps=1, attn_kq_dim=8,
        attn_v_dim=8, attn_concat_heads_output_dim=8, degree_head=degree_head))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "model_config": model.config.to_dict(),
                "vocabulary_fingerprint": vocabulary_fingerprint(),
                "step": 6000}, path)
    return path


# ---------------------------------------------------------------------------
# The shipped configuration must actually parse.
# ---------------------------------------------------------------------------
def test_shipped_configuration_builds_both_dataclasses():
    configuration = load_config(REPOSITORY / CONFIG)
    flow = FlowConfig.from_dict(configuration["flow"])
    training = FlowTrainConfig.from_dict(configuration["training"])
    training.validate()
    assert flow.condition_features == ()           # this stage is unconditional
    assert flow.conditional is False and flow.condition_dim == 0
    assert flow.node_embedding_dim == 64 and flow.num_blocks == 12
    assert flow.noise_sigma == pytest.approx(0.05)
    assert flow.whiten and flow.actnorm and flow.invertible_linear
    assert flow.max_log_scale == pytest.approx(1.0)
    assert training.ae_checkpoint == SELECTED_AUTOENCODER
    assert training.ae_subset_seed == 3 and training.ae_seen_only is False
    assert training.save_every <= 1000             # resume granularity for walltime kills
    assert training.early_stopping_patience and training.val_molecules


def test_full_size_model_is_small_enough_for_an_eight_gigabyte_gpu():
    from zinc_gnf.flow import NodeFlow
    flow = NodeFlow(FlowConfig.from_dict(load_config(REPOSITORY / CONFIG)["flow"]))
    parameters = flow.num_parameters()
    assert 100_000 < parameters < 20_000_000, parameters
    # a full-width batch of 128 molecules of 38 atoms is the training shape
    assert parameters * 4 / 1e9 < 0.5              # weights, in GB


# ---------------------------------------------------------------------------
# Scripts end to end.
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny_setup(tmp_path):
    data = write_dataset(tmp_path / "data")
    autoencoder = write_autoencoder(tmp_path / "ae.pt")
    return data, autoencoder


def run_training(tmp_path, data, autoencoder, *, output, extra=()):
    script = load_script("04_train_flow.py")
    return script.main([
        "--config", str(REPOSITORY / CONFIG), "--data-dir", str(data),
        "--ae-checkpoint", str(autoencoder), "--output-dir", str(output),
        "--device", "cpu", "--seed", "1", "--batch-size", "4", "--max-steps", "4",
        "--num-blocks", "1", "--statistics-molecules", "20", "--val-molecules", "8",
        "--eval-every", "2", "--save-every", "2", "--log-every", "2", "--warmup-steps", "0",
        *extra])


def test_training_script_runs_and_is_resumable(tiny_setup, tmp_path):
    data, autoencoder = tiny_setup
    output = tmp_path / "run"
    assert run_training(tmp_path, data, autoencoder, output=output) == 0
    for name in ("config.json", "train.log", "metrics.jsonl", "summary.json", "best.pt", "last.pt"):
        assert (output / name).exists(), name
    configuration = json.loads((output / "config.json").read_text())
    assert configuration["autoencoder"]["sha256"]
    assert configuration["flow"]["num_blocks"] == 1
    assert configuration["training"]["ae_checkpoint"] == str(autoencoder)
    # resuming the same command continues instead of restarting
    assert run_training(tmp_path, data, autoencoder, output=output,
                        extra=("--max-steps", "6")) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["final_step"] == 6 and summary["completed"]


def test_training_script_exit_codes(tiny_setup, tmp_path):
    data, autoencoder = tiny_setup
    script = load_script("04_train_flow.py")
    assert script.main(["--config", "configs/does_not_exist.yaml"]) == 4
    missing = script.main([
        "--config", str(REPOSITORY / CONFIG), "--data-dir", str(tmp_path / "absent"),
        "--ae-checkpoint", str(autoencoder), "--output-dir", str(tmp_path / "r2"),
        "--device", "cpu", "--max-steps", "2", "--num-blocks", "1"])
    assert missing == 2                                  # dataset missing
    no_encoder = script.main([
        "--config", str(REPOSITORY / CONFIG), "--data-dir", str(data),
        "--ae-checkpoint", str(tmp_path / "nope.pt"), "--output-dir", str(tmp_path / "r3"),
        "--device", "cpu", "--max-steps", "2", "--num-blocks", "1"])
    assert no_encoder == 2                               # autoencoder missing


def test_generation_script_produces_metrics_for_every_decoder(tiny_setup, tmp_path):
    data, autoencoder = tiny_setup
    output = tmp_path / "run"
    assert run_training(tmp_path, data, autoencoder, output=output) == 0
    script = load_script("05_generate.py")
    samples = tmp_path / "samples"
    code = script.main(["--flow-checkpoint", str(output / "best.pt"),
                        "--ae-checkpoint", str(autoencoder), "--data-dir", str(data),
                        "--output-dir", str(samples), "--count", "6",
                        "--temperature", "0.85,1.0", "--batch-size", "3", "--seed", "0",
                        "--device", "cpu"])
    assert code == 0
    for temperature in ("T0.85", "T1"):
        directory = samples / temperature
        metrics = json.loads((directory / "metrics.json").read_text())
        assert metrics["samples"] == 6
        assert set(metrics["decoders"]) == {"raw", "constrained", "degree"}
        assert metrics["headline_decoder"] == "degree"
        assert metrics["autoencoder"]["sha256"]
        assert metrics["settings"]["temperature"] == pytest.approx(float(temperature[1:]))
        assert (directory / "metrics.txt").exists()
        assert (directory / "samples.jsonl").exists()
        assert len((directory / "samples.jsonl").read_text().splitlines()) == 6
        assert (directory / "valid_smiles.txt").exists()
        # novelty is measured against the DISTINCT SMILES of the training split only
        # (the tiny fixture repeats 16 molecules across its 20 training records)
        assert metrics["train_reference_molecules"] == 16
    assert json.loads((samples / "summary.json").read_text())["by_temperature"].keys() == {"T0.85", "T1"}


def test_generation_script_refuses_a_different_autoencoder(tiny_setup, tmp_path):
    data, autoencoder = tiny_setup
    output = tmp_path / "run"
    assert run_training(tmp_path, data, autoencoder, output=output) == 0
    other = write_autoencoder(tmp_path / "other.pt", seed=5)
    script = load_script("05_generate.py")
    arguments = ["--flow-checkpoint", str(output / "best.pt"), "--ae-checkpoint", str(other),
                 "--data-dir", str(data), "--output-dir", str(tmp_path / "s2"),
                 "--count", "2", "--device", "cpu"]
    assert script.main(arguments) == 3
    assert script.main(arguments + ["--allow-ae-mismatch"]) == 0     # explicit opt-out only


def test_generation_script_argument_validation(tiny_setup, tmp_path):
    data, autoencoder = tiny_setup
    script = load_script("05_generate.py")
    base = ["--flow-checkpoint", str(tmp_path / "none.pt"), "--output-dir", str(tmp_path / "s3"),
            "--data-dir", str(data)]
    assert script.main(base + ["--count", "0"]) == 4
    assert script.main(base + ["--temperature", ""]) == 4
    assert script.main(base + ["--decoders", "magic"]) == 4
    assert script.main(base) == 2                                   # checkpoint missing


def test_summary_script_aggregates_across_seeds(tmp_path):
    root = tmp_path / "samples"
    for seed, validity in ((1, 0.90), (2, 0.94)):
        directory = root / f"seed_{seed}" / "T0.85"
        directory.mkdir(parents=True)
        (directory / "metrics.json").write_text(json.dumps({
            "samples": 100, "headline_decoder": "degree",
            "decoders": {"degree": {
                "strict_validity": {"value": validity}, "uniqueness": {"value": 0.98},
                "novelty_vs_train": {"value": 0.99}, "nuv": validity * 0.98 * 0.99,
                "fallback_rate": {"value": 0.05}, "sanitizable": {"value": 0.97},
                "diversity": {"internal_diversity": 0.88}}}}))
    script = load_script("06_summarize_generation.py")
    assert script.main(["--root", str(root)]) == 0
    summary = json.loads((root / "generation_summary.json").read_text())
    entry = summary["temperatures"]["T0.85"]
    assert entry["runs"] == ["seed_1", "seed_2"]
    assert entry["metrics"]["strict_validity"]["mean"] == pytest.approx(0.92)
    assert entry["metrics"]["strict_validity"]["std"] == pytest.approx(0.02)
    assert entry["metrics"]["internal_diversity"]["mean"] == pytest.approx(0.88)
    assert "across flow seeds" in summary["uncertainty"]
    assert (root / "generation_summary.txt").exists()
    assert script.main(["--root", str(tmp_path / "empty")]) == 2


# ---------------------------------------------------------------------------
# Static SLURM checks.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", GPU_JOBS + CPU_JOBS)
def test_slurm_syntax_and_common_rules(name):
    source = text(name)
    assert subprocess.run(["bash", "-n", str(SLURM / name)]).returncode == 0
    assert "SLURM_JOB_ID" in source and "exit 64" in source
    assert not re.search(r"\bpip\b|\bconda (install|create)|\bwget\b|\bcurl\b|git (pull|clone)", source)
    # a job must never submit or cancel other jobs; mentions in comments/help text are fine
    assert not re.search(r"\bscancel\b|\bsbatch\b|\bssh\b", code_lines(source))
    assert SELECTED_AUTOENCODER in source                  # the frozen checkpoint, by name
    assert "--split test" not in source and "load_test_split" not in source
    # the autoencoder is read-only: nothing may delete, move or write to it
    assert not re.search(r"(rm|mv|cp|chmod|truncate)[^\n]*ae_degree", source)
    assert not re.search(r">\s*\"?\$\{?AE_CHECKPOINT", source)


@pytest.mark.parametrize("name", GPU_JOBS)
def test_gpu_jobs_request_one_gpu_on_the_student_partition(name):
    source, found = text(name), directives(text(name))
    assert found["partition"] == "STUD" and found["gres"] == "gpu:1"
    assert found["cpus-per-task"] == "4" and found["nodes"] == "1"
    assert re.match(r"\d{2}:\d{2}:\d{2}", found["time"])
    assert "nvidia-smi" in source and "torch.cuda.is_available()" in source
    assert "--device cuda" in source
    assert "--resume auto" in source                        # walltime kills must be recoverable


def test_training_job_is_resumable_and_skips_finished_seeds():
    source = text("04_flow_uncond_gpu.sh")
    assert "--resume auto" in source
    assert "summary.json" in source and "skipping" in source
    assert "resubmit" in source.lower()
    assert directives(source)["mem"] == "16G"
    hours = int(directives(source)["time"].split(":")[0])
    assert hours >= 4, "leave room for a slower-than-estimated run"
    assert 'SEEDS="${SEEDS:-1}"' in source                  # later: 5 seeds, same script


def test_smoke_job_checks_gpu_numerics_and_generates():
    source = text("04_flow_smoke_gpu.sh")
    assert "slogdet" in source and "inverse" in source      # invertibility + log-det on the GPU
    assert "scripts/05_generate.py" in source               # the pipeline is exercised end to end
    assert "--max-steps 300" in source
    assert "NOT a result" in source
    minutes = directives(source)["time"]
    assert minutes in ("00:30:00", "00:40:00", "00:45:00", "01:00:00")


@pytest.mark.parametrize("name", CPU_JOBS)
def test_generation_job_runs_on_cpu_without_a_gpu(name):
    source, found = text(name), directives(text(name))
    assert found["partition"] == "CPU" and "gres" not in found
    assert "--device cpu" in source and "--device cuda" not in source
    assert "--headline degree" in source
    assert "scripts/06_summarize_generation.py" in source
    assert "93.4% exact" in source or "93.4%" in source     # the honest caveat is stated


# ---------------------------------------------------------------------------
# The smoke job must stay a smoke job.
# ---------------------------------------------------------------------------
def test_smoke_job_cannot_turn_into_a_full_training_run():
    source = text("04_flow_smoke_gpu.sh")
    code = code_lines(source)
    # the budget is literal, never read from the environment
    assert "--max-steps 300" in code and "--max-train-molecules 20000" in code
    assert not re.search(r"--max-steps\s+\"?\$", code)
    assert not re.search(r"MAX_STEPS|SEEDS", code)
    # it writes to its own directories, so it can never overwrite a real run
    assert 'OUT="outputs/checkpoints/flow_smoke"' in code
    assert 'SAMPLES="outputs/samples/flow_smoke"' in code
    assert "outputs/checkpoints/flow_uncond" not in code      # only its own directories
    assert "outputs/samples/flow_uncond" not in code
    # and its generation call is small
    assert "--count 200" in code
    assert directives(source)["time"] < "01:00:00"


def test_full_training_job_and_smoke_job_do_not_share_output_directories():
    smoke = code_lines(text("04_flow_smoke_gpu.sh"))
    full = code_lines(text("04_flow_uncond_gpu.sh"))
    assert "outputs/checkpoints/flow_smoke" in smoke
    assert "outputs/checkpoints/flow_uncond" in full
    assert "flow_smoke" not in full
    assert "outputs/checkpoints/flow_uncond" not in smoke


def test_every_flow_job_uses_the_same_dataset_directory_and_config():
    for name in GPU_JOBS + CPU_JOBS:
        code = code_lines(text(name))
        assert 'DATA_DIR="data/processed/zinc250k_charge_aware_moflow_fixed"' in code, name
    for name in GPU_JOBS:
        assert "configs/zinc250k_flow_uncond.yaml" in code_lines(text(name)), name


def test_jobs_reference_only_scripts_that_exist():
    for name in GPU_JOBS + CPU_JOBS:
        for script in re.findall(r"scripts/([\w.]+\.py)", text(name)):
            assert (REPOSITORY / "scripts" / script).exists(), (name, script)


def test_gitignore_covers_every_artifact_these_jobs_write():
    patterns = (REPOSITORY / ".gitignore").read_text()
    for needed in ("outputs/checkpoints/*", "outputs/samples/*", "outputs/logs/*",
                   "*.pt", "*.pkl", "data/processed/", "__pycache__/", ".venv/"):
        assert needed in patterns, needed


def test_generation_refuses_the_degree_decoder_without_a_milp_solver(tiny_setup, tmp_path, monkeypatch):
    """No solver must be a loud failure, not a silent fall back to greedy decoding."""
    data, autoencoder = tiny_setup
    output = tmp_path / "run"
    assert run_training(tmp_path, data, autoencoder, output=output) == 0
    script = load_script("05_generate.py")
    real_import = __import__

    def blocked(name, *args, **kwargs):
        if name == "scipy.optimize" or name.startswith("scipy.optimize"):
            raise ImportError("no scipy for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked)
    code = script.main(["--flow-checkpoint", str(output / "best.pt"),
                        "--ae-checkpoint", str(autoencoder), "--data-dir", str(data),
                        "--output-dir", str(tmp_path / "s_nosolver"), "--count", "2",
                        "--device", "cpu"])
    monkeypatch.undo()
    assert code == 4


def test_cpu_generation_job_checks_its_dependencies():
    source = text("05_generate_uncond.sh")
    assert "scipy.optimize" in source and "milp" in source
    assert "fall back to greedy" in source


# ---------------------------------------------------------------------------
# The pause mechanism, through the script and the SLURM jobs.
# ---------------------------------------------------------------------------
def test_training_script_pauses_and_then_resumes_to_the_target(tiny_setup, tmp_path):
    data, autoencoder = tiny_setup
    output = tmp_path / "paused_run"
    # --max-steps stays the target; --pause-at stops short of it
    assert run_training(tmp_path, data, autoencoder, output=output,
                        extra=("--max-steps", "8", "--pause-at", "4")) == 0   # a pause exits 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["paused"] and summary["paused_at_step"] == 4
    assert summary["completed"] is False and summary["max_steps"] == 8
    configuration = json.loads((output / "config.json").read_text())
    assert configuration["training"]["pause_at_step"] == 4
    assert configuration["training"]["max_steps"] == 8

    # generation from the paused checkpoint, into its own directory
    script = load_script("05_generate.py")
    intermediate = tmp_path / "samples_pause"
    assert script.main(["--flow-checkpoint", str(output / "last.pt"),
                        "--ae-checkpoint", str(autoencoder), "--data-dir", str(data),
                        "--output-dir", str(intermediate), "--count", "4",
                        "--temperature", "0.85", "--batch-size", "4", "--device", "cpu"]) == 0
    assert (intermediate / "T0.85" / "metrics.json").exists()

    # resuming with the pause flag still set reaches the target
    assert run_training(tmp_path, data, autoencoder, output=output,
                        extra=("--max-steps", "8", "--pause-at", "4")) == 0
    finished = json.loads((output / "summary.json").read_text())
    assert finished["completed"] and finished["final_step"] == 8 and not finished["paused"]


def test_training_script_rejects_an_impossible_pause(tiny_setup, tmp_path):
    data, autoencoder = tiny_setup
    assert run_training(tmp_path, data, autoencoder, output=tmp_path / "bad",
                        extra=("--max-steps", "4", "--pause-at", "9")) == 4


def test_training_job_wires_the_pause_without_changing_the_target():
    source = text("04_flow_uncond_gpu.sh")
    code = code_lines(source)
    assert 'PAUSE_AT="${PAUSE_AT:-}"' in code                  # off unless asked for
    assert "--pause-at" in code and "PAUSE_ARGS" in code
    assert 'MAX_STEPS="${MAX_STEPS:-50000}"' in code            # target still 50,000
    assert "--max-steps \"${MAX_STEPS}\"" in code
    # a paused seed is resumed, never skipped
    assert "'paused' if d.get('paused')" in code
    assert 'if [[ "${STATE}" == "paused" ]]' in code
    assert "resuming from" in source
    assert "PAUSE_AT=10000" in source                           # the documented example


def test_generation_job_can_write_intermediate_results_elsewhere():
    source = text("05_generate_uncond.sh")
    code = code_lines(source)
    assert 'SAMPLES="${SAMPLES_ROOT:-outputs/samples/flow_uncond}"' in code
    assert 'CHECKPOINT_NAME="${CHECKPOINT_NAME:-best.pt}"' in code     # final default unchanged
    assert "INTERMEDIATE check" in source                              # labelled as not final
    assert "flow_uncond_pause" in source                               # the documented example


def test_pause_configuration_knob_is_documented_in_the_shipped_config():
    configuration = load_config(REPOSITORY / CONFIG)
    assert "pause_at_step" in configuration["training"]
    assert configuration["training"]["pause_at_step"] is None          # off by default
    training = FlowTrainConfig.from_dict(configuration["training"])
    training.validate()
    assert training.max_steps == 50_000
