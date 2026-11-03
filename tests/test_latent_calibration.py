"""Tests for post-hoc latent recalibration of the flow's prior.

The claims under test are the ones the method rests on:
  * installing the prior's own (mu, sigma) as the calibration is an exact no-op,
    for log_prob AND for sampling;
  * calibrating with the measured latent statistics cannot lower the likelihood of
    the measurement data (maximum likelihood over a family containing the old prior);
  * sampling really uses the calibrated mean/std, so generation and scoring agree;
  * the calibration survives a checkpoint round-trip, and checkpoints written before
    calibration existed still load;
  * the measurement reproduces hand-computed statistics and never touches the test split.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from zinc_gnf.flow import FlowConfig, NodeFlow, NodePrior, node_mask_from_counts
from zinc_gnf.latent_calibration import (
    CALIBRATION_NOISE_SEED,
    calibrate_prior,
    dataset_negative_log_likelihood,
    format_calibration,
    latent_statistics,
)
from zinc_gnf.utils.checkpointing import load_checkpoint, save_checkpoint

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def small_flow(seed: int = 0, dimension: int = 8) -> NodeFlow:
    config = FlowConfig(node_embedding_dim=dimension, num_blocks=2, hidden_dim=16,
                        attention_dim=16, num_heads=2, noise_sigma=0.05)
    flow = NodeFlow(config, init_seed=seed)
    generator = torch.Generator().manual_seed(seed + 1)
    mean = torch.randn(dimension, generator=generator)
    std = torch.rand(dimension, generator=generator) + 0.5
    flow.set_data_statistics(mean, std)
    flow.eval()
    return flow


class StubBatch:
    """Minimal stand-in for MolecularBatch: the calibration code only needs these."""

    def __init__(self, embeddings: torch.Tensor, node_mask: torch.Tensor) -> None:
        self.embeddings, self.node_mask = embeddings, node_mask
        self.batch_size = int(embeddings.shape[0])

    def to(self, device):  # noqa: ANN001
        return self


class StubEncoder:
    def encode(self, batch: StubBatch) -> torch.Tensor:
        return batch.embeddings


@pytest.fixture()
def stub_data(monkeypatch):
    """Patches `iter_eval_batches` as used by the calibration module only."""
    torch.manual_seed(7)
    counts = [5, 7, 4, 7, 6, 5]
    mask = node_mask_from_counts(counts)
    embeddings = torch.randn(len(counts), mask.shape[1], 8) * 1.3 + 0.4
    embeddings = embeddings * mask.unsqueeze(-1)
    batches = [StubBatch(embeddings[:3], mask[:3]), StubBatch(embeddings[3:], mask[3:])]

    def fake_iter(dataset, batch_size, indices=None):  # noqa: ANN001
        assert dataset == "train-set", "calibration must read the dataset it was given"
        return iter(batches)

    monkeypatch.setattr("zinc_gnf.latent_calibration.iter_eval_batches", fake_iter)
    return "train-set", embeddings, mask


# ---------------------------------------------------------------- no-op ----
def test_setting_the_priors_own_parameters_is_a_noop_for_log_prob():
    flow = small_flow()
    mask = node_mask_from_counts([6, 4])
    z = torch.randn(2, mask.shape[1], 8)
    before = flow.prior.log_prob(z, mask, None)
    mu, sigma = flow.prior.parameters_for(1, None, "cpu")
    flow.prior.set_calibration(mu[0], sigma[0])
    after = flow.prior.log_prob(z, mask, None)
    assert torch.allclose(before, after, atol=1e-6)


def test_setting_the_priors_own_parameters_is_a_noop_for_sampling():
    flow = small_flow()
    mask = node_mask_from_counts([6, 4])
    draw = lambda: flow.prior.sample(  # noqa: E731
        mask, None, generator=torch.Generator().manual_seed(11))
    before = draw()
    mu, sigma = flow.prior.parameters_for(1, None, "cpu")
    flow.prior.set_calibration(mu[0], sigma[0])
    assert torch.allclose(before, draw(), atol=1e-6)


def test_clear_calibration_restores_the_trained_prior():
    flow = small_flow()
    mu, sigma = flow.prior.parameters_for(1, None, "cpu")
    flow.prior.set_calibration(torch.full((8,), 3.0), torch.full((8,), 2.0))
    assert flow.prior.is_calibrated
    flow.prior.clear_calibration()
    assert not flow.prior.is_calibrated
    again = flow.prior.parameters_for(1, None, "cpu")
    assert torch.allclose(mu, again[0]) and torch.allclose(sigma, again[1])


def test_trained_parameters_ignore_calibration_without_changing_state():
    flow = small_flow()
    expected = flow.prior.parameters_for(1, None, "cpu")
    flow.prior.set_calibration(torch.full((8,), 3.0), torch.full((8,), 2.0))
    trained = flow.prior.trained_parameters_for(1, None, "cpu")
    assert torch.allclose(trained[0], expected[0]) and torch.allclose(trained[1], expected[1])
    assert flow.prior.is_calibrated, "querying the trained prior must not clear the calibration"
    assert float(flow.prior.parameters_for(1, None, "cpu")[1][0, 0]) == pytest.approx(2.0)


# ------------------------------------------------------- the likelihood ----
def test_calibration_does_not_lower_the_measured_likelihood(stub_data):
    dataset, _, _ = stub_data
    flow = small_flow()
    encoder = StubEncoder()
    before = dataset_negative_log_likelihood(flow, encoder, dataset, device="cpu")
    statistics = latent_statistics(flow, encoder, dataset, device="cpu")
    calibrate_prior(flow, statistics)
    after = dataset_negative_log_likelihood(flow, encoder, dataset, device="cpu")
    assert after["nll_per_node"] <= before["nll_per_node"] + 1e-6
    assert after["nodes"] == before["nodes"] == statistics["nodes"]


def test_calibration_is_a_noop_for_an_already_calibrated_flow(stub_data):
    dataset, _, _ = stub_data
    flow = small_flow()
    encoder = StubEncoder()
    calibrate_prior(flow, latent_statistics(flow, encoder, dataset, device="cpu"))
    first = dataset_negative_log_likelihood(flow, encoder, dataset, device="cpu")
    # Re-measuring the same latents must reproduce the same prior: the fixed point.
    second_statistics = latent_statistics(flow, encoder, dataset, device="cpu")
    calibrate_prior(flow, second_statistics)
    second = dataset_negative_log_likelihood(flow, encoder, dataset, device="cpu")
    assert second["nll_per_node"] == pytest.approx(first["nll_per_node"], abs=1e-6)


def test_log_det_is_untouched_by_calibration(stub_data):
    """Calibration changes the prior only; the flow map must be bit-identical."""
    dataset, embeddings, mask = stub_data
    flow = small_flow()
    _, log_det_before = flow(embeddings, mask)
    calibrate_prior(flow, latent_statistics(flow, StubEncoder(), dataset, device="cpu"))
    _, log_det_after = flow(embeddings, mask)
    assert torch.equal(log_det_before, log_det_after)


# ---------------------------------------------------------- measurement ----
def test_latent_statistics_match_a_direct_computation(stub_data):
    dataset, embeddings, mask = stub_data
    flow = small_flow()
    statistics = latent_statistics(flow, StubEncoder(), dataset, device="cpu")

    generator = torch.Generator().manual_seed(CALIBRATION_NOISE_SEED)
    latents = []
    for start, stop in ((0, 3), (3, 6)):
        noisy = flow.add_smoothing_noise(embeddings[start:stop], mask[start:stop],
                                         generator=generator)
        z, _ = flow(noisy, mask[start:stop])
        latents.append(z[mask[start:stop]])
    pooled = torch.cat(latents).to(torch.float64)
    assert statistics["nodes"] == pooled.shape[0]
    assert torch.allclose(statistics["mean"], pooled.mean(0).to(torch.float32), atol=1e-5)
    assert torch.allclose(statistics["std"],
                          pooled.std(0, unbiased=False).to(torch.float32), atol=1e-5)


def test_zscore_ratio_reports_a_known_mismatch():
    """A prior twice as wide as the latents must report zscore_rms ~ 0.5."""
    prior = NodePrior(4, init_sigma=2.0)
    z = torch.randn(10000, 4, generator=torch.Generator().manual_seed(3))
    mu, sigma = prior.parameters_for(1, None, "cpu")
    ratio = float((((z - mu[0]) / sigma[0]) ** 2).mean().sqrt().detach())
    assert ratio == pytest.approx(0.5, abs=0.02)


def test_padding_nodes_are_excluded_from_the_measurement(stub_data):
    dataset, embeddings, mask = stub_data
    flow = small_flow()
    statistics = latent_statistics(flow, StubEncoder(), dataset, device="cpu")
    assert statistics["nodes"] == int(mask.sum())
    assert statistics["molecules"] == int(mask.shape[0])


def test_format_calibration_mentions_the_ratio_and_its_limits(stub_data):
    dataset, _, _ = stub_data
    statistics = latent_statistics(small_flow(), StubEncoder(), dataset, device="cpu")
    text = format_calibration(statistics)
    assert "zscore_rms" in text and "kurtosis" in text
    assert "inter-node dependence" in text


# ----------------------------------------------------------- sampling ------
def test_sampling_follows_the_calibrated_mean_and_std():
    flow = small_flow()
    mean = torch.linspace(-1.0, 1.0, 8)
    std = torch.linspace(0.5, 2.0, 8)
    flow.prior.set_calibration(mean, std)
    mask = node_mask_from_counts([10] * 4000)
    z = flow.prior.sample(mask, None, generator=torch.Generator().manual_seed(5))
    flat = z.reshape(-1, 8)
    assert torch.allclose(flat.mean(0), mean, atol=0.05)
    assert torch.allclose(flat.std(0), std, atol=0.05)


def test_temperature_still_scales_the_calibrated_std():
    flow = small_flow()
    flow.prior.set_calibration(torch.zeros(8), torch.ones(8))
    mask = node_mask_from_counts([10] * 4000)
    z = flow.prior.sample(mask, None, temperature=0.5,
                          generator=torch.Generator().manual_seed(5))
    assert float(z.reshape(-1, 8).std()) == pytest.approx(0.5, abs=0.03)


# ------------------------------------------------------------ guards -------
def test_calibration_is_refused_for_a_conditional_prior():
    prior = NodePrior(8, condition_dim=4)
    with pytest.raises(NotImplementedError):
        prior.set_calibration(torch.zeros(8), torch.ones(8))


@pytest.mark.parametrize("mean, std", [
    (torch.zeros(7), torch.ones(8)),
    (torch.zeros(8), torch.ones(7)),
    (torch.full((8,), float("nan")), torch.ones(8)),
    (torch.zeros(8), torch.full((8,), -1.0)),
])
def test_bad_calibration_statistics_are_rejected(mean, std):
    prior = NodePrior(8)
    with pytest.raises(ValueError):
        prior.set_calibration(mean, std)


def test_tiny_std_is_floored_and_counted():
    prior = NodePrior(8)
    std = torch.ones(8)
    std[0] = 0.0
    floored = prior.set_calibration(torch.zeros(8), std, std_floor=1e-3)
    assert floored == 1
    assert float(prior.calibration_std[0]) == pytest.approx(1e-3)


# ----------------------------------------------------- checkpoint I/O ------
def test_calibration_survives_a_checkpoint_round_trip(tmp_path):
    flow = small_flow()
    mean, std = torch.linspace(-1, 1, 8), torch.linspace(0.5, 2.0, 8)
    flow.prior.set_calibration(mean, std)
    path = tmp_path / "calibrated.pt"
    save_checkpoint(path, flow, None, epoch=0, best_metric=1.0, configuration={})
    restored = small_flow()
    load_checkpoint(path, restored)
    assert restored.prior.is_calibrated
    assert torch.allclose(restored.prior.calibration_mean, mean)
    assert torch.allclose(restored.prior.calibration_std, std)


def test_checkpoints_written_before_calibration_existed_still_load(tmp_path):
    """Strict loading must not break on the three new buffers."""
    flow = small_flow()
    state = {k: v for k, v in flow.state_dict().items() if "calibration" not in k
             and not k.endswith("prior.calibrated")}
    assert not any("calibrat" in key for key in state)
    path = tmp_path / "old.pt"
    torch.save({"epoch": 0, "best_metric": 1.0, "configuration": {},
                "model_state_dict": state, "optimizer_state_dict": None}, path)
    restored = small_flow()
    restored.prior.set_calibration(torch.full((8,), 9.0), torch.full((8,), 9.0))
    load_checkpoint(path, restored)
    assert not restored.prior.is_calibrated, "a legacy checkpoint means 'not calibrated'"
    assert torch.allclose(restored.prior.calibration_std, torch.ones(8))


# ------------------------------------------------------------- script ------
def test_calibration_script_never_loads_the_test_split():
    source = (REPOSITORY_ROOT / "scripts" / "10_calibrate_flow.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                for alias in node.names}
    assert "load_training_splits" in imported
    assert not any("test" in name for name in imported)
    assert "test_records" not in source and "load_test_split" not in source


def test_calibration_script_refuses_to_overwrite_its_input(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "calibrate_cli", REPOSITORY_ROOT / "scripts" / "10_calibrate_flow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"not really a checkpoint")
    assert module.main(["--flow-checkpoint", str(checkpoint),
                        "--output-checkpoint", str(checkpoint),
                        "--output-dir", str(tmp_path / "out")]) == 4


def test_training_indices_reproduce_the_checkpoints_restriction(tmp_path):
    import importlib.util

    from zinc_gnf.trainer_flow import autoencoder_seen_indices

    spec = importlib.util.spec_from_file_location(
        "calibrate_cli2", REPOSITORY_ROOT / "scripts" / "10_calibrate_flow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    probe = {"train_config": {"ae_seen_only": True, "ae_subset_seed": 11, "ae_subset_size": 20}}
    assert module.training_indices(100, probe) == autoencoder_seen_indices(100, 11, 20)
    assert module.training_indices(10, {"train_config": {}}) == list(range(10))
    assert module.training_indices(10, {"train_config": {"max_train_molecules": 3}}) == [0, 1, 2]


# ------------------------------------------------- the script end to end ---
def test_calibration_script_end_to_end_and_then_generation(tmp_path):
    """Train 4 steps, calibrate, sample from the calibrated checkpoint.

    This is the exact chain the SLURM jobs run, on tiny data: it proves the new
    checkpoint is loadable by `05_generate.py` and that the calibrated prior is what
    generation draws from.
    """
    import json

    from tests.test_flow_scripts_and_slurm import (
        load_script,
        run_training,
        write_autoencoder,
    )
    from tests.test_autoencoder_training import write_dataset

    data = write_dataset(tmp_path / "data")
    autoencoder = write_autoencoder(tmp_path / "ae.pt")
    run = tmp_path / "run"
    assert run_training(tmp_path, data, autoencoder, output=run) == 0

    calibrate = load_script("10_calibrate_flow.py")
    arguments = ["--flow-checkpoint", str(run / "last.pt"), "--data-dir", str(data),
                 "--ae-checkpoint", str(autoencoder), "--molecules", "16",
                 "--val-molecules", "8", "--batch-size", "4", "--device", "cpu"]
    assert calibrate.main(arguments) == 0

    produced = run / "last_calibrated.pt"
    assert produced.exists(), "the calibrated checkpoint must be written next to the input"
    payload = json.loads((run / "calibration.json").read_text())
    assert payload["calibration"]["calibrated"] is True
    assert payload["likelihood"]["train_gain_nats_per_node"] >= -1e-6
    assert payload["statistics"]["zscore_rms"] > 0
    assert payload["output_checkpoint_sha256"]
    assert (run / "calibration.txt").exists()

    from zinc_gnf.trainer_flow import load_flow

    original, _ = load_flow(run / "last.pt")
    calibrated, probe = load_flow(produced)
    assert not original.prior.is_calibrated
    assert calibrated.prior.is_calibrated
    assert probe["latent_calibration"]["measurement"]["molecules"] == 16
    assert probe["step"] == 4, "the calibrated checkpoint keeps the training metadata"
    assert torch.equal(original.blocks[0].act_norm.bias, calibrated.blocks[0].act_norm.bias)

    # a second run must refuse to clobber, and accept --force
    assert calibrate.main(arguments) == 4
    assert calibrate.main(arguments + ["--force"]) == 0
    # --measure-only writes no checkpoint
    measure_dir = tmp_path / "measure"
    assert calibrate.main(arguments + ["--measure-only", "--output-checkpoint",
                                       str(tmp_path / "unwritten.pt"),
                                       "--output-dir", str(measure_dir)]) == 0
    assert not (tmp_path / "unwritten.pt").exists()
    assert json.loads((measure_dir / "calibration.json").read_text())["measure_only"] is True

    generate = load_script("05_generate.py")
    samples = tmp_path / "samples"
    assert generate.main(["--flow-checkpoint", str(produced), "--ae-checkpoint", str(autoencoder),
                          "--data-dir", str(data), "--output-dir", str(samples),
                          "--count", "4", "--temperature", "0.2,1.0", "--batch-size", "2",
                          "--device", "cpu"]) == 0
    assert json.loads((samples / "T1" / "metrics.json").read_text())["samples"] == 4


# -------------------------------------------------------------- SLURM ------
def slurm_source() -> str:
    return (REPOSITORY_ROOT / "slurm" / "10_calibrate_flow.sh").read_text(encoding="utf-8")


def test_calibration_slurm_job_follows_the_cluster_rules():
    import re
    import subprocess

    source = slurm_source()
    directives = {}
    for line in source.splitlines():
        match = re.match(r"#SBATCH\s+--([\w-]+)(?:=(.*))?$", line.strip())
        if match:
            directives[match.group(1)] = (match.group(2) or "").strip()
    assert directives["partition"] == "CPU", "calibration is a CPU job; do not take a GPU"
    assert "gres" not in directives
    assert directives["output"].startswith("outputs/logs/")
    assert directives["error"].startswith("outputs/logs/")
    assert directives["time"] and directives["mem"]
    assert 'if [[ -z "${SLURM_JOB_ID:-}" ]]' in source, "must refuse to run on the master node"
    assert "scripts/10_calibrate_flow.py" in source
    assert "srun " in source
    assert subprocess.run(["bash", "-n", str(REPOSITORY_ROOT / "slurm" / "10_calibrate_flow.sh")],
                          capture_output=True).returncode == 0


def test_calibration_slurm_job_never_writes_over_a_training_checkpoint():
    source = slurm_source()
    executable = "\n".join(line for line in source.splitlines()
                           if not line.lstrip().startswith("#") and "echo " not in line)
    assert "--output-checkpoint" in executable       # only ever an explicit, separate path
    assert "rm " not in executable and "mv " not in executable
    assert "--force" in executable and 'FORCE:-0' in executable, "overwriting must be opt-in"
    assert "--device cpu" in executable
