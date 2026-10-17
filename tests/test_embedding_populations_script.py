"""The shuffled-node control script end to end, and static checks of its CPU job."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import subprocess

import pytest
import torch

from tests.test_autoencoder_training import write_dataset
from tests.test_flow_scripts_and_slurm import write_autoencoder
from zinc_gnf.constants_charge_aware import vocabulary_fingerprint
from zinc_gnf.flow import FlowConfig, NodeFlow
from zinc_gnf.trainer_flow import file_sha256

REPOSITORY = Path(__file__).resolve().parents[1]
SLURM_JOB = "08_embedding_populations.sh"
SCRIPT = "08_embedding_populations.py"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""),
                                                  REPOSITORY / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def text(name: str) -> str:
    return (REPOSITORY / "slurm" / name).read_text(encoding="utf-8")


def directives(source: str) -> dict[str, str]:
    found = {}
    for line in source.splitlines():
        match = re.match(r"#SBATCH\s+--([\w-]+)(?:=(.*))?$", line.strip())
        if match:
            found[match.group(1)] = (match.group(2) or "").strip()
    return found


def code_lines(source: str) -> str:
    return "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith("#") and "echo " not in line)


def write_flow(path: Path, autoencoder_path: Path, *, dimension: int = 64) -> Path:
    """A flow checkpoint in the format `load_flow` expects, tied to an autoencoder."""
    flow = NodeFlow(FlowConfig(node_embedding_dim=dimension, num_blocks=1, attention_dim=8,
                               num_heads=2, hidden_dim=16), init_seed=0)
    flow.set_data_statistics(torch.zeros(dimension), torch.ones(dimension))
    torch.save({"model_state_dict": flow.state_dict(), "flow_config": flow.config.to_dict(),
                "vocabulary_fingerprint": vocabulary_fingerprint(), "step": 300,
                "autoencoder": {"checkpoint": str(autoencoder_path),
                                "sha256": file_sha256(autoencoder_path)}}, path)
    return path


@pytest.fixture
def setup(tmp_path):
    data = write_dataset(tmp_path / "data")
    autoencoder = write_autoencoder(tmp_path / "ae.pt")
    flow = write_flow(tmp_path / "flow.pt", autoencoder)
    return data, autoencoder, flow


def run(tmp_path, data, autoencoder, *extra, output="pop"):
    script = load_script(SCRIPT)
    return script.main(["--ae-checkpoint", str(autoencoder), "--data-dir", str(data),
                        "--output-dir", str(tmp_path / output), "--molecules", "8",
                        "--reference-molecules", "20", "--batch-size", "4",
                        "--seed", "0", "--device", "cpu", *extra])


def test_script_compares_real_shuffled_and_flow(setup, tmp_path):
    data, autoencoder, flow = setup
    assert run(tmp_path, data, autoencoder, "--flow-checkpoint", str(flow),
               "--temperature", "0.85,1.0") == 0
    result = json.loads((tmp_path / "pop" / "embedding_populations.json").read_text())
    names = [row["name"] for row in result["embedding_statistics"]]
    assert names == ["real", "shuffled", "flow_T0.85", "flow_T1"]

    # all populations share the same molecules and sizes
    assert {row["molecules"] for row in result["embedding_statistics"]} == {8}
    assert {row["nodes"] for row in result["embedding_statistics"]} == \
        {result["embedding_statistics"][0]["nodes"]}

    # shuffling preserves every per-node statistic exactly
    real, shuffled = result["embedding_statistics"][0], result["embedding_statistics"][1]
    assert shuffled["mean_norm"] == pytest.approx(real["mean_norm"], rel=1e-9)
    assert shuffled["participation_ratio"] == pytest.approx(real["participation_ratio"], rel=1e-6)
    assert shuffled["within_over_between_distance"] == pytest.approx(1.0, abs=0.4)

    # pre-assembly predictions and assembled molecules exist for every population
    assert [row["name"] for row in result["decoder_predictions"]] == names
    assert set(result["profiles"]["profiles"]) == set(names)
    for name in names:
        profile = result["profiles"]["profiles"][name]
        assert "qed_at_least_0p7" in profile and "aromatic_ring_fraction" in profile
        assert "ring_bond_fractions" in profile          # connectivity vs bond orders
        assert 0.0 <= profile["degree_fallback_rate"] <= 1.0
        assert profile["decoder"] == "degree"

    # provenance and protocol
    assert result["autoencoder"]["sha256"] and result["flow"]["step"] == 300
    assert result["settings"]["reference_basis_split"] == "train"
    assert result["settings"]["population_split"] == "validation"
    assert result["shuffle"]["permuted_nodes"] == real["nodes"]
    assert "participation_ratio" in result["reference_spectrum"]

    report = (tmp_path / "pop" / "embedding_populations.txt").read_text()
    for fragment in ("within / between", "near-identical pair", "bond argmax class",
                     "aromatic / all rings", "degree-decoder fallback",
                     "WHAT THIS SHOWS AND WHAT IT DOES NOT", "SUFFICIENT"):
        assert fragment in report, fragment


def test_the_control_runs_without_any_flow(setup, tmp_path):
    data, autoencoder, _ = setup
    assert run(tmp_path, data, autoencoder) == 0
    result = json.loads((tmp_path / "pop" / "embedding_populations.json").read_text())
    assert [row["name"] for row in result["embedding_statistics"]] == ["real", "shuffled"]
    assert result["flow"] == {}


def test_a_foreign_autoencoder_is_refused_unless_allowed(setup, tmp_path):
    data, autoencoder, flow = setup
    other = write_autoencoder(tmp_path / "other.pt", seed=7)
    arguments = ("--flow-checkpoint", str(flow), "--temperature", "0.85")
    assert run(tmp_path, data, other, *arguments, output="p2") == 3
    assert run(tmp_path, data, other, *arguments, "--allow-ae-mismatch", output="p3") == 0


def test_argument_and_path_validation(setup, tmp_path):
    data, autoencoder, _ = setup
    script = load_script(SCRIPT)
    base = ["--ae-checkpoint", str(autoencoder), "--data-dir", str(data), "--device", "cpu"]
    assert script.main(base + ["--output-dir", str(tmp_path / "v1"), "--molecules", "1"]) == 4
    assert script.main(base + ["--output-dir", str(tmp_path / "v2"),
                               "--flow-checkpoint", str(tmp_path / "ghost.pt")]) == 2
    assert script.main(["--ae-checkpoint", str(tmp_path / "ghost_ae.pt"), "--data-dir", str(data),
                        "--output-dir", str(tmp_path / "v3"), "--device", "cpu"]) == 2
    assert script.main(base + ["--output-dir", str(tmp_path / "v4"),
                               "--data-dir", str(tmp_path / "nodata")]) == 2


def test_the_script_never_reads_the_test_split():
    source = (REPOSITORY / "scripts" / SCRIPT).read_text()
    assert "load_test_split" not in source and "load_training_splits" in source


# ---------------------------------------------------------------------------
# Static checks of the CPU job.
# ---------------------------------------------------------------------------
def test_job_is_cpu_only_and_valid():
    source = text(SLURM_JOB)
    found = directives(source)
    assert subprocess.run(["bash", "-n", str(REPOSITORY / "slurm" / SLURM_JOB)]).returncode == 0
    assert found["partition"] == "CPU" and "gres" not in found
    assert found["cpus-per-task"] == "4" and found["mem"] == "16G"
    assert re.match(r"\d{2}:\d{2}:\d{2}", found["time"])
    assert "--device cpu" in source and "--device cuda" not in source
    assert "nvidia-smi" not in source
    assert "SLURM_JOB_ID" in source and "exit 64" in source


def test_job_is_read_only_and_degrades_without_a_flow():
    source = text(SLURM_JOB)
    code = code_lines(source)
    assert "outputs/ablation/ae_degree/seed_3/last.pt" in source
    assert "outputs/checkpoints/flow_smoke/best.pt" in source
    assert not re.search(r"(rm|mv|cp|chmod|truncate)[^\n]*(ae_degree|flow_smoke|splits\.pkl)", code)
    assert not re.search(r"\bscancel\b|\bsbatch\b|\bssh\b", code)
    assert not re.search(r"\bpip\b|\bconda (install|create)|\bwget\b|\bcurl\b|git (pull|clone)", code)
    assert 'OUT="outputs/metrics/embedding_populations"' in code
    assert "outputs/metrics/latent_diagnostics" not in code      # does not touch job 332840
    assert 'if [[ -f "${FLOW_CHECKPOINT}" ]]' in code and "without flow populations" in source
    assert "scipy.optimize" in source and "milp" in source


def test_job_states_what_the_control_can_and_cannot_prove():
    source = text(SLURM_JOB)
    assert "SUFFICIENT" in source
    assert "not proof" in source
    assert "not trained enough" in source and "cannot be" in source
    assert "aromatic" in source and "rings" in source


def test_job_exposes_every_sample_size_as_an_override():
    code = code_lines(text(SLURM_JOB))
    for name in ("MOLECULES", "REFERENCE_MOLECULES", "TEMPERATURES", "FLOW_CHECKPOINT"):
        assert f'{name}="${{{name}:-' in code, name
