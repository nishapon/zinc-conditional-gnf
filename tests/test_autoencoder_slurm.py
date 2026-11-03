"""Static ISML-rule checks for the derivation (CPU) and autoencoder smoke (GPU)
SLURM scripts. No job is submitted; scripts are parsed and `bash -n` checked."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
SLURM = REPOSITORY / "slurm"
CPU_JOB = "01d_derive_moflow_split.sh"
GPU_JOB = "02_ae_smoke_gpu.sh"


def text(name: str) -> str:
    return (SLURM / name).read_text(encoding="utf-8")


def directives(source: str) -> dict[str, str]:
    found = {}
    for line in source.splitlines():
        match = re.match(r"#SBATCH\s+--([\w-]+)(?:=(.*))?$", line.strip())
        if match:
            found[match.group(1)] = (match.group(2) or "").strip()
    return found


def code(source: str) -> str:
    """Executable shell lines: no comments, no `echo` text, no heredoc bodies
    (the Python preflight blocks print hints that mention install commands but
    never run them)."""
    lines, delimiter = [], None
    for line in source.splitlines():
        stripped = line.strip()
        if delimiter is not None:
            if stripped == delimiter:
                delimiter = None
            continue
        match = re.search(r"<<-?\s*'?\"?(\w+)'?\"?\s*$", stripped)
        if match:
            delimiter = match.group(1)
            lines.append(stripped)
            continue
        if stripped and not stripped.startswith(("#", "echo")):
            lines.append(stripped)
    return "\n".join(lines)


def seconds(limit: str) -> int:
    h, m, s = (int(v) for v in limit.split(":"))
    return h * 3600 + m * 60 + s


@pytest.mark.parametrize("name", [CPU_JOB, GPU_JOB])
def test_syntax_logging_and_failure_handling(name):
    source = text(name)
    assert source.startswith("#!/bin/bash")
    assert subprocess.run(["bash", "-n", str(SLURM / name)], capture_output=True).returncode == 0
    d, body = directives(source), code(source)
    assert d["output"].startswith("outputs/logs/") and "%j" in d["output"]
    assert d["error"].startswith("outputs/logs/") and "%j" in d["error"]
    assert (REPOSITORY / "outputs" / "logs" / ".gitkeep").exists()
    assert "set -euo pipefail" in body and "trap" in body and "ERR" in body
    assert "PYTHONUNBUFFERED=1" in body
    assert "zinc_gnf/bin/python" in source and "SLURM_SUBMIT_DIR" in source
    assert re.search(r'srun\s+"\$\{PYTHON\}"\s+-u\s+scripts/', body)
    assert 0 < seconds(d["time"]) <= 24 * 3600
    assert "mkdir -p outputs/logs" in body


@pytest.mark.parametrize("name", [CPU_JOB, GPU_JOB])
def test_no_network_installs_or_submission_from_inside_jobs(name):
    forbidden = (r"\b(wget|curl|git (clone|pull|fetch)|pip3? install|conda (install|create)|"
                 r"apt|yum|dnf|sbatch)\b")
    executable = code(text(name))
    assert not re.search(forbidden, executable)
    # Download / install instructions may only appear as printed hints.
    assert "wget" not in executable and "pip3 install" not in executable


@pytest.mark.parametrize("name", [CPU_JOB, GPU_JOB])
def test_refuse_to_run_outside_slurm(name):
    environment = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    completed = subprocess.run(["bash", str(SLURM / name)], capture_output=True, text=True,
                               env=environment, cwd=str(REPOSITORY), timeout=60)
    assert completed.returncode == 64
    assert "sbatch" in completed.stderr


def test_derivation_job_is_cpu_only_and_minimal():
    d, body = directives(text(CPU_JOB)), code(text(CPU_JOB))
    assert d["partition"] == "CPU"
    assert not any(k in d for k in ("gres", "gpus", "gpus-per-node"))
    assert not re.search(r"--gres|--gpus|STUD", body)
    assert d["nodes"] == "1" and d["ntasks"] == "1" and int(d["cpus-per-task"]) == 1
    memory = int(d["mem"].rstrip("G"))
    assert 4 <= memory <= 8                           # measured peak 3.1 GB
    assert seconds(d["time"]) <= 3600                 # measured ~20 s
    source = text(CPU_JOB)
    assert "scripts/01d_derive_moflow_split.py" in source
    assert "data/external/moflow/valid_idx_zinc.json" in source
    assert "login node" in source                      # tells where to fetch MoFlow's files


def test_derivation_job_does_not_touch_other_datasets():
    body = code(text(CPU_JOB))
    for forbidden in ("rm ", "mv ", "01b_preprocess", "01_preprocess.py"):
        assert forbidden not in body
    assert "--output-dir" not in body or "zinc250k_charge_aware_moflow_fixed" in text(CPU_JOB)


def test_gpu_job_requests_one_gpu_on_the_student_partition():
    d, source = directives(text(GPU_JOB)), text(GPU_JOB)
    assert d["partition"] == "STUD"                    # cluster tutorial: STUD + gres gpu:1
    assert d["gres"] == "gpu:1"
    assert d["nodes"] == "1" and d["ntasks"] == "1"
    assert 1 <= int(d["cpus-per-task"]) <= 8
    assert int(d["mem"].rstrip("G")) <= 16
    assert seconds(d["time"]) <= 2 * 3600              # a smoke test, not a research run
    assert "#SBATCH --partition=CPU" not in source


def test_gpu_job_fails_fast_without_a_gpu_and_never_trains_on_cpu():
    source, body = text(GPU_JOB), code(text(GPU_JOB))
    assert "nvidia-smi" in body and "torch.cuda.is_available()" in source
    assert re.search(r"--device\s+cuda", body)
    assert "--device cpu" not in body and "--device auto" not in body
    assert "GPU build of PyTorch" in source or "whl/cu" in source


def test_gpu_job_uses_validation_only_and_the_primary_dataset():
    source, body = text(GPU_JOB), code(text(GPU_JOB))
    assert "zinc250k_charge_aware_moflow_fixed" in source
    assert "zinc250k_charge_aware/" not in body                      # not the random-split dataset
    assert "--split val" in body
    assert "--allow-test" not in source and "--split test" not in source
    assert "--max-steps" in body and "--max-train-molecules" in body        # bounded smoke test
    assert "--resume auto" in body                                          # resubmission resumes
    assert "outputs/checkpoints/ae_smoke_gpu" in source


def test_gpu_job_has_a_data_preflight():
    source = text(GPU_JOB)
    assert "splits.pkl not found" in source and "01d_derive_moflow_split.sh" in source


def test_readme_documents_protocol_and_run_order():
    readme = (SLURM / "README_ae.md").read_text()
    for needle in ("MoFlow", "1-based", "01d_derive_moflow_split.sh", "02_ae_smoke_gpu.sh",
                   "afterok", "equivariant_exact_ceiling", "not** autoencoder variability",
                   "never loaded by training"):
        assert needle in readme, needle
