"""Static ISML-rule checks for the charge-aware SLURM scripts, and guards that
the neutral-only baseline is untouched.

No job is submitted; scripts are parsed and `bash -n` checked. Behaviour on
the cluster itself can only be established by running there.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
SLURM = REPOSITORY / "slurm"
JOB = "01b_preprocess_charge_aware.sh"
PREP = "01b_prepare_charge_aware.sh"
PINNED_SHA = "35e3f1a52b1badc0697e373d73a18ad773f415936ff992f4c6baa2e067b3e6ae"


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
    return "\n".join(
        line.strip() for line in source.splitlines()
        if line.strip() and not line.strip().startswith(("#", "echo"))
    )


@pytest.mark.parametrize("name", [JOB, PREP])
def test_bash_syntax(name):
    assert (SLURM / name).read_text().startswith("#!/bin/bash")
    completed = subprocess.run(["bash", "-n", str(SLURM / name)],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_job_is_cpu_only_and_minimal():
    d = directives(text(JOB))
    assert d["partition"] == "CPU"
    assert "gres" not in d and "gpus" not in d and "gpus-per-node" not in d
    assert not re.search(r"--gres|--gpus|STUD", code(text(JOB)))
    assert d["nodes"] == "1" and d["ntasks"] == "1" and int(d["cpus-per-task"]) == 1
    assert re.fullmatch(r"\d+G", d["mem"]) and int(d["mem"][:-1]) <= 8
    hours, minutes, seconds = (int(v) for v in d["time"].split(":"))
    assert 0 < hours * 3600 + minutes * 60 + seconds <= 24 * 3600


def test_job_logs_to_precreated_directory_and_prepare_makes_all_outputs():
    d = directives(text(JOB))
    assert d["output"].startswith("outputs/logs/") and "%j" in d["output"]
    assert d["error"].startswith("outputs/logs/") and "%j" in d["error"]
    assert (REPOSITORY / "outputs" / "logs" / ".gitkeep").exists()
    prep = text(PREP)
    for directory in ("outputs/logs", "data/processed/zinc250k_charge_aware",
                      "outputs/metrics/preprocessing_charge_aware"):
        assert directory in prep
    assert "mkdir -p" in code(prep)


def test_job_uses_conda_env_srun_unbuffered_and_traps_failures():
    source = text(JOB)
    body = code(source)
    assert "set -euo pipefail" in body and "trap" in body and "ERR" in body
    assert "zinc_gnf/bin/python" in source
    assert "PYTHONUNBUFFERED=1" in body
    assert re.search(r'srun\s+"\$\{PYTHON\}"\s+-u\s+scripts/01b_preprocess_charge_aware\.py', body)
    assert "data/raw/zinc250k.csv" in source
    assert "SLURM_JOB_ID" in source
    assert "configs/zinc250k_charge_aware.yaml" in source


@pytest.mark.parametrize("name", [JOB, PREP])
def test_no_network_or_installs_executed(name):
    forbidden = (r"\b(wget|curl|git (clone|pull|fetch)|pip3? install|"
                 r"conda (install|create)|apt|yum|dnf)\b")
    assert not re.search(forbidden, code(text(name)))


def test_wget_instructions_are_login_node_only_text():
    for name in (JOB, PREP):
        source = text(name)
        for line in source.splitlines():
            if "wget" in line:
                stripped = line.strip()
                assert stripped.startswith(("echo", "#")), line


def test_job_never_references_baseline_output_for_writing():
    body = code(text(JOB))
    assert "data/processed/zinc250k_charge_aware" in text(JOB)
    # No command in the job touches the baseline directory.
    assert not re.search(r"data/processed/zinc250k(?!_charge_aware)", body)
    assert "outputs/metrics/preprocessing_charge_aware" in text(JOB)
    assert not re.search(r"outputs/metrics/preprocessing(?!_charge_aware)", body)


def test_job_refuses_to_run_outside_slurm():
    environment = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    completed = subprocess.run(
        ["bash", str(SLURM / JOB)], capture_output=True, text=True,
        env=environment, cwd=str(REPOSITORY), timeout=60,
    )
    assert completed.returncode == 64
    assert "sbatch" in completed.stderr and "master node" in completed.stderr


def test_prepare_script_creates_directories_and_guides_downloads(tmp_path):
    repo = tmp_path / "repo"
    (repo / "slurm").mkdir(parents=True)
    shutil.copy(SLURM / PREP, repo / "slurm")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)

    run = lambda: subprocess.run(  # noqa: E731
        ["bash", str(repo / "slurm" / PREP)], capture_output=True, text=True,
        cwd=str(tmp_path),
    )
    missing = run()
    assert missing.returncode == 1 and "login node" in missing.stderr
    for directory in ("outputs/logs", "data/processed/zinc250k_charge_aware",
                      "outputs/metrics/preprocessing_charge_aware",
                      "data/external/moflow"):
        assert (repo / directory).is_dir()
    assert not (repo / "data" / "processed" / "zinc250k").exists()      # baseline untouched

    (repo / "data" / "raw" / "zinc250k.csv").write_text("smiles,logP,qed,SAS\n")
    ready = run()
    assert ready.returncode == 0, ready.stderr
    assert "sbatch slurm/01b_preprocess_charge_aware.sh" in ready.stdout
    assert "MoFlow files: not present" in ready.stdout


# ---------------------------------------------------------------------------
# The neutral-only baseline is a clearly identified, unmodified reference.
# ---------------------------------------------------------------------------
def git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=REPOSITORY, capture_output=True,
                          text=True).stdout


@pytest.mark.skipif(not (REPOSITORY / ".git").exists(), reason="needs a git checkout")
@pytest.mark.parametrize("path", [
    "zinc_gnf/constants.py", "zinc_gnf/chemistry.py", "zinc_gnf/preprocessing.py",
    "scripts/01_preprocess.py", "slurm/01_preprocess.sh", "configs/zinc250k.yaml",
])
def test_baseline_files_unchanged_since_phase0_commit(path):
    """Baseline sources must be identical to their content at the Phase 0 commit."""
    baseline_commit = "8be3f03a52a9d34b3f06529360593318ab0cba0f"
    if not git("cat-file", "-t", baseline_commit).strip() == "commit":
        pytest.skip("Phase 0 commit not available in this checkout")
    committed = subprocess.run(["git", "show", f"{baseline_commit}:{path}"],
                               cwd=REPOSITORY, capture_output=True)
    assert committed.returncode == 0
    assert (REPOSITORY / path).read_bytes() == committed.stdout, (
        f"{path} changed: the neutral-only baseline must not be modified"
    )


def test_baseline_configuration_still_neutral_only():
    from zinc_gnf.utils.config import load_config

    config = load_config(REPOSITORY / "configs" / "zinc250k.yaml")
    assert config["data"]["neutral_only"] is True
    assert config["data"]["processed_dir"] == "data/processed/zinc250k"
    assert len(config["data"]["atom_vocab"]) == 9
