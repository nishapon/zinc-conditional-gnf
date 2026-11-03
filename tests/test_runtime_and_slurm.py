"""Tests for runtime helpers and static compliance of the Phase 0 SLURM
scripts with the ISML cluster rules.

These tests never submit a job and never need SLURM: they parse the scripts
and run `bash -n`. Whether a script passes ON the cluster can only be
established by running it there.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess

import numpy as np
import pytest

from zinc_gnf.utils.runtime import (
    StopFlag,
    assert_output_outside,
    atomic_write_json,
    collect_environment,
    format_duration,
    json_safe,
    read_cgroup_limits,
    usable_cpu_count,
)

SLURM_DIR = Path(__file__).resolve().parents[1] / "slurm"
JOB_SCRIPTS = ["00_profile_preprocessing.sh", "00_chemistry_diagnostics.sh"]
ALL_SCRIPTS = JOB_SCRIPTS + ["00_prepare_phase0.sh"]
PINNED_SHA = "35e3f1a52b1badc0697e373d73a18ad773f415936ff992f4c6baa2e067b3e6ae"


# ---------------------------------------------------------------------------
# Runtime helpers.
# ---------------------------------------------------------------------------
def test_json_safe_makes_strict_json():
    payload = {
        "nan": float("nan"), "inf": float("inf"), "np_int": np.int64(3),
        "np_float": np.float32(1.5), "np_nan": np.float64("nan"),
        "array": np.array([1, 2]), "set": {"b", "a"}, "tuple": (1, 2),
        3: "int key", "path": Path("x"), "flag": np.bool_(True),
    }
    safe = json_safe(payload)
    text = json.dumps(safe, allow_nan=False)
    parsed = json.loads(text)
    assert parsed["nan"] is None and parsed["inf"] is None
    assert parsed["np_nan"] is None
    assert parsed["np_int"] == 3 and parsed["array"] == [1, 2]
    assert parsed["set"] == ["a", "b"] and parsed["3"] == "int key"
    assert parsed["flag"] is True


def test_atomic_write_json_leaves_no_temp_file(tmp_path):
    target = tmp_path / "a" / "b.json"
    atomic_write_json({"x": float("nan"), "y": 1}, target)
    assert json.loads(target.read_text()) == {"x": None, "y": 1}
    assert not list(target.parent.glob("*.tmp"))


def test_assert_output_outside(tmp_path):
    protected = tmp_path / "processed"
    assert_output_outside(tmp_path / "diagnostics", protected)
    for bad in (protected, protected / "sub", protected / "a" / "b"):
        with pytest.raises(ValueError, match="processed-data"):
            assert_output_outside(bad, protected)
    # A sibling that merely shares a prefix is allowed.
    assert_output_outside(tmp_path / "processed_diagnostics", protected)


def test_stop_flag_records_sigterm_and_restores_handler():
    before = signal.getsignal(signal.SIGTERM)
    with StopFlag() as flag:
        assert not flag.requested
        signal.raise_signal(signal.SIGTERM)
        assert flag.requested and flag.signal_name == "SIGTERM"
    assert signal.getsignal(signal.SIGTERM) == before


def test_environment_reports_usable_cpus_and_slurm(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "CPU")
    environment = collect_environment()
    assert environment["slurm"]["SLURM_JOB_ID"] == "12345"
    assert environment["slurm"]["SLURM_JOB_PARTITION"] == "CPU"
    assert environment["usable_cpu_count"] >= 1
    assert environment["usable_cpu_count"] <= (environment["os_cpu_count"] or 10**6)
    assert environment["libraries"]["rdkit"] is not None
    assert "python" in environment and "hostname" in environment
    json.dumps(json_safe(environment), allow_nan=False)


def test_usable_cpu_count_respects_affinity_and_quota(monkeypatch):
    if hasattr(os, "sched_getaffinity"):
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2, 3})
    monkeypatch.setattr(
        "zinc_gnf.utils.runtime.read_cgroup_limits",
        lambda: {"cpu_quota_cpus": 1.0},
    )
    assert usable_cpu_count() == 1
    monkeypatch.setattr(
        "zinc_gnf.utils.runtime.read_cgroup_limits", lambda: {},
    )
    if hasattr(os, "sched_getaffinity"):
        assert usable_cpu_count() == 4
    assert isinstance(read_cgroup_limits(), dict)


def test_format_duration():
    assert format_duration(5.25) == "5.2s" or format_duration(5.25) == "5.3s"
    assert format_duration(125) == "2m 05s"
    assert format_duration(3725) == "1h 02m 05s"
    assert format_duration(float("nan")) == "n/a"
    assert format_duration(None) == "n/a"


# ---------------------------------------------------------------------------
# SLURM scripts: ISML cluster etiquette, checked statically.
# ---------------------------------------------------------------------------
def read_script(name: str) -> str:
    return (SLURM_DIR / name).read_text(encoding="utf-8")


def sbatch_directives(text: str) -> dict[str, str]:
    directives = {}
    for line in text.splitlines():
        match = re.match(r"#SBATCH\s+--([\w-]+)(?:=(.*))?$", line.strip())
        if match:
            directives[match.group(1)] = (match.group(2) or "").strip()
    return directives


def executable_lines(text: str) -> list[str]:
    """Script lines that are neither comments nor `echo` text."""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("echo"):
            continue
        lines.append(stripped)
    return lines


@pytest.mark.parametrize("name", ALL_SCRIPTS)
def test_scripts_exist_and_pass_bash_syntax_check(name):
    path = SLURM_DIR / name
    assert path.exists()
    assert path.read_text().startswith("#!/bin/bash")
    bash = shutil.which("bash")
    assert bash
    completed = subprocess.run([bash, "-n", str(path)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("name", JOB_SCRIPTS)
def test_job_scripts_are_cpu_only_on_cpu_partition(name):
    text = read_script(name)
    directives = sbatch_directives(text)
    assert directives["partition"] == "CPU"
    assert "gres" not in directives, "no GPU may be requested"
    assert "gpus" not in directives and "gpus-per-node" not in directives
    assert not re.search(r"--gres|--gpus", "\n".join(executable_lines(text)))
    assert "STUD" not in "\n".join(executable_lines(text))
    assert directives["nodes"] == "1" and directives["ntasks"] == "1"


@pytest.mark.parametrize("name", JOB_SCRIPTS)
def test_job_scripts_request_minimal_resources_within_24h(name):
    directives = sbatch_directives(read_script(name))
    assert int(directives["cpus-per-task"]) == 1       # nothing is parallel
    memory = directives["mem"]
    assert re.fullmatch(r"\d+G", memory) and int(memory[:-1]) <= 8
    hours, minutes, seconds = (int(v) for v in directives["time"].split(":"))
    total = hours * 3600 + minutes * 60 + seconds
    assert 0 < total <= 24 * 3600


@pytest.mark.parametrize("name", JOB_SCRIPTS)
def test_job_scripts_log_to_precreated_directory(name):
    text = read_script(name)
    directives = sbatch_directives(text)
    assert directives["output"].startswith("outputs/logs/")
    assert directives["error"].startswith("outputs/logs/")
    assert "%j" in directives["output"] and "%j" in directives["error"]
    # The directory is tracked in git (.gitkeep) and created by the prepare
    # script BEFORE submission; SLURM itself will not create it.
    repository = SLURM_DIR.parent
    assert (repository / "outputs" / "logs" / ".gitkeep").exists()
    assert "mkdir -p outputs/logs" in read_script("00_prepare_phase0.sh")


@pytest.mark.parametrize("name", JOB_SCRIPTS)
def test_job_scripts_use_conda_env_srun_and_unbuffered_python(name):
    text = read_script(name)
    code = "\n".join(executable_lines(text))
    assert "set -euo pipefail" in code
    assert "trap" in code and "ERR" in code
    assert "zinc_gnf/bin/python" in text           # existing Conda env
    assert "PYTHONUNBUFFERED=1" in code
    assert re.search(r'srun\s+"\$\{PYTHON\}"\s+-u\s+scripts/00_', code)
    assert "data/raw/zinc250k.csv" in text
    assert PINNED_SHA in text
    assert "scontrol show job" in code or "seff" in text


@pytest.mark.parametrize("name", ALL_SCRIPTS)
def test_scripts_make_no_network_or_install_calls(name):
    code = "\n".join(executable_lines(read_script(name)))
    forbidden = r"\b(wget|curl|git (clone|pull|fetch)|pip3? install|conda (install|create)|apt|yum|dnf)\b"
    assert not re.search(forbidden, code), (
        "compute-node scripts must not use the network or install packages"
    )


def test_download_instructions_target_the_login_node_only():
    for name in ALL_SCRIPTS:
        text = read_script(name)
        if "wget" in text:
            assert "login node" in text


@pytest.mark.parametrize("name", JOB_SCRIPTS)
def test_job_scripts_refuse_to_run_outside_slurm(name):
    """Running the script directly (e.g. on the master node) must stop
    immediately, before doing any work."""
    environment = {k: v for k, v in os.environ.items()
                   if not k.startswith("SLURM_")}
    completed = subprocess.run(
        ["bash", str(SLURM_DIR / name)], capture_output=True, text=True,
        env=environment, cwd=str(SLURM_DIR.parent), timeout=60,
    )
    assert completed.returncode == 64
    assert "sbatch" in completed.stderr
    assert "master node" in completed.stderr


def test_prepare_script_creates_directories_without_running_workloads(tmp_path):
    repository = tmp_path / "repo"
    (repository / "slurm").mkdir(parents=True)
    shutil.copy(SLURM_DIR / "00_prepare_phase0.sh", repository / "slurm")
    subprocess.run(["git", "init", "-q", str(repository)], check=True)

    completed = subprocess.run(
        ["bash", str(repository / "slurm" / "00_prepare_phase0.sh")],
        capture_output=True, text=True, cwd=str(tmp_path),
    )
    # No dataset yet: directories are created, then it explains how to fetch.
    assert completed.returncode == 1
    assert "login node" in completed.stderr
    for directory in ("outputs/logs", "outputs/diagnostics/chemistry",
                      "outputs/diagnostics/profiling", "outputs/figures/diagnostics"):
        assert (repository / directory).is_dir()

    (repository / "data" / "raw" / "zinc250k.csv").write_text("smiles,logP,qed,SAS\n")
    completed = subprocess.run(
        ["bash", str(repository / "slurm" / "00_prepare_phase0.sh")],
        capture_output=True, text=True, cwd=str(tmp_path),
    )
    assert completed.returncode == 0, completed.stderr
    assert "sbatch slurm/00_profile_preprocessing.sh" in completed.stdout


def test_existing_slice1_slurm_script_is_unchanged():
    """Phase 0 must not alter the production preprocessing job script."""
    text = read_script("01_preprocess.sh")
    assert "scripts/01_preprocess.py" in text
    assert "00_" not in text
