"""Static ISML-rule checks for the same-budget autoencoder ablation SLURM scripts.
No job is submitted; scripts are parsed and `bash -n` checked."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess

import pytest

SLURM = Path(__file__).resolve().parents[1] / "slurm"
GPU_ARMS = {"baseline": "03_ae_ablation_baseline_gpu.sh", "degree": "03_ae_ablation_degree_gpu.sh",
            "noise4": "03_ae_ablation_noise4_gpu.sh"}
CPU_JOBS = ["03_analyze_reconstruction.sh", "03c_summarize_ablation.sh"]


def text(name):
    return (SLURM / name).read_text(encoding="utf-8")


def directives(source):
    found = {}
    for line in source.splitlines():
        m = re.match(r"#SBATCH\s+--([\w-]+)(?:=(.*))?$", line.strip())
        if m:
            found[m.group(1)] = (m.group(2) or "").strip()
    return found


@pytest.mark.parametrize("name", list(GPU_ARMS.values()) + CPU_JOBS)
def test_syntax_and_common_rules(name):
    source = text(name)
    assert subprocess.run(["bash", "-n", str(SLURM / name)]).returncode == 0
    assert "SLURM_JOB_ID" in source and "exit 64" in source          # refuses to run outside SLURM
    assert not re.search(r"\bpip\b|\bconda (install|create)|\bwget\b|\bcurl\b|git (pull|clone)", source)
    assert not re.search(r"\bscancel\b|\bssh\b", source)
    assert "--test" not in source and "split test" not in source


@pytest.mark.parametrize("variant,name", list(GPU_ARMS.items()))
def test_gpu_arm(variant, name):
    source, d = text(name), directives(text(name))
    assert d["partition"] == "STUD" and d["gres"] == "gpu:1"
    assert d["cpus-per-task"] == "4" and d["mem"] == "16G" and d["time"] == "01:30:00"
    assert "--device cuda" in source and "nvidia-smi" in source
    assert f"--variant {variant}" in source
    assert f'ROOT="outputs/ablation/ae_{variant}"' in source          # separate output dirs per arm
    assert 'if [[ -f "${OUT}/result.json" ]]' in source and "continue" in source   # never overwrites a finished seed
    assert "for SEED in 1 2 3" in source
    for budget in ("--train-molecules 100000", "--val-molecules 2000", "--steps 6000", "--batch-size 32"):
        assert budget in source                                       # identical budget in every arm


def test_arms_share_one_budget_and_differ_only_in_variant_and_names():
    def normalised(name, variant):
        s = text(name).replace(variant, "VARIANT")
        return "\n".join(l for l in s.splitlines() if not l.lstrip().startswith("#") and "echo" not in l)
    ref = normalised(GPU_ARMS["baseline"], "baseline")
    assert normalised(GPU_ARMS["degree"], "degree") == ref


@pytest.mark.parametrize("name", CPU_JOBS)
def test_cpu_jobs_use_cpu_partition_and_no_gpu(name):
    d = directives(text(name))
    assert d["partition"] == "CPU" and "gres" not in d
    assert "srun" not in text(name) or "--device cuda" not in text(name)


def test_summary_links_arms_and_writes_outside_the_arm_directories():
    source = text("03c_summarize_ablation.sh")
    assert "ln -s" in source and "outputs/ablation/summary" in source
    assert "rm -rf outputs/ablation/ae_" not in source
