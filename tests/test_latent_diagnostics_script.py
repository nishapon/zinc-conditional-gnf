"""The diagnostic script end to end on tiny data, and static checks of its CPU job.

The protocol assertions matter as much as the plumbing: the covariance basis must
come from the TRAIN split, the robustness and profile measurements from VALIDATION,
and the test split must never be read.
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
from tests.test_flow_scripts_and_slurm import write_autoencoder
from zinc_gnf.dataset import load_training_splits

REPOSITORY = Path(__file__).resolve().parents[1]
SLURM_JOB = "07_latent_diagnostics.sh"
SCRIPT = "07_latent_diagnostics.py"


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


def write_samples(path: Path, smiles: list[str | None]) -> Path:
    """A samples.jsonl in the format scripts/05_generate.py writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(smiles):
            entry = {"valid": item is not None, "smiles": item, "sanitizable": item is not None,
                     "connected": item is not None, "fallback": index % 2 == 0,
                     "fallback_reason": "infeasible_degrees" if index % 2 == 0 else None,
                     "bonds": 2, "qed": 0.3, "rings": 0}
            handle.write(json.dumps({"index": index, "n_node": 6, "degree": entry,
                                     "constrained": entry}) + "\n")
    return path


@pytest.fixture
def setup(tmp_path):
    data = write_dataset(tmp_path / "data")
    autoencoder = write_autoencoder(tmp_path / "ae.pt")
    samples = write_samples(tmp_path / "samples.jsonl",
                            ["c1ccccc1", "CCO", None, "CC(=O)Nc1ccccc1"])
    return data, autoencoder, samples


def run(tmp_path, data, autoencoder, *extra):
    script = load_script(SCRIPT)
    return script.main(["--ae-checkpoint", str(autoencoder), "--data-dir", str(data),
                        "--output-dir", str(tmp_path / "diag"), "--spectrum-molecules", "20",
                        "--robustness-molecules", "6", "--profile-molecules", "8",
                        "--reconstruction-molecules", "4", "--k", "3",
                        "--magnitudes", "0.0,0.1", "--batch-size", "4", "--device", "cpu",
                        *extra])


def test_script_produces_all_three_parts(setup, tmp_path):
    data, autoencoder, samples = setup
    assert run(tmp_path, data, autoencoder, "--generated", f"flow_smoke={samples}") == 0
    output = tmp_path / "diag"
    assert (output / "diagnostics.txt").exists() and (output / "diagnostics.log").exists()
    result = json.loads((output / "diagnostics.json").read_text())

    # A: spectrum, from the TRAIN split
    spectrum = result["spectrum"]
    assert spectrum["dimension"] == 64 and spectrum["molecules"] == 20
    assert spectrum["nodes"] > 0 and len(spectrum["eigenvalues"]) == 64
    assert spectrum["eigenvalues"] == sorted(spectrum["eigenvalues"], reverse=True)
    assert 1.0 <= spectrum["participation_ratio"] <= 64.0
    assert 1 <= spectrum["effective_rank_99"] <= 64
    assert "strongest_correlations" in spectrum

    # B: robustness rows, on VALIDATION, with one shared baseline
    rows = result["robustness"]
    assert sum(1 for r in rows if r["mode"] == "baseline") == 1
    assert {r["group"] for r in rows if r["mode"] != "baseline"} == {"isotropic", "top", "bottom"}
    assert all(r["molecules"] == 6 for r in rows)
    assert all("degree_fallback_rate" in r for r in rows)

    # C: profiles, including the generated population read from disk
    profiles = result["profiles"]["profiles"]
    assert set(profiles) == {"real_validation", "ae_reconstruction", "real_matched_subset",
                             "flow_smoke"}
    assert profiles["real_validation"]["submitted"] == 8
    assert profiles["ae_reconstruction"]["submitted"] == 4
    assert profiles["flow_smoke"]["submitted"] == 4 and profiles["flow_smoke"]["unparsable"] == 1
    assert "qed_at_least_0p7" in profiles["real_validation"]
    assert result["profiles"]["reference"] == "real_validation"
    assert "flow_smoke" in result["profiles"]["differences_vs_reference"]

    # provenance and protocol are recorded
    assert result["autoencoder"]["sha256"] and result["autoencoder"]["model_config"]["degree_head"]
    assert result["settings"]["spectrum_split"] == "train"
    assert result["settings"]["robustness_split"] == "validation"
    assert result["settings"]["profile_split"] == "validation"
    report = (output / "diagnostics.txt").read_text()
    assert "participation ratio" in report and "QED >= 0.7" in report
    assert "basis from TRAIN" in report


def test_the_script_never_reads_the_test_split(setup, tmp_path):
    data, autoencoder, _ = setup
    source = (REPOSITORY / "scripts" / SCRIPT).read_text()
    assert "load_test_split" not in source and "load_training_splits" in source
    assert run(tmp_path, data, autoencoder) == 0
    train, validation, _ = load_training_splits(data)
    result = json.loads((tmp_path / "diag" / "diagnostics.json").read_text())
    # the profile cannot have seen more molecules than the validation split holds
    assert result["profiles"]["profiles"]["real_validation"]["submitted"] <= len(validation)
    assert result["spectrum"]["molecules"] <= len(train)


def test_skip_flags_and_argument_validation(setup, tmp_path):
    data, autoencoder, _ = setup
    assert run(tmp_path, data, autoencoder, "--skip-robustness", "--skip-profiles") == 0
    result = json.loads((tmp_path / "diag" / "diagnostics.json").read_text())
    assert "spectrum" in result and "robustness" not in result and "profiles" not in result

    script = load_script(SCRIPT)
    base = ["--ae-checkpoint", str(autoencoder), "--data-dir", str(data),
            "--output-dir", str(tmp_path / "d2"), "--device", "cpu"]
    assert script.main(base + ["--magnitudes", ""]) == 4
    assert script.main(base + ["--k", "0"]) == 4
    assert script.main(base + ["--generated", "no-equals-sign"]) == 4
    assert script.main(["--ae-checkpoint", str(tmp_path / "absent.pt"), "--data-dir", str(data),
                        "--output-dir", str(tmp_path / "d3"), "--device", "cpu"]) == 2
    assert script.main(["--ae-checkpoint", str(autoencoder), "--data-dir", str(tmp_path / "nope"),
                        "--output-dir", str(tmp_path / "d4"), "--device", "cpu"]) == 2


def test_a_missing_generated_file_is_reported_clearly(setup, tmp_path):
    data, autoencoder, _ = setup
    assert run(tmp_path, data, autoencoder, "--generated", f"ghost={tmp_path / 'ghost.jsonl'}") == 2


def test_generated_population_must_contain_the_requested_decoder(setup, tmp_path):
    data, autoencoder, samples = setup
    assert run(tmp_path, data, autoencoder, "--generated", f"flow={samples}",
               "--generated-decoder", "raw") == 4


# ---------------------------------------------------------------------------
# Static checks of the CPU job.
# ---------------------------------------------------------------------------
def test_slurm_job_is_cpu_only_and_syntactically_valid():
    source = text(SLURM_JOB)
    found = directives(source)
    assert subprocess.run(["bash", "-n", str(REPOSITORY / "slurm" / SLURM_JOB)]).returncode == 0
    assert found["partition"] == "CPU" and "gres" not in found
    assert found["cpus-per-task"] == "4" and found["mem"] == "16G"
    assert re.match(r"\d{2}:\d{2}:\d{2}", found["time"])
    assert "--device cpu" in source and "--device cuda" not in source
    assert "nvidia-smi" not in source
    assert "SLURM_JOB_ID" in source and "exit 64" in source


def test_slurm_job_is_read_only_and_touches_nothing_existing():
    source = text(SLURM_JOB)
    code = code_lines(source)
    assert "outputs/ablation/ae_degree/seed_3/last.pt" in source
    assert not re.search(r"(rm|mv|cp|chmod|truncate)[^\n]*(ae_degree|splits\.pkl)", code)
    assert not re.search(r"\bscancel\b|\bsbatch\b|\bssh\b", code)
    assert not re.search(r"\bpip\b|\bconda (install|create)|\bwget\b|\bcurl\b|git (pull|clone)", code)
    assert "load_test_split" not in source and "--split test" not in source
    assert 'OUT="outputs/metrics/latent_diagnostics"' in code      # its own output directory
    assert "outputs/checkpoints" not in code and "outputs/samples/flow_uncond" not in code


def test_slurm_job_checks_dependencies_and_degrades_without_samples():
    source = text(SLURM_JOB)
    assert "scipy.optimize" in source and "milp" in source
    assert "fall back to greedy" in source
    # a missing samples.jsonl must omit that population rather than fail the job
    assert 'if [[ -f "${GEN_PATH}" ]]' in source and "omitted" in source
    assert "MEASUREMENTS, not a diagnosis" in source


def test_slurm_job_passes_every_sample_size_as_an_override():
    code = code_lines(text(SLURM_JOB))
    for name in ("SPECTRUM_MOLECULES", "ROBUSTNESS_MOLECULES", "PROFILE_MOLECULES",
                 "RECONSTRUCTION_MOLECULES", "K"):
        assert f'{name}="${{{name}:-' in code, name
        assert f'"${{{name}}}"' in code, name
