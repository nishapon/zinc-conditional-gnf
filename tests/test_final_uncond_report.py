"""Stage 11: the final 10k-vs-50k unconditional report.

The figures cannot be asserted pixel by pixel, so the tests check what the report
*claims*: that every number is recomputed from the saved samples with RDKit, that the
three validities stay apart, that metrics which got worse are printed, that the grid
states its selection rule, and that the job refuses two runs sampled at different
temperatures.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import subprocess

import pytest
from rdkit import Chem
from rdkit.Chem import QED

REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = "11_final_uncond_report.py"
SLURM = REPOSITORY / "slurm" / "11_final_uncond_report.sh"

# Real molecules, so RDKit recomputes genuine QED and aromaticity.
AROMATIC = ["c1ccccc1C(=O)NC", "c1ccc2ccccc2c1", "Cc1ccc(O)cc1", "c1ccncc1C(=O)O",
            "CCOc1ccccc1N", "c1ccc(cc1)S(=O)(=O)N"]
ALIPHATIC = ["CCCCO", "CC(C)CC(N)=O", "OCCNCC", "CC(=O)OCC", "CCCCCCN", "CC1CCCCC1O"]


def load_script():
    spec = importlib.util.spec_from_file_location("final_report", REPOSITORY / "scripts" / SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_run(directory: Path, *, smiles: list[str], samples: int, raw_validity: float,
              fallback: float, novelty: float, temperature: float = 0.2,
              step: int = 10000) -> Path:
    """A saved generation directory in the exact shape scripts/05_generate.py writes."""
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(samples):
        valid = index < round(len(smiles) / len(smiles) * samples)   # all degree-valid
        smile = smiles[index % len(smiles)]
        degree = {"sanitizable": True, "connected": True, "valid": bool(valid),
                  "smiles": smile, "fragments": 1,
                  "fallback": index < round(fallback * samples),
                  "fallback_reason": None, "bonds": 7,
                  "qed": float(QED.qed(Chem.MolFromSmiles(smile)))}
        raw_valid = index < round(raw_validity * samples)
        raw = dict(degree, valid=bool(raw_valid),
                   smiles=smile if raw_valid else None)
        records.append({"index": index, "n_node": 20, "degree": degree,
                        "constrained": dict(degree), "raw": raw})
    with (directory / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    block = lambda validity: {  # noqa: E731
        "strict_validity": {"value": validity}, "sanitizable": {"value": 1.0},
        "uniqueness": {"value": 1.0}, "novelty_vs_train": {"value": novelty},
        "nuv": validity * novelty, "fallback_rate": {"value": fallback},
        "diversity": {"internal_diversity": 0.9}, "qed": {"mean": 0.5}}
    (directory / "metrics.json").write_text(json.dumps({
        "samples": samples, "headline_decoder": "degree",
        "flow_checkpoint": f"outputs/checkpoints/flow_uncond_ar/seed_1/step{step}.pt",
        "flow_step": step, "settings": {"temperature": temperature},
        "autoencoder": {"sha256": "b" * 64},
        "decoders": {"raw": block(raw_validity), "constrained": block(1.0),
                     "degree": block(1.0)}}))
    return directory


@pytest.fixture
def saved(tmp_path):
    early = write_run(tmp_path / "t10k", smiles=ALIPHATIC, samples=24, raw_validity=0.0,
                      fallback=0.65, novelty=0.99, step=10000)
    late = write_run(tmp_path / "t50k", smiles=AROMATIC, samples=24, raw_validity=0.25,
                     fallback=0.30, novelty=0.95, step=50000)
    return early, late


def run(saved, tmp_path, *extra):
    early, late = saved
    return load_script().main(["--early", str(early), "--late", str(late),
                               "--output-dir", str(tmp_path / "final"), *extra])


def test_every_artifact_is_produced(saved, tmp_path):
    assert run(saved, tmp_path) == 0
    output = tmp_path / "final"
    for name in ("arm_b_10k_vs_50k.png", "arm_b_10k_vs_50k.pdf", "arm_b_50k_molecules.png",
                 "arm_b_50k_molecules.svg", "arm_b_50k_molecules_caption.txt",
                 "final_summary.md", "figure_data.json"):
        assert (output / name).exists(), name
        assert (output / name).stat().st_size > 0, name


def test_chemistry_is_recomputed_from_the_samples_not_taken_from_metrics(saved, tmp_path):
    """metrics.json claims qed mean 0.5 for both runs; the truth is in the SMILES."""
    assert run(saved, tmp_path) == 0
    data = json.loads((tmp_path / "final" / "figure_data.json").read_text())
    late = data["runs"]["50k"]["decoders"]["degree"]
    expected = [QED.qed(Chem.MolFromSmiles(s)) for s in AROMATIC]
    assert late["qed_mean"] == pytest.approx(sum(expected) / len(expected), abs=1e-6)
    assert late["saved_qed_mean"] == 0.5                     # kept, but not what is plotted
    assert late["qed_mean"] != pytest.approx(0.5, abs=1e-3)
    assert data["runs"]["50k"]["recomputed_from_samples"] is True


def test_aromaticity_and_qed_fraction_are_measured(saved, tmp_path):
    assert run(saved, tmp_path) == 0
    data = json.loads((tmp_path / "final" / "figure_data.json").read_text())
    early = data["runs"]["10k"]["decoders"]["degree"]
    late = data["runs"]["50k"]["decoders"]["degree"]
    # the 10k fixture is purely aliphatic, the 50k one purely aromatic
    assert early["aromatic_rings_mean"] == pytest.approx(0.0)
    assert early["molecules_with_aromatic_ring"] == pytest.approx(0.0)
    assert late["aromatic_rings_mean"] >= 1.0
    assert late["molecules_with_aromatic_ring"] == pytest.approx(1.0)
    assert 0.0 <= late["qed_at_least_0p7"] <= 1.0
    assert late["aromatic_ring_fraction"] == pytest.approx(1.0)


def test_the_three_validities_stay_apart(saved, tmp_path):
    assert run(saved, tmp_path) == 0
    summary = (tmp_path / "final" / "final_summary.md").read_text()
    assert "argmax bond matrix, no chemistry applied" in summary
    assert "greedy valence-aware assembly" in summary
    assert "degree-constrained integer program" in summary
    assert "Raw validity is the only number that reflects the model's own bond predictions" \
        in summary
    assert "| raw | 0.0000 | 0.2500 | +0.2500 |" in summary
    assert "fallback" in summary


def test_metrics_that_worsened_are_not_hidden(saved, tmp_path):
    assert run(saved, tmp_path) == 0
    summary = (tmp_path / "final" / "final_summary.md").read_text()
    assert "## What got WORSE or did not move" in summary
    section = summary.split("## What got WORSE or did not move", 1)[1]
    # novelty fell 0.99 -> 0.95 in the fixture; it must appear as a worsening
    assert "novelty vs train" in section and "-0.0400" in section
    improved = summary.split("## What improved with 5x the training", 1)[1].split("##", 1)[0]
    assert "aromatic rings / molecule" in improved
    assert "degree-decoder fallback" in improved          # 0.65 -> 0.30 counts as better


def test_the_remaining_gap_to_real_zinc_is_stated(saved, tmp_path):
    assert run(saved, tmp_path) == 0
    summary = (tmp_path / "final" / "final_summary.md").read_text()
    assert "## Remaining gap to real ZINC250K" in summary
    assert "0.7314" in summary                            # real mean QED
    assert "1.8305" in summary or "1.830" in summary      # real aromatic rings
    data = json.loads((tmp_path / "final" / "figure_data.json").read_text())
    assert data["reference"]["label"] == "real ZINC250K (validation)"
    assert data["reference"]["qed_mean"] == pytest.approx(0.7314)
    assert "job 332841" in data["reference"]["source"]


def test_a_supplied_reference_file_overrides_the_builtin_constants(saved, tmp_path):
    reference = tmp_path / "ref.json"
    reference.write_text(json.dumps({"profiles": {"profiles": {"real_validation": {
        "qed": {"mean": 0.7000}, "qed_at_least_0p7": 0.6,
        "aromatic_rings": {"mean": 1.5}, "aromatic_ring_fraction": 0.6}}}}))
    assert run(saved, tmp_path, "--reference", str(reference)) == 0
    data = json.loads((tmp_path / "final" / "figure_data.json").read_text())
    assert data["reference"]["qed_mean"] == pytest.approx(0.7)


def test_the_grid_has_twelve_molecules_with_qed_and_states_its_rule(saved, tmp_path):
    assert run(saved, tmp_path, "--grid-molecules", "4") == 0
    data = json.loads((tmp_path / "final" / "figure_data.json").read_text())
    assert 1 <= len(data["grid"]) <= 4
    values = [entry["qed"] for entry in data["grid"]]
    assert values == sorted(values)                       # spread, ascending
    for entry in data["grid"]:
        assert Chem.MolFromSmiles(entry["smiles"]) is not None
        assert entry["qed"] == pytest.approx(
            QED.qed(Chem.MolFromSmiles(entry["smiles"])), abs=1e-6)
    caption = (tmp_path / "final" / "arm_b_50k_molecules_caption.txt").read_text()
    assert "not cherry-picked" in caption and "unique valid molecules" in caption
    assert "50k steps" in caption


def test_grid_default_is_twelve(saved, tmp_path):
    assert run(saved, tmp_path) == 0
    data = json.loads((tmp_path / "final" / "figure_data.json").read_text())
    assert len(data["grid"]) == min(12, data["grid_unique_pool"])


def test_runs_at_different_temperatures_are_refused(tmp_path):
    early = write_run(tmp_path / "a", smiles=ALIPHATIC, samples=8, raw_validity=0.0,
                      fallback=0.5, novelty=1.0, temperature=0.2)
    late = write_run(tmp_path / "b", smiles=AROMATIC, samples=8, raw_validity=0.1,
                     fallback=0.3, novelty=1.0, temperature=0.85)
    code = load_script().main(["--early", str(early), "--late", str(late),
                               "--output-dir", str(tmp_path / "out")])
    assert code == 4


def test_recomputation_disagreements_are_reported(tmp_path):
    early = write_run(tmp_path / "a", smiles=ALIPHATIC, samples=20, raw_validity=0.0,
                      fallback=0.5, novelty=1.0)
    late = write_run(tmp_path / "b", smiles=AROMATIC, samples=20, raw_validity=0.25,
                     fallback=0.3, novelty=1.0, step=50000)
    # corrupt the saved raw validity of the late run: the recomputation must notice
    metrics = json.loads((late / "metrics.json").read_text())
    metrics["decoders"]["raw"]["strict_validity"]["value"] = 0.99
    (late / "metrics.json").write_text(json.dumps(metrics))
    assert load_script().main(["--early", str(early), "--late", str(late),
                               "--output-dir", str(tmp_path / "out")]) == 0
    summary = (tmp_path / "out" / "final_summary.md").read_text()
    assert "Recomputation disagreements" in summary
    assert "recomputed strict_validity 0.2500 vs 0.9900" in summary


def test_missing_inputs_are_reported(saved, tmp_path):
    early, late = saved
    script = load_script()
    assert script.main(["--early", str(tmp_path / "ghost"), "--late", str(late),
                        "--output-dir", str(tmp_path / "o1")]) == 2
    assert script.main(["--early", str(early), "--late", str(late),
                        "--output-dir", str(tmp_path / "o2"), "--grid-molecules", "0"]) == 4
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "metrics.json").write_text("{not json")
    assert script.main(["--early", str(early), "--late", str(broken),
                        "--output-dir", str(tmp_path / "o3")]) == 4


def test_it_works_without_samples_files(tmp_path):
    """Degrades to metrics.json numbers and says so, instead of failing."""
    early = write_run(tmp_path / "a", smiles=ALIPHATIC, samples=8, raw_validity=0.0,
                      fallback=0.5, novelty=1.0)
    late = write_run(tmp_path / "b", smiles=AROMATIC, samples=8, raw_validity=0.2,
                     fallback=0.3, novelty=1.0, step=50000)
    for directory in (early, late):
        (directory / "samples.jsonl").unlink()
    assert load_script().main(["--early", str(early), "--late", str(late),
                               "--output-dir", str(tmp_path / "out")]) == 0
    output = tmp_path / "out"
    assert (output / "arm_b_10k_vs_50k.png").exists()
    assert not (output / "arm_b_50k_molecules.png").exists()
    summary = (output / "final_summary.md").read_text()
    assert "from metrics.json only" in summary


def test_nothing_is_written_outside_the_output_directory(saved, tmp_path):
    early, late = saved
    before = {p: p.stat().st_mtime_ns for p in (early, late) for p in p.iterdir()}
    assert run(saved, tmp_path) == 0
    after = {p: p.stat().st_mtime_ns for p in (early, late) for p in p.iterdir()}
    assert before == after, "the inputs must be read-only"


def test_slurm_job_follows_the_cluster_rules():
    source = SLURM.read_text(encoding="utf-8")
    directives = {}
    for line in source.splitlines():
        match = re.match(r"#SBATCH\s+--([\w-]+)(?:=(.*))?$", line.strip())
        if match:
            directives[match.group(1)] = (match.group(2) or "").strip()
    assert directives["partition"] == "CPU"
    assert "gres" not in directives
    assert directives["output"].startswith("outputs/logs/")
    assert directives["time"] and directives["mem"]
    assert 'if [[ -z "${SLURM_JOB_ID:-}" ]]' in source
    assert "scripts/11_final_uncond_report.py" in source
    assert "outputs/figures/arm_b_final" in source
    assert "flow_uncond_ar_lowtemp" in source and "flow_uncond_ar_50k" in source
    assert "T0.2" in source
    assert subprocess.run(["bash", "-n", str(SLURM)], capture_output=True).returncode == 0
