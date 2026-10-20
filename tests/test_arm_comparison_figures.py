"""The presentation-figure script, driven by synthetic saved generation outputs.

The figures themselves cannot be asserted pixel by pixel, so the tests check the two
things that can be wrong in a way that matters: the numbers written alongside the
figures must equal the numbers in the saved inputs, and the summary must keep raw,
greedy-constrained and degree-constrained validity apart.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = "09_arm_comparison_figures.py"

DRUG_LIKE = ["CC(C)Cc1ccc(C(C)C(=O)O)cc1", "O=C(Nc1ccc(F)cc1)c1cc(Cl)ccc1Br",
             "CC(=O)Nc1ccccc1", "CCN(CC)CC", "c1ccccc1", "CCO",
             "c1ccncc1", "CC(C)C", "CCOC(=O)c1ccccc1", "Clc1ccc(N)cc1",
             "CC(=O)Oc1ccccc1C(=O)O", "COc1ccc(CCN)cc1", "CCCCO", "c1ccsc1"]


def load_script():
    spec = importlib.util.spec_from_file_location("figures", REPOSITORY / "scripts" / SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_arm(directory: Path, *, validity, qed_mean, fallback, diversity, uniqueness=0.97,
              novelty=0.99, qed_high=0.2, samples=20, with_records=True, step=10000):
    """A metrics.json (and samples.jsonl) in the format scripts/05_generate.py writes."""
    directory.mkdir(parents=True, exist_ok=True)
    def block(value, **extra):
        return {"strict_validity": {"value": value}, "sanitizable": {"value": min(1.0, value + 0.05)},
                "uniqueness": {"value": uniqueness}, "novelty_vs_train": {"value": novelty},
                "nuv": value * uniqueness * novelty, "fallback_rate": {"value": 0.0},
                "qed": {"mean": qed_mean}, "qed_at_least_0p7": {"value": qed_high},
                "diversity": {"internal_diversity": diversity}, **extra}
    metrics = {
        "samples": samples, "headline_decoder": "degree",
        "decoders": {"raw": block(validity["raw"]), "constrained": block(validity["constrained"]),
                     "degree": block(validity["degree"], fallback_rate={"value": fallback})},
        "settings": {"temperature": 0.85, "count": samples},
        "flow_checkpoint": f"outputs/checkpoints/x/seed_1/last.pt", "flow_step": step,
        "autoencoder": {"sha256": "a" * 64},
    }
    (directory / "metrics.json").write_text(json.dumps(metrics))
    if with_records:
        with (directory / "samples.jsonl").open("w", encoding="utf-8") as handle:
            for index in range(samples):
                smiles = DRUG_LIKE[index % len(DRUG_LIKE)]
                valid = index < round(validity["degree"] * samples)
                entry = {"valid": valid, "smiles": smiles if valid else None,
                         "sanitizable": valid, "connected": valid,
                         "fallback": index % 4 == 0, "fallback_reason": None, "bonds": 5,
                         "qed": 0.3 + 0.05 * (index % 10), "rings": 1}
                handle.write(json.dumps({"index": index, "n_node": 12, "degree": entry,
                                         "constrained": entry, "raw": entry}) + "\n")
    return directory


def write_reference(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {"qed": {"mean": 0.7314}, "qed_at_least_0p7": 0.6595,
               "aromatic_rings": {"mean": 1.830}, "aromatic_ring_fraction": 0.631,
               "degree_fallback_rate": 0.044}
    path.write_text(json.dumps({"profiles": {"profiles": {"real_validation": profile}}}))
    return path


@pytest.fixture
def saved(tmp_path):
    arm_a = write_arm(tmp_path / "a", validity={"raw": 0.0, "constrained": 0.98, "degree": 0.99},
                      qed_mean=0.24, fallback=0.79, diversity=0.90)
    arm_b = write_arm(tmp_path / "b", validity={"raw": 0.12, "constrained": 0.98, "degree": 0.99},
                      qed_mean=0.47, fallback=0.31, diversity=0.88, qed_high=0.18)
    return arm_a, arm_b, write_reference(tmp_path / "ref" / "diagnostics.json")


def run(saved, tmp_path, *extra):
    arm_a, arm_b, reference = saved
    return load_script().main(["--arm-a", str(arm_a), "--arm-b", str(arm_b),
                               "--output-dir", str(tmp_path / "figures"),
                               "--reference", str(reference), *extra])


def test_every_artifact_is_produced(saved, tmp_path):
    assert run(saved, tmp_path) == 0
    output = tmp_path / "figures"
    for name in ("arm_comparison.png", "arm_comparison.pdf", "arm_b_molecules.png",
                 "arm_b_molecules.svg", "arm_b_molecules_caption.txt", "results_summary.md",
                 "figure_data.json"):
        assert (output / name).exists(), name
        assert (output / name).stat().st_size > 0, name


def test_plotted_numbers_equal_the_saved_inputs(saved, tmp_path):
    assert run(saved, tmp_path) == 0
    data = json.loads((tmp_path / "figures" / "figure_data.json").read_text())
    arm_a, arm_b = data["arms"]["A"], data["arms"]["B"]
    assert arm_a["raw"]["strict_validity"] == pytest.approx(0.0)
    assert arm_b["raw"]["strict_validity"] == pytest.approx(0.12)
    assert arm_a["degree"]["strict_validity"] == pytest.approx(0.99)
    assert arm_a["degree"]["qed_mean"] == pytest.approx(0.24)
    assert arm_b["degree"]["qed_mean"] == pytest.approx(0.47)
    assert arm_a["degree"]["fallback_rate"] == pytest.approx(0.79)
    assert arm_b["degree"]["fallback_rate"] == pytest.approx(0.31)
    assert arm_b["degree"]["internal_diversity"] == pytest.approx(0.88)
    assert data["references"]["real ZINC250K (validation)"]["qed_mean"] == pytest.approx(0.7314)
    assert data["references"]["real ZINC250K (validation)"]["aromatic_ring_fraction"] == \
        pytest.approx(0.631)


def test_the_summary_keeps_the_three_validities_apart(saved, tmp_path):
    assert run(saved, tmp_path) == 0
    summary = (tmp_path / "figures" / "results_summary.md").read_text()
    assert "argmax bond matrix, no chemistry applied" in summary
    assert "greedy valence-aware assembly" in summary
    assert "degree-constrained integer program" in summary
    assert "Raw validity is the only one that reflects the model's own bond predictions" in summary
    # the three numbers appear as separate rows
    assert "| raw | 0.0000 | 0.1200 |" in summary
    assert "| degree | 0.9900 | 0.9900 |" in summary


def test_the_summary_reports_improvements_and_limitations(saved, tmp_path):
    assert run(saved, tmp_path) == 0
    summary = (tmp_path / "figures" / "results_summary.md").read_text()
    assert "+0.1200" in summary and "raw validity" in summary        # improvement, signed
    assert "-0.4800" in summary                                      # fallback fell
    assert "10,000-step" in summary and "50,000-step" in summary
    assert "67.6% more parameters" in summary                        # capacity confound
    assert "node-order dependent" in summary
    assert "greedy fallback" in summary                              # fallback 0.31 > 0.25
    assert "65.9%" in summary or "66.0%" in summary                  # QED>=0.7 reference
    assert "never the held-out test split" in summary
    assert "a" * 16 in summary                                       # autoencoder provenance


def test_the_molecule_grid_states_its_selection_rule(saved, tmp_path):
    assert run(saved, tmp_path, "--grid-molecules", "6") == 0
    caption = (tmp_path / "figures" / "arm_b_molecules_caption.txt").read_text()
    assert "not cherry-picked" in caption and "unique valid molecules" in caption
    assert "degree-constrained assembly" in caption
    data = json.loads((tmp_path / "figures" / "figure_data.json").read_text())
    assert 1 <= len(data["grid"]) <= 6
    qed_values = [entry["qed"] for entry in data["grid"]]
    assert qed_values == sorted(qed_values)                          # spread, ascending
    assert data["selection"] == "spread"


def test_selection_modes_are_deterministic_and_differ(saved, tmp_path):
    first = tmp_path / "f1"
    assert run(saved, first, "--select", "best", "--grid-molecules", "4") == 0
    best = json.loads((first / "figures" / "figure_data.json").read_text())["grid"]
    second = tmp_path / "f2"
    assert run(saved, second, "--select", "best", "--grid-molecules", "4") == 0
    assert json.loads((second / "figures" / "figure_data.json").read_text())["grid"] == best
    third = tmp_path / "f3"
    assert run(saved, third, "--select", "spread", "--grid-molecules", "4") == 0
    spread = json.loads((third / "figures" / "figure_data.json").read_text())["grid"]
    assert [e["qed"] for e in best] >= [e["qed"] for e in spread]


def test_it_works_without_samples_and_without_references(tmp_path):
    arm_a = write_arm(tmp_path / "a", validity={"raw": 0.0, "constrained": 0.9, "degree": 0.95},
                      qed_mean=0.3, fallback=0.5, diversity=0.8, with_records=False)
    arm_b = write_arm(tmp_path / "b", validity={"raw": 0.1, "constrained": 0.9, "degree": 0.96},
                      qed_mean=0.4, fallback=0.4, diversity=0.8, with_records=False)
    code = load_script().main(["--arm-a", str(arm_a), "--arm-b", str(arm_b),
                               "--output-dir", str(tmp_path / "figures")])
    assert code == 0
    output = tmp_path / "figures"
    assert (output / "arm_comparison.png").exists()
    assert not (output / "arm_b_molecules.png").exists()             # no samples, no grid
    summary = (output / "results_summary.md").read_text()
    assert "| raw |" in summary and "—" in summary                   # no reference column values


def test_missing_or_malformed_inputs_are_reported(saved, tmp_path):
    arm_a, arm_b, reference = saved
    script = load_script()
    assert script.main(["--arm-a", str(tmp_path / "ghost"), "--arm-b", str(arm_b),
                        "--output-dir", str(tmp_path / "f")]) == 2
    assert script.main(["--arm-a", str(arm_a), "--arm-b", str(arm_b),
                        "--output-dir", str(tmp_path / "f"),
                        "--reference", str(tmp_path / "ghost.json")]) == 2
    assert script.main(["--arm-a", str(arm_a), "--arm-b", str(arm_b),
                        "--output-dir", str(tmp_path / "f"), "--grid-molecules", "0"]) == 4
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "metrics.json").write_text(json.dumps(
        {"samples": 5, "headline_decoder": "degree", "decoders": {}}))
    assert script.main(["--arm-a", str(arm_a), "--arm-b", str(broken),
                        "--output-dir", str(tmp_path / "f")]) == 4


def test_nothing_is_written_outside_the_output_directory(saved, tmp_path):
    before = {p for p in REPOSITORY.rglob("*") if ".git" not in p.parts}
    assert run(saved, tmp_path) == 0
    after = {p for p in REPOSITORY.rglob("*") if ".git" not in p.parts}
    assert after - before == set()


def test_every_grid_cell_contains_a_molecule(saved, tmp_path):
    """Regression: with only a canvas size RDKit draws all molecules on top of each other.

    The panel-size constructor is what tiles them, so each of the 12 cells must contain
    ink. Before the fix, eleven cells were blank.
    """
    import numpy as np
    from PIL import Image

    assert run(saved, tmp_path, "--grid-molecules", "12") == 0
    image = Image.open(tmp_path / "figures" / "arm_b_molecules.png").convert("L")
    data = json.loads((tmp_path / "figures" / "figure_data.json").read_text())
    drawn = len(data["grid"])
    columns = 4
    rows = -(-drawn // columns)
    panel_width, panel_height = image.width // columns, image.height // rows
    assert (image.width, image.height) == (columns * 300, rows * 300)
    occupied = 0
    for index in range(drawn):
        row, column = divmod(index, columns)
        cell = image.crop((column * panel_width, row * panel_height,
                           (column + 1) * panel_width, (row + 1) * panel_height))
        dark = int((np.asarray(cell) < 200).sum())
        if dark > 50:
            occupied += 1
    assert occupied == drawn, f"only {occupied} of {drawn} grid cells contain a drawing"


def test_the_comparison_figure_has_the_expected_canvas():
    """A four-panel figure at publication resolution, not a default-size plot."""
    from PIL import Image

    import json as json_module
    from pathlib import Path as PathType

    # rendered by the fixture-driven tests above; re-render into a temp place cheaply
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        base = PathType(directory)
        arm_a = write_arm(base / "a", validity={"raw": 0.0, "constrained": 0.9, "degree": 0.95},
                          qed_mean=0.3, fallback=0.5, diversity=0.8)
        arm_b = write_arm(base / "b", validity={"raw": 0.1, "constrained": 0.9, "degree": 0.96},
                          qed_mean=0.4, fallback=0.4, diversity=0.8)
        assert load_script().main(["--arm-a", str(arm_a), "--arm-b", str(arm_b),
                                   "--output-dir", str(base / "figures")]) == 0
        image = Image.open(base / "figures" / "arm_comparison.png")
        assert image.width >= 2000 and image.height >= 1400          # 200 dpi, 4 panels
        assert (base / "figures" / "arm_comparison.pdf").stat().st_size > 5000
        assert json_module.loads(
            (base / "figures" / "figure_data.json").read_text())["arms"]["B"]["raw"][
                "strict_validity"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Static checks of the CPU job.
# ---------------------------------------------------------------------------
def test_the_figure_job_is_cpu_only_and_read_only():
    import re
    import subprocess

    source = (REPOSITORY / "slurm" / "09_arm_comparison_figures.sh").read_text()
    code = "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith("#") and "echo " not in line)
    assert subprocess.run(["bash", "-n", str(REPOSITORY / "slurm" /
                                             "09_arm_comparison_figures.sh")]).returncode == 0
    assert "#SBATCH --partition=CPU" in source and "--gres" not in source
    assert "SLURM_JOB_ID" in source and "exit 64" in source
    assert "MPLBACKEND=Agg" in source                       # headless rendering
    assert 'OUT="outputs/figures/arm_comparison"' in code
    assert "gitignored" in source                           # figures are not committed
    # reads saved outputs only: no training, no sampling, no job submission
    assert "04_train_flow.py" not in code and "05_generate.py" not in code
    assert not re.search(r"\bscancel\b|\bsbatch\b|\bssh\b", code)
    assert not re.search(r"\bpip\b|\bconda (install|create)|\bwget\b|\bcurl\b", code)
    assert not re.search(r"(rm|mv|truncate)[^\n]*(ae_degree|samples|checkpoints)", code)
    # a missing reference file is omitted rather than fatal
    assert 'if [[ -f "${PATH_ENTRY}" ]]' in code and "omitted" in source
