"""Stage 14: QED support analysis, label verification, and latent QED guidance.

The analysis only earns trust if it would actually detect each thing it claims to test,
so the tests construct the failure modes: a target outside the data's support, a
corrupted QED label, a distribution that matches in mean but not in shape, and a
predictor-guided sample that must never be presentable as native conditioning.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest
import torch

from tests.test_autoencoder_training import write_dataset
from zinc_gnf.preprocessing import ConditionScaler
from zinc_gnf.qed_guidance import (
    METHOD_LABELS,
    LatentQedPredictor,
    equal_budget_note,
    fit_latent_predictor,
    guide_latents,
    select_best,
)
from zinc_gnf.qed_support import (
    by_n,
    compare_distributions,
    feasibility,
    format_support_report,
    n_qed_heatmap,
    qed_histogram,
    read_generated_qed,
    support_table,
    verify_labels,
)

REPOSITORY = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""),
                                                  REPOSITORY / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def records_with(n_and_qed) -> list[dict]:
    out = []
    for index, (n, qed) in enumerate(n_and_qed):
        out.append({"n_node": int(n), "qed": float(qed), "row_id": index})
    return out


# -------------------------------------------------------------- histograms ----
def test_histogram_reports_counts_quantiles_and_high_qed_mass():
    values = [0.1] * 10 + [0.75] * 30 + [0.85] * 10
    result = qed_histogram(values)
    assert result["molecules"] == 50
    assert result["count_at_least_0p7"] == 40
    assert result["fraction_at_least_0p7"] == pytest.approx(0.8)
    assert result["count_at_least_0p8"] == 10
    assert result["quantiles"]["0.5"] == pytest.approx(0.75)
    assert sum(result["counts"]) == 50


def test_support_table_is_per_n_and_sums_to_the_split():
    records = records_with([(15, 0.5)] * 4 + [(23, 0.8)] * 6)
    rows = support_table(by_n(records))
    assert [r["n"] for r in rows] == [15, 23]
    assert sum(r["molecules"] for r in rows) == 10
    assert sum(r["share_of_split"] for r in rows) == pytest.approx(1.0)
    small, large = rows
    assert small["fraction_ge_0p7"] == 0.0
    assert large["fraction_ge_0p7"] == pytest.approx(1.0)
    assert large["count_ge_0p8"] == 6


def test_heatmap_has_conditional_density_and_quantiles():
    records = records_with([(15, 0.3)] * 5 + [(23, 0.9)] * 5)
    heatmap = n_qed_heatmap(by_n(records))
    assert heatmap["n_values"] == [15, 23]
    density = np.asarray(heatmap["conditional_density"])
    assert density.shape[0] == 2
    # each row is a distribution over QED WITHIN that N
    assert np.allclose(density.sum(axis=1), 1.0)
    assert heatmap["conditional_quantiles"]["0.5"] == [pytest.approx(0.3), pytest.approx(0.9)]


# ------------------------------------------------------------- feasibility ----
def test_feasibility_separates_well_represented_thin_tail_and_outside_support():
    # at N=23: 1000 molecules centred on 0.75, none below 0.5
    rng = np.random.default_rng(0)
    values = list(np.clip(rng.normal(0.75, 0.05, 1000), 0.5, 0.95))
    grouped = {23: values}
    rows = feasibility(grouped, requests=[(23, 0.75), (23, 0.93), (23, 0.20), (30, 0.6)])
    verdicts = {(r["n"], r["requested_qed"]): r["verdict"] for r in rows}
    assert verdicts[(23, 0.75)] == "well_represented"      # the conditional mode
    assert verdicts[(23, 0.20)] == "outside_support"       # nothing within +-0.05
    assert verdicts[(30, 0.6)] == "n_absent"               # that N does not occur
    # 0.93 is inside the clipped range but carries under 1% of the mass at this N
    assert verdicts[(23, 0.93)] == "thin_tail"
    thin = next(r for r in rows if r["requested_qed"] == 0.93)
    assert 0 < thin["fraction_in_window"] < 0.01
    outside = next(r for r in rows if r["requested_qed"] == 0.20)
    assert outside["molecules_in_window"] == 0
    assert outside["target_z_at_this_n"] < -5          # far outside the conditional mass
    assert outside["observed_range"][0] >= 0.5


def test_feasibility_reports_the_window_it_used():
    rows = feasibility({10: [0.5, 0.52]}, requests=[(10, 0.5)], neighbourhood=0.05)
    assert rows[0]["window"] == [pytest.approx(0.45), pytest.approx(0.55)]
    assert rows[0]["molecules_in_window"] == 2


# ---------------------------------------------------------------- labels ------
def test_label_verification_passes_on_the_real_pipeline_and_catches_corruption(tmp_path):
    from zinc_gnf.dataset import load_training_splits

    data = write_dataset(tmp_path / "data")
    train, _, _ = load_training_splits(data)
    scaler = ConditionScaler.fit(train)
    for record in train:
        record["condition"] = scaler.transform(record["qed"], record["n_node"])

    good = verify_labels(train, scaler=scaler, sample=len(train))
    assert good["unbuildable_graphs"] == 0
    assert good["qed_label_max_abs_difference"] < 1e-6
    assert "agree with RDKit" in good["verdict"]
    assert good["condition_matching_within_tolerance"] == pytest.approx(1.0)

    corrupted = [dict(r) for r in train]
    corrupted[0]["qed"] = float(corrupted[0]["qed"]) + 0.25
    bad = verify_labels(corrupted, scaler=scaler, sample=len(corrupted))
    assert bad["qed_label_max_abs_difference"] > 0.2
    assert "MISMATCH" in bad["verdict"]


def test_label_verification_uses_the_stored_graph_not_the_smiles(tmp_path):
    """Changing only the SMILES string must not hide a graph/label mismatch."""
    from zinc_gnf.dataset import load_training_splits

    data = write_dataset(tmp_path / "data")
    train, _, _ = load_training_splits(data)
    altered = [dict(r) for r in train]
    for record in altered:
        record["smiles"] = "C"                 # nonsense, deliberately
    result = verify_labels(altered, sample=len(altered))
    assert result["qed_label_max_abs_difference"] < 1e-6   # graph-based, so still fine


# ------------------------------------------------- distribution comparison ----
def test_distribution_comparison_catches_a_matching_mean_with_the_wrong_shape():
    rng = np.random.default_rng(1)
    real = rng.normal(0.7, 0.12, 4000)
    narrow = rng.normal(0.7, 0.01, 4000)              # same mean, 12x too narrow
    result = compare_distributions(narrow, real, requested_qed=0.7)
    assert abs(result["mean_shift"]) < 0.01
    assert result["generated_std_over_real_std"] < 0.2
    assert result["histogram_overlap"] < 0.5
    shifted = compare_distributions(rng.normal(0.5, 0.12, 4000), real, requested_qed=0.7)
    assert shifted["mean_shift"] < -0.1
    assert shifted["distance_of_generated_mean_from_request"] < -0.1


def test_distribution_comparison_handles_an_empty_population():
    result = compare_distributions([], [0.5, 0.6])
    assert result["generated_molecules"] == 0 and "no comparison" in result["note"]


def test_generated_qed_is_read_from_samples_and_recomputed_when_absent(tmp_path):
    path = tmp_path / "samples.jsonl"
    rows = [
        {"degree": {"valid": True, "smiles": "c1ccccc1", "qed": 0.44}},
        {"degree": {"valid": True, "smiles": "CCO"}},                  # no stored qed
        {"degree": {"valid": False, "smiles": None}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    values = read_generated_qed(path)
    assert len(values) == 2
    assert values[0] == pytest.approx(0.44)
    assert 0.0 < values[1] < 1.0


def test_the_report_states_the_feasibility_verdicts_and_the_label_check():
    records = records_with([(23, 0.8)] * 10)
    text = format_support_report(
        {"train": qed_histogram([0.8] * 10)}, support_table(by_n(records)),
        feasibility(by_n(records), requests=[(23, 0.4)]),
        {"verdict": "labels and scaler agree with RDKit on the stored graph",
         "qed_label_max_abs_difference": 1e-16, "checked": 10})
    assert "feasibility" in text.lower()
    assert "outside_support" in text
    assert "agree with RDKit" in text


# ------------------------------------------------------------- guidance -------
def tiny_flow_and_data():
    from tests.test_conditional_flow import StubEncoder, conditional_dataset, conditional_flow

    flow = conditional_flow(features=("n", "qed"))
    return flow, StubEncoder(), conditional_dataset()


def test_the_predictor_reports_its_own_held_out_accuracy():
    flow, encoder, dataset = tiny_flow_and_data()
    predictor, report = fit_latent_predictor(flow, encoder, dataset,
                                             indices=list(range(len(dataset))),
                                             device="cpu", batch_size=8, epochs=2)
    assert isinstance(predictor, LatentQedPredictor)
    assert report["held_out"]["molecules"] > 0
    assert np.isfinite(report["held_out"]["mae"])
    assert "GUIDE's accuracy" in report["note"]
    assert len(report["history"]) == 2


def test_guidance_moves_the_predicted_property_towards_the_target():
    torch.manual_seed(0)
    predictor = LatentQedPredictor(8)
    mask = torch.ones(4, 6, dtype=torch.bool)
    z = torch.randn(4, 6, 8)
    with torch.no_grad():
        before = float(predictor(z, mask).mean())
    target = before + 0.5
    result = guide_latents(predictor, z, mask, target=target, steps=15, step_size=0.2)
    after = float(result["predicted_qed"].mean())
    assert abs(after - target) < abs(before - target), "guidance must reduce the gap"
    assert result["z"].shape == z.shape
    assert len(result["trace"]) == 15
    # it is a TARGET, not a maximisation: a lower target must move it down
    lower = guide_latents(predictor, z, mask, target=before - 0.5, steps=15, step_size=0.2)
    assert float(lower["predicted_qed"].mean()) < before


def test_guidance_respects_padding_and_rejects_bad_arguments():
    predictor = LatentQedPredictor(8)
    mask = torch.tensor([[True, True, False]])
    z = torch.randn(1, 3, 8)
    result = guide_latents(predictor, z, mask, target=0.5, steps=3)
    assert torch.allclose(result["z"][0, 2], torch.zeros(8)), "padding must stay zero"
    with pytest.raises(ValueError):
        guide_latents(predictor, z, mask, target=0.5, steps=0)


def test_reranking_picks_the_closest_valid_candidate_and_counts_its_budget():
    records = [
        {"degree": {"valid": True, "qed": 0.30}},
        {"degree": {"valid": True, "qed": 0.58}},
        {"degree": {"valid": False, "qed": None}},
        {"degree": {"valid": True, "qed": 0.90}},
    ]
    kept = select_best(records, target=0.6, group=2)
    assert len(kept) == 2
    assert kept[0]["degree"]["qed"] == pytest.approx(0.58)       # closest in group 1
    assert kept[0]["selection"]["chosen"] == 1
    assert kept[1]["degree"]["qed"] == pytest.approx(0.90)       # the only valid one
    assert kept[1]["selection"]["candidates_valid"] == 1
    empty = select_best([{"degree": {"valid": False}}], target=0.6, group=1)
    assert empty[0]["selection"]["chosen"] is None


def test_the_budget_note_makes_the_comparison_fair():
    note = equal_budget_note(100, group=5, steps=10)
    assert note["native_candidates"] == 100
    assert note["rerank_candidates"] == 500
    assert note["guided_candidates"] == 100
    assert note["guided_extra_predictor_passes"] == 1000
    assert "fixes the number of DECODED candidates" in note["note"]


def test_method_labels_never_call_selection_native_conditioning():
    assert "native" in METHOD_LABELS["native"]
    for key in ("rerank", "guided"):
        assert "not learned conditioning" in METHOD_LABELS[key]
    assert "SELECTION" in METHOD_LABELS["rerank"]
    assert "GUIDED" in METHOD_LABELS["guided"]


# --------------------------------------------------- the script, end to end ---
def test_the_analysis_script_runs_and_writes_every_artifact(tmp_path):
    import pickle

    data = write_dataset(tmp_path / "data")
    payload = pickle.loads((Path(data) / "splits.pkl").read_bytes())
    fitted = ConditionScaler.fit(payload["train"])
    (Path(data) / "scaler.json").write_text(json.dumps(fitted.to_dict()))

    generated = tmp_path / "gen" / "T0.2_N9_QED0.4"
    generated.mkdir(parents=True)
    (generated / "samples.jsonl").write_text("\n".join(
        json.dumps({"n_node": 9, "degree": {"valid": True, "smiles": "c1ccccc1", "qed": 0.44}})
        for _ in range(5)))

    script = load_script("14_qed_support_analysis.py")
    output = tmp_path / "out"
    code = script.main(["--data-dir", str(data), "--output-dir", str(output),
                        "--target-n", "9,12", "--target-qed", "0.4,0.8",
                        "--generated-root", str(tmp_path / "gen"),
                        "--label-sample", "20"])
    assert code == 0
    for name in ("qed_distribution.png", "n_qed_heatmap.png", "generated_overlay.png",
                 "qed_support.json", "qed_support.txt"):
        assert (output / name).exists(), name
        assert (output / name).stat().st_size > 0
    payload = json.loads((output / "qed_support.json").read_text())
    assert payload["overall"]["train"]["molecules"] == 20
    assert payload["labels"]["qed_label_max_abs_difference"] < 1e-6
    assert payload["scaler"]["fit_on"] == "train_only"
    assert payload["feasibility"] and all("verdict" in r for r in payload["feasibility"])
    assert payload["generated"] and payload["generated"][0]["n"] == 9
    assert payload["stability"] is None                      # no checkpoint given


def test_the_analysis_script_validates_arguments_and_missing_data(tmp_path):
    script = load_script("14_qed_support_analysis.py")
    assert script.main(["--data-dir", str(tmp_path / "nothing"),
                        "--output-dir", str(tmp_path / "o")]) == 2
    assert script.main(["--data-dir", str(tmp_path), "--output-dir", str(tmp_path / "o2"),
                        "--target-n", ""]) == 4


def test_generated_cell_discovery_handles_both_stage_layouts(tmp_path):
    script = load_script("14_qed_support_analysis.py")
    joint = tmp_path / "T0.85_N23_QED0.6"
    joint.mkdir(parents=True)
    (joint / "samples.jsonl").write_text("{}\n")
    n_only = tmp_path / "N15"
    n_only.mkdir()
    (n_only / "samples.jsonl").write_text("{}\n")
    (n_only / "metrics.json").write_text(json.dumps({"requested": {"qed": 0.7}}))
    found = {(n, qed) for n, qed, _, _ in script.find_generated_cells(tmp_path)}
    assert (23, 0.6) in found
    assert (15, 0.7) in found


# ------------------------------------------------------------------ SLURM -----
def test_the_analysis_slurm_job_is_cpu_only_and_login_node_safe():
    path = REPOSITORY / "slurm" / "14_qed_support_analysis.sh"
    source = path.read_text(encoding="utf-8")
    directives = {}
    for line in source.splitlines():
        match = re.match(r"#SBATCH\s+--([\w-]+)(?:=(.*))?$", line.strip())
        if match:
            directives[match.group(1)] = (match.group(2) or "").strip()
    assert directives["partition"] == "CPU"
    assert "gres" not in directives
    assert directives["output"].startswith("outputs/logs/")
    assert 'if [[ -z "${SLURM_JOB_ID:-}" ]]' in source
    assert "scripts/14_qed_support_analysis.py" in source
    assert "srun " in source
    executable = "\n".join(line for line in source.splitlines()
                           if not line.lstrip().startswith("#") and "echo " not in line)
    assert "rm " not in executable and "mv " not in executable
    assert "--device cpu" in executable
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0


def test_the_literature_comparison_names_sources_and_its_own_limits():
    text = (REPOSITORY / "docs" / "stage14_literature_and_code_comparison.md").read_text()
    for marker in ("conditional_prior.py", "property_conditioning.py", "qm9_chem.py",
                   "null_conditioning.py", "optimize_property.py"):
        assert marker in text, marker
    assert "not retrievable" in text, "the unreadable GNF PDF must be declared"
    assert "not verified in source" in text, "GraphAF/GraphNVP status must be stated"
    assert "does not condition" in text.lower() or "does NOT" in text
