"""Tests for the conditional-QED diagnostics.

The diagnostics exist to stop a bare MAE being read as evidence of control, so the tests
pin the three claims that do the work:

  * the constant-predictor floor is the L1 optimum, which sits at the MEDIAN target. Get
    this wrong and the central comparison is wrong.
  * responsiveness is measured at FIXED N from the lowest to the highest request, against
    the ideal, and non-monotone rows are reported as such rather than smoothed.
  * size-versus-request attribution is a measured ratio, not an assertion.

Pure arithmetic: no torch, no rdkit, no numpy.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "39_conditioning_diagnostics.py"
REPORTED_PATH = ROOT / "docs" / "reported_results.json"


def diagnostics():
    spec = importlib.util.spec_from_file_location("conditioning_diagnostics",
                                                   SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flat_grid(value):
    return {"15": {"0.4": value, "0.6": value, "0.8": value}}


# ------------------------------------------------------- constant-predictor floor ---
def test_the_floor_sits_at_the_median_target():
    """argmin_c mean_t |c - t| is the median of t. For {0.4,0.6,0.8} that is 0.6."""
    module = diagnostics()
    floor = module.constant_predictor_floor([0.4, 0.6, 0.8])
    assert floor["best_constant"] == pytest.approx(0.6)
    assert floor["mae_floor"] == pytest.approx(0.4 / 3)


def test_no_constant_on_the_grid_beats_the_reported_floor():
    module = diagnostics()
    floor = module.constant_predictor_floor([0.4, 0.6, 0.8])
    assert min(floor["mae_by_constant"].values()) >= floor["mae_floor"] - 1e-12


def test_the_floor_handles_an_even_number_of_targets():
    module = diagnostics()
    floor = module.constant_predictor_floor([0.4, 0.8])
    assert floor["mae_floor"] == pytest.approx(0.2)


def test_a_single_target_has_a_zero_floor():
    module = diagnostics()
    assert module.constant_predictor_floor([0.6])["mae_floor"] == pytest.approx(0.0)


# ------------------------------------------------------------------ responsiveness ---
def test_a_perfect_model_delivers_the_full_ideal_response():
    module = diagnostics()
    grid = {"15": {"0.4": 0.4, "0.6": 0.6, "0.8": 0.8}}
    result = module.responsiveness(grid, [0.4, 0.6, 0.8])
    assert result["by_n"]["15"]["response"] == pytest.approx(0.4)
    assert result["by_n"]["15"]["fraction_of_ideal"] == pytest.approx(1.0)
    assert result["by_n"]["15"]["monotonic"] is True


def test_a_model_that_ignores_the_request_delivers_zero_response():
    module = diagnostics()
    result = module.responsiveness(flat_grid(0.55), [0.4, 0.6, 0.8])
    assert result["by_n"]["15"]["response"] == pytest.approx(0.0)
    assert result["mean_fraction_of_ideal"] == pytest.approx(0.0)
    assert result["positive_at_n"] == []


def test_a_negative_response_is_reported_as_negative():
    module = diagnostics()
    grid = {"23": {"0.4": 0.609, "0.6": 0.541, "0.8": 0.597}}
    result = module.responsiveness(grid, [0.4, 0.6, 0.8])
    assert result["by_n"]["23"]["response"] == pytest.approx(-0.012)
    assert result["by_n"]["23"]["fraction_of_ideal"] < 0
    assert result["by_n"]["23"]["monotonic"] is False


def test_non_monotone_rows_are_flagged_even_when_the_endpoints_rise():
    """0.600 -> 0.569 -> 0.644 rises overall but dips in the middle."""
    module = diagnostics()
    grid = {"15": {"0.4": 0.600, "0.6": 0.569, "0.8": 0.644}}
    result = module.responsiveness(grid, [0.4, 0.6, 0.8])
    assert result["by_n"]["15"]["response"] > 0
    assert result["by_n"]["15"]["monotonic"] is False


def test_a_missing_endpoint_is_reported_not_guessed():
    module = diagnostics()
    grid = {"15": {"0.4": 0.5, "0.6": 0.5}}
    result = module.responsiveness(grid, [0.4, 0.6, 0.8])
    assert "note" in result["by_n"]["15"]


# ------------------------------------------------------------ size vs the request ---
def test_a_size_driven_model_is_identified_as_size_driven():
    module = diagnostics()
    grid = {"15": {"0.4": 0.70, "0.6": 0.70, "0.8": 0.70},
            "30": {"0.4": 0.40, "0.6": 0.40, "0.8": 0.40}}
    result = module.variance_attribution(grid, [0.4, 0.6, 0.8])
    assert result["spread_across_n"] == pytest.approx(0.30)
    assert result["spread_across_request"] == pytest.approx(0.0)
    assert result["size_over_request_ratio"] is None   # undefined, not infinite


def test_a_request_driven_model_has_a_ratio_below_one():
    module = diagnostics()
    grid = {"15": {"0.4": 0.4, "0.6": 0.6, "0.8": 0.8},
            "30": {"0.4": 0.4, "0.6": 0.6, "0.8": 0.8}}
    result = module.variance_attribution(grid, [0.4, 0.6, 0.8])
    assert result["spread_across_n"] == pytest.approx(0.0)
    assert result["size_over_request_ratio"] == pytest.approx(0.0)
    assert result["monotone_in_request"] is True


def test_non_monotone_request_means_are_flagged():
    module = diagnostics()
    grid = {"15": {"0.4": 0.60, "0.6": 0.53, "0.8": 0.58}}
    result = module.variance_attribution(grid, [0.4, 0.6, 0.8])
    assert result["monotone_in_request"] is False


# ----------------------------------------------------------------- cell-mean MAE ---
def test_cell_mean_mae_excludes_within_cell_scatter():
    module = diagnostics()
    grid = {"15": {"0.4": 0.5, "0.6": 0.6, "0.8": 0.7}}
    result = module.cell_mean_mae(grid, [0.4, 0.6, 0.8])
    assert result["overall"] == pytest.approx((0.1 + 0.0 + 0.1) / 3)


def test_cell_mean_mae_is_zero_for_a_perfect_model():
    module = diagnostics()
    grid = {"15": {"0.4": 0.4, "0.6": 0.6, "0.8": 0.8}}
    assert module.cell_mean_mae(grid, [0.4, 0.6, 0.8])["overall"] == pytest.approx(0.0)


# ------------------------------------------------------------ selection leverage ---
def test_selection_leverage_separates_support_from_steering():
    module = diagnostics()
    result = module.selection_leverage(0.1848, {"10": 0.0327}, 0.1333)
    entry = result["by_k"]["10"]
    assert entry["improvement_over_native"] == pytest.approx(0.1521)
    assert entry["beats_constant_floor"] is True
    assert "does NOT show conditional control" in result["interpretation"]


def test_selection_leverage_handles_no_selected_results():
    module = diagnostics()
    assert module.selection_leverage(0.18, None, 0.13)["by_k"] == {}


# --------------------------------------------------------- end-to-end on our data ---
def test_the_reported_numbers_reproduce_the_headline_findings():
    module = diagnostics()
    blocks = module.analyse(json.loads(REPORTED_PATH.read_text()))
    assert blocks["constant_floor"]["mae_floor"] == pytest.approx(0.1333, abs=1e-4)
    assert blocks["cell_mean_mae"]["overall"] == pytest.approx(0.1508, abs=1e-3)
    assert blocks["responsiveness"]["mean_response"] == pytest.approx(0.0140, abs=1e-4)
    assert blocks["variance_attribution"]["size_over_request_ratio"] == pytest.approx(
        2.22, abs=0.02)
    assert blocks["variance_attribution"]["monotone_in_request"] is False
    # the central claim: worse than a constant, on both bases
    assert blocks["reported"]["native_qed_mae"] > blocks["constant_floor"]["mae_floor"]
    assert blocks["cell_mean_mae"]["overall"] > blocks["constant_floor"]["mae_floor"]


def test_the_reported_input_records_that_it_is_unverified():
    reported = json.loads(REPORTED_PATH.read_text())
    assert "NOT YET RECOMPUTED" in reported["status"]
    assert reported["literature_reference_disagreement"]["moflow_raw_validity"][
        "status"].startswith("UNRESOLVED")


def test_the_moflow_disagreement_is_recorded_without_picking_a_side():
    reported = json.loads(REPORTED_PATH.read_text())
    block = reported["literature_reference_disagreement"]["moflow_raw_validity"]
    assert block["moflow_paper"] == pytest.approx(0.8176)
    assert block["graphdf_table1_reporting_moflow"] == pytest.approx(0.503)
    assert "Do not pick one" in block["status"]


def test_the_verdict_names_the_constant_baseline_explicitly():
    module = diagnostics()
    blocks = module.analyse(json.loads(REPORTED_PATH.read_text()))
    text = " ".join(blocks["verdict"]["findings"])
    assert "WORSE than always emitting" in text
    assert "does not demonstrate conditional control" in blocks["verdict"]["conclusion"]
    assert "recomputed from the evaluation artefacts" in blocks["verdict"]["caveat"]


# ----------------------------------------------------------------- CLI and purity ---
def test_main_returns_exit_codes_rather_than_raising(tmp_path):
    module = diagnostics()
    assert module.main(["--input", str(tmp_path / "absent.json"),
                        "--output-dir", str(tmp_path / "o")]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"qed_targets": [0.5]}))
    assert module.main(["--input", str(bad), "--output-dir", str(tmp_path / "o")]) == 4
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert module.main(["--input", str(broken),
                        "--output-dir", str(tmp_path / "o")]) == 2


def test_a_complete_run_writes_both_outputs(tmp_path):
    module = diagnostics()
    out = tmp_path / "out"
    assert module.main(["--input", str(REPORTED_PATH), "--output-dir", str(out)]) == 0
    blocks = json.loads((out / "conditioning_diagnostics.json").read_text())
    assert blocks["constant_floor"]["best_constant"] == pytest.approx(0.6)
    text = (out / "conditioning_diagnostics.txt").read_text()
    assert "BASELINES THE MODEL MUST BEAT" in text
    assert "IS IT SIZE OR IS IT THE REQUEST?" in text


def test_the_diagnostics_touch_no_model_or_dataset():
    tree = ast.parse(SCRIPT_PATH.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not imported & {"torch", "rdkit", "numpy", "zinc_gnf"}, imported
