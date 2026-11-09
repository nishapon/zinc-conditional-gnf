"""Tests for the final consolidation and statistical correction.

The consolidation exists to fix two defects, so the tests are mostly about those:

  * the model-to-model test must be on the DIFFERENCE between models. Two responses that
    both exclude zero can be statistically indistinguishable from each other, and the
    original `response_improved` flag would have called that an improvement.
  * when the per-molecule deltas were not persisted the bound must be CONSERVATIVE and
    the non-significant outcome must be labelled INCONCLUSIVE, never "no difference".

Pure JSON and arithmetic; torch is needed only for the checkpoint audit, which is
imported lazily and skipped when no checkpoint is supplied.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "37_final_consolidation.py"


def analysis():
    spec = importlib.util.spec_from_file_location("final_consolidation", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def response(mean, half_width, *, pairs=100, deltas=None):
    block = {"mean": mean, "ci95": [mean - half_width, mean + half_width],
             "jointly_valid_pairs": pairs, "excludes_zero": bool(
                 mean - half_width > 0 or mean + half_width < 0)}
    if deltas is not None:
        block["paired_deltas"] = deltas
    return block


def evaluation_payload(candidate, reference, *, n="23", settings=None):
    def model(block, step, validation):
        return {"step": step, "checkpoint": f"/x/{step}.pt",
                "real_validation": validation,
                "by_n": {n: {"paired_response": {"degree": block},
                             "noise_seed": 777000111 + 7919 * int(n),
                             "cells": {}}}}
    return {"settings": settings or {"samples_per_cell": 100, "temperature": 0.2},
            "models": {
                "finetuned": model(candidate, 400, {"val_loss": 1.23,
                                                     "val_condition_gap": 0.01}),
                "final": model(reference, 50000, {"val_loss": 1.20,
                                                   "val_condition_gap": 0.02})}}


# ====================================================================================
# 1. The statistical correction.
# ====================================================================================
def test_standard_error_is_recovered_from_a_symmetric_interval():
    module = analysis()
    # a 95% interval of half-width 1.96*se
    se = module.standard_error_from_ci([0.10 - 1.959963984540054 * 0.05,
                                        0.10 + 1.959963984540054 * 0.05])
    assert se == pytest.approx(0.05, rel=1e-9)


def test_standard_error_is_none_for_a_missing_interval():
    module = analysis()
    assert module.standard_error_from_ci(None) is None
    assert module.standard_error_from_ci([None, None]) is None
    assert module.standard_error_from_ci([]) is None


def test_two_models_whose_responses_both_exclude_zero_can_be_indistinguishable():
    """The exact defect in the original verdict, demonstrated numerically."""
    module = analysis()
    candidate = response(0.25, 0.10)          # excludes zero
    reference = response(0.20, 0.10)          # excludes zero
    assert candidate["excludes_zero"] and reference["excludes_zero"]
    test = module.paired_difference_conservative(candidate, reference)
    assert test["mean"] == pytest.approx(0.05)
    assert test["significant"] is False
    assert test["inconclusive"] is True


def test_a_genuinely_large_difference_is_significant_even_under_the_bound():
    module = analysis()
    test = module.paired_difference_conservative(response(0.60, 0.05),
                                                 response(0.10, 0.05))
    assert test["significant"] is True
    assert test["inconclusive"] is False
    assert test["ci95"][0] > 0.0


def test_the_bound_is_conservative_and_says_so():
    module = analysis()
    test = module.paired_difference_conservative(response(0.3, 0.08),
                                                 response(0.2, 0.06))
    expected = math.sqrt((0.08 / 1.959963984540054) ** 2
                         + (0.06 / 1.959963984540054) ** 2)
    assert test["standard_error_upper_bound"] == pytest.approx(expected, rel=1e-9)
    assert "UPPER bound" in test["note"]
    assert "positively correlated" in test["note"]
    assert "INCONCLUSIVE" in test["note"]


def test_a_missing_marginal_makes_the_result_inconclusive_not_significant():
    module = analysis()
    test = module.paired_difference_conservative({"mean": None, "ci95": None},
                                                 response(0.2, 0.05))
    assert test["significant"] is False and test["inconclusive"] is True
    assert test["mean"] is None


def test_the_exact_paired_test_is_used_when_deltas_are_present():
    module = analysis()
    candidate = response(0.5, 0.1, deltas=[0.5] * 40)
    reference = response(0.2, 0.1, deltas=[0.2] * 40)
    test = module.paired_difference_exact(candidate["paired_deltas"],
                                          reference["paired_deltas"])
    assert test["method"] == "exact_paired"
    assert test["mean"] == pytest.approx(0.3)
    assert test["pairs"] == 40
    assert test["significant"] is True


def test_the_exact_paired_test_detects_a_null_difference():
    module = analysis()
    candidate = [0.1, 0.5, 0.3, 0.2, 0.4] * 8
    reference = [0.5, 0.1, 0.2, 0.4, 0.3] * 8
    test = module.paired_difference_exact(candidate, reference)
    assert test["mean"] == pytest.approx(0.0, abs=1e-9)
    assert test["significant"] is False


def test_the_exact_paired_test_refuses_too_few_pairs():
    module = analysis()
    test = module.paired_difference_exact([0.3], [0.1])
    assert test["pairs"] == 1 and test["significant"] is False
    assert test["mean"] is None


def test_the_exact_paired_test_drops_pairs_with_a_missing_side():
    module = analysis()
    test = module.paired_difference_exact([0.3, None, 0.4, 0.5],
                                          [0.1, 0.2, None, 0.2])
    assert test["pairs"] == 2


def test_two_sided_p_matches_known_normal_values():
    module = analysis()
    assert module.two_sided_p(0.0) == pytest.approx(1.0)
    assert module.two_sided_p(1.959963984540054) == pytest.approx(0.05, abs=1e-6)
    assert module.two_sided_p(2.5758293035489) == pytest.approx(0.01, abs=1e-6)


def test_holm_adjustment_controls_the_family_and_is_monotone():
    module = analysis()
    adjusted = module.holm_adjust({"15": 0.01, "23": 0.04, "30": 0.60})
    assert adjusted["15"] == pytest.approx(0.03)
    assert adjusted["23"] == pytest.approx(0.08)
    assert adjusted["30"] == pytest.approx(0.60)
    assert adjusted["15"] <= adjusted["23"] <= adjusted["30"]


def test_holm_adjustment_caps_at_one_and_keeps_missing_keys():
    module = analysis()
    adjusted = module.holm_adjust({"a": 0.5, "b": 0.9, "c": None})
    assert adjusted["a"] <= 1.0 and adjusted["b"] <= 1.0
    assert adjusted["c"] is None


# ====================================================================================
# 2. The comparison replaces the invalid flag.
# ====================================================================================
def test_the_comparison_marks_the_original_flag_invalid():
    module = analysis()
    payload = evaluation_payload(response(0.25, 0.10), response(0.20, 0.10))
    responses = module.extract_responses(payload, decoder="degree")
    comparison = module.compare_models(responses, candidate="finetuned",
                                       reference="final")
    test = comparison["by_n"]["23"]
    assert "INVALID" in test["original_flag_response_improved"]
    assert "against zero" in test["original_flag_response_improved"]


def test_the_comparison_reports_not_established_when_nothing_is_significant():
    module = analysis()
    payload = evaluation_payload(response(0.25, 0.10), response(0.20, 0.10))
    responses = module.extract_responses(payload, decoder="degree")
    comparison = module.compare_models(responses, candidate="finetuned",
                                       reference="final")
    assert comparison["conclusion"] == "IMPROVEMENT NOT ESTABLISHED"
    assert comparison["inconclusive_n"] == ["23"]
    assert comparison["significant_n"] == []


def test_the_comparison_reports_a_real_improvement_when_the_data_support_it():
    module = analysis()
    payload = evaluation_payload(response(0.60, 0.04), response(0.10, 0.04))
    responses = module.extract_responses(payload, decoder="degree")
    comparison = module.compare_models(responses, candidate="finetuned",
                                       reference="final")
    assert comparison["significant_n"] == ["23"]
    assert "improvement established" in comparison["conclusion"]


def test_the_comparison_needs_both_models():
    module = analysis()
    payload = evaluation_payload(response(0.3, 0.05), response(0.2, 0.05))
    responses = module.extract_responses(payload, decoder="degree")
    comparison = module.compare_models(responses, candidate="absent",
                                       reference="final")
    assert comparison["evaluated"] is False
    assert "absent" in comparison["note"]


def test_the_comparison_prefers_exact_deltas_when_both_models_have_them():
    module = analysis()
    payload = evaluation_payload(response(0.4, 0.1, deltas=[0.4] * 30),
                                 response(0.1, 0.1, deltas=[0.1] * 30))
    responses = module.extract_responses(payload, decoder="degree")
    comparison = module.compare_models(responses, candidate="finetuned",
                                       reference="final")
    assert comparison["by_n"]["23"]["method"] == "exact_paired"


def test_the_comparison_falls_back_when_only_one_side_has_deltas():
    module = analysis()
    payload = evaluation_payload(response(0.4, 0.1, deltas=[0.4] * 30),
                                 response(0.1, 0.1))
    responses = module.extract_responses(payload, decoder="degree")
    comparison = module.compare_models(responses, candidate="finetuned",
                                       reference="final")
    assert comparison["by_n"]["23"]["method"] == "conservative_independent_bound"


def test_extract_responses_keeps_the_seed_and_pair_count():
    module = analysis()
    payload = evaluation_payload(response(0.3, 0.05, pairs=87), response(0.2, 0.05))
    responses = module.extract_responses(payload, decoder="degree")
    entry = responses["finetuned"]["by_n"]["23"]
    assert entry["jointly_valid_pairs"] == 87
    assert entry["noise_seed"] == 777000111 + 7919 * 23
    assert responses["final"]["step"] == 50000


# ====================================================================================
# 3. The definitive table keeps the protocol explicit.
# ====================================================================================
def test_the_table_has_one_row_per_checkpoint_and_n():
    module = analysis()
    payload = evaluation_payload(response(0.25, 0.10), response(0.20, 0.10))
    responses = module.extract_responses(payload, decoder="degree")
    comparison = module.compare_models(responses, candidate="finetuned",
                                       reference="final")
    table = module.definitive_table(responses, comparison,
                                    settings=payload["settings"], decoder="degree")
    assert len(table["rows"]) == 2
    assert {row["checkpoint"] for row in table["rows"]} == {"finetuned", "final"}


def test_only_the_candidate_row_carries_the_difference():
    module = analysis()
    payload = evaluation_payload(response(0.25, 0.10), response(0.20, 0.10))
    responses = module.extract_responses(payload, decoder="degree")
    comparison = module.compare_models(responses, candidate="finetuned",
                                       reference="final")
    table = module.definitive_table(responses, comparison,
                                    settings=payload["settings"], decoder="degree")
    rows = {row["checkpoint"]: row for row in table["rows"]}
    assert rows["finetuned"]["vs_reference_difference"] is not None
    assert rows["final"]["vs_reference_difference"] is None


def test_the_table_records_the_protocol_and_the_seed_caveat():
    module = analysis()
    payload = evaluation_payload(response(0.25, 0.10), response(0.20, 0.10))
    responses = module.extract_responses(payload, decoder="degree")
    comparison = module.compare_models(responses, candidate="finetuned",
                                       reference="final")
    table = module.definitive_table(responses, comparison,
                                    settings=payload["settings"], decoder="degree")
    assert "jointly valid pairs only" in table["protocol_note"]
    assert "must not be pooled" in table["seed_note"]
    for row in table["rows"]:
        assert row["temperature"] == 0.2
        assert row["samples_per_cell"] == 100
        assert row["decoder"] == "degree"
        assert "shared prior draw" in row["pairing"]


def test_the_markdown_table_renders_without_crashing_on_missing_values():
    module = analysis()
    table = {"rows": [{"checkpoint": "early", "step": None, "n": 15,
                       "paired_response_mean": None, "paired_response_ci95": [None, None],
                       "jointly_valid_pairs": None, "vs_reference_difference": None,
                       "vs_reference_ci95": None, "vs_reference_method": None,
                       "vs_reference_significant": None, "real_validation_nll": None,
                       "condition_gap": None}],
             "protocol_note": "x", "seed_note": "y"}
    rendered = module.markdown_table(table)
    assert rendered.count("\n") == 2            # header, divider, one row
    assert "n/a" in rendered


# ====================================================================================
# 4. Checkpoint identity, including the absent-best.pt case.
# ====================================================================================
def test_an_absent_best_checkpoint_is_explained_not_treated_as_an_error(tmp_path):
    """An absent best.pt means real validation NLL never improved. That is a finding."""
    module = analysis()
    audit = module.checkpoint_audit({"best": tmp_path / "best.pt",
                                     "last": tmp_path / "last.pt"})
    assert audit["absent"] == ["best", "last"]
    assert "never improved real validation likelihood" in audit["best_absent_meaning"]
    assert "describe last.pt" in audit["best_absent_meaning"]


def test_the_audit_reports_no_readable_checkpoint_rather_than_crashing(tmp_path):
    module = analysis()
    audit = module.checkpoint_audit({"last": tmp_path / "absent.pt"})
    assert audit["present"] == []
    assert "nothing to compare" in audit["note"]


def test_identical_checkpoints_are_detected_as_identical(tmp_path):
    torch = pytest.importorskip("torch")
    module = analysis()
    state = {"w": torch.ones(3), "b": torch.zeros(2)}
    for name in ("best.pt", "last.pt"):
        torch.save({"model_state_dict": state, "step": 400,
                    "relabel_finetune": {"parent_step": 50000,
                                          "label_decoder": "degree",
                                          "relabel_weight": 0.5}},
                   tmp_path / name)
    audit = module.checkpoint_audit({"best": tmp_path / "best.pt",
                                     "last": tmp_path / "last.pt"})
    assert sorted(audit["present"]) == ["best", "last"]
    comparison = audit["best_vs_last"]
    assert comparison["identical"] is True
    assert comparison["max_absolute_weight_difference"] == 0.0
    assert audit["checkpoints"]["best"]["parameters"] == 5


def test_differing_checkpoints_report_where_they_differ(tmp_path):
    torch = pytest.importorskip("torch")
    module = analysis()
    torch.save({"model_state_dict": {"w": torch.ones(3)}, "step": 100},
               tmp_path / "best.pt")
    torch.save({"model_state_dict": {"w": torch.ones(3) * 1.25}, "step": 400},
               tmp_path / "last.pt")
    audit = module.checkpoint_audit({"best": tmp_path / "best.pt",
                                     "last": tmp_path / "last.pt"})
    comparison = audit["best_vs_last"]
    assert comparison["identical"] is False
    assert comparison["max_absolute_weight_difference"] == pytest.approx(0.25)
    assert comparison["largest_difference_in"] == "w"
    assert audit["checkpoints"]["best"]["step"] == 100
    assert audit["checkpoints"]["last"]["step"] == 400


def test_the_audit_surfaces_the_relabel_provenance(tmp_path):
    torch = pytest.importorskip("torch")
    module = analysis()
    torch.save({"model_state_dict": {"w": torch.zeros(1)}, "step": 400,
                "relabel_finetune": {"parent_step": 50000, "label_decoder": "degree",
                                      "relabel_weight": 0.5,
                                      "real_validation": {"val_loss": 1.3}}},
               tmp_path / "last.pt")
    audit = module.checkpoint_audit({"last": tmp_path / "last.pt"})
    block = audit["checkpoints"]["last"]
    assert block["parent_step"] == 50000
    assert block["label_decoder"] == "degree"
    assert block["real_validation_at_save"]["val_loss"] == 1.3


# ====================================================================================
# 5. CLI contract and read-only guarantees.
# ====================================================================================
def test_main_returns_an_exit_code_rather_than_raising(tmp_path):
    module = analysis()
    assert module.main(["--evaluation", str(tmp_path / "absent.json"),
                        "--output-dir", str(tmp_path / "out")]) == 2


def test_bad_arguments_are_rejected_with_code_four(tmp_path):
    module = analysis()
    base = ["--evaluation", str(tmp_path / "e.json"), "--output-dir", str(tmp_path / "o")]
    assert module.main(base + ["--checkpoint", "nopath"]) == 4
    assert module.main(base + ["--artifact", "nopath"]) == 4
    assert module.main(base + ["--candidate", "x", "--reference", "x"]) == 4


def test_an_evaluation_without_models_is_exit_two(tmp_path):
    module = analysis()
    path = tmp_path / "e.json"
    path.write_text(json.dumps({"models": {}}))
    assert module.main(["--evaluation", str(path),
                        "--output-dir", str(tmp_path / "out")]) == 2


def test_a_complete_run_writes_all_three_outputs(tmp_path):
    module = analysis()
    path = tmp_path / "relabel_evaluation.json"
    path.write_text(json.dumps(evaluation_payload(response(0.25, 0.10),
                                                   response(0.20, 0.10))))
    out = tmp_path / "out"
    assert module.main(["--evaluation", str(path), "--output-dir", str(out),
                        "--skip-checkpoint-audit"]) == 0
    payload = json.loads((out / "final_consolidation.json").read_text())
    assert payload["model_comparison"]["conclusion"] == "IMPROVEMENT NOT ESTABLISHED"
    assert sorted(payload["models_found"]) == ["final", "finetuned"]
    table = (out / "definitive_table.md").read_text()
    assert table.startswith("| checkpoint |")
    report = (out / "final_consolidation.txt").read_text()
    assert "CORRECTED MODEL-TO-MODEL COMPARISON" in report
    assert "INCONCLUSIVE" in report


def test_a_missing_artifact_is_recorded_not_fatal(tmp_path):
    module = analysis()
    path = tmp_path / "e.json"
    path.write_text(json.dumps(evaluation_payload(response(0.3, 0.05),
                                                   response(0.2, 0.05))))
    out = tmp_path / "out"
    assert module.main(["--evaluation", str(path), "--output-dir", str(out),
                        "--artifact", f"acp_gate={tmp_path / 'nope.json'}",
                        "--skip-checkpoint-audit"]) == 0
    payload = json.loads((out / "final_consolidation.json").read_text())
    assert payload["artifacts"]["acp_gate"]["present"] is False


def test_the_script_trains_nothing_and_generates_nothing():
    calls = [ast.unparse(n.func) for n in ast.walk(ast.parse(SCRIPT_PATH.read_text()))
             if isinstance(n, ast.Call)]
    for forbidden in ("torch.save", "decode_embeddings", "load_flow",
                      "load_autoencoder", "backward", "inverse"):
        assert not any(forbidden in call for call in calls), forbidden
    source = SCRIPT_PATH.read_text()
    for forbidden in ("torch.optim", "zero_grad", "flow.inverse", "save_checkpoint"):
        assert forbidden not in source, forbidden


def test_torch_is_imported_lazily_so_the_statistics_run_without_it():
    tree = ast.parse(SCRIPT_PATH.read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(not a.name.startswith("torch") for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("torch")
    hosts = {function.name for function in ast.walk(tree)
             if isinstance(function, ast.FunctionDef)
             for sub in ast.walk(function)
             if isinstance(sub, ast.Import)
             and any(a.name == "torch" for a in sub.names)}
    assert hosts == {"checkpoint_audit"}


def test_the_report_states_the_limits_of_the_evidence():
    source = " ".join(SCRIPT_PATH.read_text().split())
    assert "absence of evidence here is not evidence of equality" in source
    assert "INVALID wherever it appears" in source
    assert "CANNOT be computed from the saved artefacts" in source
