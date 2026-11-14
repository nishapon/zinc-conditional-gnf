"""Tests for the conditioning sensitivity probe and the metric reproducibility audit.

The audit's arithmetic runs anywhere, because RDKit is imported lazily inside `score`.
The probe needs torch and skips without it.

The property that matters most here is the one that nearly went unnoticed: the
constant-predictor floor depends on the per-SAMPLE target weighting, and validity
collapses with N, so the cells contribute wildly unequal numbers of scored samples. A
floor computed with uniform weights can be far from the floor the evaluation's own
aggregation implies. These tests pin that behaviour so the number cannot drift back.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "scripts" / "40_condition_sensitivity_probe.py"
AUDIT_PATH = ROOT / "scripts" / "41_metric_reproducibility_audit.py"


def audit():
    spec = importlib.util.spec_from_file_location("metric_audit", AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probe():
    pytest.importorskip("torch")
    pytest.importorskip("rdkit")
    spec = importlib.util.spec_from_file_location("sensitivity_probe", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cell(*, n, target, scored, qed, candidates=100):
    return {"n_node": n, "target": target, "candidates": candidates,
            "scored": scored, "scored_fraction": scored / candidates,
            "samples": [{"qed": qed, "aromatic_rings": 1, "rings": 2,
                         "heavy_atoms": n, "aromatic_atom_fraction": 0.4,
                         "smiles": f"C{n}"} for _ in range(scored)]}


# ================================= the weighting subtlety =========================
def test_equal_counts_reproduce_the_uniform_floor():
    """With balanced cells the two floors must agree; otherwise the maths is wrong."""
    module = audit()
    cells = {f"c{i}": cell(n=15, target=t, scored=10, qed=0.55)
             for i, t in enumerate((0.4, 0.6, 0.8))}
    result = module.constant_predictor_audit(cells)
    assert result["sample_weighted_floor"]["mae"] == pytest.approx(0.4 / 3, abs=1e-3)
    assert result["uniform_weight_floor"]["mae"] == pytest.approx(0.4 / 3, abs=1e-9)
    assert result["weighting_matters"] is False


def test_unequal_counts_move_the_floor_and_the_shift_is_flagged():
    """IF the chosen decoder's scored counts are skewed, the floor moves with them.

    Whether they are skewed is counted, never inferred from raw validity: the QED metric
    runs on the corrected decoder, whose validity is high at every N.
    """
    module = audit()
    cells = {"big": cell(n=15, target=0.4, scored=96, qed=0.55),
             "small": cell(n=30, target=0.8, scored=4, qed=0.50)}
    result = module.constant_predictor_audit(cells)
    weights = result["target_weights_from_scored_samples"]
    assert weights["0.4"] == pytest.approx(0.96)
    assert result["sample_weighted_floor"]["mae"] < 0.05     # just predict 0.4
    assert result["uniform_weight_floor"]["mae"] == pytest.approx(0.2)
    assert result["weighting_matters"] is True


def test_the_exact_minimiser_is_the_weighted_median():
    module = audit()
    cells = {"a": cell(n=15, target=0.4, scored=70, qed=0.5),
             "b": cell(n=23, target=0.8, scored=30, qed=0.5)}
    result = module.constant_predictor_audit(cells)
    assert result["sample_weighted_floor"]["exact_weighted_median"] == pytest.approx(0.4)
    grid_best = result["sample_weighted_floor"]["mae"]
    assert result["sample_weighted_floor"]["mae_at_exact"] <= grid_best + 1e-9


def test_no_constant_on_the_grid_beats_the_reported_weighted_floor():
    module = audit()
    cells = {"a": cell(n=15, target=0.4, scored=40, qed=0.5),
             "b": cell(n=23, target=0.6, scored=35, qed=0.5),
             "c": cell(n=30, target=0.8, scored=5, qed=0.5)}
    result = module.constant_predictor_audit(cells)
    assert min(result["mae_by_constant"].values()) >= \
        result["sample_weighted_floor"]["mae"] - 1e-12


def test_the_model_mae_uses_the_same_sample_aggregation():
    module = audit()
    cells = {"a": cell(n=15, target=0.4, scored=3, qed=0.5)}
    result = module.constant_predictor_audit(cells)
    assert result["model_mae_recomputed"] == pytest.approx(0.1)
    assert result["samples"] == 3
    assert "SCORED SAMPLES" in result["aggregation"]


def test_a_model_that_beats_the_floor_is_reported_as_beating_it():
    module = audit()
    cells = {"a": cell(n=15, target=0.4, scored=10, qed=0.40),
             "b": cell(n=15, target=0.8, scored=10, qed=0.80)}
    result = module.constant_predictor_audit(cells)
    assert result["model_mae_recomputed"] == pytest.approx(0.0)
    assert result["model_beats_sample_weighted_floor"] is True


def test_no_scored_samples_is_reported_not_divided_by_zero():
    module = audit()
    result = module.constant_predictor_audit(
        {"a": {"n_node": 15, "target": 0.4, "candidates": 10, "scored": 0,
               "scored_fraction": 0.0, "samples": []}})
    assert result["samples"] == 0 and "no scored samples" in result["note"]


def test_weighted_median_minimises_absolute_deviation():
    module = audit()
    values = [0.4] * 7 + [0.8] * 3
    best = module.weighted_median(values)
    reference = sum(abs(best - v) for v in values) / len(values)
    for candidate in (0.3, 0.5, 0.6, 0.8):
        assert sum(abs(candidate - v) for v in values) / len(values) >= reference - 1e-12


# ===================================== aromaticity ================================
def test_aromaticity_is_reported_per_cell_and_per_n():
    module = audit()
    cells = {"N15_QED0.4": cell(n=15, target=0.4, scored=4, qed=0.5),
             "N30_QED0.8": cell(n=30, target=0.8, scored=2, qed=0.5)}
    result = module.aromaticity_audit(cells)
    assert set(result["by_cell"]) == {"N15_QED0.4", "N30_QED0.8"}
    assert set(result["by_n"]) == {"15", "30"}
    assert result["by_cell"]["N15_QED0.4"]["aromatic_rings_mean"] == pytest.approx(1.0)
    assert result["pooled_mean"] == pytest.approx(1.0)


def test_a_pooled_mean_cannot_hide_a_size_dependent_collapse():
    """The reason per-N reporting exists: identical pooled means, different stories."""
    module = audit()
    good = cell(n=15, target=0.4, scored=10, qed=0.5)
    bad = cell(n=30, target=0.4, scored=10, qed=0.5)
    for row in bad["samples"]:
        row["aromatic_rings"] = 0
    for row in good["samples"]:
        row["aromatic_rings"] = 2
    result = module.aromaticity_audit({"a": good, "b": bad})
    assert result["pooled_mean"] == pytest.approx(1.0)
    assert result["by_n"]["15"]["mean"] == pytest.approx(2.0)
    assert result["by_n"]["30"]["mean"] == pytest.approx(0.0)
    assert result["by_n"]["30"]["fraction_with_no_aromatic_ring"] == pytest.approx(1.0)


def test_the_histogram_covers_the_declared_bins():
    module = audit()
    result = module.aromaticity_audit({"a": cell(n=15, target=0.4, scored=3, qed=0.5)})
    assert set(result["by_cell"]["a"]["histogram"]) == {"0", "1", "2", "3", "4", "5"}


def test_the_real_reference_is_attached_when_supplied():
    module = audit()
    result = module.aromaticity_audit(
        {"a": cell(n=15, target=0.4, scored=2, qed=0.5)},
        {"15": {"molecules": 500, "mean": 1.88}})
    assert result["real_reference_by_n"]["15"]["mean"] == pytest.approx(1.88)


# ================================= audit CLI and purity ===========================
def test_the_audit_cell_naming_matches_the_generator():
    module = audit()
    assert module.cell_directory(Path("/x"), temperature=0.2, n=15, qed=0.4).name \
        == "T0.2_N15_QED0.4"


def test_the_audit_refuses_to_generate_and_names_save_samples(tmp_path):
    module = audit()
    code = module.main(["--samples-root", str(tmp_path), "--output-dir",
                        str(tmp_path / "out"), "--skip-real-reference",
                        "--target-n", "15", "--qed-values", "0.4,0.8"])
    assert code == 2


def test_the_audit_rejects_bad_arguments(tmp_path):
    module = audit()
    assert module.main(["--samples-root", str(tmp_path),
                        "--output-dir", str(tmp_path / "o"),
                        "--qed-values", "0.4"]) == 4


def test_the_audit_imports_no_model_stack_at_module_scope():
    tree = ast.parse(AUDIT_PATH.read_text())
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        if isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not imported & {"torch", "rdkit", "numpy", "zinc_gnf"}, imported


def test_the_audit_never_loads_a_checkpoint():
    calls = [ast.unparse(n.func) for n in ast.walk(ast.parse(AUDIT_PATH.read_text()))
             if isinstance(n, ast.Call)]
    for forbidden in ("load_flow", "load_autoencoder", "decode_embeddings",
                      "torch.save", "torch.load"):
        assert not any(forbidden in call for call in calls), forbidden


# ================================= the probe's verdicts ===========================
def test_the_probe_calls_a_sub_noise_response_a_failure():
    module = probe()
    assert module._verdict_snr(0.4).startswith("NO")
    assert module._verdict_snr(1.5).startswith("YES, weakly")
    assert module._verdict_snr(5.0).startswith("YES, strongly")
    assert module._verdict_snr(None) == "not measurable"


def test_the_probe_distinguishes_a_collapsed_decoder_from_an_orthogonal_response():
    module = probe()
    collapsed = module._verdict_transmission(latent=5.0, raw_delta=0.0, identity=0.01)
    assert collapsed.startswith("NO")
    assert "collapses" in collapsed
    orthogonal = module._verdict_transmission(latent=5.0, raw_delta=0.001, identity=0.9)
    assert orthogonal.startswith("PARTIALLY")
    assert "QED-orthogonal" in orthogonal
    carried = module._verdict_transmission(latent=5.0, raw_delta=0.2, identity=0.9)
    assert carried.startswith("YES")


def test_the_probe_reports_correction_attenuation_in_both_directions():
    module = probe()
    assert module._verdict_correction(0.20, 0.05).startswith("YES")
    assert module._verdict_correction(0.10, 0.30).startswith("IT AMPLIFIES")
    assert module._verdict_correction(0.20, 0.19).startswith("NO")
    assert module._verdict_correction(0.001, 0.001).startswith("NOTHING TO ERASE")


def test_the_probe_issues_no_causal_verdict_about_n_conditioning():
    """The two causal verdicts were unsupported and must stay deleted.

    A larger latent response to the N direction is a sensitivity, not evidence that N
    and QED compete for conditioner capacity. And mismatching the condition's N against
    the mask is an out-of-distribution probe: a null response cannot distinguish an
    unused channel from a saturated or badly extrapolating conditioner.
    """
    module = probe()
    assert not hasattr(module, "_verdict_dominance")
    assert not hasattr(module, "_verdict_redundancy")


def test_the_synthesis_labels_the_n_arm_as_out_of_distribution():
    module = probe()
    payload = {"by_n": {"15": {"arms": {
        "qed_low_to_high": {"latent_snr_vs_noise": 1.2,
                             "latent_rms_per_unit_condition": 0.01,
                             "decoded": {"raw": {"delta_qed_mean": 0.0,
                                                  "smiles_change_rate": 0.5,
                                                  "delta_heavy_atoms": 0.0},
                                          "degree": {"delta_qed_mean": 0.0}}},
        "condition_n_15_to_30": {"latent_snr_vs_noise": 3.0,
                                  "latent_rms_per_unit_condition": 0.05,
                                  "decoded": {"raw": {"delta_qed_mean": 0.0,
                                                       "smiles_change_rate": 0.02,
                                                       "delta_heavy_atoms": 0.0},
                                               "degree": {"delta_qed_mean": 0.0}}}}}}}
    synthesis = module.synthesise(payload)
    assert "n_dominates_the_conditional_representation" not in synthesis
    assert "explicit_n_conditioning_is_redundant" not in synthesis
    ood = synthesis["n_mismatch_ood_response"]["answer"]
    assert "OUT-OF-DISTRIBUTION" in ood
    assert "does NOT establish redundancy" in ood
    assert "does NOT license removing" in ood
    sensitivity = synthesis["latent_sensitivity_per_unit_condition"]
    assert "no causal claim" in sensitivity["answer"]
    assert "NOT evidence that the two inputs compete" in sensitivity["criterion"]


def test_the_probe_latent_change_is_zero_for_an_identical_pair():
    module = probe()
    import torch

    x = torch.randn(4, 6, 8)
    mask = torch.ones(4, 6, dtype=torch.bool)
    result = module.latent_change(x, x.clone(), mask)
    assert result["rms_per_element"] == pytest.approx(0.0)
    assert result["relative_to_embedding_norm"] == pytest.approx(0.0)


def test_the_probe_latent_change_ignores_masked_padding():
    module = probe()
    import torch

    a = torch.zeros(2, 5, 4)
    b = a.clone()
    b[:, 3:, :] = 99.0                       # only padding differs
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[:, :3] = True
    assert module.latent_change(a, b, mask)["rms_per_element"] == pytest.approx(0.0)


def test_the_probe_detects_a_consistent_translation():
    module = probe()
    import torch

    a = torch.randn(8, 5, 4)
    direction = torch.randn(1, 1, 4)
    result = module.latent_change(a, a + direction, torch.ones(8, 5, dtype=torch.bool))
    assert result["rms_per_element"] > 0
    assert result["direction_consistency"] == pytest.approx(1.0, abs=1e-4)


def test_decoded_change_uses_only_jointly_scorable_pairs():
    module = probe()
    before = {"raw": [{"scorable": True, "valid": True, "smiles": "CCO", "qed": 0.3,
                       "aromatic_rings": 0.0, "heavy_atoms": 3.0},
                      {"scorable": False, "valid": False, "smiles": None, "qed": None,
                       "aromatic_rings": None, "heavy_atoms": None}]}
    after = {"raw": [{"scorable": True, "valid": True, "smiles": "CCN", "qed": 0.5,
                      "aromatic_rings": 0.0, "heavy_atoms": 3.0},
                     {"scorable": True, "valid": True, "smiles": "CCC", "qed": 0.9,
                      "aromatic_rings": 0.0, "heavy_atoms": 3.0}]}
    result = module.decoded_change(before, after, decoder="raw")
    assert result["jointly_scorable"] == 1
    assert result["delta_qed_mean"] == pytest.approx(0.2)
    assert result["smiles_change_rate"] == pytest.approx(1.0)


def test_the_probe_is_read_only_with_respect_to_models():
    source = PROBE_PATH.read_text()
    assert "requires_grad_(False)" in source
    for forbidden in ("torch.optim", "zero_grad", "save_checkpoint", "backward()"):
        assert forbidden not in source, forbidden
    calls = [ast.unparse(n.func) for n in ast.walk(ast.parse(source))
             if isinstance(n, ast.Call)]
    assert not any("torch.save" in call for call in calls)


def test_the_probe_states_what_it_cannot_settle():
    source = " ".join(PROBE_PATH.read_text().split())
    assert "CANNOT: establish that the explicit N channel is redundant" in source
    assert "OUT-OF-DISTRIBUTION SENSITIVITY TEST AND NOTHING MORE" in source
    assert "explicitly declines to issue a recommendation about N conditioning" in source


def test_the_audit_weights_from_the_chosen_decoders_scored_counts():
    """Raw validity must not be used to infer a corrected-QED metric's weighting."""
    module = audit()
    # checked on runtime string values: adjacent-literal boundaries make raw source
    # matching unreliable
    import ast
    strings = [n.value for n in ast.walk(ast.parse(AUDIT_PATH.read_text()))
               if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    blob = " ".join(" ".join(value.split()) for value in strings)
    assert "SCORED BY THE DECODER NAMED IN --decoder" in blob
    assert "Raw validity must not be used to infer the weighting" in blob
    assert "That inference was invalid" in blob
    result = module.constant_predictor_audit(
        {"a": cell(n=15, target=0.4, scored=10, qed=0.5),
         "b": cell(n=23, target=0.6, scored=10, qed=0.5),
         "c": cell(n=30, target=0.8, scored=10, qed=0.5)})
    # balanced corrected-decoder counts -> the uniform floor is the right one
    assert result["weighting_matters"] is False
    assert result["sample_weighted_floor"]["mae"] == pytest.approx(
        result["uniform_weight_floor"]["mae"], abs=1e-3)
