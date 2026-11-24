"""Logic tests for the cluster pipeline's shared helpers.

These run WITHOUT torch, numpy or rdkit by design: the functions under test operate on
saved records, which is what lets the summariser run on a login node. The numpy-dependent
functions (`percentile_targets`, `grid_target_support`, `joint_association`) are skipped
automatically where numpy is absent.

What is checked here is exactly what would be hardest to notice if it were wrong:
denominators, failed attempts surviving into the yield, macro vs pooled weighting,
duplicate preservation, reproducible size sequences, and the median-vs-mean property of
the diagnostic reference.
"""

from __future__ import annotations

import json

import pytest

from zinc_gnf.eval_support import (
    COMPLETE,
    FixedCounts,
    artefact_fingerprint,
    completed_artefact,
    aggregate_runs,
    already_done,
    constant_reference,
    macro_and_pooled,
    manifest_entry,
    qed_cell_metrics,
    read_jsonl,
    response_fit,
    run_level_summary,
    write_jsonl,
)


def attempt(*, valid, qed=None, smiles=None, heavy=None, decoder="degree"):
    """One saved attempt. An invalid attempt still occupies a row."""
    entry = {"valid": bool(valid), "smiles": smiles, "qed": qed, "heavy_atoms": heavy}
    return {"n_node": 23, "requested_qed": 0.6, decoder: entry}


# ============================== denominators ==================================
def test_failed_attempts_stay_in_every_per_attempt_denominator():
    """Eight attempts, two valid, one on target. Nothing may silently disappear."""
    records = [attempt(valid=False) for _ in range(6)]
    records += [attempt(valid=True, qed=0.62, smiles="CCO", heavy=23),
                attempt(valid=True, qed=0.20, smiles="CCN", heavy=23)]
    metrics = qed_cell_metrics(records, requested_qed=0.6, requested_n=23,
                               train_smiles=set())
    assert metrics["attempted"] == 8
    assert metrics["valid"] == 2
    assert metrics["validity_over_attempts"] == pytest.approx(2 / 8)
    # MAE and hit rate are over SCORED molecules
    assert metrics["qed_mae"] == pytest.approx((0.02 + 0.40) / 2)
    assert metrics["hit_rate_within_0p1_among_scored"] == pytest.approx(1 / 2)
    # the yield is over ALL attempts, so the six failures count against it
    assert metrics["valid_hit_yield_within_0p1_over_all_attempts"] == pytest.approx(1 / 8)
    assert metrics["valid_hit_yield_within_0p01_over_all_attempts"] == 0.0


def test_mae_is_the_mean_of_individual_errors_not_the_error_of_the_mean():
    """Two molecules straddling the target: the cell mean looks perfect, MAE does not."""
    records = [attempt(valid=True, qed=0.4, smiles="A", heavy=23),
               attempt(valid=True, qed=0.8, smiles="B", heavy=23)]
    metrics = qed_cell_metrics(records, requested_qed=0.6, requested_n=23,
                               train_smiles=set())
    assert metrics["achieved_qed_mean"] == pytest.approx(0.6)
    assert metrics["qed_bias"] == pytest.approx(0.0)
    assert metrics["cell_mean_abs_error"] == pytest.approx(0.0)
    assert metrics["qed_mae"] == pytest.approx(0.2)      # the honest number


def test_duplicates_are_preserved_in_accuracy_but_not_in_uniqueness():
    records = [attempt(valid=True, qed=0.6, smiles="CCO", heavy=23) for _ in range(4)]
    records.append(attempt(valid=True, qed=0.1, smiles="CCN", heavy=23))
    metrics = qed_cell_metrics(records, requested_qed=0.6, requested_n=23,
                              train_smiles={"CCO"})
    assert metrics["scored"] == 5                        # duplicates kept
    assert metrics["qed_mae"] == pytest.approx((0 + 0 + 0 + 0 + 0.5) / 5)
    assert metrics["duplicates_preserved"] is True
    # only uniqueness and novelty deduplicate
    assert metrics["unique_valid"] == 2
    assert metrics["uniqueness_over_valid"] == pytest.approx(2 / 5)
    assert metrics["novel_unique_valid"] == 1            # CCO is in train
    assert metrics["novelty_over_unique_valid"] == pytest.approx(1 / 2)
    assert metrics["nuv_over_attempts"] == pytest.approx(1 / 5)


def test_size_match_is_reported_over_valid_and_over_all_attempts():
    records = [attempt(valid=True, qed=0.6, smiles="A", heavy=23),
               attempt(valid=True, qed=0.6, smiles="B", heavy=22),
               attempt(valid=False)]
    metrics = qed_cell_metrics(records, requested_qed=0.6, requested_n=23,
                              train_smiles=set())
    assert metrics["size_match_among_valid"] == pytest.approx(1 / 2)
    assert metrics["size_match_over_all_attempts"] == pytest.approx(1 / 3)
    assert metrics["size_is_structural"] is True


def test_a_cell_with_no_valid_molecule_reports_zero_yield_not_a_missing_row():
    metrics = qed_cell_metrics([attempt(valid=False) for _ in range(5)],
                               requested_qed=0.6, requested_n=30, train_smiles=set())
    assert metrics["attempted"] == 5 and metrics["valid"] == 0
    assert metrics["qed_mae"] is None
    assert metrics["valid_hit_yield_within_0p1_over_all_attempts"] == 0.0


# ============================ macro vs pooled =================================
def test_macro_and_pooled_differ_when_validity_differs_across_cells():
    """Pooling shifts the answer toward whichever cell produced more molecules."""
    cells = [
        qed_cell_metrics([attempt(valid=True, qed=0.45, smiles=f"A{i}", heavy=15)
                          for i in range(90)] + [attempt(valid=False)] * 10,
                         requested_qed=0.4, requested_n=15, train_smiles=set()),
        qed_cell_metrics([attempt(valid=True, qed=0.50, smiles="B", heavy=30)] * 2
                         + [attempt(valid=False)] * 98,
                         requested_qed=0.8, requested_n=30, train_smiles=set()),
    ]
    both = macro_and_pooled(cells)
    macro, pooled = both["macro"], both["pooled"]
    # macro: (0.05 + 0.30)/2 ; pooled: weighted by 90 and 2 scored molecules
    assert macro["qed_mae"] == pytest.approx((0.05 + 0.30) / 2)
    assert pooled["qed_mae"] == pytest.approx((0.05 * 90 + 0.30 * 2) / 92)
    assert pooled["qed_mae"] < macro["qed_mae"]
    assert "equal weight" in macro["weighting"]
    assert "small-N" in pooled["weighting"]


# ========================= repetitions and aggregation ========================
def test_run_level_summary_uses_the_population_sd():
    values = [0.10, 0.12, 0.14, 0.11, 0.13]
    mean = sum(values) / len(values)
    expected = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    summary = run_level_summary(values)
    assert summary["mean"] == pytest.approx(mean)
    assert summary["sd"] == pytest.approx(expected)
    assert summary["runs"] == 5
    assert run_level_summary([0.5])["sd"] == 0.0
    assert run_level_summary([])["runs"] == 0


def test_aggregate_runs_labels_the_variability_as_sampling_only():
    def run(raw, degree):
        return {"metrics": {"decoders": {
                    "raw": {"strict_validity": raw},
                    "degree": {"strict_validity": degree, "uniqueness": 0.99,
                               "novelty_vs_train": 1.0, "nuv": degree * 0.99}}},
                "sizes": {"atom_count_match_over_all_attempts": 0.95}}
    summary = aggregate_runs([run(0.13, 0.97), run(0.15, 0.99)], "degree")
    assert summary["raw_validity"]["mean"] == pytest.approx(0.14)
    assert summary["raw_validity"]["sd"] == pytest.approx(0.01)
    assert "no training-seed variability" in summary["note"]


# ============================ reproducible sizes ==============================
def test_fixed_counts_returns_the_predrawn_sequence_in_order():
    counts = [15] * 3 + [23] * 2
    sampler = FixedCounts(counts)
    assert list(sampler.sample(2)) == [15, 15]
    assert list(sampler.sample(3)) == [15, 23, 23]
    with pytest.raises(ValueError, match="only 5 were pre-drawn"):
        sampler.sample(1)
    # two instances over the same request give byte-identical vectors, which is what
    # makes the two arms comparable
    assert list(FixedCounts(counts).sample(5)) == list(FixedCounts(counts).sample(5))


# ========================== diagnostic references =============================
def test_the_optimal_constant_is_the_median_not_the_mean_under_uneven_weights():
    """The correction that matters: median and mean coincide only by symmetry."""
    symmetric = constant_reference([0.4, 0.6, 0.8])
    assert symmetric["optimal_constant"] == pytest.approx(0.6, abs=1e-3)
    assert symmetric["mae_at_optimal_constant"] == pytest.approx(0.13333, abs=1e-4)
    assert symmetric["weighted_median_of_targets"] == pytest.approx(0.6)
    assert symmetric["weighted_mean_of_targets"] == pytest.approx(0.6)

    skewed = constant_reference([0.4, 0.6, 0.8], weights=[0.6, 0.3, 0.1])
    assert skewed["weighted_median_of_targets"] == pytest.approx(0.4)
    assert skewed["mae_at_weighted_median"] == pytest.approx(0.10, abs=1e-6)
    assert skewed["weighted_mean_of_targets"] == pytest.approx(0.5)
    # the mean is strictly worse, so quoting it as "the" reference understates the bar
    assert skewed["mae_at_weighted_mean"] > skewed["mae_at_weighted_median"]


def test_spread_raises_the_reference_and_it_is_never_called_a_lower_bound():
    sharp = constant_reference([0.4, 0.6, 0.8], sigma=0.0)
    wide = constant_reference([0.4, 0.6, 0.8], sigma=0.20)
    assert wide["mae_at_optimal_constant"] > sharp["mae_at_optimal_constant"]
    assert "not a lower bound" in sharp["kind"]
    assert "never appear as a model row" in sharp["note"]


def test_response_fit_recovers_a_known_line_and_reports_uncertainty():
    requested = [0.4, 0.4, 0.6, 0.6, 0.8, 0.8]
    achieved = [0.5 + 0.25 * r for r in requested]
    fit = response_fit(requested, achieved)
    assert fit["slope"] == pytest.approx(0.25)
    assert fit["intercept"] == pytest.approx(0.5)
    assert fit["slope_se"] == pytest.approx(0.0, abs=1e-9)
    assert "does not prove low error" in fit["note"]
    assert response_fit([0.5] * 4, [0.5] * 4)["slope"] is None   # no request variation


# ============================ storage / resume ================================
def test_saved_attempts_round_trip_and_completed_work_is_not_redone(tmp_path):
    rows = [attempt(valid=False), attempt(valid=True, qed=0.6, smiles="CCO", heavy=23)]
    path = tmp_path / "cell" / "samples.jsonl"
    assert write_jsonl(path, rows) == 2
    assert not list(path.parent.glob("*.partial"))     # atomic write left no debris
    back = read_jsonl(path)
    assert len(back) == 2 and back[0]["degree"]["valid"] is False
    assert already_done(path) is True
    assert already_done(path, force=True) is False
    assert already_done(tmp_path / "missing.jsonl") is False


def test_the_manifest_requires_full_provenance():
    with pytest.raises(ValueError, match="missing"):
        manifest_entry(flow_checkpoint="a.pt")
    entry = manifest_entry(
        flow_checkpoint="a.pt", flow_sha256="x" * 64, decoder_checkpoint="ae.pt",
        decoder_sha256="y" * 64, temperature=0.2, sampling_seed=7, attempts=2000,
        repetitions=5, decoder="degree")
    assert entry["training_seed"] is None                # explicit, not omitted
    assert "did not record it" in entry["training_seed_note"]
    assert json.loads(json.dumps(entry))                 # serialisable


# ============================ numpy-dependent parts ===========================
def test_target_support_marks_thin_corners():
    np = pytest.importorskip("numpy")
    from zinc_gnf.eval_support import grid_target_support, joint_association

    # QED falls with size, as in ZINC: the (30, 0.8) corner must come out unsupported
    counts = np.asarray([15] * 400 + [30] * 400)
    qed = np.asarray([0.45] * 400 + [0.50] * 400)
    rows = grid_target_support(counts, qed, min_support=50)
    corner = next(r for r in rows if r["n"] == 30 and r["qed"] == 0.8)
    assert corner["support"] == 0 and corner["extrapolation"] is True
    supported = next(r for r in rows if r["n"] == 15 and r["qed"] == 0.4)
    assert supported["support"] > 0
    association = joint_association(counts, qed)
    assert association["spearman_n_qed"] > 0.0           # not independent
    assert "eight components" in association["note"]


# ===================== the two fixes from the cluster run =====================
def test_fixed_counts_exposes_mean_like_the_real_sampler():
    """generation.py logs `node_counts.mean`, so its absence crashed AFTER generating."""
    sampler = FixedCounts([15, 23, 30, 23])
    assert sampler.mean == pytest.approx((15 + 23 + 30 + 23) / 4)
    assert FixedCounts([]).mean == 0.0
    assert sampler.state_dict()["sampler"] == "FixedCounts"
    # the attribute the real NodeCountSampler also exposes
    assert sampler.values == [15, 23, 30]


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_resume_requires_completion_budget_and_matching_settings(tmp_path):
    """File presence alone must NOT count as done."""
    metrics = tmp_path / "metrics.json"
    samples = tmp_path / "samples.jsonl"
    fingerprint = artefact_fingerprint(flow_sha256="a", temperature=0.2, seed=1)
    write_jsonl(samples, [attempt(valid=False) for _ in range(10)])

    # 1. a crashed job left no status
    _write(metrics, {"attempted": 10})
    assert completed_artefact(metrics, expected_attempts=10,
                              fingerprint=fingerprint, samples_path=samples
                              )["reuse"] is False

    # 2. complete and matching -> reused
    _write(metrics, {"status": COMPLETE, "attempted": 10,
                     "artefact_fingerprint": fingerprint})
    good = completed_artefact(metrics, expected_attempts=10, fingerprint=fingerprint,
                              samples_path=samples)
    assert good["reuse"] is True and good["payload"]["attempted"] == 10

    # 3. a bigger budget is now requested -> recompute
    assert completed_artefact(metrics, expected_attempts=2000,
                              fingerprint=fingerprint,
                              samples_path=samples)["reuse"] is False

    # 4. a different checkpoint or temperature -> recompute
    other = artefact_fingerprint(flow_sha256="b", temperature=0.2, seed=1)
    decision = completed_artefact(metrics, expected_attempts=10, fingerprint=other,
                                  samples_path=samples)
    assert decision["reuse"] is False and "fingerprint" in decision["reason"]

    # 5. metrics say 10 but samples.jsonl holds fewer -> recompute
    write_jsonl(samples, [attempt(valid=False) for _ in range(4)])
    decision = completed_artefact(metrics, expected_attempts=10,
                                  fingerprint=fingerprint, samples_path=samples)
    assert decision["reuse"] is False and "4 rows" in decision["reason"]

    # 6. --force always recomputes
    assert completed_artefact(metrics, expected_attempts=10, fingerprint=fingerprint,
                              force=True)["reuse"] is False


def test_fingerprint_changes_with_any_setting_and_is_order_independent():
    base = dict(flow_sha256="a", temperature=0.2, seed=1, decoder="degree")
    assert artefact_fingerprint(**base) == artefact_fingerprint(
        **{k: base[k] for k in reversed(list(base))})
    for key, value in (("temperature", 0.85), ("seed", 2), ("decoder", "raw"),
                       ("flow_sha256", "b")):
        assert artefact_fingerprint(**{**base, key: value}) != artefact_fingerprint(**base)


def test_arm_labels_name_the_contrastive_objective():
    from zinc_gnf.eval_support import ARM_LABELS

    assert "contrastive" in ARM_LABELS["conditional"]
    assert "NLL" in ARM_LABELS["baseline"]
    assert "NOT a FiLM-only contrast" in ARM_LABELS["comparison"]
