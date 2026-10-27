"""Tests for the Phase 0 CPU preprocessing profiler (zinc_gnf.profiling)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from zinc_gnf import chemistry
from zinc_gnf.diagnostics import DatasetIdentityError
from zinc_gnf.profiling import (
    PHASES,
    PhaseTimer,
    benchmark_pickle,
    extrapolate,
    instrument_chemistry,
    latency_statistics,
    machine_calibration,
    run_preprocessing_profile,
    suggest_time_limit,
)
from zinc_gnf.utils.runtime import StopFlag


def profile(tmp_path, csv_path, logger, **overrides):
    arguments = dict(
        csv_path=csv_path,
        output_dir=tmp_path / "out" / "profile",
        processed_dir=tmp_path / "data" / "processed",
        logger=logger,
        sample_size=20,
        sample_seed=5,
        expected_sha256=None,
        expected_rows=None,
        scratch_dirs=[tmp_path / "scratch_a", tmp_path / "scratch_b"],
        progress_every=7,
    )
    arguments.update(overrides)
    return run_preprocessing_profile(**arguments)


def test_profile_report_is_complete_and_consistent(tmp_path, profiling_csv, logger):
    report = profile(tmp_path, profiling_csv, logger)

    assert report["complete"] is True
    assert report["writes_production_dataset"] is False
    assert report["sample"]["size"] == 20
    assert report["sample"]["population"] == 25

    clean = report["clean_pass"]
    assert clean["molecules"] == 20
    assert clean["throughput_mol_per_s"] > 0
    assert clean["latency"]["p99_ms"] >= clean["latency"]["median_ms"]
    assert clean["accepted"] + sum(
        v["count"] for k, v in clean["outcomes"].items() if k != "accepted"
    ) == 20
    assert len(clean["slowest_molecules"]) == 5

    breakdown = report["phase_breakdown"]
    assert set(breakdown["phases"]) == set(PHASES)
    shares = [breakdown["phases"][p]["share_of_total"] for p in PHASES]
    assert all(0.0 <= s <= 1.0 for s in shares)
    assert sum(shares) + breakdown["other_share"] == pytest.approx(1.0, abs=1e-6)
    assert breakdown["phases"]["parse"]["calls"] >= 20
    assert breakdown["phases"]["qed"]["calls"] > 0
    assert breakdown["phases"]["reconstruction"]["calls"] > 0

    estimate = report["full_dataset_estimate"]
    assert estimate["population_rows"] == 25
    assert estimate["usable_cpus"] >= 1
    assert estimate["total_serial_seconds"] > 0
    assert estimate["suggested_time_limit"].endswith(":00")
    assert estimate["ideal_parallel_encoding_seconds"] <= (
        estimate["encoding"]["estimate_seconds"] + 1e-12
    )
    assert report["calibration"]["rdkit_parse_and_write_per_second"] > 0
    assert report["environment"]["usable_cpu_count"] >= 1
    assert "repository_commit" in report["environment"]


def test_profile_json_is_strict(tmp_path, profiling_csv, logger):
    profile(tmp_path, profiling_csv, logger)
    path = tmp_path / "out" / "profile" / "preprocessing_profile.json"

    def reject_constant(name):
        raise AssertionError(f"non-strict JSON constant: {name}")

    parsed = json.loads(path.read_text(encoding="utf-8"),
                        parse_constant=reject_constant)
    assert parsed["kind"] == "preprocessing_profile"
    assert (tmp_path / "out" / "profile" / "preprocessing_profile.txt").exists()


def test_profile_does_not_touch_production_or_leave_scratch(tmp_path, profiling_csv, logger):
    profiling_csv_bytes = profiling_csv.read_bytes()
    report = profile(tmp_path, profiling_csv, logger)

    assert profiling_csv.read_bytes() == profiling_csv_bytes
    assert not (tmp_path / "data").exists()
    assert not list(tmp_path.rglob("splits.pkl"))
    # Pickle benchmark ran on both locations and cleaned up after itself.
    assert len(report["pickle_benchmarks"]) == 2
    for item in report["pickle_benchmarks"]:
        assert item["bytes"] > 0 and item["write_seconds"] >= 0
    for scratch in ("scratch_a", "scratch_b"):
        assert list((tmp_path / scratch).iterdir()) == []


def test_profile_refuses_output_or_scratch_inside_processed_dir(tmp_path, profiling_csv, logger):
    processed = tmp_path / "data" / "processed"
    with pytest.raises(ValueError, match="processed-data"):
        profile(tmp_path, profiling_csv, logger, output_dir=processed / "p")
    with pytest.raises(ValueError, match="processed-data"):
        profile(tmp_path, profiling_csv, logger, scratch_dirs=[processed])
    assert not processed.exists()


def test_profile_sample_is_deterministic(tmp_path, profiling_csv, logger):
    one = profile(tmp_path, profiling_csv, logger, skip_io=True,
                  skip_calibration=True)
    two = profile(tmp_path, profiling_csv, logger, skip_io=True,
                  skip_calibration=True)
    other = profile(tmp_path, profiling_csv, logger, skip_io=True,
                    skip_calibration=True, sample_seed=6, sample_size=10)
    assert one["sample"]["index_fingerprint"] == two["sample"]["index_fingerprint"]
    assert one["sample"]["index_fingerprint"] != other["sample"]["index_fingerprint"]
    assert one["clean_pass"]["outcomes"].keys() == two["clean_pass"]["outcomes"].keys()


def test_profile_wrong_sha256_fails(tmp_path, profiling_csv, logger):
    with pytest.raises(DatasetIdentityError):
        profile(tmp_path, profiling_csv, logger, expected_sha256="0" * 64)


def test_interrupted_profile_writes_partial_report(tmp_path, profiling_csv, logger):
    flag = StopFlag()
    flag.signal_name = "SIGTERM"
    report = profile(tmp_path, profiling_csv, logger, stop_flag=flag)
    assert report["complete"] is False
    assert report["stopped_by"] == "SIGTERM"
    out = tmp_path / "out" / "profile"
    assert (out / "preprocessing_profile.partial.json").exists()
    assert not (out / "preprocessing_profile.json").exists()


def test_small_sample_warns_about_recommended_range(tmp_path, profiling_csv, logger):
    report = profile(tmp_path, profiling_csv, logger)
    assert any("recommended" in w for w in report["warnings"])
    assert any("SLURM" in w for w in report["warnings"])  # not under sbatch


# ---------------------------------------------------------------------------
# Instrumentation must time the real functions and always restore them.
# ---------------------------------------------------------------------------
def test_instrumentation_wraps_real_functions_and_restores():
    original = (chemistry.Chem, chemistry.QED,
                chemistry.canonical_connectivity_smiles, chemistry.graph_to_mol)
    timer = PhaseTimer()
    with instrument_chemistry(timer):
        assert chemistry.Chem is not original[0]
        record, reason = chemistry.encode_zinc_smiles("CCO")
        assert record is not None and reason is None
    assert (chemistry.Chem, chemistry.QED,
            chemistry.canonical_connectivity_smiles,
            chemistry.graph_to_mol) == original
    assert timer.calls["parse"] == 2            # parse + canonical re-parse
    assert timer.calls["canonicalization"] == 2
    assert timer.calls["kekulize"] == 1
    assert timer.calls["reconstruction"] == 1
    assert timer.calls["qed"] == 1
    assert all(timer.seconds[p] > 0 for p in PHASES)


def test_instrumentation_restores_after_exception():
    original_chem = chemistry.Chem
    with pytest.raises(RuntimeError, match="boom"):
        with instrument_chemistry(PhaseTimer()):
            raise RuntimeError("boom")
    assert chemistry.Chem is original_chem
    assert chemistry.Chem is Chem


def test_instrumented_encoding_gives_identical_results():
    smiles = ["CCO", "c1ccccc1", "CC(=O)Nc1ccc(O)cc1", "C[NH3+]", "CC.O"]
    plain = [chemistry.encode_zinc_smiles(s) for s in smiles]
    with instrument_chemistry(PhaseTimer()):
        timed = [chemistry.encode_zinc_smiles(s) for s in smiles]
    for (rec_a, why_a), (rec_b, why_b) in zip(plain, timed):
        assert why_a == why_b
        if rec_a is not None:
            assert rec_a["smiles"] == rec_b["smiles"]
            assert rec_a["qed"] == rec_b["qed"]
            assert np.array_equal(rec_a["bond_matrix"], rec_b["bond_matrix"])


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def test_extrapolate_scales_linearly_with_interval():
    seconds = np.full(100, 0.002)
    result = extrapolate(seconds, 249_455)
    assert result["estimate_seconds"] == pytest.approx(0.002 * 249_455)
    assert result["ci95_low_seconds"] == pytest.approx(result["estimate_seconds"])
    noisy = extrapolate(np.array([0.001, 0.003] * 50), 1000)
    assert noisy["ci95_low_seconds"] < noisy["estimate_seconds"] < noisy["ci95_high_seconds"]
    assert np.isnan(extrapolate(np.array([]), 10)["estimate_seconds"])


def test_suggest_time_limit_rounds_and_caps():
    assert suggest_time_limit(10.0) == "00:15:00"
    assert suggest_time_limit(1800.0) == "01:00:00"
    assert suggest_time_limit(3601.0) == "02:15:00"
    assert suggest_time_limit(10 ** 9) == "24:00:00"
    assert suggest_time_limit(float("nan")) == "n/a"


def test_latency_statistics():
    stats = latency_statistics([0.001, 0.002, 0.003, 0.100])
    assert stats["n"] == 4
    assert stats["max_ms"] == pytest.approx(100.0)
    assert stats["total_seconds"] == pytest.approx(0.106)
    assert latency_statistics([]) == {"n": 0}


def test_machine_calibration_is_fixed_workload():
    result = machine_calibration(repetitions=2, python_loop=1000)
    assert result["python_loop_checksum"] == sum(v * v for v in range(1000))
    assert result["rdkit_parse_and_write_molecules"] == 2 * 10
    assert result["rdkit_parse_and_write_per_second"] > 0


def test_benchmark_pickle_cleans_up_and_reports_errors(tmp_path):
    ok = benchmark_pickle({"a": list(range(1000))}, tmp_path / "s", label="x")
    assert ok["bytes"] > 0 and "error" not in ok
    assert list((tmp_path / "s").iterdir()) == []
    blocked = tmp_path / "file_not_dir"
    blocked.write_text("x")
    with pytest.raises(OSError):
        benchmark_pickle({}, blocked, label="bad")


def test_cli_profile_exit_codes(tmp_path, profiling_csv, repository_root):
    path = repository_root / "scripts" / "00_profile_preprocessing.py"
    spec = importlib.util.spec_from_file_location("profile_script", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    config = str(repository_root / "configs" / "zinc250k.yaml")
    base = ["--config", config, "--csv", str(profiling_csv),
            "--output-dir", str(tmp_path / "cli_prof"),
            "--sample-size", "15", "--skip-calibration",
            "--scratch-dir", str(tmp_path / "cli_scratch")]

    assert script.main(base + ["--expected-sha256", "skip"]) == 0
    assert (tmp_path / "cli_prof" / "preprocessing_profile.json").exists()
    assert script.main(base + ["--expected-sha256", "e" * 64]) == 3
    assert script.main(["--config", config, "--csv", str(tmp_path / "x.csv"),
                        "--output-dir", str(tmp_path / "cli_prof2"),
                        "--expected-sha256", "skip"]) == 2
    assert script.main(base + ["--sample-size", "0"]) == 4
