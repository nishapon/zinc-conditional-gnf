"""Tests for the MoFlow-fixed split derivation (zinc_gnf.split_derivation)."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import shutil

import numpy as np
import pytest

from tests.test_charge_aware_preprocessing import ALL_SMILES, write_csv, write_moflow_csv
from zinc_gnf.constants_charge_aware import vocabulary_fingerprint
from zinc_gnf.preprocessing import ConditionScaler
from zinc_gnf.preprocessing_charge_aware import (
    OutputCollisionError,
    prepare_zinc_dataset_charge_aware,
)
from zinc_gnf.split_audit import MoflowVerificationError
from zinc_gnf.split_derivation import (
    DerivationError,
    SourceCollisionError,
    derive_moflow_fixed_dataset,
)

# 1-based MoFlow indices. Positions 1,3,...,15 plus position 32 (a REJECTED row).
HELDOUT_1BASED = [2, 4, 6, 8, 10, 12, 14, 16, 33]
HELDOUT_POSITIONS = {i - 1 for i in HELDOUT_1BASED}
REJECTED_POSITIONS = set(range(32, 38))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def world(tmp_path, logger):
    """A completed random-split charge-aware dataset + MoFlow files, all synthetic."""
    csv_path = write_csv(tmp_path / "zinc.csv", ALL_SMILES, qed_offset={"CCO": 0.2})
    source = tmp_path / "random_split"
    prepare_zinc_dataset_charge_aware(
        csv_path=csv_path, processed_dir=source, metrics_dir=tmp_path / "random_metrics",
        baseline_processed_dir=tmp_path / "baseline", seed=42, logger=logger, progress_every=100,
    )
    moflow = write_moflow_csv(tmp_path / "moflow.csv", csv_path)
    idx = tmp_path / "valid_idx.json"
    idx.write_text(json.dumps(HELDOUT_1BASED))
    return {"csv": csv_path, "source": source, "moflow": moflow, "idx": idx,
            "out": tmp_path / "derived", "metrics": tmp_path / "derived_metrics",
            "baseline": tmp_path / "baseline", "tmp": tmp_path}


def derive(world, logger, **overrides):
    arguments = dict(
        source_dir=world["source"], output_dir=world["out"], metrics_dir=world["metrics"],
        csv_path=world["csv"], moflow_csv=world["moflow"], moflow_valid_idx=world["idx"],
        baseline_processed_dir=world["baseline"], logger=logger, seed=42,
        moflow_reference_train_count=None, moflow_reference_heldout_count=None,
    )
    arguments.update(overrides)
    return derive_moflow_fixed_dataset(**arguments)


def load(world):
    with (world["out"] / "splits.pkl").open("rb") as handle:
        return pickle.load(handle)


# ===========================================================================
def test_exact_membership_counts_and_rejected_assignment(world, logger):
    report = derive(world, logger)
    payload = load(world)
    ids = {name: {r["row_id"] for r in payload[name]} for name in ("train", "val", "test")}

    accepted = set(range(38)) - REJECTED_POSITIONS
    heldout_accepted = HELDOUT_POSITIONS & accepted
    assert ids["test"] == heldout_accepted                    # official held-out, exactly
    assert not HELDOUT_POSITIONS & (ids["train"] | ids["val"])
    assert ids["train"] | ids["val"] == accepted - HELDOUT_POSITIONS
    assert not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"])
    assert ids["train"] | ids["val"] | ids["test"] == accepted

    pool = len(accepted - HELDOUT_POSITIONS)
    assert len(payload["val"]) == int(0.10 * pool) >= 1
    assert report["test_molecules"] == len(heldout_accepted) == 8
    assert report["train_molecules"] + report["validation_molecules"] == pool
    assert report["train_molecules"] + report["validation_molecules"] + report["test_molecules"] == 32

    # Every rejected molecule is accounted for; the one inside the held-out set is flagged.
    rejected = {r["row_id"]: r for r in report["rejected_row_assignment"]}
    assert set(rejected) == REJECTED_POSITIONS
    assert rejected[32]["moflow_heldout"] is True and "TEST" in rejected[32]["assignment"]
    assert rejected[33]["moflow_heldout"] is False and "training pool" in rejected[33]["assignment"]
    assert all(r["original_smiles"] for r in rejected.values())
    assert all(v["ok"] for v in report["checks"].values())


def test_index_base_is_one_based_and_reported(world, logger):
    report = derive(world, logger)
    base = report["index_base"]
    assert base["applied_base"] == 1
    assert "idx - 1" in base["basis"]
    assert base["confirmed_by_data"] is False
    assert base["alternative_base_0_in_range"] is True
    assert base["rows_whose_membership_would_change_under_base_0"] > 0
    assert base["min_index"] == 2 and base["max_index"] == 33
    assert base["positions_min"] == 1 and base["positions_max"] == 32
    # Position 0 must NOT be held out (it would be, under the wrong base).
    assert 0 not in {r["row_id"] for r in load(world)["test"]}


def test_condition_scaler_refit_on_new_train_only(world, logger):
    report = derive(world, logger)
    payload = load(world)
    scaler = ConditionScaler.fit(payload["train"])
    assert report["condition_scaler"]["fit_on"] == "train_only"
    assert tuple(report["condition_scaler"]["mean"]) == pytest.approx(scaler.mean)
    assert tuple(report["condition_scaler"]["std"]) == pytest.approx(scaler.std)
    assert report["previous_condition_scaler_random_split"] is not None
    assert tuple(report["condition_scaler"]["mean"]) != pytest.approx(
        tuple(report["previous_condition_scaler_random_split"]["mean"]), abs=1e-12)
    everything = ConditionScaler.fit(payload["train"] + payload["val"] + payload["test"])
    assert tuple(everything.mean) != pytest.approx(scaler.mean, abs=1e-12)
    for name in ("train", "val", "test"):                   # conditions use the NEW statistics
        for record in payload[name]:
            assert np.allclose(record["condition"], scaler.transform(record["qed"], record["n_node"]))
    assert json.loads((world["out"] / "scaler.json").read_text()) == json.loads(
        json.dumps(report["condition_scaler"]))


def test_records_are_copied_not_reencoded_and_nothing_deduplicated(world, logger):
    derive(world, logger)
    with (world["source"] / "splits.pkl").open("rb") as handle:
        source = pickle.load(handle)
    src = {r["row_id"]: r for n in ("train", "val", "test") for r in source[n]}
    new = load(world)
    derived = {r["row_id"]: r for n in ("train", "val", "test") for r in new[n]}
    assert set(derived) == set(src) and len(derived) == 32          # nothing dropped
    for row_id, record in derived.items():
        original = src[row_id]
        assert np.array_equal(record["atom_types"], original["atom_types"])
        assert np.array_equal(record["bond_matrix"], original["bond_matrix"])
        assert record["smiles"] == original["smiles"] and record["qed"] == original["qed"]
        assert record["original_smiles"] == original["original_smiles"]
    # The stereoisomer pair (rows 30, 31) is kept: duplicates are reported, not removed.
    assert {30, 31} <= set(derived)
    assert derived[30]["smiles"] == derived[31]["smiles"]


def test_overlap_and_novelty_reference_are_reported_separately(world, logger):
    report = derive(world, logger)
    overlap = json.loads((world["metrics"] / "split_overlap.json").read_text())
    assert set(overlap["split_sizes"]) == {"train", "val", "test"}
    reference = report["novelty_reference"]
    assert reference["main_reference"] == "train"
    assert reference["train"]["records"] == report["train_molecules"]
    assert reference["train_plus_val"]["records"] == report["train_molecules"] + report["validation_molecules"]
    assert "full dataset including test" in reference["note"]
    assert reference["moflow_training_rows"] == 38 - len(HELDOUT_POSITIONS)
    assert report["deduplication"].startswith("none")
    assert (world["metrics"] / "moflow_split_verification.json").exists()
    assert (world["metrics"] / "split_derivation_checks.json").exists()


def test_row_index_marks_every_row(world, logger):
    derive(world, logger)
    with (world["out"] / "row_index.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(r["row_id"]) for r in rows] == list(range(38))
    for row in rows:
        position = int(row["row_id"])
        assert int(row["moflow_heldout"]) == int(position in HELDOUT_POSITIONS)
        if position in REJECTED_POSITIONS:
            assert row["status"] == "rejected" and row["split"] == "" and row["reason"]
        else:
            assert row["status"] == "accepted"
            assert (row["split"] == "test") == (position in HELDOUT_POSITIONS)


def test_source_dataset_and_baseline_are_never_written(world, logger):
    before = {p.name: sha(p) for p in world["source"].iterdir()}
    derive(world, logger)
    assert {p.name: sha(p) for p in world["source"].iterdir()} == before
    assert not world["baseline"].exists()
    assert world["out"].resolve() != world["source"].resolve()


def test_derivation_is_deterministic(world, logger):
    derive(world, logger)
    first = sha(world["out"] / "splits.pkl")
    shutil.rmtree(world["out"])
    shutil.rmtree(world["metrics"])
    derive(world, logger)
    assert sha(world["out"] / "splits.pkl") == first


def test_seed_changes_only_the_validation_carve_out(world, logger):
    derive(world, logger, seed=1)
    one = load(world)
    shutil.rmtree(world["out"])
    shutil.rmtree(world["metrics"])
    derive(world, logger, seed=2)
    two = load(world)
    assert {r["row_id"] for r in one["test"]} == {r["row_id"] for r in two["test"]}
    pool = {r["row_id"] for r in one["train"] + one["val"]}
    assert pool == {r["row_id"] for r in two["train"] + two["val"]}


# ---------------------------------------------------------------------------
# Output collisions.
# ---------------------------------------------------------------------------
def test_refuses_to_overwrite_the_source_dataset(world, logger):
    for bad in (world["source"], world["source"] / "nested"):
        with pytest.raises(SourceCollisionError, match="source"):
            derive(world, logger, output_dir=bad)
    with pytest.raises(SourceCollisionError):
        derive(world, logger, metrics_dir=world["source"])
    with pytest.raises(SourceCollisionError, match="must differ"):
        derive(world, logger, metrics_dir=world["out"])


def test_refuses_the_neutral_baseline_directory(world, logger):
    with pytest.raises(OutputCollisionError, match="baseline"):
        derive(world, logger, output_dir=world["baseline"])
    assert not world["baseline"].exists()


# ---------------------------------------------------------------------------
# Verification gates: nothing is written when any check fails.
# ---------------------------------------------------------------------------
def assert_nothing_written(world):
    assert not world["out"].exists() or not any(world["out"].iterdir())


def test_unverified_moflow_rows_are_refused(world, logger):
    bad = write_moflow_csv(world["tmp"] / "moflow_bad.csv", world["csv"], change_row=5)
    with pytest.raises((MoflowVerificationError, DerivationError)):
        derive(world, logger, moflow_csv=bad)
    assert_nothing_written(world)


def test_out_of_range_heldout_indices_are_refused(world, logger):
    world["idx"].write_text(json.dumps([2, 4, 999]))
    with pytest.raises((MoflowVerificationError, DerivationError)):
        derive(world, logger)
    assert_nothing_written(world)


def test_zero_index_is_refused_under_the_one_based_rule(world, logger):
    world["idx"].write_text(json.dumps([0, 4, 6]))              # 0 is invalid if 1-based
    with pytest.raises((MoflowVerificationError, DerivationError)):
        derive(world, logger)
    assert_nothing_written(world)


def test_changed_csv_is_detected(world, logger):
    lines = world["csv"].read_text().splitlines(keepends=True)
    world["csv"].write_text("".join(lines[:-1]))                 # drop a row
    with pytest.raises((DerivationError, MoflowVerificationError)):
        derive(world, logger)
    assert_nothing_written(world)


def test_csv_content_change_with_same_length_is_detected(world, logger):
    text = world["csv"].read_text()
    world["csv"].write_text(text.replace("CCCO", "CCCN", 1))
    with pytest.raises((DerivationError, MoflowVerificationError)):
        derive(world, logger)
    assert_nothing_written(world)


def test_vocabulary_or_representation_mismatch_is_refused(world, logger):
    path = world["source"] / "splits.pkl"
    payload = pickle.loads(path.read_bytes())
    payload["vocabulary_fingerprint"] = "0" * 16
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(DerivationError, match="source_vocabulary_fingerprint"):
        derive(world, logger)
    payload["vocabulary_fingerprint"] = vocabulary_fingerprint()
    payload["representation"] = "zinc250k_neutral_kekule_v1"
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(DerivationError, match="source_representation"):
        derive(world, logger)
    assert_nothing_written(world)


def test_reference_count_mismatches_are_refused(world, logger):
    with pytest.raises(DerivationError, match="heldout_count_matches_moflow_reference"):
        derive(world, logger, moflow_reference_heldout_count=24_887)
    with pytest.raises(DerivationError, match="pool_rows_equal_moflow_train_count"):
        derive(world, logger, moflow_reference_train_count=224_568)
    assert_nothing_written(world)


def test_missing_inputs_raise_file_not_found(world, logger):
    with pytest.raises(FileNotFoundError):
        derive(world, logger, moflow_csv=world["tmp"] / "absent.csv")
    with pytest.raises(FileNotFoundError):
        derive(world, logger, source_dir=world["tmp"] / "no_such_dir")


def test_sha256_pin_is_enforced(world, logger):
    with pytest.raises(DerivationError, match="csv_sha256_matches_pinned"):
        derive(world, logger, expected_sha256="0" * 64)
    assert_nothing_written(world)


# ---------------------------------------------------------------------------
# Command line.
# ---------------------------------------------------------------------------
def load_script(repository_root: Path):
    spec = importlib.util.spec_from_file_location(
        "derive_script", repository_root / "scripts" / "01d_derive_moflow_split.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_exit_codes(world, repository_root, monkeypatch):
    script = load_script(repository_root)
    monkeypatch.chdir(world["tmp"])
    real_config = str(repository_root / "configs" / "zinc250k_charge_aware.yaml")
    paths = ["--csv", str(world["csv"]), "--source-dir", str(world["source"]),
             "--output-dir", str(world["out"]), "--metrics-dir", str(world["metrics"]),
             "--moflow-csv", str(world["moflow"]), "--moflow-valid-idx", str(world["idx"]),
             "--skip-sha256"]

    # The real config pins MoFlow's published counts (224,568 / 24,887): a 38-row
    # synthetic dataset cannot reproduce them, so verification must fail and write nothing.
    assert script.main(["--config", real_config] + paths) == 5
    assert_nothing_written(world)

    # A config for synthetic data explicitly disables the published-count gate.
    synthetic = world["tmp"] / "synthetic.yaml"
    synthetic.write_text(
        "experiment: {seed: 42}\n"
        "representation: {baseline_processed_dir: data/processed/zinc250k}\n"
        "data:\n  csv_path: x\n  processed_dir: x\n  moflow_fixed_dir: x\n"
        "  moflow_fixed_metrics_dir: x\n  moflow_reference_train_count: null\n"
        "  moflow_reference_heldout_count: null\n")
    assert script.main(["--config", str(synthetic)] + paths) == 0
    assert (world["out"] / "splits.pkl").exists() and (world["metrics"] / "derive.log").exists()
    report = json.loads((world["out"] / "report.json").read_text())
    assert report["test_molecules"] == 8 and report["split_protocol"] == "moflow_fixed"

    # Failure exit codes.
    missing_source = [a if a != str(world["source"]) else str(world["tmp"] / "none") for a in paths]
    assert script.main(["--config", str(synthetic)] + missing_source) == 2
    assert script.main(["--config", str(world["tmp"] / "no.yaml")]) == 4
    colliding = [a if a != str(world["out"]) else str(world["source"]) for a in paths]
    assert script.main(["--config", str(synthetic)] + colliding) == 4
    shutil.rmtree(world["out"])
    tampered = write_moflow_csv(world["tmp"] / "tampered.csv", world["csv"], change_row=3)
    bad = [a if a != str(world["moflow"]) else str(tampered) for a in paths]
    assert script.main(["--config", str(synthetic)] + bad) == 5
    assert_nothing_written(world)
