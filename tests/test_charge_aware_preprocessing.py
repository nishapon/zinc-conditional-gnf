"""Tests for the charge-aware preprocessing path and the split audits.

All data is synthetic and tiny: no dataset, network or cluster needed.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import pickle

import numpy as np
import pytest
from rdkit import Chem, rdBase
from rdkit.Chem import QED

from tests.conftest import sha256_of
from zinc_gnf.constants_charge_aware import ATOM_VOCAB, NUM_ATOM_TYPES
from zinc_gnf.preprocessing import ConditionScaler, random_split
from zinc_gnf.preprocessing_charge_aware import (
    OutputCollisionError,
    PreprocessingInterrupted,
    SPLIT_MOFLOW,
    SPLIT_RANDOM,
    assert_not_baseline_dir,
    fixed_heldout_split,
    prepare_zinc_dataset_charge_aware,
    qed_discrepancy,
)
from zinc_gnf.split_audit import (
    MoflowVerificationError,
    classify_duplicate_group,
    duplicate_audit,
    moflow_heldout_positions,
    require_verified,
    split_overlap_audit,
    verify_moflow_row_identity,
)
from zinc_gnf.utils.runtime import StopFlag

# 30 distinct molecules incl. charged ones, plus duplicates and rejects.
BASE = [
    "CCO", "c1ccccc1", "CC(=O)Nc1ccc(O)cc1", "C[NH3+]", "CC(=O)[O-]",
    "C[N+](=O)[O-]", "C[N+](C)(C)C", "c1cc[nH+]cc1", "[NH3+]CC(=O)[O-]",
    "CN=[N+]=[N-]", "C[O+](C)C", "C[S+](C)C", "[C-]#[N+]C", "C[P+](C)(C)C",
    "CCN", "CCCO", "c1ccncc1", "Fc1ccccc1", "Clc1ccccc1", "CC(=O)O",
    "c1ccsc1", "c1ccoc1", "C=CC#N", "O=C(Nc1ccc(F)cc1)c1cc(Cl)ccc1Br",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "c1cnc[nH]1", "CS(=O)(=O)[N-]c1ccccc1",
    "c1ccccc1[S-]", "[O-][N+](=O)c1ccccc1", "C[n+]1ccccc1",
]
# Stereoisomer pair -> a duplicate group under non-stereo identity.
STEREO_PAIR = ["C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O"]
REJECTS = ["not a smiles", "C[SiH3]", "[2H]CCO", "CC.O", "C", "C[Cl+]C"]
ALL_SMILES = BASE + STEREO_PAIR + REJECTS      # 30 + 2 + 6 = 38 rows


def write_csv(path: Path, smiles_list, qed_offset: dict[str, float] | None = None) -> Path:
    qed_offset = qed_offset or {}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["smiles", "logP", "qed", "SAS"])
        for index, smiles in enumerate(smiles_list):
            with rdBase.BlockLogs():
                molecule = Chem.MolFromSmiles(smiles)
            qed_value = (QED.qed(molecule) if molecule is not None else 0.5) + qed_offset.get(smiles, 0.0)
            writer.writerow([smiles + "\n", round(1.0 + index / 100, 4),
                             round(qed_value, 6), 2.0])
    return path


def write_moflow_csv(path: Path, source_csv: Path, *, change_row: int | None = None) -> Path:
    """MoFlow's layout: pandas index column, no embedded newlines."""
    with source_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "smiles", "logP", "qed", "SAS"])
        for i, row in enumerate(rows):
            smiles = row["smiles"].strip()
            if change_row == i:
                smiles = "CCCCCCCC"
            writer.writerow([i, smiles, row["logP"], row["qed"], row["SAS"]])
    return path


@pytest.fixture
def dataset(tmp_path):
    return write_csv(tmp_path / "zinc.csv", ALL_SMILES, qed_offset={"CCO": 0.2})


def run(tmp_path, csv_path, logger, **overrides):
    arguments = dict(
        csv_path=csv_path,
        processed_dir=tmp_path / "processed_charge_aware",
        metrics_dir=tmp_path / "metrics_charge_aware",
        baseline_processed_dir=tmp_path / "processed_baseline",
        seed=42,
        logger=logger,
        progress_every=10,
    )
    arguments.update(overrides)
    return prepare_zinc_dataset_charge_aware(**arguments)


def load_splits(tmp_path):
    with (tmp_path / "processed_charge_aware" / "splits.pkl").open("rb") as handle:
        return pickle.load(handle)


# ===========================================================================
# End-to-end behaviour.
# ===========================================================================
def test_every_row_is_accounted_for_and_charged_molecules_kept(tmp_path, dataset, logger):
    report = run(tmp_path, dataset, logger)
    assert report["candidate_rows"] == len(ALL_SMILES) == 38
    assert report["accepted_molecules"] + report["rejected_molecules"] == 38
    assert report["rejected_molecules"] == len(REJECTS) == 6
    assert report["rejection_reasons"]["parse_failure"] == 1
    assert report["rejection_reasons"]["unsupported_element"] == 1
    assert report["rejection_reasons"]["isotope"] == 1
    assert report["rejection_reasons"]["multiple_fragments"] == 1
    assert report["rejection_reasons"]["node_count_below_min"] == 1
    assert report["rejection_reasons"]["unsupported_charge_type"] == 1
    assert report["unsupported_charge_types"] == {"Cl+1": 1}
    assert sum(report["rejection_reasons"].values()) == report["rejected_molecules"]

    charged = report["charge_composition"]["molecules_with_charged_atoms"]
    # Exactly the 15 charged fixtures: the neutral-only rule would drop them all.
    assert charged == 15
    assert report["representation_version"] == "zinc250k_charge_aware_kekule_v2"
    assert report["num_atom_types"] == NUM_ATOM_TYPES == 17
    assert report["atom_vocab"] == list(ATOM_VOCAB)
    assert report["deduplication"].startswith("none")


def test_row_index_lists_every_row_with_status_reason_split(tmp_path, dataset, logger):
    run(tmp_path, dataset, logger)
    with (tmp_path / "processed_charge_aware" / "row_index.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(r["row_id"]) for r in rows] == list(range(38))      # order preserved
    accepted = [r for r in rows if r["status"] == "accepted"]
    rejected = [r for r in rows if r["status"] == "rejected"]
    assert len(accepted) == 32 and len(rejected) == 6
    assert all(r["split"] in ("train", "val", "test") and r["reason"] == "" for r in accepted)
    assert all(r["split"] == "" and r["reason"] for r in rejected)
    # Rejected rows are the last six, with their original SMILES preserved.
    with (tmp_path / "metrics_charge_aware" / "rejected_rows.csv").open(newline="") as handle:
        rej = list(csv.DictReader(handle))
    assert [int(r["row_id"]) for r in rej] == list(range(32, 38))
    assert [r["original_smiles"] for r in rej] == REJECTS


def test_records_keep_row_id_original_smiles_and_csv_qed(tmp_path, dataset, logger):
    run(tmp_path, dataset, logger)
    payload = load_splits(tmp_path)
    records = payload["train"] + payload["val"] + payload["test"]
    by_row = {r["row_id"]: r for r in records}
    assert len(by_row) == len(records) == 32                      # no duplicates of row ids
    for row_id, record in by_row.items():
        assert record["original_smiles"].strip() == ALL_SMILES[row_id]
        assert record["representation"] == "zinc250k_charge_aware_kekule_v2"
        assert record["qed_csv"] is not None
        assert record["condition"].shape == (2,)
        assert record["atom_types"].max() < NUM_ATOM_TYPES
    assert payload["vocabulary_fingerprint"] == report_fingerprint()
    assert payload["atom_vocab"] == list(ATOM_VOCAB)
    assert payload["software"]["rdkit"]


def report_fingerprint():
    from zinc_gnf.constants_charge_aware import vocabulary_fingerprint
    return vocabulary_fingerprint()


def test_qed_is_the_reconstructed_molecule_qed_and_discrepancy_is_reported(tmp_path, dataset, logger):
    report = run(tmp_path, dataset, logger)
    payload = load_splits(tmp_path)
    cco = next(r for s in ("train", "val", "test") for r in payload[s]
               if r["original_smiles"].strip() == "CCO")
    assert cco["qed"] == pytest.approx(QED.qed(Chem.MolFromSmiles("CCO")), abs=1e-12)
    assert cco["qed_csv"] - cco["qed"] == pytest.approx(0.2, abs=1e-5)

    discrepancy = report["qed_csv_vs_rdkit"]
    assert discrepancy["molecules_compared"] == 32
    assert discrepancy["max_abs_delta"] == pytest.approx(0.2, abs=1e-4)
    assert discrepancy["above_1e-1"] == 1
    assert report["software"]["rdkit"]
    assert (tmp_path / "metrics_charge_aware" / "qed_discrepancy.json").exists()


def test_deterministic_outputs(tmp_path, dataset, logger):
    first = run(tmp_path / "a", dataset, logger)
    second = run(tmp_path / "b", dataset, logger)
    pa = pickle.loads((tmp_path / "a" / "processed_charge_aware" / "splits.pkl").read_bytes())
    pb = pickle.loads((tmp_path / "b" / "processed_charge_aware" / "splits.pkl").read_bytes())
    for name in ("train", "val", "test"):
        assert [r["row_id"] for r in pa[name]] == [r["row_id"] for r in pb[name]]
        for ra, rb in zip(pa[name], pb[name]):
            assert np.array_equal(ra["atom_types"], rb["atom_types"])
            assert np.array_equal(ra["bond_matrix"], rb["bond_matrix"])
            assert ra["qed"] == rb["qed"]
    assert first["condition_scaler"] == second["condition_scaler"]
    assert (tmp_path / "a" / "processed_charge_aware" / "row_index.csv").read_bytes() == \
        (tmp_path / "b" / "processed_charge_aware" / "row_index.csv").read_bytes()


def test_default_split_matches_the_baseline_algorithm(tmp_path, dataset, logger):
    run(tmp_path, dataset, logger)
    payload = load_splits(tmp_path)
    records = sorted(
        payload["train"] + payload["val"] + payload["test"], key=lambda r: r["row_id"],
    )
    expected = random_split(records, train_fraction=0.8, validation_fraction=0.1, seed=42)
    for name in ("train", "val", "test"):
        assert [r["row_id"] for r in payload[name]] == [r["row_id"] for r in expected[name]]
    assert payload["split_protocol"] == SPLIT_RANDOM


def test_scaler_is_fit_on_training_split_only(tmp_path, dataset, logger):
    report = run(tmp_path, dataset, logger)
    payload = load_splits(tmp_path)
    train_scaler = ConditionScaler.fit(payload["train"])
    assert report["condition_scaler"]["fit_on"] == "train_only"
    assert tuple(report["condition_scaler"]["mean"]) == pytest.approx(train_scaler.mean)
    assert tuple(report["condition_scaler"]["std"]) == pytest.approx(train_scaler.std)
    everything = ConditionScaler.fit(payload["train"] + payload["val"] + payload["test"])
    assert tuple(everything.mean) != pytest.approx(train_scaler.mean, abs=1e-12)
    # The stored conditions use the train-only statistics.
    record = payload["val"][0]
    expected = train_scaler.transform(record["qed"], record["n_node"])
    assert np.allclose(record["condition"], expected)
    assert json.loads((tmp_path / "processed_charge_aware" / "scaler.json").read_text()) == \
        json.loads(json.dumps(report["condition_scaler"]))


def test_duplicates_are_reported_not_removed(tmp_path, dataset, logger):
    report = run(tmp_path, dataset, logger)
    summary = report["duplicates_summary"]
    assert summary["duplicate_groups"] == 1
    assert summary["excess_records"] == 1
    assert summary["group_kinds"] == {"stereoisomers": 1}
    assert report["accepted_molecules"] == 32            # both stereoisomers kept
    detail = json.loads((tmp_path / "metrics_charge_aware" / "duplicates.json").read_text())
    assert detail["listed_groups"][0]["row_ids"] == [30, 31]
    assert "REPORT ONLY" in detail["policy"]


def test_split_overlap_is_reported(tmp_path, dataset, logger):
    report = run(tmp_path, dataset, logger)
    overlap = report["split_overlap_summary"]
    assert set(overlap["split_sizes"]) == {"train", "val", "test"}
    assert sum(overlap["split_sizes"].values()) == 32
    assert "records_with_train_twin" in overlap
    assert (tmp_path / "metrics_charge_aware" / "split_overlap.json").exists()


def test_source_csv_is_not_modified(tmp_path, dataset, logger):
    before = dataset.read_bytes()
    run(tmp_path, dataset, logger, expected_sha256=sha256_of(dataset))
    assert dataset.read_bytes() == before


def test_wrong_sha256_and_row_count_fail_before_writing(tmp_path, dataset, logger):
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        run(tmp_path, dataset, logger, expected_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="expected 249,455"):
        run(tmp_path, dataset, logger, expected_rows=249_455)
    assert not (tmp_path / "processed_charge_aware").exists()


def test_subsample_is_flagged(tmp_path, dataset, logger):
    report = run(tmp_path, dataset, logger, max_molecules=10)
    assert report["subsampled"] is True and report["candidate_rows"] == 10
    assert report["accepted_molecules"] + report["rejected_molecules"] == 10


def test_interruption_writes_nothing(tmp_path, dataset, logger):
    flag = StopFlag()
    flag.signal_name = "SIGTERM"
    with pytest.raises(PreprocessingInterrupted):
        run(tmp_path, dataset, logger, stop_flag=flag)
    assert not (tmp_path / "processed_charge_aware").exists()


# ===========================================================================
# The neutral-only baseline is never touched.
# ===========================================================================
def test_refuses_to_use_or_enclose_the_baseline_directory(tmp_path, dataset, logger):
    baseline = tmp_path / "processed_baseline"
    for bad in (baseline, baseline / "sub", tmp_path):
        with pytest.raises(OutputCollisionError, match="baseline"):
            run(tmp_path, dataset, logger, processed_dir=bad)
    with pytest.raises(OutputCollisionError):
        run(tmp_path, dataset, logger, metrics_dir=baseline / "metrics")
    assert not baseline.exists()
    assert_not_baseline_dir(tmp_path / "x", baseline)               # unrelated: fine


def test_default_config_paths_are_separate_from_the_baseline(repository_root):
    from zinc_gnf.utils.config import load_config

    baseline = load_config(repository_root / "configs" / "zinc250k.yaml")["data"]
    charge = load_config(repository_root / "configs" / "zinc250k_charge_aware.yaml")
    assert charge["data"]["processed_dir"] != baseline["processed_dir"]
    assert charge["representation"]["baseline_processed_dir"] == baseline["processed_dir"]
    assert_not_baseline_dir(charge["data"]["processed_dir"], baseline["processed_dir"])
    assert charge["data"]["metrics_dir"] != "outputs/metrics/preprocessing"
    assert charge["representation"]["num_atom_types"] == NUM_ATOM_TYPES
    assert charge["data"]["split_protocol"] == "random_80_10_10"
    assert charge["data"]["expected_sha256"] == \
        "35e3f1a52b1badc0697e373d73a18ad773f415936ff992f4c6baa2e067b3e6ae"


# ===========================================================================
# MoFlow verification and the fixed split.
# ===========================================================================
def _our_columns(path: Path):
    from zinc_gnf.diagnostics import read_zinc_table

    table = read_zinc_table(path)
    return table.smiles, {"logP": table.logp, "qed": table.qed, "SAS": table.sas}


def test_moflow_verification_passes_for_identical_rows(tmp_path, dataset):
    moflow = write_moflow_csv(tmp_path / "moflow.csv", dataset)
    idx = tmp_path / "idx.json"
    idx.write_text(json.dumps([3, 8, 15, 22]))                  # 1-based
    smiles, numeric = _our_columns(dataset)
    result = verify_moflow_row_identity(smiles, numeric, moflow, idx)
    assert result["verified"] is True
    assert result["row_count_equal"] and result["smiles_mismatches"] == 0
    assert result["moflow_index_column_is_0_to_n_minus_1"] is True
    assert result["heldout_count"] == 4
    assert result["index_base_assumed"] == 1
    assert result["index_base_confirmed_by_data"] is False     # both bases in range
    assert "cannot confirm" in result["index_base_note"]
    require_verified(result)


def test_moflow_verification_detects_a_changed_row(tmp_path, dataset):
    moflow = write_moflow_csv(tmp_path / "moflow.csv", dataset, change_row=5)
    idx = tmp_path / "idx.json"
    idx.write_text(json.dumps([3, 8]))
    smiles, numeric = _our_columns(dataset)
    result = verify_moflow_row_identity(smiles, numeric, moflow, idx)
    assert result["verified"] is False
    assert result["smiles_mismatches"] == 1
    assert result["first_smiles_mismatches"][0]["row"] == 5
    with pytest.raises(MoflowVerificationError, match="NOT verified"):
        require_verified(result)


def test_moflow_verification_detects_row_count_and_numeric_differences(tmp_path, dataset):
    smiles, numeric = _our_columns(dataset)
    short = tmp_path / "short.csv"
    full = write_moflow_csv(tmp_path / "full.csv", dataset)
    lines = full.read_text().splitlines()
    short.write_text("\n".join(lines[:-3]) + "\n")
    idx = tmp_path / "idx.json"
    idx.write_text(json.dumps([3]))
    assert verify_moflow_row_identity(smiles, numeric, short, idx)["verified"] is False

    altered = tmp_path / "altered.csv"
    text = full.read_text().splitlines()
    parts = text[2].split(",")
    parts[3] = str(float(parts[3]) + 0.01)                       # qed column
    text[2] = ",".join(parts)
    altered.write_text("\n".join(text) + "\n")
    result = verify_moflow_row_identity(smiles, numeric, altered, idx)
    assert result["numeric_columns"]["qed"]["rows_exceeding_tolerance"] == 1
    assert result["verified"] is False


def test_moflow_heldout_positions_base_and_range():
    assert moflow_heldout_positions([2, 5, 10], 10) == [1, 4, 9]          # 1-based
    assert moflow_heldout_positions([0, 5], 10, index_base=0) == [0, 5]
    with pytest.raises(ValueError, match="outside"):
        moflow_heldout_positions([0, 5], 10)                                # 0 invalid if 1-based
    with pytest.raises(ValueError, match="outside"):
        moflow_heldout_positions([11], 10)
    with pytest.raises(ValueError, match="duplicates"):
        moflow_heldout_positions([3, 3], 10)
    with pytest.raises(ValueError, match="index_base"):
        moflow_heldout_positions([1], 10, index_base=2)


def test_fixed_split_protocol_uses_moflow_heldout_as_test(tmp_path, dataset, logger):
    moflow = write_moflow_csv(tmp_path / "moflow.csv", dataset)
    idx = tmp_path / "idx.json"
    heldout_1based = [2, 4, 6, 8, 10, 12, 14, 16]           # rows 1,3,...,15 (0-based)
    idx.write_text(json.dumps(heldout_1based))
    report = run(tmp_path, dataset, logger, split_protocol=SPLIT_MOFLOW,
                 moflow_csv=moflow, moflow_valid_idx=idx)
    payload = load_splits(tmp_path)
    held_rows = {i - 1 for i in heldout_1based}
    assert {r["row_id"] for r in payload["test"]} == held_rows
    assert not held_rows & {r["row_id"] for r in payload["train"] + payload["val"]}
    assert payload["split_protocol"] == SPLIT_MOFLOW
    assert report["split_protocol_detail"]["moflow_index_base_assumed"] == 1
    assert report["moflow_verification"]["verified"] is True
    assert report["train_molecules"] + report["validation_molecules"] + report["test_molecules"] == 32
    # Scaler still uses train only.
    assert report["condition_scaler"]["fit_on"] == "train_only"
    assert (tmp_path / "metrics_charge_aware" / "moflow_split_verification.json").exists()


def test_fixed_split_is_refused_when_verification_fails(tmp_path, dataset, logger):
    moflow = write_moflow_csv(tmp_path / "moflow.csv", dataset, change_row=4)
    idx = tmp_path / "idx.json"
    idx.write_text(json.dumps([2, 4]))
    with pytest.raises(MoflowVerificationError):
        run(tmp_path, dataset, logger, split_protocol=SPLIT_MOFLOW,
            moflow_csv=moflow, moflow_valid_idx=idx)
    assert not (tmp_path / "processed_charge_aware").exists()


def test_verification_alone_never_changes_the_split(tmp_path, dataset, logger):
    moflow = write_moflow_csv(tmp_path / "moflow.csv", dataset)
    idx = tmp_path / "idx.json"
    idx.write_text(json.dumps([2, 4, 6]))
    run(tmp_path / "with", dataset, logger, moflow_csv=moflow, moflow_valid_idx=idx)
    run(tmp_path / "without", dataset, logger)
    a = pickle.loads((tmp_path / "with" / "processed_charge_aware" / "splits.pkl").read_bytes())
    b = pickle.loads((tmp_path / "without" / "processed_charge_aware" / "splits.pkl").read_bytes())
    for name in ("train", "val", "test"):
        assert [r["row_id"] for r in a[name]] == [r["row_id"] for r in b[name]]
    assert a["split_protocol"] == SPLIT_RANDOM


def test_fixed_split_argument_validation(tmp_path, dataset, logger):
    with pytest.raises(ValueError, match="requires both"):
        run(tmp_path, dataset, logger, split_protocol=SPLIT_MOFLOW)
    with pytest.raises(ValueError, match="split_protocol"):
        run(tmp_path, dataset, logger, split_protocol="bogus")
    moflow = write_moflow_csv(tmp_path / "m.csv", dataset)
    idx = tmp_path / "i.json"
    idx.write_text("[2]")
    with pytest.raises(ValueError, match="max-molecules"):
        run(tmp_path, dataset, logger, split_protocol=SPLIT_MOFLOW,
            moflow_csv=moflow, moflow_valid_idx=idx, max_molecules=5)


def test_fixed_heldout_split_unit():
    records = [{"row_id": i, "smiles": f"s{i}"} for i in range(20)]
    split = fixed_heldout_split(records, {1, 5, 9}, validation_fraction_of_pool=0.25, seed=1)
    assert {r["row_id"] for r in split["test"]} == {1, 5, 9}
    assert len(split["val"]) == int(0.25 * 17) and len(split["train"]) == 17 - len(split["val"])
    again = fixed_heldout_split(records, {1, 5, 9}, validation_fraction_of_pool=0.25, seed=1)
    assert [r["row_id"] for r in again["val"]] == [r["row_id"] for r in split["val"]]
    other = fixed_heldout_split(records, {1, 5, 9}, validation_fraction_of_pool=0.25, seed=2)
    assert [r["row_id"] for r in other["val"]] != [r["row_id"] for r in split["val"]]
    with pytest.raises(RuntimeError, match="empty"):
        fixed_heldout_split(records, set(), validation_fraction_of_pool=0.25, seed=1)
    with pytest.raises(ValueError):
        fixed_heldout_split(records, {1}, validation_fraction_of_pool=1.5, seed=1)


# ===========================================================================
# Duplicate and overlap audits (unit level).
# ===========================================================================
def rec(row_id, original, canonical, qed=0.5):
    return {"row_id": row_id, "original_smiles": original, "smiles": canonical, "qed_csv": qed}


def test_duplicate_classification():
    exact = [rec(0, "CCO", "CCO"), rec(1, "CCO\n", "CCO")]
    assert classify_duplicate_group(exact) == "exact_repeat"
    rewritten = [rec(0, "CCO", "CCO"), rec(1, "OCC", "CCO")]
    assert classify_duplicate_group(rewritten) == "same_molecule_rewritten"
    stereo = [rec(0, "C[C@H](N)CC", "CCC(C)N"), rec(1, "C[C@@H](N)CC", "CCC(C)N")]
    assert classify_duplicate_group(stereo) == "stereoisomers"
    # Markers on an atom that is not a stereocentre change nothing: that is a
    # rewrite of one molecule, not a stereoisomer pair.
    no_centre = [rec(0, "C[C@H](N)C", "CC(C)N"), rec(1, "C[C@@H](N)C", "CC(C)N")]
    assert classify_duplicate_group(no_centre) == "same_molecule_rewritten"


def test_duplicate_audit_groups_and_never_mutates():
    records = [rec(0, "CCO", "CCO"), rec(1, "CCO", "CCO", 0.6), rec(2, "CCN", "CCN"),
               rec(3, "C[C@H](N)CC", "CCC(C)N"), rec(4, "C[C@@H](N)CC", "CCC(C)N"),
               rec(5, "CCO", "CCO")]
    snapshot = [dict(r) for r in records]
    audit = duplicate_audit(records)
    assert audit["records"] == 6 and audit["unique_canonical_smiles"] == 3
    assert audit["duplicate_groups"] == 2 and audit["excess_records"] == 3
    assert audit["group_size_histogram"] == {"2": 1, "3": 1}
    assert audit["group_kinds"] == {"exact_repeat": 1, "stereoisomers": 1}
    assert audit["listed_groups"][0]["row_ids"] == [0, 1, 5]              # biggest first
    assert audit["csv_qed_spread_max"] == pytest.approx(0.1)
    assert records == snapshot
    assert duplicate_audit([])["records"] == 0


def test_split_overlap_audit_counts_twins():
    splits = {
        "train": [rec(0, "a", "A"), rec(1, "b", "B"), rec(2, "c", "C")],
        "val": [rec(3, "a2", "A"), rec(4, "d", "D")],
        "test": [rec(5, "b2", "B"), rec(6, "a3", "A"), rec(7, "e", "E")],
    }
    audit = split_overlap_audit(splits)
    assert audit["shared_unique_smiles_pairwise"] == {
        "train&val": 1, "train&test": 2, "val&test": 1,
    }
    assert audit["records_with_train_twin"]["val"]["records_with_train_twin"] == 1
    assert audit["records_with_train_twin"]["test"]["records_with_train_twin"] == 2
    assert audit["canonical_smiles_spanning_multiple_splits"] == 2         # A and B
    assert {e["canonical_smiles"] for e in audit["examples"]} == {"A", "B"}
    assert "REPORT ONLY" in audit["policy"]


def test_qed_discrepancy_helper():
    records = [{"qed": 0.5, "qed_csv": 0.5}, {"qed": 0.6, "qed_csv": 0.4},
               {"qed": 0.7, "qed_csv": None}]
    stats = qed_discrepancy(records)
    assert stats["molecules_compared"] == 2
    assert stats["max_abs_delta"] == pytest.approx(0.2)
    assert qed_discrepancy([]) == {"molecules_compared": 0}


# ===========================================================================
# Command-line interfaces.
# ===========================================================================
def load_script(repository_root: Path, name: str):
    spec = importlib.util.spec_from_file_location(
        name.replace(".py", ""), repository_root / "scripts" / name,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_preprocess_end_to_end_and_exit_codes(tmp_path, dataset, repository_root, monkeypatch):
    script = load_script(repository_root, "01b_preprocess_charge_aware.py")
    base = ["--csv", str(dataset),
            "--processed-dir", str(tmp_path / "p"), "--metrics-dir", str(tmp_path / "m"),
            "--expected-sha256", "skip", "--progress-every", "5"]
    # The pinned row count (249,455) legitimately fails on a 38-row file.
    assert script.main(base) == 1
    assert not (tmp_path / "p").exists()
    assert script.main(base + ["--max-molecules", "20"]) == 0
    assert (tmp_path / "p" / "splits.pkl").exists()
    assert (tmp_path / "m" / "run.log").exists()
    assert script.main(["--csv", str(dataset), "--processed-dir", str(tmp_path / "q"),
                        "--metrics-dir", str(tmp_path / "m2"), "--max-molecules", "20"]) == 3
    assert script.main(["--csv", str(tmp_path / "nope.csv"), "--max-molecules", "5",
                        "--expected-sha256", "skip", "--processed-dir", str(tmp_path / "r"),
                        "--metrics-dir", str(tmp_path / "m3")]) == 2
    assert script.main(["--config", str(tmp_path / "no.yaml")]) == 4
    # Writing into the baseline directory is a hard error (exit 4) and creates
    # nothing. Run from a scratch working directory so the relative baseline
    # path from the config resolves there, not in the real repository.
    monkeypatch.chdir(tmp_path)
    config = str(repository_root / "configs" / "zinc250k_charge_aware.yaml")
    assert script.main(base + ["--config", config, "--max-molecules", "20",
                               "--processed-dir", "data/processed/zinc250k"]) == 4
    assert not Path("data/processed/zinc250k").exists()


def test_cli_verify_moflow_split(tmp_path, dataset, repository_root):
    script = load_script(repository_root, "01c_verify_moflow_split.py")
    moflow = write_moflow_csv(tmp_path / "moflow.csv", dataset)
    idx = tmp_path / "idx.json"
    idx.write_text(json.dumps([2, 4, 6]))
    out = tmp_path / "verification.json"
    args = ["--csv", str(dataset), "--moflow-csv", str(moflow),
            "--moflow-valid-idx", str(idx), "--output", str(out)]
    assert script.main(args) == 0
    assert json.loads(out.read_text())["verified"] is True

    bad = write_moflow_csv(tmp_path / "bad.csv", dataset, change_row=2)
    assert script.main(["--csv", str(dataset), "--moflow-csv", str(bad),
                        "--moflow-valid-idx", str(idx), "--output", str(out)]) == 5
    assert json.loads(out.read_text())["verified"] is False
    assert script.main(["--csv", str(dataset), "--moflow-csv", str(tmp_path / "x.csv"),
                        "--moflow-valid-idx", str(idx)]) == 2
