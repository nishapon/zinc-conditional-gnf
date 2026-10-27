"""Tests for the Phase 0 chemistry diagnostics (zinc_gnf.diagnostics)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from tests.conftest import SYNTHETIC_MOLECULES, sha256_of
from zinc_gnf.chemistry import encode_zinc_smiles
from zinc_gnf.diagnostics import (
    ChemistryCensus,
    DatasetIdentityError,
    atom_type_key,
    delta_statistics,
    deterministic_sample_indices,
    indices_fingerprint,
    read_zinc_table,
    run_chemistry_diagnostics,
    strip_formal_charges,
)
from zinc_gnf.utils.runtime import StopFlag

EXPECTED_REJECTIONS = {
    "non_neutral": 4,
    "unsupported_element": 1,
    "isotope": 1,
    "multiple_fragments": 1,
    "parse_failure": 1,
    "node_count_below_min": 1,
    "empty_smiles": 1,
}


def run(tmp_path, csv_path, logger, **overrides):
    arguments = dict(
        csv_path=csv_path,
        output_dir=tmp_path / "out" / "chem",
        processed_dir=tmp_path / "data" / "processed",
        figures_dir=tmp_path / "out" / "figs",
        logger=logger,
        expected_sha256=None,
        expected_rows=None,
        progress_every=3,
        make_figures=False,
    )
    arguments.update(overrides)
    return run_chemistry_diagnostics(**arguments)


# ---------------------------------------------------------------------------
# Counts agree with the EXISTING encoder, one molecule at a time.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", SYNTHETIC_MOLECULES)
def test_fixture_verdicts_match_existing_encoder(smiles, expected):
    record, reason = encode_zinc_smiles(smiles)
    assert (reason or "accepted") == expected


def test_report_counts_and_rejections(tmp_path, synthetic_csv, logger):
    report = run(tmp_path, synthetic_csv, logger)
    counts = report["counts"]

    assert report["complete"] is True
    assert counts["total_molecules"] == len(SYNTHETIC_MOLECULES) == 13
    assert counts["accepted_neutral_only"] == 3
    assert counts["rejected_total"] == 10
    assert counts["parse_failures"] == 1
    assert counts["empty_smiles"] == 1
    assert counts["parseable_molecules"] == 11
    assert counts["charged_molecules"] == 4
    assert counts["neutral_molecules"] == 7
    assert counts["rejected_as_non_neutral"] == 4
    assert counts["accepted_fraction"] == pytest.approx(3 / 13)

    reasons = {k: v["count"] for k, v in report["rejection_reasons"].items()}
    assert reasons == EXPECTED_REJECTIONS
    assert report["rejection_reasons"]["non_neutral"]["of_which_charged"] == 4
    assert sum(reasons.values()) == counts["rejected_total"]


def test_charge_census_and_charge_typed_atoms(tmp_path, synthetic_csv, logger):
    report = run(tmp_path, synthetic_csv, logger)
    census = report["charge_census"]

    assert census["molecule_classes"] == {
        "neutral": 7,
        "net_cation": 2,          # methylammonium, tetramethylammonium
        "net_anion": 1,           # acetate
        "charged_net_neutral": 1,  # nitromethane (N+ / O-)
    }
    assert census["net_charge_distribution"] == {"-1": 1, "0": 8, "1": 2}
    assert census["atom_formal_charge_distribution"]["1"] == 3
    assert census["atom_formal_charge_distribution"]["-1"] == 2

    signatures = {s["signature"]: s["molecules"]
                  for s in census["top_charge_signatures"]}
    assert signatures == {"N+1": 2, "O-1": 1, "N+1|O-1": 1}

    types = {t["atom_type"]: t for t in report["atom_types"]["charged_atom_types"]}
    assert set(types) == {"N+1", "O-1"}
    assert types["N+1"]["atoms"] == 3 and types["N+1"]["molecules"] == 3
    assert types["O-1"]["atoms"] == 2 and types["O-1"]["molecules"] == 2
    assert report["atom_types"]["all_parseable"]["atoms"]["N+1"] == 3
    # Neutral-only accepted set cannot contain charge-typed atoms.
    assert not any(
        "+" in k or "-" in k
        for k in report["atom_types"]["accepted_neutral_only"]["atoms"]
    )
    # Si from methylsilane; H from "[2H]CCO", where RDKit keeps the
    # deuterium as an explicit heavy-list atom rather than folding it away.
    assert report["atom_types"]["out_of_vocabulary_elements"] == {"Si": 1, "H": 1}


def test_charge_strip_counterfactual_is_diagnostic_only(tmp_path, synthetic_csv, logger):
    report = run(tmp_path, synthetic_csv, logger)
    cf = report["charge_strip_counterfactual"]

    assert cf["diagnostic_only"] is True
    assert cf["charged_molecules_tested"] == 4
    assert cf["recovered_by_existing_rules"] == 2   # CN and acetic acid
    assert cf["still_rejected_reasons"] == {"strip_sanitize_failure": 2}
    assert cf["hypothetical_accepted_total"] == 3 + 2
    # The production result is unchanged by the what-if.
    assert report["counts"]["accepted_neutral_only"] == 3
    assert report["encoding_rules"]["neutral_only"] is True

    signatures = {s["signature"]: s["recovered"] for s in cf["signatures"]}
    assert signatures["N+1|O-1"] == 0
    assert signatures["O-1"] == 1


def test_counterfactual_can_be_disabled(tmp_path, synthetic_csv, logger):
    report = run(tmp_path, synthetic_csv, logger, counterfactual=False)
    cf = report["charge_strip_counterfactual"]
    assert cf["performed"] is False
    assert cf["charged_molecules_tested"] == 0
    assert report["counts"]["accepted_neutral_only"] == 3


def test_qed_discrepancy_statistics(tmp_path, synthetic_csv, logger):
    report = run(tmp_path, synthetic_csv, logger)
    qed = report["qed_discrepancy"]

    # CCO has a deliberate +0.2 error in the CSV; RDKit - CSV = -0.2.
    original = qed["csv_vs_rdkit_original_molecule"]["all_parseable"]
    assert original["molecules_compared"] == 11
    assert original["max_abs_delta"] == pytest.approx(0.2, abs=2e-3)
    assert original["above_1e-1"] == 1

    # The reconstructed path (production label) sees the same disagreement
    # for the one accepted molecule with a wrong CSV value.
    reconstructed = qed["csv_vs_rdkit_reconstructed_accepted"]
    assert reconstructed["molecules_compared"] == 3
    assert reconstructed["above_1e-1"] == 1

    # Charged molecules are evaluated on the molecule as written in the CSV,
    # which is exactly where the CSV values were computed: tiny discrepancy.
    charged = qed["csv_vs_rdkit_original_molecule"]["charged"]
    assert charged["molecules_compared"] == 4
    assert charged["max_abs_delta"] < 1e-3

    stripped = qed["csv_vs_rdkit_after_charge_strip_recovered"]
    assert stripped["molecules_compared"] == 2


def test_all_vs_accepted_distribution_comparison(tmp_path, synthetic_csv, logger):
    report = run(tmp_path, synthetic_csv, logger)
    metrics = report["distribution_comparison"]["metrics"]

    # CSV-derived metrics cover EVERY row, even unparseable/empty SMILES.
    assert metrics["qed_csv"]["all"]["n"] == 13
    # Molecule-derived metrics exist only for parseable molecules.
    assert metrics["n_atoms"]["all"]["n"] == 11
    assert metrics["qed_csv"]["accepted"]["n"] == 3
    assert metrics["qed_csv"]["charged"]["n"] == 4
    assert metrics["n_atoms"]["accepted"]["n"] == 3
    assert "ks_all_vs_accepted" in metrics["qed_csv"]
    assert 0.0 <= metrics["qed_csv"]["ks_all_vs_accepted"]["statistic"] <= 1.0
    assert metrics["qed_csv"]["wasserstein_all_vs_accepted"] >= 0.0

    composition = report["distribution_comparison"]["element_composition"]
    assert composition["total_variation_distance"] >= 0.0
    assert pytest.approx(sum(composition["all_parseable"].values())) == 1.0
    assert pytest.approx(sum(composition["accepted_neutral_only"].values())) == 1.0
    assert "Si" in composition["all_parseable"]
    assert "Si" not in composition["accepted_neutral_only"]

    hist = report["distribution_comparison"]["histograms"]["all"]["qed_csv"]
    assert sum(hist["counts"]) == hist["n"] - hist["outside_range"]


def test_json_is_strict_and_file_written(tmp_path, synthetic_csv, logger):
    run(tmp_path, synthetic_csv, logger)
    path = tmp_path / "out" / "chem" / "chemistry_report.json"
    assert path.exists()

    def reject_constant(name):  # NaN / Infinity must never appear
        raise AssertionError(f"non-strict JSON constant: {name}")

    parsed = json.loads(path.read_text(encoding="utf-8"),
                        parse_constant=reject_constant)
    assert parsed["kind"] == "chemistry_diagnostics"
    assert (tmp_path / "out" / "chem" / "chemistry_summary.txt").exists()


def test_figures_are_written(tmp_path, synthetic_csv, logger):
    report = run(tmp_path, synthetic_csv, logger, make_figures=True)
    figures = sorted(p.name for p in (tmp_path / "out" / "figs").glob("*.png"))
    assert figures == sorted([
        "charge_overview.png", "charged_atom_types.png",
        "rejection_reasons.png", "distribution_all_vs_accepted.png",
        "element_composition_all_vs_accepted.png",
        "qed_csv_vs_rdkit_charge.png", "charge_strip_counterfactual.png",
    ])
    assert all((tmp_path / "out" / "figs" / n).stat().st_size > 1000
               for n in figures)
    assert len(report["figures"]) == 7


# ---------------------------------------------------------------------------
# The source dataset is never altered and no molecule is silently dropped.
# ---------------------------------------------------------------------------
def test_source_dataset_unchanged_and_every_row_counted(tmp_path, synthetic_csv, logger):
    before_bytes = synthetic_csv.read_bytes()
    before_digest = sha256_of(synthetic_csv)
    report = run(tmp_path, synthetic_csv, logger, expected_sha256=before_digest)

    assert synthetic_csv.read_bytes() == before_bytes
    assert report["dataset"]["sha256_matches"] is True
    assert report["dataset"]["source_unchanged"] is True
    assert report["dataset"]["molecules_removed_from_source"] == 0
    assert report["counts"]["total_molecules"] == report["dataset"]["records_in_file"]
    accounted = (report["counts"]["accepted_neutral_only"]
                 + report["counts"]["rejected_total"])
    assert accounted == report["counts"]["total_molecules"]


def test_wrong_sha256_is_a_hard_failure(tmp_path, synthetic_csv, logger):
    with pytest.raises(DatasetIdentityError):
        run(tmp_path, synthetic_csv, logger, expected_sha256="0" * 64)
    assert not (tmp_path / "out" / "chem" / "chemistry_report.json").exists()


def test_row_count_mismatch_is_reported_not_fatal(tmp_path, synthetic_csv, logger):
    report = run(tmp_path, synthetic_csv, logger, expected_rows=249_455)
    assert report["dataset"]["record_count_matches_expected"] is False
    assert report["complete"] is True


def test_refuses_to_write_inside_processed_dir(tmp_path, synthetic_csv, logger):
    processed = tmp_path / "data" / "processed"
    with pytest.raises(ValueError, match="processed-data"):
        run(tmp_path, synthetic_csv, logger,
            output_dir=processed / "zinc250k" / "diag")
    assert not processed.exists()


def test_never_creates_processed_dataset(tmp_path, synthetic_csv, logger):
    run(tmp_path, synthetic_csv, logger, make_figures=True)
    assert not (tmp_path / "data").exists()
    assert not list(tmp_path.rglob("splits.pkl"))
    assert not list(tmp_path.rglob("scaler.json"))


# ---------------------------------------------------------------------------
# Subsampling, determinism, and interruption.
# ---------------------------------------------------------------------------
def test_deterministic_sample_indices():
    first = deterministic_sample_indices(1000, 100, seed=7)
    second = deterministic_sample_indices(1000, 100, seed=7)
    other = deterministic_sample_indices(1000, 100, seed=8)
    assert first == second
    assert first != other
    assert first == sorted(first) and len(set(first)) == 100
    assert deterministic_sample_indices(10, 50, seed=1) == list(range(10))
    assert indices_fingerprint(first) == indices_fingerprint(second)
    assert indices_fingerprint(first) != indices_fingerprint(other)


def test_subsample_is_flagged_and_reproducible(tmp_path, synthetic_csv, logger):
    one = run(tmp_path, synthetic_csv, logger, max_molecules=6, sample_seed=3)
    two = run(tmp_path, synthetic_csv, logger, max_molecules=6, sample_seed=3)
    assert one["subsampled"] is True
    assert one["counts"]["total_molecules"] == 6
    assert one["sample_index_fingerprint"] == two["sample_index_fingerprint"]
    assert one["counts"] == two["counts"]


def test_interrupted_run_writes_partial_report(tmp_path, synthetic_csv, logger):
    flag = StopFlag()
    flag.signal_name = "SIGTERM"            # as if SLURM had sent it
    report = run(tmp_path, synthetic_csv, logger, stop_flag=flag)

    assert report["complete"] is False
    assert report["stopped_by"] == "SIGTERM"
    assert report["molecules_processed"] == 1
    out = tmp_path / "out" / "chem"
    assert (out / "chemistry_report.partial.json").exists()
    assert not (out / "chemistry_report.json").exists()


# ---------------------------------------------------------------------------
# Unit tests for helpers.
# ---------------------------------------------------------------------------
def test_atom_type_key():
    assert atom_type_key("C", 0) == "C"
    assert atom_type_key("N", 1) == "N+1"
    assert atom_type_key("O", -1) == "O-1"
    assert atom_type_key("N", 2) == "N+2"


@pytest.mark.parametrize("smiles,expected", [
    ("C[NH3+]", "CN"),
    ("CC(=O)[O-]", "CC(=O)O"),
    ("c1cc[nH+]cc1", "c1ccncc1"),
])
def test_strip_formal_charges_recovers_neutral_form(smiles, expected):
    stripped = strip_formal_charges(Chem.MolFromSmiles(smiles))
    assert Chem.MolToSmiles(stripped) == expected
    assert all(a.GetFormalCharge() == 0 for a in stripped.GetAtoms())


@pytest.mark.parametrize("smiles", ["C[N+](=O)[O-]", "C[N+](C)(C)C"])
def test_strip_fails_when_charge_is_needed_for_valence(smiles):
    with pytest.raises(Exception):
        strip_formal_charges(Chem.MolFromSmiles(smiles))


def test_strip_does_not_mutate_input():
    molecule = Chem.MolFromSmiles("C[NH3+]")
    strip_formal_charges(molecule)
    assert molecule.GetAtomWithIdx(1).GetFormalCharge() == 1


def test_delta_statistics_handles_nan_and_empty():
    stats = delta_statistics([0.5, float("nan"), 0.7], [0.5, 0.1, 0.6])
    assert stats["molecules_compared"] == 2
    assert stats["max_abs_delta"] == pytest.approx(0.1)
    assert stats["mean_signed_delta"] == pytest.approx(-0.05)
    assert delta_statistics([], []) == {"molecules_compared": 0}


def test_read_zinc_table_keeps_every_row(synthetic_csv):
    table = read_zinc_table(synthetic_csv)
    assert len(table) == 13
    assert table.empty_smiles_rows == 1
    assert table.smiles[0] == "CCO"           # trailing newline stripped
    assert np.isfinite(table.qed[0])


def test_read_zinc_table_missing_file_and_columns(tmp_path):
    with pytest.raises(FileNotFoundError, match="internet"):
        read_zinc_table(tmp_path / "absent.csv")
    bad = tmp_path / "bad.csv"
    bad.write_text("smiles,qed\nCCO,0.4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        read_zinc_table(bad)


def test_census_matches_encoder_directly():
    census = ChemistryCensus()
    for smiles, _ in SYNTHETIC_MOLECULES:
        census.add(smiles, 0.5, 1.0, 2.0)
    expected = {}
    for smiles, _ in SYNTHETIC_MOLECULES:
        record, reason = encode_zinc_smiles(smiles)
        if record is None:
            expected[reason] = expected.get(reason, 0) + 1
    assert dict(census.rejection_counts) == expected


# ---------------------------------------------------------------------------
# Command-line interface (exit codes).
# ---------------------------------------------------------------------------
def load_script(repository_root: Path, name: str):
    path = repository_root / "scripts" / name
    spec = importlib.util.spec_from_file_location(
        name.replace(".py", ""), path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_success_and_exit_codes(tmp_path, synthetic_csv, repository_root):
    script = load_script(repository_root, "00_chemistry_diagnostics.py")
    config = str(repository_root / "configs" / "zinc250k.yaml")
    base = ["--config", config, "--csv", str(synthetic_csv),
            "--output-dir", str(tmp_path / "cli_out"),
            "--figures-dir", str(tmp_path / "cli_figs")]

    assert script.main(base + ["--expected-sha256", "skip"]) == 0
    assert (tmp_path / "cli_out" / "chemistry_report.json").exists()
    assert len(list((tmp_path / "cli_figs").glob("*.png"))) == 7

    assert script.main(base + ["--expected-sha256", "f" * 64]) == 3
    missing = ["--config", config, "--csv", str(tmp_path / "nope.csv"),
               "--output-dir", str(tmp_path / "cli_out2"),
               "--expected-sha256", "skip"]
    assert script.main(missing) == 2
    assert script.main(["--config", str(tmp_path / "no.yaml")]) == 4
    assert script.main(base + ["--max-molecules", "0"]) == 4
