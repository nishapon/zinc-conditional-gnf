"""Tests for the QED support / decoder agreement / best-of-k rerank analysis.

The analysis has to be trustworthy in four specific ways, so each is pinned here:

  * the rerank protocol must be honest about SELECTION. Groups are disjoint, every
    candidate is consumed exactly once, a group with nothing scorable yields nothing and
    is still paid for, candidates never cross requested conditions, and k=1 must BE the
    native baseline rather than a separate code path that could drift from it.
  * the finite-sample bookkeeping must be visible: `independent_accepted_ceiling` is
    floor(pool/k), and the min/max across partitions is labelled a protocol spread
    rather than a confidence interval, because the partitions reuse the same candidates.
  * cross-decoder disagreement must be measured PER EMBEDDING, and the two different
    problems -- the label not existing under one decoder, and the label disagreeing in
    value -- must be counted separately.
  * the script must stay read-only and model-free: no torch at module scope, no
    checkpoint, no generation, and a corrupt sample file must not be misreported as a
    provenance mismatch.

Requires rdkit (the cluster env). Torch is NOT required by the script under test.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import QED, rdMolDescriptors

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "33_qed_support_and_rerank.py"
SLURM_PATH = ROOT / "slurm" / "33_qed_support_and_rerank.sh"

# Three real, distinct, decodable molecules with clearly different QED.
BENZENE = "c1ccccc1"
PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
IBUPROFEN = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
ETHANOL = "CCO"


def analysis():
    spec = importlib.util.spec_from_file_location("qed_support_rerank", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rdkit_qed(smiles: str) -> float:
    return float(QED.qed(Chem.MolFromSmiles(smiles)))


def row(qed, *, smiles="CCO", scorable=True):
    """A scored candidate, shaped exactly as `score_entry` returns one."""
    return {"valid": scorable, "sanitizable": scorable, "connected": scorable,
            "fallback": False, "smiles": smiles if scorable else None,
            "qed_stored": qed, "qed": qed if scorable else None,
            "aromatic_rings": 1.0 if scorable else None,
            "rings": 1.0 if scorable else None,
            "heavy_atoms": 10.0 if scorable else None, "scorable": scorable}


def entry_for(smiles, *, valid=True, fallback=False):
    if not valid:
        return {"valid": False, "sanitizable": False, "connected": False,
                "smiles": None, "fallback": fallback, "bonds": 0}
    molecule = Chem.MolFromSmiles(smiles)
    return {"valid": True, "sanitizable": True, "connected": True, "smiles": smiles,
            "qed": float(QED.qed(molecule)),
            "rings": int(molecule.GetRingInfo().NumRings()), "fallback": fallback}


def write_grid(root: Path, *, per_decoder, n_values=(15,), qed_values=(0.4, 0.8),
               temperature=0.2, step=4000, requested=True, count=4):
    """A saved grid in the exact layout `13_generate_nqed.py` writes."""
    for n in n_values:
        for qed in qed_values:
            directory = root / f"T{temperature:g}_N{n}_QED{qed:g}"
            directory.mkdir(parents=True, exist_ok=True)
            metrics = {"flow_checkpoint": "flow/best.pt", "flow_step": step,
                       "autoencoder": {"sha256": "93a89bba" * 8}}
            if requested:
                metrics["requested"] = {"n": n, "qed": qed}
            (directory / "metrics.json").write_text(json.dumps(metrics))
            with (directory / "samples.jsonl").open("w", encoding="utf-8") as handle:
                for index in range(count):
                    record = {"n_node": n, "atom_types": []}
                    for decoder, smiles_list in per_decoder.items():
                        record[decoder] = entry_for(smiles_list[index % len(smiles_list)])
                    handle.write(json.dumps(record) + "\n")
    return root


@pytest.fixture
def grid(tmp_path):
    return write_grid(tmp_path / "early", per_decoder={
        "raw": [BENZENE, ETHANOL, PARACETAMOL, IBUPROFEN],
        "constrained": [BENZENE, ETHANOL, PARACETAMOL, IBUPROFEN],
        "degree": [BENZENE, ETHANOL, PARACETAMOL, IBUPROFEN]})


# ====================================================================================
# 1. The rerank protocol: disjoint, fully paid for, never crossing conditions.
# ====================================================================================
def test_groups_are_disjoint_and_every_candidate_is_consumed_once():
    module = analysis()
    rows = [row(0.1 * i) for i in range(1, 11)]          # 10 scorable candidates
    result = module.rerank_once(rows, target=0.55, k=2, order=list(range(10)))
    assert result["groups_formed"] == 5
    assert result["candidates_consumed"] == 10
    assert result["candidates_discarded_as_remainder"] == 0
    assert result["accepted"] == 5


def test_a_remainder_that_cannot_fill_a_group_is_discarded_and_reported():
    module = analysis()
    rows = [row(0.1 * i) for i in range(1, 11)]
    result = module.rerank_once(rows, target=0.5, k=3, order=list(range(10)))
    assert result["groups_formed"] == 3
    assert result["candidates_consumed"] == 9
    assert result["candidates_discarded_as_remainder"] == 1


def test_best_of_k_picks_the_candidate_closest_to_the_REQUEST():
    """Target matching, not maximisation: the pick must track the request."""
    module = analysis()
    rows = [row(q) for q in (0.10, 0.20, 0.30, 0.40, 0.50,
                             0.60, 0.70, 0.80, 0.90, 1.00)]
    low = module.rerank_once(rows, target=0.22, k=5, order=list(range(10)))
    high = module.rerank_once(rows, target=0.95, k=5, order=list(range(10)))
    assert low["qed_mean"] == pytest.approx((0.20 + 0.60) / 2)
    assert high["qed_mean"] == pytest.approx((0.50 + 1.00) / 2)
    # a maximiser would give the same answer for both, which is the bug this catches
    assert low["qed_mean"] != high["qed_mean"]


def test_a_group_with_nothing_scorable_yields_nothing_and_is_still_paid_for():
    module = analysis()
    rows = [row(None, scorable=False) for _ in range(5)] + [row(0.1 * i) for i in range(5)]
    result = module.rerank_once(rows, target=0.3, k=5, order=list(range(10)))
    assert result["empty_groups"] == 1
    assert result["accepted"] == 1
    assert result["candidates_consumed"] == 10
    assert result["candidates_per_accepted"] == pytest.approx(10.0)


def test_selection_cost_counts_invalid_candidates_even_at_k_one():
    """Native decoding already wastes candidates; the cost column must show it."""
    module = analysis()
    rows = [row(0.5)] * 3 + [row(None, scorable=False)] * 7
    result = module.rerank_once(rows, target=0.5, k=1, order=list(range(10)))
    assert result["accepted"] == 3
    assert result["candidates_consumed"] == 10
    assert result["candidates_per_accepted"] == pytest.approx(10 / 3)


def test_k_one_is_exactly_the_native_baseline_and_ignores_the_partition():
    module = analysis()
    rows = [row(0.1 * i) for i in range(1, 11)] + [row(None, scorable=False)]
    first = module.rerank_cell(rows, target=0.5, k=1, trials=5, seed=1, train_smiles=None)
    second = module.rerank_cell(rows, target=0.5, k=1, trials=1, seed=999,
                                train_smiles=None)
    assert first["is_native_baseline"] is True
    assert first["partitions"] == 1                      # more would be duplicates
    assert first["accepted_per_partition"]["mean"] == 10  # every scorable candidate
    assert first["qed_mean"] == second["qed_mean"]


def test_the_independent_molecule_ceiling_is_reported_for_every_k():
    module = analysis()
    rows = [row(0.5) for _ in range(100)]
    for k, expected in ((1, 100), (2, 50), (5, 20), (10, 10), (20, 5)):
        result = module.rerank_cell(rows, target=0.5, k=k, trials=2, seed=3,
                                    train_smiles=None)
        assert result["independent_accepted_ceiling"] == expected, k


def test_a_k_larger_than_the_pool_is_skipped_not_silently_wrapped():
    module = analysis()
    result = module.rerank_cell([row(0.5)] * 4, target=0.5, k=20, trials=2, seed=3,
                                train_smiles=None)
    assert "skipped" in result and result["k"] == 20


def test_the_partition_spread_is_labelled_as_protocol_not_as_a_confidence_interval():
    module = analysis()
    rows = [row(0.1 * i) for i in range(1, 21)]
    result = module.rerank_cell(rows, target=0.5, k=4, trials=4, seed=7,
                               train_smiles=None)
    assert result["partitions"] == 4
    assert {"mean", "min", "max"} == set(result["qed_mean"])
    note = result["spread_note"].lower()
    assert "not a confidence interval" in note
    assert "same saved candidates" in note


def test_rerank_is_deterministic_for_a_fixed_seed():
    module = analysis()
    rows = [row(0.05 * i) for i in range(1, 21)]
    first = module.rerank_cell(rows, target=0.5, k=4, trials=3, seed=11,
                               train_smiles=None)
    second = module.rerank_cell(rows, target=0.5, k=4, trials=3, seed=11,
                                train_smiles=None)
    assert first["qed_mean"] == second["qed_mean"]
    assert first["uniqueness"] == second["uniqueness"]


def test_different_seeds_change_the_partition_but_not_the_candidate_pool():
    module = analysis()
    rows = [row(0.05 * i) for i in range(1, 21)]
    first = module.rerank_cell(rows, target=0.5, k=4, trials=1, seed=11,
                               train_smiles=None)
    second = module.rerank_cell(rows, target=0.5, k=4, trials=1, seed=12,
                                train_smiles=None)
    assert first["accepted_per_partition"] == second["accepted_per_partition"]
    assert first["independent_accepted_ceiling"] == second["independent_accepted_ceiling"]


def test_reranking_cannot_use_candidates_from_another_requested_condition(grid):
    """Structural: each cell is reranked against its own pool only."""
    module = analysis()
    cells = {}
    for qed in (0.4, 0.8):
        cell = module.load_cell(
            module.cell_directory(grid, temperature=0.2, n=15, qed=qed),
            n=15, qed=qed, decoders=("degree",))
        cells[qed] = module.rerank_cell(cell["candidates"]["degree"], target=qed, k=2,
                                        trials=2, seed=5, train_smiles=None)
    for qed, result in cells.items():
        assert result["pooled_unique_molecules"] <= 4
        assert result["independent_accepted_ceiling"] == 2
    # the only call site passes one cell's own rows and that cell's own target
    source = SCRIPT_PATH.read_text()
    assert source.count("rerank_cell(") == 2            # one definition, one call
    assert "rows, target=qed, k=k" in source


def test_novelty_is_computed_over_the_pooled_unique_accepted_molecules():
    module = analysis()
    rows = [row(0.5, smiles=BENZENE), row(0.5, smiles=PARACETAMOL)]
    result = module.rerank_cell(rows, target=0.5, k=1, trials=1, seed=1,
                                train_smiles={BENZENE})
    assert result["novelty_vs_train_over_pooled_unique"] == pytest.approx(0.5)
    without = module.rerank_cell(rows, target=0.5, k=1, trials=1, seed=1,
                                 train_smiles=None)
    assert without["novelty_vs_train_over_pooled_unique"] is None


def test_uniqueness_falls_when_rerank_keeps_choosing_the_same_molecule():
    """The diversity cost of selection has to be visible, not hidden."""
    module = analysis()
    # one molecule is nearest the target in every group, so selection collapses onto it
    rows = []
    for index in range(20):
        rows.append(row(0.5, smiles=BENZENE) if index % 5 == 0
                    else row(0.1, smiles=f"C{'C' * (index + 1)}O"))
    native = module.rerank_once(rows, target=0.5, k=1, order=list(range(20)))
    selected = module.rerank_once(rows, target=0.5, k=5, order=list(range(20)))
    assert selected["unique_molecules"] == 1
    assert selected["uniqueness"] == pytest.approx(0.25)
    assert native["unique_molecules"] > selected["unique_molecules"]


# ====================================================================================
# 2. Support statistics.
# ====================================================================================
def test_summarise_values_reports_quantiles_iqr_and_high_qed_mass():
    module = analysis()
    values = list(np.linspace(0.0, 1.0, 101))
    block = module.summarise_values(values)
    assert block["molecules"] == 101
    assert block["quantiles"]["p50"] == pytest.approx(0.5)
    assert block["iqr"] == pytest.approx(0.5, abs=1e-9)
    assert block["fraction_at_least_0p75"] == pytest.approx(26 / 101)
    assert block["min"] == 0.0 and block["max"] == 1.0


def test_summarise_values_ignores_missing_and_non_finite_entries():
    module = analysis()
    block = module.summarise_values([0.2, None, float("nan"), float("inf"), 0.8])
    assert block["molecules"] == 2
    assert block["mean"] == pytest.approx(0.5)


def test_summarise_values_handles_an_empty_population():
    module = analysis()
    assert module.summarise_values([])["molecules"] == 0
    assert module.summarise_values([None, float("nan")])["molecules"] == 0


def test_support_is_stratified_by_validity_and_scorability():
    module = analysis()
    rows = [row(0.5, smiles=BENZENE), row(0.9, smiles=PARACETAMOL),
            row(None, scorable=False), row(None, scorable=False)]
    block = module.support_block(rows)
    assert block["candidates"] == 4 and block["valid"] == 2 and block["scorable"] == 2
    assert block["validity"] == pytest.approx(0.5)
    assert block["qed"]["molecules"] == 2
    assert block["unique_molecules"] == 2
    assert block["uniqueness_over_scorable"] == pytest.approx(1.0)


def test_support_is_reported_separately_for_each_decoder(grid):
    module = analysis()
    cell = module.load_cell(module.cell_directory(grid, temperature=0.2, n=15, qed=0.4),
                            n=15, qed=0.4, decoders=("raw", "constrained", "degree"))
    for decoder in ("raw", "constrained", "degree"):
        block = module.support_block(cell["candidates"][decoder])
        assert block["candidates"] == 4
        assert block["qed"]["molecules"] == 4


# ====================================================================================
# 3. Ingestion: recomputation, provenance, and the per-sample record shape.
# ====================================================================================
def test_score_entry_recomputes_qed_and_aromatic_rings_from_the_smiles():
    module = analysis()
    scored = module.score_entry(entry_for(PARACETAMOL))
    assert scored["scorable"] is True
    assert scored["qed"] == pytest.approx(rdkit_qed(PARACETAMOL))
    assert scored["aromatic_rings"] == pytest.approx(
        float(rdMolDescriptors.CalcNumAromaticRings(Chem.MolFromSmiles(PARACETAMOL))))
    assert scored["heavy_atoms"] == pytest.approx(
        float(Chem.MolFromSmiles(PARACETAMOL).GetNumHeavyAtoms()))


def test_score_entry_does_not_trust_the_stored_qed():
    """The recomputation is independent, and the disagreement is reported not hidden."""
    module = analysis()
    tampered = entry_for(BENZENE) | {"qed": 0.123}
    scored = module.score_entry(tampered)
    assert scored["qed_stored"] == 0.123
    assert scored["qed"] == pytest.approx(rdkit_qed(BENZENE))
    assert scored["qed"] != scored["qed_stored"]


def test_an_invalid_or_unparseable_entry_is_not_scorable():
    module = analysis()
    assert module.score_entry(None)["scorable"] is False
    assert module.score_entry({})["scorable"] is False
    assert module.score_entry(entry_for(ETHANOL, valid=False))["scorable"] is False
    nonsense = {"valid": True, "smiles": "not a molecule"}
    assert module.score_entry(nonsense)["scorable"] is False


def test_load_cell_flags_a_stored_qed_that_disagrees_with_recomputation(tmp_path):
    module = analysis()
    root = tmp_path / "tampered"
    directory = root / "T0.2_N15_QED0.4"
    directory.mkdir(parents=True)
    (directory / "metrics.json").write_text(json.dumps({"requested": {"n": 15, "qed": 0.4}}))
    bad = entry_for(BENZENE) | {"qed": 0.0}
    (directory / "samples.jsonl").write_text(json.dumps({"n_node": 15, "degree": bad}) + "\n")
    cell = module.load_cell(directory, n=15, qed=0.4, decoders=("degree",))
    assert cell["stored_qed_matches_recomputation"] is False
    assert cell["max_stored_vs_recomputed_qed_gap"] > 0.1


def test_load_cell_accepts_a_consistent_grid(grid):
    module = analysis()
    cell = module.load_cell(module.cell_directory(grid, temperature=0.2, n=15, qed=0.8),
                            n=15, qed=0.8, decoders=("degree",))
    assert cell["candidates_in_pool"] == 4
    assert cell["requested_n_consistent"] is True
    assert cell["stored_qed_matches_recomputation"] is True
    assert cell["flow_step"] == 4000


def test_a_mislabelled_cell_is_a_provenance_error(tmp_path):
    """A directory named for one condition but recording another invalidates every
    per-condition number, so it must stop the run rather than be averaged in."""
    module = analysis()
    root = tmp_path / "mislabelled"
    directory = root / "T0.2_N15_QED0.4"
    directory.mkdir(parents=True)
    (directory / "metrics.json").write_text(
        json.dumps({"requested": {"n": 15, "qed": 0.8}}))      # says 0.8, named 0.4
    (directory / "samples.jsonl").write_text(
        json.dumps({"n_node": 15, "degree": entry_for(BENZENE)}) + "\n")
    with pytest.raises(module.ProvenanceError):
        module.load_cell(directory, n=15, qed=0.4, decoders=("degree",))


def test_a_missing_sample_file_names_save_samples_in_the_error(tmp_path):
    module = analysis()
    directory = tmp_path / "T0.2_N15_QED0.4"
    directory.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="save-samples"):
        module.load_cell(directory, n=15, qed=0.4, decoders=("degree",))


def test_an_empty_sample_file_is_rejected(tmp_path):
    module = analysis()
    directory = tmp_path / "T0.2_N15_QED0.4"
    directory.mkdir(parents=True)
    (directory / "samples.jsonl").write_text("")
    with pytest.raises(FileNotFoundError):
        module.load_cell(directory, n=15, qed=0.4, decoders=("degree",))


def test_the_cell_directory_matches_the_generator_naming_convention():
    module = analysis()
    assert module.cell_directory(Path("/x"), temperature=0.2, n=15, qed=0.4).name \
        == "T0.2_N15_QED0.4"
    assert module.cell_directory(Path("/x"), temperature=1.0, n=30, qed=0.8).name \
        == "T1_N30_QED0.8"
    # the same f-string format as scripts/13_generate_nqed.py
    generator = (ROOT / "scripts" / "13_generate_nqed.py").read_text()
    assert 'f"T{temperature:g}_N{target_n}_QED{target_qed:g}"' in generator


# ====================================================================================
# 4. Cross-decoder disagreement, per embedding.
# ====================================================================================
def two_decoder_cell(module, pairs):
    """One synthetic cell where index i is the same embedding under both decoders."""
    candidates = {"constrained": [], "degree": []}
    for left, right in pairs:
        candidates["constrained"].append(
            module.score_entry(entry_for(left) if left else entry_for(ETHANOL, valid=False)))
        candidates["degree"].append(
            module.score_entry(entry_for(right) if right else entry_for(ETHANOL, valid=False)))
    return {"candidates": candidates}


def test_disagreement_is_measured_per_embedding_not_between_population_means():
    module = analysis()
    cell = two_decoder_cell(module, [(BENZENE, PARACETAMOL), (BENZENE, PARACETAMOL)])
    block = module.disagreement_block([cell], ("constrained", "degree"))
    pair = block["pairs"]["constrained_vs_degree"]
    expected = abs(rdkit_qed(BENZENE) - rdkit_qed(PARACETAMOL))
    assert pair["jointly_scorable"] == 2
    assert pair["absolute_qed_difference"]["median"] == pytest.approx(expected)
    assert pair["identical_molecule_fraction"] == pytest.approx(0.0)
    assert "per generated embedding" in block["note"]


def test_identical_decodings_give_zero_disagreement():
    module = analysis()
    cell = two_decoder_cell(module, [(IBUPROFEN, IBUPROFEN)] * 3)
    pair = module.disagreement_block([cell], ("constrained", "degree"))["pairs"][
        "constrained_vs_degree"]
    assert pair["absolute_qed_difference"]["max"] == pytest.approx(0.0, abs=1e-12)
    assert pair["identical_molecule_fraction"] == pytest.approx(1.0)


def test_a_label_existing_under_only_one_decoder_is_counted_separately():
    """Two different problems: the label missing, and the label disagreeing."""
    module = analysis()
    cell = two_decoder_cell(module, [(BENZENE, None), (None, PARACETAMOL),
                                     (None, None), (BENZENE, BENZENE)])
    pair = module.disagreement_block([cell], ("constrained", "degree"))["pairs"][
        "constrained_vs_degree"]
    assert pair["jointly_scorable"] == 1
    assert pair["scorable_only_under_first"] == 1
    assert pair["scorable_only_under_second"] == 1
    assert pair["scorable_under_neither"] == 1
    assert pair["label_exists_under_exactly_one"] == 2


def test_every_decoder_pair_is_reported():
    module = analysis()
    cell = {"candidates": {d: [module.score_entry(entry_for(BENZENE))]
                           for d in ("raw", "constrained", "degree")}}
    block = module.disagreement_block([cell], ("raw", "constrained", "degree"))
    assert set(block["pairs"]) == {"raw_vs_constrained", "raw_vs_degree",
                                   "constrained_vs_degree"}


def test_disagreement_thresholds_are_reported_as_fractions():
    module = analysis()
    cell = two_decoder_cell(module, [(BENZENE, PARACETAMOL), (IBUPROFEN, IBUPROFEN)])
    difference = module.disagreement_block([cell], ("constrained", "degree"))["pairs"][
        "constrained_vs_degree"]["absolute_qed_difference"]
    for key in ("fraction_above_0p05", "fraction_above_0p1", "fraction_above_0p2"):
        assert 0.0 <= difference[key] <= 1.0


# ====================================================================================
# 5. Responsiveness at fixed N.
# ====================================================================================
def test_responsiveness_is_the_difference_of_cell_means_and_says_so():
    module = analysis()
    cells = {"0.40": {"rerank": {"1": {"qed_mean": {"mean": 0.30}}}},
             "0.60": {"rerank": {"1": {"qed_mean": {"mean": 0.45}}}},
             "0.80": {"rerank": {"1": {"qed_mean": {"mean": 0.55}}}}}
    result = module.responsiveness(cells, k=1)
    assert result["delta_low_to_high"] == pytest.approx(0.25)
    assert result["monotonic_increasing"] is True
    assert "not a per-molecule paired statistic" in result["note"].lower()


def test_responsiveness_flags_a_non_monotonic_response():
    module = analysis()
    cells = {"0.40": {"rerank": {"1": {"qed_mean": {"mean": 0.43}}}},
             "0.60": {"rerank": {"1": {"qed_mean": {"mean": 0.61}}}},
             "0.80": {"rerank": {"1": {"qed_mean": {"mean": 0.56}}}}}
    result = module.responsiveness(cells, k=1)
    assert result["monotonic_increasing"] is False
    assert result["delta_low_to_high"] == pytest.approx(0.13)


def test_responsiveness_refuses_to_report_when_a_condition_has_no_result():
    module = analysis()
    cells = {"0.40": {"rerank": {"1": {"qed_mean": None}}},
             "0.80": {"rerank": {"1": {"qed_mean": {"mean": 0.5}}}}}
    assert "note" in module.responsiveness(cells, k=1)
    assert "delta_low_to_high" not in module.responsiveness(cells, k=1)


# ====================================================================================
# 6. The script is read-only, model-free, and honest about what rerank is.
# ====================================================================================
def _calls():
    tree = ast.parse(SCRIPT_PATH.read_text())
    return [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]


def flat_source(path=SCRIPT_PATH) -> str:
    """Source with runs of whitespace collapsed, so a wrapped sentence still matches."""
    return " ".join(path.read_text().split())


def test_the_script_never_imports_torch_at_module_scope():
    """It loads no model, so it must not pay for -- or depend on -- torch."""
    tree = ast.parse(SCRIPT_PATH.read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(not a.name.startswith("torch") for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("torch")
            assert (node.module or "") not in {"zinc_gnf.dataset", "zinc_gnf.generation",
                                               "zinc_gnf.metrics_generation"}


def test_torch_only_dependencies_are_imported_lazily():
    """`zinc_gnf.dataset` and `zinc_gnf.metrics_generation` both pull in torch."""
    source = SCRIPT_PATH.read_text()
    for module_name in ("zinc_gnf.dataset", "zinc_gnf.metrics_generation"):
        assert f"from {module_name} import" in source
    hosts = set()
    for function in [n for n in ast.walk(ast.parse(source))
                     if isinstance(n, ast.FunctionDef)]:
        for node in ast.walk(function):
            if isinstance(node, ast.ImportFrom) and (node.module or "") in {
                    "zinc_gnf.dataset", "zinc_gnf.metrics_generation"}:
                hosts.add(function.name)
    assert hosts == {"internal_diversity_of", "load_train_smiles"}


def test_the_script_loads_no_checkpoint_and_generates_nothing():
    calls = _calls()
    for forbidden in ("torch.save", "torch.load", "load_flow", "load_autoencoder",
                      "decode_embeddings", "backward"):
        assert not any(forbidden in call for call in calls), forbidden
    source = SCRIPT_PATH.read_text()
    for forbidden in ("flow.inverse", "torch.optim", "Optimizer", "Trainer(",
                      "zero_grad", "requires_grad"):
        assert forbidden not in source, forbidden


def test_rerank_is_labelled_postprocessing_and_not_an_upper_bound():
    """Both claims must be in the artefact itself, not only in the covering message."""
    source = flat_source()
    assert "POSTPROCESSING" in source
    assert "not intrinsic conditional generation" in source
    assert "FINITE-CANDIDATE PERFORMANCE REFERENCE" in source
    assert "NOT a universal upper bound" in source
    assert "A retrained model can place mass where the saved pool happens to be thin" \
        in source


def test_the_selection_bias_of_reranking_is_stated_not_implied():
    source = flat_source()
    assert "optimistically biased BY CONSTRUCTION" in source
    assert "rerank does not optimise" in source


def test_the_precommitted_thresholds_are_constants_and_framed_as_evidence():
    module = analysis()
    assert module.SUPPORT_MIN_FRACTION_HIGH == 0.05
    assert module.SUPPORT_MIN_IQR == 0.15
    assert module.LABEL_MAX_MEDIAN_DISAGREEMENT == 0.05
    assert module.RERANK_TARGET_MAE == 0.05
    assert module.DIVERSITY_RETENTION == 0.90
    assert "EVIDENCE INPUTS, not an automatic GO/NO-GO" in flat_source()


def test_the_verdict_separates_support_from_label_definedness():
    module = analysis()
    payload = {"runs": {"early": {
        "support_by_n": {"15": {"degree": {"qed": {
            "fraction_at_least_0p75": 0.20, "iqr": 0.25, "quantiles": {"p50": 0.6},
            "molecules": 100}}}},
        "decoder_disagreement": {"pairs": {"constrained_vs_degree": {
            "absolute_qed_difference": {"median": 0.01}}}},
        "cells": {"15": {"0.40": {"rerank": {
            "1": {"internal_diversity_over_pooled_unique": 0.8},
            "10": {"qed_mae_vs_target": {"mean": 0.02},
                   "internal_diversity_over_pooled_unique": 0.78}}}}},
    }}}
    verdict = module.verdict(payload, decoder="degree")
    checks = verdict["by_checkpoint"]["early"]
    assert checks["by_n"]["15"]["support_sufficient"] is True
    assert checks["label_well_defined"] is True
    assert checks["rerank_k10_hits_target_mae"] is True
    assert checks["rerank_k10_diversity_retention"] == pytest.approx(0.975)
    assert checks["rerank_k10_holds_diversity"] is True


def test_the_verdict_marks_thin_support_and_an_ambiguous_label_as_failures():
    module = analysis()
    payload = {"runs": {"final": {
        "support_by_n": {"30": {"degree": {"qed": {
            "fraction_at_least_0p75": 0.001, "iqr": 0.06, "quantiles": {"p50": 0.5},
            "molecules": 300}}}},
        "decoder_disagreement": {"pairs": {"constrained_vs_degree": {
            "absolute_qed_difference": {"median": 0.18}}}},
        "cells": {"30": {"0.80": {"rerank": {
            "1": {"internal_diversity_over_pooled_unique": 0.85},
            "10": {"qed_mae_vs_target": {"mean": 0.21},
                   "internal_diversity_over_pooled_unique": 0.50}}}}},
    }}}
    checks = module.verdict(payload, decoder="degree")["by_checkpoint"]["final"]
    assert checks["by_n"]["30"]["support_sufficient"] is False
    assert checks["label_well_defined"] is False
    assert checks["rerank_k10_hits_target_mae"] is False
    assert checks["rerank_k10_holds_diversity"] is False


def test_the_verdict_explains_which_failure_retraining_could_escape():
    module = analysis()
    text = module.verdict({"runs": {}}, decoder="degree")["interpretation"]
    assert "could in principle escape" in text
    assert "no amount of retraining fixes that" in text


# ====================================================================================
# 7. CLI contract and the SLURM wrapper.
# ====================================================================================
def test_main_returns_an_exit_code_rather_than_raising():
    module = analysis()
    assert module.main(["--run", "a=/nope", "--output-dir", "/dev/null/x"]) == 4


def test_bad_arguments_are_rejected_with_code_four(tmp_path):
    module = analysis()
    base = ["--output-dir", str(tmp_path / "out"), "--skip-novelty"]
    assert module.main(["--run", "nopath", *base]) == 4
    assert module.main(["--run", "a=/x", "--k-values", "2,5", *base]) == 4
    assert module.main(["--run", "a=/x", "--headline", "raw",
                        "--decoders", "degree", *base]) == 4
    assert module.main(["--run", "a=/x", "--qed-values", "0.4", *base]) == 4
    assert module.main(["--run", "a=/x", "--decoders", "nonsense", *base]) == 4
    assert module.main(["--run", "a=/x", "--partitions", "0", *base]) == 4


def test_a_missing_grid_directory_is_exit_two(tmp_path):
    module = analysis()
    assert module.main(["--run", f"a={tmp_path / 'absent'}",
                        "--output-dir", str(tmp_path / "out"), "--skip-novelty"]) == 2


def test_a_mislabelled_grid_is_exit_three(tmp_path):
    module = analysis()
    root = write_grid(tmp_path / "bad", per_decoder={"degree": [BENZENE]},
                      qed_values=(0.4,), count=2)
    metrics = root / "T0.2_N15_QED0.4" / "metrics.json"
    metrics.write_text(json.dumps({"requested": {"n": 15, "qed": 0.7}}))
    assert module.main(["--run", f"bad={root}", "--output-dir", str(tmp_path / "out"),
                        "--target-n", "15", "--qed-values", "0.4,0.8",
                        "--decoders", "degree", "--headline", "degree",
                        "--k-values", "1", "--skip-novelty"]) == 3


def test_a_corrupt_sample_file_is_not_misreported_as_a_provenance_error(tmp_path):
    """`json.JSONDecodeError` is a ValueError; it must not masquerade as exit 3."""
    module = analysis()
    root = write_grid(tmp_path / "corrupt", per_decoder={"degree": [BENZENE]},
                      qed_values=(0.4, 0.8), count=2)
    (root / "T0.2_N15_QED0.4" / "samples.jsonl").write_text("{not json\n")
    code = module.main(["--run", f"c={root}", "--output-dir", str(tmp_path / "out"),
                        "--target-n", "15", "--qed-values", "0.4,0.8",
                        "--decoders", "degree", "--headline", "degree",
                        "--k-values", "1", "--skip-novelty"])
    assert code == 1


def test_a_complete_run_writes_both_reports_and_exits_zero(tmp_path):
    module = analysis()
    root = write_grid(tmp_path / "ok", per_decoder={
        "constrained": [BENZENE, ETHANOL, PARACETAMOL, IBUPROFEN],
        "degree": [BENZENE, ETHANOL, PARACETAMOL, IBUPROFEN]},
        qed_values=(0.4, 0.8), count=4)
    out = tmp_path / "out"
    code = module.main(["--run", f"early={root}", "--output-dir", str(out),
                        "--target-n", "15", "--qed-values", "0.4,0.8",
                        "--decoders", "constrained,degree", "--headline", "degree",
                        "--k-values", "1,2", "--partitions", "2", "--skip-novelty"])
    assert code == 0
    payload = json.loads((out / "qed_support_rerank.json").read_text())
    assert "POSTPROCESSING" in payload["framing"]
    assert set(payload["runs"]) == {"early"}
    run = payload["runs"]["early"]
    assert set(run["cells"]["15"]) == {"0.40", "0.80"}
    assert run["support_by_n"]["15"]["degree"]["candidates"] == 8
    assert run["cells"]["15"]["0.40"]["rerank"]["1"]["is_native_baseline"] is True
    assert "constrained_vs_degree" in run["decoder_disagreement"]["pairs"]
    assert "15" in run["responsiveness_by_n"]
    text = (out / "qed_support_rerank.txt").read_text()
    assert "BEST-OF-k RERANK (POSTPROCESSING)" in text
    assert "k=1 IS the native model" in text


def test_the_run_argument_accepts_two_checkpoints(tmp_path):
    module = analysis()
    early = write_grid(tmp_path / "early", per_decoder={"degree": [BENZENE, ETHANOL]},
                       qed_values=(0.4, 0.8), count=2, step=4000)
    final = write_grid(tmp_path / "final", per_decoder={"degree": [BENZENE, IBUPROFEN]},
                       qed_values=(0.4, 0.8), count=2, step=50000)
    out = tmp_path / "out"
    code = module.main(["--run", f"early={early}", "--run", f"final={final}",
                        "--output-dir", str(out), "--target-n", "15",
                        "--qed-values", "0.4,0.8", "--decoders", "degree",
                        "--headline", "degree", "--k-values", "1,2",
                        "--skip-novelty"])
    assert code == 0
    payload = json.loads((out / "qed_support_rerank.json").read_text())
    assert set(payload["runs"]) == {"early", "final"}
    assert payload["runs"]["early"]["provenance"]["flow_step"] == 4000
    assert payload["runs"]["final"]["provenance"]["flow_step"] == 50000
    assert set(payload["verdict"]["by_checkpoint"]) == {"early", "final"}


def test_slurm_flags_are_a_subset_of_the_scripts_arguments():
    parser_flags = {line.strip().split('"')[1]
                    for line in SCRIPT_PATH.read_text().splitlines()
                    if line.strip().startswith('parser.add_argument("--')}
    assert parser_flags
    used = {line.strip().split()[0] for line in SLURM_PATH.read_text().splitlines()
            if line.strip().startswith("--")}
    assert used <= parser_flags, used - parser_flags
    for required in ("--run", "--output-dir", "--k-values", "--temperature", "--seed"):
        assert required in used, required


def test_slurm_refuses_the_login_node_and_names_both_saved_grids():
    source = SLURM_PATH.read_text()
    assert 'if [[ -z "${SLURM_JOB_ID:-}" ]]' in source
    assert "set -euo pipefail" in source
    assert "qedcontrast_seed3_early_paired" in source
    assert "qedcontrast_seed3_full_paired" in source
    assert "never regenerates them" in source


def test_slurm_is_read_only_and_writes_only_its_own_output():
    source = SLURM_PATH.read_text()
    assert 'OUT="${OUT:-outputs/metrics/qed_support_rerank}"' in source
    for line in [l for l in source.splitlines()
                 if l.strip() and not l.strip().startswith("#")]:
        assert line.strip().split()[0] not in {"rm", "mv", "cp", "truncate", "dd"}, line
    assert "imports no torch" in source


def test_slurm_checks_its_inputs_and_explains_the_temperature_trap():
    source = SLURM_PATH.read_text()
    assert "exit 66" in source
    assert "TEMP must match the temperature the grids were generated at" in source
