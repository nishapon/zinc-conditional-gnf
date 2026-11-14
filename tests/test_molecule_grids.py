"""Tests for the molecule-grid rendering.

The images are meant to be evidence, so the tests pin the three properties that make
them evidence rather than decoration:

  * the row selection is QUALITY-BLIND. It depends only on the noise id, so it cannot
    drift toward the best-looking molecules however often it is re-run.
  * failures are DRAWN IN PLACE. An invalid or undrawable sample becomes a labelled
    placeholder in its own grid position, never a substituted neighbour. At N=30 raw
    validity is 0-1%, so most raw cells should be placeholders, and a renderer that
    silently backfilled them would hide exactly the result we are reporting.
  * baseline and candidate are PAIRED BY NOISE ID, and the renderer refuses to build a
    comparison when the two models share no draws.

`select_noise_ids`, `Cell` and `cell_from_record` are pure. Rendering needs Pillow;
drawing an actual molecule needs RDKit and skips without it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zinc_gnf.molecule_grids import (
    Cell,
    best_of_k_gallery,
    cell_from_record,
    comparison_grid,
    render_grid,
    select_noise_ids,
)


def record(noise_id, *, raw_valid=True, degree_valid=True, qed=0.55, atoms=15,
           rings=1, smiles="c1ccccc1"):
    def entry(valid):
        return {"valid": valid, "fallback": False,
                "smiles": smiles if valid else None,
                "qed": qed if valid else None,
                "heavy_atoms": atoms if valid else None,
                "aromatic_rings": rings if valid else None}
    return {"noise_id": noise_id, "requested_n": atoms, "requested_qed": 0.6,
            "raw": entry(raw_valid), "degree": entry(degree_valid)}


# ---------------------------------------------------------- quality-blind choice ---
def test_selection_depends_only_on_the_noise_id():
    """Re-ordering the input, or changing molecule quality, cannot change the rows."""
    forward = select_noise_ids([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 4)
    backward = select_noise_ids(list(reversed(range(10))), 4)
    assert forward == backward


def test_selection_is_evenly_spaced_not_the_first_k():
    chosen = select_noise_ids(list(range(100)), 5)
    assert chosen == [0, 20, 40, 60, 80]
    assert chosen != list(range(5))


def test_selection_returns_everything_when_the_pool_is_small():
    assert select_noise_ids([3, 1, 2], 10) == [1, 2, 3]
    assert select_noise_ids([], 4) == []
    assert select_noise_ids([1, 2, 3], 0) == []


def test_selection_deduplicates_and_sorts():
    assert select_noise_ids([5, 5, 1, 1, 9], 3) == [1, 5, 9]


def test_selection_is_stable_across_repeated_calls():
    pool = list(range(37))
    assert select_noise_ids(pool, 6) == select_noise_ids(pool, 6)


# ------------------------------------------------------------------ cell labels ---
def test_a_valid_cell_labels_every_requested_quantity():
    cell = Cell(smiles="c1ccccc1", model="best", decoder="degree",
                requested_qed=0.8, actual_qed=0.512, heavy_atoms=23,
                aromatic_rings=2)
    lines = cell.legend()
    assert "best / degree" in lines[0]
    assert "0.80" in lines[1] and "0.512" in lines[1]
    assert "atoms 23" in lines[1] and "arom 2" in lines[1]


def test_an_invalid_cell_says_so_and_keeps_the_request():
    cell = Cell(smiles=None, model="cand", decoder="raw", requested_qed=0.4,
                actual_qed=None, heavy_atoms=None, aromatic_rings=None,
                status="invalid")
    lines = cell.legend()
    assert "INVALID" in lines[1]
    assert "0.40" in lines[1]


def test_a_cell_with_missing_chemistry_renders_na_rather_than_zero():
    cell = Cell(smiles="CCO", model="m", decoder="degree", requested_qed=0.6,
                actual_qed=None, heavy_atoms=None, aromatic_rings=None)
    assert "n/a" in cell.legend()[1]


def test_cell_from_record_marks_an_invalid_decoder_entry():
    built = cell_from_record(record(0, raw_valid=False), model="best", decoder="raw",
                             requested_qed=0.4)
    assert built.status == "invalid" and built.smiles is None
    good = cell_from_record(record(0), model="best", decoder="degree",
                            requested_qed=0.4)
    assert good.status == "ok" and good.smiles == "c1ccccc1"


def test_cell_from_record_carries_the_requested_qed_not_the_stored_one():
    """The figure must label what was ASKED for, independent of the record."""
    built = cell_from_record(record(0), model="m", decoder="degree",
                             requested_qed=0.8)
    assert built.requested_qed == pytest.approx(0.8)


# --------------------------------------------------------------- failures drawn ---
def test_invalid_cells_become_placeholders_in_their_own_position(tmp_path):
    pytest.importorskip("PIL")
    rows = [(7, [Cell(None, "best", "raw", 0.4, None, None, None, status="invalid"),
                 Cell(None, "cand", "raw", 0.4, None, None, None, status="invalid")])]
    manifest = render_grid(rows, title="t", subtitle="s",
                           output=tmp_path / "grid.png")
    assert Path(manifest["path"]).is_file()
    assert manifest["cells"] == 2
    assert manifest["invalid"] == 2 and manifest["drawn"] == 0
    assert manifest["placeholder_rate"] == pytest.approx(1.0)


def test_the_manifest_reports_the_placeholder_rate(tmp_path):
    pytest.importorskip("PIL")
    pytest.importorskip("rdkit")
    rows = [(1, [Cell("c1ccccc1", "m", "degree", 0.6, 0.55, 6, 1),
                 Cell(None, "m", "raw", 0.6, None, None, None, status="invalid")])]
    manifest = render_grid(rows, title="t", subtitle="s", output=tmp_path / "g.png")
    assert manifest["drawn"] == 1 and manifest["invalid"] == 1
    assert manifest["placeholder_rate"] == pytest.approx(0.5)


def test_an_undrawable_smiles_is_marked_not_dropped(tmp_path):
    pytest.importorskip("PIL")
    pytest.importorskip("rdkit")
    rows = [(1, [Cell("this is not a molecule", "m", "raw", 0.6, 0.5, 5, 0)])]
    manifest = render_grid(rows, title="t", subtitle="s", output=tmp_path / "g.png")
    assert manifest["drawn"] == 0
    assert manifest["invalid"] + manifest["undrawable"] == 1


def test_ragged_rows_are_refused(tmp_path):
    pytest.importorskip("PIL")
    rows = [(1, [Cell(None, "a", "raw", 0.4, None, None, None, status="invalid")]),
            (2, [Cell(None, "a", "raw", 0.4, None, None, None, status="invalid"),
                 Cell(None, "b", "raw", 0.4, None, None, None, status="invalid")])]
    with pytest.raises(ValueError, match="same number of cells"):
        render_grid(rows, title="t", subtitle="s", output=tmp_path / "g.png")


def test_an_empty_grid_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no rows"):
        render_grid([], title="t", subtitle="s", output=tmp_path / "g.png")


# ------------------------------------------------------------------ pairing ---
def test_the_comparison_refuses_models_with_no_shared_noise(tmp_path):
    pytest.importorskip("PIL")
    with pytest.raises(ValueError, match="share no noise ids"):
        comparison_grid([record(0)], [record(99)], n=15, requested_qed=0.4,
                        output=tmp_path / "g.png")


def test_the_comparison_pairs_by_noise_id_and_labels_the_rule(tmp_path):
    """All-invalid records, so the layout is exercised without needing RDKit."""
    pytest.importorskip("PIL")
    baseline = [record(i, raw_valid=False, degree_valid=False) for i in range(8)]
    candidate = [record(i, raw_valid=False, degree_valid=False) for i in range(8)]
    manifest = comparison_grid(baseline, candidate, n=15, requested_qed=0.4,
                               samples=3, output=tmp_path / "cmp.png")
    # 3 rows x (2 models x 2 decoders)
    assert manifest["rows"] == 3 and manifest["columns"] == 4
    assert manifest["cells"] == 12


def test_the_comparison_only_uses_draws_present_in_both(tmp_path):
    pytest.importorskip("PIL")
    baseline = [record(i, raw_valid=False, degree_valid=False) for i in range(6)]
    candidate = [record(i, raw_valid=False, degree_valid=False) for i in (0, 1, 2)]
    manifest = comparison_grid(baseline, candidate, n=15, requested_qed=0.4,
                               samples=6, output=tmp_path / "cmp.png")
    assert manifest["rows"] == 3


# ------------------------------------------------------- selection kept separate ---
def test_the_selection_gallery_states_its_budget_and_yield(tmp_path):
    pytest.importorskip("PIL")
    pytest.importorskip("rdkit")
    records = [record(i, raw_valid=False, qed=0.3 + 0.02 * i) for i in range(20)]
    manifest = best_of_k_gallery(records, n=15, requested_qed=0.6, k=10,
                                 decoder="degree", samples=2,
                                 output=tmp_path / "sel.png")
    assert manifest["rows"] == 2 and manifest["columns"] == 1


def test_the_selection_gallery_refuses_an_incomplete_group(tmp_path):
    pytest.importorskip("PIL")
    records = [record(i) for i in range(4)]
    with pytest.raises(ValueError, match="fewer than one full group"):
        best_of_k_gallery(records, n=15, requested_qed=0.6, k=10,
                          output=tmp_path / "sel.png")


def test_the_selection_gallery_refuses_unscorable_records(tmp_path):
    records = [record(i, degree_valid=False) for i in range(20)]
    with pytest.raises(ValueError, match="no scorable records"):
        best_of_k_gallery(records, n=15, requested_qed=0.6, k=10,
                          output=tmp_path / "sel.png")


def test_the_selection_gallery_keeps_the_nearest_to_target():
    """The selection rule itself, checked without rendering."""
    records = [record(i, qed=q) for i, q in enumerate(
        [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])]
    groups = [records[0:10]]
    kept = [min(group, key=lambda r: abs(r["degree"]["qed"] - 0.62))
            for group in groups]
    assert kept[0]["degree"]["qed"] == pytest.approx(0.60)


# ------------------------------------------------------------- module hygiene ---
def test_the_module_loads_no_model_and_generates_nothing():
    import ast

    source = (Path(__file__).resolve().parents[1] / "zinc_gnf"
              / "molecule_grids.py").read_text()
    tree = ast.parse(source)
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top.update(a.name.split(".")[0] for a in node.names)
        if isinstance(node, ast.ImportFrom):
            top.add((node.module or "").split(".")[0])
    assert not top & {"torch", "rdkit", "PIL"}, top      # all lazy
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    for forbidden in ("load_flow", "decode_embeddings", "torch.load"):
        assert not any(forbidden in call for call in calls), forbidden


def test_the_module_documents_why_placeholders_are_not_substituted():
    source = " ".join((Path(__file__).resolve().parents[1] / "zinc_gnf"
                       / "molecule_grids.py").read_text().split())
    assert "FAILURES ARE DRAWN, NOT DROPPED" in source
    assert "most raw cells are expected to be placeholders" in source
    assert "THE SAMPLE SELECTION IS FIXED AND QUALITY-BLIND" in source


# ================== corrections: selection rate vs target-hit yield ==============
def test_the_gallery_reports_selection_rate_and_target_hit_yield_separately(tmp_path):
    """Two different quantities. Conflating them hides the cost of selection."""
    pytest.importorskip("PIL")
    pytest.importorskip("rdkit")
    # 20 candidates, only 2 anywhere near 0.60; best-of-10 keeps 2 of 20
    qeds = [0.10] * 9 + [0.61] + [0.12] * 9 + [0.59]
    records = [record(i, qed=q) for i, q in enumerate(qeds)]
    manifest = best_of_k_gallery(records, n=15, requested_qed=0.60, k=10,
                                 decoder="degree", samples=2, tolerance=0.10,
                                 output=tmp_path / "sel.png")
    assert manifest["candidates_drawn"] == 20
    assert manifest["kept"] == 2
    assert manifest["selection_rate"] == pytest.approx(2 / 20)
    # both kept molecules are within tolerance, so target-hit yield is 2/20 as well,
    # but it is computed over ALL DRAWN and is a distinct number
    assert manifest["target_hit_yield_over_all_drawn"] == pytest.approx(2 / 20)
    assert manifest["tolerance"] == pytest.approx(0.10)


def test_target_hit_yield_is_lower_than_selection_rate_when_picks_miss(tmp_path):
    pytest.importorskip("PIL")
    pytest.importorskip("rdkit")
    records = [record(i, qed=0.20) for i in range(20)]      # nothing near 0.80
    manifest = best_of_k_gallery(records, n=15, requested_qed=0.80, k=10,
                                 decoder="degree", samples=2, tolerance=0.10,
                                 output=tmp_path / "sel.png")
    assert manifest["selection_rate"] == pytest.approx(2 / 20)
    assert manifest["target_hit_yield_over_all_drawn"] == pytest.approx(0.0)


def test_the_module_no_longer_claims_yield_is_unaffected():
    source = (Path('/projects/sandbox/zinc-conditional-gnf') / "zinc_gnf"
              / "molecule_grids.py").read_text()
    assert "yield is unaffected" not in source
    assert "SELECTION RATE" in source and "TARGET-HIT" in source
