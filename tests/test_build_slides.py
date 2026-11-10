"""Tests for the slide builder.

The deck is a deliverable that goes to the supervisor, so the things worth pinning are
the ones that would quietly corrupt it: a malformed table emitted as a broken file, a
missing number rendered as a guess instead of a gap, and a highlight index pointing at a
cell that does not exist.

The validator and the pending-count logic are pure and run anywhere. The rendering tests
need python-pptx and skip without it.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "38_build_slides.py"
CONTENT_PATH = ROOT / "docs" / "slides_content.json"


def builder():
    spec = importlib.util.spec_from_file_location("build_slides", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def deck(**overrides):
    slide = {"title": "A slide",
             "table": {"headers": ["a", "b"], "rows": [["1", "2"]]}}
    slide.update(overrides)
    return {"slides": [slide]}


# ------------------------------------------------------------------- validator ---
def test_a_deck_without_slides_is_rejected():
    module = builder()
    for bad in ({}, {"slides": []}, {"slides": "nope"}):
        with pytest.raises(ValueError, match="non-empty 'slides'"):
            module.validate(bad)


def test_a_slide_without_a_title_is_rejected():
    module = builder()
    with pytest.raises(ValueError, match="needs a title"):
        module.validate({"slides": [{"bullets": []}]})


def test_a_ragged_table_is_rejected_with_the_offending_row():
    module = builder()
    content = deck(table={"headers": ["a", "b", "c"], "rows": [["1", "2", "3"],
                                                                ["1", "2"]]})
    with pytest.raises(ValueError, match="row 2 has 2 cells but there are 3"):
        module.validate(content)


def test_a_table_without_headers_or_rows_is_rejected():
    module = builder()
    with pytest.raises(ValueError, match="needs 'headers'"):
        module.validate(deck(table={"rows": [["1"]]}))
    with pytest.raises(ValueError, match="needs 'headers'"):
        module.validate(deck(table={"headers": ["a"], "rows": "nope"}))


def test_an_out_of_range_highlight_is_rejected():
    module = builder()
    for highlight in ([[0, 5]], [[9, 0]], [[0]], ["x"]):
        with pytest.raises(ValueError, match="out of range"):
            module.validate(deck(table={"headers": ["a", "b"], "rows": [["1", "2"]],
                                         "highlight": highlight}))


def test_a_valid_highlight_passes():
    module = builder()
    module.validate(deck(table={"headers": ["a", "b"], "rows": [["1", "2"]],
                                 "highlight": [[0, 1]]}))


def test_a_slide_with_only_bullets_is_valid():
    module = builder()
    module.validate({"slides": [{"title": "t", "bullets": [{"text": "x"}]}]})


# ---------------------------------------------------------------- pending logic ---
def test_pending_cells_counts_only_nulls():
    module = builder()
    table = {"headers": ["a", "b"], "rows": [["1", None], [None, None], ["3", "4"]]}
    assert module.pending_cells(table) == 3


def test_pending_cells_is_zero_for_a_complete_table():
    module = builder()
    assert module.pending_cells({"headers": ["a"], "rows": [["1"], ["2"]]}) == 0
    assert module.pending_cells({}) == 0


def test_an_empty_string_is_not_treated_as_pending():
    """An intentionally blank cell must not be reported as a missing number."""
    module = builder()
    assert module.pending_cells({"headers": ["a"], "rows": [[""]]}) == 0


# ------------------------------------------------------------- shipped content ---
def test_the_shipped_content_is_valid():
    module = builder()
    module.validate(json.loads(CONTENT_PATH.read_text()))


def test_every_slide_with_a_gap_names_the_artefact_it_comes_from():
    """A gap with no source is useless to whoever has to fill it."""
    module = builder()
    content = json.loads(CONTENT_PATH.read_text())
    for slide in content["slides"]:
        table = slide.get("table")
        if table and module.pending_cells(table):
            assert slide.get("pending_from"), slide["title"]


def test_the_content_never_invents_a_number_for_a_missing_value():
    """Gaps must be null, never a placeholder string that reads as data."""
    content = json.loads(CONTENT_PATH.read_text())
    for slide in content["slides"]:
        for row in (slide.get("table") or {}).get("rows", []):
            for cell in row:
                if cell is None:
                    continue
                assert "FILL" not in str(cell), slide["title"]
                assert "TODO" not in str(cell), slide["title"]
                assert "???" not in str(cell), slide["title"]


def test_the_deck_keeps_both_raw_and_corrected_validity():
    """The raw-corrected gap is a finding; the deck must not quietly drop it."""
    content = json.loads(CONTENT_PATH.read_text())
    headers = [h for slide in content["slides"]
               for h in ((slide.get("table") or {}).get("headers") or [])]
    assert any("Raw" in h for h in headers)
    assert any("Corrected" in h for h in headers)


def test_the_deck_states_the_sample_size_caveat_rather_than_burying_it():
    content = json.loads(CONTENT_PATH.read_text())
    titles = [slide["title"] for slide in content["slides"]]
    assert any("Scale dependence" in title for title in titles)
    text = json.dumps(content)
    assert "+/-0.1" in text or "+/- 0.1" in text
    assert "would not survive the first question about n" in text


def test_the_deck_carries_the_published_baselines():
    """The supervisor's actual question was what to compare against."""
    text = json.dumps(json.loads(CONTENT_PATH.read_text()))
    for number in ("42.6%", "50.3%", "68.0%", "89.0%"):
        assert number in text, number
    for number in ("0.493", "0.796", "0.011", "0.22"):
        assert number in text, number


def test_the_deck_lists_the_files_still_needed():
    content = json.loads(CONTENT_PATH.read_text())
    titles = [slide["title"] for slide in content["slides"]]
    assert any("Data still needed" in title for title in titles)


# ------------------------------------------------------------------- rendering ---
def test_the_deck_renders_to_a_readable_pptx(tmp_path):
    pytest.importorskip("pptx")
    module = builder()
    output = tmp_path / "deck.pptx"
    stats = module.build(json.loads(CONTENT_PATH.read_text()), output)
    assert output.is_file() and output.stat().st_size > 10_000
    assert stats["slides"] >= 20
    assert stats["tables"] >= 10

    from pptx import Presentation
    presentation = Presentation(str(output))
    assert len(presentation.slides) == stats["slides"]
    assert round(presentation.slide_width.inches, 2) == 13.33      # 16:9


def test_no_slide_overflows_the_canvas(tmp_path):
    """A table running off the bottom is the usual way a generated deck breaks."""
    pytest.importorskip("pptx")
    module = builder()
    output = tmp_path / "deck.pptx"
    module.build(json.loads(CONTENT_PATH.read_text()), output)

    from pptx import Presentation
    presentation = Presentation(str(output))
    height = presentation.slide_height
    for index, slide in enumerate(presentation.slides, start=1):
        bottom = max((shape.top + shape.height for shape in slide.shapes), default=0)
        assert bottom <= height, f"slide {index} overflows by {bottom - height} EMU"


def test_a_missing_value_renders_as_a_dash_not_as_text(tmp_path):
    pytest.importorskip("pptx")
    module = builder()
    output = tmp_path / "deck.pptx"
    module.build({"slides": [{"title": "t", "pending_from": "somefile.json",
                               "table": {"headers": ["a", "b"],
                                         "rows": [["1", None]]}}]}, output)

    from pptx import Presentation
    table = next(shape.table for shape in Presentation(str(output)).slides[0].shapes
                 if shape.has_table)
    assert table.cell(1, 1).text == "\u2014"
    texts = " ".join(shape.text_frame.text
                     for shape in Presentation(str(output)).slides[0].shapes
                     if shape.has_text_frame)
    assert "Pending (1 values): somefile.json" in texts


def test_an_empty_header_does_not_crash_the_renderer(tmp_path):
    """The early-vs-final table has a blank corner cell; an empty string creates no run."""
    pytest.importorskip("pptx")
    module = builder()
    output = tmp_path / "deck.pptx"
    module.build({"slides": [{"title": "t",
                               "table": {"headers": ["", "a"],
                                         "rows": [["x", "y"]]}}]}, output)
    assert output.is_file()


# ----------------------------------------------------------------- CLI contract ---
def test_main_returns_an_exit_code_rather_than_raising(tmp_path):
    module = builder()
    assert module.main(["--content", str(tmp_path / "absent.json"),
                        "--output", str(tmp_path / "x.pptx")]) == 2


def test_malformed_content_is_exit_four(tmp_path):
    module = builder()
    path = tmp_path / "content.json"
    path.write_text(json.dumps({"slides": [{"bullets": []}]}))
    assert module.main(["--content", str(path),
                        "--output", str(tmp_path / "x.pptx")]) == 4


def test_unparseable_json_is_exit_two(tmp_path):
    module = builder()
    path = tmp_path / "content.json"
    path.write_text("{not json")
    assert module.main(["--content", str(path),
                        "--output", str(tmp_path / "x.pptx")]) == 2


def test_the_builder_touches_no_model_or_dataset():
    """Checked on the AST, not on the text: the docstring mentions these by name."""
    import ast

    tree = ast.parse(SCRIPT_PATH.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not imported & {"torch", "rdkit", "numpy", "zinc_gnf"}, imported

    calls = [ast.unparse(node.func) for node in ast.walk(tree)
             if isinstance(node, ast.Call)]
    for forbidden in ("load_flow", "load_autoencoder", "decode_embeddings"):
        assert not any(forbidden in call for call in calls), forbidden
