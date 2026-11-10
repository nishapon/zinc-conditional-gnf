#!/usr/bin/env python3
"""Render the results deck to .pptx from a JSON content file.

Content and rendering are separate on purpose: all text and numbers live in
`docs/slides_content.json`, so filling in a missing number means editing JSON and
re-running this script -- no code change, and no hand-editing of a binary that then
drifts from the repository.

READ-ONLY with respect to the research: no model, no dataset, no torch, no rdkit.

MISSING NUMBERS ARE RENDERED AS A DASH, NOT AS A MARKER
-------------------------------------------------------
A table cell whose value is `null` renders as an em dash and the slide grows a small
"Pending" line naming the artefact it comes from. That reads as a deliberate template
rather than as a broken placeholder, which is what `[[FILL: ...]]` looked like on screen.
A deck with honest gaps is still presentable; a deck with fake numbers is not, so the
gaps stay visible and labelled.

    python scripts/38_build_slides.py \
        --content docs/slides_content.json \
        --output docs/final_results_slides.pptx

Exit codes: 0 ok; 2 missing input; 4 bad content; 1 unexpected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

TITLE_SIZE = 26
KICKER_SIZE = 13
BULLET_SIZE = 15
SUB_BULLET_SIZE = 13
TABLE_HEADER_SIZE = 12
TABLE_BODY_SIZE = 11
FOOT_SIZE = 10
DASH = "\u2014"

INK = (0x1F, 0x24, 0x33)
MUTED = (0x5B, 0x65, 0x70)
ACCENT = (0x0B, 0x5F, 0xA5)
GOOD = (0x1B, 0x7F, 0x3B)
WARN = (0xA6, 0x3D, 0x00)
HEADER_FILL = (0xEF, 0xF2, 0xF6)
HIGHLIGHT_FILL = (0xFF, 0xF4, 0xCC)


def validate(content: dict[str, Any]) -> None:
    """Fail on a malformed deck rather than emitting a silently broken file."""
    if not isinstance(content.get("slides"), list) or not content["slides"]:
        raise ValueError("content must contain a non-empty 'slides' list")
    for index, slide in enumerate(content["slides"]):
        where = f"slide {index + 1}"
        if not slide.get("title"):
            raise ValueError(f"{where}: every slide needs a title")
        table = slide.get("table")
        if table is None:
            continue
        headers = table.get("headers")
        rows = table.get("rows")
        if not headers or not isinstance(rows, list):
            raise ValueError(f"{where}: a table needs 'headers' and a 'rows' list")
        for row_index, row in enumerate(rows):
            if len(row) != len(headers):
                raise ValueError(
                    f"{where}: row {row_index + 1} has {len(row)} cells but there are "
                    f"{len(headers)} headers")
        for pair in table.get("highlight", []):
            if (not isinstance(pair, list) or len(pair) != 2
                    or not 0 <= pair[0] < len(rows)
                    or not 0 <= pair[1] < len(headers)):
                raise ValueError(f"{where}: highlight {pair} is out of range")


def pending_cells(table: dict[str, Any]) -> int:
    return sum(1 for row in table.get("rows", []) for cell in row if cell is None)


def build(content: dict[str, Any], output: Path) -> dict[str, Any]:
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Emu, Inches, Pt

    presentation = Presentation()
    presentation.slide_width = Inches(13.333)      # 16:9
    presentation.slide_height = Inches(7.5)
    blank = presentation.slide_layouts[6]

    def colour(rgb):
        return RGBColor(*rgb)

    def textbox(slide, left, top, width, height):
        box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width),
                                       Inches(height))
        frame = box.text_frame
        frame.word_wrap = True
        return frame

    stats = {"slides": 0, "tables": 0, "pending_cells": 0, "pending_slides": []}

    for spec in content["slides"]:
        slide = presentation.slides.add_slide(blank)
        stats["slides"] += 1
        cursor = 0.45

        frame = textbox(slide, 0.55, cursor, 12.2, 0.95)
        paragraph = frame.paragraphs[0]
        run = paragraph.add_run()
        run.text = spec["title"]
        run.font.size = Pt(TITLE_SIZE)
        run.font.bold = True
        run.font.color.rgb = colour(INK)
        cursor += 0.95 if len(spec["title"]) < 70 else 1.25

        if spec.get("kicker"):
            frame = textbox(slide, 0.55, cursor, 12.2, 0.4)
            run = frame.paragraphs[0].add_run()
            run.text = spec["kicker"]
            run.font.size = Pt(KICKER_SIZE)
            run.font.italic = True
            run.font.color.rgb = colour(MUTED)
            cursor += 0.42

        bullets = spec.get("bullets") or []
        if bullets:
            height = 0.30 * len(bullets) + 0.25
            frame = textbox(slide, 0.55, cursor, 12.2, height)
            for position, bullet in enumerate(bullets):
                paragraph = (frame.paragraphs[0] if position == 0
                             else frame.add_paragraph())
                level = int(bullet.get("level", 0))
                paragraph.level = level
                run = paragraph.add_run()
                prefix = "" if bullet.get("plain") else ("\u2022  " if level == 0
                                                          else "\u2013  ")
                run.text = prefix + bullet["text"]
                run.font.size = Pt(BULLET_SIZE if level == 0 else SUB_BULLET_SIZE)
                run.font.bold = bool(bullet.get("bold"))
                tone = bullet.get("tone")
                run.font.color.rgb = colour(
                    GOOD if tone == "good" else WARN if tone == "warn"
                    else ACCENT if tone == "accent" else INK)
                paragraph.space_after = Pt(5)
            cursor += height + 0.12

        table_spec = spec.get("table")
        if table_spec:
            stats["tables"] += 1
            headers, rows = table_spec["headers"], table_spec["rows"]
            highlight = {tuple(pair) for pair in table_spec.get("highlight", [])}
            available = 6.95 - cursor
            row_height = min(0.34, max(0.22, available / (len(rows) + 1.6)))
            height = row_height * (len(rows) + 1)
            shape = slide.shapes.add_table(len(rows) + 1, len(headers),
                                           Inches(0.55), Inches(cursor),
                                           Inches(12.2), Inches(height))
            table = shape.table
            first_width = 3.4 if len(headers) <= 5 else 2.8
            rest = (12.2 - first_width) / max(1, len(headers) - 1)
            table.columns[0].width = Emu(int(Inches(first_width)))
            for index in range(1, len(headers)):
                table.columns[index].width = Emu(int(Inches(rest)))

            def cell_run(cell, text, *, align_left):
                """Add the run explicitly: an empty string creates no run to format."""
                paragraph = cell.text_frame.paragraphs[0]
                paragraph.alignment = PP_ALIGN.LEFT if align_left else PP_ALIGN.CENTER
                run = paragraph.add_run()
                run.text = str(text)
                return run

            for index, header in enumerate(headers):
                cell = table.cell(0, index)
                run = cell_run(cell, header or "", align_left=index == 0)
                run.font.size = Pt(TABLE_HEADER_SIZE)
                run.font.bold = True
                run.font.color.rgb = colour(INK)
                cell.fill.solid()
                cell.fill.fore_color.rgb = colour(HEADER_FILL)

            for row_index, row in enumerate(rows):
                for column_index, value in enumerate(row):
                    cell = table.cell(row_index + 1, column_index)
                    missing = value is None
                    if missing:
                        stats["pending_cells"] += 1
                    run = cell_run(cell, DASH if missing else value,
                                   align_left=column_index == 0)
                    run.font.size = Pt(TABLE_BODY_SIZE)
                    emphasised = (row_index, column_index) in highlight
                    run.font.bold = emphasised
                    run.font.color.rgb = colour(
                        MUTED if missing else ACCENT if emphasised else INK)
                    if emphasised and not missing:
                        cell.fill.solid()
                        cell.fill.fore_color.rgb = colour(HIGHLIGHT_FILL)
            cursor += height + 0.12

        footers = []
        if spec.get("note"):
            footers.append(("note", spec["note"]))
        if spec.get("source"):
            footers.append(("source", f"Source: {spec['source']}"))
        if table_spec and pending_cells(table_spec):
            where = spec.get("pending_from") or "the cluster artefacts"
            footers.append(("pending",
                            f"Pending ({pending_cells(table_spec)} values): {where}"))
            stats["pending_slides"].append(spec["title"])
        elif spec.get("pending_from"):
            footers.append(("pending", f"Pending: {spec['pending_from']}"))
            stats["pending_slides"].append(spec["title"])

        if footers:
            top = min(cursor + 0.05, 6.72)
            frame = textbox(slide, 0.55, top, 12.2, 0.72)
            for position, (kind, text) in enumerate(footers):
                paragraph = (frame.paragraphs[0] if position == 0
                             else frame.add_paragraph())
                run = paragraph.add_run()
                run.text = text
                run.font.size = Pt(FOOT_SIZE)
                run.font.italic = kind != "note"
                run.font.color.rgb = colour(WARN if kind == "pending" else MUTED)

    output.parent.mkdir(parents=True, exist_ok=True)
    presentation.save(str(output))
    return stats


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--content", default="docs/slides_content.json")
    parser.add_argument("--output", default="docs/final_results_slides.pptx")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    content_path = Path(args.content)
    try:
        if not content_path.is_file():
            print(f"FATAL: content not found: {content_path}", file=sys.stderr)
            return 2
        content = json.loads(content_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2
    try:
        validate(content)
    except ValueError as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 4
    try:
        stats = build(content, Path(args.output))
    except ImportError as error:
        print(f"FATAL: python-pptx is required: {error}", file=sys.stderr)
        return 2
    except Exception:  # noqa: BLE001
        print("FATAL: unexpected\n" + traceback.format_exc(), file=sys.stderr)
        return 1

    print(f"wrote {args.output}")
    print(f"  slides: {stats['slides']}  tables: {stats['tables']}")
    print(f"  pending values: {stats['pending_cells']}")
    for title in stats["pending_slides"]:
        print(f"    - {title}")
    if stats["pending_cells"]:
        print("  Those cells render as an em dash with a labelled Pending line. Fill "
              "them in docs/slides_content.json and re-run this script.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
