"""RDKit molecule grids for checkpoint comparison, rendered from SAVED sample records.

No model, no generation: this module takes sample-level records that an evaluation
already wrote and draws them. Images can therefore be re-rendered, relabelled or
re-laid-out without spending a single GPU second.

THREE RULES THAT MAKE THE IMAGES EVIDENCE RATHER THAN DECORATION
----------------------------------------------------------------
1. THE SAMPLE SELECTION IS FIXED AND QUALITY-BLIND. Rows are chosen by even spacing over
   the SORTED NOISE IDs (`select_noise_ids`). The choice depends only on the noise
   identifier, never on QED, validity or aesthetics, so it cannot drift toward the
   best-looking molecules. The same seed and budget always give the same rows, and the
   selection rule is printed on the figure.

2. FAILURES ARE DRAWN, NOT DROPPED. A sample that did not decode, or decoded to something
   RDKit cannot render, becomes a bordered placeholder reading INVALID or UNDRAWABLE in
   that exact grid position. Silently substituting the next valid molecule would turn a
   validity problem into an invisible one, and at N=30 raw validity is 0-1%, so most raw
   cells are expected to be placeholders. That is the finding, not a rendering bug.

3. PAIRING IS BY NOISE ID. Every column in a row is the SAME prior draw, so baseline and
   candidate are compared on identical noise and raw sits beside its own corrected
   structure. The noise ID is printed on each row.

SELECTION GALLERIES ARE KEPT APART
----------------------------------
`best_of_k_gallery` renders a separate figure with its own banner naming the candidate
budget and the number kept. Best-of-k molecules are chosen by oracle RDKit QED, so they
are not comparable with native samples and must never share a figure with them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

CELL = (320, 300)              # molecule box, then a label strip underneath
LABEL_HEIGHT = 58
HEADER_HEIGHT = 86
ROW_LABEL_WIDTH = 104
PLACEHOLDER_INVALID = "INVALID"
PLACEHOLDER_UNDRAWABLE = "UNDRAWABLE"


@dataclass
class Cell:
    """One grid position: either a molecule or an explicit failure."""

    smiles: str | None
    model: str
    decoder: str
    requested_qed: float
    actual_qed: float | None
    heavy_atoms: int | None
    aromatic_rings: int | None
    status: str = "ok"            # ok | invalid | undrawable
    stored_mismatch: dict[str, Any] | None = None

    def legend(self) -> list[str]:
        """Two short lines. Every quantity the reviewer needs is on the image."""
        head = f"{self.model} / {self.decoder}"
        if self.status != "ok":
            return [head, f"{self.status.upper()}  req QED {self.requested_qed:.2f}"]
        actual = "n/a" if self.actual_qed is None else f"{self.actual_qed:.3f}"
        atoms = "n/a" if self.heavy_atoms is None else str(self.heavy_atoms)
        rings = "n/a" if self.aromatic_rings is None else str(self.aromatic_rings)
        return [head,
                f"QED {self.requested_qed:.2f}\u2192{actual}  atoms {atoms}  arom {rings}"]


def select_noise_ids(noise_ids: Sequence[Any], count: int) -> list[Any]:
    """Evenly spaced over the SORTED noise ids. Deterministic and quality-blind.

    Even spacing rather than the first `count` so the sample is spread across the draw
    rather than clustered at one end, while still depending on nothing but the id.
    """
    ordered = sorted(set(noise_ids), key=lambda value: (isinstance(value, str), value))
    if count <= 0:
        return []
    if len(ordered) <= count:
        return ordered
    step = len(ordered) / count
    return [ordered[min(len(ordered) - 1, int(index * step))] for index in range(count)]


def cell_from_record(record: dict[str, Any], *, model: str, decoder: str,
                     requested_qed: float, recompute: bool = True) -> Cell:
    """Build a Cell from one decoder's entry, marking failure explicitly.

    TWO FIXES HERE, KEPT APART BECAUSE THEY ARE DIFFERENT BUGS.

    (a) NO SILENT FALLBACK. A missing decoder key used to fall back to treating the whole
        record as the entry, so a mistyped decoder name rendered plausible garbage. It
        now raises.

    (b) DESCRIPTORS ARE RECOMPUTED FROM THE MOLECULE THAT IS ACTUALLY DRAWN. The stored
        fields may have been written by a different code path, a different decoder, or an
        earlier definition, so a label taken from the record is not guaranteed to
        describe the depicted structure. With `recompute=True` the QED, heavy-atom count
        and aromatic-ring count come from parsing the same SMILES the renderer draws.
        This is NOT advanced as an explanation of the project's aromaticity gap -- that
        gap is a property of the generated molecules, not of the labelling, and
        `stored_mismatch` is reported so any disagreement is visible rather than assumed
        away.
    """
    if decoder not in record:
        raise KeyError(
            f"record has no {decoder!r} entry; available: "
            f"{sorted(k for k, v in record.items() if isinstance(v, dict))}")
    entry = record.get(decoder) or {}
    smiles = entry.get("smiles")
    valid = bool(entry.get("valid")) and bool(smiles)
    qed = entry.get("qed")
    atoms = entry.get("heavy_atoms")
    rings = entry.get("aromatic_rings")
    mismatch = None
    if valid and recompute:
        fresh = descriptors_from_smiles(str(smiles))
        if fresh is None:
            return Cell(smiles=str(smiles), model=model, decoder=decoder,
                        requested_qed=float(requested_qed), actual_qed=None,
                        heavy_atoms=None, aromatic_rings=None, status="undrawable")
        if qed is not None and abs(float(qed) - fresh["qed"]) > 1e-4:
            mismatch = {"stored_qed": float(qed), "recomputed_qed": fresh["qed"]}
        qed, atoms, rings = fresh["qed"], fresh["heavy_atoms"], fresh["aromatic_rings"]
    return Cell(
        smiles=smiles if valid else None,
        model=model, decoder=decoder, requested_qed=float(requested_qed),
        actual_qed=qed, heavy_atoms=atoms, aromatic_rings=rings,
        status="ok" if valid else "invalid", stored_mismatch=mismatch,
    )


def descriptors_from_smiles(smiles: str) -> dict[str, Any] | None:
    """QED, heavy atoms and aromatic rings of the molecule the renderer will draw."""
    from rdkit import Chem, rdBase
    from rdkit.Chem import QED, rdMolDescriptors

    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return {"qed": float(QED.qed(molecule)),
            "heavy_atoms": int(molecule.GetNumHeavyAtoms()),
            "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(molecule))}


# ------------------------------------------------------------------- rendering ---
def _font(size: int):
    from PIL import ImageFont
    for name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:  # noqa: BLE001 - fall through to the bundled default
            continue
    return ImageFont.load_default()


def _molecule_image(smiles: str, size):
    """PIL image for one SMILES, or None if RDKit cannot draw it.

    `Draw.MolToImage` is used rather than the Cairo backend because it returns a PIL
    image directly and does not require RDKit to have been built with Cairo.
    """
    from rdkit import Chem, rdBase
    from rdkit.Chem import Draw

    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    try:
        with rdBase.BlockLogs():
            Chem.rdDepictor.Compute2DCoords(molecule)
            return Draw.MolToImage(molecule, size=size)
    except Exception:  # noqa: BLE001 - an undrawable molecule is marked, not dropped
        return None


def _placeholder(size, text: str):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", size, (0xF7, 0xF7, 0xF9))
    draw = ImageDraw.Draw(image)
    draw.rectangle([(1, 1), (size[0] - 2, size[1] - 2)], outline=(0xC2, 0x3B, 0x22),
                   width=2)
    font = _font(20)
    anchor = (size[0] // 2, size[1] // 2)
    draw.text(anchor, text, fill=(0xC2, 0x3B, 0x22), font=font, anchor="mm")
    return image


def render_grid(rows: Sequence[tuple[Any, Sequence[Cell]]], *, title: str,
                subtitle: str, output: str | Path) -> dict[str, Any]:
    """Draw a grid: one row per noise id, one column per (model, decoder).

    Returns a manifest counting what was drawn and what was marked, so the caller can
    report the placeholder rate instead of the reader having to count boxes.
    """
    from PIL import Image, ImageDraw

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no rows to render")
    columns = len(rows[0][1])
    if any(len(cells) != columns for _, cells in rows):
        raise ValueError("every row must have the same number of cells")

    width = ROW_LABEL_WIDTH + columns * CELL[0]
    height = HEADER_HEIGHT + len(rows) * (CELL[1] + LABEL_HEIGHT)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 14), title, fill=(0x1F, 0x24, 0x33), font=_font(26))
    draw.text((16, 50), subtitle, fill=(0x5B, 0x65, 0x70), font=_font(15))

    counts = {"drawn": 0, "invalid": 0, "undrawable": 0}
    label_font, row_font = _font(14), _font(15)
    for row_index, (noise_id, cells) in enumerate(rows):
        top = HEADER_HEIGHT + row_index * (CELL[1] + LABEL_HEIGHT)
        draw.text((12, top + CELL[1] // 2), f"noise\n#{noise_id}",
                  fill=(0x1F, 0x24, 0x33), font=row_font)
        for column_index, cell in enumerate(cells):
            left = ROW_LABEL_WIDTH + column_index * CELL[0]
            image = _molecule_image(cell.smiles, CELL) if cell.smiles else None
            if cell.smiles and image is None:
                cell.status = "undrawable"
            if image is None:
                text = (PLACEHOLDER_UNDRAWABLE if cell.status == "undrawable"
                        else PLACEHOLDER_INVALID)
                image = _placeholder(CELL, text)
                counts[cell.status if cell.status in counts else "invalid"] += 1
            else:
                counts["drawn"] += 1
            canvas.paste(image.convert("RGB"), (left, top))
            for line_index, line in enumerate(cell.legend()):
                draw.text((left + 8, top + CELL[1] + 6 + line_index * 18), line,
                          fill=(0x1F, 0x24, 0x33), font=label_font)
            draw.rectangle([(left, top), (left + CELL[0] - 1,
                                          top + CELL[1] + LABEL_HEIGHT - 1)],
                           outline=(0xDD, 0xE1, 0xE6))
    canvas.save(output)
    total = sum(counts.values())
    return {"path": str(output), "rows": len(rows), "columns": columns,
            "cells": total, **counts,
            "placeholder_rate": (counts["invalid"] + counts["undrawable"]) / max(1, total)}


def comparison_grid(baseline_records, candidate_records, *, n, requested_qed,
                    decoders=("raw", "degree"), samples=6,
                    baseline_label="baseline", candidate_label="candidate",
                    output) -> dict[str, Any]:
    """Baseline vs candidate at one (N, requested QED), paired by noise id.

    Columns are (baseline x decoders) then (candidate x decoders), so raw sits beside
    its own corrected structure and both models sit on the same prior draw.
    """
    by_id = {
        baseline_label: {record["noise_id"]: record for record in baseline_records},
        candidate_label: {record["noise_id"]: record for record in candidate_records},
    }
    shared = sorted(set(by_id[baseline_label]) & set(by_id[candidate_label]),
                    key=lambda value: (isinstance(value, str), value))
    if not shared:
        raise ValueError("baseline and candidate share no noise ids; they were not "
                         "generated on common noise and must not be compared")
    chosen = select_noise_ids(shared, samples)
    rows = []
    for noise_id in chosen:
        cells = [cell_from_record(by_id[model][noise_id], model=model, decoder=decoder,
                                  requested_qed=requested_qed)
                 for model in (baseline_label, candidate_label)
                 for decoder in decoders]
        rows.append((noise_id, cells))
    subtitle = (f"N={n}  requested QED={requested_qed:g}  |  rows are the SAME prior "
                f"draws in both models  |  selection: even spacing over sorted noise "
                f"ids, independent of QED and validity  |  "
                f"{len(shared)} shared draws available, {len(chosen)} shown")
    return render_grid(rows, title=f"Native generation \u2014 {baseline_label} vs "
                                    f"{candidate_label}",
                       subtitle=subtitle, output=output)


def best_of_k_gallery(records, *, n, requested_qed, k, decoder="degree", samples=6,
                      model_label="candidate", tolerance=0.10, output) -> dict[str, Any]:
    """A SEPARATE figure for selected molecules, with the budget on the banner.

    Kept apart from native grids on purpose: these molecules were chosen by oracle RDKit
    QED, so putting them next to native samples would invite exactly the comparison that
    is invalid.
    """
    scorable = [r for r in records
                if ((r.get(decoder) or {}).get("qed")) is not None]
    if not scorable:
        raise ValueError("no scorable records to build a selection gallery from")
    groups = [scorable[index:index + k] for index in range(0, len(scorable), k)]
    groups = [group for group in groups if len(group) == k]
    if not groups:
        raise ValueError(f"fewer than one full group of {k} scorable records")
    kept = [min(group, key=lambda r: abs(r[decoder]["qed"] - requested_qed))
            for group in groups]
    drawn = len(groups) * k
    # SELECTION RATE is how many of the drawn candidates survive the filter. TARGET-HIT
    # YIELD is a different quantity: how many of ALL drawn candidates are both valid and
    # within tolerance of the request. Conflating them would hide the cost of selection.
    on_target = sum(1 for r in kept
                    if abs(r[decoder]["qed"] - requested_qed) <= tolerance)
    chosen = select_noise_ids([r["noise_id"] for r in kept], samples)
    index = {r["noise_id"]: r for r in kept}
    rows = [(noise_id,
             [cell_from_record(index[noise_id], model=f"{model_label} best-of-{k}",
                               decoder=decoder, requested_qed=requested_qed)])
            for noise_id in chosen]
    subtitle = (f"SELECTION, NOT NATIVE GENERATION.  N={n}  requested QED="
                f"{requested_qed:g}  |  {drawn} candidates drawn, {len(groups)} kept "
                f"(best-of-{k} by oracle RDKit QED), {len(chosen)} shown  |  "
                f"selection rate {len(groups)}/{drawn}  |  target-hit yield "
                f"{on_target}/{drawn} (valid and within {tolerance:g} of request, over "
                f"ALL drawn)")
    manifest = render_grid(rows, title=f"Best-of-{k} selected molecules \u2014 "
                                        f"{model_label}",
                           subtitle=subtitle, output=output)
    manifest.update({"candidates_drawn": drawn, "kept": len(groups),
                     "selection_rate": len(groups) / drawn,
                     "target_hit_yield_over_all_drawn": on_target / drawn,
                     "tolerance": tolerance})
    return manifest


__all__ = [
    "Cell",
    "best_of_k_gallery",
    "cell_from_record",
    "comparison_grid",
    "descriptors_from_smiles",
    "render_grid",
    "select_noise_ids",
]
