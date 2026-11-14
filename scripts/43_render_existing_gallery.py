#!/usr/bin/env python3
"""Render molecule galleries from ALREADY SAVED samples.jsonl. No model, no generation.

Exists so molecules can be looked at now, without waiting for the checkpoint sweep.

ONE IMPORTANT LIMITATION, STATED UP FRONT
-----------------------------------------
The grids written by `scripts/13_generate_nqed.py` do NOT carry a noise identifier. The
row index is therefore the only handle available, and a row index is NOT a verified
shared-noise id: nothing in the saved file proves that line 7 of one run came from the
same prior draw as line 7 of another.

Consequently this script renders SINGLE-MODEL galleries only. It will not produce a
baseline-versus-candidate comparison, because that comparison is only meaningful on
verified common noise, and `scripts/42_checkpoint_eval.py` is the thing that writes
records with real noise ids. Asking for a comparison here is refused rather than
silently approximated.

What it does render, per (N, requested QED) cell:
  * a native gallery showing raw beside degree-corrected for the SAME saved sample, with
    invalid and undrawable entries drawn as labelled placeholders in place;
  * a SEPARATE best-of-k gallery carrying its selection rate and its target-hit yield
    over all drawn candidates.

Exit codes: 0 ok; 2 missing input; 4 bad arguments; 1 unexpected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_records(path: Path) -> list[dict[str, Any]]:
    """Attach the row index as `noise_id`, flagged as unverified by the caller."""
    records = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if line.strip():
                records.append({"noise_id": index, **json.loads(line)})
    if not records:
        raise FileNotFoundError(f"{path} is empty")
    return records


def score_in_place(records, *, decoders):
    """Recompute QED, aromatic rings and heavy atoms from the stored SMILES."""
    from rdkit import Chem, rdBase
    from rdkit.Chem import QED, rdMolDescriptors

    for record in records:
        for decoder in decoders:
            entry = record.get(decoder)
            if not isinstance(entry, dict):
                continue
            if not (entry.get("valid") and entry.get("smiles")):
                entry.setdefault("qed", None)
                continue
            with rdBase.BlockLogs():
                molecule = Chem.MolFromSmiles(str(entry["smiles"]))
            if molecule is None:
                entry["valid"] = False
                entry["qed"] = None
                continue
            entry["qed"] = float(QED.qed(molecule))
            entry["aromatic_rings"] = int(
                rdMolDescriptors.CalcNumAromaticRings(molecule))
            entry["heavy_atoms"] = int(molecule.GetNumHeavyAtoms())
    return records


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", default="final")
    parser.add_argument("--target-n", default="15,23,30")
    parser.add_argument("--qed-values", default="0.4,0.6,0.8")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--decoders", default="raw,degree")
    parser.add_argument("--grid-samples", type=int, default=6)
    parser.add_argument("--best-of-k", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=0.10)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        targets_n = [int(v) for v in args.target_n.split(",") if v.strip()]
        qed_values = [float(v) for v in args.qed_values.split(",") if v.strip()]
        decoders = tuple(d.strip() for d in args.decoders.split(",") if d.strip())
        if not targets_n or not qed_values or not decoders:
            raise ValueError("need at least one N, one QED value and one decoder")
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 4
    try:
        from zinc_gnf.molecule_grids import (
            Cell, best_of_k_gallery, cell_from_record, render_grid, select_noise_ids)

        root = Path(args.samples_root)
        figures: dict[str, Any] = {"native": {}, "selection": {}}
        for n in targets_n:
            for qed in qed_values:
                cell_dir = root / f"T{args.temperature:g}_N{n}_QED{qed:g}"
                path = cell_dir / "samples.jsonl"
                if not path.is_file():
                    figures["native"][f"N{n}_QED{qed:g}"] = {
                        "error": f"{path} not found"}
                    continue
                records = score_in_place(load_records(path), decoders=decoders)
                key = f"N{n}_QED{qed:g}"
                chosen = select_noise_ids([r["noise_id"] for r in records],
                                           args.grid_samples)
                index = {r["noise_id"]: r for r in records}
                rows = [(row_id,
                         [cell_from_record(index[row_id], model=args.label,
                                           decoder=decoder, requested_qed=qed)
                          for decoder in decoders])
                        for row_id in chosen]
                subtitle = (
                    f"N={n}  requested QED={qed:g}  |  saved records, {len(records)} "
                    f"available, {len(chosen)} shown  |  selection: even spacing over "
                    f"row index, independent of QED and validity  |  ROW INDEX IS NOT A "
                    f"VERIFIED SHARED-NOISE ID, so this is a single-model gallery and "
                    f"not a cross-model comparison")
                try:
                    figures["native"][key] = render_grid(
                        rows, title=f"Native generation \u2014 {args.label}",
                        subtitle=subtitle,
                        output=out_dir / "figures" / f"existing_native_{key}.png")
                except Exception as error:  # noqa: BLE001
                    figures["native"][key] = {"error": str(error)[:200]}
                try:
                    figures["selection"][key] = best_of_k_gallery(
                        records, n=n, requested_qed=qed, k=args.best_of_k,
                        decoder=decoders[-1], samples=args.grid_samples,
                        model_label=args.label, tolerance=args.tolerance,
                        output=out_dir / "figures"
                        / f"existing_bestof{args.best_of_k}_{key}.png")
                except Exception as error:  # noqa: BLE001
                    figures["selection"][key] = {"error": str(error)[:200]}
                head = figures["native"][key]
                if not head.get("error"):
                    print(f"  {key}: placeholder_rate="
                          f"{head['placeholder_rate']:.3f} "
                          f"({head['invalid']} invalid, {head['undrawable']} "
                          f"undrawable of {head['cells']})", flush=True)

        payload = {
            "framing": ("Galleries rendered from ALREADY SAVED records. No model, no "
                        "generation."),
            "limitation": ("The saved records carry no noise identifier, so the row "
                           "index is the only handle and is NOT a verified shared-noise "
                           "id. Single-model galleries only; cross-model comparison is "
                           "refused here and belongs to scripts/42, which writes real "
                           "noise ids."),
            "settings": vars(args), "figures": figures,
        }
        (out_dir / "existing_gallery.json").write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out_dir / 'existing_gallery.json'}")
        print(f"figures under {out_dir / 'figures'}")
        return 0
    except FileNotFoundError as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2
    except Exception:  # noqa: BLE001
        print("FATAL: unexpected\n" + traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
