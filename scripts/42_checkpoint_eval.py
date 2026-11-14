#!/usr/bin/env python3
"""Checkpoint selection on DECODED QED and chemistry, with molecule grids.

Two modes. Run the inventory first; it is free and decides whether the experiment is
worth anything.

  --inventory-only   lists every checkpoint in a run directory with its recorded step
                     and validation metrics, and prints a measured-assumption cost
                     estimate for evaluating them. NOTHING is generated. If too few
                     snapshots exist, STOP: do not spend training to manufacture
                     candidates.

  (default)          evaluates the named checkpoints on all nine (N, requested QED)
                     cells under SHARED prior noise, writes sample-level records keyed
                     by noise id, computes the full metric set, and renders RDKit grids
                     from those records.

Every model is opened in eval() with `requires_grad_(False)`; nothing is trained and no
checkpoint is written. Images come from the saved records, so re-rendering costs nothing.

WHY DECODED-QED SELECTION IS WORTH TESTING
------------------------------------------
`zinc_gnf/trainer_flow.py:568` saves `best.pt` on validation per-node NLL alone, and the
project has already established that better validation likelihood did not improve decoded
QED responsiveness. So the deliverable model is chosen by a criterion known to be
misaligned with the objective. Selecting among existing checkpoints on the decoded metric
is ordinary model selection, not inference-time selection: the chosen model generates
natively, every sample is kept and yield is unaffected. It belongs in the native column,
reported apart from best-of-k.

To keep that honest the split matters: candidates are ranked on the VALIDATION conditions
and the winner is reported on the untouched TEST split (`zinc_gnf/dataset.py:314`), which
no flow script has ever read. The validation-to-test gap is reported as the headline, so
selection overfitting is visible rather than hidden.

Exit codes: 0 ok; 2 missing input; 3 provenance mismatch; 4 bad arguments; 1 unexpected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from zinc_gnf.conditioning import (  # noqa: E402
    condition_embedding_for,
    load_condition_scaler,
    standardise_condition,
)
from zinc_gnf.generation import decode_embeddings  # noqa: E402
from zinc_gnf.trainer_autoencoder import ResumeMismatchError, load_autoencoder  # noqa: E402
from zinc_gnf.trainer_flow import file_sha256, load_flow  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402

EVAL_NOISE_SEED = 20261118          # validation-condition ranking
TEST_NOISE_SEED = 20261119          # final report, distinct stream
DECODERS = ("raw", "constrained", "degree")
QED_TOLERANCE = 0.10                # for the valid-and-on-target yield
# Measured on this project's own jobs: the ACP gate decoded ~4,000 generated samples in
# 8m09s on 4 CPU cores. Used only for the estimate, and labelled as an extrapolation.
SAMPLES_PER_SECOND = 8.0


def inventory(directory: Path, *, cells: int, samples: int) -> dict[str, Any]:
    """What checkpoints exist, and what would evaluating them cost?"""
    found = []
    for path in sorted(directory.glob("*.pt")):
        entry: dict[str, Any] = {"name": path.name,
                                 "size_mb": round(path.stat().st_size / 1e6, 1)}
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            entry["step"] = blob.get("step")
            metrics = blob.get("val_metrics") or {}
            entry["val_loss"] = metrics.get("val_loss")
            entry["val_condition_gap"] = metrics.get("val_condition_gap")
            entry["has_optimizer_state"] = blob.get("optimizer_state_dict") is not None
        except Exception as error:  # noqa: BLE001
            entry["unreadable"] = str(error)[:120]
        found.append(entry)
    usable = [e for e in found if e.get("step") is not None]
    per_checkpoint = cells * samples
    seconds = len(usable) * per_checkpoint / SAMPLES_PER_SECOND
    distinct_steps = sorted({e["step"] for e in usable if e.get("step") is not None})
    return {
        "directory": str(directory),
        "checkpoints": found,
        "readable_count": len(usable),
        "distinct_steps": distinct_steps,
        "samples_per_checkpoint": per_checkpoint,
        "estimated_generation_seconds": seconds,
        "estimated_generation_minutes": round(seconds / 60, 1),
        "throughput_assumption": (
            f"{SAMPLES_PER_SECOND}/s decoded, extrapolated from this project's ACP gate "
            f"(~4,000 samples in 8m09s on 4 cores). An estimate, not a measurement of "
            f"this job."),
        "verdict": (
            "ENOUGH CANDIDATES: selection has something to choose between"
            if len(distinct_steps) >= 8 else
            f"TOO FEW CANDIDATES: {len(distinct_steps)} distinct steps. Selection over "
            f"this many is not a meaningful experiment. Do NOT run extra training to "
            f"manufacture candidates without a separate decision."),
    }


def pair_masks(node_mask: torch.Tensor) -> torch.Tensor:
    width = node_mask.shape[1]
    pair = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
    eye = torch.eye(width, dtype=torch.bool, device=node_mask.device).unsqueeze(0)
    return pair & ~eye


@torch.no_grad()
def generate_cell(flow, autoencoder, *, scaler, n, qed, eps, temperature, device,
                  batch_size):
    """One cell. Records carry `noise_id` so every comparison can be paired."""
    from rdkit import Chem, rdBase
    from rdkit.Chem import QED, rdMolDescriptors

    records = []
    total = eps.shape[0]
    for start in range(0, total, batch_size):
        chunk = eps[start:start + batch_size, :n, :]
        size = chunk.shape[0]
        mask = torch.ones((size, n), dtype=torch.bool, device=device)
        values = torch.from_numpy(
            standardise_condition(flow.config.condition_features, scaler=scaler,
                                  n=[n] * size, qed=[qed] * size)).to(device)
        condition = condition_embedding_for(flow, values, torch.ones_like(values),
                                            size, device)
        mu, sigma = flow.prior.parameters_for(size, condition, device)
        z = (mu.unsqueeze(1) + float(temperature) * sigma.unsqueeze(1) * chunk) \
            * mask.unsqueeze(-1)
        embeddings = flow.inverse(z, mask, condition)
        decoded = decode_embeddings(autoencoder, embeddings, [n] * size,
                                    decoders=DECODERS)
        for offset, record in enumerate(decoded):
            row: dict[str, Any] = {"noise_id": start + offset,
                                   "requested_n": n, "requested_qed": float(qed)}
            for decoder in DECODERS:
                entry = record.get(decoder) or {}
                scored: dict[str, Any] = {
                    "valid": bool(entry.get("valid")),
                    "fallback": bool(entry.get("fallback", False)),
                    "smiles": entry.get("smiles"),
                    "qed": None, "aromatic_rings": None, "heavy_atoms": None,
                    "rings": None,
                }
                if entry.get("valid") and entry.get("smiles"):
                    with rdBase.BlockLogs():
                        molecule = Chem.MolFromSmiles(str(entry["smiles"]))
                    if molecule is not None:
                        scored.update({
                            "qed": float(QED.qed(molecule)),
                            "aromatic_rings": int(
                                rdMolDescriptors.CalcNumAromaticRings(molecule)),
                            "heavy_atoms": int(molecule.GetNumHeavyAtoms()),
                            "rings": int(molecule.GetRingInfo().NumRings())})
                row[decoder] = scored
            records.append(row)
    return records


def mean(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def internal_diversity(smiles):
    unique = sorted({s for s in smiles if s})
    if len(unique) < 2:
        return None
    from zinc_gnf.metrics_generation import internal_diversity as compute
    return float(compute(unique)["internal_diversity"])


def cell_metrics(records, *, n, qed, decoder, tolerance=QED_TOLERANCE) -> dict[str, Any]:
    """Everything requested, per cell. Yield is over ALL attempted samples."""
    attempted = len(records)
    entries = [r[decoder] for r in records]
    scored = [e for e in entries if e["qed"] is not None]
    smiles = [e["smiles"] for e in scored if e["smiles"]]
    unique = sorted(set(smiles))
    on_target = [e for e in scored if abs(e["qed"] - qed) <= tolerance]
    return {
        "requested_n": n, "requested_qed": qed, "attempted": attempted,
        "raw_validity": mean([float(r["raw"]["valid"]) for r in records]),
        "constrained_validity": mean([float(r["constrained"]["valid"]) for r in records]),
        "degree_validity": mean([float(r["degree"]["valid"]) for r in records]),
        "fallback_rate": mean([float(e["fallback"]) for e in entries]),
        "scored": len(scored),
        "qed_mae": mean([abs(e["qed"] - qed) for e in scored]),
        "qed_mean": mean([e["qed"] for e in scored]),
        # the honest denominator: all attempts, so validity loss cannot be hidden
        "valid_on_target_yield": len(on_target) / max(1, attempted),
        "tolerance": tolerance,
        "heavy_atoms_mean": mean([e["heavy_atoms"] for e in scored]),
        "atom_count_adherence": mean(
            [float(e["heavy_atoms"] == n) for e in scored]),
        "atom_count_abs_error": mean([abs(e["heavy_atoms"] - n) for e in scored]),
        "aromatic_rings_mean": mean([e["aromatic_rings"] for e in scored]),
        "fraction_no_aromatic_ring": mean(
            [float(e["aromatic_rings"] == 0) for e in scored]),
        "unique_molecules": len(unique),
        "uniqueness": (len(unique) / len(scored)) if scored else None,
        "internal_diversity": internal_diversity(smiles),
    }


def paired_response(low_records, high_records, *, decoder) -> dict[str, Any]:
    """Per-noise-id QED response, jointly scorable pairs only."""
    low = {r["noise_id"]: r[decoder] for r in low_records}
    high = {r["noise_id"]: r[decoder] for r in high_records}
    deltas = [high[key]["qed"] - low[key]["qed"]
              for key in sorted(set(low) & set(high))
              if low[key]["qed"] is not None and high[key]["qed"] is not None]
    if len(deltas) < 2:
        return {"pairs": len(deltas), "mean": mean(deltas), "ci95": [None, None]}
    array = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(20261120)
    means = array[rng.integers(0, array.size, size=(2000, array.size))].mean(axis=1)
    low_q, high_q = float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
    return {"pairs": int(array.size), "mean": float(array.mean()),
            "ci95": [low_q, high_q],
            "excludes_zero": bool(low_q > 0.0 or high_q < 0.0),
            "pairing": "per noise id, jointly scorable only"}


def summarise(cells, responses, *, decoder) -> dict[str, Any]:
    """Equally weighted macro aggregates, so a high-yield cell cannot dominate."""
    values = list(cells.values())
    return {
        "decoder": decoder,
        "macro_qed_mae": mean([c["qed_mae"] for c in values]),
        "macro_valid_on_target_yield": mean([c["valid_on_target_yield"] for c in values]),
        "macro_raw_validity": mean([c["raw_validity"] for c in values]),
        "macro_degree_validity": mean([c["degree_validity"] for c in values]),
        "macro_aromatic_rings": mean([c["aromatic_rings_mean"] for c in values]),
        "macro_uniqueness": mean([c["uniqueness"] for c in values]),
        "macro_internal_diversity": mean([c["internal_diversity"] for c in values]),
        "macro_atom_count_adherence": mean([c["atom_count_adherence"] for c in values]),
        "mean_paired_response": mean([r["mean"] for r in responses.values()]),
        "cells": len(values),
        "weighting": ("macro = equally weighted over the nine cells. Cells contribute "
                      "unequal sample counts after validity filtering, so a "
                      "sample-pooled average would be dominated by whichever cells "
                      "decode best."),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--checkpoint", action="append", default=[],
                        help="label=path, repeatable; omit to use every readable .pt")
    parser.add_argument("--baseline-label", default="best")
    parser.add_argument("--ae-checkpoint", default=None)
    parser.add_argument("--data-dir",
                        default="data/processed/zinc250k_charge_aware_moflow_fixed")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-n", default="15,23,30")
    parser.add_argument("--qed-values", default="0.4,0.6,0.8")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--decoder", default="degree", choices=DECODERS)
    parser.add_argument("--grid-samples", type=int, default=6)
    parser.add_argument("--best-of-k", type=int, default=10)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--skip-grids", action="store_true")
    parser.add_argument("--allow-ae-mismatch", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        targets_n = [int(v) for v in args.target_n.split(",") if v.strip()]
        qed_values = [float(v) for v in args.qed_values.split(",") if v.strip()]
        if not targets_n or len(qed_values) < 2:
            raise ValueError("need at least one N and two QED values")
        if args.samples < 4 or args.batch_size < 1:
            raise ValueError("samples >= 4 and batch-size >= 1")
        checkpoints: dict[str, Path] = {}
        for item in args.checkpoint:
            if "=" not in item:
                raise ValueError(f"--checkpoint must be label=path, got {item!r}")
            label, path = item.split("=", 1)
            if not label.strip() or not path.strip():
                raise ValueError(f"--checkpoint must be label=path, got {item!r}")
            checkpoints[label.strip()] = Path(path.strip())
        directory = Path(args.checkpoint_dir)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 4

    started = time.perf_counter()
    try:
        if not directory.is_dir():
            raise FileNotFoundError(f"checkpoint directory not found: {directory}")
        report = inventory(directory, cells=len(targets_n) * len(qed_values),
                           samples=args.samples)
        print(format_inventory(report))
        atomic_write_json(report, out_dir / "checkpoint_inventory.json")
        (out_dir / "checkpoint_inventory.txt").write_text(format_inventory(report))
        if args.inventory_only:
            print(f"\nwrote {out_dir / 'checkpoint_inventory.txt'}")
            print("inventory only; nothing was generated")
            return 0
        if not checkpoints:
            raise ValueError("no --checkpoint label=path given; pass them explicitly "
                             "after reading the inventory")

        device = torch.device(args.device)
        scaler = load_condition_scaler(args.data_dir)
        seed = EVAL_NOISE_SEED if args.split == "validation" else TEST_NOISE_SEED
        payload: dict[str, Any] = {
            "framing": ("Checkpoint selection on DECODED QED and chemistry. Ordinary "
                        "model selection: the chosen model generates natively, every "
                        "sample is kept, yield is unaffected. Reported apart from "
                        "best-of-k, which is inference-time selection."),
            "settings": vars(args) | {"noise_seed": seed},
            "inventory": report,
            "models": {},
        }

        all_records: dict[str, Any] = {}
        autoencoder = None
        digest = None
        for label, path in checkpoints.items():
            if not path.is_file():
                raise FileNotFoundError(f"{label}: checkpoint not found: {path}")
            flow, probe = load_flow(path, device=device)
            flow.eval().requires_grad_(False)
            recorded = probe.get("autoencoder") or {}
            if autoencoder is None:
                ae_path = Path(args.ae_checkpoint or recorded.get("checkpoint", ""))
                if not ae_path.is_file():
                    raise FileNotFoundError(f"autoencoder not found: {ae_path}")
                digest = file_sha256(ae_path)
                autoencoder, _ = load_autoencoder(ae_path, device=device)
                autoencoder.eval().requires_grad_(False)
            if recorded.get("sha256") != digest and not args.allow_ae_mismatch:
                raise ResumeMismatchError(
                    f"{label}: autoencoder SHA256 mismatch against the shared encoder")

            block: dict[str, Any] = {"checkpoint": str(path), "step": probe.get("step"),
                                     "sha256": file_sha256(path), "cells": {},
                                     "paired_response": {}}
            records_by_cell: dict[str, Any] = {}
            for n in targets_n:
                generator = torch.Generator(device=device).manual_seed(seed + 7919 * n)
                eps = torch.randn((args.samples, max(targets_n),
                                   flow.config.node_embedding_dim),
                                  device=device, generator=generator)
                for qed in qed_values:
                    records = generate_cell(
                        flow, autoencoder, scaler=scaler, n=n, qed=qed, eps=eps,
                        temperature=args.temperature, device=device,
                        batch_size=args.batch_size)
                    key = f"N{n}_QED{qed:g}"
                    records_by_cell[key] = records
                    block["cells"][key] = cell_metrics(records, n=n, qed=qed,
                                                        decoder=args.decoder)
                    head = block["cells"][key]
                    print(f"  {label} {key}: QED={head['qed_mean']} "
                          f"MAE={head['qed_mae']} "
                          f"raw={head['raw_validity']:.3f} "
                          f"deg={head['degree_validity']:.3f} "
                          f"yield={head['valid_on_target_yield']:.3f} "
                          f"arom={head['aromatic_rings_mean']}", flush=True)
                low, high = f"N{n}_QED{min(qed_values):g}", f"N{n}_QED{max(qed_values):g}"
                block["paired_response"][str(n)] = paired_response(
                    records_by_cell[low], records_by_cell[high], decoder=args.decoder)
            block["summary"] = summarise(block["cells"], block["paired_response"],
                                          decoder=args.decoder)
            all_records[label] = records_by_cell
            payload["models"][label] = block

            sample_path = out_dir / f"samples_{label}.jsonl"
            with sample_path.open("w", encoding="utf-8") as handle:
                for key, records in records_by_cell.items():
                    for record in records:
                        handle.write(json.dumps({"cell": key, "model": label,
                                                  **record}, default=float) + "\n")
            block["sample_records"] = str(sample_path)
            print(f"  {label}: macro MAE="
                  f"{block['summary']['macro_qed_mae']} "
                  f"macro yield={block['summary']['macro_valid_on_target_yield']} "
                  f"response={block['summary']['mean_paired_response']}", flush=True)

        if not args.skip_grids and len(all_records) >= 1:
            payload["figures"] = render_figures(
                all_records, out_dir=out_dir, baseline=args.baseline_label,
                targets_n=targets_n, qed_values=qed_values, decoder=args.decoder,
                grid_samples=args.grid_samples, best_of_k=args.best_of_k)

        payload["elapsed_seconds"] = time.perf_counter() - started
        atomic_write_json(payload, out_dir / "checkpoint_eval.json")
        (out_dir / "checkpoint_eval.txt").write_text(format_eval(payload))
        print("\n" + format_eval(payload))
        return 0
    except FileNotFoundError as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2
    except ResumeMismatchError as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 3
    except Exception:  # noqa: BLE001
        print("FATAL: unexpected\n" + traceback.format_exc(), file=sys.stderr)
        return 1


def render_figures(all_records, *, out_dir, baseline, targets_n, qed_values, decoder,
                   grid_samples, best_of_k) -> dict[str, Any]:
    """Native comparison grids plus a SEPARATE best-of-k gallery."""
    from zinc_gnf.molecule_grids import best_of_k_gallery, comparison_grid

    figures: dict[str, Any] = {"native_comparisons": {}, "selection_galleries": {}}
    labels = list(all_records)
    reference = baseline if baseline in all_records else labels[0]
    candidates = [label for label in labels if label != reference]
    directory = out_dir / "figures"
    for n in targets_n:
        for qed in qed_values:
            key = f"N{n}_QED{qed:g}"
            for candidate in candidates:
                try:
                    figures["native_comparisons"][f"{key}_{reference}_vs_{candidate}"] = \
                        comparison_grid(
                            all_records[reference][key], all_records[candidate][key],
                            n=n, requested_qed=qed, samples=grid_samples,
                            baseline_label=reference, candidate_label=candidate,
                            output=directory / f"native_{key}_{reference}_vs_{candidate}.png")
                except Exception as error:  # noqa: BLE001
                    figures["native_comparisons"][f"{key}_{reference}_vs_{candidate}"] = \
                        {"error": str(error)[:200]}
            try:
                figures["selection_galleries"][key] = best_of_k_gallery(
                    all_records[reference][key], n=n, requested_qed=qed, k=best_of_k,
                    decoder=decoder, samples=grid_samples, model_label=reference,
                    output=directory / f"bestof{best_of_k}_{key}_{reference}.png")
            except Exception as error:  # noqa: BLE001
                figures["selection_galleries"][key] = {"error": str(error)[:200]}
    return figures


def _f(value, spec=".4f"):
    return format(value, spec) if isinstance(value, (int, float)) else "n/a"


def format_inventory(report) -> str:
    lines = ["Checkpoint inventory", "=" * 78, "",
             f"directory: {report['directory']}",
             f"readable checkpoints: {report['readable_count']}",
             f"distinct steps: {report['distinct_steps']}", "",
             f"{'name':<22} {'step':>8} {'val_loss':>10} {'cond_gap':>10} {'opt':>5}"]
    for entry in report["checkpoints"]:
        if entry.get("unreadable"):
            lines.append(f"{entry['name']:<22} UNREADABLE: {entry['unreadable']}")
            continue
        lines.append(
            f"{entry['name']:<22} {str(entry.get('step')):>8} "
            f"{_f(entry.get('val_loss')):>10} {_f(entry.get('val_condition_gap')):>10} "
            f"{str(bool(entry.get('has_optimizer_state'))):>5}")
    lines += ["", f"evaluation cost: {report['samples_per_checkpoint']} samples per "
                  f"checkpoint, about {report['estimated_generation_minutes']} min total",
              f"  ({report['throughput_assumption']})", "",
              f"VERDICT: {report['verdict']}"]
    return "\n".join(lines)


def format_eval(payload) -> str:
    lines = ["Checkpoint selection on decoded QED and chemistry", "=" * 78, "",
             payload["framing"], "",
             f"split: {payload['settings']['split']}  "
             f"noise seed: {payload['settings']['noise_seed']}  "
             f"samples/cell: {payload['settings']['samples']}  "
             f"decoder: {payload['settings']['decoder']}", "",
             f"{'model':<14} {'step':>7} {'macroMAE':>9} {'yield':>7} {'rawVal':>7} "
             f"{'degVal':>7} {'arom':>6} {'uniq':>6} {'div':>6} {'resp':>8}"]
    for label, block in payload["models"].items():
        s = block["summary"]
        lines.append(
            f"{label:<14} {str(block.get('step')):>7} "
            f"{_f(s['macro_qed_mae']):>9} "
            f"{_f(s['macro_valid_on_target_yield'], '.3f'):>7} "
            f"{_f(s['macro_raw_validity'], '.3f'):>7} "
            f"{_f(s['macro_degree_validity'], '.3f'):>7} "
            f"{_f(s['macro_aromatic_rings'], '.2f'):>6} "
            f"{_f(s['macro_uniqueness'], '.3f'):>6} "
            f"{_f(s['macro_internal_diversity'], '.3f'):>6} "
            f"{_f(s['mean_paired_response'], '+.4f'):>8}")
    lines += ["", "  macro aggregates are EQUALLY WEIGHTED over the nine cells.",
              "  yield = valid AND within "
              f"{payload['settings'].get('samples') and QED_TOLERANCE} of target, over "
              "ALL attempted samples.", "", "-- PER CELL " + "-" * 65]
    for label, block in payload["models"].items():
        lines.append(f"  {label} (step {block.get('step')})")
        for key, cell in block["cells"].items():
            lines.append(
                f"    {key:<14} MAE={_f(cell['qed_mae'])} QED={_f(cell['qed_mean'])} "
                f"raw={_f(cell['raw_validity'], '.3f')} "
                f"deg={_f(cell['degree_validity'], '.3f')} "
                f"yield={_f(cell['valid_on_target_yield'], '.3f')} "
                f"atoms={_f(cell['heavy_atoms_mean'], '.1f')}/"
                f"{cell['requested_n']} "
                f"adher={_f(cell['atom_count_adherence'], '.2f')} "
                f"arom={_f(cell['aromatic_rings_mean'], '.2f')} "
                f"uniq={_f(cell['uniqueness'], '.2f')}")
        for n, response in block["paired_response"].items():
            lines.append(f"    paired response N={n}: {_f(response.get('mean'))} "
                         f"CI={response.get('ci95')} pairs={response.get('pairs')}")
    figures = payload.get("figures") or {}
    if figures:
        lines += ["", "-- FIGURES " + "-" * 66]
        for group, entries in figures.items():
            lines.append(f"  {group}")
            for key, info in entries.items():
                if info.get("error"):
                    lines.append(f"    {key}: NOT RENDERED ({info['error']})")
                    continue
                lines.append(
                    f"    {key}: {info['path']}  cells={info['cells']} "
                    f"drawn={info['drawn']} invalid={info['invalid']} "
                    f"undrawable={info['undrawable']} "
                    f"placeholder_rate={info['placeholder_rate']:.3f}")
        lines.append("  Native grids and best-of-k galleries are SEPARATE figures. "
                     "Placeholders are drawn in place, never substituted.")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
