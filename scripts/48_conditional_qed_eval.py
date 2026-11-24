#!/usr/bin/env python3
"""Conditional QED evaluation: matched N-only baseline vs N+QED model, identical requests.

SCALAR QED inputs throughout. The tolerances (+/-0.1 and +/-0.01) are EVALUATOR
acceptance windows applied after generation; they are never supplied to the model, and no
retraining for interval inputs is implied or required.

Two request sets, kept separate:
  --target-set grid         N in {15,23,30} x QED in {0.4,0.6,0.8}  (continuity)
  --target-set percentile   QED at training percentiles computed WITHIN each N group,
                            with per-target support counts and extrapolation flags

Presets:
  --preset full    2000 attempts per target, 5 sampling repetitions
  --preset smoke     25 attempts per target, 1 repetition (pipeline check)

Both models answer IDENTICAL requests with identical size vectors and identical sampling
seeds. The baseline is given only the conditions it was trained on: an N-only baseline
receives N and never QED; a fully unconditional baseline receives nothing.

Every attempt is saved, including failures, so the per-attempt denominators are honest.
Duplicates are preserved in the primary accuracy metrics. No first-draw or best-of-k
metric is produced.

Resumable: a cell/repetition whose `samples.jsonl` already exists is skipped unless
--force, so a requeued job never overwrites completed work.

Exit codes: 0 ok; 2 missing input; 3 refused; 4 bad arguments; 1 unexpected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zinc_gnf.eval_support import (  # noqa: E402
    DEFAULT_TOLERANCES,
    GRID_TARGETS,
    STRUCTURAL_SIZE_NOTE,
    FixedCounts,
    already_done,
    grid_target_support,
    joint_association,
    macro_and_pooled,
    manifest_entry,
    percentile_targets,
    qed_cell_metrics,
    run_level_summary,
    write_jsonl,
)

PRESETS = {"full": {"attempts": 2000, "repetitions": 5},
           "smoke": {"attempts": 25, "repetitions": 1}}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/cluster.json")
    parser.add_argument("--target-set", choices=("grid", "percentile"), default="grid")
    parser.add_argument("--preset", choices=tuple(PRESETS), default=None)
    parser.add_argument("--attempts", type=int, default=None,
                        help="attempts per target per repetition")
    parser.add_argument("--repetitions", type=int, default=None,
                        help="SAMPLING repetitions of one checkpoint, not training seeds")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="GPU minibatch; attempts are generated in chunks of this")
    parser.add_argument("--decoder", default=None, help="headline decoder")
    parser.add_argument("--baseline-checkpoint", default=None,
                        help="overrides the config; omit to use the configured one")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="evaluate the conditional model only (baseline not trained yet)")
    parser.add_argument("--force", action="store_true", help="recompute completed cells")
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def enrich(records, decoders):
    """Add chemistry fields once, at generation time, so no later step needs RDKit."""
    from rdkit import Chem, RDLogger, rdBase
    from rdkit.Chem import Crippen, Descriptors

    RDLogger.DisableLog("rdApp.*")
    with rdBase.BlockLogs():
        for record in records:
            for name in decoders:
                entry = record.get(name)
                if not entry or not entry.get("valid") or not entry.get("smiles"):
                    continue
                molecule = Chem.MolFromSmiles(entry["smiles"])
                if molecule is None:
                    entry["valid"] = False
                    continue
                rings = molecule.GetRingInfo()
                entry["heavy_atoms"] = int(molecule.GetNumHeavyAtoms())
                entry["rings"] = int(rings.NumRings())
                entry["aromatic_rings"] = int(sum(
                    1 for ring in rings.AtomRings()
                    if all(molecule.GetAtomWithIdx(i).GetIsAromatic() for i in ring)))
                entry["molecular_weight"] = float(Descriptors.MolWt(molecule))
                entry["logp"] = float(Crippen.MolLogP(molecule))
    return records


def cell_seed(base: int, repetition: int, position: int) -> int:
    """Deterministic and documented: base + 1000*repetition + position."""
    return int(base) + 1000 * int(repetition) + int(position)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        config = json.loads(Path(args.config).read_text())
        paths, run = config["paths"], config.get("run", {})
        preset = PRESETS[args.preset] if args.preset else {}
        attempts = args.attempts or preset.get("attempts") or run.get("attempts", 2000)
        repetitions = (args.repetitions or preset.get("repetitions")
                       or run.get("repetitions", 5))
        temperature = (args.temperature if args.temperature is not None
                       else run.get("conditional_temperature", 0.2))
        batch_size = args.batch_size or run.get("batch_size", 250)
        decoder = args.decoder or run.get("headline_decoder", "degree")
        base_seed = int(run.get("sampling_seed", 20261124))
        tolerances = tuple(run.get("tolerances", DEFAULT_TOLERANCES))
        if attempts < 1 or repetitions < 1 or batch_size < 1:
            raise ValueError("attempts, repetitions and batch size must all be >= 1")
    except (OSError, ValueError, KeyError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 4

    started = time.perf_counter()
    try:
        import torch

        from zinc_gnf.conditioning import load_condition_scaler, standardise_condition
        from zinc_gnf.dataset import MolecularGraphDataset, load_training_splits
        from zinc_gnf.generation import generate_molecules
        from zinc_gnf.metrics_generation import DECODERS
        from zinc_gnf.trainer_autoencoder import load_autoencoder, resolve_device
        from zinc_gnf.trainer_flow import file_sha256, load_flow

        device = resolve_device(args.device)
        out_root = Path(paths["results_dir"]) / "conditional" / args.target_set
        out_root.mkdir(parents=True, exist_ok=True)

        train_records, _, fingerprint = load_training_splits(paths["data_dir"])
        train_set = MolecularGraphDataset(train_records)
        train_smiles = {str(r["smiles"]) for r in train_records}
        scaler = load_condition_scaler(paths["data_dir"])
        autoencoder, _ = load_autoencoder(paths["ae_checkpoint"], device=device)
        autoencoder.eval()
        for parameter in autoencoder.parameters():
            parameter.requires_grad_(False)

        # ---- request set, with measured support ----
        association = joint_association(train_set.n_node, train_set.qed)
        if args.target_set == "grid":
            support = grid_target_support(train_set.n_node, train_set.qed)
            cells = [{"n": row["n"], "qed": row["qed"], "support": row["support"],
                      "extrapolation": row["extrapolation"]} for row in support]
            target_definition = {"kind": "grid", "n": list(GRID_TARGETS["n"]),
                                 "qed": list(GRID_TARGETS["qed"]), "support": support}
        else:
            chosen = percentile_targets(train_set.n_node, train_set.qed)
            cells = [{"n": row["n"], "qed": row["qed"], "percentile": row["percentile"],
                      "support": row["support"], "extrapolation": row["extrapolation"]}
                     for row in chosen["targets"] if row["qed"] is not None]
            target_definition = {"kind": "percentile", **chosen}
        if not cells:
            raise ValueError("no usable targets")

        # ---- models ----
        models = {}
        conditional, cond_probe = load_flow(paths["conditional_checkpoint"], device=device)
        models["conditional"] = (conditional, paths["conditional_checkpoint"], cond_probe)
        baseline_path = args.baseline_checkpoint or paths.get("baseline_checkpoint")
        if not args.skip_baseline and baseline_path and Path(baseline_path).is_file():
            baseline, base_probe = load_flow(baseline_path, device=device)
            models["baseline"] = (baseline, baseline_path, base_probe)
        elif not args.skip_baseline:
            print(f"WARNING: no baseline checkpoint at {baseline_path!r}; evaluating the "
                  f"conditional model only. Train it with the config exported by "
                  f"scripts/47_cluster_preflight.py, then re-run (completed cells are "
                  f"skipped).", file=sys.stderr)

        manifest = {"target_set": args.target_set, "targets": target_definition,
                    "n_qed_association": association,
                    "structural_size_note": STRUCTURAL_SIZE_NOTE,
                    "tolerance_note": ("tolerances are EVALUATOR acceptance windows "
                                       "applied after generation; the model receives a "
                                       "SCALAR target only"),
                    "models": {}, "cells": [], "elapsed_seconds": None}

        for label, (flow, path, probe) in models.items():
            features = tuple(flow.config.condition_features)
            manifest["models"][label] = manifest_entry(
                flow_checkpoint=str(path), flow_sha256=file_sha256(path),
                decoder_checkpoint=str(paths["ae_checkpoint"]),
                decoder_sha256=file_sha256(paths["ae_checkpoint"]),
                temperature=temperature, sampling_seed=base_seed,
                attempts=attempts, repetitions=repetitions, decoder=decoder,
                training_seed=(probe.get("train_config") or {}).get("seed"),
                condition_features=list(features),
                flow_config=probe["flow_config"],
                label=("N+QED conditional" if set(features) == {"n", "qed"} else
                       "matched N-only baseline" if features == ("n",) else
                       "unconditional baseline"),
                data_fingerprint=fingerprint)

        decoders = tuple(d for d in DECODERS)
        per_cell_rows = []
        for position, cell in enumerate(cells):
            n_target, qed_target = int(cell["n"]), float(cell["qed"])
            tag = f"n{n_target}_qed{qed_target:.3f}".replace(".", "p")
            for label, (flow, path, probe) in models.items():
                features = tuple(flow.config.condition_features)
                reps = []
                for repetition in range(repetitions):
                    directory = out_root / label / tag / f"rep{repetition}"
                    samples_path = directory / "samples.jsonl"
                    metrics_path = directory / "metrics.json"
                    seed = cell_seed(base_seed, repetition, position)
                    if already_done(metrics_path, force=args.force):
                        reps.append(json.loads(metrics_path.read_text()))
                        continue
                    # QED is supplied ONLY to a model trained with it
                    if not features:
                        values = None
                    else:
                        values = standardise_condition(
                            features, scaler=scaler, n=[n_target] * attempts,
                            qed=([qed_target] * attempts if "qed" in features else None))
                    drawn = generate_molecules(
                        flow, autoencoder, count=attempts,
                        node_counts=FixedCounts([n_target] * attempts),
                        target_n=n_target, temperature=temperature, seed=seed,
                        batch_size=batch_size, decoders=decoders,
                        condition_values=None if values is None
                        else torch.from_numpy(values), device=device)
                    records = enrich(drawn["records"], decoders)
                    for record in records:
                        record.update({"model": label, "cell": tag,
                                       "repetition": repetition, "sampling_seed": seed,
                                       "requested_qed": qed_target,
                                       "requested_n": n_target,
                                       "qed_supplied_to_model": "qed" in features})
                    write_jsonl(samples_path, records)
                    metrics = qed_cell_metrics(
                        records, requested_qed=qed_target, requested_n=n_target,
                        decoder=decoder, tolerances=tolerances,
                        train_smiles=train_smiles)
                    metrics.update({"model": label, "cell": tag, "repetition": repetition,
                                    "sampling_seed": seed, "temperature": temperature,
                                    "support": cell.get("support"),
                                    "extrapolation": cell.get("extrapolation"),
                                    "qed_supplied_to_model": "qed" in features,
                                    "samples_file": str(samples_path)})
                    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
                    reps.append(metrics)
                    print(f"  {label:12s} {tag} rep{repetition}: "
                          f"valid {metrics['valid']}/{metrics['attempted']}, "
                          f"MAE {metrics.get('qed_mae')}, "
                          f"yield+/-0.1 "
                          f"{metrics.get('valid_hit_yield_within_0p1_over_all_attempts')}")
                combined = {"model": label, "cell": tag, "n": n_target,
                            "qed": qed_target, "repetitions": len(reps),
                            "support": cell.get("support"),
                            "extrapolation": cell.get("extrapolation"),
                            "across_repetitions": {}}
                for field in ("qed_mae", "qed_bias", "achieved_qed_mean",
                              "achieved_qed_sd", "validity_over_attempts",
                              "achieved_atoms_mean",
                              "size_match_over_all_attempts",
                              "uniqueness_over_valid", "nuv_over_attempts",
                              *(f"hit_rate_within_{f'{t:g}'.replace('.', 'p')}_among_scored"
                                for t in tolerances),
                              *(f"valid_hit_yield_within_{f'{t:g}'.replace('.', 'p')}"
                                f"_over_all_attempts" for t in tolerances)):
                    values = [r[field] for r in reps
                              if r.get(field) is not None]
                    if values:
                        combined["across_repetitions"][field] = run_level_summary(values)
                # pooled over repetitions uses every scored attempt once
                combined["pooled_over_repetitions"] = {
                    "attempted": sum(r["attempted"] for r in reps),
                    "valid": sum(r["valid"] for r in reps),
                    "scored": sum(r["scored"] for r in reps),
                }
                per_cell_rows.append(combined)
                manifest["cells"].append({"model": label, "cell": tag,
                                          "n": n_target, "qed": qed_target,
                                          "repetitions": len(reps)})

        summary = {"per_cell": per_cell_rows, "by_model": {}}
        for label in models:
            flat = []
            for repetition in range(repetitions):
                cells_this_rep = []
                for position, cell in enumerate(cells):
                    tag = f"n{int(cell['n'])}_qed{float(cell['qed']):.3f}".replace(".", "p")
                    path = out_root / label / tag / f"rep{repetition}" / "metrics.json"
                    if path.is_file():
                        cells_this_rep.append(json.loads(path.read_text()))
                if cells_this_rep:
                    flat.append(macro_and_pooled(cells_this_rep, tolerances=tolerances))
            summary["by_model"][label] = {
                "per_repetition": flat,
                "macro_qed_mae_across_repetitions": run_level_summary(
                    [f["macro"]["qed_mae"] for f in flat
                     if f["macro"].get("qed_mae") is not None]),
                "pooled_qed_mae_across_repetitions": run_level_summary(
                    [f["pooled"]["qed_mae"] for f in flat
                     if f["pooled"].get("qed_mae") is not None]),
                "note": ("macro = equal weight per cell; pooled = weighted by scored "
                         "molecules, which shifts toward high-validity small-N cells"),
            }
        manifest["elapsed_seconds"] = time.perf_counter() - started
        (out_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        print(f"\nwrote {out_root / 'summary.json'} and manifest.json")
        for label, block in summary["by_model"].items():
            macro = block["macro_qed_mae_across_repetitions"]
            print(f"  {label:12s} macro QED MAE {macro['mean']:.4f} +/- {macro['sd']:.4f} "
                  f"over {macro['runs']} sampling repetitions")
        print("\nRun scripts/49_summarize.py for CSV tables and plots (no regeneration).")
        return 0
    except FileNotFoundError as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2
    except (ValueError, KeyError) as error:
        print(f"FATAL (refused): {error}", file=sys.stderr)
        return 3
    except Exception:  # noqa: BLE001
        print("FATAL: unexpected\n" + traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
