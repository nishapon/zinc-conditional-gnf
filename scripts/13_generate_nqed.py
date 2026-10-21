#!/usr/bin/env python3
"""Stage 13 — joint (N, QED)-conditioned generation, target achievement and controls.

Four blocks, in this order, from one trained joint checkpoint:

A. CONTROLS. Three permutation tests on the same VALIDATION molecules with the same
   smoothing noise, differing only in which condition vector a molecule is scored against:
     both      every feature permuted        - is the conditioning used at all?
     qed-only  QED permuted, N kept TRUE     - is QED used for something N cannot explain?
     n-only    N permuted, QED kept TRUE     - is N used for something QED cannot explain?
   The qed-only test is the one stage 13 stands on: N and QED are correlated in ZINC, so a
   model that learned N alone still looks conditional under a joint permutation. Plus
   mu(c), sigma(c) swept over QED at FIXED N - a flat curve there means QED never reaches
   the prior.

B. DECODE CEILING. The same three decoders run on REAL validation embeddings from the
   frozen autoencoder. 0% raw validity means something different depending on this row: if
   real embeddings decode to valid aromatic molecules under the raw argmax, the bond
   prediction works and the sampled embeddings are the problem (modelling); if they do not,
   the raw argmax itself is the bottleneck and flow training cannot fix it. Every generated
   raw validity below is reported against this ceiling, never against 100%.

C. GENERATION over a grid of requested (N, QED) - and, if asked, more than one sampling
   temperature, because temperature trades validity against aromaticity and the frozen
   unconditional result used 0.2. Each cell reports raw / greedy / degree-constrained
   validity SEPARATELY, the degree fallback rate, aromatic rings per molecule, QED,
   uniqueness, novelty against the TRAINING split and internal diversity, beside the real
   training molecules of the same N.

D. QED TARGET ACHIEVEMENT: MAE, bias, RMSE, hit rate within +-0.1 and the
   requested-vs-realised correlation, grouped by N and by requested QED, with the
   within-N correlation as the headline that the N-QED correlation in ZINC cannot explain.

The TEST split is never loaded. Exit codes: 0 ok; 2 missing checkpoint or data;
3 autoencoder mismatch; 4 bad arguments; 5 a control failed; 1 unexpected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from zinc_gnf.conditioning import (  # noqa: E402
    condition_sensitivity,
    latent_spread_by_n,
    load_condition_scaler,
    prior_response,
    standardise_condition,
)
from zinc_gnf.dataset import (  # noqa: E402
    DatasetFormatError,
    MolecularGraphDataset,
    load_training_splits,
    seeded_subset_indices,
)
from zinc_gnf.generation import (  # noqa: E402
    DECODERS,
    NodeCountSampler,
    generate_molecules,
    requested_n_report,
)
from zinc_gnf.metrics_generation import generation_metrics  # noqa: E402
from zinc_gnf.molecular_profile import profile_molecules  # noqa: E402
from zinc_gnf.qed_evaluation import (  # noqa: E402
    DEFAULT_QED_TOLERANCE,
    decode_ceiling,
    format_qed_achievement,
    qed_achievement_table,
    qed_target_metrics,
)
from zinc_gnf.trainer_autoencoder import ResumeMismatchError, load_autoencoder, resolve_device  # noqa: E402
from zinc_gnf.trainer_flow import VALIDATION_NOISE_SEED, VALIDATION_SUBSET_SEED, file_sha256, load_flow  # noqa: E402
from zinc_gnf.utils.logging import JobLogger  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402

# Sanity floors for the controls, not success criteria. A smoke run may fail them, which
# is what --allow-weak-conditioning is for.
MIN_GAP = 1e-4
MIN_WIN_RATE = 0.55
DEFAULT_QED_TARGETS = (0.4, 0.6, 0.8)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="data/processed/zinc250k_charge_aware_moflow_fixed")
    parser.add_argument("--ae-checkpoint", default=None)
    parser.add_argument("--target-n", default="15,23,30", help="comma-separated atom counts")
    parser.add_argument("--target-qed", default=",".join(str(q) for q in DEFAULT_QED_TARGETS),
                        help="comma-separated requested QED values")
    parser.add_argument("--temperature", default="0.2",
                        help="comma-separated; more than one measures the validity / "
                             "aromaticity trade-off in the same job")
    parser.add_argument("--count", type=int, default=500, help="samples per (T, N, QED) cell")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decoders", default=",".join(DECODERS))
    parser.add_argument("--headline", default="degree", choices=DECODERS)
    parser.add_argument("--qed-tolerance", type=float, default=DEFAULT_QED_TOLERANCE)
    parser.add_argument("--check-molecules", type=int, default=2000)
    parser.add_argument("--ceiling-molecules", type=int, default=1000,
                        help="validation molecules for the decode ceiling (0 = skip)")
    parser.add_argument("--no-diversity", action="store_true")
    parser.add_argument("--save-samples", action="store_true", default=True)
    parser.add_argument("--allow-ae-mismatch", action="store_true")
    parser.add_argument("--allow-weak-conditioning", action="store_true",
                        help="report a failed control instead of exiting 5 (smoke tests)")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def numbers(text: str, kind=float) -> list:
    return [kind(part) for part in str(text).split(",") if str(part).strip()]


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - a report, read top to bottom
    arguments = parse_args(argv)
    output_root = Path(arguments.output_dir)
    try:
        decoders = [d.strip() for d in arguments.decoders.split(",") if d.strip()]
        unknown = set(decoders) - set(DECODERS)
        if unknown:
            raise ValueError(f"unknown decoders {sorted(unknown)}")
        targets_n = numbers(arguments.target_n, int)
        targets_qed = numbers(arguments.target_qed, float)
        temperatures = numbers(arguments.temperature, float)
        if not targets_n or not targets_qed or not temperatures:
            raise ValueError("--target-n, --target-qed and --temperature must be non-empty")
        if min(targets_n) < 1:
            raise ValueError("every requested N must be >= 1")
        if any(not 0.0 <= q <= 1.0 for q in targets_qed):
            raise ValueError("requested QED values must lie in [0, 1]")
        if any(t <= 0 for t in temperatures):
            raise ValueError("every temperature must be > 0")
        if arguments.count < 1:
            raise ValueError("--count must be >= 1")
        output_root.mkdir(parents=True, exist_ok=True)
        logger = JobLogger("generate_nqed", output_root, filename="generate_nqed.log")
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4

    started = time.perf_counter()
    try:
        device = resolve_device(arguments.device)
        flow, probe = load_flow(arguments.flow_checkpoint, device=device)
        features = flow.config.condition_features
        if set(features) != {"n", "qed"}:
            raise ValueError(f"stage 13 needs a flow conditioned on BOTH n and qed; this one "
                             f"has {list(features)}")
        recorded = probe.get("autoencoder") or {}
        ae_path = Path(arguments.ae_checkpoint or recorded.get("checkpoint", ""))
        if not ae_path.exists():
            raise FileNotFoundError(f"autoencoder checkpoint not found: {ae_path}")
        digest = file_sha256(ae_path)
        if recorded.get("sha256") and digest != recorded["sha256"]:
            message = (f"autoencoder mismatch: flow trained with sha256 "
                       f"{recorded['sha256'][:16]} but {ae_path} has {digest[:16]}")
            if not arguments.allow_ae_mismatch:
                raise ResumeMismatchError(message)
            logger.warn(message + " (continuing because --allow-ae-mismatch was given)")
        autoencoder, _ = load_autoencoder(ae_path, device=device)
        autoencoder.eval()
        for parameter in autoencoder.parameters():
            parameter.requires_grad_(False)
        if "degree" in decoders:
            if not autoencoder.config.degree_head:
                raise ValueError("this autoencoder has no degree head; drop 'degree'")
            try:
                from scipy.optimize import milp  # noqa: F401
            except ImportError as error:
                raise ValueError("the degree decoder needs scipy.optimize.milp "
                                 f"(scipy >= 1.9): {error}") from error

        scaler = load_condition_scaler(arguments.data_dir)
        train_records, val_records, fingerprint = load_training_splits(arguments.data_dir)
        train_counts = [int(r["n_node"]) for r in train_records]
        train_smiles = {str(r["smiles"]) for r in train_records}
        validation_smiles = {str(r["smiles"]) for r in val_records}
        val_set = MolecularGraphDataset(val_records)
        real_by_n: dict[int, list[dict[str, Any]]] = {}
        for record in train_records:
            real_by_n.setdefault(int(record["n_node"]), []).append(
                {"smiles": str(record["smiles"]), "qed": float(record["qed"])})
        del train_records, val_records
        node_counts = NodeCountSampler(train_counts)

        logger.header("Joint (N, QED)-conditioned generation")
        logger.info(f"flow={arguments.flow_checkpoint} step={probe.get('step')} "
                    f"features={list(features)} "
                    f"node_autoregressive={flow.config.node_autoregressive}")
        logger.info(f"autoencoder={ae_path} sha256={digest[:16]} (frozen, read-only)")
        logger.info(f"grid: N={targets_n} x QED={targets_qed} x T={temperatures}, "
                    f"{arguments.count} samples per cell "
                    f"({len(targets_n) * len(targets_qed) * len(temperatures) * arguments.count:,} total)")

        # ---- A. controls -----------------------------------------------------
        check_indices = seeded_subset_indices(len(val_set), arguments.check_molecules,
                                              VALIDATION_SUBSET_SEED)
        controls: dict[str, Any] = {}
        for name, permute in (("both", None), ("qed_only", ["qed"]), ("n_only", ["n"])):
            controls[name] = condition_sensitivity(
                flow, autoencoder, val_set, indices=check_indices, device=device,
                batch_size=arguments.batch_size, noise_seed=VALIDATION_NOISE_SEED,
                permute_features=permute)
            logger.info(f"control '{name}' (permuting "
                        f"{controls[name]['permuted_features']}): gap "
                        f"{controls[name]['gap_nats_per_node']:+.4f} nats/node, win rate "
                        f"{controls[name]['molecules_better_with_true_condition'] * 100:.1f}%, "
                        f"ties {controls[name]['exact_ties'] * 100:.1f}%")
        # mu/sigma over QED at FIXED N, so the response cannot be N acting instead. The
        # fixed N is an actually REQUESTED count (the middle one), not an interpolated
        # median, so the curve belongs to a cell that is really generated below.
        middle_n = sorted(targets_n)[len(targets_n) // 2]
        qed_sweep = [round(float(q), 3) for q in np.linspace(0.3, 0.9, 7)]
        values = standardise_condition(features, scaler=scaler,
                                       n=[middle_n] * len(qed_sweep), qed=qed_sweep)
        with torch.no_grad():
            embedding = flow.conditioning(torch.from_numpy(values).to(device), None,
                                          len(qed_sweep), device)
            mu, sigma = flow.prior.parameters_for(len(qed_sweep), embedding, device)
            mu, sigma = mu.detach().cpu(), sigma.detach().cpu()
        prior_qed = {
            "fixed_n": middle_n,
            "qed_values": qed_sweep,
            "rows": [{"qed": float(q), "mu_norm": float(mu[i].norm()),
                      "sigma_mean": float(sigma[i].mean())} for i, q in enumerate(qed_sweep)],
            "mu_spread_across_qed": float(mu.std(dim=0).mean()),
            "sigma_spread_across_qed": float(sigma.std(dim=0).mean()),
        }
        logger.info(f"prior response to QED at fixed N={middle_n}: mu spread "
                    f"{prior_qed['mu_spread_across_qed']:.4f}, sigma spread "
                    f"{prior_qed['sigma_spread_across_qed']:.4f} (0 = QED never reaches "
                    f"the prior)")
        prior_n = prior_response(flow, scaler=scaler, n_values=targets_n,
                                 qed=float(np.median(targets_qed)), device=device)
        spread = latent_spread_by_n(flow, autoencoder, val_set, n_values=targets_n,
                                    indices=check_indices, device=device,
                                    batch_size=arguments.batch_size,
                                    noise_seed=VALIDATION_NOISE_SEED)

        passed = {
            name: bool(controls[name]["gap_nats_per_node"] > MIN_GAP
                       and controls[name]["molecules_better_with_true_condition"] > MIN_WIN_RATE)
            for name in controls
        }
        # QED-only is the decisive one for this stage.
        verification = {"controls": controls, "passed_per_control": passed,
                        "prior_response_over_qed_at_fixed_n": prior_qed,
                        "prior_response_over_n": prior_n,
                        "latent_spread_by_n": spread,
                        "thresholds": {"min_gap_nats_per_node": MIN_GAP,
                                       "min_win_rate": MIN_WIN_RATE},
                        "passed": bool(passed["both"] and passed["qed_only"]),
                        "validation_molecules": len(check_indices)}
        if not verification["passed"]:
            logger.error(
                "CONTROL FAILED: "
                + ("the joint conditioning is not used; " if not passed["both"] else "")
                + ("QED adds nothing beyond N (permuting QED alone costs nothing), so any "
                   "apparent QED control would be N acting through the N-QED correlation in "
                   "ZINC; " if not passed["qed_only"] else "")
                + "no QED-achievement table is trustworthy in this state.")
            if not arguments.allow_weak_conditioning:
                atomic_write_json({"verification": verification},
                                  output_root / "conditioning_check.json")
                return 5
            logger.warn("continuing because --allow-weak-conditioning was given; every number "
                        "below comes from a model whose QED control is unproven")

        # ---- B. decode ceiling ------------------------------------------------
        ceiling = None
        if arguments.ceiling_molecules:
            ceiling_indices = seeded_subset_indices(len(val_set), arguments.ceiling_molecules,
                                                    VALIDATION_SUBSET_SEED)
            ceiling = decode_ceiling(autoencoder, val_set, indices=ceiling_indices,
                                     decoders=tuple(decoders), device=str(device),
                                     batch_size=arguments.batch_size)
            for name, row in ceiling["decoders"].items():
                logger.info(f"decode ceiling [{name}] on {ceiling['molecules']:,} REAL "
                            f"validation embeddings: strict validity "
                            f"{row['strict_validity']:.4f}, aromatic rings "
                            f"{row['aromatic_rings_mean']:.4f}, QED {row['qed_mean']:.4f}, "
                            f"fallback {row['fallback_rate']}")
            raw_row = ceiling["decoders"].get("raw")
            if raw_row is not None:
                if raw_row["strict_validity"] > 0.5:
                    logger.info("the raw argmax decodes REAL embeddings to valid molecules, so "
                                "raw bond prediction is not broken: a generated raw validity "
                                "near 0 is a property of the SAMPLED embeddings (modelling), "
                                "not a decoding defect")
                else:
                    logger.warn(f"the raw argmax reaches only "
                                f"{raw_row['strict_validity']:.4f} validity on REAL "
                                f"embeddings: the raw decode path itself is the bottleneck, "
                                f"and no amount of flow training would raise raw validity "
                                f"above this ceiling")

        atomic_write_json({"verification": verification, "decode_ceiling": ceiling},
                          output_root / "conditioning_check.json")

        if arguments.skip_generation:
            atomic_write_json({"verification": verification, "decode_ceiling": ceiling,
                               "cells": None, "seconds": time.perf_counter() - started},
                              output_root / "summary.json")
            logger.info("--skip-generation: no molecules sampled")
            return 0

        # ---- C. generation over the grid -------------------------------------
        cells: list[dict[str, Any]] = []
        position = 0
        for temperature in temperatures:
            for target_n in targets_n:
                for target_qed in targets_qed:
                    position += 1
                    key = f"T{temperature:g}_N{target_n}_QED{target_qed:g}"
                    directory = output_root / key
                    directory.mkdir(parents=True, exist_ok=True)
                    values = standardise_condition(features, scaler=scaler,
                                                   n=[target_n] * arguments.count,
                                                   qed=[target_qed] * arguments.count)
                    sampled = generate_molecules(
                        flow, autoencoder, count=arguments.count, node_counts=node_counts,
                        temperature=temperature, seed=arguments.seed + position,
                        batch_size=arguments.batch_size, decoders=decoders,
                        condition_values=values, target_n=target_n, device=device)
                    records = sampled["records"]
                    metrics = generation_metrics(records, train_smiles=train_smiles,
                                                 validation_smiles=validation_smiles,
                                                 decoders=decoders,
                                                 headline=arguments.headline,
                                                 diversity=not arguments.no_diversity)
                    metrics["settings"] = sampled["settings"]
                    metrics["requested"] = {"n": target_n, "qed": target_qed,
                                            "temperature": temperature,
                                            "standardised": [float(v) for v in values[0]]}
                    metrics["requested_n_report"] = requested_n_report(records, target_n,
                                                                       decoders=decoders)
                    smiles = [r[arguments.headline].get("smiles")
                              if (r.get(arguments.headline) or {}).get("valid") else None
                              for r in records]
                    metrics["profile"] = profile_molecules(smiles, name=key, scaffolds=True)
                    achievement = qed_target_metrics(records, requested_qed=target_qed,
                                                     decoder=arguments.headline,
                                                     tolerance=arguments.qed_tolerance)
                    metrics["qed_achievement"] = achievement
                    reference = real_by_n.get(target_n, [])
                    metrics["real_reference_same_n"] = {
                        "molecules": len(reference),
                        "qed_mean": float(np.mean([r["qed"] for r in reference]))
                        if reference else None,
                        "profile": profile_molecules([r["smiles"] for r in reference[:1000]],
                                                     name=f"real_N{target_n}", scaffolds=False)
                        if reference else None,
                    }
                    metrics["decode_ceiling"] = ceiling
                    metrics["flow_checkpoint"] = str(arguments.flow_checkpoint)
                    metrics["flow_step"] = probe.get("step")
                    metrics["autoencoder"] = {"checkpoint": str(ae_path), "sha256": digest}
                    atomic_write_json(metrics, directory / "metrics.json")
                    if arguments.save_samples:
                        with (directory / "samples.jsonl").open("w", encoding="utf-8") as handle:
                            for record in records:
                                handle.write(json.dumps(record, default=float) + "\n")
                    head = metrics["decoders"].get(arguments.headline, {})
                    raw = metrics["decoders"].get("raw", {})
                    real_profile = (metrics["real_reference_same_n"].get("profile") or {})
                    cells.append({
                        "key": key, "n": target_n, "requested_qed": target_qed,
                        "temperature": temperature, "metrics": achievement,
                        "raw_validity": (raw.get("strict_validity") or {}).get("value"),
                        "headline_validity": (head.get("strict_validity") or {}).get("value"),
                        "fallback_rate": (head.get("fallback_rate") or {}).get("value"),
                        "uniqueness": (head.get("uniqueness") or {}).get("value"),
                        "novelty_vs_train": (head.get("novelty_vs_train") or {}).get("value"),
                        "internal_diversity": (head.get("diversity") or {}
                                               ).get("internal_diversity"),
                        "aromatic_rings_mean": metrics["profile"]["aromatic_rings"]["mean"],
                        "aromatic_ring_fraction": metrics["profile"]["aromatic_ring_fraction"],
                        "qed_mean": metrics["profile"]["qed"]["mean"],
                        "delivered_n_after_decoding": metrics["requested_n_report"][
                            "after_decoding"][arguments.headline]["match_over_all_samples"],
                        "real_qed_same_n": metrics["real_reference_same_n"]["qed_mean"],
                        "real_aromatic_rings_same_n": (real_profile.get("aromatic_rings") or {}
                                                       ).get("mean"),
                        "directory": str(directory),
                    })
                    logger.info(
                        f"{key}: raw validity {cells[-1]['raw_validity']}, "
                        f"{arguments.headline} validity {cells[-1]['headline_validity']}, "
                        f"fallback {cells[-1]['fallback_rate']}, aromatic rings "
                        f"{cells[-1]['aromatic_rings_mean']:.3f} "
                        f"(real {cells[-1]['real_aromatic_rings_same_n']}), QED "
                        f"{achievement['realised_qed_mean']:.4f} vs requested {target_qed:g} "
                        f"(MAE {achievement['mae']:.4f}, hit "
                        f"{achievement['hit_rate'] * 100:.1f}%)")

        # ---- D. achievement table --------------------------------------------
        table = qed_achievement_table(cells, tolerance=arguments.qed_tolerance)
        report = format_qed_achievement(table, tolerance=arguments.qed_tolerance)
        for line in report.splitlines():
            logger.info(line)

        lines = ["Joint (N, QED) conditional generation", "", report, "",
                 "Per cell (headline decoder: " + arguments.headline + ")", ""]
        lines.append("| T | N | requested QED | realised QED | MAE | hit +-"
                     f"{arguments.qed_tolerance:g} | raw validity | {arguments.headline}"
                     " validity | fallback | aromatic rings | real aromatic | real QED |")
        lines.append("|--:|--:|--------------:|-------------:|----:|-------:|-------------:|"
                     "------------------:|---------:|---------------:|-------------:|---------:|")

        def cell(value: Any, template: str = "{:.4f}") -> str:
            return "N/A" if value is None or (isinstance(value, float) and not np.isfinite(value)) \
                else template.format(float(value))

        for row in cells:
            lines.append(
                f"| {row['temperature']:g} | {row['n']} | {row['requested_qed']:g} | "
                f"{cell(row['metrics']['realised_qed_mean'])} | {cell(row['metrics']['mae'])} | "
                f"{cell(row['metrics']['hit_rate'], '{:.1%}')} | {cell(row['raw_validity'])} | "
                f"{cell(row['headline_validity'])} | {cell(row['fallback_rate'])} | "
                f"{cell(row['aromatic_rings_mean'], '{:.3f}')} | "
                f"{cell(row['real_aromatic_rings_same_n'], '{:.3f}')} | "
                f"{cell(row['real_qed_same_n'], '{:.3f}')} |")
        if ceiling:
            lines += ["", "Decode ceiling on REAL validation embeddings "
                      f"({ceiling['molecules']:,} molecules) - the most this decode path can do:"]
            for name, row in ceiling["decoders"].items():
                lines.append(f"  [{name}] strict validity {row['strict_validity']:.4f}, "
                             f"aromatic rings {row['aromatic_rings_mean']:.3f}, "
                             f"QED {row['qed_mean']:.4f}, fallback {row['fallback_rate']}")
        lines += ["",
                  "Raw validity is the model's own bond prediction with no chemistry applied; "
                  "the headline column adds the degree-constrained program AND its greedy "
                  "fallback, whose rate is beside it. They are different quantities and are "
                  "never merged. Aromaticity is enforced by no decoder, so the aromatic-ring "
                  "column is a model property.",
                  "",
                  "Controls (validation molecules, permuted conditions):"]
        for name, control in controls.items():
            lines.append(f"  {name:>8} (permuting {control['permuted_features']}): gap "
                         f"{control['gap_nats_per_node']:+.4f} nats/node, win rate "
                         f"{control['molecules_better_with_true_condition'] * 100:.1f}%"
                         f"{'  <- decisive for QED control' if name == 'qed_only' else ''}")
        (output_root / "nqed_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        atomic_write_json({"verification": verification, "decode_ceiling": ceiling,
                           "cells": cells, "qed_achievement": table,
                           "settings": {"targets_n": targets_n, "targets_qed": targets_qed,
                                        "temperatures": temperatures, "count": arguments.count,
                                        "seed": arguments.seed,
                                        "headline_decoder": arguments.headline,
                                        "decoders": decoders,
                                        "qed_tolerance": arguments.qed_tolerance},
                           "flow": {"checkpoint": str(arguments.flow_checkpoint),
                                    "step": probe.get("step"), "features": list(features)},
                           "autoencoder": {"checkpoint": str(ae_path), "sha256": digest},
                           "data_fingerprint": fingerprint,
                           "seconds": time.perf_counter() - started},
                          output_root / "summary.json")
        print("\n".join(lines), flush=True)
        logger.info(f"wrote {output_root}/summary.json, nqed_report.txt, "
                    f"conditioning_check.json and one directory per cell")
        return 0

    except FileNotFoundError as error:
        print(f"FATAL (missing input): {error}", file=sys.stderr, flush=True)
        return 2
    except DatasetFormatError as error:
        print(f"FATAL (dataset): {error}", file=sys.stderr, flush=True)
        return 2
    except ResumeMismatchError as error:
        print(f"FATAL (autoencoder mismatch): {error}", file=sys.stderr, flush=True)
        return 3
    except ValueError as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
