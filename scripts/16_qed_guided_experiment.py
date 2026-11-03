#!/usr/bin/env python3
"""Stage 15b — three ways to hit a QED target, at the SAME decoded-candidate budget.

Uses the existing 50k joint checkpoint read-only. Nothing is retrained except a small
latent->QED predictor, which is fitted on TRAINING molecules only.

  native   the model's own conditional sampling. This is the only arm that is evidence
           about what the GNF learned.
  rerank   k native candidates per kept molecule, the one whose RDKit QED is closest to
           the target is kept. POST-SELECTION with an oracle - never native conditioning.
  guided   MoFlow's mechanism (verified in mflow/optimize_property.py) adapted to a
           target: normalised gradient steps on (q_hat(z) - q*)^2 using the latent
           predictor, then one decode. Model-based, but still not native conditioning,
           because an external predictor supplies the direction.

Budget. `native` and `guided` decode one candidate per kept molecule; `rerank` decodes
k. The comparison therefore fixes the number of DECODED candidates: with --count m and
--rerank-group k, native and guided are run at m*k candidates too, so every arm is read
at the same cost. Both the equal-budget table and the per-slot table are written, and
the budget accounting is printed with them.

Guardrails that make this honest:
  * the predictor is fitted on TRAIN latents, and its HELD-OUT accuracy is reported; if
    the held-out MAE exceeds --predictor-mae-gate the guided arm is skipped rather than
    reported, because guidance driven by a bad guide is noise with a direction
  * no validation or test label is used to choose targets, steps, or anything else
  * reranking and guidance carry their labels from qed_guidance.METHOD_LABELS into
    every output file
  * optionally (--double-inverse) the inverse runs in float64, so a numerical artefact
    cannot be mistaken for a chemistry result

Reports per arm and per (N, QED) cell: QED MAE, bias, hit rate within +-0.1, raw
validity, degree-decoded validity, decoder fallback, aromatic rings per molecule,
novelty, uniqueness and internal diversity.

Exit codes: 0 ok; 2 missing input; 3 autoencoder mismatch; 4 bad arguments; 1 unexpected.
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
    condition_embedding_for,
    load_condition_scaler,
    standardise_condition,
)
from zinc_gnf.dataset import (  # noqa: E402
    DatasetFormatError,
    MolecularGraphDataset,
    load_training_splits,
    seeded_subset_indices,
)
from zinc_gnf.generation import DECODERS, decode_embeddings  # noqa: E402
from zinc_gnf.invertibility import double_precision_inverse  # noqa: E402
from zinc_gnf.metrics_generation import generation_metrics  # noqa: E402
from zinc_gnf.molecular_profile import profile_molecules  # noqa: E402
from zinc_gnf.qed_evaluation import qed_target_metrics  # noqa: E402
from zinc_gnf.qed_guidance import (  # noqa: E402
    METHOD_LABELS,
    equal_budget_note,
    fit_latent_predictor,
    guide_latents,
    select_best,
)
from zinc_gnf.trainer_autoencoder import ResumeMismatchError, load_autoencoder, resolve_device  # noqa: E402
from zinc_gnf.trainer_flow import STATISTICS_SEED, file_sha256, load_flow  # noqa: E402
from zinc_gnf.utils.logging import JobLogger  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="data/processed/zinc250k_charge_aware_moflow_fixed")
    parser.add_argument("--ae-checkpoint", default=None)
    parser.add_argument("--target-n", default="15,23,30")
    parser.add_argument("--target-qed", default="0.4,0.6,0.8")
    parser.add_argument("--count", type=int, default=200, help="kept molecules per cell")
    parser.add_argument("--rerank-group", type=int, default=5,
                        help="candidates per kept molecule for the reranking arm")
    parser.add_argument("--guidance-steps", type=int, default=10)
    parser.add_argument("--guidance-step-size", type=float, default=0.1)
    parser.add_argument("--predictor-molecules", type=int, default=20000)
    parser.add_argument("--predictor-epochs", type=int, default=8)
    parser.add_argument("--predictor-mae-gate", type=float, default=0.08,
                        help="skip the guided arm if the held-out MAE exceeds this")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headline", default="degree", choices=DECODERS)
    parser.add_argument("--qed-tolerance", type=float, default=0.1)
    parser.add_argument("--double-inverse", action="store_true",
                        help="run the flow inverse in float64 (no retraining, no weight change)")
    parser.add_argument("--no-diversity", action="store_true")
    parser.add_argument("--allow-ae-mismatch", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


def decode_from_latents(flow, autoencoder, z, mask, condition, counts, *, decoders,
                        double: bool) -> list[dict[str, Any]]:
    """Latents -> embeddings -> molecules, optionally inverting in float64."""
    with torch.no_grad():
        if double:
            with double_precision_inverse(flow) as doubled:
                embeddings = doubled.inverse(
                    z.double(), mask, None if condition is None else condition.double())
            embeddings = embeddings.float()
        else:
            embeddings = flow.inverse(z, mask, condition)
    return decode_embeddings(autoencoder, embeddings, counts, decoders=decoders)


def summarise(records, *, target_qed, decoder, train_smiles, validation_smiles, decoders,
              diversity, tolerance) -> dict[str, Any]:
    """Every metric the stage must report, for one arm of one cell."""
    metrics = generation_metrics(records, train_smiles=train_smiles,
                                 validation_smiles=validation_smiles, decoders=decoders,
                                 headline=decoder, diversity=diversity)
    head = metrics["decoders"].get(decoder, {})
    raw = metrics["decoders"].get("raw", {})
    smiles = [r[decoder].get("smiles") if (r.get(decoder) or {}).get("valid") else None
              for r in records]
    profile = profile_molecules(smiles, name=decoder, scaffolds=False)
    achievement = qed_target_metrics(records, requested_qed=target_qed, decoder=decoder,
                                     tolerance=tolerance)
    return {
        "kept_molecules": len(records),
        "qed_mae": achievement["mae"],
        "qed_bias": achievement["bias"],
        "qed_hit_rate": achievement["hit_rate"],
        "realised_qed_mean": achievement["realised_qed_mean"],
        "valid_molecules": achievement["valid_molecules"],
        "raw_validity": (raw.get("strict_validity") or {}).get("value"),
        "corrected_validity": (head.get("strict_validity") or {}).get("value"),
        "fallback_rate": (head.get("fallback_rate") or {}).get("value"),
        "aromatic_rings_mean": profile["aromatic_rings"]["mean"],
        "aromatic_ring_fraction": profile["aromatic_ring_fraction"],
        "uniqueness": (head.get("uniqueness") or {}).get("value"),
        "novelty_vs_train": (head.get("novelty_vs_train") or {}).get("value"),
        "internal_diversity": (head.get("diversity") or {}).get("internal_diversity"),
    }


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - one experiment, read linearly
    arguments = parse_args(argv)
    output_root = Path(arguments.output_dir)
    try:
        targets_n = [int(t) for t in str(arguments.target_n).split(",") if t.strip()]
        targets_qed = [float(t) for t in str(arguments.target_qed).split(",") if t.strip()]
        if not targets_n or not targets_qed:
            raise ValueError("--target-n and --target-qed must be non-empty")
        if arguments.count < 1 or arguments.rerank_group < 1:
            raise ValueError("--count and --rerank-group must be >= 1")
        if arguments.temperature <= 0:
            raise ValueError("--temperature must be > 0")
        output_root.mkdir(parents=True, exist_ok=True)
        logger = JobLogger("guided_qed", output_root, filename="guided_qed.log")
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4

    started = time.perf_counter()
    try:
        device = resolve_device(arguments.device)
        flow, probe = load_flow(arguments.flow_checkpoint, device=device)
        features = flow.config.condition_features
        if set(features) != {"n", "qed"}:
            raise ValueError(f"this experiment needs a joint (n, qed) flow; got {list(features)}")
        recorded = probe.get("autoencoder") or {}
        ae_path = Path(arguments.ae_checkpoint or recorded.get("checkpoint", ""))
        if not ae_path.exists():
            raise FileNotFoundError(f"autoencoder checkpoint not found: {ae_path}")
        digest = file_sha256(ae_path)
        if recorded.get("sha256") and digest != recorded["sha256"] \
                and not arguments.allow_ae_mismatch:
            raise ResumeMismatchError("autoencoder mismatch; pass --allow-ae-mismatch to override")
        autoencoder, _ = load_autoencoder(ae_path, device=device)
        autoencoder.eval()
        for parameter in autoencoder.parameters():
            parameter.requires_grad_(False)
        decoders = list(DECODERS) if autoencoder.config.degree_head else ["raw", "constrained"]
        if arguments.headline not in decoders:
            raise ValueError(f"--headline {arguments.headline} is unavailable")

        scaler = load_condition_scaler(arguments.data_dir)
        train_records, val_records, fingerprint = load_training_splits(arguments.data_dir)
        train_smiles = {str(r["smiles"]) for r in train_records}
        validation_smiles = {str(r["smiles"]) for r in val_records}
        train_set = MolecularGraphDataset(train_records)
        del train_records, val_records

        logger.header("Native vs reranked vs latent-guided QED, equal decoded budget")
        logger.info(f"flow={arguments.flow_checkpoint} step={probe.get('step')} "
                    f"T={arguments.temperature} double_inverse={arguments.double_inverse}")
        for name, label in METHOD_LABELS.items():
            logger.info(f"arm '{name}': {label}")

        # ---- the guide, fitted on TRAIN only ---------------------------------
        predictor_indices = seeded_subset_indices(len(train_set), arguments.predictor_molecules,
                                                  STATISTICS_SEED)
        predictor, predictor_report = fit_latent_predictor(
            flow, autoencoder, train_set, indices=predictor_indices, device=device,
            batch_size=arguments.batch_size, epochs=arguments.predictor_epochs)
        held_out = predictor_report["held_out"]
        logger.info(f"latent->QED predictor: fitted on {predictor_report['molecules_fitted']:,} "
                    f"TRAIN molecules; held-out MAE {held_out['mae']:.4f}, correlation "
                    f"{held_out['correlation']:.4f} ({held_out['molecules']:,} molecules)")
        guidance_enabled = bool(np.isfinite(held_out["mae"])
                                and held_out["mae"] <= arguments.predictor_mae_gate)
        if not guidance_enabled:
            logger.warn(f"held-out MAE {held_out['mae']:.4f} exceeds the gate "
                        f"{arguments.predictor_mae_gate}: the guided arm is SKIPPED rather "
                        f"than reported, because guidance from a weak guide is just noise "
                        f"with a direction")

        budget = equal_budget_note(arguments.count, group=arguments.rerank_group,
                                   steps=arguments.guidance_steps)
        logger.info(f"budget per cell: {budget['rerank_candidates']} decoded candidates for "
                    f"every arm ({arguments.count} kept x {arguments.rerank_group}); "
                    f"native and guided are also run at the per-slot budget for reference")

        cells: list[dict[str, Any]] = []
        for target_n in targets_n:
            for target_qed in targets_qed:
                key = f"N{target_n}_QED{target_qed:g}"
                candidates = arguments.count * arguments.rerank_group
                values = standardise_condition(features, scaler=scaler,
                                               n=[target_n] * candidates,
                                               qed=[target_qed] * candidates)
                mask = torch.ones((candidates, target_n), dtype=torch.bool, device=device)
                counts = [target_n] * candidates
                with torch.no_grad():
                    condition = condition_embedding_for(
                        flow, torch.from_numpy(values).to(device), None, candidates, device)
                    generator = torch.Generator(device=device).manual_seed(
                        arguments.seed + target_n * 1000 + int(target_qed * 100))
                    z = flow.prior.sample(mask, condition, temperature=arguments.temperature,
                                          generator=generator)
                native_all = decode_from_latents(flow, autoencoder, z, mask, condition, counts,
                                                 decoders=decoders,
                                                 double=arguments.double_inverse)

                arms: dict[str, Any] = {}
                common = dict(target_qed=target_qed, decoder=arguments.headline,
                              train_smiles=train_smiles, validation_smiles=validation_smiles,
                              decoders=decoders, diversity=not arguments.no_diversity,
                              tolerance=arguments.qed_tolerance)
                # Equal-budget arms: all three read at `candidates` decoded molecules.
                arms["native_equal_budget"] = summarise(native_all, **common)
                arms["rerank_equal_budget"] = summarise(
                    select_best(native_all, target=target_qed, group=arguments.rerank_group,
                                decoder=arguments.headline), **common)
                # Per-slot reference: what one native candidate per slot looks like.
                arms["native_per_slot"] = summarise(native_all[:arguments.count], **common)

                if guidance_enabled:
                    guided_z = guide_latents(predictor, z, mask, target=target_qed,
                                             steps=arguments.guidance_steps,
                                             step_size=arguments.guidance_step_size)
                    guided_records = decode_from_latents(
                        flow, autoencoder, guided_z["z"], mask, condition, counts,
                        decoders=decoders, double=arguments.double_inverse)
                    arms["guided_equal_budget"] = summarise(guided_records, **common)
                    arms["guided_equal_budget"]["predicted_qed_mean"] = float(
                        guided_z["predicted_qed"].mean())
                    arms["guided_then_rerank_equal_budget"] = summarise(
                        select_best(guided_records, target=target_qed,
                                    group=arguments.rerank_group,
                                    decoder=arguments.headline), **common)

                cells.append({"key": key, "n": target_n, "requested_qed": target_qed,
                              "decoded_candidates": candidates, "arms": arms})
                for name, row in arms.items():
                    logger.info(
                        f"{key} [{name}]: QED {row['realised_qed_mean']:.4f} "
                        f"(MAE {row['qed_mae']:.4f}, hit {row['qed_hit_rate'] * 100:.1f}%), "
                        f"raw validity {row['raw_validity']}, corrected "
                        f"{row['corrected_validity']}, fallback {row['fallback_rate']}, "
                        f"aromatic rings {row['aromatic_rings_mean']:.3f}, uniqueness "
                        f"{row['uniqueness']}, novelty {row['novelty_vs_train']}")

        # ---- aggregate -------------------------------------------------------
        def aggregate(arm: str) -> dict[str, Any] | None:
            rows = [c["arms"][arm] for c in cells if arm in c["arms"]]
            if not rows:
                return None
            def mean(key: str) -> float:
                values = [r[key] for r in rows
                          if r[key] is not None and np.isfinite(float(r[key]))]
                return float(np.mean(values)) if values else float("nan")
            return {"cells": len(rows), **{key: mean(key) for key in
                    ("qed_mae", "qed_bias", "qed_hit_rate", "realised_qed_mean",
                     "raw_validity", "corrected_validity", "fallback_rate",
                     "aromatic_rings_mean", "uniqueness", "novelty_vs_train",
                     "internal_diversity")},
                    "label": METHOD_LABELS.get(arm.split("_")[0], arm)}

        arm_names = ["native_per_slot", "native_equal_budget", "rerank_equal_budget"]
        if guidance_enabled:
            arm_names += ["guided_equal_budget", "guided_then_rerank_equal_budget"]
        summary = {name: aggregate(name) for name in arm_names}

        lines = ["Native vs reranked vs latent-guided QED", "", 
                 f"temperature {arguments.temperature}, {arguments.count} kept molecules per "
                 f"cell, {arguments.rerank_group} candidates per kept molecule, "
                 f"float64 inverse: {arguments.double_inverse}", "",
                 "ARM LABELS (these are not interchangeable):"]
        for name in arm_names:
            kind = ("native" if name.startswith("native")
                    else "guided" if name.startswith("guided") else "rerank")
            lines.append(f"  {name:<32} {METHOD_LABELS[kind]}")
        lines += ["",
                  f"latent->QED predictor held-out MAE {held_out['mae']:.4f}, correlation "
                  f"{held_out['correlation']:.4f}"
                  + ("" if guidance_enabled else "  -> GUIDED ARM SKIPPED (gate "
                     f"{arguments.predictor_mae_gate})"),
                  "",
                  "| arm | QED MAE | hit +-" + f"{arguments.qed_tolerance:g}" +
                  " | realised QED | raw validity | corrected validity | fallback | "
                  "aromatic rings | uniqueness | novelty | diversity |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

        def cell_value(value: Any, template: str = "{:.4f}") -> str:
            return ("N/A" if value is None or not np.isfinite(float(value))
                    else template.format(float(value)))

        for name in arm_names:
            row = summary[name]
            if row is None:
                continue
            lines.append(
                f"| {name} | {cell_value(row['qed_mae'])} | "
                f"{cell_value(row['qed_hit_rate'], '{:.1%}')} | "
                f"{cell_value(row['realised_qed_mean'])} | {cell_value(row['raw_validity'])} | "
                f"{cell_value(row['corrected_validity'])} | {cell_value(row['fallback_rate'])} | "
                f"{cell_value(row['aromatic_rings_mean'], '{:.3f}')} | "
                f"{cell_value(row['uniqueness'])} | {cell_value(row['novelty_vs_train'])} | "
                f"{cell_value(row['internal_diversity'])} |")
        lines += ["",
                  "Only the native rows are evidence about what the GNF learned. The rerank "
                  "rows use RDKit QED to choose among the model's own samples - that is "
                  "post-selection with an oracle, and it cannot be reported as conditioning. "
                  "The guided rows use an external latent->QED predictor to move the latent, "
                  "so they are model-based but still not native conditioning.",
                  "",
                  f"Budget: {budget['note']}"]
        (output_root / "guided_qed.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        atomic_write_json({
            "flow": {"checkpoint": str(arguments.flow_checkpoint), "step": probe.get("step")},
            "autoencoder": {"checkpoint": str(ae_path), "sha256": digest},
            "data_fingerprint": fingerprint,
            "settings": {"targets_n": targets_n, "targets_qed": targets_qed,
                         "count": arguments.count, "rerank_group": arguments.rerank_group,
                         "temperature": arguments.temperature,
                         "guidance_steps": arguments.guidance_steps,
                         "guidance_step_size": arguments.guidance_step_size,
                         "double_inverse": bool(arguments.double_inverse),
                         "qed_tolerance": arguments.qed_tolerance,
                         "headline_decoder": arguments.headline},
            "method_labels": METHOD_LABELS,
            "predictor": predictor_report,
            "guidance_enabled": guidance_enabled,
            "budget": budget,
            "cells": cells,
            "summary": summary,
            "seconds": time.perf_counter() - started,
        }, output_root / "guided_qed.json")
        print("\n".join(lines), flush=True)
        logger.info(f"wrote {output_root}/guided_qed.{{json,txt}}")
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
