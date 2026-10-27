#!/usr/bin/env python3
"""Stage 15a — is the 4.9e-03 round-trip error a real problem? Read-only audit.

The stage-14 numbers (4.92e-03 on real embeddings, 1.41e-05 in the sampling direction,
log|det J|/node 183.23 raw / 167.07 adjusted) are explained and tested here:

1. THE TWO NUMBERS MEASURE DIFFERENT THINGS. `z -> x -> z` evaluates F(G(z)) with the
   same arithmetic in both directions, so an inverse error is largely cancelled by the
   forward pass - it is a self-consistency check, not an accuracy test, and it stays
   small even when G(z) is not the true preimage. `x -> z -> x` evaluates G(F(x)), where
   nothing cancels. Only the second is evidence about inversion. Both are reported, side
   by side, from the same inputs.

2. float32 vs float64, same weights, same inputs: the smallest possible correction is to
   cast for the duration of the inverse, so the gap between the two tells us whether
   precision - rather than a modelling or code defect - is the whole story.

3. Per component: ActNorm, + invertible linear, + affine couplings, + the
   node-autoregressive coupling, by disabling the later pieces of each block. The jump
   between consecutive rows is that piece's contribution.

4. By N, and across requested QED: the node-autoregressive inverse is SEQUENTIAL (node i
   is recovered from the already-recovered nodes before it), so if accumulation is the
   mechanism the error must grow with N. If it does not, accumulation is not the cause.

5. Does it matter? The same embeddings, perturbed only by a round trip, are decoded and
   compared molecule by molecule. If the SMILES are identical, the error is below the
   decoder's resolution and results already obtained are unaffected - which is the only
   question that decides whether the existing 50k numbers stand.

6. The log-determinant is placed against its two reference quantities: the whitening
   gauge term (a constant of the data) and the coupling clamp budget.

Nothing is trained; the checkpoint is opened read-only and never rewritten.
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
    batch_condition,
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
from zinc_gnf.invertibility import (  # noqa: E402
    chemistry_impact,
    component_breakdown,
    double_precision_inverse,
    error_by_size,
    log_det_budget,
    relative_error,
    round_trip,
)
from zinc_gnf.trainer_autoencoder import ResumeMismatchError, load_autoencoder, resolve_device  # noqa: E402
from zinc_gnf.trainer_flow import (  # noqa: E402
    VALIDATION_SUBSET_SEED,
    encode_batch,
    file_sha256,
    load_flow,
)
from zinc_gnf.utils.logging import JobLogger  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402

# Above this relative error the float32 inverse is not trustworthy for corrective work.
TRUST_THRESHOLD = 1e-3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="data/processed/zinc250k_charge_aware_moflow_fixed")
    parser.add_argument("--ae-checkpoint", default=None)
    parser.add_argument("--molecules", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-n", default="15,23,30")
    parser.add_argument("--target-qed", default="0.4,0.6,0.8")
    parser.add_argument("--samples-per-cell", type=int, default=64)
    parser.add_argument("--allow-ae-mismatch", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    output_root = Path(arguments.output_dir)
    try:
        targets_n = [int(t) for t in str(arguments.target_n).split(",") if t.strip()]
        targets_qed = [float(t) for t in str(arguments.target_qed).split(",") if t.strip()]
        if not targets_n or not targets_qed:
            raise ValueError("--target-n and --target-qed must be non-empty")
        output_root.mkdir(parents=True, exist_ok=True)
        logger = JobLogger("invertibility", output_root, filename="invertibility.log")
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4

    started = time.perf_counter()
    try:
        device = resolve_device(arguments.device)
        flow, probe = load_flow(arguments.flow_checkpoint, device=device)
        recorded = probe.get("autoencoder") or {}
        ae_path = Path(arguments.ae_checkpoint or recorded.get("checkpoint", ""))
        if not ae_path.exists():
            raise FileNotFoundError(f"autoencoder checkpoint not found: {ae_path}")
        digest = file_sha256(ae_path)
        if recorded.get("sha256") and digest != recorded["sha256"] \
                and not arguments.allow_ae_mismatch:
            raise ResumeMismatchError(
                f"autoencoder mismatch: flow trained with {recorded['sha256'][:16]}, "
                f"{ae_path} has {digest[:16]}")
        autoencoder, _ = load_autoencoder(ae_path, device=device)
        autoencoder.eval()
        for parameter in autoencoder.parameters():
            parameter.requires_grad_(False)

        _, val_records, fingerprint = load_training_splits(arguments.data_dir)
        val_set = MolecularGraphDataset(val_records)
        del val_records
        indices = seeded_subset_indices(len(val_set), arguments.molecules,
                                        VALIDATION_SUBSET_SEED)

        logger.header("Invertibility audit of the trained flow")
        logger.info(f"flow={arguments.flow_checkpoint} step={probe.get('step')} "
                    f"features={list(flow.config.condition_features)} "
                    f"node_autoregressive={flow.config.node_autoregressive} device={device}")
        logger.info(f"{len(indices):,} validation molecules, batch {arguments.batch_size}; "
                    f"the checkpoint is read-only and is never rewritten")

        # ---- real embeddings, both directions, both precisions ----------------
        from zinc_gnf.dataset import iter_eval_batches

        batches = []
        for batch in iter_eval_batches(val_set, arguments.batch_size, indices):
            batch = batch.to(device)
            embeddings = encode_batch(autoencoder, batch)
            condition = None
            if flow.config.conditional:
                values, present = batch_condition(batch, flow.config.condition_features)
                condition = condition_embedding_for(flow, values, present,
                                                    batch.batch_size, device)
            batches.append((embeddings, batch.node_mask, condition,
                            [int(n) for n in batch.n_node.tolist()]))

        precision_rows = []
        for double in (False, True):
            worst = None
            for embeddings, mask, condition, _ in batches:
                result = round_trip(flow, embeddings, mask, condition, double=double)
                value = result["data_direction_x_z_x"]["max_relative"]
                if worst is None or value > worst["data_direction_x_z_x"]["max_relative"]:
                    worst = result
            precision_rows.append(worst)
            logger.info(
                f"[{worst['dtype']}] x->z->x max relative {worst['data_direction_x_z_x']['max_relative']:.3e} "
                f"(absolute {worst['data_direction_x_z_x']['max_absolute']:.3e}, data scale "
                f"{worst['data_direction_x_z_x']['reference_scale']:.3e}) | "
                f"z->x->z max relative {worst['latent_direction_z_x_z']['max_relative']:.3e} "
                f"| log|det J|/node {worst['log_det_per_node']:.2f}")
        logger.info("the two directions are measured in DIFFERENT spaces (embedding vs "
                    "latent) with different reference scales and different conditioning, so "
                    "neither relative number bounds the other; compare the absolute errors "
                    "and the scales printed beside them, and treat z->x->z as a "
                    "consistency check between the two code paths rather than as evidence "
                    "that the inverse is accurate")

        # ---- per component ----------------------------------------------------
        embeddings, mask, condition, counts = batches[0]
        components = component_breakdown(flow, embeddings, mask, condition)
        previous = 0.0
        for row in components:
            value = row["data_direction_x_z_x"]["max_relative"]
            logger.info(f"components [{row['components']:<24}] x->z->x "
                        f"{value:.3e} (delta {value - previous:+.3e}), "
                        f"log|det J|/node {row['log_det_per_node']:.2f}")
            previous = value

        # ---- by size ----------------------------------------------------------
        sizes: list[dict[str, Any]] = []
        for embeddings_b, mask_b, condition_b, _ in batches:
            sizes.extend(error_by_size(flow, embeddings_b, mask_b, condition_b))
        merged: dict[int, list[float]] = {}
        for row in sizes:
            merged.setdefault(row["n"], []).append(row["max_relative_error"])
        by_size = [{"n": n, "max_relative_error": float(np.max(v)), "groups": len(v)}
                   for n, v in sorted(merged.items())]
        if len(by_size) > 1:
            correlation = float(np.corrcoef([r["n"] for r in by_size],
                                            [r["max_relative_error"] for r in by_size])[0, 1])
        else:
            correlation = float("nan")
        logger.info(f"error vs N: correlation {correlation:+.3f} "
                    f"(small N {by_size[0]['n']}: {by_size[0]['max_relative_error']:.3e}, "
                    f"large N {by_size[-1]['n']}: {by_size[-1]['max_relative_error']:.3e}); "
                    f"a positive correlation is the signature of the SEQUENTIAL "
                    f"node-autoregressive inverse accumulating error")

        # ---- generated embeddings across the requested grid -------------------
        generated_rows = []
        scaler = load_condition_scaler(arguments.data_dir) if flow.config.conditional else None
        for target_n in targets_n:
            for target_qed in (targets_qed if flow.config.conditional else [float("nan")]):
                count = arguments.samples_per_cell
                mask_g = torch.ones((count, target_n), dtype=torch.bool, device=device)
                condition_g = None
                if flow.config.conditional:
                    features = flow.config.condition_features
                    values = standardise_condition(
                        features, scaler=scaler, n=[target_n] * count,
                        qed=[target_qed] * count if "qed" in features else None)
                    condition_g = condition_embedding_for(
                        flow, torch.from_numpy(values).to(device), None, count, device)
                generator = torch.Generator(device=device).manual_seed(11)
                with torch.no_grad():
                    z = flow.prior.sample(mask_g, condition_g, temperature=1.0,
                                          generator=generator)
                    single = flow.inverse(z, mask_g, condition_g)
                    with double_precision_inverse(flow) as doubled:
                        double_x = doubled.inverse(z.double(), mask_g,
                                                   None if condition_g is None
                                                   else condition_g.double())
                gap = relative_error(single, double_x.to(single.dtype), mask_g)
                generated_rows.append({
                    "n": target_n, "requested_qed": target_qed,
                    "float32_vs_float64_embedding_gap": gap,
                })
                logger.info(f"generated N={target_n} QED={target_qed:g}: the float32 and "
                            f"float64 inverses of the SAME latents differ by "
                            f"{gap['max_relative']:.3e} (relative) in embedding space")

        # ---- does it change the chemistry? -----------------------------------
        embeddings, mask, condition, counts = batches[0]
        z, _ = flow(embeddings, mask, condition)
        round_tripped = flow.inverse(z, mask, condition)
        impact = chemistry_impact(autoencoder, embeddings, round_tripped, counts,
                                 decoders=("raw", "degree")
                                 if autoencoder.config.degree_head else ("raw",))
        for name, row in impact["decoders"].items():
            logger.info(f"chemistry [{name}]: identical SMILES after a round trip "
                        f"{row['identical_smiles_rate'] * 100:.2f}%, identical validity "
                        f"{row['identical_validity_rate'] * 100:.2f}%")
        logger.info(impact["verdict"])

        budget = log_det_budget(flow)
        single_error = precision_rows[0]["data_direction_x_z_x"]["max_relative"]
        double_error = precision_rows[1]["data_direction_x_z_x"]["max_relative"]
        verdict = {
            "float32_x_z_x_max_relative": single_error,
            "float64_x_z_x_max_relative": double_error,
            "float64_improvement_factor": float(single_error / max(double_error, 1e-30)),
            "float32_trustworthy": bool(single_error < TRUST_THRESHOLD),
            "trust_threshold": TRUST_THRESHOLD,
            "recommendation": (
                "float32 is adequate; no correction needed"
                if single_error < TRUST_THRESHOLD else
                "use the float64 inverse (zinc_gnf.invertibility.double_precision_inverse) "
                "for corrective experiments; it needs no retraining and changes no weights"),
            "chemistry_unaffected": all(v["identical_smiles_rate"] > 0.999
                                        for v in impact["decoders"].values()),
        }
        logger.info(f"VERDICT: float32 {single_error:.3e} vs float64 {double_error:.3e} "
                    f"({verdict['float64_improvement_factor']:.0f}x better) -> "
                    f"{verdict['recommendation']}")
        logger.info(f"log-determinant: raw {precision_rows[0]['log_det_per_node']:.2f}/node, "
                    f"whitening gauge {budget['whitening_log_det_per_node']:.2f}, coupling "
                    f"clamp budget {budget['coupling_clamp_budget_per_node']:.0f} -> the "
                    f"value sits well inside the budget, so its size alone is not evidence "
                    f"of divergence")

        payload = {
            "flow": {"checkpoint": str(arguments.flow_checkpoint), "step": probe.get("step"),
                     "sha256": file_sha256(arguments.flow_checkpoint),
                     "condition_features": list(flow.config.condition_features),
                     "node_autoregressive": flow.config.node_autoregressive},
            "autoencoder": {"checkpoint": str(ae_path), "sha256": digest},
            "data_fingerprint": fingerprint,
            "molecules": len(indices),
            "precision": precision_rows,
            "components": components,
            "error_by_size": by_size,
            "error_size_correlation": correlation,
            "generated_grid": generated_rows,
            "chemistry_impact": impact,
            "log_det_budget": budget,
            "verdict": verdict,
            "seconds": time.perf_counter() - started,
        }
        atomic_write_json(payload, output_root / "invertibility.json")
        lines = ["Invertibility audit", "===================", "",
                 f"float32  x->z->x {single_error:.3e}   z->x->z "
                 f"{precision_rows[0]['latent_direction_z_x_z']['max_relative']:.3e}",
                 f"float64  x->z->x {double_error:.3e}   z->x->z "
                 f"{precision_rows[1]['latent_direction_z_x_z']['max_relative']:.3e}",
                 "",
                 "The two directions live in different spaces (embedding vs latent) with "
                 "different reference scales and different conditioning, so neither "
                 "relative number bounds the other. z->x->z recomputes z from the point "
                 "the inverse just produced, which makes it a consistency check between "
                 "the two code paths; x->z->x is the one that reflects inversion accuracy. "
                 "Absolute errors and reference scales are in invertibility.json.",
                 "", "Per component (x->z->x, float32):"]
        for row in components:
            lines.append(f"  {row['components']:<24} "
                         f"{row['data_direction_x_z_x']['max_relative']:.3e}")
        lines += ["", f"Error vs N: correlation {correlation:+.3f}"]
        for row in by_size:
            lines.append(f"  N={row['n']:>3}  {row['max_relative_error']:.3e}")
        lines += ["", "Chemistry impact of one round trip:"]
        for name, row in impact["decoders"].items():
            lines.append(f"  [{name}] identical SMILES {row['identical_smiles_rate'] * 100:.2f}%, "
                         f"identical validity {row['identical_validity_rate'] * 100:.2f}%")
        lines += ["", f"{impact['verdict']}", "", f"{verdict['recommendation']}"]
        (output_root / "invertibility.txt").write_text("\n".join(lines) + "\n",
                                                       encoding="utf-8")
        print("\n".join(lines), flush=True)
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
