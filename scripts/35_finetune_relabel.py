#!/usr/bin/env python3
"""Short forward-only fine-tune on relabeled generated pairs, anchored by real molecules.

    L = (1 - w) * NLL(x_real, c_real)  +  w * NLL(x_generated, c_actual_decoded)

Both terms are per-node NLL in nats, the same units as the original objective, so the
loss is comparable with the original run's logs. FORWARD-ONLY: the gradient path is
`flow.forward` plus `prior.log_prob`. `flow.inverse` is never called here at all -- the
generated embeddings were produced once by `34_collect_relabel_pool.py` and are read
from disk as data.

WHAT IS AND IS NOT TOUCHED
--------------------------
The ORIGINAL checkpoint is opened read-only and never written. The fine-tuned weights go
to a NEW --output-dir, and the script REFUSES to start if that directory already holds a
checkpoint (unless --allow-existing is passed explicitly), so no previous result can be
overwritten. The autoencoder is frozen throughout and is used only to encode real
molecules for the anchor term; the encoder is not in any gradient path.

THE BUDGET IS FIXED, NOT SEARCHED
---------------------------------
400 steps, learning rate 1e-5 (one hundredth of the original 1e-3), 50 warm-up steps,
gradient clipping at 5.0, relabel weight 0.5. These are recorded in the checkpoint and
in the report. There is no sweep and no early-stopping on the target metric, because
selecting a checkpoint on the quantity being tested would invalidate the test; `best.pt`
is selected on the REAL validation NLL only, and `last.pt` is always kept.

WHAT THE LABELS MEAN
--------------------
The conditioning target is QED under the DEGREE-CONSTRAINED assembler. The fine-tuned
model therefore learns "produce embeddings whose degree-decoded molecule has QED c".
Whether that is intrinsic conditioning or a correction-dependent artefact is decided by
`36_evaluate_relabel.py`, which scores the response under raw, constrained and degree
separately. This script does not and cannot settle it.

Exit codes: 0 ok; 2 missing input; 3 provenance or overwrite refusal; 4 bad arguments;
1 unexpected.
"""

from __future__ import annotations

import argparse
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
    batch_condition,
    condition_embedding_for,
    load_condition_scaler,
)
from zinc_gnf.dataset import (  # noqa: E402
    MolecularGraphDataset,
    iter_eval_batches,
    load_training_splits,
)
from zinc_gnf.relabel import (  # noqa: E402
    REAL_ANCHOR_NOISE_SEED,
    RELABEL_NOISE_SEED,
    FinetuneConfig,
    RelabelPool,
    assert_disjoint_seeds,
    batch_from_pool,
    holdout_relabel_nll,
    learning_rate_at,
    mixed_loss,
    real_validation_nll,
)
from zinc_gnf.trainer_autoencoder import ResumeMismatchError, load_autoencoder  # noqa: E402
from zinc_gnf.trainer_flow import encode_batch, file_sha256, load_flow  # noqa: E402
from zinc_gnf.utils.checkpointing import save_checkpoint  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402
from zinc_gnf.utils.seed import set_seed  # noqa: E402

CHECKPOINT_NAMES = ("last.pt", "best.pt")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flow-checkpoint", required=True,
                        help="the ORIGINAL checkpoint; opened read-only")
    parser.add_argument("--ae-checkpoint", default=None)
    parser.add_argument("--pool-dir", required=True)
    parser.add_argument("--data-dir",
                        default="data/processed/zinc250k_charge_aware_moflow_fixed")
    parser.add_argument("--output-dir", required=True,
                        help="a NEW directory; refuses to overwrite an existing run")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--relabel-weight", type=float, default=0.5)
    parser.add_argument("--real-batch-size", type=int, default=32)
    parser.add_argument("--pool-batch-size", type=int, default=32)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--val-molecules", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--allow-ae-mismatch", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    return parser.parse_args(argv)


def real_batch_stream(dataset, autoencoder, flow, *, batch_size, device, seed):
    """Infinite shuffled stream of encoded real batches with their own conditions."""
    generator = np.random.default_rng(seed)
    size = len(dataset)
    while True:
        order = generator.permutation(size)
        for start in range(0, size - batch_size + 1, batch_size):
            indices = order[start:start + batch_size].tolist()
            for batch in iter_eval_batches(dataset, batch_size, indices):
                batch = batch.to(device)
                with torch.no_grad():
                    embeddings = encode_batch(autoencoder, batch)
                values, present = batch_condition(batch, flow.config.condition_features)
                condition = condition_embedding_for(
                    flow, values, present, batch.batch_size, device)
                yield {"embeddings": embeddings, "node_mask": batch.node_mask,
                       "condition": condition}


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        assert_disjoint_seeds()
        config = FinetuneConfig(
            max_steps=args.max_steps, learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps, grad_clip_norm=args.grad_clip_norm,
            relabel_weight=args.relabel_weight,
            real_batch_size=args.real_batch_size,
            pool_batch_size=args.pool_batch_size, eval_every=args.eval_every,
            seed=args.seed)
        config.validate()
        output_dir = Path(args.output_dir)
        if Path(args.flow_checkpoint).resolve().parent == output_dir.resolve():
            raise ValueError(
                "--output-dir is the ORIGINAL checkpoint's directory; refusing to risk "
                "overwriting it. Point --output-dir at a new directory.")
        output_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4

    started = time.perf_counter()
    try:
        existing = [name for name in CHECKPOINT_NAMES if (output_dir / name).is_file()]
        if existing and not args.allow_existing:
            raise ResumeMismatchError(
                f"{output_dir} already contains {existing}; refusing to overwrite a "
                f"previous run. Choose a new --output-dir or pass --allow-existing.")

        device = torch.device("cuda" if (args.device == "auto"
                                         and torch.cuda.is_available())
                              else ("cpu" if args.device == "auto" else args.device))
        set_seed(config.seed, deterministic=False)
        scaler = load_condition_scaler(args.data_dir)

        flow, probe = load_flow(args.flow_checkpoint, device=device)
        if not flow.config.conditional:
            raise ValueError("relabeling needs a CONDITIONAL flow")
        recorded = probe.get("autoencoder") or {}
        ae_path = Path(args.ae_checkpoint or recorded.get("checkpoint", ""))
        if not ae_path.is_file():
            raise FileNotFoundError(f"autoencoder not found: {ae_path}")
        digest = file_sha256(ae_path)
        match = bool(recorded.get("sha256") == digest)
        if not match and not args.allow_ae_mismatch:
            raise ResumeMismatchError(
                f"autoencoder SHA256 mismatch: recorded "
                f"{str(recorded.get('sha256'))[:16]}, actual {digest[:16]}")
        autoencoder, _ = load_autoencoder(ae_path, device=device)
        autoencoder.eval().requires_grad_(False)

        pool = RelabelPool.load(args.pool_dir)
        pool_flow = (pool.metadata.get("flow") or {})
        if pool_flow.get("sha256") and pool_flow["sha256"] != file_sha256(
                args.flow_checkpoint):
            raise ResumeMismatchError(
                f"the pool was collected from a different flow checkpoint "
                f"({str(pool_flow.get('sha256'))[:16]}); relabeling a different model's "
                f"samples is not the intended experiment")
        train_indices = pool.indices_for("train")
        if len(train_indices) < config.pool_batch_size:
            raise FileNotFoundError(
                f"only {len(train_indices)} training samples in the pool, need at least "
                f"{config.pool_batch_size}")

        train_records, val_records, fingerprint = load_training_splits(args.data_dir)
        train_set = MolecularGraphDataset(train_records)
        val_set = MolecularGraphDataset(val_records)
        del train_records, val_records

        print(f"device={device}  flow step={probe.get('step')}  "
              f"AE match={match}", flush=True)
        print(f"pool: {len(pool)} samples ({len(train_indices)} train, "
              f"{len(pool.indices_for('holdout'))} holdout), label decoder="
              f"{pool.metadata.get('pool_config', {}).get('label_decoder')}", flush=True)

        flow.requires_grad_(True).train()
        optimizer = torch.optim.Adam(flow.parameters(), lr=config.learning_rate,
                                     weight_decay=config.weight_decay)
        real_stream = real_batch_stream(train_set, autoencoder, flow,
                                        batch_size=config.real_batch_size,
                                        device=device, seed=config.seed)
        pool_rng = np.random.default_rng(config.seed + 1)
        real_noise = torch.Generator(device=device).manual_seed(REAL_ANCHOR_NOISE_SEED)
        relabel_noise = torch.Generator(device=device).manual_seed(RELABEL_NOISE_SEED)

        baseline = real_validation_nll(flow, autoencoder, val_set, scaler=scaler,
                                       device=device, val_molecules=args.val_molecules)
        print(f"  step 0: real val NLL/node={baseline['val_loss']:.4f} "
              f"condition gap={baseline.get('val_condition_gap')}", flush=True)

        history: list[dict[str, Any]] = []
        best = {"val_loss": baseline["val_loss"], "step": 0}
        nonfinite_steps = 0
        for step in range(config.max_steps):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_at(config, step)
            chunk = pool_rng.choice(train_indices, size=config.pool_batch_size,
                                    replace=False).tolist()
            pool_batch = batch_from_pool(pool, chunk, device=device)
            loss, diagnostics = mixed_loss(
                flow, real_batch=next(real_stream), pool_batch=pool_batch,
                scaler=scaler, relabel_weight=config.relabel_weight,
                real_generator=real_noise, relabel_generator=relabel_noise,
                device=device)
            if not torch.isfinite(loss):
                nonfinite_steps += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                flow.parameters(), config.grad_clip_norm))
            if not np.isfinite(grad_norm):
                nonfinite_steps += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()

            if (step + 1) % config.eval_every == 0 or step + 1 == config.max_steps:
                validation = real_validation_nll(
                    flow, autoencoder, val_set, scaler=scaler, device=device,
                    val_molecules=args.val_molecules)
                holdout = holdout_relabel_nll(flow, pool, scaler=scaler, device=device)
                record = {"step": step + 1, "learning_rate": learning_rate_at(config, step),
                          "grad_norm": grad_norm, **diagnostics, **validation,
                          "holdout_relabel_nll_per_node": holdout.get("nll_per_node")}
                history.append(record)
                print(f"  step {step + 1}: loss={diagnostics['loss']:.4f} "
                      f"real={diagnostics.get('real_nll_per_node')} "
                      f"relabel={diagnostics.get('relabel_nll_per_node')} "
                      f"val NLL={validation['val_loss']:.4f} "
                      f"gap={validation.get('val_condition_gap')} "
                      f"holdout={holdout.get('nll_per_node')}", flush=True)
                # `best` tracks the REAL validation NLL only. Selecting on the QED
                # response would be selecting on the quantity under test.
                if validation["val_loss"] < best["val_loss"]:
                    best = {"val_loss": validation["val_loss"], "step": step + 1}
                    write_checkpoint(output_dir / "best.pt", flow, optimizer, config,
                                     probe, step + 1, validation, pool, args)

        final_validation = real_validation_nll(
            flow, autoencoder, val_set, scaler=scaler, device=device,
            val_molecules=args.val_molecules)
        write_checkpoint(output_dir / "last.pt", flow, optimizer, config, probe,
                         config.max_steps, final_validation, pool, args)
        report = {
            "framing": ("Forward-only fine-tune on generated embeddings relabelled with "
                        "their ACTUAL decoded QED under the degree-constrained "
                        "assembler, anchored by real-molecule NLL. The labels are "
                        "DECODER-INDEXED; whether this is intrinsic conditioning is "
                        "decided by scripts/36_evaluate_relabel.py, not here."),
            "original_flow": {"checkpoint": str(args.flow_checkpoint),
                              "step": probe.get("step"),
                              "sha256": file_sha256(args.flow_checkpoint)},
            "autoencoder": {"checkpoint": str(ae_path), "sha256_actual": digest,
                            "sha256_match": match},
            "pool": {"directory": str(args.pool_dir), "samples": len(pool),
                     "train": len(train_indices),
                     "holdout": len(pool.indices_for("holdout")),
                     "qed_support": pool.qed_support(),
                     "label_decoder": pool.metadata.get("pool_config", {}).get(
                         "label_decoder")},
            "config": config.to_dict(),
            "data_fingerprint": fingerprint,
            "baseline_real_validation": baseline,
            "final_real_validation": final_validation,
            "real_validation_nll_change": final_validation["val_loss"] - baseline["val_loss"],
            "best_on_real_validation": best,
            "nonfinite_steps": nonfinite_steps,
            "history": history,
            "elapsed_seconds": time.perf_counter() - started,
            "checkpoint_selection_note": (
                "best.pt is selected on REAL validation NLL only. last.pt is the end of "
                "the fixed budget. Neither is selected on the QED response, because "
                "that is the quantity under test."),
        }
        atomic_write_json(report, output_dir / "finetune_report.json")
        (output_dir / "finetune_report.txt").write_text(format_finetune(report))
        print(f"\nreal validation NLL/node {baseline['val_loss']:.4f} -> "
              f"{final_validation['val_loss']:.4f} "
              f"({report['real_validation_nll_change']:+.4f})")
        print(f"non-finite steps: {nonfinite_steps}/{config.max_steps}")
        print(f"wrote {output_dir / 'last.pt'} and {output_dir / 'finetune_report.txt'}")
        return 0
    except FileNotFoundError as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 2
    except ResumeMismatchError as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 3
    except Exception:  # noqa: BLE001
        print("FATAL: unexpected\n" + traceback.format_exc(), file=sys.stderr, flush=True)
        return 1


def write_checkpoint(path, flow, optimizer, config, probe, step, validation, pool, args):
    """A checkpoint `load_flow` can reopen, carrying its relabeling provenance."""
    save_checkpoint(
        path, flow, optimizer, epoch=0, best_metric=validation["val_loss"],
        configuration={"flow": flow.config.to_dict(),
                       "finetune": config.to_dict()},
        extra={
            "step": step,
            "flow_config": flow.config.to_dict(),
            "vocabulary_fingerprint": probe.get("vocabulary_fingerprint"),
            "embedding_statistics": probe.get("embedding_statistics"),
            "autoencoder": probe.get("autoencoder"),
            "data_fingerprint": probe.get("data_fingerprint"),
            "relabel_finetune": {
                "parent_checkpoint": str(args.flow_checkpoint),
                "parent_step": probe.get("step"),
                "pool_directory": str(args.pool_dir),
                "pool_samples": len(pool),
                "label_decoder": pool.metadata.get("pool_config", {}).get(
                    "label_decoder"),
                "relabel_weight": config.relabel_weight,
                "config": config.to_dict(),
                "real_validation": validation,
            },
        })


def format_finetune(report: dict[str, Any]) -> str:
    config = report["config"]
    lines = ["Relabeling fine-tune", "=" * 78, "", report["framing"], "",
             f"parent: {report['original_flow']['checkpoint']} "
             f"(step {report['original_flow']['step']})",
             f"pool: {report['pool']['samples']} samples "
             f"({report['pool']['train']} train / {report['pool']['holdout']} holdout), "
             f"label decoder {report['pool']['label_decoder']}",
             f"budget: {config['max_steps']} steps, lr {config['learning_rate']}, "
             f"warmup {config['warmup_steps']}, relabel weight "
             f"{config['relabel_weight']}, clip {config['grad_clip_norm']}",
             f"non-finite steps: {report['nonfinite_steps']}", "",
             "-- pool QED support (the ceiling) " + "-" * 44]
    support = report["pool"]["qed_support"]
    lines += [f"  mean={support.get('mean')} IQR={support.get('iqr')} "
              f"p5={support.get('quantiles', {}).get('p5')} "
              f"p95={support.get('quantiles', {}).get('p95')} "
              f">=0.75: {support.get('fraction_at_least_0p75')}", "",
              "-- training trace " + "-" * 60]
    for record in report["history"]:
        lines.append(
            f"  step {record['step']:<5} loss={record.get('loss'):.4f} "
            f"real={record.get('real_nll_per_node')} "
            f"relabel={record.get('relabel_nll_per_node')} "
            f"val={record.get('val_loss'):.4f} "
            f"gap={record.get('val_condition_gap')} "
            f"holdout={record.get('holdout_relabel_nll_per_node')}")
    lines += ["", "-- real validation NLL (ORIGINAL split, ORIGINAL noise seed) " + "-" * 16,
              f"  before {report['baseline_real_validation']['val_loss']:.4f}  "
              f"after {report['final_real_validation']['val_loss']:.4f}  "
              f"change {report['real_validation_nll_change']:+.4f}",
              f"  condition gap before "
              f"{report['baseline_real_validation'].get('val_condition_gap')}  "
              f"after {report['final_real_validation'].get('val_condition_gap')}",
              "", report["checkpoint_selection_note"], "",
              "This file says NOTHING about whether QED control improved. Run",
              "scripts/36_evaluate_relabel.py under common noise to find out, and read",
              "the per-decoder breakdown before believing any gain is intrinsic."]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
