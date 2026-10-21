#!/usr/bin/env python3
"""Stage 16 — the decisive bounded experiment: does a decoder-aligned structural term fix
the sampled-embedding distribution?

ONE factor is changed, and the control is built in:

    --lambda-structure 0    continued pure maximum likelihood from the same checkpoint,
                            same seed, same number of steps  (CONTROL)
    --lambda-structure L>0  the same, plus a moment-matching penalty on what the FROZEN
                            decoder says about the flow's OWN samples               (ARM)

Everything else - architecture, frozen autoencoder, data split, smoothing sigma,
whitening, optimiser, schedule, seed, step count, sampling temperature - is identical
between the two runs, so a difference between them is attributable to the structural
term and nothing else.

WHY THIS AND NOT MORE MLE. The likelihood is over 64*N per-node dimensions and is
dominated by per-node marginals; the chemistry is carried by O(N^2) pairwise relations
that barely enter it. The shuffled-node control showed the consequence: real embeddings
regrouped at random reproduce the flow's failure almost exactly, i.e. our samples behave
like a correct marginal model with the joint structure missing. The structural term is
the first thing in this project that puts the decoder's own opinion about SAMPLES into
the objective.

WHAT IS AND IS NOT CLAIMED. The objective stops being pure maximum likelihood; both
terms, and the validation NLL, are logged every evaluation so the likelihood cost is
visible. No RDKit, no validity rule and no QED value enters the loss - only differentiable
class probabilities and training-set moments - so any gain in validity, aromaticity or
QED control is a consequence, not a target. The existing checkpoints are opened
read-only; this writes to its own directory.

Exit codes: 0 ok; 2 missing input; 3 autoencoder/checkpoint mismatch; 4 bad arguments;
5 non-finite loss; 1 unexpected.
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
    DeterministicBatcher,
    MolecularGraphDataset,
    iter_eval_batches,
    load_training_splits,
    make_batch,
    seeded_subset_indices,
)
from zinc_gnf.flow import node_mask_from_counts  # noqa: E402
from zinc_gnf.generation import generation_batch  # noqa: E402
from zinc_gnf.structure_objective import (  # noqa: E402
    DEFAULT_WEIGHTS,
    StructureTargets,
    structure_loss,
    structure_statistics,
)
from zinc_gnf.trainer_autoencoder import ResumeMismatchError, load_autoencoder, resolve_device  # noqa: E402
from zinc_gnf.trainer_flow import (  # noqa: E402
    VALIDATION_NOISE_SEED,
    VALIDATION_SUBSET_SEED,
    encode_batch,
    file_sha256,
    load_flow,
)
from zinc_gnf.utils.checkpointing import save_checkpoint  # noqa: E402
from zinc_gnf.utils.logging import JobLogger  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402
from zinc_gnf.utils.seed import set_seed  # noqa: E402

RESERVED = {"epoch", "best_metric", "configuration", "model_state_dict",
            "optimizer_state_dict", "torch_rng_state", "cuda_rng_state"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flow-checkpoint", required=True, help="opened READ-ONLY")
    parser.add_argument("--output-dir", required=True, help="must not be an existing run")
    parser.add_argument("--data-dir", default="data/processed/zinc250k_charge_aware_moflow_fixed")
    parser.add_argument("--ae-checkpoint", default=None)
    parser.add_argument("--lambda-structure", type=float, required=True,
                        help="0 = the control arm (continued pure MLE)")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64, help="likelihood batch")
    parser.add_argument("--sample-batch", type=int, default=16,
                        help="molecules sampled per step for the structural term")
    parser.add_argument("--sample-temperature", type=float, default=1.0,
                        help="temperature used for the structural term's samples")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="lower than the 1e-3 of the main run: this is a fine-tune")
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--target-molecules", type=int, default=20000,
                        help="TRAINING molecules used to measure the structure targets")
    parser.add_argument("--val-molecules", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="stage A of the warm start: train ONLY the conditional adapters "
                             "(FiLM, the condition embedding and the prior heads) and leave "
                             "the pretrained generative backbone untouched")
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--allow-ae-mismatch", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def sample_conditions(rng: np.random.Generator, counts: np.ndarray, qed: np.ndarray,
                      size: int) -> tuple[list[int], np.ndarray]:
    """Draw (N, QED) pairs from the TRAINING joint distribution, by resampling rows."""
    index = rng.integers(0, len(counts), size=size)
    return [int(counts[i]) for i in index], qed[index]


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - one training loop
    arguments = parse_args(argv)
    output_root = Path(arguments.output_dir)
    try:
        if arguments.lambda_structure < 0:
            raise ValueError("--lambda-structure must be >= 0")
        if arguments.max_steps < 1 or arguments.sample_batch < 1:
            raise ValueError("--max-steps and --sample-batch must be >= 1")
        output_root.mkdir(parents=True, exist_ok=True)
        logger = JobLogger("structure_finetune", output_root, filename="train.log")
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4

    started = time.perf_counter()
    try:
        set_seed(arguments.seed, deterministic=False)
        device = resolve_device(arguments.device)
        source = Path(arguments.flow_checkpoint)
        flow, probe = load_flow(source, device=device)
        features = flow.config.condition_features
        if set(features) != {"n", "qed"}:
            raise ValueError(f"this experiment expects a joint (n, qed) flow; got {list(features)}")
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

        scaler = load_condition_scaler(arguments.data_dir)
        train_records, val_records, fingerprint = load_training_splits(arguments.data_dir)
        targets = StructureTargets.from_records(train_records, limit=arguments.target_molecules)
        train_counts = np.asarray([int(r["n_node"]) for r in train_records], dtype=np.int64)
        train_qed = np.asarray([float(r["qed"]) for r in train_records], dtype=np.float64)
        train_set = MolecularGraphDataset(train_records)
        val_set = MolecularGraphDataset(val_records)
        del train_records, val_records
        val_indices = seeded_subset_indices(len(val_set), arguments.val_molecules,
                                            VALIDATION_SUBSET_SEED)

        arm = "structure" if arguments.lambda_structure > 0 else "control_pure_mle"
        logger.header(f"Structure-aligned fine-tune [{arm}]")
        logger.info(f"source checkpoint {source} (READ-ONLY) step={probe.get('step')} "
                    f"sha256={file_sha256(source)[:16]}")
        logger.info(f"lambda_structure={arguments.lambda_structure} "
                    f"steps={arguments.max_steps} lr={arguments.learning_rate} "
                    f"batch={arguments.batch_size} sample_batch={arguments.sample_batch} "
                    f"sample_temperature={arguments.sample_temperature} seed={arguments.seed}")
        logger.info(f"structure targets measured on {targets.molecules:,} TRAINING molecules: "
                    f"bonds/atom {targets.bonds_per_atom:.4f}, cycle rank "
                    f"{targets.cycle_rank:.4f}, double-bond fraction "
                    f"{targets.double_bond_fraction:.4f}, isolated atoms "
                    f"{targets.isolated_atom_rate:.4f}")
        logger.info("the objective is NLL(data) + lambda * moment-mismatch(samples): no "
                    "longer pure maximum likelihood. Both terms and the validation NLL are "
                    "logged every evaluation, so the likelihood cost is visible.")

        parameter_split = None
        if arguments.freeze_backbone:
            from zinc_gnf.warm_start import freeze_backbone

            parameter_split = freeze_backbone(flow)
            logger.info(f"BACKBONE FROZEN: training {parameter_split['trainable_parameters']:,} "
                        f"adapter parameters "
                        f"({parameter_split['trainable_fraction'] * 100:.2f}% of the model), "
                        f"{parameter_split['frozen_parameters']:,} frozen. The pretrained "
                        f"cross-node structure cannot be overwritten by this run.")
        trainable = [p for p in flow.parameters() if p.requires_grad]
        optimiser = torch.optim.Adam(trainable, lr=arguments.learning_rate)
        batcher = DeterministicBatcher(len(train_set), arguments.batch_size, arguments.seed)
        noise_generator = torch.Generator(device=device).manual_seed(arguments.seed * 7919 + 13)
        sample_generator = torch.Generator(device=device).manual_seed(arguments.seed * 104729 + 7)
        condition_rng = np.random.default_rng(arguments.seed * 7 + 3)
        step = 0
        history: list[dict[str, Any]] = []
        last_path = output_root / "last.pt"
        if arguments.resume == "auto" and last_path.exists():
            state = torch.load(last_path, map_location=device, weights_only=False)
            flow.load_state_dict(state["model_state_dict"])
            if state.get("optimizer_state_dict"):
                optimiser.load_state_dict(state["optimizer_state_dict"])
            step = int(state.get("step", 0))
            batcher.load_state_dict(state["batcher"])
            history = list(state.get("history", []))
            logger.info(f"RESUMED from {last_path} at step {step}")

        def validate() -> dict[str, float]:
            flow.eval()
            generator = torch.Generator(device=device).manual_seed(VALIDATION_NOISE_SEED)
            total, nodes = 0.0, 0
            with torch.no_grad():
                for batch in iter_eval_batches(val_set, max(arguments.batch_size, 64),
                                               val_indices):
                    batch = batch.to(device)
                    embeddings = encode_batch(autoencoder, batch)
                    embeddings = flow.add_smoothing_noise(embeddings, batch.node_mask,
                                                          generator=generator)
                    values, present = batch_condition(batch, features)
                    condition = condition_embedding_for(flow, values, present,
                                                        batch.batch_size, device)
                    parts = flow.log_prob(embeddings, batch.node_mask, condition)
                    total += float(parts["log_prob"].sum())
                    nodes += int(batch.node_mask.sum())
            flow.train()
            return {"val_nll_per_node": -total / max(nodes, 1), "val_nodes": nodes}

        atomic_write_json({
            "arm": arm, "lambda_structure": arguments.lambda_structure,
            "freeze_backbone": bool(arguments.freeze_backbone),
            "parameter_split": parameter_split,
            "warm_start": probe.get("warm_start"),
            "source_checkpoint": str(source), "source_sha256": file_sha256(source),
            "source_step": probe.get("step"),
            "flow_config": flow.config.to_dict(),
            "structure_targets": targets.to_dict(),
            "structure_weights": DEFAULT_WEIGHTS,
            "settings": vars(arguments),
            "autoencoder": {"checkpoint": str(ae_path), "sha256": digest},
            "data_fingerprint": fingerprint,
        }, output_root / "config.json")

        flow.train()
        metrics_path = output_root / "metrics.jsonl"
        window: list[dict[str, float]] = []
        while step < arguments.max_steps:
            batch = make_batch(train_set, batcher.next_indices()).to(device)
            embeddings = encode_batch(autoencoder, batch)
            embeddings = flow.add_smoothing_noise(embeddings, batch.node_mask,
                                                  generator=noise_generator)
            values, present = batch_condition(batch, features)
            condition = condition_embedding_for(flow, values, present, batch.batch_size, device)
            parts = flow.negative_log_likelihood(embeddings, batch.node_mask, condition)
            nll = parts["loss"]
            loss = nll
            report: dict[str, float] = {}
            if arguments.lambda_structure > 0:
                counts, qed_values = sample_conditions(condition_rng, train_counts, train_qed,
                                                       arguments.sample_batch)
                sample_values = standardise_condition(features, scaler=scaler, n=counts,
                                                      qed=qed_values)
                mask = node_mask_from_counts(counts, device=device)
                sample_condition = condition_embedding_for(
                    flow, torch.from_numpy(sample_values).to(device), None,
                    len(counts), device)
                mu, sigma = flow.prior.parameters_for(len(counts), sample_condition, device)
                noise = torch.randn((len(counts), mask.shape[1], flow.config.node_embedding_dim),
                                    device=device, generator=sample_generator)
                # Reparameterised draw: the gradient reaches mu, sigma and every coupling.
                z = (mu.unsqueeze(1)
                     + arguments.sample_temperature * sigma.unsqueeze(1) * noise)
                z = z * mask.unsqueeze(-1).to(z.dtype)
                sampled = flow.inverse(z, mask, sample_condition)
                decode_batch = generation_batch(counts, device=device)
                outputs = autoencoder.decode(sampled, decode_batch, teacher_forcing=False)
                statistics = structure_statistics(outputs["bond_logits"],
                                                  outputs["atom_logits"], mask)
                term, report = structure_loss(statistics, targets)
                loss = nll + arguments.lambda_structure * term

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            if not torch.isfinite(loss):
                save_checkpoint(last_path, flow, optimiser, epoch=batcher.epoch,
                                best_metric=float("inf"),
                                configuration={"flow": flow.config.to_dict()},
                                extra={"step": step, "batcher": batcher.state_dict(),
                                       "history": history})
                logger.error(f"non-finite loss at step {step + 1}; last state saved")
                return 5
            if arguments.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(trainable, arguments.grad_clip_norm)
            optimiser.step()
            step += 1
            window.append({"loss": float(loss.detach()), "nll": float(nll.detach()),
                           **{k: v for k, v in report.items()
                              if k.endswith("_value") or k == "structure_loss"}})

            if step % arguments.log_every == 0 or step == arguments.max_steps:
                mean = {k: float(np.mean([w[k] for w in window if k in w])) for k in window[-1]}
                logger.info(f"step {step}/{arguments.max_steps} loss={mean['loss']:.4f} "
                            f"nll/node={mean['nll']:.4f}"
                            + (f" structure={mean.get('structure_loss', float('nan')):.5f} "
                               f"bonds/atom={mean.get('bonds_per_atom_value', float('nan')):.4f}"
                               f"(target {targets.bonds_per_atom:.4f}) "
                               f"double={mean.get('double_bond_fraction_value', float('nan')):.4f}"
                               f"(target {targets.double_bond_fraction:.4f}) "
                               f"isolated={mean.get('isolated_atom_rate_value', float('nan')):.4f}"
                               if arguments.lambda_structure > 0 else ""))
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"kind": "train", "step": step, **mean}) + "\n")
                window = []

            if step % arguments.eval_every == 0 or step == arguments.max_steps:
                evaluation = validate()
                record = {"kind": "val", "step": step, **evaluation, **report}
                history.append(record)
                logger.info(f"VAL step {step}: nll/node={evaluation['val_nll_per_node']:.4f} "
                            f"(the likelihood cost of the structural term is visible here)")
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")

            if step % arguments.save_every == 0 or step == arguments.max_steps:
                extra = {k: v for k, v in probe.items() if k not in RESERVED}
                extra.update({"step": step, "batcher": batcher.state_dict(),
                              "history": history, "structure_finetune": {
                                  "arm": arm, "lambda_structure": arguments.lambda_structure,
                                  "source_checkpoint": str(source),
                                  "structure_targets": targets.to_dict(),
                                  "sample_temperature": arguments.sample_temperature,
                                  "seed": arguments.seed}})
                save_checkpoint(last_path, flow, optimiser, epoch=batcher.epoch,
                                best_metric=history[-1]["val_nll_per_node"] if history
                                else float("inf"),
                                configuration={"flow": flow.config.to_dict(),
                                               "training": vars(arguments)},
                                extra=extra)

        summary = {
            "arm": arm, "lambda_structure": arguments.lambda_structure,
            "final_step": step, "completed": step >= arguments.max_steps,
            "history": history,
            "structure_targets": targets.to_dict(),
            "source_checkpoint": str(source), "source_step": probe.get("step"),
            "seconds": time.perf_counter() - started,
        }
        atomic_write_json(summary, output_root / "summary.json")
        logger.info(f"finished {step} steps in {summary['seconds']:.0f}s; "
                    f"checkpoint {last_path} (the source checkpoint is unchanged)")
        logger.info("evaluate BOTH arms with scripts/13_generate_nqed.py at the same "
                    "temperature and sample count before drawing any conclusion")
        return 0

    except FileNotFoundError as error:
        print(f"FATAL (missing input): {error}", file=sys.stderr, flush=True)
        return 2
    except DatasetFormatError as error:
        print(f"FATAL (dataset): {error}", file=sys.stderr, flush=True)
        return 2
    except ResumeMismatchError as error:
        print(f"FATAL (mismatch): {error}", file=sys.stderr, flush=True)
        return 3
    except ValueError as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
