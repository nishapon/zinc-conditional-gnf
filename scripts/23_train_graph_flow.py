#!/usr/bin/env python3
"""Stage 22a — train the direct conditional graph flow. Bounded, resumable, two arms.

Stages 17-21 all failed the same way: the conditional signal is present in embedding
space (stage 18: delivered/required 0.665, cosine +0.926), the decoder preserves QED
differences on real held-out embeddings (stage 20: 97.8-99.1%), and realised QED still
does not follow the request. Stage 20 located the gap - a real-versus-generated embedding
classifier reaches AUC 0.979 - and stage 21 showed that removing the decoder's assembly
discontinuity makes conditioning WORSE, because pinning the bond set to the degree
sequence everywhere removes the channel the signal travelled through.

This stage removes the decoder entirely: atoms and bonds are the flow's own variables.

WHAT THIS SCRIPT DOES NOT ASSUME. No published molecular flow established native
target-property conditional generation - MoFlow optimises QED by latent gradient ascent
with a separately trained regressor, GraphAF and GraphDF use RL - so nothing licenses the
assumption that attaching a QED embedding produces control. The design is therefore built
to be falsified:

  * every evaluation measures the CONDITIONAL LIKELIHOOD GAP, scoring the same validation
    molecules under their true QED and under QED permuted WITHIN N. Zero means the
    density ignores QED. Permuting within N is what makes it a test of QED and not of N.
  * `--shuffle-qed` trains the SAME architecture, budget and seed on QED labels permuted
    within N. That arm is the null for everything the real arm reports.
  * the gap's trend over the run is the pre-registered undertraining discriminator: still
    rising means undertrained and nothing may be concluded; plateaued and not positive
    beyond twice its standard error means the architecture is not using the condition.

TRAIN split for fitting, VALIDATION for model selection and diagnostics. The test split
cannot be returned by `load_training_splits` and is never loaded.

Exit codes: 0 ok; 2 missing input; 4 bad arguments; 1 unexpected.
"""

from __future__ import annotations

import argparse
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

from zinc_gnf.conditioning import load_condition_scaler  # noqa: E402
from zinc_gnf.dataset import MolecularGraphDataset, load_training_splits  # noqa: E402
from zinc_gnf.graph_flow import (  # noqa: E402
    BOND_CONDITIONERS,
    ConditionalGraphFlow,
    GraphFlowConfig,
)
from zinc_gnf.trainer_graph_flow import (  # noqa: E402
    GraphFlowTrainConfig,
    GraphFlowTrainer,
    shuffle_qed_within_n,
)
from zinc_gnf.utils.logging import JobLogger  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402
from zinc_gnf.utils.seed import set_seed  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="data/processed/zinc250k_charge_aware_moflow_fixed")
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--bond-blocks", type=int, default=6)
    parser.add_argument("--atom-blocks", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--bond-conditioner", choices=BOND_CONDITIONERS,
                        default=GraphFlowConfig().bond_conditioner,
                        help="topology = stage-23 masked 2-D convolution over the "
                             "adjacency grid; pairwise = the stage-22 conditioner, kept "
                             "as a baseline")
    parser.add_argument("--conv-hidden", type=int, default=GraphFlowConfig().conv_hidden)
    parser.add_argument("--conv-layers", type=int, default=GraphFlowConfig().conv_layers)
    parser.add_argument("--conv-kernel", type=int, default=GraphFlowConfig().conv_kernel)
    parser.add_argument("--condition-dim", type=int, default=64)
    parser.add_argument("--condition-dropout", type=float, default=0.1)
    parser.add_argument("--noise-scale", type=float, default=0.6,
                        help="uniform dequantization width; MoFlow used 0.6")
    # Defaults are taken FROM the dataclass so the two can never diverge again. They did:
    # this script hard-coded 2.0 while GraphFlowConfig was changed to 0.75 after 2.0 was
    # measured to destroy the inverse at depth, and the stage-22a checkpoint was trained
    # at 2.0 as a result.
    parser.add_argument("--max-log-scale", type=float,
                        default=GraphFlowConfig().max_log_scale)
    parser.add_argument("--max-shift", type=float, default=GraphFlowConfig().max_shift)
    parser.add_argument("--no-conditional-prior", action="store_true",
                        help="condition the couplings only, not the prior")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--val-molecules", type=int, default=1024)
    parser.add_argument("--max-nodes", type=int, default=None,
                        help="defaults to the representation's MAX_NODES (38)")
    parser.add_argument("--shuffle-qed", action="store_true",
                        help="THE NULL ARM: permute QED within N in the training data")
    parser.add_argument("--shuffle-seed", type=int, default=99)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    arguments = parse_args(argv)
    output_root = Path(arguments.output_dir)
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        logger = JobLogger("graph_flow", output_root, filename="train.log")
    except OSError as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4

    started = time.perf_counter()
    try:
        if arguments.max_steps < 1 or arguments.batch_size < 1:
            raise ValueError("--max-steps and --batch-size must be positive")
        if not 0.0 <= arguments.condition_dropout < 1.0:
            raise ValueError("--condition-dropout must lie in [0, 1)")
        if arguments.noise_scale <= 0.0:
            raise ValueError("--noise-scale must be positive")
        set_seed(arguments.seed, deterministic=False)
        device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu")
            if arguments.device == "auto" else arguments.device)

        scaler = load_condition_scaler(arguments.data_dir)
        train_records, val_records, fingerprint = load_training_splits(arguments.data_dir)
        observed = max(int(r["n_node"]) for r in train_records + val_records)
        max_nodes = int(arguments.max_nodes or GraphFlowConfig().max_nodes)
        if observed > max_nodes:
            raise ValueError(f"the data contains a molecule with {observed} atoms, above "
                             f"--max-nodes {max_nodes}")

        logger.header("Stage 22a: direct conditional graph flow")
        logger.info(f"device={device} seed={arguments.seed} steps={arguments.max_steps} "
                    f"batch={arguments.batch_size} lr={arguments.learning_rate}")
        logger.info(f"data {arguments.data_dir}: train={len(train_records)} "
                    f"val={len(val_records)} (test split NOT loaded) "
                    f"protocol={fingerprint.get('split_protocol')}")
        logger.info(f"largest molecule {observed} atoms; max_nodes={max_nodes}")

        arm = "shuffled_qed" if arguments.shuffle_qed else "real_conditions"
        if arguments.shuffle_qed:
            train_records = shuffle_qed_within_n(train_records,
                                                 seed=arguments.shuffle_seed,
                                                 scaler=scaler)
            logger.warn("THE NULL ARM: QED labels permuted WITHIN N in the training data. "
                        "Every marginal is identical and the association is destroyed, so "
                        "any QED response this arm shows is an artefact of the "
                        "architecture or the measurement, not learned conditioning.")
        logger.info(f"arm = {arm}")

        config = GraphFlowConfig(
            max_nodes=max_nodes, bond_blocks=arguments.bond_blocks,
            atom_blocks=arguments.atom_blocks, hidden=arguments.hidden,
            condition_dim=arguments.condition_dim,
            condition_dropout=arguments.condition_dropout,
            conditional_prior=not arguments.no_conditional_prior,
            noise_scale=arguments.noise_scale, max_log_scale=arguments.max_log_scale,
            max_shift=arguments.max_shift,
            bond_conditioner=arguments.bond_conditioner,
            conv_hidden=arguments.conv_hidden, conv_layers=arguments.conv_layers,
            conv_kernel=arguments.conv_kernel)
        model = ConditionalGraphFlow(config)
        parameters = sum(p.numel() for p in model.parameters())
        logger.info(f"model parameters: {parameters:,} "
                    f"(bond_blocks={config.bond_blocks} atom_blocks={config.atom_blocks} "
                    f"hidden={config.hidden} conditional_prior={config.conditional_prior} "
                    f"condition_dropout={config.condition_dropout})")
        logger.info(f"bond conditioner: {config.bond_conditioner}"
                    + (f" (masked 2-D conv, {config.conv_layers} layers, "
                       f"{config.conv_kernel}x{config.conv_kernel}, "
                       f"{config.conv_hidden} channels)"
                       if config.bond_conditioner == "topology" else
                       " (stage-22 pairwise aggregates: provably blind to topology at a "
                       "fixed degree sequence)"))
        logger.info(f"numerics: max_log_scale={config.max_log_scale} "
                    f"max_shift={config.max_shift}; inverse condition-number bound "
                    f"exp(bond_blocks * max_log_scale) = "
                    f"{float(np.exp(config.bond_blocks * config.max_log_scale)):.3g}")
        logger.info(f"dequantization U[0,{config.noise_scale}] -> the reported NLL is an "
                    f"UPPER BOUND on the discrete NLL and is NOT comparable with the "
                    f"embedding flow's continuous density")
        logger.info("every evaluation scores the same molecules under QED PERMUTED WITHIN "
                    "N; a condition gap of 0 means the density ignores QED")

        train_config = GraphFlowTrainConfig(
            max_steps=arguments.max_steps, batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate, warmup_steps=arguments.warmup_steps,
            grad_clip=arguments.grad_clip, eval_every=arguments.eval_every,
            save_every=arguments.save_every, log_every=arguments.log_every,
            val_molecules=arguments.val_molecules, seed=arguments.seed,
            shuffle_qed=bool(arguments.shuffle_qed),
            shuffle_seed=arguments.shuffle_seed)
        provenance = {
            "arm": arm, "data_dir": str(arguments.data_dir),
            "data_fingerprint": fingerprint, "stage": 22,
            "condition_features": list(config.condition_features),
            "parameters": int(parameters),
            "representation": "charge_aware_kekule_direct_graph",
            "note": ("atoms and bonds are modelled directly; there is no autoencoder and "
                     "no decoder anywhere in this model"),
        }
        trainer = GraphFlowTrainer(
            model=model, config=train_config,
            train_set=MolecularGraphDataset(train_records),
            val_set=MolecularGraphDataset(val_records),
            output_dir=output_root, logger=logger, device=device,
            provenance=provenance)
        result = trainer.train()

        summary = {
            "stage": 22, "arm": arm, "completed": result.get("completed", False),
            "step": result.get("step"), "failure": result.get("failure"),
            "best_validation_nll_per_dim": trainer.best,
            "condition_gap_trend": result.get("condition_gap_trend"),
            "model_config": config.to_dict(), "train_config": train_config.to_dict(),
            "parameters": int(parameters), "provenance": provenance,
            "history": trainer.history,
            "elapsed_seconds": time.perf_counter() - started,
            "reading": {
                "undertraining_test": ("if condition_gap_trend.status is "
                                       "'rising_undertrained', the budget was too small "
                                       "and NOTHING about the architecture may be "
                                       "concluded"),
                "null_arm": ("compare every number against the --shuffle-qed arm; a real "
                             "arm that does not separate from it has learned nothing "
                             "about QED"),
                "likelihood": ("nll_per_dim is a dequantized-continuous density and "
                               "bound_per_dim is an upper bound on the discrete NLL. "
                               "Neither is comparable with the embedding flow"),
            },
        }
        atomic_write_json(summary, output_root / "training_summary.json")
        trend = result.get("condition_gap_trend") or {}
        logger.info(f"wrote {output_root / 'training_summary.json'}")
        logger.info(f"CONDITION GAP: {trend.get('reading', 'not evaluated')}")
        if not result.get("completed"):
            logger.error(f"run did not complete: {result.get('failure')}")
            return 1
        logger.info(f"elapsed {summary['elapsed_seconds']:.1f}s")
        return 0
    except FileNotFoundError as error:
        logger.error(f"missing input: {error}")
        return 2
    except ValueError as error:
        logger.error(f"bad arguments: {error}")
        return 4
    except Exception:  # noqa: BLE001
        logger.error("unexpected failure:\n" + traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
