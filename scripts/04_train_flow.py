#!/usr/bin/env python3
"""Stage 4 — train the UNCONDITIONAL node-feature graph normalizing flow.

Trains the flow on the node embeddings of the frozen learned-degree
autoencoder over the TRAIN split of the MoFlow-fixed ZINC250K dataset,
selects on the VALIDATION split, and never loads the test split. Resumes
exactly from `last.pt` (`--resume auto`, the default), so a walltime limit is
not a lost run.

The autoencoder is never modified: it is loaded in eval mode with gradients
disabled, and its file SHA-256 is written into config.json and every
checkpoint. Resuming against a different encoder is refused.

`--node-autoregressive` selects ARM B (one node-autoregressive coupling per block,
targeting inter-node dependence). Without it the run is ARM A, the committed baseline.
The two arms share every other setting, and Arm B's new couplings are identity at
initialisation, so both arms start from the same density.

`--pause-at N` stops cleanly at step N while keeping `--max-steps` as the
target: `summary.json` records `paused: true` with `completed: false`, so the
seed still counts as unfinished and resubmitting resumes at step N+1. Use it to
sample from a mid-run checkpoint before spending the rest of the GPU budget.

Settings come from `configs/zinc250k_flow_uncond.yaml`; command-line flags
override them for smoke tests.

GPU jobs run only through sbatch (slurm/04_flow_*.sh). `--device cuda` fails
loudly if no GPU is visible instead of silently training on CPU.

Exit codes: 0 completed; 130 interrupted (checkpoint saved, resubmit to
resume); 2 data or checkpoint missing; 3 resume mismatch; 4 bad
arguments/config/data; 5 non-finite loss (last finite state saved);
1 unexpected error (traceback in train.log).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from zinc_gnf.dataset import DatasetFormatError  # noqa: E402
from zinc_gnf.flow import FlowConfig  # noqa: E402
from zinc_gnf.trainer_autoencoder import ResumeMismatchError  # noqa: E402
from zinc_gnf.trainer_flow import FlowTrainConfig, FlowTrainer  # noqa: E402
from zinc_gnf.utils.config import load_config  # noqa: E402
from zinc_gnf.utils.logging import JobLogger  # noqa: E402
from zinc_gnf.utils.runtime import StopFlag  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/zinc250k_flow_uncond.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--ae-checkpoint", default=None, help="frozen autoencoder checkpoint")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None, help="0 disables clipping")
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--val-molecules", type=int, default=None)
    parser.add_argument("--statistics-molecules", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None, help="early stopping, in evaluations")
    parser.add_argument("--pause-at", type=int, default=None,
                        help="stop cleanly at this step WITHOUT completing the run, keeping "
                             "--max-steps as the target; resubmit the same command to resume. "
                             "A pause point already passed is ignored.")
    parser.add_argument("--max-train-molecules", type=int, default=None, help="smoke tests only")
    parser.add_argument("--ae-seen-only", action="store_true",
                        help="train only on the molecules the autoencoder itself saw")
    parser.add_argument("--noise-sigma", type=float, default=None,
                        help="smoothing noise in per-dimension std units (default 0.05)")
    parser.add_argument("--node-autoregressive", dest="node_autoregressive",
                        action="store_true", default=None,
                        help="ARM B: add one node-autoregressive coupling per block")
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--max-log-scale", type=float, default=None)
    parser.add_argument("--condition-features", default=None,
                        help="comma-separated subset of 'n,qed' (empty string = unconditional); "
                             "overrides the config's condition_features")
    parser.add_argument("--resume", default="auto",
                        help="'auto' (resume last.pt if present), 'none', or a checkpoint path")
    return parser.parse_args(argv)


def build_configs(arguments: argparse.Namespace) -> tuple[FlowConfig, FlowTrainConfig]:
    configuration = load_config(arguments.config)
    flow_values = dict(configuration.get("flow", {}))
    train_values = dict(configuration.get("training", {}))
    overrides_train = {
        "data_dir": arguments.data_dir, "output_dir": arguments.output_dir,
        "ae_checkpoint": arguments.ae_checkpoint, "seed": arguments.seed,
        "device": arguments.device, "batch_size": arguments.batch_size,
        "max_steps": arguments.max_steps, "learning_rate": arguments.learning_rate,
        "warmup_steps": arguments.warmup_steps, "eval_every": arguments.eval_every,
        "save_every": arguments.save_every, "log_every": arguments.log_every,
        "val_molecules": arguments.val_molecules,
        "statistics_molecules": arguments.statistics_molecules,
        "early_stopping_patience": arguments.patience,
        "pause_at_step": arguments.pause_at,
        "max_train_molecules": arguments.max_train_molecules,
    }
    train_values.update({k: v for k, v in overrides_train.items() if v is not None})
    if arguments.ae_seen_only:
        train_values["ae_seen_only"] = True
    if arguments.grad_clip_norm is not None:
        train_values["grad_clip_norm"] = arguments.grad_clip_norm or None
    for name in ("noise_sigma", "num_blocks", "max_log_scale", "node_autoregressive"):
        value = getattr(arguments, name)
        if value is not None:
            flow_values[name] = value
    if arguments.condition_features is not None:
        # "" means unconditional on purpose, so the flag can also turn conditioning OFF.
        flow_values["condition_features"] = [f.strip() for f in
                                             arguments.condition_features.split(",") if f.strip()]
    return FlowConfig.from_dict(flow_values), FlowTrainConfig.from_dict(train_values)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        flow_config, train_config = build_configs(arguments)
        train_config.validate()
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print(f"FATAL: invalid configuration: {error}", file=sys.stderr, flush=True)
        return 4

    try:
        logger = JobLogger("flow", train_config.output_dir, filename="train.log")
    except OSError as error:
        print(f"FATAL: cannot create {train_config.output_dir}: {error}", file=sys.stderr, flush=True)
        return 4

    try:
        trainer = FlowTrainer(flow_config, train_config, logger)
        with StopFlag(logger) as stop_flag:
            result = trainer.train(resume=arguments.resume, stop_flag=stop_flag)
    except FileNotFoundError as error:
        logger.error(f"FATAL: {error}")
        return 2
    except ResumeMismatchError as error:
        logger.error(f"FATAL: {error}")
        return 3
    except FloatingPointError as error:
        logger.error(f"FATAL: {error}")
        return 5
    except (DatasetFormatError, ValueError, RuntimeError) as error:
        logger.error(f"FATAL: {type(error).__name__}: {error}")
        return 4
    except Exception:  # noqa: BLE001
        logger.error("FATAL: unexpected error\n" + traceback.format_exc())
        return 1
    # A pause is a planned, successful stop, so it must exit 0 and not look like a failure
    # to the cluster script; the run is still recorded as unfinished in summary.json.
    if result["completed"] or result["early_stopped"] or result.get("paused"):
        return 0
    return 130


if __name__ == "__main__":
    sys.exit(main())
