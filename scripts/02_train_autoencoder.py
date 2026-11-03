#!/usr/bin/env python3
"""Stage 2 — train the charge-aware bond-aware graph autoencoder.

Trains on the TRAIN split of the MoFlow-fixed ZINC250K dataset, selects the
best checkpoint on the VALIDATION split, and never loads the test split.
Resumes exactly from `last.pt` (`--resume auto`, the default).

Settings come from `configs/zinc250k_charge_aware_ae.yaml` (teammates' v5
values) and can be overridden on the command line for smoke tests.

GPU jobs: run only through sbatch (slurm/02_ae_smoke_gpu.sh). `--device cuda`
fails loudly if no GPU is visible instead of silently training on CPU.

Exit codes: 0 completed; 130 interrupted (checkpoint saved, resume with the
same command); 2 data missing; 3 resume mismatch; 4 bad arguments/config or
dataset format; 1 unexpected error (traceback in train.log).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from zinc_gnf.autoencoder import AutoencoderConfig  # noqa: E402
from zinc_gnf.dataset import DatasetFormatError  # noqa: E402
from zinc_gnf.trainer_autoencoder import (  # noqa: E402
    SELECTION_KEYS,
    AutoencoderTrainer,
    ResumeMismatchError,
    TrainConfig,
)
from zinc_gnf.utils.config import load_config  # noqa: E402
from zinc_gnf.utils.logging import JobLogger  # noqa: E402
from zinc_gnf.utils.runtime import StopFlag  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/zinc250k_charge_aware_ae.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None,
                        help="global gradient-norm clip; 0 disables clipping")
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--val-loss-molecules", type=int, default=None)
    parser.add_argument("--val-recon-molecules", type=int, default=None)
    parser.add_argument("--selection-key", choices=SELECTION_KEYS, default=None)
    parser.add_argument("--patience", type=int, default=None, help="early stopping, in evaluations")
    parser.add_argument("--max-train-molecules", type=int, default=None, help="smoke tests only")
    parser.add_argument("--max-val-molecules", type=int, default=None, help="smoke tests only")
    parser.add_argument("--symmetry-noise-dim", type=int, default=None,
                        help="ablation: extra noise channels beside one-hot identity (default 0)")
    parser.add_argument("--num-processing-steps", type=int, default=None)
    parser.add_argument("--resume", default="auto",
                        help="'auto' (resume last.pt if present), 'none', or a checkpoint path")
    return parser.parse_args(argv)


def build_configs(arguments: argparse.Namespace) -> tuple[AutoencoderConfig, TrainConfig]:
    configuration = load_config(arguments.config)
    model_values = dict(configuration.get("autoencoder", {}))
    train_values = dict(configuration.get("training", {}))
    overrides_train = {
        "data_dir": arguments.data_dir, "output_dir": arguments.output_dir, "seed": arguments.seed,
        "device": arguments.device, "batch_size": arguments.batch_size, "max_steps": arguments.max_steps,
        "learning_rate": arguments.learning_rate, "eval_every": arguments.eval_every,
        "save_every": arguments.save_every, "log_every": arguments.log_every,
        "val_loss_molecules": arguments.val_loss_molecules,
        "val_recon_molecules": arguments.val_recon_molecules,
        "selection_key": arguments.selection_key, "early_stopping_patience": arguments.patience,
        "max_train_molecules": arguments.max_train_molecules,
        "max_val_molecules": arguments.max_val_molecules,
    }
    train_values.update({k: v for k, v in overrides_train.items() if v is not None})
    if arguments.grad_clip_norm is not None:
        train_values["grad_clip_norm"] = arguments.grad_clip_norm or None
    for name in ("symmetry_noise_dim", "num_processing_steps"):
        value = getattr(arguments, name)
        if value is not None:
            model_values[name] = value
    return AutoencoderConfig.from_dict(model_values), TrainConfig.from_dict(train_values)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        model_config, train_config = build_configs(arguments)
        train_config.validate()
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print(f"FATAL: invalid configuration: {error}", file=sys.stderr, flush=True)
        return 4

    try:
        logger = JobLogger("autoencoder", train_config.output_dir, filename="train.log")
    except OSError as error:
        print(f"FATAL: cannot create {train_config.output_dir}: {error}", file=sys.stderr, flush=True)
        return 4

    try:
        trainer = AutoencoderTrainer(model_config, train_config, logger)
        with StopFlag(logger) as stop_flag:
            result = trainer.train(resume=arguments.resume, stop_flag=stop_flag)
    except FileNotFoundError as error:
        logger.error(f"FATAL: {error}")
        return 2
    except ResumeMismatchError as error:
        logger.error(f"FATAL: {error}")
        return 3
    except (DatasetFormatError, ValueError, RuntimeError) as error:
        logger.error(f"FATAL: {type(error).__name__}: {error}")
        return 4
    except Exception:  # noqa: BLE001
        logger.error("FATAL: unexpected error\n" + traceback.format_exc())
        return 1
    return 0 if result["completed"] or result["early_stopped"] else 130


if __name__ == "__main__":
    sys.exit(main())
