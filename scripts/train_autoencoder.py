#!/usr/bin/env python3
"""Train the bond-aware ZINC molecular autoencoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch

from zinc_gnf.data.datasets import make_dataloaders
from zinc_gnf.models.autoencoder import (
    build_autoencoder_from_config,
)
from zinc_gnf.training.autoencoder import (
    evaluate_autoencoder,
    metric_summary,
    train_autoencoder_epoch,
)
from zinc_gnf.utils.checkpointing import (
    load_checkpoint,
    save_checkpoint,
)
from zinc_gnf.utils.config import load_config
from zinc_gnf.utils.seed import set_seed


ARCHITECTURE_KEYS = (
    "node_noise_dim",
    "embedding_dim",
    "hidden_dim",
    "message_passing_steps",
    "attention_heads",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--config",
        default="configs/zinc250k.yaml",
        help="Experiment YAML configuration.",
    )
    parser.add_argument(
        "--splits",
        default=None,
        help=(
            "Preprocessed split pickle. By default, "
            "<processed_dir>/zinc_splits.pkl is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/checkpoints/autoencoder",
        help="Checkpoint and history directory.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint to resume, normally last.pt.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a device such as cuda:0.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override of the configured epoch limit.",
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=5.0,
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(requested)

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is unavailable."
        )

    return device


def architecture_signature(
    configuration: dict[str, Any],
) -> dict[str, Any]:
    autoencoder = configuration["autoencoder"]

    return {
        key: autoencoder[key]
        for key in ARCHITECTURE_KEYS
    }


def verify_resume_configuration(
    checkpoint: dict[str, Any],
    configuration: dict[str, Any],
) -> None:
    saved_signature = checkpoint.get(
        "architecture"
    )

    if saved_signature is None:
        saved_configuration = checkpoint.get(
            "configuration",
            {},
        )
        if "autoencoder" in saved_configuration:
            saved_signature = architecture_signature(
                saved_configuration
            )

    current_signature = architecture_signature(
        configuration
    )

    if (
        saved_signature is not None
        and saved_signature != current_signature
    ):
        raise RuntimeError(
            "Resume checkpoint architecture does not match "
            "the current configuration.\n"
            f"Saved: {saved_signature}\n"
            f"Current: {current_signature}"
        )


def restore_rng_state(
    checkpoint: dict[str, Any],
) -> None:
    if checkpoint.get("torch_rng_state") is not None:
        torch.set_rng_state(
            checkpoint["torch_rng_state"]
        )

    if (
        torch.cuda.is_available()
        and checkpoint.get("cuda_rng_state") is not None
    ):
        torch.cuda.set_rng_state_all(
            checkpoint["cuda_rng_state"]
        )


def write_json_atomic(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            allow_nan=False,
        )

    temporary_path.replace(path)


def main() -> None:
    arguments = parse_args()

    if arguments.num_workers < 0:
        raise ValueError(
            "num-workers cannot be negative."
        )
    if arguments.gradient_clip <= 0:
        raise ValueError(
            "gradient-clip must be positive."
        )

    configuration = load_config(
        arguments.config
    )

    seed = int(configuration["seed"])
    set_seed(seed)

    device = resolve_device(arguments.device)

    autoencoder_configuration = configuration[
        "autoencoder"
    ]

    epochs = (
        int(arguments.epochs)
        if arguments.epochs is not None
        else int(
            autoencoder_configuration["epochs"]
        )
    )

    if epochs < 1:
        raise ValueError(
            "The epoch limit must be positive."
        )

    processed_directory = Path(
        configuration["data"]["processed_dir"]
    )
    split_path = (
        Path(arguments.splits)
        if arguments.splits is not None
        else processed_directory / "zinc_splits.pkl"
    )

    if not split_path.exists():
        raise FileNotFoundError(
            f"Preprocessed split not found: {split_path}\n"
            "Run scripts/preprocess_zinc.py first."
        )

    output_directory = Path(
        arguments.output_dir
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_path = output_directory / "best.pt"
    last_path = output_directory / "last.pt"
    history_path = output_directory / "history.json"
    report_path = output_directory / "report.json"

    batch_size = int(
        autoencoder_configuration["batch_size"]
    )

    loaders = make_dataloaders(
        split_path,
        batch_size=batch_size,
        seed=seed,
        num_workers=arguments.num_workers,
        include_test=False,
        pin_memory=(device.type == "cuda"),
    )

    model = build_autoencoder_from_config(
        autoencoder_configuration
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(
            autoencoder_configuration[
                "learning_rate"
            ]
        ),
        weight_decay=1e-5,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        )
    )

    noise_dim = int(
        autoencoder_configuration["node_noise_dim"]
    )
    bond_class_weight = float(
        autoencoder_configuration[
            "bond_class_weight"
        ]
    )
    patience = int(
        autoencoder_configuration["patience"]
    )

    start_epoch = 1
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    if arguments.resume is not None:
        checkpoint = load_checkpoint(
            arguments.resume,
            model,
            optimizer,
            device=device,
        )

        verify_resume_configuration(
            checkpoint,
            configuration,
        )
        restore_rng_state(checkpoint)

        if checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(
                checkpoint[
                    "scheduler_state_dict"
                ]
            )

        start_epoch = (
            int(checkpoint["epoch"]) + 1
        )
        best_validation_loss = float(
            checkpoint["best_metric"]
        )
        epochs_without_improvement = int(
            checkpoint.get(
                "epochs_without_improvement",
                0,
            )
        )
        history = list(
            checkpoint.get("history", [])
        )

        print(
            f"Resumed from {arguments.resume} "
            f"at epoch {start_epoch}."
        )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("ZINC MOLECULAR AUTOENCODER TRAINING")
    print("Configuration:", arguments.config)
    print("Split:", split_path)
    print("Output:", output_directory)
    print("Device:", device)
    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(device),
        )
    print("Trainable parameters:", f"{parameter_count:,}")
    print("Training molecules:", len(loaders["train"].dataset))
    print("Validation molecules:", len(loaders["val"].dataset))
    print("Batch size:", batch_size)
    print("Epoch limit:", epochs)
    print("Starting epoch:", start_epoch)
    print("Test split accessed: no")

    if start_epoch > epochs:
        print(
            "Checkpoint already reached the requested "
            "epoch limit; no training required."
        )
        return

    training_started = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_started = time.time()

        # Epoch-specific seeds make interrupted/resumed runs
        # reproducible from epoch boundaries.
        if getattr(
            loaders["train"],
            "generator",
            None,
        ) is not None:
            loaders["train"].generator.manual_seed(
                seed + 10_000 + epoch
            )

        generator_device = (
            "cuda"
            if device.type == "cuda"
            else "cpu"
        )
        training_noise_generator = torch.Generator(
            device=generator_device
        )
        training_noise_generator.manual_seed(
            seed + 20_000 + epoch
        )

        training_metrics = train_autoencoder_epoch(
            model,
            loaders["train"],
            optimizer,
            device=device,
            noise_dim=noise_dim,
            bond_class_weight=bond_class_weight,
            gradient_clip=arguments.gradient_clip,
            noise_generator=training_noise_generator,
        )

        validation_metrics = evaluate_autoencoder(
            model,
            loaders["val"],
            device=device,
            noise_dim=noise_dim,
            bond_class_weight=bond_class_weight,
            noise_seed=seed + 30_000,
        )

        validation_loss = float(
            validation_metrics["total_loss"]
        )
        scheduler.step(validation_loss)

        improved = (
            validation_loss
            < best_validation_loss - 1e-8
        )

        if improved:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        elapsed = time.time() - epoch_started
        learning_rate = float(
            optimizer.param_groups[0]["lr"]
        )

        epoch_record = {
            "epoch": epoch,
            "seconds": elapsed,
            "learning_rate": learning_rate,
            "improved": improved,
            "train": training_metrics,
            "validation": validation_metrics,
        }
        history.append(epoch_record)

        checkpoint_extra = {
            "architecture": architecture_signature(
                configuration
            ),
            "scheduler_state_dict": (
                scheduler.state_dict()
            ),
            "epochs_without_improvement": (
                epochs_without_improvement
            ),
            "history": history,
            "train_metrics": training_metrics,
            "validation_metrics": validation_metrics,
            "split_path": str(split_path),
            "test_split_accessed": False,
        }

        save_checkpoint(
            last_path,
            model,
            optimizer,
            epoch,
            best_validation_loss,
            configuration,
            extra=checkpoint_extra,
        )

        if improved:
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                best_validation_loss,
                configuration,
                extra=checkpoint_extra,
            )

        write_json_atomic(
            history_path,
            history,
        )

        marker = " *best" if improved else ""

        print(
            f"Epoch {epoch:03d} | "
            f"train {metric_summary(training_metrics)} | "
            f"val {metric_summary(validation_metrics)} | "
            f"lr={learning_rate:.2e} | "
            f"{elapsed:.1f}s{marker}",
            flush=True,
        )

        if (
            epochs_without_improvement
            >= patience
        ):
            print(
                f"Early stopping after {patience} "
                "epochs without improvement."
            )
            break

    total_seconds = time.time() - training_started

    best_checkpoint = load_checkpoint(
        best_path,
        model,
        optimizer=None,
        device=device,
    )

    report = {
        "configuration": str(arguments.config),
        "split_path": str(split_path),
        "device": str(device),
        "trainable_parameters": parameter_count,
        "epochs_completed": len(history),
        "best_epoch": int(
            best_checkpoint["epoch"]
        ),
        "best_validation_loss": float(
            best_checkpoint["best_metric"]
        ),
        "best_validation_metrics": (
            best_checkpoint[
                "validation_metrics"
            ]
        ),
        "training_seconds_this_invocation": (
            total_seconds
        ),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "test_split_accessed": False,
    }

    write_json_atomic(
        report_path,
        report,
    )

    print("\nAUTOENCODER TRAINING RESULT")
    print(
        "Best epoch:",
        report["best_epoch"],
    )
    print(
        "Best validation loss:",
        round(
            report["best_validation_loss"],
            6,
        ),
    )
    print(
        "Best checkpoint:",
        best_path,
    )
    print(
        "Last checkpoint:",
        last_path,
    )
    print(
        "History:",
        history_path,
    )
    print("Test split accessed: no")
    print(
        "AUTOENCODER TRAINING COMPLETE"
    )


if __name__ == "__main__":
    main()
