#!/usr/bin/env python3
"""Train the conditional node-level GNF on extracted ZINC embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch

from zinc_gnf.data.flow_dataset import (
    make_flow_dataloaders,
)
from zinc_gnf.models.flow import (
    build_flow_from_config,
)
from zinc_gnf.training.flow import (
    evaluate_flow,
    flow_metric_summary,
    train_flow_epoch,
)
from zinc_gnf.utils.config import load_config
from zinc_gnf.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--config",
        default="configs/zinc250k.yaml",
    )
    parser.add_argument(
        "--embeddings",
        default="outputs/embeddings/zinc250k",
        help=(
            "Directory containing manifest.json, "
            "normalization.pt and embedding shards."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/checkpoints/"
            "conditional_flow"
        ),
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Flow last.pt checkpoint to resume.",
    )
    parser.add_argument(
        "--device",
        default="auto",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help=(
            "Optional per-epoch training-batch limit for smoke tests."
        ),
    )
    parser.add_argument(
        "--max-validation-batches",
        type=int,
        default=None,
        help=(
            "Optional validation-batch limit for smoke tests."
        ),
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


def flow_configuration(
    configuration: dict[str, Any],
) -> dict[str, Any]:
    configured = dict(
        configuration.get("flow", {})
    )

    configured.setdefault(
        "embedding_dim",
        int(
            configuration["autoencoder"][
                "embedding_dim"
            ]
        ),
    )
    configured.setdefault(
        "condition_dim",
        2,
    )
    configured.setdefault(
        "hidden_dim",
        128,
    )
    configured.setdefault(
        "context_dim",
        64,
    )
    configured.setdefault(
        "attention_heads",
        4,
    )
    configured.setdefault(
        "coupling_blocks",
        12,
    )
    configured.setdefault(
        "max_log_scale",
        0.25,
    )
    configured.setdefault(
        "minimum_sigma",
        0.1,
    )
    configured.setdefault(
        "batch_size",
        32,
    )
    configured.setdefault(
        "epochs",
        40,
    )
    configured.setdefault(
        "learning_rate",
        2e-4,
    )
    configured.setdefault(
        "patience",
        8,
    )

    return configured


def architecture_signature(
    configured_flow: dict[str, Any],
) -> dict[str, Any]:
    keys = (
        "embedding_dim",
        "condition_dim",
        "hidden_dim",
        "context_dim",
        "attention_heads",
        "coupling_blocks",
        "max_log_scale",
        "minimum_sigma",
    )

    return {
        key: configured_flow[key]
        for key in keys
    }


def atomic_torch_save(
    payload: Any,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )
    torch.save(payload, temporary)
    temporary.replace(path)


def write_json_atomic(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            allow_nan=False,
        )

    temporary.replace(path)


def load_raw_checkpoint(
    path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


def restore_rng_state(
    checkpoint: dict[str, Any],
) -> None:
    if checkpoint.get(
        "torch_rng_state"
    ) is not None:
        torch.set_rng_state(
            checkpoint["torch_rng_state"]
        )

    if (
        torch.cuda.is_available()
        and checkpoint.get(
            "cuda_rng_state"
        ) is not None
    ):
        torch.cuda.set_rng_state_all(
            checkpoint["cuda_rng_state"]
        )


def save_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    configuration: dict[str, Any],
    configured_flow: dict[str, Any],
    embeddings: Path,
    epoch: int,
    best_metric: float,
    epochs_without_improvement: int,
    history: list[dict[str, Any]],
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
) -> None:
    atomic_torch_save(
        {
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "scheduler_state_dict": (
                scheduler.state_dict()
            ),
            "configuration": configuration,
            "flow_configuration": (
                configured_flow
            ),
            "architecture": (
                architecture_signature(
                    configured_flow
                )
            ),
            "embedding_root": str(
                embeddings
            ),
            "epoch": int(epoch),
            "best_metric": float(
                best_metric
            ),
            "epochs_without_improvement": int(
                epochs_without_improvement
            ),
            "history": history,
            "train_metrics": train_metrics,
            "validation_metrics": (
                validation_metrics
            ),
            "torch_rng_state": (
                torch.get_rng_state()
            ),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
            "test_split_accessed": False,
        },
        path,
    )


def main() -> None:
    arguments = parse_args()

    if arguments.num_workers < 0:
        raise ValueError(
            "num-workers cannot be negative."
        )
    if arguments.cache_size <= 0:
        raise ValueError(
            "cache-size must be positive."
        )
    if arguments.gradient_clip <= 0:
        raise ValueError(
            "gradient-clip must be positive."
        )

    configuration = load_config(
        arguments.config
    )
    configured_flow = flow_configuration(
        configuration
    )

    seed = int(configuration["seed"])
    set_seed(seed)

    device = resolve_device(
        arguments.device
    )
    embedding_root = Path(
        arguments.embeddings
    )
    output_directory = Path(
        arguments.output_dir
    )

    if not (
        embedding_root / "manifest.json"
    ).exists():
        raise FileNotFoundError(
            "Embedding manifest not found under "
            f"{embedding_root}. Run "
            "scripts/extract_embeddings.py first."
        )

    epochs = (
        int(arguments.epochs)
        if arguments.epochs is not None
        else int(configured_flow["epochs"])
    )
    batch_size = (
        int(arguments.batch_size)
        if arguments.batch_size is not None
        else int(
            configured_flow["batch_size"]
        )
    )
    learning_rate = (
        float(arguments.learning_rate)
        if arguments.learning_rate is not None
        else float(
            configured_flow[
                "learning_rate"
            ]
        )
    )

    if epochs <= 0:
        raise ValueError(
            "epochs must be positive."
        )
    if batch_size <= 0:
        raise ValueError(
            "batch-size must be positive."
        )
    if learning_rate <= 0:
        raise ValueError(
            "learning-rate must be positive."
        )
    if (
        arguments.max_train_batches is not None
        and arguments.max_train_batches <= 0
    ):
        raise ValueError(
            "max-train-batches must be positive."
        )
    if (
        arguments.max_validation_batches is not None
        and arguments.max_validation_batches <= 0
    ):
        raise ValueError(
            "max-validation-batches must be positive."
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    best_path = output_directory / "best.pt"
    last_path = output_directory / "last.pt"
    history_path = output_directory / "history.json"
    report_path = output_directory / "report.json"

    loaders = make_flow_dataloaders(
        embedding_root,
        batch_size=batch_size,
        seed=seed,
        num_workers=arguments.num_workers,
        pin_memory=(device.type == "cuda"),
        cache_size=arguments.cache_size,
    )

    model = build_flow_from_config(
        configured_flow
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-6,
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

    start_epoch = 1
    best_metric = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    if arguments.resume is not None:
        checkpoint = load_raw_checkpoint(
            arguments.resume,
            device,
        )

        saved_architecture = checkpoint.get(
            "architecture"
        )
        current_architecture = (
            architecture_signature(
                configured_flow
            )
        )

        if (
            saved_architecture is not None
            and saved_architecture
            != current_architecture
        ):
            raise RuntimeError(
                "Resume checkpoint architecture does "
                "not match the current configuration.\n"
                f"Saved: {saved_architecture}\n"
                f"Current: {current_architecture}"
            )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        if checkpoint.get(
            "scheduler_state_dict"
        ):
            scheduler.load_state_dict(
                checkpoint[
                    "scheduler_state_dict"
                ]
            )

        restore_rng_state(checkpoint)

        start_epoch = (
            int(checkpoint["epoch"]) + 1
        )
        best_metric = float(
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

    print(
        "ZINC CONDITIONAL NODE-LEVEL GNF TRAINING"
    )
    print("Configuration:", arguments.config)
    print("Embeddings:", embedding_root)
    print("Output:", output_directory)
    print("Device:", device)
    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(device),
        )
    print(
        "Trainable parameters:",
        f"{parameter_count:,}",
    )
    print(
        "Coupling blocks:",
        configured_flow["coupling_blocks"],
    )
    print(
        "Embedding dimension:",
        configured_flow["embedding_dim"],
    )
    print(
        "Condition:",
        "standardized QED + log1p(node count)",
    )
    print(
        "Training molecules:",
        len(loaders["train"].dataset),
    )
    print(
        "Validation molecules:",
        len(loaders["val"].dataset),
    )
    print("Batch size:", batch_size)
    print("Epoch limit:", epochs)
    print("Starting epoch:", start_epoch)
    print("Learning rate:", learning_rate)
    print("Maximum train batches:", arguments.max_train_batches)
    print("Maximum validation batches:", arguments.max_validation_batches)
    print("Objective: conditional NLL")
    print("Test split accessed: no")

    if start_epoch > epochs:
        print(
            "Checkpoint has already reached the "
            "requested epoch limit."
        )
        return

    started = time.time()
    patience = int(
        configured_flow["patience"]
    )

    for epoch in range(
        start_epoch,
        epochs + 1,
    ):
        epoch_started = time.time()

        if getattr(
            loaders["train"],
            "generator",
            None,
        ) is not None:
            loaders["train"].generator.manual_seed(
                seed + 70_000 + epoch
            )

        train_metrics = train_flow_epoch(
            model,
            loaders["train"],
            optimizer,
            device=device,
            gradient_clip=(
                arguments.gradient_clip
            ),
            max_batches=(
                arguments.max_train_batches
            ),
        )

        validation_metrics = evaluate_flow(
            model,
            loaders["val"],
            device=device,
            max_batches=(
                arguments.max_validation_batches
            ),
        )

        validation_metric = float(
            validation_metrics[
                "nll_per_dimension"
            ]
        )
        scheduler.step(validation_metric)

        improved = (
            validation_metric
            < best_metric - 1e-8
        )

        if improved:
            best_metric = validation_metric
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        elapsed = time.time() - epoch_started
        learning_rate_now = float(
            optimizer.param_groups[0]["lr"]
        )

        record = {
            "epoch": epoch,
            "seconds": elapsed,
            "learning_rate": (
                learning_rate_now
            ),
            "improved": improved,
            "train": train_metrics,
            "validation": (
                validation_metrics
            ),
        }
        history.append(record)

        checkpoint_arguments = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "configuration": configuration,
            "configured_flow": (
                configured_flow
            ),
            "embeddings": embedding_root,
            "epoch": epoch,
            "best_metric": best_metric,
            "epochs_without_improvement": (
                epochs_without_improvement
            ),
            "history": history,
            "train_metrics": train_metrics,
            "validation_metrics": (
                validation_metrics
            ),
        }

        save_training_checkpoint(
            last_path,
            **checkpoint_arguments,
        )

        if improved:
            save_training_checkpoint(
                best_path,
                **checkpoint_arguments,
            )

        write_json_atomic(
            history_path,
            history,
        )

        marker = " *best" if improved else ""

        print(
            f"Epoch {epoch:03d} | "
            f"train "
            f"{flow_metric_summary(train_metrics)} | "
            f"val "
            f"{flow_metric_summary(validation_metrics)} | "
            f"lr={learning_rate_now:.2e} | "
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

    total_seconds = time.time() - started

    best_checkpoint = load_raw_checkpoint(
        best_path,
        device,
    )
    model.load_state_dict(
        best_checkpoint["model_state_dict"]
    )

    report = {
        "configuration": str(arguments.config),
        "embedding_root": str(
            embedding_root
        ),
        "device": str(device),
        "trainable_parameters": (
            parameter_count
        ),
        "epochs_completed": len(history),
        "best_epoch": int(
            best_checkpoint["epoch"]
        ),
        "best_validation_nll_per_node": float(
            best_checkpoint[
                "validation_metrics"
            ]["nll_per_node"]
        ),
        "best_validation_nll_per_dimension": float(
            best_checkpoint[
                "validation_metrics"
            ]["nll_per_dimension"]
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

    print("\nCONDITIONAL GNF RESULT")
    print(
        "Best epoch:",
        report["best_epoch"],
    )
    print(
        "Best validation NLL/node:",
        round(
            report[
                "best_validation_nll_per_node"
            ],
            6,
        ),
    )
    print(
        "Best validation NLL/dimension:",
        round(
            report[
                "best_validation_nll_per_dimension"
            ],
            6,
        ),
    )
    print("Best checkpoint:", best_path)
    print("Last checkpoint:", last_path)
    print("History:", history_path)
    print("Test split accessed: no")
    print(
        "CONDITIONAL GNF TRAINING COMPLETE"
    )


if __name__ == "__main__":
    main()
