#!/usr/bin/env python3
"""Train the property-aware ZINC molecular autoencoder.

Stage A:
    Freeze the pretrained autoencoder and warm up the QED head.

Stage B:
    Jointly fine-tune the autoencoder and QED head with reconstruction
    and QED supervision.

Only the training and validation splits are loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch

from zinc_gnf.data.datasets import make_dataloaders
from zinc_gnf.models.autoencoder import (
    MolecularEmbeddingQEDHead,
    build_autoencoder_from_config,
)
from zinc_gnf.training.property_aware import (
    evaluate_property_aware,
    evaluate_qed_head,
    property_metric_summary,
    train_property_aware_epoch,
    train_qed_head_epoch,
)
from zinc_gnf.utils.checkpointing import load_checkpoint
from zinc_gnf.utils.config import load_config
from zinc_gnf.utils.seed import set_seed


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
        "--source-checkpoint",
        default="outputs/checkpoints/autoencoder/best.pt",
        help="Best pretrained autoencoder checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/checkpoints/"
            "property_aware_autoencoder"
        ),
    )
    parser.add_argument(
        "--resume",
        default=None,
        help=(
            "Property-aware last.pt checkpoint to resume."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or e.g. cuda:0.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--joint-epochs",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--qed-loss-weight",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--qed-balanced-sampling",
        action="store_true",
        help="Use capped inverse-frequency QED sampling for training.",
    )
    parser.add_argument(
        "--qed-balance-bins",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--qed-balance-power",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--qed-balance-max-weight",
        type=float,
        default=4.0,
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


def property_configuration(
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """Read property settings while retaining safe defaults."""
    configured = configuration.get(
        "property_aware_autoencoder",
        configuration.get(
            "property_aware",
            {},
        ),
    )

    return {
        "warmup_epochs": int(
            configured.get("warmup_epochs", 5)
        ),
        "joint_epochs": int(
            configured.get("joint_epochs", 25)
        ),
        "head_learning_rate": float(
            configured.get(
                "head_learning_rate",
                1e-3,
            )
        ),
        "joint_learning_rate": float(
            configured.get(
                "joint_learning_rate",
                1e-4,
            )
        ),
        "qed_loss_weight": float(
            configured.get(
                "qed_loss_weight",
                2.0,
            )
        ),
        "patience": int(
            configured.get("patience", 8)
        ),
        "dropout": float(
            configured.get("dropout", 0.1)
        ),
    }


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


def save_property_checkpoint(
    path: Path,
    *,
    autoencoder: torch.nn.Module,
    qed_head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    configuration: dict[str, Any],
    source_checkpoint: str,
    source_epoch: int,
    stage: str,
    stage_epoch: int,
    warmup_completed: int,
    joint_completed: int,
    best_metric: float,
    epochs_without_improvement: int,
    history: list[dict[str, Any]],
    validation_metrics: dict[str, float],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    payload = {
        "autoencoder_state_dict": (
            autoencoder.state_dict()
        ),
        "qed_head_state_dict": (
            qed_head.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),
        "configuration": configuration,
        "source_checkpoint": source_checkpoint,
        "source_autoencoder_epoch": source_epoch,
        "stage": stage,
        "stage_epoch": stage_epoch,
        "warmup_completed": warmup_completed,
        "joint_completed": joint_completed,
        "best_metric": best_metric,
        "epochs_without_improvement": (
            epochs_without_improvement
        ),
        "history": history,
        "validation_metrics": validation_metrics,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
        "test_split_accessed": False,
    }

    torch.save(payload, temporary)
    temporary.replace(path)


def load_raw_checkpoint(
    path: str | Path,
    *,
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


def make_epoch_generator(
    *,
    device: torch.device,
    seed: int,
) -> torch.Generator:
    generator = torch.Generator(
        device=(
            "cuda"
            if device.type == "cuda"
            else "cpu"
        )
    )
    generator.manual_seed(int(seed))
    return generator


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
    property_config = property_configuration(
        configuration
    )

    warmup_epochs = (
        int(arguments.warmup_epochs)
        if arguments.warmup_epochs is not None
        else property_config["warmup_epochs"]
    )
    joint_epochs = (
        int(arguments.joint_epochs)
        if arguments.joint_epochs is not None
        else property_config["joint_epochs"]
    )
    qed_loss_weight = (
        float(arguments.qed_loss_weight)
        if arguments.qed_loss_weight is not None
        else property_config["qed_loss_weight"]
    )

    if warmup_epochs < 0:
        raise ValueError(
            "warmup-epochs cannot be negative."
        )
    if joint_epochs < 1:
        raise ValueError(
            "joint-epochs must be positive."
        )
    if qed_loss_weight <= 0:
        raise ValueError(
            "qed-loss-weight must be positive."
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
            f"Preprocessed split not found: {split_path}"
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

    loaders = make_dataloaders(
        split_path,
        batch_size=int(
            autoencoder_configuration["batch_size"]
        ),
        seed=seed,
        num_workers=arguments.num_workers,
        include_test=False,
        pin_memory=(device.type == "cuda"),
        qed_balanced_sampling=(
            arguments.qed_balanced_sampling
        ),
        qed_balance_bins=(
            arguments.qed_balance_bins
        ),
        qed_balance_power=(
            arguments.qed_balance_power
        ),
        qed_balance_max_weight=(
            arguments.qed_balance_max_weight
        ),
    )

    autoencoder = build_autoencoder_from_config(
        autoencoder_configuration
    ).to(device)

    qed_head = MolecularEmbeddingQEDHead(
        embedding_dim=int(
            autoencoder_configuration["embedding_dim"]
        ),
        hidden_dim=int(
            autoencoder_configuration["hidden_dim"]
        ),
        dropout=property_config["dropout"],
    ).to(device)

    source_checkpoint_path = str(
        arguments.source_checkpoint
    )
    source_epoch = -1
    history: list[dict[str, Any]] = []
    warmup_completed = 0
    joint_completed = 0
    best_metric = float("inf")
    epochs_without_improvement = 0
    resume_stage = None
    resume_checkpoint = None

    if arguments.resume is None:
        source_path = Path(source_checkpoint_path)

        if not source_path.exists():
            raise FileNotFoundError(
                "Pretrained autoencoder checkpoint not found: "
                f"{source_path}\n"
                "Run scripts/train_autoencoder.py first."
            )

        source_checkpoint = load_checkpoint(
            source_path,
            autoencoder,
            optimizer=None,
            device=device,
        )
        source_epoch = int(
            source_checkpoint["epoch"]
        )
    else:
        resume_checkpoint = load_raw_checkpoint(
            arguments.resume,
            device=device,
        )

        autoencoder.load_state_dict(
            resume_checkpoint[
                "autoencoder_state_dict"
            ]
        )
        qed_head.load_state_dict(
            resume_checkpoint[
                "qed_head_state_dict"
            ]
        )

        source_checkpoint_path = str(
            resume_checkpoint["source_checkpoint"]
        )
        source_epoch = int(
            resume_checkpoint[
                "source_autoencoder_epoch"
            ]
        )
        history = list(
            resume_checkpoint.get("history", [])
        )
        warmup_completed = int(
            resume_checkpoint.get(
                "warmup_completed",
                0,
            )
        )
        joint_completed = int(
            resume_checkpoint.get(
                "joint_completed",
                0,
            )
        )
        best_metric = float(
            resume_checkpoint.get(
                "best_metric",
                float("inf"),
            )
        )
        epochs_without_improvement = int(
            resume_checkpoint.get(
                "epochs_without_improvement",
                0,
            )
        )
        resume_stage = str(
            resume_checkpoint["stage"]
        )

        restore_rng_state(resume_checkpoint)

        print(
            f"Resumed {resume_stage} checkpoint "
            f"from {arguments.resume}."
        )

    noise_dim = int(
        autoencoder_configuration["node_noise_dim"]
    )
    bond_class_weight = float(
        autoencoder_configuration[
            "bond_class_weight"
        ]
    )

    autoencoder_parameters = sum(
        parameter.numel()
        for parameter in autoencoder.parameters()
    )
    qed_parameters = sum(
        parameter.numel()
        for parameter in qed_head.parameters()
    )

    print(
        "PROPERTY-AWARE ZINC AUTOENCODER TRAINING"
    )
    print("Configuration:", arguments.config)
    print("Split:", split_path)
    print("Source checkpoint:", source_checkpoint_path)
    print("Source epoch:", source_epoch)
    print("Output:", output_directory)
    print("Device:", device)
    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(device),
        )
    print(
        "Autoencoder parameters:",
        f"{autoencoder_parameters:,}",
    )
    print(
        "QED-head parameters:",
        f"{qed_parameters:,}",
    )
    print(
        "Training molecules:",
        len(loaders["train"].dataset),
    )
    print(
        "Validation molecules:",
        len(loaders["val"].dataset),
    )
    print("Warm-up epochs:", warmup_epochs)
    print("Joint epochs:", joint_epochs)
    print("QED loss weight:", qed_loss_weight)
    print(
        "QED-balanced sampling:",
        arguments.qed_balanced_sampling,
    )
    print(
        "QED balance bins/power/cap:",
        arguments.qed_balance_bins,
        arguments.qed_balance_power,
        arguments.qed_balance_max_weight,
    )
    print("Test split accessed: no")

    training_started = time.time()

    # --------------------------------------------------------
    # Stage A: QED-head warm-up.
    # --------------------------------------------------------

    if joint_completed == 0 and warmup_completed < warmup_epochs:
        for parameter in autoencoder.parameters():
            parameter.requires_grad_(False)
        for parameter in qed_head.parameters():
            parameter.requires_grad_(True)

        warmup_optimizer = torch.optim.AdamW(
            qed_head.parameters(),
            lr=property_config[
                "head_learning_rate"
            ],
            weight_decay=1e-5,
        )

        if (
            resume_checkpoint is not None
            and resume_stage == "warmup"
        ):
            warmup_optimizer.load_state_dict(
                resume_checkpoint[
                    "optimizer_state_dict"
                ]
            )

        print("\nSTAGE A — QED HEAD WARM-UP")

        for epoch in range(
            warmup_completed + 1,
            warmup_epochs + 1,
        ):
            epoch_started = time.time()

            if getattr(
                loaders["train"],
                "generator",
                None,
            ) is not None:
                loaders["train"].generator.manual_seed(
                    seed + 40_000 + epoch
                )

            metrics = train_qed_head_epoch(
                autoencoder,
                qed_head,
                loaders["train"],
                warmup_optimizer,
                device=device,
                noise_dim=noise_dim,
                noise_generator=make_epoch_generator(
                    device=device,
                    seed=seed + 41_000 + epoch,
                ),
                gradient_clip=arguments.gradient_clip,
            )

            validation = evaluate_qed_head(
                autoencoder,
                qed_head,
                loaders["val"],
                device=device,
                noise_dim=noise_dim,
                noise_seed=seed + 42_000,
            )

            warmup_completed = epoch
            elapsed = time.time() - epoch_started

            record = {
                "stage": "warmup",
                "epoch": epoch,
                "seconds": elapsed,
                "train": metrics,
                "validation": validation,
            }
            history.append(record)

            save_property_checkpoint(
                last_path,
                autoencoder=autoencoder,
                qed_head=qed_head,
                optimizer=warmup_optimizer,
                scheduler=None,
                configuration=configuration,
                source_checkpoint=source_checkpoint_path,
                source_epoch=source_epoch,
                stage="warmup",
                stage_epoch=epoch,
                warmup_completed=warmup_completed,
                joint_completed=joint_completed,
                best_metric=best_metric,
                epochs_without_improvement=(
                    epochs_without_improvement
                ),
                history=history,
                validation_metrics=validation,
            )
            write_json_atomic(
                history_path,
                history,
            )

            print(
                f"Warm-up {epoch:03d} | "
                f"train MAE={metrics['qed_mae']:.5f} | "
                f"val MAE={validation['qed_mae']:.5f} | "
                f"corr="
                f"{validation['qed_correlation']:.4f} | "
                f"R²="
                f"{validation['qed_r_squared']:.4f} | "
                f"{elapsed:.1f}s",
                flush=True,
            )

    # --------------------------------------------------------
    # Stage B: joint property-aware fine-tuning.
    # --------------------------------------------------------

    for parameter in autoencoder.parameters():
        parameter.requires_grad_(True)
    for parameter in qed_head.parameters():
        parameter.requires_grad_(True)

    joint_optimizer = torch.optim.AdamW(
        list(autoencoder.parameters())
        + list(qed_head.parameters()),
        lr=property_config[
            "joint_learning_rate"
        ],
        weight_decay=1e-5,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            joint_optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        )
    )

    if (
        resume_checkpoint is not None
        and resume_stage == "joint"
    ):
        joint_optimizer.load_state_dict(
            resume_checkpoint[
                "optimizer_state_dict"
            ]
        )
        if resume_checkpoint.get(
            "scheduler_state_dict"
        ):
            scheduler.load_state_dict(
                resume_checkpoint[
                    "scheduler_state_dict"
                ]
            )

    print(
        "\nSTAGE B — JOINT PROPERTY-AWARE FINE-TUNING"
    )

    for epoch in range(
        joint_completed + 1,
        joint_epochs + 1,
    ):
        epoch_started = time.time()

        if getattr(
            loaders["train"],
            "generator",
            None,
        ) is not None:
            loaders["train"].generator.manual_seed(
                seed + 50_000 + epoch
            )

        training_metrics = (
            train_property_aware_epoch(
                autoencoder,
                qed_head,
                loaders["train"],
                joint_optimizer,
                device=device,
                noise_dim=noise_dim,
                bond_class_weight=bond_class_weight,
                qed_loss_weight=qed_loss_weight,
                noise_generator=make_epoch_generator(
                    device=device,
                    seed=seed + 51_000 + epoch,
                ),
                gradient_clip=arguments.gradient_clip,
            )
        )

        validation_metrics = (
            evaluate_property_aware(
                autoencoder,
                qed_head,
                loaders["val"],
                device=device,
                noise_dim=noise_dim,
                bond_class_weight=bond_class_weight,
                qed_loss_weight=qed_loss_weight,
                noise_seed=seed + 52_000,
            )
        )

        validation_objective = float(
            validation_metrics["total_loss"]
        )
        scheduler.step(validation_objective)

        improved = (
            validation_objective
            < best_metric - 1e-8
        )

        if improved:
            best_metric = validation_objective
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        joint_completed = epoch
        elapsed = time.time() - epoch_started

        record = {
            "stage": "joint",
            "epoch": epoch,
            "seconds": elapsed,
            "learning_rate": float(
                joint_optimizer.param_groups[0]["lr"]
            ),
            "improved": improved,
            "train": training_metrics,
            "validation": validation_metrics,
        }
        history.append(record)

        checkpoint_arguments = {
            "autoencoder": autoencoder,
            "qed_head": qed_head,
            "optimizer": joint_optimizer,
            "scheduler": scheduler,
            "configuration": configuration,
            "source_checkpoint": source_checkpoint_path,
            "source_epoch": source_epoch,
            "stage": "joint",
            "stage_epoch": epoch,
            "warmup_completed": warmup_completed,
            "joint_completed": joint_completed,
            "best_metric": best_metric,
            "epochs_without_improvement": (
                epochs_without_improvement
            ),
            "history": history,
            "validation_metrics": validation_metrics,
        }

        save_property_checkpoint(
            last_path,
            **checkpoint_arguments,
        )

        if improved:
            save_property_checkpoint(
                best_path,
                **checkpoint_arguments,
            )

        write_json_atomic(
            history_path,
            history,
        )

        marker = " *best" if improved else ""

        print(
            f"Joint {epoch:03d} | "
            f"train "
            f"{property_metric_summary(training_metrics)} | "
            f"val "
            f"{property_metric_summary(validation_metrics)} | "
            f"{elapsed:.1f}s{marker}",
            flush=True,
        )

        if (
            epochs_without_improvement
            >= property_config["patience"]
        ):
            print(
                "Early stopping after "
                f"{property_config['patience']} "
                "epochs without improvement."
            )
            break

    if not best_path.exists():
        raise RuntimeError(
            "No best joint checkpoint was created. "
            "If resuming, ensure the original best.pt is "
            "also available in the output directory."
        )

    best_checkpoint = load_raw_checkpoint(
        best_path,
        device=device,
    )
    autoencoder.load_state_dict(
        best_checkpoint[
            "autoencoder_state_dict"
        ]
    )
    qed_head.load_state_dict(
        best_checkpoint[
            "qed_head_state_dict"
        ]
    )

    total_seconds = time.time() - training_started

    report = {
        "configuration": str(arguments.config),
        "split_path": str(split_path),
        "source_checkpoint": source_checkpoint_path,
        "source_autoencoder_epoch": source_epoch,
        "device": str(device),
        "warmup_epochs_completed": warmup_completed,
        "joint_epochs_completed": joint_completed,
        "best_joint_epoch": int(
            best_checkpoint["joint_completed"]
        ),
        "best_validation_objective": float(
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

    print(
        "\nPROPERTY-AWARE AUTOENCODER RESULT"
    )
    print(
        "Source autoencoder epoch:",
        source_epoch,
    )
    print(
        "Best joint epoch:",
        report["best_joint_epoch"],
    )
    print(
        "Best validation objective:",
        round(
            report["best_validation_objective"],
            6,
        ),
    )
    print(
        "Validation QED MAE:",
        round(
            report[
                "best_validation_metrics"
            ]["qed_mae"],
            6,
        ),
    )
    print("Best checkpoint:", best_path)
    print("Last checkpoint:", last_path)
    print("History:", history_path)
    print("Test split accessed: no")
    print(
        "PROPERTY-AWARE AUTOENCODER TRAINING COMPLETE"
    )


if __name__ == "__main__":
    main()
