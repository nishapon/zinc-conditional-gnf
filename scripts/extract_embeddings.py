#!/usr/bin/env python3
"""Extract property-aware ZINC embeddings for conditional GNF training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time
from typing import Any

import torch

from zinc_gnf.data.datasets import make_dataloaders
from zinc_gnf.data.embeddings import (
    RunningFeatureMoments,
    atomic_torch_save,
    extract_embedding_split,
)
from zinc_gnf.models.autoencoder import (
    MolecularEmbeddingQEDHead,
    build_autoencoder_from_config,
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
        "--splits",
        default=None,
        help=(
            "Preprocessed split pickle. Defaults to "
            "<processed_dir>/zinc_splits.pkl."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "outputs/checkpoints/"
            "property_aware_autoencoder/best.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/embeddings/zinc250k",
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
        "--batch-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing embedding output directory.",
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


def write_json_atomic(
    path: Path,
    payload: Any,
) -> None:
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


def main() -> None:
    arguments = parse_args()

    if arguments.num_workers < 0:
        raise ValueError(
            "num-workers cannot be negative."
        )
    if arguments.shard_size <= 0:
        raise ValueError(
            "shard-size must be positive."
        )
    if (
        arguments.batch_size is not None
        and arguments.batch_size <= 0
    ):
        raise ValueError(
            "batch-size must be positive."
        )

    configuration = load_config(
        arguments.config
    )
    seed = int(configuration["seed"])
    set_seed(seed)

    device = resolve_device(
        arguments.device
    )

    autoencoder_configuration = configuration[
        "autoencoder"
    ]

    processed_directory = Path(
        configuration["data"]["processed_dir"]
    )
    split_path = (
        Path(arguments.splits)
        if arguments.splits is not None
        else processed_directory / "zinc_splits.pkl"
    )
    checkpoint_path = Path(
        arguments.checkpoint
    )
    output_directory = Path(
        arguments.output_dir
    )

    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}"
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Property-aware checkpoint not found: "
            f"{checkpoint_path}"
        )

    if output_directory.exists():
        existing_files = list(
            output_directory.iterdir()
        )

        if existing_files:
            if not arguments.overwrite:
                raise FileExistsError(
                    f"{output_directory} is not empty. "
                    "Use --overwrite to replace it."
                )

            shutil.rmtree(
                output_directory
            )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    batch_size = (
        int(arguments.batch_size)
        if arguments.batch_size is not None
        else int(
            autoencoder_configuration[
                "batch_size"
            ]
        )
    )
    noise_seed = (
        int(arguments.noise_seed)
        if arguments.noise_seed is not None
        else seed + 60_000
    )

    loaders = make_dataloaders(
        split_path,
        batch_size=batch_size,
        seed=seed,
        num_workers=arguments.num_workers,
        include_test=False,
        pin_memory=(device.type == "cuda"),
    )

    checkpoint = load_raw_checkpoint(
        checkpoint_path,
        device,
    )

    required_checkpoint_keys = {
        "autoencoder_state_dict",
        "qed_head_state_dict",
    }
    missing_keys = (
        required_checkpoint_keys
        - checkpoint.keys()
    )

    if missing_keys:
        raise KeyError(
            "Property-aware checkpoint is missing: "
            + ", ".join(sorted(missing_keys))
        )

    autoencoder = build_autoencoder_from_config(
        autoencoder_configuration
    ).to(device)

    autoencoder.load_state_dict(
        checkpoint["autoencoder_state_dict"]
    )

    property_configuration = configuration.get(
        "property_aware_autoencoder",
        configuration.get(
            "property_aware",
            {},
        ),
    )

    qed_head = MolecularEmbeddingQEDHead(
        embedding_dim=int(
            autoencoder_configuration[
                "embedding_dim"
            ]
        ),
        hidden_dim=int(
            autoencoder_configuration[
                "hidden_dim"
            ]
        ),
        dropout=float(
            property_configuration.get(
                "dropout",
                0.1,
            )
        ),
    ).to(device)

    qed_head.load_state_dict(
        checkpoint["qed_head_state_dict"]
    )

    autoencoder.eval()
    qed_head.eval()

    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)
    for parameter in qed_head.parameters():
        parameter.requires_grad_(False)

    embedding_dim = int(
        autoencoder_configuration[
            "embedding_dim"
        ]
    )

    moments = RunningFeatureMoments(
        embedding_dim
    )

    print(
        "PROPERTY-AWARE EMBEDDING EXTRACTION"
    )
    print("Configuration:", arguments.config)
    print("Split:", split_path)
    print("Checkpoint:", checkpoint_path)
    print(
        "Checkpoint joint epoch:",
        checkpoint.get(
            "joint_completed",
            checkpoint.get("epoch", "unknown"),
        ),
    )
    print("Output:", output_directory)
    print("Device:", device)
    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(device),
        )
    print("Batch size:", batch_size)
    print("Shard size:", arguments.shard_size)
    print("Embedding dimension:", embedding_dim)
    print(
        "Training molecules:",
        len(loaders["train"].dataset),
    )
    print(
        "Validation molecules:",
        len(loaders["val"].dataset),
    )
    print("Test split accessed: no")

    started = time.time()

    train_manifest = extract_embedding_split(
        autoencoder,
        qed_head,
        loaders["train"],
        output_directory=output_directory,
        split_name="train",
        device=device,
        noise_dim=int(
            autoencoder_configuration[
                "node_noise_dim"
            ]
        ),
        noise_seed=noise_seed,
        shard_size=arguments.shard_size,
        moments=moments,
    )

    embedding_mean, embedding_std = (
        moments.finalize()
    )

    validation_manifest = (
        extract_embedding_split(
            autoencoder,
            qed_head,
            loaders["val"],
            output_directory=output_directory,
            split_name="val",
            device=device,
            noise_dim=int(
                autoencoder_configuration[
                    "node_noise_dim"
                ]
            ),
            noise_seed=noise_seed + 1,
            shard_size=arguments.shard_size,
            moments=None,
        )
    )

    normalization_path = (
        output_directory
        / "normalization.pt"
    )

    atomic_torch_save(
        {
            "embedding_mean": embedding_mean,
            "embedding_std": embedding_std,
            "training_node_count": (
                moments.count
            ),
            "embedding_dim": embedding_dim,
            "checkpoint": str(
                checkpoint_path
            ),
            "checkpoint_joint_epoch": (
                checkpoint.get(
                    "joint_completed",
                    checkpoint.get(
                        "epoch",
                        None,
                    ),
                )
            ),
            "noise_seed": noise_seed,
        },
        normalization_path,
    )

    elapsed = time.time() - started

    manifest = {
        "configuration": str(arguments.config),
        "split_path": str(split_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_joint_epoch": (
            checkpoint.get(
                "joint_completed",
                checkpoint.get("epoch", None),
            )
        ),
        "embedding_dim": embedding_dim,
        "batch_size": batch_size,
        "shard_size": arguments.shard_size,
        "noise_seed": noise_seed,
        "normalization": str(
            normalization_path.name
        ),
        "training": train_manifest,
        "validation": validation_manifest,
        "embedding_std_min": float(
            embedding_std.min().item()
        ),
        "embedding_std_max": float(
            embedding_std.max().item()
        ),
        "elapsed_seconds": elapsed,
        "test_split_accessed": False,
    }

    manifest_path = (
        output_directory / "manifest.json"
    )
    write_json_atomic(
        manifest_path,
        manifest,
    )

    assert embedding_mean.shape == (
        embedding_dim,
    )
    assert embedding_std.shape == (
        embedding_dim,
    )
    assert torch.isfinite(
        embedding_mean
    ).all()
    assert torch.isfinite(
        embedding_std
    ).all()
    assert (
        embedding_std > 0
    ).all()

    print("\nEMBEDDING EXTRACTION RESULT")
    print(
        "Training embeddings:",
        train_manifest["molecules"],
        "molecules /",
        train_manifest["nodes"],
        "nodes /",
        len(train_manifest["shards"]),
        "shards",
    )
    print(
        "Validation embeddings:",
        validation_manifest["molecules"],
        "molecules /",
        validation_manifest["nodes"],
        "nodes /",
        len(validation_manifest["shards"]),
        "shards",
    )
    print(
        "Training QED-head MAE:",
        round(
            train_manifest["qed_mae"],
            6,
        ),
    )
    print(
        "Validation QED-head MAE:",
        round(
            validation_manifest["qed_mae"],
            6,
        ),
    )
    print(
        "Validation QED correlation:",
        round(
            validation_manifest[
                "qed_correlation"
            ],
            6,
        ),
    )
    print(
        "Embedding std min/max:",
        round(
            manifest["embedding_std_min"],
            6,
        ),
        "/",
        round(
            manifest["embedding_std_max"],
            6,
        ),
    )
    print(
        "Normalization:",
        normalization_path,
    )
    print("Manifest:", manifest_path)
    print(
        "Elapsed seconds:",
        round(elapsed, 1),
    )
    print("Training performed: no")
    print("Test split accessed: no")
    print(
        "EMBEDDING EXTRACTION COMPLETE"
    )


if __name__ == "__main__":
    main()
