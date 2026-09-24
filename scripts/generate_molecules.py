#!/usr/bin/env python3
"""Generate and evaluate molecules from the conditional ZINC GNF."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import pickle
import time
from typing import Any

import numpy as np
import torch

from zinc_gnf.evaluation.generation import (
    canonical_training_smiles,
    generate_conditioned_batch,
    summarize_generation,
)
from zinc_gnf.models.autoencoder import (
    build_autoencoder_from_config,
)
from zinc_gnf.models.flow import (
    build_flow_from_config,
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
            "Preprocessed zinc_splits.pkl. Defaults to "
            "<processed_dir>/zinc_splits.pkl."
        ),
    )
    parser.add_argument(
        "--autoencoder-checkpoint",
        default=(
            "outputs/checkpoints/"
            "property_aware_autoencoder/best.pt"
        ),
    )
    parser.add_argument(
        "--flow-checkpoint",
        default=(
            "outputs/checkpoints/"
            "conditional_flow/best.pt"
        ),
    )
    parser.add_argument(
        "--embeddings",
        default="outputs/embeddings/zinc250k",
        help=(
            "Directory containing normalization.pt."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/samples/zinc250k",
    )
    parser.add_argument(
        "--device",
        default="auto",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--target-tolerance",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--fixed-qed",
        type=float,
        default=None,
        help=(
            "If supplied, use this QED target for every sample. "
            "Otherwise QED and node count are sampled jointly from "
            "the training distribution."
        ),
    )
    parser.add_argument(
        "--fixed-n-node",
        type=int,
        default=None,
        help=(
            "If supplied, use this heavy-atom count for every sample."
        ),
    )

    return parser.parse_args()


def resolve_device(
    requested: str,
) -> torch.device:
    if requested == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(
        requested
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is unavailable."
        )

    return device


def torch_load(
    path: str | Path,
    *,
    map_location: Any = "cpu",
) -> Any:
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


def load_split_payload(
    path: str | Path,
) -> dict[str, Any]:
    with Path(path).open(
        "rb"
    ) as handle:
        payload = pickle.load(
            handle
        )

    required = {
        "train",
        "condition_mean",
        "condition_std",
    }
    missing = required.difference(
        payload
    )

    if missing:
        raise KeyError(
            "Split payload is missing: "
            + ", ".join(
                sorted(missing)
            )
        )

    return payload


def json_safe(
    value: Any,
) -> Any:
    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): json_safe(item)
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        np.ndarray,
    ):
        return json_safe(
            value.tolist()
        )

    if isinstance(
        value,
        np.generic,
    ):
        return json_safe(
            value.item()
        )

    if torch.is_tensor(value):
        return json_safe(
            value.detach()
            .cpu()
            .tolist()
        )

    if isinstance(
        value,
        float,
    ):
        if not math.isfinite(
            value
        ):
            return None

    return value


def write_json(
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

    temporary.write_text(
        json.dumps(
            json_safe(payload),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(
        path
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
        return

    fieldnames = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(
                    key
                )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: json_safe(
                        row.get(key)
                    )
                    for key in fieldnames
                }
            )

    temporary.replace(
        path
    )


def resolve_paths(
    arguments: argparse.Namespace,
    configuration: dict[str, Any],
) -> tuple[Path, Path, Path, Path, Path]:
    processed_directory = Path(
        configuration["data"][
            "processed_dir"
        ]
    )

    split_path = Path(
        arguments.splits
    ) if arguments.splits else (
        processed_directory
        / "zinc_splits.pkl"
    )

    return (
        split_path,
        Path(
            arguments.autoencoder_checkpoint
        ),
        Path(
            arguments.flow_checkpoint
        ),
        Path(
            arguments.embeddings
        ),
        Path(
            arguments.output_dir
        ),
    )


def validate_paths(
    paths: list[Path],
) -> None:
    missing = [
        path
        for path in paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required files are missing:\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )


def main() -> None:
    arguments = parse_args()

    configuration = load_config(
        arguments.config
    )
    generation_configuration = dict(
        configuration.get(
            "generation",
            {},
        )
    )

    seed = int(
        arguments.seed
        if arguments.seed is not None
        else configuration.get(
            "seed",
            42,
        )
    )

    num_samples = int(
        arguments.num_samples
        if arguments.num_samples
        is not None
        else generation_configuration.get(
            "samples",
            10_000,
        )
    )

    batch_size = int(
        arguments.batch_size
        if arguments.batch_size
        is not None
        else generation_configuration.get(
            "batch_size",
            64,
        )
    )

    temperature = float(
        arguments.temperature
        if arguments.temperature
        is not None
        else generation_configuration.get(
            "temperature",
            1.0,
        )
    )

    target_tolerance = float(
        arguments.target_tolerance
        if arguments.target_tolerance
        is not None
        else generation_configuration.get(
            "target_tolerance",
            0.05,
        )
    )

    if num_samples <= 0:
        raise ValueError(
            "num-samples must be positive."
        )
    if batch_size <= 0:
        raise ValueError(
            "batch-size must be positive."
        )
    if temperature <= 0:
        raise ValueError(
            "temperature must be positive."
        )
    if target_tolerance <= 0:
        raise ValueError(
            "target-tolerance must be positive."
        )
    if (
        arguments.fixed_qed
        is not None
        and not (
            0.0
            <= arguments.fixed_qed
            <= 1.0
        )
    ):
        raise ValueError(
            "fixed-qed must lie in [0, 1]."
        )
    if (
        arguments.fixed_n_node
        is not None
        and arguments.fixed_n_node
        <= 0
    ):
        raise ValueError(
            "fixed-n-node must be positive."
        )

    device = resolve_device(
        arguments.device
    )

    set_seed(
        seed
    )

    (
        split_path,
        autoencoder_path,
        flow_path,
        embedding_directory,
        output_directory,
    ) = resolve_paths(
        arguments,
        configuration,
    )

    normalization_path = (
        embedding_directory
        / "normalization.pt"
    )

    validate_paths(
        [
            split_path,
            autoencoder_path,
            flow_path,
            normalization_path,
        ]
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_payload = load_split_payload(
        split_path
    )
    training_records = list(
        split_payload["train"]
    )

    if not training_records:
        raise RuntimeError(
            "Training split is empty."
        )

    condition_mean = torch.as_tensor(
        split_payload[
            "condition_mean"
        ],
        dtype=torch.float32,
        device=device,
    )
    condition_std = torch.as_tensor(
        split_payload[
            "condition_std"
        ],
        dtype=torch.float32,
        device=device,
    )

    training_smiles = (
        canonical_training_smiles(
            record["smiles"]
            for record
            in training_records
        )
    )

    autoencoder_checkpoint = torch_load(
        autoencoder_path,
        map_location=device,
    )

    autoencoder = (
        build_autoencoder_from_config(
            configuration[
                "autoencoder"
            ]
        )
        .to(device)
    )

    autoencoder.load_state_dict(
        autoencoder_checkpoint[
            "autoencoder_state_dict"
        ],
        strict=True,
    )

    autoencoder.eval()

    for parameter in (
        autoencoder.parameters()
    ):
        parameter.requires_grad_(
            False
        )

    flow_checkpoint = torch_load(
        flow_path,
        map_location=device,
    )

    flow_configuration = dict(
        flow_checkpoint.get(
            "flow_configuration",
            configuration["flow"],
        )
    )

    flow_configuration.setdefault(
        "embedding_dim",
        int(
            configuration[
                "autoencoder"
            ]["embedding_dim"]
        ),
    )
    flow_configuration.setdefault(
        "condition_dim",
        2,
    )

    flow = (
        build_flow_from_config(
            flow_configuration
        )
        .to(device)
    )

    flow.load_state_dict(
        flow_checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    flow.eval()

    for parameter in flow.parameters():
        parameter.requires_grad_(
            False
        )

    normalization = torch_load(
        normalization_path,
        map_location=device,
    )

    embedding_mean = torch.as_tensor(
        normalization[
            "embedding_mean"
        ],
        dtype=torch.float32,
        device=device,
    )
    embedding_std = torch.as_tensor(
        normalization[
            "embedding_std"
        ],
        dtype=torch.float32,
        device=device,
    )

    rng = np.random.default_rng(
        seed
    )

    selected_indices = rng.integers(
        0,
        len(training_records),
        size=num_samples,
    )

    requested_qed = np.asarray(
        [
            float(
                training_records[
                    int(index)
                ]["qed"]
            )
            for index
            in selected_indices
        ],
        dtype=np.float32,
    )

    requested_n_node = np.asarray(
        [
            int(
                training_records[
                    int(index)
                ]["n_node"]
            )
            for index
            in selected_indices
        ],
        dtype=np.int64,
    )

    if arguments.fixed_qed is not None:
        requested_qed.fill(
            float(
                arguments.fixed_qed
            )
        )

    if arguments.fixed_n_node is not None:
        requested_n_node.fill(
            int(
                arguments.fixed_n_node
            )
        )

    torch_generator = torch.Generator(
        device=device
    )
    torch_generator.manual_seed(
        seed
    )

    print(
        "ZINC CONDITIONAL MOLECULAR GENERATION"
    )
    print(
        "Device:",
        device,
    )
    print(
        "Samples:",
        num_samples,
    )
    print(
        "Batch size:",
        batch_size,
    )
    print(
        "Temperature:",
        temperature,
    )
    print(
        "Training molecules:",
        len(training_records),
    )
    print(
        "Training canonical molecules:",
        len(training_smiles),
    )
    print(
        "QED target range:",
        round(
            float(
                requested_qed.min()
            ),
            4,
        ),
        "to",
        round(
            float(
                requested_qed.max()
            ),
            4,
        ),
    )
    print(
        "Node-count range:",
        int(
            requested_n_node.min()
        ),
        "to",
        int(
            requested_n_node.max()
        ),
    )
    print(
        "Autoencoder checkpoint:",
        autoencoder_path,
    )
    print(
        "Flow checkpoint:",
        flow_path,
    )
    print(
        "Test split accessed: no",
    )

    all_records: list[
        dict[str, Any]
    ] = []

    start_time = time.time()

    for start in range(
        0,
        num_samples,
        batch_size,
    ):
        end = min(
            start + batch_size,
            num_samples,
        )

        batch_qed = torch.as_tensor(
            requested_qed[
                start:end
            ],
            dtype=torch.float32,
            device=device,
        )
        batch_n_node = torch.as_tensor(
            requested_n_node[
                start:end
            ],
            dtype=torch.long,
            device=device,
        )

        records, _ = (
            generate_conditioned_batch(
                flow=flow,
                decoder=autoencoder.decoder,
                target_qed=batch_qed,
                n_node=batch_n_node,
                condition_mean=condition_mean,
                condition_std=condition_std,
                embedding_mean=embedding_mean,
                embedding_std=embedding_std,
                temperature=temperature,
                generator=torch_generator,
                training_smiles=training_smiles,
                sample_offset=start,
            )
        )

        all_records.extend(
            records
        )

        completed = end

        if (
            completed % 1000 == 0
            or completed
            == num_samples
        ):
            print(
                f"Generated "
                f"{completed}/"
                f"{num_samples} "
                f"| elapsed "
                f"{time.time() - start_time:.1f}s"
            )

    summaries = summarize_generation(
        all_records,
        target_tolerance=(
            target_tolerance
        ),
    )

    molecule_path = (
        output_directory
        / "generated_molecules.csv"
    )
    summary_csv_path = (
        output_directory
        / "generation_summary.csv"
    )
    result_json_path = (
        output_directory
        / "generation_results.json"
    )

    write_csv(
        molecule_path,
        all_records,
    )
    write_csv(
        summary_csv_path,
        summaries,
    )

    payload = {
        "configuration": {
            "seed": seed,
            "num_samples": num_samples,
            "batch_size": batch_size,
            "temperature": temperature,
            "target_tolerance": (
                target_tolerance
            ),
            "fixed_qed": (
                arguments.fixed_qed
            ),
            "fixed_n_node": (
                arguments.fixed_n_node
            ),
            "conditioning": (
                "joint empirical training "
                "QED and node count"
                if (
                    arguments.fixed_qed
                    is None
                    and arguments.fixed_n_node
                    is None
                )
                else "partially or fully fixed"
            ),
            "autoencoder_checkpoint": (
                str(autoencoder_path)
            ),
            "autoencoder_epoch": (
                autoencoder_checkpoint.get(
                    "stage_epoch"
                )
            ),
            "flow_checkpoint": (
                str(flow_path)
            ),
            "flow_epoch": (
                flow_checkpoint.get(
                    "epoch"
                )
            ),
            "test_split_accessed": False,
        },
        "summaries": summaries,
        "molecules": all_records,
    }

    write_json(
        result_json_path,
        payload,
    )

    print()
    print(
        "GENERATION RESULTS"
    )

    for summary in summaries:
        print()
        print(
            "Method:",
            summary["method"],
        )

        for key, value in (
            summary.items()
        ):
            if key == "method":
                continue
            print(
                f"  {key}: {value}"
            )

    print()
    print(
        "Saved molecules:",
        molecule_path,
    )
    print(
        "Saved summary:",
        summary_csv_path,
    )
    print(
        "Saved JSON:",
        result_json_path,
    )
    print(
        "Generation time:",
        round(
            time.time()
            - start_time,
            1,
        ),
        "seconds",
    )
    print(
        "Test split accessed: no"
    )
    print(
        "GENERATION COMPLETE"
    )


if __name__ == "__main__":
    main()
