"""Extraction and sharding of molecular node embeddings.

Embeddings are stored as concatenated valid nodes rather than padded
[B, max_nodes, embedding_dim] tensors. This reduces disk usage for the
full ZINC250K dataset.

Each shard contains:

    node_embeddings: [total_valid_nodes, embedding_dim]
    offsets:         [num_molecules + 1]
    condition:       [num_molecules, 2]
    n_node:          [num_molecules]
    qed:             [num_molecules]
    smiles:          list[str]

Normalization statistics are calculated only from training nodes.
Validation embeddings never influence the mean or standard deviation.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from zinc_gnf.data.datasets import batch_to_device
from zinc_gnf.models.autoencoder import make_node_noise


class RunningFeatureMoments:
    """Streaming per-feature mean and standard deviation."""

    def __init__(
        self,
        feature_dim: int,
    ) -> None:
        if feature_dim <= 0:
            raise ValueError(
                "feature_dim must be positive."
            )

        self.feature_dim = int(feature_dim)
        self.count = 0
        self.sum = torch.zeros(
            self.feature_dim,
            dtype=torch.float64,
        )
        self.square_sum = torch.zeros(
            self.feature_dim,
            dtype=torch.float64,
        )

    def update(
        self,
        values: torch.Tensor,
    ) -> None:
        values = values.detach().to(
            device="cpu",
            dtype=torch.float64,
        )

        if values.ndim != 2:
            raise ValueError(
                "RunningFeatureMoments expects a "
                "two-dimensional tensor."
            )
        if values.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected feature dimension "
                f"{self.feature_dim}, received "
                f"{values.shape[1]}."
            )
        if values.shape[0] == 0:
            return
        if not torch.isfinite(values).all():
            raise FloatingPointError(
                "Non-finite embedding values encountered."
            )

        self.count += int(values.shape[0])
        self.sum += values.sum(dim=0)
        self.square_sum += (
            values.square().sum(dim=0)
        )

    def finalize(
        self,
        *,
        minimum_standard_deviation: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count <= 0:
            raise ValueError(
                "Cannot finalize empty feature moments."
            )
        if minimum_standard_deviation <= 0:
            raise ValueError(
                "minimum_standard_deviation must be positive."
            )

        mean = self.sum / self.count
        variance = (
            self.square_sum / self.count
            - mean.square()
        ).clamp_min(0.0)

        standard_deviation = torch.sqrt(
            variance
        ).clamp_min(
            float(minimum_standard_deviation)
        )

        return (
            mean.float(),
            standard_deviation.float(),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "count": self.count,
            "sum": self.sum.clone(),
            "square_sum": self.square_sum.clone(),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
    ) -> "RunningFeatureMoments":
        result = cls(
            int(state["feature_dim"])
        )
        result.count = int(state["count"])
        result.sum = torch.as_tensor(
            state["sum"],
            dtype=torch.float64,
        ).clone()
        result.square_sum = torch.as_tensor(
            state["square_sum"],
            dtype=torch.float64,
        ).clone()
        return result


def atomic_torch_save(
    payload: Any,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )
    torch.save(payload, temporary)
    temporary.replace(path)


def pack_embedding_records(
    embeddings: list[torch.Tensor],
    conditions: list[torch.Tensor],
    qeds: list[float],
    smiles: list[str],
) -> dict[str, Any]:
    """Pack variable-sized molecular embeddings into one shard."""
    molecule_count = len(embeddings)

    if molecule_count == 0:
        raise ValueError(
            "Cannot create an empty embedding shard."
        )

    if not (
        len(conditions)
        == len(qeds)
        == len(smiles)
        == molecule_count
    ):
        raise ValueError(
            "Embedding shard fields have inconsistent lengths."
        )

    feature_dim = int(
        embeddings[0].shape[-1]
    )
    node_counts: list[int] = []

    for embedding in embeddings:
        if embedding.ndim != 2:
            raise ValueError(
                "Each molecular embedding must have shape "
                "[n_node, embedding_dim]."
            )
        if embedding.shape[0] <= 0:
            raise ValueError(
                "A molecule cannot have zero nodes."
            )
        if embedding.shape[1] != feature_dim:
            raise ValueError(
                "Embedding dimensions differ within a shard."
            )
        if not torch.isfinite(embedding).all():
            raise FloatingPointError(
                "Non-finite molecular embedding."
            )

        node_counts.append(
            int(embedding.shape[0])
        )

    n_node = torch.tensor(
        node_counts,
        dtype=torch.long,
    )
    offsets = torch.zeros(
        molecule_count + 1,
        dtype=torch.long,
    )
    offsets[1:] = torch.cumsum(
        n_node,
        dim=0,
    )

    condition = torch.stack(
        [
            value.detach().cpu().float()
            for value in conditions
        ],
        dim=0,
    )

    if condition.shape != (
        molecule_count,
        2,
    ):
        raise ValueError(
            "Conditions must have shape "
            f"{(molecule_count, 2)}, received "
            f"{tuple(condition.shape)}."
        )

    return {
        "node_embeddings": torch.cat(
            [
                value.detach().cpu().float()
                for value in embeddings
            ],
            dim=0,
        ),
        "offsets": offsets,
        "condition": condition,
        "n_node": n_node,
        "qed": torch.tensor(
            qeds,
            dtype=torch.float32,
        ),
        "smiles": list(smiles),
    }


def unpack_molecule_embedding(
    shard: Mapping[str, Any],
    index: int,
    *,
    mean: torch.Tensor | None = None,
    standard_deviation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Read one molecular embedding from a packed shard."""
    molecule_count = int(
        shard["n_node"].shape[0]
    )

    if index < 0 or index >= molecule_count:
        raise IndexError(index)

    offsets = shard["offsets"]
    start = int(offsets[index].item())
    end = int(offsets[index + 1].item())

    embedding = shard[
        "node_embeddings"
    ][start:end]

    if (
        mean is None
        and standard_deviation is None
    ):
        return embedding

    if (
        mean is None
        or standard_deviation is None
    ):
        raise ValueError(
            "Both mean and standard_deviation are required "
            "for normalization."
        )

    return (
        embedding
        - mean.to(embedding)
    ) / standard_deviation.to(embedding)


def _qed_summary(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
) -> dict[str, float]:
    prediction = torch.cat(
        predictions
    ).float()
    target = torch.cat(
        targets
    ).float()

    residual = prediction - target
    mse = float(
        residual.square().mean().item()
    )
    mae = float(
        residual.abs().mean().item()
    )

    centered_prediction = (
        prediction - prediction.mean()
    )
    centered_target = (
        target - target.mean()
    )

    denominator = torch.sqrt(
        centered_prediction.square().sum()
        * centered_target.square().sum()
    )

    correlation = (
        float(
            (
                centered_prediction
                * centered_target
            ).sum().item()
            / denominator.item()
        )
        if denominator.item() > 0
        else float("nan")
    )

    return {
        "qed_mae": mae,
        "qed_rmse": math.sqrt(mse),
        "qed_correlation": correlation,
    }


@torch.inference_mode()
def extract_embedding_split(
    autoencoder: nn.Module,
    qed_head: nn.Module,
    loader: DataLoader,
    *,
    output_directory: str | Path,
    split_name: str,
    device: torch.device | str,
    noise_dim: int,
    noise_seed: int,
    shard_size: int = 2048,
    moments: RunningFeatureMoments | None = None,
) -> dict[str, Any]:
    """Extract one split into deterministic embedding shards."""
    if shard_size <= 0:
        raise ValueError(
            "shard_size must be positive."
        )
    if not split_name:
        raise ValueError(
            "split_name cannot be empty."
        )

    resolved_device = torch.device(device)
    output_directory = Path(
        output_directory
    )
    split_directory = (
        output_directory / split_name
    )
    split_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    generator = torch.Generator(
        device=(
            "cuda"
            if resolved_device.type == "cuda"
            else "cpu"
        )
    )
    generator.manual_seed(
        int(noise_seed)
    )

    autoencoder.eval()
    qed_head.eval()

    buffered_embeddings: list[
        torch.Tensor
    ] = []
    buffered_conditions: list[
        torch.Tensor
    ] = []
    buffered_qeds: list[float] = []
    buffered_smiles: list[str] = []

    prediction_groups: list[
        torch.Tensor
    ] = []
    target_groups: list[
        torch.Tensor
    ] = []

    manifest_shards: list[
        dict[str, Any]
    ] = []

    molecule_total = 0
    node_total = 0
    shard_index = 0

    def flush() -> None:
        nonlocal shard_index

        if not buffered_embeddings:
            return

        payload = pack_embedding_records(
            buffered_embeddings,
            buffered_conditions,
            buffered_qeds,
            buffered_smiles,
        )

        filename = (
            f"{split_name}-{shard_index:05d}.pt"
        )
        shard_path = (
            split_directory / filename
        )

        atomic_torch_save(
            payload,
            shard_path,
        )

        manifest_shards.append(
            {
                "path": str(
                    Path(split_name) / filename
                ),
                "molecules": int(
                    payload["n_node"].shape[0]
                ),
                "nodes": int(
                    payload[
                        "node_embeddings"
                    ].shape[0]
                ),
            }
        )

        buffered_embeddings.clear()
        buffered_conditions.clear()
        buffered_qeds.clear()
        buffered_smiles.clear()
        shard_index += 1

    for raw_batch in loader:
        batch = batch_to_device(
            raw_batch,
            resolved_device,
        )

        noise = make_node_noise(
            batch,
            noise_dim=noise_dim,
            generator=generator,
        )

        node_embeddings = (
            autoencoder.encoder(
                batch,
                noise,
            )
        )
        qed_prediction = qed_head(
            node_embeddings,
            batch["node_mask"],
        )

        if not torch.isfinite(
            node_embeddings
        ).all():
            raise FloatingPointError(
                "Non-finite encoder output."
            )
        if not torch.isfinite(
            qed_prediction
        ).all():
            raise FloatingPointError(
                "Non-finite QED prediction."
            )

        prediction_groups.append(
            qed_prediction.detach().cpu()
        )
        target_groups.append(
            batch["qed"].detach().cpu()
        )

        batch_size = int(
            node_embeddings.shape[0]
        )

        for index in range(batch_size):
            n_node = int(
                batch["n_node"][index].item()
            )
            molecular_embedding = (
                node_embeddings[
                    index,
                    :n_node,
                ].detach().cpu().float()
            )

            if moments is not None:
                moments.update(
                    molecular_embedding
                )

            buffered_embeddings.append(
                molecular_embedding
            )
            buffered_conditions.append(
                batch["condition"][
                    index
                ].detach().cpu().float()
            )
            buffered_qeds.append(
                float(
                    batch["qed"][
                        index
                    ].item()
                )
            )
            buffered_smiles.append(
                str(
                    batch["smiles"][index]
                )
            )

            molecule_total += 1
            node_total += n_node

            if (
                len(buffered_embeddings)
                >= shard_size
            ):
                flush()

    flush()

    if molecule_total <= 0:
        raise ValueError(
            f"The {split_name} loader was empty."
        )

    result = {
        "split": split_name,
        "molecules": molecule_total,
        "nodes": node_total,
        "embedding_dim": int(
            autoencoder.encoder.embedding_dim
        ),
        "noise_seed": int(noise_seed),
        "shard_size": int(shard_size),
        "shards": manifest_shards,
    }
    result.update(
        _qed_summary(
            prediction_groups,
            target_groups,
        )
    )

    return result


__all__ = [
    "RunningFeatureMoments",
    "atomic_torch_save",
    "extract_embedding_split",
    "pack_embedding_records",
    "unpack_molecule_embedding",
]
