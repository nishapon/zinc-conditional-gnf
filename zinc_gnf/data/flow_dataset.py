"""Lazy datasets for sharded molecular node embeddings."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import bisect
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
)

from zinc_gnf.data.embeddings import (
    unpack_molecule_embedding,
)
from zinc_gnf.data.sampling import (
    make_capped_qed_sampler,
)


def _torch_load(
    path: str | Path,
) -> Any:
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


class EmbeddingShardDataset(Dataset):
    """Lazy molecule-level view over embedding shards."""

    SPLIT_MANIFEST_KEYS = {
        "train": "training",
        "val": "validation",
    }

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        cache_size: int = 2,
    ) -> None:
        if split not in self.SPLIT_MANIFEST_KEYS:
            raise ValueError(
                "split must be 'train' or 'val'."
            )
        if cache_size <= 0:
            raise ValueError(
                "cache_size must be positive."
            )

        self.root = Path(root)
        self.split = split
        self.cache_size = int(cache_size)

        manifest_path = (
            self.root / "manifest.json"
        )
        normalization_path = (
            self.root / "normalization.pt"
        )

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Embedding manifest not found: "
                f"{manifest_path}"
            )
        if not normalization_path.exists():
            raise FileNotFoundError(
                f"Normalization file not found: "
                f"{normalization_path}"
            )

        with manifest_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            self.manifest = json.load(handle)

        split_key = self.SPLIT_MANIFEST_KEYS[
            split
        ]
        split_manifest = self.manifest[
            split_key
        ]

        self.shards = list(
            split_manifest["shards"]
        )

        if not self.shards:
            raise ValueError(
                f"No shards found for split {split}."
            )

        normalization = _torch_load(
            normalization_path
        )

        self.embedding_mean = torch.as_tensor(
            normalization["embedding_mean"],
            dtype=torch.float32,
        )
        self.embedding_std = torch.as_tensor(
            normalization["embedding_std"],
            dtype=torch.float32,
        )

        if (
            self.embedding_mean.ndim != 1
            or self.embedding_std.shape
            != self.embedding_mean.shape
        ):
            raise ValueError(
                "Invalid embedding normalization shapes."
            )
        if not torch.isfinite(
            self.embedding_mean
        ).all():
            raise FloatingPointError(
                "Embedding mean is non-finite."
            )
        if (
            not torch.isfinite(
                self.embedding_std
            ).all()
            or (self.embedding_std <= 0).any()
        ):
            raise FloatingPointError(
                "Embedding standard deviation is invalid."
            )

        self.embedding_dim = int(
            self.embedding_mean.shape[0]
        )

        self.cumulative_molecules = [0]

        for shard in self.shards:
            molecule_count = int(
                shard["molecules"]
            )
            if molecule_count <= 0:
                raise ValueError(
                    "A shard has no molecules."
                )

            self.cumulative_molecules.append(
                self.cumulative_molecules[-1]
                + molecule_count
            )

        expected_molecules = int(
            split_manifest["molecules"]
        )

        if (
            self.cumulative_molecules[-1]
            != expected_molecules
        ):
            raise ValueError(
                "Shard molecule counts do not match "
                "the embedding manifest."
            )

        self._cache: OrderedDict[
            int,
            Mapping[str, Any],
        ] = OrderedDict()

    def __len__(self) -> int:
        return self.cumulative_molecules[-1]

    def qed_values(self) -> torch.Tensor:
        values = []

        for shard_index, metadata in enumerate(
            self.shards
        ):
            shard = self._load_shard(
                shard_index
            )
            shard_values = torch.as_tensor(
                shard["qed"],
                dtype=torch.float64,
            ).flatten()

            expected = int(
                metadata["molecules"]
            )

            if shard_values.numel() != expected:
                raise ValueError(
                    "QED count does not match shard metadata."
                )

            values.append(shard_values)

        return torch.cat(values, dim=0)

    def _load_shard(
        self,
        shard_index: int,
    ) -> Mapping[str, Any]:
        if shard_index in self._cache:
            shard = self._cache.pop(
                shard_index
            )
            self._cache[shard_index] = shard
            return shard

        relative_path = Path(
            self.shards[
                shard_index
            ]["path"]
        )
        shard_path = self.root / relative_path

        if not shard_path.exists():
            raise FileNotFoundError(
                f"Embedding shard not found: "
                f"{shard_path}"
            )

        shard = _torch_load(shard_path)

        required = {
            "node_embeddings",
            "offsets",
            "condition",
            "n_node",
            "qed",
            "smiles",
        }
        missing = required - shard.keys()

        if missing:
            raise KeyError(
                "Embedding shard is missing: "
                + ", ".join(sorted(missing))
            )

        self._cache[shard_index] = shard

        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

        return shard

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        if index < 0:
            index += len(self)

        if index < 0 or index >= len(self):
            raise IndexError(index)

        shard_index = (
            bisect.bisect_right(
                self.cumulative_molecules,
                index,
            )
            - 1
        )
        local_index = (
            index
            - self.cumulative_molecules[
                shard_index
            ]
        )

        shard = self._load_shard(
            shard_index
        )

        embedding = unpack_molecule_embedding(
            shard,
            local_index,
            mean=self.embedding_mean,
            standard_deviation=(
                self.embedding_std
            ),
        )

        n_node = int(
            shard["n_node"][
                local_index
            ].item()
        )

        if embedding.shape != (
            n_node,
            self.embedding_dim,
        ):
            raise ValueError(
                "Unpacked embedding shape is invalid."
            )

        return {
            "h": embedding.float(),
            "condition": shard[
                "condition"
            ][local_index].float(),
            "n_node": n_node,
            "qed": float(
                shard["qed"][
                    local_index
                ].item()
            ),
            "smiles": str(
                shard["smiles"][
                    local_index
                ]
            ),
        }


def collate_flow_embeddings(
    records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Pad variable-size embeddings for conditional flow training."""
    if not records:
        raise ValueError(
            "Cannot collate an empty flow batch."
        )

    batch_size = len(records)
    maximum_nodes = max(
        int(record["n_node"])
        for record in records
    )
    embedding_dim = int(
        records[0]["h"].shape[-1]
    )

    embeddings = torch.zeros(
        batch_size,
        maximum_nodes,
        embedding_dim,
        dtype=torch.float32,
    )
    mask = torch.zeros(
        batch_size,
        maximum_nodes,
        dtype=torch.bool,
    )

    for index, record in enumerate(records):
        n_node = int(record["n_node"])
        molecular_embedding = torch.as_tensor(
            record["h"],
            dtype=torch.float32,
        )

        if molecular_embedding.shape != (
            n_node,
            embedding_dim,
        ):
            raise ValueError(
                "Invalid molecular embedding shape."
            )

        embeddings[
            index,
            :n_node,
        ] = molecular_embedding
        mask[
            index,
            :n_node,
        ] = True

    condition = torch.stack(
        [
            torch.as_tensor(
                record["condition"],
                dtype=torch.float32,
            )
            for record in records
        ],
        dim=0,
    )

    if condition.shape != (
        batch_size,
        2,
    ):
        raise ValueError(
            "Flow conditions must have shape [B, 2]."
        )

    return {
        "h": embeddings,
        "mask": mask,
        "condition": condition,
        "n_node": torch.tensor(
            [
                int(record["n_node"])
                for record in records
            ],
            dtype=torch.long,
        ),
        "qed": torch.tensor(
            [
                float(record["qed"])
                for record in records
            ],
            dtype=torch.float32,
        ),
        "smiles": [
            str(record["smiles"])
            for record in records
        ],
    }


def flow_batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device | str,
) -> dict[str, Any]:
    return {
        key: (
            value.to(device)
            if torch.is_tensor(value)
            else value
        )
        for key, value in batch.items()
    }


def make_flow_dataloaders(
    root: str | Path,
    *,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    cache_size: int = 2,
    qed_balanced_sampling: bool = False,
    qed_balance_bins: int = 20,
    qed_balance_power: float = 0.5,
    qed_balance_max_weight: float = 4.0,
) -> dict[str, DataLoader]:
    """Build train/validation flow loaders; never load test data."""
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )
    if num_workers < 0:
        raise ValueError(
            "num_workers cannot be negative."
        )

    train_dataset = EmbeddingShardDataset(
        root,
        split="train",
        cache_size=cache_size,
    )
    validation_dataset = (
        EmbeddingShardDataset(
            root,
            split="val",
            cache_size=cache_size,
        )
    )

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    common = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "collate_fn": (
            collate_flow_embeddings
        ),
        "pin_memory": bool(pin_memory),
    }

    if qed_balanced_sampling:
        sampler = make_capped_qed_sampler(
            train_dataset.qed_values(),
            bins=qed_balance_bins,
            power=qed_balance_power,
            max_weight=qed_balance_max_weight,
            generator=generator,
        )

        train_loader = DataLoader(
            train_dataset,
            shuffle=False,
            sampler=sampler,
            generator=generator,
            **common,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            generator=generator,
            **common,
        )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **common,
    )

    return {
        "train": train_loader,
        "val": validation_loader,
    }


__all__ = [
    "EmbeddingShardDataset",
    "collate_flow_embeddings",
    "flow_batch_to_device",
    "make_flow_dataloaders",
]
