
"""Conditional molecular generation and evaluation for ZINC."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED

from zinc_gnf.chemistry import (
    canonical_connectivity_smiles,
    decode_constrained_molecule,
    inspect_raw_molecule,
)
from zinc_gnf.constants import (
    CHARGE_VALUES,
    HYDROGEN_VALUES,
)


def make_condition(
    qed: float,
    n_node: int,
    *,
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
    device: torch.device | str,
) -> torch.Tensor:
    """Create one standardized [QED, log1p(N)] condition."""
    if not math.isfinite(float(qed)):
        raise ValueError("QED must be finite.")
    if int(n_node) <= 0:
        raise ValueError("n_node must be positive.")

    mean = torch.as_tensor(
        condition_mean,
        dtype=torch.float32,
        device=device,
    )
    std = torch.as_tensor(
        condition_std,
        dtype=torch.float32,
        device=device,
    )

    if mean.shape != (2,) or std.shape != (2,):
        raise ValueError(
            "Condition statistics must have shape (2,)."
        )
    if not torch.isfinite(mean).all():
        raise FloatingPointError(
            "Condition mean is non-finite."
        )
    if (
        not torch.isfinite(std).all()
        or (std <= 0).any()
    ):
        raise FloatingPointError(
            "Condition standard deviation is invalid."
        )

    raw = torch.tensor(
        [
            float(qed),
            math.log1p(int(n_node)),
        ],
        dtype=torch.float32,
        device=device,
    )

    return (raw - mean) / std


def make_condition_batch(
    target_qed: torch.Tensor,
    n_node: torch.Tensor,
    *,
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
) -> torch.Tensor:
    """Create standardized [QED, log1p(N)] conditions."""
    target_qed = torch.as_tensor(
        target_qed,
        dtype=torch.float32,
    )
    n_node = torch.as_tensor(
        n_node,
        dtype=torch.long,
        device=target_qed.device,
    )

    if target_qed.ndim != 1 or n_node.ndim != 1:
        raise ValueError(
            "target_qed and n_node must have shape [B]."
        )
    if target_qed.shape != n_node.shape:
        raise ValueError(
            "target_qed and n_node batch sizes differ."
        )
    if not torch.isfinite(target_qed).all():
        raise FloatingPointError(
            "Target QED contains non-finite values."
        )
    if (n_node <= 0).any():
        raise ValueError(
            "Every graph must contain at least one node."
        )

    mean = torch.as_tensor(
        condition_mean,
        dtype=torch.float32,
        device=target_qed.device,
    )
    std = torch.as_tensor(
        condition_std,
        dtype=torch.float32,
        device=target_qed.device,
    )

    if mean.shape != (2,) or std.shape != (2,):
        raise ValueError(
            "Condition statistics must have shape (2,)."
        )
    if (
        not torch.isfinite(std).all()
        or (std <= 0).any()
    ):
        raise FloatingPointError(
            "Condition standard deviation is invalid."
        )

    raw = torch.stack(
        [
            target_qed,
            torch.log1p(n_node.float()),
        ],
        dim=-1,
    )

    return (raw - mean) / std


def symmetrize_bond_probabilities(
    bond_logits: torch.Tensor,
) -> torch.Tensor:
    """Convert pairwise bond logits to symmetric probabilities."""
    if bond_logits.ndim != 4:
        raise ValueError(
            "bond_logits must have shape [B, N, N, C]."
        )
    if bond_logits.shape[1] != bond_logits.shape[2]:
        raise ValueError(
            "Bond-logit node dimensions must be square."
        )

    probabilities = torch.softmax(
        bond_logits,
        dim=-1,
    )

    return 0.5 * (
        probabilities
        + probabilities.transpose(1, 2)
    )


def actual_qed_from_smiles(
    smiles: str | None,
) -> float | None:
    """Calculate RDKit QED for a valid SMILES string."""
    if not smiles:
        return None

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        return None

    try:
        value = float(QED.qed(molecule))
    except Exception:
        return None

    if not math.isfinite(value):
        return None

    return value


def canonical_training_smiles(
    smiles_values: Iterable[str],
) -> set[str]:
    """Canonicalize training SMILES for novelty evaluation."""
    canonical = set()

    for smiles in smiles_values:
        molecule = Chem.MolFromSmiles(
            str(smiles)
        )

        if molecule is None:
            continue

        canonical.add(
            canonical_connectivity_smiles(
                molecule
            )
        )

    return canonical


def decode_prediction_batch(
    predictions: Mapping[str, torch.Tensor],
    mask: torch.Tensor,
    *,
    requested_qed: torch.Tensor,
    n_node: torch.Tensor,
    training_smiles: set[str] | None = None,
    sample_offset: int = 0,
) -> list[dict[str, Any]]:
    """Decode one batch through raw and constrained decoders."""
    required = {
        "atom_logits",
        "charge_logits",
        "h_logits",
        "bond_logits",
    }
    missing = required.difference(
        predictions
    )

    if missing:
        raise KeyError(
            "Decoder output is missing: "
            + ", ".join(sorted(missing))
        )

    mask = torch.as_tensor(
        mask,
        dtype=torch.bool,
    )
    requested_qed = torch.as_tensor(
        requested_qed,
        dtype=torch.float32,
    )
    n_node = torch.as_tensor(
        n_node,
        dtype=torch.long,
    )

    batch_size = mask.shape[0]

    if requested_qed.shape != (batch_size,):
        raise ValueError(
            "requested_qed has an invalid shape."
        )
    if n_node.shape != (batch_size,):
        raise ValueError(
            "n_node has an invalid shape."
        )

    atom_prediction = predictions[
        "atom_logits"
    ].argmax(dim=-1)

    charge_prediction = predictions[
        "charge_logits"
    ].argmax(dim=-1)

    hydrogen_prediction = predictions[
        "h_logits"
    ].argmax(dim=-1)

    bond_probabilities = (
        symmetrize_bond_probabilities(
            predictions["bond_logits"]
        )
    )

    records: list[dict[str, Any]] = []

    for batch_index in range(batch_size):
        molecule_nodes = int(
            n_node[batch_index].item()
        )

        if molecule_nodes != int(
            mask[batch_index].sum().item()
        ):
            raise ValueError(
                "n_node does not match the generated mask."
            )

        atoms = (
            atom_prediction[
                batch_index,
                :molecule_nodes,
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )

        charge_classes = (
            charge_prediction[
                batch_index,
                :molecule_nodes,
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )

        hydrogen_classes = (
            hydrogen_prediction[
                batch_index,
                :molecule_nodes,
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )

        charges = np.asarray(
            [
                CHARGE_VALUES[int(index)]
                for index in charge_classes
            ],
            dtype=np.int64,
        )

        hydrogens = np.asarray(
            [
                HYDROGEN_VALUES[int(index)]
                for index
                in hydrogen_classes
            ],
            dtype=np.int64,
        )

        molecule_bond_probabilities = (
            bond_probabilities[
                batch_index,
                :molecule_nodes,
                :molecule_nodes,
            ]
            .detach()
            .cpu()
            .numpy()
        )

        raw_bonds = np.argmax(
            molecule_bond_probabilities,
            axis=-1,
        ).astype(np.int64)

        raw_bonds = np.triu(
            raw_bonds,
            k=1,
        )
        raw_bonds = (
            raw_bonds
            + raw_bonds.T
        )

        raw_result = inspect_raw_molecule(
            atoms,
            charges,
            hydrogens,
            raw_bonds,
        )

        constrained_result = (
            decode_constrained_molecule(
                atoms,
                charges,
                molecule_bond_probabilities,
            )
        )

        target = float(
            requested_qed[
                batch_index
            ].item()
        )
        sample_index = (
            int(sample_offset)
            + batch_index
        )

        for method, result in (
            ("raw", raw_result),
            (
                "constrained",
                constrained_result,
            ),
        ):
            smiles = result.get("smiles")
            actual_qed = (
                actual_qed_from_smiles(
                    smiles
                )
            )
            domain_valid = bool(
                result.get(
                    "domain_valid",
                    False,
                )
            )

            novel = None

            if (
                domain_valid
                and smiles is not None
                and training_smiles
                is not None
            ):
                novel = (
                    smiles
                    not in training_smiles
                )

            records.append({
                "sample_index": (
                    sample_index
                ),
                "method": method,
                "requested_qed": target,
                "n_node": molecule_nodes,
                "smiles": smiles,
                "sanitizable": bool(
                    result.get(
                        "sanitizable",
                        False,
                    )
                ),
                "connected": bool(
                    result.get(
                        "connected",
                        False,
                    )
                ),
                "radical_free": bool(
                    result.get(
                        "radical_free",
                        False,
                    )
                ),
                "domain_valid": (
                    domain_valid
                ),
                "novel": novel,
                "actual_qed": actual_qed,
                "qed_absolute_error": (
                    abs(
                        actual_qed
                        - target
                    )
                    if actual_qed
                    is not None
                    else None
                ),
                "error": str(
                    result.get(
                        "error",
                        "",
                    )
                ),
                "connected_by_decoder": (
                    result.get(
                        "connected_by_decoder"
                    )
                ),
                "tree_edges": (
                    result.get(
                        "tree_edges"
                    )
                ),
                "extra_edges": (
                    result.get(
                        "extra_edges"
                    )
                ),
                "upgraded_edges": (
                    result.get(
                        "upgraded_edges"
                    )
                ),
                "decoder_reason": str(
                    result.get(
                        "reason",
                        "",
                    )
                ),
            })

    return records


@torch.no_grad()
def generate_conditioned_batch(
    *,
    flow: torch.nn.Module,
    decoder: torch.nn.Module,
    target_qed: torch.Tensor,
    n_node: torch.Tensor,
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
    embedding_mean: torch.Tensor,
    embedding_std: torch.Tensor,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
    training_smiles: set[str] | None = None,
    sample_offset: int = 0,
) -> tuple[
    list[dict[str, Any]],
    dict[str, torch.Tensor],
]:
    """Generate and decode a genuine conditional-prior batch."""
    if temperature <= 0:
        raise ValueError(
            "temperature must be positive."
        )

    try:
        device = next(
            flow.parameters()
        ).device
    except StopIteration as error:
        raise ValueError(
            "The flow has no parameters."
        ) from error

    target_qed = torch.as_tensor(
        target_qed,
        dtype=torch.float32,
        device=device,
    )
    n_node = torch.as_tensor(
        n_node,
        dtype=torch.long,
        device=device,
    )

    condition = make_condition_batch(
        target_qed,
        n_node,
        condition_mean=torch.as_tensor(
            condition_mean,
            device=device,
        ),
        condition_std=torch.as_tensor(
            condition_std,
            device=device,
        ),
    )

    normalized_embeddings, mask = (
        flow.sample(
            condition,
            n_node,
            temperature=float(
                temperature
            ),
            generator=generator,
        )
    )

    embedding_mean = torch.as_tensor(
        embedding_mean,
        dtype=normalized_embeddings.dtype,
        device=device,
    )
    embedding_std = torch.as_tensor(
        embedding_std,
        dtype=normalized_embeddings.dtype,
        device=device,
    )

    if (
        embedding_mean.ndim != 1
        or embedding_std.shape
        != embedding_mean.shape
        or embedding_mean.shape[0]
        != normalized_embeddings.shape[-1]
    ):
        raise ValueError(
            "Embedding normalization dimensions are invalid."
        )

    raw_embeddings = (
        normalized_embeddings
        * embedding_std.view(
            1,
            1,
            -1,
        )
        + embedding_mean.view(
            1,
            1,
            -1,
        )
    )

    raw_embeddings = (
        raw_embeddings
        * mask.unsqueeze(-1).to(
            raw_embeddings.dtype
        )
    )

    predictions = decoder(
        raw_embeddings
    )

    records = decode_prediction_batch(
        predictions,
        mask,
        requested_qed=target_qed,
        n_node=n_node,
        training_smiles=(
            training_smiles
        ),
        sample_offset=sample_offset,
    )

    tensors = {
        "condition": (
            condition.detach()
        ),
        "normalized_embeddings": (
            normalized_embeddings.detach()
        ),
        "raw_embeddings": (
            raw_embeddings.detach()
        ),
        "mask": mask.detach(),
    }

    return records, tensors


def safe_correlation(
    first: Iterable[float],
    second: Iterable[float],
) -> float:
    """Pearson correlation with constant-array protection."""
    first_array = np.asarray(
        list(first),
        dtype=np.float64,
    )
    second_array = np.asarray(
        list(second),
        dtype=np.float64,
    )

    if (
        len(first_array) < 2
        or first_array.shape
        != second_array.shape
        or np.std(first_array) == 0
        or np.std(second_array) == 0
    ):
        return float("nan")

    return float(
        np.corrcoef(
            first_array,
            second_array,
        )[0, 1]
    )


def summarize_generation(
    records: list[Mapping[str, Any]],
    *,
    target_tolerance: float = 0.05,
) -> list[dict[str, Any]]:
    """Summarize raw and constrained molecular generation."""
    if target_tolerance <= 0:
        raise ValueError(
            "target_tolerance must be positive."
        )

    summaries = []

    for method in (
        "raw",
        "constrained",
    ):
        selected = [
            record
            for record in records
            if record["method"] == method
        ]

        if not selected:
            continue

        valid = [
            record
            for record in selected
            if record["domain_valid"]
            and record["smiles"]
            and record["actual_qed"]
            is not None
        ]

        valid_smiles = [
            str(record["smiles"])
            for record in valid
        ]
        unique_smiles = set(
            valid_smiles
        )

        requested = [
            float(
                record["requested_qed"]
            )
            for record in valid
        ]
        actual = [
            float(
                record["actual_qed"]
            )
            for record in valid
        ]

        errors = [
            abs(target - value)
            for target, value in zip(
                requested,
                actual,
            )
        ]

        novel_values = [
            bool(record["novel"])
            for record in valid
            if record["novel"]
            is not None
        ]

        attempts = len(selected)

        sanitizable_count = sum(
            bool(record["sanitizable"])
            for record in selected
        )
        connected_count = sum(
            bool(record["connected"])
            for record in selected
        )

        summaries.append({
            "method": method,
            "attempts": attempts,
            "sanitizable_count": (
                sanitizable_count
            ),
            "sanitizable_percent": (
                100.0
                * sanitizable_count
                / attempts
            ),
            "connected_count": (
                connected_count
            ),
            "connected_percent": (
                100.0
                * connected_count
                / attempts
            ),
            "domain_valid_count": (
                len(valid)
            ),
            "domain_validity_percent": (
                100.0
                * len(valid)
                / attempts
            ),
            "unique_valid": len(
                unique_smiles
            ),
            "uniqueness_percent": (
                100.0
                * len(unique_smiles)
                / len(valid)
                if valid
                else float("nan")
            ),
            "novelty_percent": (
                100.0
                * sum(novel_values)
                / len(novel_values)
                if novel_values
                else float("nan")
            ),
            "actual_qed_mean": (
                float(np.mean(actual))
                if actual
                else float("nan")
            ),
            "actual_qed_std": (
                float(np.std(actual))
                if actual
                else float("nan")
            ),
            "qed_mae": (
                float(np.mean(errors))
                if errors
                else float("nan")
            ),
            "qed_rmse": (
                float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                errors
                            )
                        )
                    )
                )
                if errors
                else float("nan")
            ),
            "requested_actual_correlation": (
                safe_correlation(
                    requested,
                    actual,
                )
            ),
            "target_hit_percent_all": (
                100.0
                * sum(
                    error
                    <= target_tolerance
                    for error in errors
                )
                / attempts
            ),
        })

    return summaries
