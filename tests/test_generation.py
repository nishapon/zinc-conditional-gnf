
"""Tests for conditional ZINC molecular generation."""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from zinc_gnf.evaluation.generation import (
    actual_qed_from_smiles,
    canonical_training_smiles,
    decode_prediction_batch,
    generate_conditioned_batch,
    make_condition,
    make_condition_batch,
    safe_correlation,
    summarize_generation,
    symmetrize_bond_probabilities,
)


def test_make_condition_matches_manual_standardization():
    condition = make_condition(
        0.75,
        9,
        condition_mean=torch.tensor(
            [0.50, 2.0]
        ),
        condition_std=torch.tensor(
            [0.25, 0.5]
        ),
        device="cpu",
    )

    expected = torch.tensor(
        [
            1.0,
            (math.log1p(9) - 2.0) / 0.5,
        ],
        dtype=torch.float32,
    )

    assert condition.shape == (2,)
    assert torch.allclose(
        condition,
        expected,
    )


def test_make_condition_batch():
    target_qed = torch.tensor(
        [0.4, 0.8]
    )
    n_node = torch.tensor(
        [5, 10]
    )

    conditions = make_condition_batch(
        target_qed,
        n_node,
        condition_mean=torch.tensor(
            [0.6, 2.0]
        ),
        condition_std=torch.tensor(
            [0.2, 0.5]
        ),
    )

    expected = torch.stack(
        [
            (
                target_qed - 0.6
            ) / 0.2,
            (
                torch.log1p(
                    n_node.float()
                )
                - 2.0
            ) / 0.5,
        ],
        dim=-1,
    )

    assert conditions.shape == (2, 2)
    assert torch.allclose(
        conditions,
        expected,
    )


def test_invalid_condition_statistics_are_rejected():
    with pytest.raises(
        FloatingPointError
    ):
        make_condition(
            0.5,
            10,
            condition_mean=torch.zeros(2),
            condition_std=torch.tensor(
                [1.0, 0.0]
            ),
            device="cpu",
        )


def test_bond_probabilities_are_symmetric():
    generator = torch.Generator().manual_seed(
        91
    )

    logits = torch.randn(
        3,
        5,
        5,
        4,
        generator=generator,
    )

    probabilities = (
        symmetrize_bond_probabilities(
            logits
        )
    )

    assert probabilities.shape == (
        3,
        5,
        5,
        4,
    )
    assert torch.allclose(
        probabilities,
        probabilities.transpose(1, 2),
    )
    assert torch.allclose(
        probabilities.sum(dim=-1),
        torch.ones(3, 5, 5),
        atol=1e-6,
    )


def test_qed_and_canonical_training_smiles():
    qed = actual_qed_from_smiles(
        "CCO"
    )

    assert qed is not None
    assert 0.0 <= qed <= 1.0

    canonical = (
        canonical_training_smiles(
            [
                "OCC",
                "CCO",
                "CC",
                "invalid-smiles",
            ]
        )
    )

    assert "CCO" in canonical
    assert "CC" in canonical
    assert len(canonical) == 2


def ethane_predictions(
    batch_size=1,
    n_node=2,
):
    atom_logits = torch.full(
        (batch_size, n_node, 9),
        -10.0,
    )
    atom_logits[:, :, 0] = 10.0

    charge_logits = torch.full(
        (batch_size, n_node, 3),
        -10.0,
    )
    charge_logits[:, :, 1] = 10.0

    h_logits = torch.full(
        (batch_size, n_node, 5),
        -10.0,
    )
    h_logits[:, :, 3] = 10.0

    bond_logits = torch.full(
        (
            batch_size,
            n_node,
            n_node,
            4,
        ),
        -10.0,
    )

    bond_logits[:, :, :, 0] = 10.0

    for batch_index in range(
        batch_size
    ):
        bond_logits[
            batch_index,
            0,
            1,
            0,
        ] = -10.0
        bond_logits[
            batch_index,
            1,
            0,
            0,
        ] = -10.0
        bond_logits[
            batch_index,
            0,
            1,
            1,
        ] = 10.0
        bond_logits[
            batch_index,
            1,
            0,
            1,
        ] = 10.0

    return {
        "atom_logits": atom_logits,
        "charge_logits": charge_logits,
        "h_logits": h_logits,
        "bond_logits": bond_logits,
    }


def test_decode_prediction_batch_raw_and_constrained():
    predictions = ethane_predictions()

    records = decode_prediction_batch(
        predictions,
        torch.tensor(
            [[True, True]]
        ),
        requested_qed=torch.tensor(
            [0.4]
        ),
        n_node=torch.tensor(
            [2]
        ),
        training_smiles={"CC"},
    )

    assert len(records) == 2

    by_method = {
        record["method"]: record
        for record in records
    }

    assert set(by_method) == {
        "raw",
        "constrained",
    }

    for record in by_method.values():
        assert record["sanitizable"]
        assert record["connected"]
        assert record["domain_valid"]
        assert record["smiles"] == "CC"
        assert record["actual_qed"] is not None
        assert record["novel"] is False

    assert by_method[
        "constrained"
    ]["connected_by_decoder"] is True


class DummyFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(
            torch.zeros(())
        )

    def sample(
        self,
        condition,
        n_node,
        *,
        temperature=1.0,
        generator=None,
    ):
        del temperature
        del generator

        maximum_nodes = int(
            n_node.max().item()
        )

        mask = (
            torch.arange(
                maximum_nodes,
                device=condition.device,
            ).unsqueeze(0)
            < n_node.unsqueeze(1)
        )

        embeddings = torch.zeros(
            condition.shape[0],
            maximum_nodes,
            64,
            dtype=condition.dtype,
            device=condition.device,
        )

        embeddings = (
            embeddings
            + self.anchor
        )

        return embeddings, mask


class DummyDecoder(nn.Module):
    def forward(self, embeddings):
        batch_size = embeddings.shape[0]
        n_node = embeddings.shape[1]

        if n_node != 2:
            raise ValueError(
                "Dummy decoder expects two nodes."
            )

        predictions = ethane_predictions(
            batch_size=batch_size,
            n_node=n_node,
        )

        return {
            key: value.to(
                embeddings.device
            )
            for key, value
            in predictions.items()
        }


def test_end_to_end_generation_batch():
    flow = DummyFlow()
    decoder = DummyDecoder()

    records, tensors = (
        generate_conditioned_batch(
            flow=flow,
            decoder=decoder,
            target_qed=torch.tensor(
                [0.3, 0.7]
            ),
            n_node=torch.tensor(
                [2, 2]
            ),
            condition_mean=torch.tensor(
                [0.5, 1.0]
            ),
            condition_std=torch.tensor(
                [0.2, 0.5]
            ),
            embedding_mean=torch.zeros(
                64
            ),
            embedding_std=torch.ones(
                64
            ),
            temperature=1.0,
            training_smiles={"CC"},
        )
    )

    assert len(records) == 4

    assert tensors[
        "condition"
    ].shape == (2, 2)

    assert tensors[
        "normalized_embeddings"
    ].shape == (2, 2, 64)

    assert tensors[
        "raw_embeddings"
    ].shape == (2, 2, 64)

    assert tensors[
        "mask"
    ].shape == (2, 2)

    assert tensors[
        "mask"
    ].all()


def test_safe_correlation():
    correlation = safe_correlation(
        [0.1, 0.2, 0.3],
        [0.2, 0.4, 0.6],
    )

    assert correlation == pytest.approx(
        1.0
    )

    constant = safe_correlation(
        [1.0, 1.0],
        [0.2, 0.4],
    )

    assert math.isnan(constant)


def test_summarize_generation():
    records = [
        {
            "method": "raw",
            "sanitizable": False,
            "connected": False,
            "domain_valid": False,
            "smiles": None,
            "actual_qed": None,
            "requested_qed": 0.5,
            "novel": None,
        },
        {
            "method": "constrained",
            "sanitizable": True,
            "connected": True,
            "domain_valid": True,
            "smiles": "CC",
            "actual_qed": 0.45,
            "requested_qed": 0.5,
            "novel": False,
        },
        {
            "method": "constrained",
            "sanitizable": True,
            "connected": True,
            "domain_valid": True,
            "smiles": "CCC",
            "actual_qed": 0.75,
            "requested_qed": 0.8,
            "novel": True,
        },
    ]

    summaries = summarize_generation(
        records,
        target_tolerance=0.051,
    )

    by_method = {
        summary["method"]: summary
        for summary in summaries
    }

    assert by_method[
        "raw"
    ]["attempts"] == 1

    assert by_method[
        "raw"
    ]["domain_valid_count"] == 0

    constrained = by_method[
        "constrained"
    ]

    assert constrained[
        "attempts"
    ] == 2

    assert constrained[
        "domain_validity_percent"
    ] == pytest.approx(100.0)

    assert constrained[
        "uniqueness_percent"
    ] == pytest.approx(100.0)

    assert constrained[
        "novelty_percent"
    ] == pytest.approx(50.0)

    assert constrained[
        "qed_mae"
    ] == pytest.approx(0.05)

    assert constrained[
        "target_hit_percent_all"
    ] == pytest.approx(100.0)
