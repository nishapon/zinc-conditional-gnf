"""Tests for the bond-aware molecular autoencoder."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from zinc_gnf.chemistry import encode_zinc_smiles
from zinc_gnf.constants import (
    NUM_ATOM_TYPES,
    NUM_BOND_TYPES,
    NUM_CHARGE_TYPES,
    NUM_HYDROGEN_TYPES,
)
from zinc_gnf.data.datasets import collate_molecules
from zinc_gnf.models.autoencoder import (
    MolecularAutoencoder,
    MolecularEmbeddingQEDHead,
    build_autoencoder_from_config,
    make_node_noise,
)


def _record(
    smiles: str,
    condition: tuple[float, float] = (0.0, 0.0),
) -> dict:
    record, reason = encode_zinc_smiles(smiles)

    assert reason is None
    assert record is not None

    record["condition"] = np.asarray(
        condition,
        dtype=np.float32,
    )
    return record


def _batch() -> dict:
    return collate_molecules(
        [
            _record("CC", (-1.0, -0.5)),
            _record("CCO", (0.0, 0.0)),
            _record("c1ccccc1", (1.0, 0.5)),
        ]
    )


def _small_model() -> MolecularAutoencoder:
    torch.manual_seed(42)

    return MolecularAutoencoder(
        noise_dim=4,
        embedding_dim=16,
        hidden_dim=32,
        message_passing_steps=2,
        num_heads=4,
    )


def test_autoencoder_shapes_and_bond_symmetry():
    batch = _batch()
    model = _small_model()
    model.eval()

    generator = torch.Generator().manual_seed(11)
    noise = make_node_noise(
        batch,
        noise_dim=4,
        generator=generator,
    )

    with torch.no_grad():
        embeddings, predictions = model(
            batch,
            noise,
        )

    batch_size, padded_nodes = batch["node_mask"].shape

    assert embeddings.shape == (
        batch_size,
        padded_nodes,
        16,
    )
    assert predictions["atom_logits"].shape == (
        batch_size,
        padded_nodes,
        NUM_ATOM_TYPES,
    )
    assert predictions["charge_logits"].shape == (
        batch_size,
        padded_nodes,
        NUM_CHARGE_TYPES,
    )
    assert predictions["h_logits"].shape == (
        batch_size,
        padded_nodes,
        NUM_HYDROGEN_TYPES,
    )
    assert predictions["bond_logits"].shape == (
        batch_size,
        padded_nodes,
        padded_nodes,
        NUM_BOND_TYPES,
    )

    assert torch.isfinite(embeddings).all()

    for tensor in predictions.values():
        assert torch.isfinite(tensor).all()

    torch.testing.assert_close(
        predictions["bond_logits"],
        predictions["bond_logits"].transpose(1, 2),
        atol=1e-6,
        rtol=1e-5,
    )


def test_padding_and_noise_are_masked():
    batch = _batch()
    model = _small_model()
    model.eval()

    generator = torch.Generator().manual_seed(12)
    noise = make_node_noise(
        batch,
        noise_dim=4,
        generator=generator,
    )

    assert torch.all(
        noise[~batch["node_mask"]] == 0
    )

    with torch.no_grad():
        embeddings, _ = model(batch, noise)

    assert torch.all(
        embeddings[~batch["node_mask"]] == 0
    )


def test_encoder_rejects_wrong_noise_shape():
    batch = _batch()
    model = _small_model()

    wrong_noise = torch.zeros(
        batch["node_features"].shape[0],
        batch["node_features"].shape[1],
        3,
    )

    with pytest.raises(ValueError, match="noise shape"):
        model(batch, wrong_noise)


def test_autoencoder_is_atom_permutation_equivariant():
    original = collate_molecules(
        [_record("CCO")]
    )

    model = _small_model()
    model.eval()

    generator = torch.Generator().manual_seed(13)
    original_noise = make_node_noise(
        original,
        noise_dim=4,
        generator=generator,
    )

    permutation = torch.tensor(
        [2, 0, 1],
        dtype=torch.long,
    )

    permuted = {
        key: value
        for key, value in original.items()
    }

    for key in (
        "node_features",
        "atom_targets",
        "charge_targets",
        "h_targets",
        "node_mask",
    ):
        permuted[key] = original[key][:, permutation]

    for key in (
        "bond_targets",
        "pair_mask",
        "bond_loss_mask",
        "encoder_attention_mask",
        "flow_attention_mask",
    ):
        permuted[key] = (
            original[key][:, permutation][:, :, permutation]
        )

    permuted_noise = original_noise[:, permutation]

    with torch.no_grad():
        original_embeddings, original_predictions = model(
            original,
            original_noise,
        )
        permuted_embeddings, permuted_predictions = model(
            permuted,
            permuted_noise,
        )

    torch.testing.assert_close(
        permuted_embeddings,
        original_embeddings[:, permutation],
        atol=1e-5,
        rtol=1e-5,
    )

    for key in (
        "atom_logits",
        "charge_logits",
        "h_logits",
    ):
        torch.testing.assert_close(
            permuted_predictions[key],
            original_predictions[key][:, permutation],
            atol=1e-5,
            rtol=1e-5,
        )

    expected_bonds = (
        original_predictions["bond_logits"]
        [:, permutation]
        [:, :, permutation]
    )

    torch.testing.assert_close(
        permuted_predictions["bond_logits"],
        expected_bonds,
        atol=1e-5,
        rtol=1e-5,
    )


def test_qed_head_is_permutation_invariant_and_differentiable():
    torch.manual_seed(14)

    embeddings = torch.randn(
        2,
        5,
        16,
        requires_grad=True,
    )
    node_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, True],
        ]
    )

    qed_head = MolecularEmbeddingQEDHead(
        embedding_dim=16,
        hidden_dim=32,
        dropout=0.0,
    )
    qed_head.eval()

    permutation = torch.tensor(
        [4, 2, 0, 3, 1],
        dtype=torch.long,
    )

    original_qed = qed_head(
        embeddings,
        node_mask,
    )
    permuted_qed = qed_head(
        embeddings[:, permutation],
        node_mask[:, permutation],
    )

    assert original_qed.shape == (2,)
    assert torch.all(original_qed >= 0)
    assert torch.all(original_qed <= 1)

    torch.testing.assert_close(
        original_qed,
        permuted_qed,
        atol=1e-6,
        rtol=1e-6,
    )

    original_qed.sum().backward()

    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    assert embeddings.grad.abs().sum() > 0


def test_build_autoencoder_from_configuration():
    configuration = {
        "node_noise_dim": 8,
        "embedding_dim": 64,
        "hidden_dim": 256,
        "message_passing_steps": 6,
        "attention_heads": 4,
    }

    model = build_autoencoder_from_config(
        configuration
    )

    assert model.noise_dim == 8
    assert model.embedding_dim == 64
    assert len(model.encoder.blocks) == 6
    assert model.encoder.blocks[0].num_heads == 4
