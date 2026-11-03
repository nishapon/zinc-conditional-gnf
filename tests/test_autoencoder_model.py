"""Tests for the dataset/collate, encoder, decoder, loss, constrained decoding
and symmetry ceiling of the charge-aware graph autoencoder. CPU only."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem, rdBase
import torch

from zinc_gnf.autoencoder import (
    AutoencoderConfig,
    GraphAutoencoder,
    evaluate_reconstruction,
    reconstruction_loss,
)
from zinc_gnf.chemistry_charge_aware import encode_zinc_smiles_charge_aware, graph_to_mol_charge_aware
from zinc_gnf.constants_charge_aware import ATOM_VALENCE, NUM_ATOM_TYPES
from zinc_gnf.dataset import (
    DatasetFormatError,
    DeterministicBatcher,
    MolecularGraphDataset,
    atom_one_hot,
    collate_molecules,
    iter_eval_batches,
    make_batch,
    seeded_subset_indices,
)
from zinc_gnf.decoding_charge_aware import valence_aware_bond_decode_typed
from zinc_gnf.symmetry import (
    has_indistinguishable_atoms,
    is_decodable,
    reconstruction_ceiling,
    refine_colors,
)

SMILES = [
    "CCO", "c1ccccc1", "CC(=O)Nc1ccc(O)cc1", "C[NH3+]", "CC(=O)[O-]", "C[N+](=O)[O-]",
    "c1cc[nH+]cc1", "[NH3+]CC(=O)[O-]", "C[S+](C)C", "CN=[N+]=[N-]", "Fc1ccccc1",
    "O=C(Nc1ccc(F)cc1)c1cc(Cl)ccc1Br", "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "c1ccncc1", "CCN", "CCCO",
]


def make_records(smiles=SMILES):
    records = []
    for index, s in enumerate(smiles):
        record, reason = encode_zinc_smiles_charge_aware(s, row_id=index)
        assert record is not None, (s, reason)
        record["condition"] = np.array([0.1 * index, -0.1 * index], dtype=np.float32)
        records.append(record)
    return records


@pytest.fixture(scope="module")
def dataset():
    return MolecularGraphDataset(make_records())


@pytest.fixture
def tiny_config():
    return AutoencoderConfig(node_embedding_dim=16, latent_dim=32, num_processing_steps=2,
                             attn_kq_dim=8, attn_v_dim=8, attn_concat_heads_output_dim=8)


# ===========================================================================
# Dataset and collate.
# ===========================================================================
def test_dataset_basics_and_compact_storage(dataset):
    assert len(dataset) == len(SMILES)
    assert dataset._atom_types[0].dtype == np.int8 and dataset._bond_matrix[0].dtype == np.int8
    item = dataset[3]
    assert item["n_node"] == 2 and item["row_id"] == 3 and item["condition"].shape == (2,)
    atoms, bonds = dataset.true_graph(3)
    assert atoms.dtype == np.int64 and bonds.dtype == np.int64
    assert dataset.smiles[3] == encode_zinc_smiles_charge_aware("C[NH3+]")[0]["smiles"]
    assert dataset.qed.shape == (len(SMILES),) and dataset.n_node[3] == 2


def test_collate_pads_and_masks(dataset):
    batch = collate_molecules([dataset[i] for i in (0, 11, 3)])      # n = 3, 18?, 2
    n = [dataset[i]["n_node"] for i in (0, 11, 3)]
    width = max(n)
    assert batch.atom_types.shape == (3, width) and batch.bond_types.shape == (3, width, width)
    assert batch.node_mask.sum(1).tolist() == n and batch.n_node.tolist() == n
    for b, size in enumerate(n):
        assert batch.node_mask[b, :size].all() and not batch.node_mask[b, size:].any()
        assert (batch.atom_types[b, size:] == 0).all() and (batch.bond_types[b, size:, :] == 0).all()
        assert (batch.bond_types[b, :, size:] == 0).all()
        # pair mask: real ordered pairs, no diagonal, no padding
        assert batch.pair_mask[b].sum() == size * (size - 1)
        assert not batch.pair_mask[b].diagonal().any()
        assert not batch.pair_mask[b, size:, :].any() and not batch.pair_mask[b, :, size:].any()
    assert batch.condition.shape == (3, 2) and batch.qed.shape == (3,)
    assert torch.equal(batch.bond_types, batch.bond_types.transpose(1, 2))
    with pytest.raises(ValueError):
        collate_molecules([])


def test_one_hot_is_atomic_identity_and_zero_on_padding(dataset):
    batch = make_batch(dataset, [0, 11])
    one_hot = atom_one_hot(batch.atom_types, batch.node_mask)
    assert one_hot.shape == (2, batch.max_nodes, NUM_ATOM_TYPES) and NUM_ATOM_TYPES == 17
    assert one_hot.sum(-1).tolist() == batch.node_mask.to(torch.float32).tolist()
    real = batch.node_mask
    assert torch.equal(one_hot.argmax(-1)[real], batch.atom_types[real])
    # Charged atoms are distinct types, not their neutral element.
    ammonium = make_batch(dataset, [3])
    types = ammonium.atom_types[0, :2].tolist()
    assert 9 in types                                         # N+1 is type index 9


def test_dataset_validation_rejects_malformed_records():
    good = make_records(["CCO"])[0]
    bad_range = dict(good, atom_types=np.array([0, 0, 99]))
    with pytest.raises(DatasetFormatError, match="atom type"):
        MolecularGraphDataset([bad_range])
    asym = dict(good, bond_matrix=np.triu(good["bond_matrix"]))
    with pytest.raises(DatasetFormatError, match="symmetric"):
        MolecularGraphDataset([asym])
    with pytest.raises(DatasetFormatError, match="at least one"):
        MolecularGraphDataset([])
    too_big = dict(good, atom_types=np.zeros(39, dtype=np.int64), bond_matrix=np.zeros((39, 39), dtype=np.int64))
    with pytest.raises(DatasetFormatError, match="bad atom vector"):
        MolecularGraphDataset([too_big])


def test_eval_batches_cover_everything_in_order(dataset):
    rows = [r for b in iter_eval_batches(dataset, 5) for r in b.row_id.tolist()]
    assert rows == list(range(len(dataset)))
    assert [r for b in iter_eval_batches(dataset, 4, [7, 3, 9]) for r in b.row_id.tolist()] == [7, 3, 9]


def test_batcher_is_deterministic_resumable_and_covers_epochs():
    a = DeterministicBatcher(20, 4, seed=5)
    seen = [a.next_indices().tolist() for _ in range(5)]
    assert sorted(i for b in seen for i in b) == list(range(20))       # an epoch covers all
    assert a.epoch == 0
    sixth = a.next_indices().tolist()
    assert a.epoch == 1                                                # new permutation
    assert sixth != seen[0]                                            # epoch 1 order differs from epoch 0
    b = DeterministicBatcher(20, 4, seed=5)
    assert [b.next_indices().tolist() for _ in range(5)] == seen        # same seed -> same order
    c = DeterministicBatcher(20, 4, seed=6)
    assert c.next_indices().tolist() != seen[0]

    resumed = DeterministicBatcher(20, 4, seed=5)
    for _ in range(7):
        resumed.next_indices()
    state = resumed.state_dict()
    fresh = DeterministicBatcher(20, 4, seed=5)
    fresh.load_state_dict(state)
    assert [fresh.next_indices().tolist() for _ in range(4)] == [resumed.next_indices().tolist() for _ in range(4)]
    with pytest.raises(ValueError, match="mismatch"):
        DeterministicBatcher(20, 4, seed=9).load_state_dict(state)
    with pytest.raises(ValueError):
        DeterministicBatcher(3, 4, seed=1)


def test_seeded_subset_indices():
    a = seeded_subset_indices(100, 10, 1)
    assert a == seeded_subset_indices(100, 10, 1) and a == sorted(a) and len(set(a)) == 10
    assert a != seeded_subset_indices(100, 10, 2)
    assert seeded_subset_indices(5, None, 1) == list(range(5)) == seeded_subset_indices(5, 50, 1)


# ===========================================================================
# Architecture properties.
# ===========================================================================
def test_architecture_matches_teammates_baseline_and_17_types():
    config = AutoencoderConfig()
    assert config.node_embedding_dim == 64 and config.latent_dim == 256
    assert config.num_processing_steps == 6 and config.num_mlp_layers == 2
    assert (config.attn_kq_dim, config.attn_v_dim, config.attn_num_heads,
            config.attn_concat_heads_output_dim) == (32, 32, 2, 32)
    assert config.weight_sharing and not config.residual
    assert config.bond_class_weight == 5.0 and config.num_bond_refine_steps == 1
    assert config.num_atom_types == 17 and config.symmetry_noise_dim == 0
    model = GraphAutoencoder(config)
    assert model.encoder.embed[0].in_features == 17          # one-hot identity only, no noise
    assert model.decoder.atom_head[-1].out_features == 17
    assert model.decoder.bond_head[0].in_features == 3 * 64
    assert len(model.encoder.blocks) == 1                      # shared across the 6 steps
    assert model.encoder.blocks[0].logit_bias.shape == (4, 2)
    assert model.encoder.blocks[0].value_bias.shape == (4, 2, 32)
    assert 100_000 < model.num_parameters() < 400_000


def test_forward_shapes_and_padding_is_zeroed(dataset, tiny_config):
    model = GraphAutoencoder(tiny_config).eval()
    batch = make_batch(dataset, [0, 11, 3, 12])
    out = model(batch)
    b, n = batch.atom_types.shape
    assert out["nodes"].shape == (b, n, 16) and out["atom_logits"].shape == (b, n, 17)
    assert out["bond_logits"].shape == (b, n, n, 4)
    assert torch.isfinite(out["nodes"]).all() and torch.isfinite(out["bond_logits"]).all()
    assert (out["nodes"][~batch.node_mask] == 0).all()
    assert torch.allclose(out["bond_logits"], out["bond_logits"].transpose(1, 2), atol=1e-6)   # symmetric


def test_padding_does_not_change_real_outputs(dataset, tiny_config):
    torch.manual_seed(0)
    model = GraphAutoencoder(tiny_config).eval()
    alone = make_batch(dataset, [3])                       # n = 2
    padded = make_batch(dataset, [3, 11])                  # padded to the larger molecule
    out_a, out_p = model(alone), model(padded)
    n = int(alone.n_node[0])
    assert torch.allclose(out_a["nodes"][0, :n], out_p["nodes"][0, :n], atol=1e-5)
    assert torch.allclose(out_a["atom_logits"][0, :n], out_p["atom_logits"][0, :n], atol=1e-5)
    assert torch.allclose(out_a["bond_logits"][0, :n, :n], out_p["bond_logits"][0, :n, :n], atol=1e-5)


def test_encoder_is_permutation_equivariant(dataset, tiny_config):
    torch.manual_seed(1)
    model = GraphAutoencoder(tiny_config).eval()
    batch = make_batch(dataset, [11])
    n = int(batch.n_node[0])
    permutation = torch.randperm(n)
    permuted = make_batch(dataset, [11])
    permuted.atom_types[0, :n] = batch.atom_types[0, :n][permutation]
    permuted.bond_types[0, :n, :n] = batch.bond_types[0, :n, :n][permutation][:, permutation]
    base, perm = model(batch), model(permuted)
    assert torch.allclose(base["nodes"][0, :n][permutation], perm["nodes"][0, :n], atol=1e-5)
    assert torch.allclose(base["bond_logits"][0, :n, :n][permutation][:, permutation],
                          perm["bond_logits"][0, :n, :n], atol=1e-5)


def test_bond_type_changes_the_embedding(dataset, tiny_config):
    """Bond-aware attention: the same connectivity with different bond orders must differ."""
    torch.manual_seed(2)
    model = GraphAutoencoder(tiny_config).eval()
    with torch.no_grad():
        model.encoder.blocks[0].logit_bias.normal_()
        model.encoder.blocks[0].value_bias.normal_()
    ethanol = make_batch(dataset, [0])
    other = make_batch(dataset, [0])
    other.bond_types[0, 0, 1] = other.bond_types[0, 1, 0] = 2              # single -> double
    assert not torch.allclose(model(ethanol)["nodes"], model(other)["nodes"], atol=1e-5)


def test_atom_identity_matters_including_charge(dataset, tiny_config):
    torch.manual_seed(3)
    model = GraphAutoencoder(tiny_config).eval()
    neutral = make_batch(dataset, [14])                     # CCN
    charged = make_batch(dataset, [14])
    charged.atom_types[0, neutral.atom_types[0] == 1] = 9   # neutral N -> N+1
    assert not torch.allclose(model(neutral)["nodes"], model(charged)["nodes"], atol=1e-6)


def test_symmetry_noise_is_optional_extra_channels_not_a_substitute(dataset):
    config = AutoencoderConfig(node_embedding_dim=16, latent_dim=32, num_processing_steps=1,
                               attn_kq_dim=8, attn_v_dim=8, attn_concat_heads_output_dim=8,
                               symmetry_noise_dim=4)
    model = GraphAutoencoder(config).eval()
    assert model.encoder.embed[0].in_features == 17 + 4            # identity kept, noise added
    batch = make_batch(dataset, [1])
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(2)
    a, b = model.encode(batch, generator=g1), model.encode(batch, generator=g2)
    assert not torch.allclose(a, b)                                 # noise breaks ties
    c = model.encode(batch, generator=torch.Generator().manual_seed(1))
    assert torch.allclose(a, c)                                     # but is reproducible
    plain = GraphAutoencoder(AutoencoderConfig(node_embedding_dim=16, latent_dim=32,
                                               num_processing_steps=1, attn_kq_dim=8, attn_v_dim=8,
                                               attn_concat_heads_output_dim=8)).eval()
    assert torch.allclose(plain.encode(batch), plain.encode(batch))  # default: deterministic


def test_refinement_runs_and_uses_typed_valence(dataset):
    config = AutoencoderConfig(node_embedding_dim=16, latent_dim=32, num_processing_steps=1,
                               attn_kq_dim=8, attn_v_dim=8, attn_concat_heads_output_dim=8,
                               num_bond_refine_steps=3)
    model = GraphAutoencoder(config)
    assert len(model.decoder.update_mlps) == 2
    batch = make_batch(dataset, [11, 3])
    free = model(batch, teacher_forcing=False)
    forced = model(batch, teacher_forcing=True)
    assert torch.isfinite(free["bond_logits"]).all() and torch.isfinite(forced["bond_logits"]).all()
    assert torch.equal(forced["atoms_used"], batch.atom_types)
    assert model.decoder.valence.tolist() == [float(v) for v in ATOM_VALENCE]
    single = GraphAutoencoder(AutoencoderConfig(node_embedding_dim=16, latent_dim=32,
                                                num_processing_steps=1, attn_kq_dim=8, attn_v_dim=8,
                                                attn_concat_heads_output_dim=8))
    assert len(single.decoder.update_mlps) == 0


def test_config_roundtrip_and_unknown_keys():
    config = AutoencoderConfig(symmetry_noise_dim=3, node_embedding_dim=48)
    assert AutoencoderConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="unknown autoencoder settings"):
        AutoencoderConfig.from_dict({"nope": 1})


# ===========================================================================
# Loss.
# ===========================================================================
def test_loss_matches_manual_computation_and_ignores_padding(dataset, tiny_config):
    torch.manual_seed(4)
    model = GraphAutoencoder(tiny_config)
    batch = make_batch(dataset, [3, 0])
    out = model(batch)
    losses = model.reconstruction_loss(batch, out)

    import torch.nn.functional as F
    atom_ce = F.cross_entropy(out["atom_logits"].reshape(-1, 17), batch.atom_types.reshape(-1),
                              reduction="none").view_as(batch.atom_types)
    expected_atom = (atom_ce * batch.node_mask).sum()
    bond_ce = F.cross_entropy(out["bond_logits"].reshape(-1, 4), batch.bond_types.reshape(-1),
                              reduction="none").view_as(batch.bond_types)
    weight = torch.tensor([1.0, 5.0, 5.0, 5.0])[batch.bond_types]
    expected_bond = (bond_ce * weight * batch.pair_mask).sum()
    assert torch.allclose(losses["atom_loss"], expected_atom)
    assert torch.allclose(losses["bond_loss"], expected_bond)
    assert torch.allclose(losses["total_loss"], expected_atom + expected_bond)
    assert torch.allclose(losses["mean_loss"], losses["total_loss"] / batch.node_mask.sum())
    assert losses["n_nodes"].item() == int(batch.node_mask.sum())


def test_padding_cannot_leak_into_the_loss(dataset, tiny_config):
    torch.manual_seed(5)
    model = GraphAutoencoder(tiny_config)
    batch = make_batch(dataset, [3, 11])
    out = model(batch)
    base = model.reconstruction_loss(batch, out)["total_loss"].item()
    corrupted = {k: v.clone() for k, v in out.items() if isinstance(v, torch.Tensor)}
    corrupted["atom_logits"][~batch.node_mask] += 1000.0
    corrupted["bond_logits"][~batch.pair_mask] += 1000.0
    assert reconstruction_loss(batch, corrupted, bond_class_weight=5.0)["total_loss"].item() == pytest.approx(base)


def test_real_bond_weight_is_applied(dataset, tiny_config):
    torch.manual_seed(6)
    model = GraphAutoencoder(tiny_config)
    batch = make_batch(dataset, [11])
    out = model(batch)
    w1 = reconstruction_loss(batch, out, bond_class_weight=1.0)["bond_loss"]
    w5 = reconstruction_loss(batch, out, bond_class_weight=5.0)["bond_loss"]
    assert w5 > w1


def test_perfect_logits_give_zero_loss_and_full_accuracy(dataset, tiny_config):
    batch = make_batch(dataset, [11, 3])
    big = 50.0
    out = {
        "atom_logits": torch.nn.functional.one_hot(batch.atom_types, 17).float() * big,
        "bond_logits": torch.nn.functional.one_hot(batch.bond_types, 4).float() * big,
    }
    losses = reconstruction_loss(batch, out, bond_class_weight=5.0)
    assert losses["total_loss"].item() < 1e-6
    assert losses["atom_accuracy"].item() == 1.0 and losses["bond_accuracy"].item() == 1.0
    assert losses["real_bond_accuracy"].item() == 1.0


def test_gradients_flow_to_every_parameter(dataset, tiny_config):
    config = AutoencoderConfig(**{**tiny_config.to_dict(), "num_bond_refine_steps": 2})
    model = GraphAutoencoder(config)
    batch = make_batch(dataset, [11, 3, 12])
    model.reconstruction_loss(batch, model(batch))["mean_loss"].backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert missing == []
    assert model.encoder.blocks[0].logit_bias.grad.abs().sum() > 0       # bond-aware biases are trained


def test_overfitting_one_batch_reaches_exact_reconstruction(dataset):
    torch.manual_seed(0)
    config = AutoencoderConfig(node_embedding_dim=32, latent_dim=64, num_processing_steps=3,
                               attn_kq_dim=16, attn_v_dim=16, attn_concat_heads_output_dim=16)
    model = GraphAutoencoder(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    indices = [0, 3, 4, 5, 6, 7, 14, 15]          # small, asymmetric-enough molecules
    batch = make_batch(dataset, indices)
    first = None
    for step in range(400):
        loss = model.reconstruction_loss(batch, model(batch))["mean_loss"]
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.2 * first
    result = evaluate_reconstruction(model, dataset, indices=indices, batch_size=4)
    assert result["atom_accuracy"] > 0.95
    assert result["constrained"]["validity"] == 1.0
    assert result["constrained"]["exact_molecule_match"] >= 0.75


# ===========================================================================
# Constrained decoding.
# ===========================================================================
def one_hot_probs(bonds: np.ndarray) -> np.ndarray:
    return np.eye(4)[bonds]


def test_constrained_decode_reproduces_confident_correct_predictions():
    for smiles in ("CCO", "CC(=O)[O-]", "C[N+](=O)[O-]", "c1cc[nH+]cc1", "CN=[N+]=[N-]"):
        record, _ = encode_zinc_smiles_charge_aware(smiles)
        bonds, info = valence_aware_bond_decode_typed(record["atom_types"], one_hot_probs(record["bond_matrix"]))
        assert np.array_equal(bonds, record["bond_matrix"]), smiles
        assert info["connected"] and info["rescue_edges"] == 0


def test_constrained_decode_never_exceeds_typed_valence():
    rng = np.random.default_rng(0)
    record, _ = encode_zinc_smiles_charge_aware("O=C(Nc1ccc(F)cc1)c1cc(Cl)ccc1Br")
    atoms = record["atom_types"]
    caps = np.array([ATOM_VALENCE[a] for a in atoms])
    for _ in range(20):
        probs = rng.dirichlet(np.ones(4), size=(len(atoms), len(atoms)))
        bonds, info = valence_aware_bond_decode_typed(atoms, probs)
        assert (bonds.sum(1) <= caps).all()
        assert np.array_equal(bonds, bonds.T) and (np.diag(bonds) == 0).all()


def test_constrained_decode_respects_charged_caps():
    atoms = np.array([10, 0, 0])                       # O-1 (cap 1) with two carbons
    probs = np.zeros((3, 3, 4)); probs[..., 3] = 1.0   # asks for triple bonds everywhere
    bonds, _ = valence_aware_bond_decode_typed(atoms, probs)
    assert bonds[0].sum() <= 1


def test_constrained_decode_makes_graph_connected_when_valence_allows():
    atoms = np.array([0, 0, 0, 0])                     # four carbons
    probs = np.zeros((4, 4, 4)); probs[..., 0] = 0.9; probs[..., 1] = 0.1   # argmax: no bonds at all
    bonds, info = valence_aware_bond_decode_typed(atoms, probs)
    assert info["connected"] and info["rescue_edges"] == 3
    assert (bonds > 0).sum() // 2 == 3


def test_constrained_decode_reports_disconnected_when_valence_exhausted():
    atoms = np.array([3, 3])                           # two fluorines already saturated?  F cap 1
    probs = np.zeros((2, 2, 4)); probs[..., 0] = 1.0
    bonds, info = valence_aware_bond_decode_typed(atoms, probs)
    assert info["connected"] and bonds[0, 1] == 1      # F-F can still join once
    three = np.array([3, 3, 3])                         # three fluorines cannot all connect
    p3 = np.zeros((3, 3, 4)); p3[..., 0] = 1.0
    bonds3, info3 = valence_aware_bond_decode_typed(three, p3)
    assert not info3["connected"]


def test_constrained_decode_input_validation():
    with pytest.raises(ValueError, match="shape"):
        valence_aware_bond_decode_typed(np.array([0, 0]), np.zeros((3, 3, 4)))
    bad = np.zeros((2, 2, 4)); bad[0, 1, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        valence_aware_bond_decode_typed(np.array([0, 0]), bad)


def test_decoded_graphs_convert_to_valid_molecules():
    record, _ = encode_zinc_smiles_charge_aware("[NH3+]CC(=O)[O-]")
    bonds, _ = valence_aware_bond_decode_typed(record["atom_types"], one_hot_probs(record["bond_matrix"]))
    with rdBase.BlockLogs():
        mol = graph_to_mol_charge_aware(record["atom_types"], bonds)
    assert len(Chem.GetMolFrags(mol)) == 1


# ===========================================================================
# Symmetry ceiling.
# ===========================================================================
def graph(smiles):
    r, _ = encode_zinc_smiles_charge_aware(smiles)
    return r["atom_types"], r["bond_matrix"]


def test_symmetry_ceiling_known_cases():
    assert is_decodable(*graph("CCO"))                              # all atoms distinguishable
    assert is_decodable(*graph("CC(=O)[O-]"))
    assert not is_decodable(*graph("c1ccccc1"))                     # benzene: all six carbons equal
    assert not is_decodable(*graph("Fc1ccc(F)cc1"))                 # ortho/meta carbons equivalent
    assert is_decodable(*graph("C[NH3+]"))
    assert has_indistinguishable_atoms(*graph("CC(C)C"))            # the methyl groups
    assert not has_indistinguishable_atoms(*graph("CCO"))


def test_symmetry_colours_use_bond_orders_and_charge():
    a, b = graph("C=CC")                                            # double vs single bond differ
    colours = refine_colors(a, b)
    assert len(set(colours)) == 3
    # Acetate's two oxygens differ only by bond order and charge: they must stay distinguishable.
    acetate_atoms, acetate_bonds = graph("CC(=O)[O-]")
    assert len(set(refine_colors(acetate_atoms, acetate_bonds))) == 4
    assert is_decodable(acetate_atoms, acetate_bonds)
    # A charge-carrying atom type is a different colour from its neutral element.
    assert graph("CC[NH3+]")[0].tolist() != graph("CCN")[0].tolist()
    assert reconstruction_ceiling([graph("CCO"), graph("c1ccccc1")]) == 0.5
    assert np.isnan(reconstruction_ceiling([]))


def test_ceiling_is_an_upper_bound_for_a_symmetric_model(dataset, tiny_config):
    """An untrained-but-symmetric model gives equal embeddings to equivalent atoms,
    so benzene cannot be decoded exactly regardless of the decoder."""
    torch.manual_seed(0)
    model = GraphAutoencoder(tiny_config).eval()
    batch = make_batch(dataset, [1])                                # benzene
    nodes = model.encode(batch)[0, :6]
    assert torch.allclose(nodes, nodes[0].expand_as(nodes), atol=1e-5)    # all six identical
    # while ethanol's atoms are all distinct
    ethanol = model.encode(make_batch(dataset, [0]))[0, :3]
    assert (ethanol[0] - ethanol[1]).abs().sum() > 1e-4
    # symmetry noise (opt-in) breaks the tie
    noisy = GraphAutoencoder(AutoencoderConfig(**{**tiny_config.to_dict(), "symmetry_noise_dim": 4})).eval()
    nodes_n = noisy.encode(batch)[0, :6]
    assert (nodes_n - nodes_n[0]).abs().sum() > 1e-3


# ===========================================================================
# Reconstruction evaluation.
# ===========================================================================
def test_evaluate_reconstruction_reports_raw_and_constrained_separately(dataset, tiny_config):
    torch.manual_seed(0)
    model = GraphAutoencoder(tiny_config)
    result = evaluate_reconstruction(model, dataset, indices=list(range(8)), batch_size=3)
    assert result["molecules"] == 8
    for group in ("raw", "constrained"):
        assert set(result[group]) == {"validity", "sanitizable", "exact_graph_match", "exact_molecule_match"}
        assert all(0.0 <= v <= 1.0 for v in result[group].values())
    assert 0.0 <= result["equivariant_exact_ceiling"] <= 1.0
    # Evaluation leaves the module in the mode it found it in, whichever that was.
    model.train()
    evaluate_reconstruction(model, dataset, indices=[0, 1], compute_ceiling=False)
    assert model.training
    model.eval()
    evaluate_reconstruction(model, dataset, indices=[0, 1], compute_ceiling=False)
    assert not model.training
    skip = evaluate_reconstruction(model, dataset, indices=[0, 1], decode_molecules=False,
                                   compute_ceiling=False)
    assert "raw" not in skip and "constrained" not in skip


def test_constrained_validity_is_not_below_raw_for_an_untrained_model(dataset, tiny_config):
    torch.manual_seed(1)
    model = GraphAutoencoder(tiny_config)
    result = evaluate_reconstruction(model, dataset, indices=list(range(len(dataset))), batch_size=8)
    assert result["constrained"]["validity"] >= result["raw"]["validity"]
