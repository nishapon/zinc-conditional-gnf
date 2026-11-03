"""Generation pipeline and metric definitions.

The decoding tests drive the decoder with hand-built logits, so the expected
molecule is known exactly and raw / greedy-constrained / degree-constrained
assembly can be told apart. The metric tests use hand-built records whose
validity, uniqueness and novelty can be counted by hand.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from zinc_gnf.autoencoder import AutoencoderConfig, GraphAutoencoder
from zinc_gnf.chemistry import canonical_connectivity_smiles
from zinc_gnf.chemistry_charge_aware import (
    encode_zinc_smiles_charge_aware,
    graph_to_mol_charge_aware,
)
from zinc_gnf.constants_charge_aware import NUM_ATOM_TYPES, NUM_BOND_TYPES
from zinc_gnf.decoder import MAX_DEGREE_CLASS
from zinc_gnf.flow import FlowConfig, NodeFlow
from zinc_gnf.generation import (
    DECODERS,
    NodeCountSampler,
    decode_embeddings,
    generate_molecules,
    generation_batch,
    node_count_agreement,
    size_distribution,
)
from zinc_gnf.metrics_generation import (
    decoder_metrics,
    format_metrics,
    generation_metrics,
    internal_diversity,
    scaffold_statistics,
)

MOLECULES = ["CCO", "c1ccccc1", "CC(=O)Nc1ccccc1", "CCN(CC)CC"]


# ---------------------------------------------------------------------------
# A decoder stub driven by known graphs.
# ---------------------------------------------------------------------------
class StubAutoencoder:
    """Returns logits that encode exactly the graphs it was given.

    `bond_confidence=None` makes every bond class equally likely, which is how
    the "raw argmax is not a molecule" case is produced.
    """

    def __init__(self, graphs, *, degree_head=True, bond_confidence=8.0, degrees=None):
        self.graphs = graphs
        self.config = AutoencoderConfig(degree_head=degree_head)
        self.training = False
        self.bond_confidence = bond_confidence
        self.degrees = degrees

    def eval(self):
        return self

    def train(self):
        return self

    def decode(self, embeddings, batch, *, teacher_forcing):
        assert teacher_forcing is False, "generation must never teacher-force"
        count, width = batch.node_mask.shape
        atom_logits = torch.zeros((count, width, NUM_ATOM_TYPES))
        bond_logits = torch.zeros((count, width, width, NUM_BOND_TYPES))
        degree_logits = torch.zeros((count, width, MAX_DEGREE_CLASS + 1))
        for index in range(count):      # the graph list cycles, so any batch size works
            atoms, bonds = self.graphs[index % len(self.graphs)]
            n = len(atoms)
            for i, atom in enumerate(atoms):
                atom_logits[index, i, int(atom)] = 10.0
            degrees = (self.degrees[index % len(self.degrees)] if self.degrees is not None
                       else (np.asarray(bonds) > 0).sum(1))
            for i in range(n):
                degree_logits[index, i, min(int(degrees[i]), MAX_DEGREE_CLASS)] = 10.0
            if self.bond_confidence is not None:
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            bond_logits[index, i, j, int(bonds[i][j])] = self.bond_confidence
        outputs = {"atom_logits": atom_logits, "bond_logits": bond_logits}
        if self.config.degree_head:
            outputs["degree_logits"] = degree_logits
        return outputs


def graph_of(smiles: str):
    record, reason = encode_zinc_smiles_charge_aware(smiles, row_id=0)
    assert record is not None, (smiles, reason)
    return np.asarray(record["atom_types"]), np.asarray(record["bond_matrix"])


def reference_smiles(smiles: str) -> str:
    atoms, bonds = graph_of(smiles)
    return canonical_connectivity_smiles(graph_to_mol_charge_aware(atoms, bonds))


# ---------------------------------------------------------------------------
# Masks and batches.
# ---------------------------------------------------------------------------
def test_generation_batch_carries_only_shapes_and_masks():
    batch = generation_batch([3, 1])
    assert batch.node_mask.tolist() == [[True, True, True], [True, False, False]]
    assert batch.pair_mask[0].tolist() == [[False, True, True], [True, False, True],
                                          [True, True, False]]
    assert batch.pair_mask[1].sum() == 0                     # a single atom has no pairs
    assert torch.equal(batch.atom_types, torch.zeros((2, 3), dtype=torch.long))
    assert torch.equal(batch.bond_types, torch.zeros((2, 3, 3), dtype=torch.long))
    assert batch.condition is None and bool(torch.isnan(batch.qed).all())
    assert batch.n_node.tolist() == [3, 1]


# ---------------------------------------------------------------------------
# Decoding.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("smiles", MOLECULES)
def test_confident_logits_decode_to_the_intended_molecule_with_every_decoder(smiles):
    atoms, bonds = graph_of(smiles)
    stub = StubAutoencoder([(atoms, bonds)])
    records = decode_embeddings(stub, torch.zeros((1, len(atoms), 64)), [len(atoms)])
    assert len(records) == 1
    for decoder in DECODERS:
        entry = records[0][decoder]
        assert entry["valid"], (decoder, entry)
        assert entry["smiles"] == reference_smiles(smiles), decoder
        assert entry["connected"] and entry["sanitizable"] and entry["fragments"] == 1
        assert 0.0 <= entry["qed"] <= 1.0
    assert records[0]["predicted_degrees"] == (bonds > 0).sum(1).tolist()


def test_raw_argmax_can_be_invalid_while_degree_constrained_assembly_is_valid():
    """Why raw validity and post-decoding validity must be reported separately."""
    atoms, bonds = graph_of("CC(=O)Nc1ccccc1")
    stub = StubAutoencoder([(atoms, bonds)], bond_confidence=None)      # uniform bond classes
    records = decode_embeddings(stub, torch.zeros((1, len(atoms), 64)), [len(atoms)])
    entry = records[0]
    assert not entry["raw"]["valid"], "a uniform bond argmax should not be a molecule"
    assert entry["degree"]["valid"]
    assert entry["degree"]["bonds"] == int((bonds > 0).sum() // 2)       # right number of bonds
    assert entry["constrained"]["valid"]


def test_infeasible_predicted_degrees_fall_back_and_say_why():
    atoms, bonds = graph_of("CCO")
    odd = np.array([1, 1, 1])                       # odd degree sum: no graph can satisfy it
    stub = StubAutoencoder([(atoms, bonds)], degrees=[odd])
    records = decode_embeddings(stub, torch.zeros((1, 3, 64)), [3])
    entry = records[0]["degree"]
    assert entry["fallback"] and entry["fallback_reason"] == "infeasible_degrees"
    assert records[0]["constrained"]["fallback"] is False


def test_decoder_selection_is_validated():
    atoms, bonds = graph_of("CCO")
    stub = StubAutoencoder([(atoms, bonds)])
    with pytest.raises(ValueError, match="unknown decoders"):
        decode_embeddings(stub, torch.zeros((1, 3, 64)), [3], decoders=("magic",))
    plain = StubAutoencoder([(atoms, bonds)], degree_head=False)
    with pytest.raises(ValueError, match="no degree head"):
        decode_embeddings(plain, torch.zeros((1, 3, 64)), [3], decoders=("degree",))
    records = decode_embeddings(plain, torch.zeros((1, 3, 64)), [3], decoders=("raw", "constrained"))
    assert "degree" not in records[0]
    with pytest.raises(ValueError, match="do not match"):
        decode_embeddings(stub, torch.zeros((1, 5, 64)), [3])


def test_real_decoder_handles_arbitrary_embeddings_without_raising():
    """An untrained decoder produces mostly nonsense; it must still be scored, not crash."""
    model = GraphAutoencoder(AutoencoderConfig(degree_head=True)).eval()
    generator = torch.Generator().manual_seed(0)
    embeddings = torch.randn((3, 9, model.config.node_embedding_dim), generator=generator)
    records = decode_embeddings(model, embeddings, [9, 7, 4])
    assert len(records) == 3
    for record in records:
        for decoder in DECODERS:
            entry = record[decoder]
            assert set(entry) >= {"valid", "sanitizable", "connected", "smiles", "bonds"}
            assert isinstance(entry["valid"], bool)
            if not entry["valid"]:
                assert entry["smiles"] is None


# ---------------------------------------------------------------------------
# Node-count sampling.
# ---------------------------------------------------------------------------
def test_node_count_sampler_reproduces_the_training_histogram():
    counts = [5] * 70 + [9] * 30
    sampler = NodeCountSampler(counts)
    assert sampler.histogram() == {5: 0.7, 9: 0.3}
    assert sampler.mean == pytest.approx(6.2)
    drawn = sampler.sample(4000, np.random.default_rng(0))
    assert set(np.unique(drawn)) <= {5, 9}
    assert abs((drawn == 5).mean() - 0.7) < 0.03
    again = sampler.sample(4000, np.random.default_rng(0))
    assert np.array_equal(drawn, again)


@pytest.mark.parametrize("bad", [[], [0, 5], [5, 99]])
def test_node_count_sampler_validates_its_input(bad):
    with pytest.raises(ValueError):
        NodeCountSampler(bad)


def test_size_distribution_counts_requested_sizes():
    records = [{"n_node": 4}, {"n_node": 4}, {"n_node": 7}]
    assert size_distribution(records) == {"histogram": {4: 2, 7: 1}, "mean": pytest.approx(5.0)}


# ---------------------------------------------------------------------------
# Sampling end to end.
# ---------------------------------------------------------------------------
def tiny_flow(**overrides) -> NodeFlow:
    config = FlowConfig(**{"node_embedding_dim": 64, "num_blocks": 1, "attention_dim": 8,
                           "num_heads": 2, "hidden_dim": 16, **overrides})
    flow = NodeFlow(config, init_seed=0)
    flow.set_data_statistics(torch.zeros(config.node_embedding_dim),
                             torch.ones(config.node_embedding_dim))
    return flow


def test_generation_puts_the_flow_in_eval_mode_itself():
    """`generate_molecules` must not depend on the caller remembering `.eval()`."""
    atoms, bonds = graph_of("CCO")
    flow = tiny_flow()
    flow.train()
    generate_molecules(flow, StubAutoencoder([(atoms, bonds)]), count=2,
                       node_counts=NodeCountSampler([3, 3]), seed=0, batch_size=2)
    assert not flow.training


def test_generation_is_reproducible_and_respects_the_budget():
    atoms, bonds = graph_of("CCO")
    stub = StubAutoencoder([(atoms, bonds)] * 8)
    flow = tiny_flow()
    sampler = NodeCountSampler([3] * 10)
    first = generate_molecules(flow, stub, count=5, node_counts=sampler, seed=7, batch_size=2)
    second = generate_molecules(flow, stub, count=5, node_counts=sampler, seed=7, batch_size=2)
    assert len(first["records"]) == 5
    assert [r["index"] for r in first["records"]] == [0, 1, 2, 3, 4]
    assert [r["degree"]["smiles"] for r in first["records"]] == \
           [r["degree"]["smiles"] for r in second["records"]]
    assert first["settings"]["temperature"] == 1.0
    assert first["settings"]["node_count_source"] == "train_histogram"


def test_target_n_fixes_every_sample_size():
    atoms, bonds = graph_of("CCO")
    stub = StubAutoencoder([(atoms, bonds)] * 4)
    result = generate_molecules(tiny_flow(), stub, count=4, node_counts=NodeCountSampler([9, 12]),
                                seed=1, batch_size=4, target_n=3)
    assert [r["n_node"] for r in result["records"]] == [3, 3, 3, 3]
    assert result["settings"]["node_count_source"] == "fixed"


def test_temperature_changes_the_samples():
    model = GraphAutoencoder(AutoencoderConfig(degree_head=True)).eval()
    flow = tiny_flow()
    sampler = NodeCountSampler([6] * 5)
    cold = generate_molecules(flow, model, count=3, node_counts=sampler, seed=0, temperature=0.1)
    hot = generate_molecules(flow, model, count=3, node_counts=sampler, seed=0, temperature=1.5)
    assert ([r["atom_types"] for r in cold["records"]] !=
            [r["atom_types"] for r in hot["records"]])


def test_the_two_sampling_modes_never_mix():
    """A condition is either required or rejected; it is never silently ignored.

    Stage 12 added the conditional path, so a conditional flow now samples - but only
    when it is GIVEN conditions, and an unconditional flow still refuses them.
    """
    atoms, bonds = graph_of("CCO")
    stub = StubAutoencoder([(atoms, bonds)])
    sampler = NodeCountSampler([3])
    with pytest.raises(ValueError, match="condition_values is required"):
        generate_molecules(tiny_flow(condition_features=("n",)), stub, count=1,
                           node_counts=sampler, seed=0)
    with pytest.raises(ValueError, match="unconditional"):
        generate_molecules(tiny_flow(), stub, count=1, node_counts=sampler, seed=0,
                           condition_values=torch.zeros(1, 1))
    with pytest.raises(ValueError):
        generate_molecules(tiny_flow(), stub, count=0, node_counts=sampler, seed=0)


def test_node_count_agreement_is_reported_as_structural():
    atoms, bonds = graph_of("CCO")
    stub = StubAutoencoder([(atoms, bonds)])
    records = decode_embeddings(stub, torch.zeros((1, 3, 64)), [3])
    agreement = node_count_agreement(records, "degree")
    assert agreement["molecules"] == 1 and agreement["agreement"] == 1.0
    assert "structural" in agreement["note"]


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------
def entry(smiles=None, *, valid=True, fallback=False, reason=None, qed=0.5, rings=0):
    return {"valid": valid and smiles is not None, "smiles": smiles if valid else None,
            "sanitizable": valid or smiles is not None, "connected": valid,
            "fragments": 1 if valid else 2, "fallback": fallback, "fallback_reason": reason,
            "bonds": 2, "qed": qed, "rings": rings}


def hand_built_records():
    """8 valid of 10; 4 distinct; 2 of those 4 absent from the training set."""
    smiles = ["CCO", "CCO", "CCO", "CCO", "CCC", "CCC", "c1ccccc1", "CCN"]
    records = [{"n_node": 3, "degree": entry(s)} for s in smiles]
    records.append({"n_node": 5, "degree": entry("CCCCC", valid=False, fallback=True,
                                                 reason="not_connected")})
    records.append({"n_node": 4, "degree": entry(None, valid=False)})
    return records


def test_metric_values_are_exactly_what_counting_by_hand_gives():
    records = hand_built_records()
    metrics = decoder_metrics(records, "degree", train_smiles={"CCO", "CCC"},
                              reference_smiles={"train_and_val": {"CCO", "CCC", "CCN"}})
    assert metrics["samples"] == 10
    assert metrics["strict_validity"]["value"] == pytest.approx(0.8)
    assert metrics["valid_molecules"] == 8 and metrics["unique_molecules"] == 4
    assert metrics["uniqueness"]["value"] == pytest.approx(4 / 8)
    assert metrics["novelty_vs_train"]["value"] == pytest.approx(2 / 4)
    assert metrics["novelty_vs_train_and_val"]["value"] == pytest.approx(1 / 4)
    assert metrics["nuv"] == pytest.approx(0.8 * 0.5 * 0.5)
    assert metrics["fallback_rate"]["value"] == pytest.approx(0.1)
    assert metrics["fallback_reasons"] == {"not_connected": 1}
    assert metrics["sanitizable"]["value"] == pytest.approx(0.9)      # the 9th is sanitizable
    low, high = metrics["strict_validity"]["ci95"]
    assert low < 0.8 < high
    assert metrics["qed"]["molecules"] == 8


def test_novelty_requires_a_reference_and_every_record_needs_the_decoder():
    records = hand_built_records()
    with pytest.raises(ValueError, match="train_smiles is empty"):
        generation_metrics(records, train_smiles=[])
    with pytest.raises(ValueError, match="have no 'raw' result"):
        decoder_metrics(records, "raw", train_smiles={"CCO"})


def test_generation_metrics_reports_each_decoder_and_a_headline():
    atoms, bonds = graph_of("CCO")
    stub = StubAutoencoder([(atoms, bonds)] * 3)
    records = decode_embeddings(stub, torch.zeros((3, 3, 64)), [3, 3, 3])
    metrics = generation_metrics(records, train_smiles={"CCN"}, validation_smiles={"CCC"},
                                 headline="degree")
    assert set(metrics["decoders"]) == set(DECODERS)
    assert metrics["headline_decoder"] == "degree"
    assert metrics["headline"]["strict_validity"] == pytest.approx(1.0)
    assert metrics["headline"]["uniqueness"] == pytest.approx(1 / 3)    # three copies of one molecule
    assert metrics["headline"]["novelty_vs_train"] == pytest.approx(1.0)
    assert metrics["train_reference_molecules"] == 1
    report = format_metrics(metrics)
    assert "<- headline" in report and "N.U.V." in report


def test_internal_diversity_is_between_zero_and_one_and_handles_degenerate_input():
    assert internal_diversity(["CCO"])["internal_diversity"] != internal_diversity(["CCO"])[
        "internal_diversity"]                                        # NaN for a single molecule
    identical = internal_diversity(["CCO", "CCO"])
    assert identical["molecules"] == 1                               # set() removes the duplicate
    mixed = internal_diversity(["CCO", "c1ccccc1", "CC(=O)Nc1ccccc1", "CCN(CC)CC"])
    assert 0.0 < mixed["internal_diversity"] <= 1.0
    assert mixed["pairs"] == 6
    assert mixed["mean_tanimoto"] == pytest.approx(1.0 - mixed["internal_diversity"])
    far = internal_diversity(["C", "c1ccccc1"])["internal_diversity"]
    near = internal_diversity(["CCCC", "CCCCC"])["internal_diversity"]
    assert far > near


def test_internal_diversity_subsamples_deterministically():
    many = [f"C{'C' * (i % 7)}O{'C' * (i % 3)}" for i in range(50)]
    many = [s for s in many if s.count("C") <= 8]
    first = internal_diversity(many, sample=5)
    second = internal_diversity(many, sample=5)
    assert first == second and first["molecules"] <= 5


def test_scaffold_statistics_counts_distinct_ring_systems():
    statistics = scaffold_statistics(["c1ccccc1", "CC(=O)Nc1ccccc1", "CCO"])
    assert statistics["molecules"] == 3
    assert statistics["distinct_scaffolds"] == 2        # benzene for both aromatics, empty for CCO
    assert 0 < statistics["scaffolds_per_molecule"] <= 1


def test_metrics_survive_a_sample_with_nothing_valid():
    records = [{"n_node": 4, "degree": entry(None, valid=False)} for _ in range(3)]
    metrics = decoder_metrics(records, "degree", train_smiles={"CCO"})
    assert metrics["strict_validity"]["value"] == 0.0
    assert math.isnan(metrics["uniqueness"]["value"]) and math.isnan(metrics["nuv"])
    assert "diversity" not in metrics


# ---------------------------------------------------------------------------
# Numerical safety at sampling time.
# ---------------------------------------------------------------------------
def test_non_finite_embeddings_are_counted_invalid_without_poisoning_the_batch():
    """A flow in a bad regime can emit NaN/inf; those samples must not reach RDKit."""
    atoms, bonds = graph_of("CCO")
    stub = StubAutoencoder([(atoms, bonds)] * 3)
    embeddings = torch.zeros((3, 3, 64))
    embeddings[0, 1, 5] = float("nan")
    embeddings[1, 2, 0] = float("inf")
    records = decode_embeddings(stub, embeddings, [3, 3, 3])
    for index in (0, 1):
        assert records[index]["nonfinite_embedding"] is True
        for decoder in DECODERS:
            entry = records[index][decoder]
            assert entry["valid"] is False and entry["smiles"] is None
            assert entry["fallback_reason"] == "nonfinite_embedding"
    good = records[2]                                  # the finite sample still decodes
    assert good["nonfinite_embedding"] is False
    assert good["degree"]["valid"] and good["degree"]["smiles"] == reference_smiles("CCO")


def test_generation_reports_the_non_finite_rate_and_a_latent_round_trip_check():
    atoms, bonds = graph_of("CCO")
    stub = StubAutoencoder([(atoms, bonds)] * 4)
    result = generate_molecules(tiny_flow(), stub, count=4,
                                node_counts=NodeCountSampler([3] * 4), seed=0, batch_size=2)
    settings = result["settings"]
    assert settings["nonfinite_embeddings"] == 0
    assert settings["nonfinite_embedding_rate"] == 0.0
    # re-encoding the sampled embeddings must return the latents they were drawn from
    assert settings["latent_round_trip_relative_error"] < 1e-4


def test_sample_can_return_its_latents():
    from zinc_gnf.flow import node_mask_from_counts
    flow = tiny_flow()
    flow.eval()          # sampling is an eval-mode operation: a train()-mode forward would
                         # data-dependently (re)initialize ActNorm and change the map
    mask = node_mask_from_counts([4, 2])
    embeddings, latents = flow.sample(mask, temperature=0.9,
                                      generator=torch.Generator().manual_seed(1),
                                      return_latent=True)
    assert embeddings.shape == latents.shape == (2, 4, 64)
    with torch.no_grad():
        recovered, _ = flow(embeddings, mask)
    assert torch.allclose(recovered, latents, atol=1e-5)
    plain = flow.sample(mask, temperature=0.9, generator=torch.Generator().manual_seed(1))
    assert torch.allclose(plain, embeddings)
