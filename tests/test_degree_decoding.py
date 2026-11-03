"""Tests for the degree head, the degree-constrained assembler, and the ablation tools."""

from __future__ import annotations

import json
from pathlib import Path
import importlib.util

import numpy as np
import pytest
from rdkit import Chem, rdBase
import torch

from tests.test_autoencoder_model import make_records
from zinc_gnf.autoencoder import (
    AutoencoderConfig,
    GraphAutoencoder,
    evaluate_reconstruction,
    reconstruction_loss,
    true_degrees,
)
from zinc_gnf.chemistry import canonical_connectivity_smiles
from zinc_gnf.chemistry_charge_aware import encode_zinc_smiles_charge_aware, graph_to_mol_charge_aware
from zinc_gnf.constants_charge_aware import ATOM_VALENCE
from zinc_gnf.dataset import MolecularGraphDataset, make_batch
from zinc_gnf.decoder import MAX_DEGREE_CLASS
from zinc_gnf.decoding_charge_aware import (
    degree_constrained_bond_decode,
    valence_aware_bond_decode_typed,
)
from zinc_gnf.symmetry import is_decodable

TINY = dict(node_embedding_dim=16, latent_dim=32, num_processing_steps=2,
            attn_kq_dim=8, attn_v_dim=8, attn_concat_heads_output_dim=8)
RING = ["C1CCNCC1", "N1CCNCC1", "O1CCNCC1", "Fc1ccc(F)cc1", "c1ccccc1", "CCN(CC)CC", "CCO", "C[NH3+]"]


def graph(smiles):
    record, _ = encode_zinc_smiles_charge_aware(smiles)
    return record["smiles"], record["atom_types"], record["bond_matrix"]


def uniform_probs(bonds, bonded=0.5):
    """What a deterministic equivariant decoder can at best say about symmetric pairs: for every
    pair of equivalent atoms the SAME probability, here by averaging over colour-pair classes."""
    from zinc_gnf.symmetry import refine_colors
    atoms_dummy = None
    return atoms_dummy


def class_averaged_probs(atoms, bonds):
    """Per-colour-pair average of one-hot bond classes: the Bayes-optimal output of any
    permutation-equivariant pairwise decoder trained to convergence."""
    from collections import defaultdict
    from zinc_gnf.symmetry import refine_colors
    colors = refine_colors(atoms, bonds)
    n = len(atoms)
    groups = defaultdict(list)
    for i in range(n):
        for j in range(n):
            if i != j:
                groups[(min(colors[i], colors[j]), max(colors[i], colors[j]))].append((i, j))
    probs = np.zeros((n, n, 4))
    for pairs in groups.values():
        average = np.mean([np.eye(4)[bonds[i, j]] for i, j in pairs], axis=0)
        for i, j in pairs:
            probs[i, j] = average
    return probs


def to_smiles(atoms, bonds):
    with rdBase.BlockLogs():
        mol = graph_to_mol_charge_aware(atoms, bonds)
    return canonical_connectivity_smiles(mol) if len(Chem.GetMolFrags(mol)) == 1 else None


# ===========================================================================
# The assembler.
# ===========================================================================
@pytest.mark.parametrize("smiles", [r for r in RING if r != "c1ccccc1"])
def test_degree_assembly_recovers_the_molecule_with_class_averaged_probabilities(smiles):
    """The decisive property: for the best a deterministic decoder can do (identical output for
    symmetric pairs), degree-constrained assembly still yields the right MOLECULE."""
    canonical, atoms, bonds = graph(smiles)
    probs = class_averaged_probs(atoms, bonds)
    degrees = (bonds > 0).sum(1)
    assembled, info = degree_constrained_bond_decode(atoms, probs, degrees)
    assert (assembled > 0).sum(1).tolist() == degrees.tolist()          # degrees honoured exactly
    assert to_smiles(atoms, assembled) == canonical, smiles
    assert not info["fallback"]


def test_documented_limit_unsubstituted_benzene_gets_the_right_ring_but_not_the_kekule_orders():
    """Fully symmetric aromatic rings tie on BOND ORDER too, which degrees cannot break. Measured on
    real data this affects ~0.1% of molecules (substituents break the tie), so no repair is added."""
    canonical, atoms, bonds = graph("c1ccccc1")
    assembled, _ = degree_constrained_bond_decode(atoms, class_averaged_probs(atoms, bonds), (bonds > 0).sum(1))
    assert (assembled > 0).sum(1).tolist() == [2] * 6                       # connectivity: correct degrees
    assert to_smiles(atoms, assembled) == "C1CCCCC1"                       # bond orders: the known limitation


def test_greedy_assembly_fails_on_exactly_the_cases_the_degree_assembly_solves():
    failures = 0
    for smiles in ("C1CCNCC1", "N1CCNCC1", "O1CCNCC1", "Fc1ccc(F)cc1", "c1ccccc1"):
        canonical, atoms, bonds = graph(smiles)
        probs = class_averaged_probs(atoms, bonds)
        assert not is_decodable(atoms, bonds)
        greedy, _ = valence_aware_bond_decode_typed(atoms, probs)
        failures += to_smiles(atoms, greedy) != canonical
    assert failures >= 4


def test_degree_assembly_is_exact_when_probabilities_are_confident():
    for smiles in ("CCO", "CC(=O)[O-]", "C[N+](=O)[O-]", "O=C(Nc1ccc(F)cc1)c1cc(Cl)ccc1Br"):
        canonical, atoms, bonds = graph(smiles)
        assembled, info = degree_constrained_bond_decode(atoms, np.eye(4)[bonds], (bonds > 0).sum(1))
        assert np.array_equal(assembled, bonds) and not info["fallback"]


def test_degree_assembly_output_is_symmetric_and_valence_respecting():
    rng = np.random.default_rng(0)
    canonical, atoms, bonds = graph("O=C(Nc1ccc(F)cc1)c1cc(Cl)ccc1Br")
    caps = np.array([ATOM_VALENCE[a] for a in atoms])
    degrees = (bonds > 0).sum(1)
    for _ in range(10):
        probs = rng.dirichlet(np.ones(4), size=(len(atoms), len(atoms)))
        assembled, _ = degree_constrained_bond_decode(atoms, probs, degrees)
        assert np.array_equal(assembled, assembled.T) and (np.diag(assembled) == 0).all()
        assert (assembled.sum(1) <= caps).all()


@pytest.mark.parametrize("degrees,reason", [
    ([1, 1, 1], "infeasible_degrees"),        # odd degree sum
    ([5, 1, 1], "infeasible_degrees"),        # degree larger than n - 1
    ([1, 1], "bad_degrees"),                  # wrong length
])
def test_infeasible_degrees_fall_back_to_the_greedy_decoder(degrees, reason):
    atoms = np.array([0, 0, 0])
    probs = np.zeros((3, 3, 4)); probs[..., 1] = 0.6; probs[..., 0] = 0.4
    bonds, info = degree_constrained_bond_decode(atoms, probs, np.array(degrees))
    assert info["fallback"] is True and info["fallback_reason"] in (reason, "infeasible_degrees", "bad_degrees")
    assert bonds.shape == (3, 3)


def test_feasible_degrees_do_not_fall_back():
    atoms = np.array([0, 0, 0])
    probs = np.zeros((3, 3, 4)); probs[..., 1] = 0.9; probs[..., 0] = 0.1
    bonds, info = degree_constrained_bond_decode(atoms, probs, np.array([2, 2, 2]))      # a triangle
    assert info["fallback"] is False and (bonds > 0).sum() == 6


def test_disconnected_degree_solution_falls_back():
    atoms = np.array([0, 0, 0, 0])
    probs = np.zeros((4, 4, 4)); probs[..., 0] = 0.9; probs[..., 1] = 0.1
    # degrees [1,1,1,1] allow two disjoint bonds -> not connected -> fall back to greedy (which connects)
    bonds, info = degree_constrained_bond_decode(atoms, probs, np.array([1, 1, 1, 1]))
    assert info["fallback"] is True and info["fallback_reason"] == "not_connected"
    assert info["connected"]


def test_wrong_predicted_degrees_degrade_gracefully_not_catastrophically():
    canonical, atoms, bonds = graph("O1CCNCC1")
    probs = class_averaged_probs(atoms, bonds)
    wrong = (bonds > 0).sum(1).copy()
    wrong[0] += 1
    wrong[1] += 1                                   # still an even total, now inconsistent with the molecule
    assembled, info = degree_constrained_bond_decode(atoms, probs, wrong)
    assert assembled.shape == bonds.shape            # always returns a bond matrix


# ===========================================================================
# Head and loss.
# ===========================================================================
def test_degree_head_is_optional_and_off_by_default():
    plain = GraphAutoencoder(AutoencoderConfig(**TINY))
    assert plain.decoder.degree_head is None
    batch = make_batch(MolecularGraphDataset(make_records(RING)), [0, 1])
    assert "degree_logits" not in plain(batch)
    with_head = GraphAutoencoder(AutoencoderConfig(**TINY, degree_head=True))
    out = with_head(batch)
    assert out["degree_logits"].shape == (2, batch.max_nodes, MAX_DEGREE_CLASS + 1)
    # same encoder parameters either way: the flow's data is unchanged by the head
    plain_keys = set(plain.encoder.state_dict())
    assert plain_keys == set(with_head.encoder.state_dict())


def test_true_degrees_and_loss(tmp_path):
    dataset = MolecularGraphDataset(make_records(RING))
    batch = make_batch(dataset, [0, 7])            # piperidine, methylammonium
    degrees = true_degrees(batch)
    assert degrees[0, :6].tolist() == [2] * 6      # every ring atom has two heavy neighbours
    assert degrees[1, :2].tolist() == [1, 1]
    assert (degrees[~batch.node_mask] == 0).all()
    model = GraphAutoencoder(AutoencoderConfig(**TINY, degree_head=True))
    out = model(batch)
    losses = model.reconstruction_loss(batch, out)
    assert losses["degree_loss"].item() > 0 and np.isfinite(float(losses["degree_accuracy"]))
    total = losses["atom_loss"] + losses["bond_loss"] + losses["degree_loss"]
    assert torch.allclose(losses["total_loss"], total)
    # the weight scales only the degree term
    heavier = reconstruction_loss(batch, out, bond_class_weight=5.0, degree_loss_weight=3.0)
    assert heavier["degree_loss"].item() == pytest.approx(3.0 * losses["degree_loss"].item())
    assert heavier["atom_loss"].item() == pytest.approx(losses["atom_loss"].item())
    # padding cannot leak into the degree loss
    corrupted = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in out.items()}
    corrupted["degree_logits"][~batch.node_mask] += 1000.0
    assert reconstruction_loss(batch, corrupted, bond_class_weight=5.0)["degree_loss"].item() == pytest.approx(
        losses["degree_loss"].item())
    # no head -> no degree term
    plain = GraphAutoencoder(AutoencoderConfig(**TINY))
    assert plain.reconstruction_loss(batch, plain(batch))["degree_loss"].item() == 0.0


def test_degree_head_learns_on_one_batch():
    torch.manual_seed(0)
    dataset = MolecularGraphDataset(make_records(RING + ["CC(C)Cc1ccc(C(C)C(=O)O)cc1"]))
    model = GraphAutoencoder(AutoencoderConfig(**TINY, degree_head=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    batch = make_batch(dataset, list(range(len(dataset))))
    for _ in range(250):
        losses = model.reconstruction_loss(batch, model(batch))
        optimizer.zero_grad(); losses["mean_loss"].backward(); optimizer.step()
    assert float(losses["degree_accuracy"]) > 0.97


# ===========================================================================
# Evaluation wiring: the degree decoder uses PREDICTED degrees, never true ones.
# ===========================================================================
def test_evaluation_reports_a_separate_degree_decoder():
    torch.manual_seed(1)
    dataset = MolecularGraphDataset(make_records(RING))
    model = GraphAutoencoder(AutoencoderConfig(**TINY, degree_head=True))
    result = evaluate_reconstruction(model, dataset, batch_size=3)
    assert set(result) >= {"raw", "constrained", "degree"}
    assert set(result["degree"]) == {"validity", "sanitizable", "exact_graph_match", "exact_molecule_match"}
    assert 0.0 <= result["degree_accuracy"] <= 1.0 and 0.0 <= result["degree_decode_fallback_rate"] <= 1.0
    strata = result["exact_molecule_by_symmetry"]
    assert "degree_exact" in strata["decodable"] and "degree_exact" in strata["non_decodable"]
    plain = evaluate_reconstruction(GraphAutoencoder(AutoencoderConfig(**TINY)), dataset)
    assert "degree" not in plain and "degree_accuracy" not in plain


def test_evaluation_cannot_see_true_degrees():
    """Corrupting every TRUE degree must not change the degree decoder's predictions."""
    torch.manual_seed(2)
    dataset = MolecularGraphDataset(make_records(RING))
    model = GraphAutoencoder(AutoencoderConfig(**TINY, degree_head=True))
    first = evaluate_reconstruction(model, dataset)["degree"]
    import zinc_gnf.autoencoder as module
    original = module.true_degrees
    module.true_degrees = lambda batch: original(batch) * 0 + 3
    try:
        second = evaluate_reconstruction(model, dataset)["degree"]
    finally:
        module.true_degrees = original
    assert first == second                       # only the head's own predictions are used


# ===========================================================================
# Experiment tools.
# ===========================================================================
def load_script(repository_root: Path, name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), repository_root / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_result(variant, seed, exact, **extra):
    final = {"constrained_exact": exact, "raw_exact": exact - 0.1, "constrained_validity": 0.99,
             "raw_validity": 0.6, "exact_among_decodable": exact + 0.1, "exact_among_unreachable": 0.0,
             "atom_accuracy": 0.998, "real_bond_accuracy": 0.995}
    final.update(extra)
    final.setdefault("headline_exact", final.get("degree_exact", exact))
    final.setdefault("headline_validity", 0.99)
    return {"variant": variant, "steps": 100, "train_molecules": 10, "val_molecules": 5,
            "parameters": 1000, "final": final}


def test_summariser_pairs_by_seed_and_handles_degree_metrics(tmp_path, repository_root, capsys):
    root = tmp_path / "ablation"
    for seed, base in ((1, 0.50), (2, 0.60)):
        for variant, gain, extra in (("baseline", 0.0, {}),
                                     ("degree", 0.1, {"degree_exact": 0.9, "degree_validity": 0.99,
                                                       "degree_exact_among_unreachable": 0.75,
                                                       "degree_accuracy": 0.998})):
            d = root / variant / f"seed_{seed}"
            d.mkdir(parents=True)
            (d / "result.json").write_text(json.dumps(fake_result(variant, seed, base + gain, **extra)))
    script = load_script(repository_root, "03c_summarize_ablation.py")
    out = tmp_path / "summary.json"
    assert script.main(["--root", str(root), "--reference", "baseline", "--output", str(out)]) == 0
    summary = json.loads(out.read_text())
    paired = summary["variants"]["degree"]["paired_vs_reference"]["constrained_exact"]
    assert paired["mean"] == pytest.approx(0.1) and paired["seeds_where_better"] == 2 and paired["n"] == 2
    headline = summary["variants"]["degree"]["paired_vs_reference"]["headline_exact"]
    assert headline["mean"] == pytest.approx(0.9 - 0.55) and headline["seeds_where_better"] == 2
    assert summary["variants"]["degree"]["metrics"]["degree_exact"]["mean"] == pytest.approx(0.9)
    assert "baseline" in summary["variants"] and not summary["variants"]["baseline"]["paired_vs_reference"]
    assert out.with_suffix(".txt").exists()
    assert "better in 2/2 seeds" in capsys.readouterr().out
    assert script.main(["--root", str(tmp_path / "empty")]) == 2
    assert script.main(["--root", str(root), "--reference", "nope"]) == 4


def test_experiment_script_variants_and_no_overwrite(tmp_path, repository_root):
    script = load_script(repository_root, "03b_reconstruction_experiment.py")
    assert {"baseline", "noise4", "noise8", "deepbond", "refine3", "wide128", "steps12", "position",
            "degree", "degree_noise4"} <= set(script.VARIANTS)
    assert script.VARIANTS["baseline"] == {}
    assert script.VARIANTS["degree"] == {"degree_head": True}
    # every variant must be a valid AutoencoderConfig
    for overrides in script.VARIANTS.values():
        AutoencoderConfig.from_dict(overrides)
    existing = tmp_path / "done"
    existing.mkdir()
    (existing / "result.json").write_text("{}")
    assert script.main(["--output-dir", str(existing), "--data-dir", str(tmp_path)]) == 4
    assert (existing / "result.json").read_text() == "{}"                 # untouched
    assert script.main(["--output-dir", str(tmp_path / "x"), "--data-dir", str(tmp_path / "none"),
                        "--device", "cpu"]) == 2


def test_experiment_end_to_end_on_tiny_data(tmp_path, repository_root):
    from tests.test_autoencoder_training import write_dataset
    data = write_dataset(tmp_path / "data")
    script = load_script(repository_root, "03b_reconstruction_experiment.py")
    out = tmp_path / "exp" / "degree" / "seed_1"
    code = script.main(["--data-dir", str(data), "--output-dir", str(out), "--variant", "degree",
                        "--train-molecules", "20", "--val-molecules", "8", "--steps", "4", "--batch-size", "4",
                        "--eval-every", "4", "--device", "cpu", "--seed", "1", "--no-flow-metrics",
                        "--set", "node_embedding_dim=16", "--set", "latent_dim=32",
                        "--set", "num_processing_steps=1"])
    assert code == 0
    result = json.loads((out / "result.json").read_text())
    assert result["model_config"]["degree_head"] is True and result["model_config"]["node_embedding_dim"] == 16
    assert result["final"]["degree_exact"] is not None and result["final"]["degree_accuracy"] is not None
    assert result["final"]["headline_decoder"] == "degree"
    assert result["final"]["headline_exact"] == result["final"]["degree_exact"]
    assert "flow_relevant" not in result
    assert script.main(["--data-dir", str(data), "--output-dir", str(out), "--steps", "2"]) == 4   # no overwrite
    # with flow metrics enabled
    out2 = tmp_path / "exp" / "noise4" / "seed_1"
    assert script.main(["--data-dir", str(data), "--output-dir", str(out2), "--variant", "noise4",
                        "--train-molecules", "20", "--val-molecules", "8", "--steps", "4", "--batch-size", "4",
                        "--eval-every", "4", "--device", "cpu", "--seed", "1",
                        "--set", "node_embedding_dim=16", "--set", "latent_dim=32",
                        "--set", "num_processing_steps=1"]) == 0
    result2 = json.loads((out2 / "result.json").read_text())
    assert result2["final"]["headline_decoder"] == "constrained"
    assert result2["final"]["headline_exact"] == result2["final"]["constrained_exact"]
    flow = result2["flow_relevant"]
    assert {r["decoder"] for r in flow["decoder_robustness_to_embedding_error"]} == {"constrained"}
    assert flow["embedding_statistics"]["noise_variance_fraction"] > 0
    assert len(flow["decoder_robustness_to_embedding_error"]) == 5
    assert flow["noise_draw_spread"]["draws"] == 5


def test_robustness_uses_the_models_own_decoder():
    from zinc_gnf.reconstruction_analysis import embedding_robustness
    dataset = MolecularGraphDataset(make_records(RING))
    plain = GraphAutoencoder(AutoencoderConfig(**TINY))
    with_head = GraphAutoencoder(AutoencoderConfig(**TINY, degree_head=True))
    rows_plain = embedding_robustness(plain, dataset, sigmas=(0.0,), batch_size=4)
    rows_head = embedding_robustness(with_head, dataset, sigmas=(0.0, 0.1), batch_size=4)
    assert rows_plain[0]["decoder"] == "constrained"
    assert all(r["decoder"] == "degree" for r in rows_head)
    forced = embedding_robustness(with_head, dataset, sigmas=(0.0,), batch_size=4, use_degree_decoder=False)
    assert forced[0]["decoder"] == "constrained"
