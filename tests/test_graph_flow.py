"""Stage 22: the direct conditional graph flow.

A flow is only a flow if it is invertible and its log-determinant is right, and a
conditional flow is only conditional if the condition reaches the samples. Everything
below is a property that the stage's conclusions depend on:

  INVERTIBILITY   encode then invert reproduces the input to float32 precision, for every
                  molecule size in the batch, including the smallest.
  LOG-DETERMINANT checked numerically against the Jacobian of the actual map, so a wrong
                  logdet cannot hide behind a plausible loss curve.
  MASKING         padded atoms and padded pairs are untouched, contribute nothing to the
                  density, and garbage written into them cannot change the likelihood.
  SYMMETRY        bonds come back symmetric with a zero diagonal BY CONSTRUCTION, because
                  only the strict upper triangle is modelled.
  EXACT N         a request for N atoms yields N atoms, before any chemistry.
  ATOM/BOND       generated graphs decode through the project's own chemistry, and the
  VALIDITY        valency corrector never leaves an over-valent atom behind.
  CONDITIONING    the condition changes the samples; the null embedding is a separate
                  path; neutralising the coupling condition is reversible; and a
                  within-N-permuted condition is what the likelihood gap measures.
  STABILITY       the density is finite for degenerate inputs, and a deep stack does not
                  overflow in float32.
  RESUMPTION      training resumes at the exact next batch and reaches the same step.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import pickle
import re
import subprocess

import numpy as np
import pytest
import torch

from tests.test_autoencoder_training import write_dataset
from zinc_gnf.constants_charge_aware import ATOM_VALENCE, NUM_ATOM_TYPES, NUM_BOND_TYPES
from zinc_gnf.graph_flow import (
    BondSubnet,
    ConditionalCoupling,
    ConditionalGraphFlow,
    GraphFlowConfig,
    dequantize,
    neutralise_coupling_condition,
    pack_upper,
    pair_mask_from_counts,
    unpack_upper,
    upper_triangle_indices,
)
from zinc_gnf.graph_flow_generation import (
    generate_conditional,
    graphs_to_records,
    valency_correction,
)
from zinc_gnf.preprocessing import ConditionScaler

REPOSITORY = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""),
                                                  REPOSITORY / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_flow(*, max_nodes: int = 10, blocks: int = 3, seed: int = 0
              ) -> ConditionalGraphFlow:
    torch.manual_seed(seed)
    return ConditionalGraphFlow(GraphFlowConfig(
        max_nodes=max_nodes, bond_blocks=blocks, atom_blocks=blocks, hidden=32,
        condition_dim=16, condition_hidden=32))


def make_inputs(flow: ConditionalGraphFlow, counts: list[int], *, seed: int = 0):
    torch.manual_seed(seed)
    width = flow.config.max_nodes
    batch = len(counts)
    node_mask = torch.zeros(batch, width, dtype=torch.bool)
    for row, count in enumerate(counts):
        node_mask[row, :count] = True
    pair_mask = pair_mask_from_counts(counts, width)
    atom_types = torch.randint(0, flow.config.num_atom_types, (batch, width)) * node_mask
    dense = torch.randint(0, flow.config.num_bond_types, (batch, width, width))
    dense = torch.triu(dense, 1)
    dense = dense + dense.transpose(1, 2)
    atoms, bonds = dequantize(atom_types, dense, node_mask, pair_mask,
                              num_atom_types=flow.config.num_atom_types,
                              num_bond_types=flow.config.num_bond_types,
                              noise_scale=flow.config.noise_scale)
    context = flow.conditioning(torch.randn(batch, 2))
    flow.train()
    flow.log_likelihood(atoms, bonds, node_mask, pair_mask, context)   # init actnorm
    flow.eval()
    return atoms, bonds, node_mask, pair_mask, context


# ------------------------------------------------------------- invertibility ----
@pytest.mark.parametrize("counts", [[10, 7, 4, 2], [10], [2, 2]])
def test_encode_then_invert_reproduces_the_graph(counts):
    flow = tiny_flow()
    atoms, bonds, node_mask, pair_mask, context = make_inputs(flow, counts)
    encoded = flow.encode(atoms, bonds, node_mask, pair_mask, context)
    recovered_bonds = flow._invert(flow.bond_layers, encoded["z_bond"], pair_mask,
                                   context, None)
    recovered_atoms = flow._invert(flow.atom_layers, encoded["z_atom"], node_mask,
                                   context, unpack_upper(bonds, flow.config.max_nodes))
    bond_error = float(((recovered_bonds - bonds).abs()
                        * pair_mask.unsqueeze(-1)).max().detach())
    atom_error = float(((recovered_atoms - atoms).abs()
                        * node_mask.unsqueeze(-1)).max().detach())
    assert bond_error < 1e-4, f"bond inverse drifted by {bond_error}"
    assert atom_error < 1e-4, f"atom inverse drifted by {atom_error}"


def _wake_couplings(flow: ConditionalGraphFlow, scale: float = 0.5) -> None:
    """Move the zero-initialised coupling heads so the flow is a non-trivial map."""
    with torch.no_grad():
        for coupling in (m for m in flow.modules() if isinstance(m, ConditionalCoupling)):
            for parameter in coupling.subnet.net[-1].parameters():
                parameter.add_(torch.randn_like(parameter) * scale)


def test_a_deep_stack_stays_invertible_at_the_default_log_scale_bound():
    """The default bound must keep a DEEP perturbed stack invertible in float32.

    A stack of K couplings each bounded by L has inverse condition number up to
    exp(K*L); float32 carries ~1e-7 relative precision. This pins the default L against
    a deeper stack than the configuration actually uses.
    """
    flow = tiny_flow(blocks=10, seed=3)
    assert flow.config.max_log_scale <= 1.0, "the default bound must stay small"
    atoms, bonds, node_mask, pair_mask, context = make_inputs(flow, [10, 6])
    _wake_couplings(flow)
    encoded = flow.encode(atoms, bonds, node_mask, pair_mask, context)
    assert torch.isfinite(encoded["z_bond"]).all()
    assert torch.isfinite(encoded["z_atom"]).all()
    recovered = flow._invert(flow.bond_layers, encoded["z_bond"], pair_mask, context, None)
    assert torch.isfinite(recovered).all()
    error = float(((recovered - bonds).abs() * pair_mask.unsqueeze(-1)).max().detach())
    assert error < 1e-2, f"a perturbed deep stack lost invertibility: {error}"


def test_the_sampler_reports_a_small_round_trip_error_by_default():
    """Sampling IS the inverse, so this number is reported on every generation call."""
    flow = tiny_flow(blocks=6, seed=4)
    make_inputs(flow, [10, 10])
    _wake_couplings(flow)
    context = flow.conditioning(torch.randn(2, 2))
    sampled = flow.sample([10, 7], context, temperature=0.8,
                          generator=torch.Generator().manual_seed(2))
    error = sampled["latent_round_trip_relative_error"]
    assert error["bond"] < 1e-3 and error["atom"] < 1e-3, error


def test_the_round_trip_check_detects_an_ill_conditioned_inverse():
    """The guard must FIRE on a configuration that float32 cannot invert.

    This is the limitation stated honestly: a coupling stack is only invertible in
    practice while exp(blocks * max_log_scale) stays inside float32 precision. A model
    configured past that point produces molecules that are not drawn from its own
    density, and the only defence is to measure it - which `sample` does.
    """
    torch.manual_seed(9)
    flow = ConditionalGraphFlow(GraphFlowConfig(
        max_nodes=10, bond_blocks=12, atom_blocks=12, hidden=32, condition_dim=16,
        condition_hidden=32, max_log_scale=4.0))
    make_inputs(flow, [10, 10])
    _wake_couplings(flow, scale=1.0)
    context = flow.conditioning(torch.randn(2, 2))
    sampled = flow.sample([10, 10], context, temperature=1.0,
                          generator=torch.Generator().manual_seed(3))
    error = sampled["latent_round_trip_relative_error"]
    assert max(error["bond"], error["atom"]) > 1e-3, (
        "an ill-conditioned stack must be detected by the round-trip check, not pass "
        f"silently: {error}")


def test_the_log_determinant_matches_the_true_jacobian():
    """Numerical check: a wrong logdet would otherwise hide behind a plausible loss."""
    flow = tiny_flow(max_nodes=4, blocks=2, seed=5)
    counts = [4]
    atoms, bonds, node_mask, pair_mask, context = make_inputs(flow, counts)
    with torch.no_grad():
        for coupling in (m for m in flow.modules() if isinstance(m, ConditionalCoupling)):
            for parameter in coupling.subnet.net[-1].parameters():
                parameter.add_(torch.randn_like(parameter) * 0.2)

    def bond_map(flat: torch.Tensor) -> torch.Tensor:
        shaped = flat.reshape(bonds.shape)
        out, _ = flow._run(flow.bond_layers, shaped, pair_mask, context, None)
        return out.reshape(-1)

    flat = bonds.reshape(-1).double()
    flow_double = flow.double()
    jacobian = torch.autograd.functional.jacobian(
        lambda v: bond_map(v.float().reshape(bonds.shape)).double(), flat)
    flow.float()
    live = pair_mask.reshape(-1).repeat_interleave(1)
    index = torch.nonzero(pair_mask.unsqueeze(-1).expand_as(bonds).reshape(-1)).flatten()
    reduced = jacobian[index][:, index]
    numeric = torch.slogdet(reduced)[1]
    _, analytic = flow._run(flow.bond_layers, bonds, pair_mask, context, None)
    measured = float(analytic[0].detach())
    assert measured == pytest.approx(float(numeric), abs=2e-2), (
        f"analytic logdet {measured} vs numeric {float(numeric)}")
    assert live.any()
    _ = flow_double


# ------------------------------------------------------------------- masking ----
def test_padding_is_untouched_and_contributes_nothing():
    flow = tiny_flow()
    atoms, bonds, node_mask, pair_mask, context = make_inputs(flow, [10, 6, 3])
    encoded = flow.encode(atoms, bonds, node_mask, pair_mask, context)
    bond_leak = float((encoded["z_bond"] * (~pair_mask).unsqueeze(-1)).abs().max().detach())
    atom_leak = float((encoded["z_atom"] * (~node_mask).unsqueeze(-1)).abs().max().detach())
    assert bond_leak == 0.0 and atom_leak == 0.0


def test_garbage_in_the_padding_cannot_change_the_likelihood():
    flow = tiny_flow()
    atoms, bonds, node_mask, pair_mask, context = make_inputs(flow, [10, 6, 3])
    clean = flow.log_likelihood(atoms, bonds, node_mask, pair_mask, context)["log_prob"]
    dirty_atoms, dirty_bonds = atoms.clone(), bonds.clone()
    dirty_atoms[~node_mask] = torch.randn_like(dirty_atoms[~node_mask]) * 100
    dirty_bonds[~pair_mask] = torch.randn_like(dirty_bonds[~pair_mask]) * 100
    dirty = flow.log_likelihood(dirty_atoms, dirty_bonds, node_mask, pair_mask,
                                context)["log_prob"]
    assert float((clean - dirty).abs().max()) < 1e-3


def test_the_dimension_count_and_the_dequantization_bound():
    flow = tiny_flow()
    counts = [10, 6, 3]
    atoms, bonds, node_mask, pair_mask, context = make_inputs(flow, counts)
    result = flow.log_likelihood(atoms, bonds, node_mask, pair_mask, context)
    expected = [float(c * NUM_ATOM_TYPES + c * (c - 1) // 2 * NUM_BOND_TYPES)
                for c in counts]
    assert result["dimensions"].tolist() == expected
    # log P(discrete) >= log p(continuous) + D log s, and log s < 0 for s < 1
    assert bool((result["discrete_lower_bound"] <= result["log_prob"]).all())


def test_the_upper_triangle_round_trips():
    width = 6
    rows, cols = upper_triangle_indices(width)
    dense = torch.zeros(2, width, width)
    dense[:, rows, cols] = torch.arange(1.0, rows.numel() + 1).expand(2, -1)
    dense = dense + dense.transpose(1, 2)
    packed = pack_upper(dense, width)
    restored = unpack_upper(packed.unsqueeze(-1), width).squeeze(-1)
    assert torch.allclose(restored, dense)
    assert torch.allclose(restored, restored.transpose(1, 2))
    assert float(restored.diagonal(dim1=1, dim2=2).abs().max()) == 0.0


def test_the_pair_mask_needs_both_atoms_real():
    mask = pair_mask_from_counts([3], 5)
    rows, cols = upper_triangle_indices(5)
    for position in range(rows.numel()):
        expected = bool(rows[position] < 3 and cols[position] < 3)
        assert bool(mask[0, position]) == expected


# ------------------------------------------------- symmetry, exact N, validity ---
def test_samples_are_symmetric_with_exact_n_and_no_self_bonds():
    flow = tiny_flow()
    counts = [10, 7, 4, 2]
    context = flow.conditioning(torch.randn(len(counts), 2))
    make_inputs(flow, counts)
    sampled = flow.sample(counts, context, temperature=0.8,
                          generator=torch.Generator().manual_seed(1))
    for graph, count in zip(flow.discrete_graphs(sampled), counts):
        bonds = graph["bond_matrix"]
        assert graph["atom_types"].shape == (count,), "exact N is structural"
        assert bonds.shape == (count, count)
        assert np.array_equal(bonds, bonds.T), "bond matrix must be symmetric"
        assert (np.diag(bonds) == 0).all(), "no self bonds"
        assert bonds.min() >= 0 and bonds.max() < NUM_BOND_TYPES
        assert graph["atom_types"].min() >= 0
        assert graph["atom_types"].max() < NUM_ATOM_TYPES


def test_sampling_rejects_an_impossible_n():
    flow = tiny_flow(max_nodes=8)
    context = flow.conditioning(torch.zeros(1, 2))
    with pytest.raises(ValueError, match="must lie in"):
        flow.sample([9], context)
    with pytest.raises(ValueError, match="must lie in"):
        flow.sample([0], context)


def test_the_corrector_never_leaves_an_over_valent_atom():
    """Over-valent by construction: three oxygens (cap 2) with triple bonds."""
    atoms = np.array([2, 2, 2])
    bonds = np.array([[0, 3, 3], [3, 0, 3], [3, 3, 0]])
    fixed_atoms, fixed_bonds, info = valency_correction(atoms, bonds)
    caps = np.asarray([ATOM_VALENCE[int(a)] for a in fixed_atoms])
    assert (fixed_bonds.sum(axis=1) <= caps).all(), "correction left an over-valent atom"
    assert info["bond_orders_removed"] > 0
    assert info["changed"] is True
    assert np.array_equal(fixed_bonds, fixed_bonds.T)


def test_the_corrector_keeps_the_largest_component_and_says_so():
    """Correction can DROP ATOMS, so exact N must be measured after it too."""
    atoms = np.array([0, 0, 0, 0, 0])
    bonds = np.zeros((5, 5), dtype=np.int64)
    for a, b in ((0, 1), (1, 2)):
        bonds[a, b] = bonds[b, a] = 1
    bonds[3, 4] = bonds[4, 3] = 1
    fixed_atoms, fixed_bonds, info = valency_correction(atoms, bonds)
    assert info["atoms_kept"] == 3
    assert info["atoms_total"] == 5
    assert info["components_before_pruning"] == 2
    assert fixed_bonds.shape == (3, 3)
    assert "can change N" in info["note"]


def test_a_valid_graph_is_left_alone_by_the_corrector():
    atoms = np.array([0, 0, 0])
    bonds = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    fixed_atoms, fixed_bonds, info = valency_correction(atoms, bonds)
    assert info["changed"] is False
    assert np.array_equal(fixed_bonds, bonds)
    assert np.array_equal(fixed_atoms, atoms)


def test_records_report_raw_and_corrected_separately():
    graphs = [{"atom_types": np.array([2, 2, 2]),
               "bond_matrix": np.array([[0, 3, 3], [3, 0, 3], [3, 3, 0]]),
               "finite": True}]
    records = graphs_to_records(graphs, requested_n=[3], correct=True)
    record = records[0]
    assert "raw" in record and "corrected" in record
    assert record["raw"]["valid"] is False, "the over-valent graph is not valid raw"
    assert "exact_n" in record["raw"] and "exact_n" in record["corrected"]
    assert record["corrected"]["correction"]["bond_orders_removed"] > 0


def test_a_non_finite_sample_is_invalid_and_never_dressed_up():
    graphs = [{"atom_types": np.array([0, 0]), "bond_matrix": np.zeros((2, 2), dtype=int),
               "finite": False}]
    record = graphs_to_records(graphs, requested_n=[2])[0]
    assert record["nonfinite"] is True
    assert record["raw"]["valid"] is False
    assert record["raw"]["fallback_reason"] == "nonfinite_sample"


# --------------------------------------------------------------- conditioning ----
def test_the_condition_reaches_the_samples_once_the_couplings_are_awake():
    flow = tiny_flow(seed=11)
    make_inputs(flow, [10, 10])
    with torch.no_grad():                        # wake the zero-initialised couplings
        for coupling in (m for m in flow.modules() if isinstance(m, ConditionalCoupling)):
            for parameter in coupling.subnet.net[-1].parameters():
                parameter.add_(torch.randn_like(parameter) * 0.5)
    low = flow.conditioning(torch.tensor([[0.0, -2.0]]))
    high = flow.conditioning(torch.tensor([[0.0, 2.0]]))
    first = flow.sample([10], low, temperature=1.0,
                        generator=torch.Generator().manual_seed(7))
    second = flow.sample([10], high, temperature=1.0,
                         generator=torch.Generator().manual_seed(7))
    assert not torch.equal(first["bonds_continuous"], second["bonds_continuous"]), (
        "identical noise and different QED produced identical bonds: the condition is "
        "not reaching the output")


def test_a_zero_initialised_model_is_the_identity_in_its_couplings():
    """Couplings start as the identity, so training begins from a well-posed map."""
    flow = tiny_flow(seed=2)
    atoms, bonds, node_mask, pair_mask, context = make_inputs(flow, [10])
    flow_fresh = tiny_flow(seed=2)
    flow_fresh.eval()
    for layer in flow_fresh.bond_layers:
        scale, shift = layer["coupling"].subnet(
            bonds[..., :layer["coupling"].passthrough], pair_mask, context)
        assert float(scale.abs().max()) == 0.0
        assert float(shift.abs().max()) == 0.0


def test_the_null_condition_is_a_separate_path():
    flow = tiny_flow()
    values = torch.randn(4, 2)
    present = torch.tensor([True, False, True, False])
    context = flow.conditioning(values, present)
    direct = flow.conditioning(values)
    assert torch.allclose(context[0], direct[0])
    assert torch.allclose(context[1], flow.null_condition)
    assert torch.allclose(context[3], flow.null_condition)
    assert torch.allclose(flow.null_context(3, "cpu")[0], flow.null_condition)


def test_dropout_keeps_everything_when_disabled():
    flow = ConditionalGraphFlow(GraphFlowConfig(max_nodes=6, bond_blocks=1, atom_blocks=1,
                                                hidden=16, condition_dim=8,
                                                condition_dropout=0.0))
    assert bool(flow.dropout_mask(16, "cpu").all())


def test_neutralising_the_coupling_condition_is_reversible():
    flow = tiny_flow()
    couplings = [m for m in flow.modules() if isinstance(m, ConditionalCoupling)]
    assert all(bool(m.condition_enabled) for m in couplings)
    with neutralise_coupling_condition(flow):
        assert all(not bool(m.condition_enabled) for m in couplings)
    assert all(bool(m.condition_enabled) for m in couplings)


def test_neutralising_restores_state_even_on_an_exception():
    flow = tiny_flow()
    couplings = [m for m in flow.modules() if isinstance(m, ConditionalCoupling)]
    with pytest.raises(RuntimeError):
        with neutralise_coupling_condition(flow):
            raise RuntimeError("boom")
    assert all(bool(m.condition_enabled) for m in couplings)


def test_neutralising_leaves_the_prior_conditional():
    """The point of the ablation: couplings off, conditional prior still on."""
    flow = tiny_flow()
    low = flow.conditioning(torch.tensor([[0.0, -2.0]]))
    high = flow.conditioning(torch.tensor([[0.0, 2.0]]))
    with torch.no_grad():
        flow.bond_prior.mu_head.weight.add_(torch.randn_like(
            flow.bond_prior.mu_head.weight) * 0.5)
    with neutralise_coupling_condition(flow):
        a = flow.bond_prior.parameters_for(low)[0]
        b = flow.bond_prior.parameters_for(high)[0]
    assert not torch.allclose(a, b), "the conditional prior must survive the ablation"


# ------------------------------------------------------------------ stability ----
def test_the_density_is_finite_for_degenerate_inputs():
    flow = tiny_flow()
    counts = [10, 2]
    width = flow.config.max_nodes
    node_mask = torch.zeros(len(counts), width, dtype=torch.bool)
    for row, count in enumerate(counts):
        node_mask[row, :count] = True
    pair_mask = pair_mask_from_counts(counts, width)
    atoms = torch.zeros(len(counts), width, dtype=torch.long)
    bonds = torch.zeros(len(counts), width, width, dtype=torch.long)
    x_atom, x_bond = dequantize(atoms, bonds, node_mask, pair_mask,
                                num_atom_types=flow.config.num_atom_types,
                                num_bond_types=flow.config.num_bond_types,
                                noise_scale=flow.config.noise_scale)
    flow.train()
    result = flow.log_likelihood(x_atom, x_bond, node_mask, pair_mask,
                                 flow.conditioning(torch.zeros(len(counts), 2)))
    assert torch.isfinite(result["log_prob"]).all()
    assert torch.isfinite(result["discrete_lower_bound"]).all()


def test_gradients_are_finite_and_non_zero():
    flow = tiny_flow()
    atoms, bonds, node_mask, pair_mask, context = make_inputs(flow, [10, 5])
    flow.train()
    loss = -flow.log_likelihood(atoms, bonds, node_mask, pair_mask,
                                context)["log_prob"].mean()
    loss.backward()
    norms = [float(p.grad.norm()) for p in flow.parameters() if p.grad is not None]
    assert norms and all(np.isfinite(n) for n in norms)
    assert sum(norms) > 0.0


def test_dequantization_stays_inside_its_box_and_only_on_real_entries():
    flow = tiny_flow()
    counts = [10, 4]
    width = flow.config.max_nodes
    node_mask = torch.zeros(len(counts), width, dtype=torch.bool)
    for row, count in enumerate(counts):
        node_mask[row, :count] = True
    pair_mask = pair_mask_from_counts(counts, width)
    atoms = torch.randint(0, NUM_ATOM_TYPES, (len(counts), width)) * node_mask
    dense = torch.randint(0, NUM_BOND_TYPES, (len(counts), width, width))
    dense = torch.triu(dense, 1)
    dense = dense + dense.transpose(1, 2)
    x_atom, x_bond = dequantize(atoms, dense, node_mask, pair_mask,
                                num_atom_types=NUM_ATOM_TYPES,
                                num_bond_types=NUM_BOND_TYPES, noise_scale=0.6)
    assert float(x_atom.min()) >= 0.0 and float(x_atom.max()) <= 1.6 + 1e-6
    assert float(x_bond.min()) >= 0.0 and float(x_bond.max()) <= 1.6 + 1e-6
    assert float(x_atom[~node_mask].abs().max()) == 0.0
    assert float(x_bond[~pair_mask].abs().max()) == 0.0


# --------------------------------------------------------- the null arm and gap ---
def test_shuffling_qed_preserves_n_and_the_qed_marginal(tmp_path):
    from zinc_gnf.conditioning import load_condition_scaler
    from zinc_gnf.dataset import load_training_splits
    from zinc_gnf.trainer_graph_flow import shuffle_qed_within_n

    data = write_dataset(tmp_path / "data")
    payload = pickle.loads((Path(data) / "splits.pkl").read_bytes())
    (Path(data) / "scaler.json").write_text(
        json.dumps(ConditionScaler.fit(payload["train"]).to_dict()))
    scaler = load_condition_scaler(data)
    train, _, _ = load_training_splits(data)
    shuffled = shuffle_qed_within_n(train, seed=5, scaler=scaler)
    assert [r["n_node"] for r in shuffled] == [r["n_node"] for r in train], \
        "N must be preserved exactly"
    assert sorted(r["qed"] for r in shuffled) == sorted(r["qed"] for r in train), \
        "the QED marginal must be preserved exactly"
    for record in shuffled:
        assert record["condition"] is not None
    # within each N bucket the multiset of QED values is unchanged
    for count in {r["n_node"] for r in train}:
        before = sorted(r["qed"] for r in train if r["n_node"] == count)
        after = sorted(r["qed"] for r in shuffled if r["n_node"] == count)
        assert before == after


# -------------------------------------------------------- training end to end ----
@pytest.fixture
def fixture_data(tmp_path):
    from zinc_gnf.conditioning import load_condition_scaler

    data = write_dataset(tmp_path / "data")
    payload = pickle.loads((Path(data) / "splits.pkl").read_bytes())
    (Path(data) / "scaler.json").write_text(
        json.dumps(ConditionScaler.fit(payload["train"]).to_dict()))
    return {"dir": data, "scaler": load_condition_scaler(data)}


def test_training_runs_resumes_and_reaches_the_requested_step(fixture_data, tmp_path):
    script = load_script("23_train_graph_flow.py")
    run = tmp_path / "real"
    common = ["--output-dir", str(run), "--data-dir", str(fixture_data["dir"]),
              "--batch-size", "4", "--bond-blocks", "2", "--atom-blocks", "2",
              "--hidden", "32", "--condition-dim", "16", "--val-molecules", "8",
              "--warmup-steps", "1", "--device", "cpu", "--max-nodes", "18"]
    assert script.main(common + ["--max-steps", "4", "--eval-every", "2",
                                 "--save-every", "2", "--log-every", "2"]) == 0
    first = json.loads((run / "training_summary.json").read_text())
    assert first["step"] == 4 and first["completed"] is True
    assert first["arm"] == "real_conditions"

    # resume to 8 and confirm it continued rather than restarted
    assert script.main(common + ["--max-steps", "8", "--eval-every", "2",
                                 "--save-every", "2", "--log-every", "2"]) == 0
    second = json.loads((run / "training_summary.json").read_text())
    assert second["step"] == 8
    assert "resumed from" in (run / "train.log").read_text()
    assert (run / "last.pt").exists() and (run / "best.pt").exists()

    # re-running a finished arm is a no-op
    assert script.main(common + ["--max-steps", "8", "--eval-every", "2",
                                 "--save-every", "2", "--log-every", "2"]) == 0

    history = second["history"]
    assert history and "condition_gap_per_atom" in history[-1]
    assert "condition_win_rate" in history[-1]
    trend = second["condition_gap_trend"]
    assert trend["status"] in {"unknown", "rising_undertrained", "plateaued_zero",
                              "plateaued_positive"}


def test_the_null_arm_trains_and_is_labelled(fixture_data, tmp_path):
    script = load_script("23_train_graph_flow.py")
    run = tmp_path / "null"
    assert script.main([
        "--output-dir", str(run), "--data-dir", str(fixture_data["dir"]),
        "--max-steps", "4", "--batch-size", "4", "--bond-blocks", "2",
        "--atom-blocks", "2", "--hidden", "32", "--condition-dim", "16",
        "--val-molecules", "8", "--eval-every", "2", "--save-every", "2",
        "--log-every", "2", "--warmup-steps", "1", "--device", "cpu",
        "--max-nodes", "18", "--shuffle-qed"]) == 0
    summary = json.loads((run / "training_summary.json").read_text())
    assert summary["arm"] == "shuffled_qed"
    assert summary["train_config"]["shuffle_qed"] is True
    assert "permuted WITHIN N" in (run / "train.log").read_text()


def test_training_validates_its_arguments(fixture_data, tmp_path):
    script = load_script("23_train_graph_flow.py")
    assert script.main(["--output-dir", str(tmp_path / "a"),
                        "--data-dir", str(fixture_data["dir"]),
                        "--max-steps", "0"]) == 4
    assert script.main(["--output-dir", str(tmp_path / "b"),
                        "--data-dir", str(fixture_data["dir"]),
                        "--condition-dropout", "1.5"]) == 4
    assert script.main(["--output-dir", str(tmp_path / "c"),
                        "--data-dir", str(tmp_path / "ghost")]) == 2


def test_a_molecule_larger_than_max_nodes_is_refused(fixture_data, tmp_path):
    script = load_script("23_train_graph_flow.py")
    assert script.main(["--output-dir", str(tmp_path / "small"),
                        "--data-dir", str(fixture_data["dir"]),
                        "--max-steps", "2", "--max-nodes", "4",
                        "--device", "cpu"]) == 4


def test_generation_is_native_and_reports_its_guarantees(fixture_data, tmp_path):
    from zinc_gnf.trainer_graph_flow import load_graph_flow

    script = load_script("23_train_graph_flow.py")
    run = tmp_path / "run"
    assert script.main([
        "--output-dir", str(run), "--data-dir", str(fixture_data["dir"]),
        "--max-steps", "4", "--batch-size", "4", "--bond-blocks", "2",
        "--atom-blocks", "2", "--hidden", "32", "--condition-dim", "16",
        "--val-molecules", "8", "--eval-every", "4", "--save-every", "4",
        "--log-every", "4", "--warmup-steps", "1", "--device", "cpu",
        "--max-nodes", "18"]) == 0
    model, meta = load_graph_flow(run / "last.pt")
    assert meta["step"] == 4
    output = generate_conditional(model, scaler=fixture_data["scaler"], n=9, qed=0.8,
                                  count=8, temperature=0.8, seed=3)
    assert len(output["records"]) == 8
    assert output["settings"]["decoders"] == ["raw", "corrected"]
    assert "no candidate selection" in output["settings"]["note"]
    for record in output["records"]:
        assert record["requested_n"] == 9
        assert len(record["atom_types"]) == 9, "exact N before any chemistry"
    # determinism: same seed, same molecules
    again = generate_conditional(model, scaler=fixture_data["scaler"], n=9, qed=0.8,
                                 count=8, temperature=0.8, seed=3)
    assert [r["raw"]["smiles"] for r in again["records"]] == \
        [r["raw"]["smiles"] for r in output["records"]]
    # the unconditional reference is a separate path
    reference = generate_conditional(model, scaler=fixture_data["scaler"], n=9, qed=0.8,
                                     count=4, seed=3, unconditional=True)
    assert all(r["unconditional"] for r in reference["records"])


def test_the_evaluation_runs_and_reports_every_required_column(fixture_data, tmp_path):
    train = load_script("23_train_graph_flow.py")
    arms = {}
    for name, extra in (("real", []), ("null", ["--shuffle-qed"])):
        run = tmp_path / name
        assert train.main([
            "--output-dir", str(run), "--data-dir", str(fixture_data["dir"]),
            "--max-steps", "4", "--batch-size", "4", "--bond-blocks", "2",
            "--atom-blocks", "2", "--hidden", "32", "--condition-dim", "16",
            "--val-molecules", "8", "--eval-every", "2", "--save-every", "2",
            "--log-every", "2", "--warmup-steps", "1", "--device", "cpu",
            "--max-nodes", "18"] + extra) == 0
        arms[name] = run / "last.pt"

    output = tmp_path / "eval"
    assert load_script("24_evaluate_graph_flow.py").main([
        "--graph-flow-checkpoint", str(arms["real"]),
        "--null-checkpoint", str(arms["null"]),
        "--output-dir", str(output), "--data-dir", str(fixture_data["dir"]),
        "--target-n", "9", "--target-qed", "0.4,0.8", "--temperature", "0.8",
        "--count", "8", "--bootstrap-resamples", "100", "--qed-window", "0.5",
        "--seed", "17", "--device", "cpu"]) == 0
    payload = json.loads((output / "graph_flow_evaluation.json").read_text())
    assert payload["stage"] == 22
    assert set(payload["summary"]) == {"graph_flow", "graph_flow_null",
                                       "graph_flow_prior"}
    for arm in payload["summary"].values():
        for decoder in ("raw", "corrected"):
            entry = arm["decoders"][decoder]
            for key in ("strict_validity", "exact_n_rate", "qed_mae_weighted",
                        "hit_rate_tol_10", "hit_rate_tol_05", "within_n_correlation",
                        "response_slope", "aromatic_rings_mean", "uniqueness",
                        "novelty_vs_train", "internal_diversity", "repair_rate"):
                assert key in entry, (decoder, key)
            slope = entry["response_slope"]
            assert {"slope", "ci_low", "ci_high", "excludes_zero"} <= set(slope)
    assert payload["arms"]["graph_flow_prior"]["prior_only"] is True
    for arm in payload["cost"].values():
        assert arm["seconds_per_molecule"] > 0
    assert "NOT comparable" in payload["likelihood"]["warning"]
    text = (output / "graph_flow_evaluation.txt").read_text()
    assert "READ THIS TABLE FIRST" in text
    assert "no_oracle" in text and "no_selection" in text
    assert "PROOF OF CONCEPT" in text


def test_the_evaluation_validates_its_arguments(tmp_path):
    script = load_script("24_evaluate_graph_flow.py")
    assert script.main(["--graph-flow-checkpoint", str(tmp_path / "ghost.pt"),
                        "--output-dir", str(tmp_path / "o"), "--target-n", ""]) == 4
    assert script.main(["--graph-flow-checkpoint", str(tmp_path / "ghost.pt"),
                        "--output-dir", str(tmp_path / "o2")]) == 2


def test_the_response_slope_recovers_a_known_slope():
    """A cell set built with a known slope must be recovered, with a CI that excludes 0."""
    evaluate = load_script("24_evaluate_graph_flow.py")
    rng = np.random.default_rng(0)
    cells = []
    for n in (15, 23):
        for requested in (0.4, 0.6, 0.8):
            realised = 0.3 + 0.5 * requested + rng.normal(scale=0.01, size=200)
            cells.append({"n": n, "requested_qed": requested,
                          "realised": {"raw": realised.tolist()}})
    block = evaluate.response_slope(cells, "raw", resamples=200, seed=1)
    assert block["slope"] == pytest.approx(0.5, abs=0.05)
    assert block["excludes_zero"] is True
    assert block["ci_low"] < block["slope"] < block["ci_high"]


def test_the_response_slope_reports_no_response_as_zero():
    evaluate = load_script("24_evaluate_graph_flow.py")
    rng = np.random.default_rng(1)
    cells = [{"n": 23, "requested_qed": q,
              "realised": {"raw": (0.5 + rng.normal(scale=0.05, size=200)).tolist()}}
             for q in (0.4, 0.6, 0.8)]
    block = evaluate.response_slope(cells, "raw", resamples=200, seed=2)
    assert abs(block["slope"]) < 0.1
    assert block["excludes_zero"] is False


# ------------------------------------------------------------------- SLURM ------
def test_the_stage22_job_is_bounded_resumable_and_estimates_its_cost():
    path = REPOSITORY / "slurm" / "22_graph_flow_poc.sh"
    source = path.read_text(encoding="utf-8")
    directives = {}
    for line in source.splitlines():
        match = re.match(r"#SBATCH\s+--([\w-]+)(?:=(.*))?$", line.strip())
        if match:
            directives[match.group(1)] = (match.group(2) or "").strip()
    assert directives["gres"] == "gpu:1", "training needs a GPU"
    assert directives["time"] and directives["mem"]
    assert 'if [[ -z "${SLURM_JOB_ID:-}" ]]' in source, "must refuse the login node"
    assert "Runtime estimate" in source
    assert "Memory estimate" in source
    assert "Resumable" in source
    executable = "\n".join(line for line in source.splitlines()
                           if not line.lstrip().startswith("#") and "echo " not in line)
    assert "scripts/23_train_graph_flow.py" in executable
    assert "scripts/24_evaluate_graph_flow.py" in executable
    assert "--shuffle-qed" in executable, "the null arm must be part of the job"
    assert not re.search(r"(?m)^\s*(rm|mv)\s", executable)
    # it must not write into any stage 1-21 output directory
    for protected in ("decoder_information", "qed_pathway", "decoder_experiment",
                      "warm_start_stageA", "flow_nqed", "flow_uncond_ar"):
        assert f"OUT=\"${{OUT:-outputs/metrics/{protected}" not in source
        assert f"REAL_DIR=\"${{REAL_DIR:-outputs/checkpoints/{protected}" not in source
    assert "graph_flow" in source
    assert "READ-ONLY" in source
    assert "STEPS=" in source, "the budget must be overridable for a staged re-run"
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0


def test_the_stage22_job_reuses_the_stage17_checkpoint_read_only():
    source = (REPOSITORY / "slurm" / "22_graph_flow_poc.sh").read_text(encoding="utf-8")
    assert "warm_start_stageA" in source
    assert "--baseline-flow-checkpoint" in source


def test_the_architecture_review_states_what_is_not_established():
    """The review must not claim a published native-QED benchmark, because none exists."""
    text = (REPOSITORY / "docs" / "stage22_architecture_review.md").read_text(
        encoding="utf-8")
    assert "latent gradient ascent" in text
    assert "no published" in text.lower()
    for name in ("MoFlow", "GraphNVP", "GraphAF", "GraphDF", "cINN"):
        assert name in text
    assert "arxiv.org" in text, "the review must cite its sources"


# ===========================================================================
# Stage 22c: the defects found while diagnosing the stage-22a run, and the
# limitation that is NOT a defect but is also not undertraining.
# ===========================================================================
def test_the_representation_round_trips_on_real_molecules():
    """THE decisive representation test: real molecule -> tensors -> argmax -> molecule.

    Stage 22b reported raw validity 0.0012, which raises the question of whether padding,
    masking, symmetry, dequantization or discretisation are wrong. They are not: this
    drives real molecules - charged species, halogens, Kekulé aromatics - through exactly
    the path the trainer and sampler use and requires the canonical SMILES back.
    """
    from rdkit import Chem, rdBase

    from zinc_gnf.chemistry_charge_aware import (
        encode_zinc_smiles_charge_aware,
        graph_to_mol_charge_aware,
    )

    smiles = ["CC(=O)Nc1ccc(O)cc1", "c1ccccc1", "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
              "O=C(Nc1ccc(F)cc1)c1cc(Cl)ccc1Br", "CCOC(=O)c1ccccc1", "c1ccsc1"]
    width = 38
    rows, cols = (t.numpy() for t in upper_triangle_indices(width))
    checked = 0
    for text in smiles:
        outcome = encode_zinc_smiles_charge_aware(text)
        record = outcome[0] if isinstance(outcome, tuple) else outcome
        if not isinstance(record, dict) or "atom_types" not in record:
            continue
        checked += 1
        truth_atoms = np.asarray(record["atom_types"], dtype=np.int64)
        truth_bonds = np.asarray(record["bond_matrix"], dtype=np.int64)
        count = int(truth_atoms.size)

        atoms = torch.zeros(1, width, dtype=torch.long)
        atoms[0, :count] = torch.from_numpy(truth_atoms)
        bonds = torch.zeros(1, width, width, dtype=torch.long)
        bonds[0, :count, :count] = torch.from_numpy(truth_bonds)
        node_mask = torch.zeros(1, width, dtype=torch.bool)
        node_mask[0, :count] = True
        pair_mask = pair_mask_from_counts([count], width)

        x_atom, x_bond = dequantize(
            atoms, bonds, node_mask, pair_mask, num_atom_types=NUM_ATOM_TYPES,
            num_bond_types=NUM_BOND_TYPES, noise_scale=0.6,
            generator=torch.Generator().manual_seed(0))
        decoded_atoms = x_atom.argmax(-1)[0, :count].numpy()
        choice = x_bond.argmax(-1)[0].numpy()
        decoded_bonds = np.zeros((count, count), dtype=np.int64)
        for position in np.flatnonzero(pair_mask[0].numpy() & (choice > 0)):
            i, j = int(rows[position]), int(cols[position])
            decoded_bonds[i, j] = decoded_bonds[j, i] = int(choice[position])

        assert np.array_equal(decoded_atoms, truth_atoms), text
        assert np.array_equal(decoded_bonds, truth_bonds), text
        with rdBase.BlockLogs():
            rebuilt = Chem.MolToSmiles(
                graph_to_mol_charge_aware(decoded_atoms, decoded_bonds))
            original = Chem.MolToSmiles(Chem.MolFromSmiles(record["smiles"]))
        assert rebuilt == original, f"{text}: {rebuilt} != {original}"
    assert checked >= 5, "the round trip must actually have been exercised"


def test_the_bond_conditioner_is_blind_to_topology_at_a_fixed_degree_sequence():
    """A LIMITATION, verified - and one that training cannot remove.

    The bond coupling's conditioner gives a pair only its own channels plus the per-atom
    sums `r_i + r_j` and `|r_i - r_j|`. Two graphs with the same degree sequence are
    therefore identical to it: a six-ring and two three-rings get byte-identical scales
    and shifts on every pair they share. Connectivity and ring closure are exactly the
    properties that need that distinction, so no amount of further training can make this
    conditioner express them.

    Note what this is NOT: the symmetry argument that motivated the sum aggregates is
    unnecessary. Only the strict upper triangle is modelled and `unpack_upper` mirrors it,
    so each pair appears once and matrix symmetry holds whatever the conditioner does.
    """
    torch.manual_seed(0)
    width = 6
    subnet = BondSubnet(passthrough=2, outputs=2, hidden=32, condition_dim=8, width=width)
    with torch.no_grad():
        for parameter in subnet.net[-1].parameters():
            parameter.add_(torch.randn_like(parameter) * 0.5)
    rows, cols = upper_triangle_indices(width)
    position_of = {
        (min(a, b), max(a, b)): int(torch.nonzero(
            (rows == min(a, b)) & (cols == max(a, b))).flatten()[0])
        for a in range(width) for b in range(width) if a != b}

    def configuration(edges):
        values = torch.zeros(1, rows.numel(), 2)
        for a, b in edges:
            values[0, position_of[(min(a, b), max(a, b))], 0] = 1.0
        return values

    ring = configuration([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)])
    triangles = configuration([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])

    def degrees(values):
        return [int(values[0, (rows == i) | (cols == i), 0].sum()) for i in range(width)]

    assert degrees(ring) == degrees(triangles) == [2] * width
    mask = torch.ones(1, rows.numel(), dtype=torch.bool)
    context = torch.randn(1, 8)
    with torch.no_grad():
        ring_scale, ring_shift = subnet(ring, mask, context)
        triangle_scale, triangle_shift = subnet(triangles, mask, context)
    for edge in ((0, 1), (1, 2), (3, 4), (4, 5)):          # present in BOTH graphs
        position = position_of[edge]
        assert torch.equal(ring_scale[0, position], triangle_scale[0, position]), edge
        assert torch.equal(ring_shift[0, position], triangle_shift[0, position]), edge


def test_the_training_script_default_tracks_the_dataclass():
    """The stage-22a defect: the script hard-coded 2.0 while the dataclass said 0.75."""
    script = load_script("23_train_graph_flow.py")
    arguments = script.parse_args(["--output-dir", "x"])
    reference = GraphFlowConfig()
    assert arguments.max_log_scale == reference.max_log_scale
    assert arguments.max_shift == reference.max_shift
    assert arguments.noise_scale == reference.noise_scale


def test_resuming_with_a_different_configuration_is_refused(fixture_data, tmp_path):
    """`max_log_scale` is a plain float, so `load_state_dict` would accept it silently.

    That is how the stage-22a run came to be trained at 2.0 while the dataclass said
    0.75: nothing compared the two. Resuming must refuse rather than quietly implement a
    different function with the same weights.
    """
    script = load_script("23_train_graph_flow.py")
    run = tmp_path / "run"
    common = ["--output-dir", str(run), "--data-dir", str(fixture_data["dir"]),
              "--batch-size", "4", "--bond-blocks", "2", "--atom-blocks", "2",
              "--hidden", "32", "--condition-dim", "16", "--val-molecules", "8",
              "--warmup-steps", "1", "--device", "cpu", "--max-nodes", "18",
              "--eval-every", "2", "--save-every", "2", "--log-every", "2"]
    assert script.main(common + ["--max-steps", "2", "--max-log-scale", "0.75"]) == 0
    # same directory, different numerical configuration -> must fail, not continue
    assert script.main(common + ["--max-steps", "4", "--max-log-scale", "2.0"]) == 4
    log = (run / "train.log").read_text()
    assert "DIFFERENT model configuration" in log
    assert "max_log_scale" in log
    # the checkpoint's own value still resumes cleanly
    assert script.main(common + ["--max-steps", "4", "--max-log-scale", "0.75"]) == 0


def test_a_saturated_stack_at_a_large_log_scale_breaks_the_inverse():
    """Quantifies why the prior-only ablation reported 1.4e-02 while the full arm had 3.3e-05.

    The subnets were trained WITH a context vector; neutralising it pushes them
    off-distribution, the log-scales saturate toward their bound, and the inverse
    condition number is ~exp(blocks * max_log_scale). At 2.0 over 6 blocks that is
    exp(12) ~ 1.6e5 against float32's ~1e-7 precision, leaving no margin.
    """
    errors = {}
    for bound in (0.75, 2.0):
        torch.manual_seed(5)
        flow = ConditionalGraphFlow(GraphFlowConfig(
            max_nodes=12, bond_blocks=6, atom_blocks=6, hidden=48, condition_dim=16,
            condition_hidden=32, max_log_scale=bound, max_shift=3.0))
        make_inputs(flow, [12, 12])
        _wake_couplings(flow, scale=1.2)
        context = flow.conditioning(torch.randn(2, 2))
        with neutralise_coupling_condition(flow):
            sampled = flow.sample([12, 12], context, temperature=1.0,
                                  generator=torch.Generator().manual_seed(1))
        errors[bound] = max(sampled["latent_round_trip_relative_error"].values())
    assert errors[0.75] < 1e-2, errors
    assert errors[2.0] > errors[0.75] * 100, (
        f"a large log-scale bound must visibly degrade the inverse: {errors}")


def test_the_diagnostic_attributes_raw_failures_and_runs_read_only(fixture_data, tmp_path):
    train = load_script("23_train_graph_flow.py")
    run = tmp_path / "run"
    assert train.main([
        "--output-dir", str(run), "--data-dir", str(fixture_data["dir"]),
        "--max-steps", "4", "--batch-size", "4", "--bond-blocks", "2",
        "--atom-blocks", "2", "--hidden", "32", "--condition-dim", "16",
        "--val-molecules", "8", "--eval-every", "4", "--save-every", "4",
        "--log-every", "4", "--warmup-steps", "1", "--device", "cpu",
        "--max-nodes", "18"]) == 0
    before = (run / "last.pt").read_bytes()

    output = tmp_path / "diag"
    assert load_script("25_graph_flow_diagnostics.py").main([
        "--checkpoint", str(run / "last.pt"), "--output-dir", str(output),
        "--data-dir", str(fixture_data["dir"]), "--target-n", "9",
        "--target-qed", "0.8", "--temperature", "0.85", "--count", "16",
        "--real-molecules", "20", "--seed", "17"]) == 0
    assert (run / "last.pt").read_bytes() == before, "the diagnostic must be READ-ONLY"

    payload = json.loads((output / "graph_flow_diagnostics.json").read_text())
    assert payload["stage"] == "22c"
    assert payload["inverse_condition_number_bound"] > 0
    assert payload["cells"]
    cell = payload["cells"][0]
    for key in ("bonds_mean", "bond_density_mean", "bond_type_fraction",
                "atom_type_fraction", "causes", "components_mean",
                "largest_component_mean", "largest_component_fraction_of_n",
                "over_valent_atoms_mean", "correction_atoms_lost_mean",
                "correction_exact_n_rate", "latent_round_trip"):
        assert key in cell, key
    # every sample must be attributed to exactly one named cause
    assert sum(cell["causes"].values()) == cell["samples"]
    assert set(cell["causes"]) <= {"valid", "over_valent_only", "disconnected_only",
                                   "over_valent_and_disconnected",
                                   "other_sanitisation_failure"}
    assert set(payload["inverse_stability"]) == {"with_context", "condition_neutralised"}
    for block in payload["inverse_stability"].values():
        assert "log_scale_fraction_at_bound" in block
        assert "latent_round_trip" in block


def test_the_diagnostic_validates_its_arguments(tmp_path):
    script = load_script("25_graph_flow_diagnostics.py")
    assert script.main(["--checkpoint", str(tmp_path / "ghost.pt"),
                        "--output-dir", str(tmp_path / "o"), "--target-n", ""]) == 4
    assert script.main(["--checkpoint", str(tmp_path / "ghost.pt"),
                        "--output-dir", str(tmp_path / "o2")]) == 2
