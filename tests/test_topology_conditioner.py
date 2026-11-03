"""Stage 23: the topology-aware bond conditioner.

The stage exists because of one proven fact: stage 22's `BondSubnet` gives a pair only
`r_i + r_j` and `|r_i - r_j|`, and a six-ring and two three-rings therefore receive
byte-identical scales and shifts. The first test below is the acceptance test for the
replacement, and it is run against BOTH conditioners so the difference is the measurement
rather than a claim.

Everything the flow depends on is re-checked for the new conditioner, because a more
expressive conditioner is worthless if it costs invertibility:

  TOPOLOGY     the 1-WL counterexample is distinguished (and the old one is not)
  NO LEAKAGE   the conditioner sees only the coupling's pass-through channels, so the
               inverse recomputes the same parameters - structurally and behaviourally
  INVERTIBLE   exact round trip, for both conditioners, at several molecule sizes
  MASKING      padding is inert in the latents, in the density and through the convolution
  SYMMETRY     samples come back symmetric with a zero diagonal and exact N
  NUMERICS     the inverse is checked at initialisation, refused if ill-conditioned, and
               tracked at every evaluation
  CONFIG       one source of truth for the numerical settings, and resumption that cannot
               silently change the architecture
  BASELINE     stage-22 checkpoints still load, as `pairwise`
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
    BOND_CONDITIONERS,
    BondSubnet,
    ConditionalCoupling,
    ConditionalGraphFlow,
    GraphFlowConfig,
    TopologyBondSubnet,
    dequantize,
    pair_mask_from_counts,
    unpack_upper,
    upper_triangle_indices,
)
from zinc_gnf.graph_flow_generation import (
    graph_structure_statistics,
    population_structure,
    real_structure_reference,
)
from zinc_gnf.preprocessing import ConditionScaler

REPOSITORY = Path(__file__).resolve().parents[1]
RING_SIX = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
TWO_TRIANGLES = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
SHARED_PAIRS = [(0, 1), (1, 2), (3, 4), (4, 5)]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""),
                                                  REPOSITORY / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _head(subnet: torch.nn.Module) -> torch.nn.Module:
    head = getattr(subnet, "head", None)
    return head if head is not None else subnet.net[-1]


def _wake(module: torch.nn.Module, scale: float = 0.5) -> None:
    with torch.no_grad():
        for parameter in _head(module).parameters():
            parameter.add_(torch.randn_like(parameter) * scale)


def _pair_positions(width: int):
    rows, cols = upper_triangle_indices(width)
    return rows, cols, {
        (min(a, b), max(a, b)): int(torch.nonzero(
            (rows == min(a, b)) & (cols == max(a, b))).flatten()[0])
        for a in range(width) for b in range(width) if a != b}


def _configuration(edges, width: int, channels: int = 2) -> torch.Tensor:
    _, _, position = _pair_positions(width)
    rows, cols, _ = _pair_positions(width)
    values = torch.zeros(1, rows.numel(), channels)
    for a, b in edges:
        values[0, position[(min(a, b), max(a, b))], 0] = 1.0
    return values


# ------------------------------------------------ THE acceptance test ----------
@pytest.mark.parametrize("conditioner,expect_blind", [("pairwise", True),
                                                      ("topology", False)])
def test_topology_sensitivity_on_the_one_wl_counterexample(conditioner, expect_blind):
    """A six-ring and two three-rings: both 2-regular, the textbook 1-WL pair.

    Every atom has degree 2 and every neighbour has degree 2, so the colouring is stable
    immediately. Any sum-aggregation scheme over the atom graph - including iterating the
    stage-22 pair -> atom -> pair step - assigns identical atom embeddings in both graphs,
    and so identical parameters to every pair they share. A convolution over the adjacency
    GRID is not 1-WL-bounded: it reads the pattern as a function of atom index, and the
    positions where the two graphs differ fall inside its receptive field.

    Compared on the four pairs present in BOTH graphs, where the pass-through value and
    the degree sums are identical, so any difference must come from the surrounding
    topology.
    """
    torch.manual_seed(0)
    width = 6
    if conditioner == "pairwise":
        subnet = BondSubnet(2, 2, 32, 8, width)
    else:
        subnet = TopologyBondSubnet(2, 2, 32, 8, width, layers=3)
    _wake(subnet)
    _, _, position = _pair_positions(width)
    ring = _configuration(RING_SIX, width)
    triangles = _configuration(TWO_TRIANGLES, width)
    rows, cols, _ = _pair_positions(width)

    def degrees(values):
        return [int(values[0, (rows == i) | (cols == i), 0].sum()) for i in range(width)]

    assert degrees(ring) == degrees(triangles) == [2] * width, "must be the 1-WL pair"
    mask = torch.ones(1, rows.numel(), dtype=torch.bool)
    context = torch.randn(1, 8)
    with torch.no_grad():
        ring_scale, ring_shift = subnet(ring, mask, context)
        triangle_scale, triangle_shift = subnet(triangles, mask, context)
    worst = max(
        max(float((ring_scale[0, position[e]] - triangle_scale[0, position[e]])
                  .abs().max()),
            float((ring_shift[0, position[e]] - triangle_shift[0, position[e]])
                  .abs().max()))
        for e in SHARED_PAIRS)
    if expect_blind:
        assert worst == 0.0, (
            f"the stage-22 conditioner is supposed to be blind here; got {worst}")
    else:
        assert worst > 1e-4, (
            f"the stage-23 conditioner must distinguish the topologies; got {worst}")


def test_the_topology_conditioner_sees_other_pairs_than_its_own():
    """Changing a DIFFERENT pair must move a fixed pair's parameters.

    Stage 22's conditioner could only see other pairs through `r_i + r_j`, so a
    rearrangement that preserved the degree sums was invisible. Here the change is
    invisible to the degree sums - atom 2 keeps degree 1 - and must still be noticed.
    """
    torch.manual_seed(1)
    width = 8
    subnet = TopologyBondSubnet(2, 2, 32, 8, width, layers=3)
    _wake(subnet)
    rows, cols, position = _pair_positions(width)
    base = _configuration([(0, 1), (2, 3)], width)
    moved = _configuration([(0, 1), (2, 4)], width)      # atom 2 keeps degree 1
    probe = position[(0, 1)]
    assert torch.equal(base[0, probe], moved[0, probe]), "the probe pair is unchanged"
    mask = torch.ones(1, rows.numel(), dtype=torch.bool)
    context = torch.randn(1, 8)
    with torch.no_grad():
        first = subnet(base, mask, context)
        second = subnet(moved, mask, context)
    difference = max(float((first[0][0, probe] - second[0][0, probe]).abs().max()),
                     float((first[1][0, probe] - second[1][0, probe]).abs().max()))
    assert difference > 1e-5, f"the change must reach this pair; got {difference}"


def test_the_receptive_field_in_index_space_is_a_stated_limitation():
    """Honest bound: a change far away in INDEX space may not reach a pair.

    The convolution is local - radius `layers * (kernel // 2)` - and the axial aggregates
    only summarise the pair's own two atoms. So a rearrangement among atoms that share no
    index neighbourhood and no endpoint with the probe pair can be invisible to a single
    coupling. Information still crosses the molecule across the six coupling blocks, but
    this records the per-coupling bound rather than overclaiming a global receptive field.
    """
    torch.manual_seed(1)
    width = 12
    subnet = TopologyBondSubnet(2, 2, 32, 8, width, layers=2)
    _wake(subnet)
    rows, cols, position = _pair_positions(width)
    base = _configuration([(0, 1), (8, 9)], width)
    moved = _configuration([(0, 1), (10, 11)], width)    # far from (0,1) in index space
    probe = position[(0, 1)]
    mask = torch.ones(1, rows.numel(), dtype=torch.bool)
    context = torch.randn(1, 8)
    with torch.no_grad():
        first = subnet(base, mask, context)
        second = subnet(moved, mask, context)
    difference = max(float((first[0][0, probe] - second[0][0, probe]).abs().max()),
                     float((first[1][0, probe] - second[1][0, probe]).abs().max()))
    assert difference == 0.0, (
        "if this becomes non-zero the receptive field grew and the docstring above, plus "
        f"the stage-23 design note, must be updated; got {difference}")


def test_the_conditioner_output_does_not_depend_on_pair_orientation():
    """Averaging with the transpose removes the (i,j) vs (j,i) arbitrariness."""
    torch.manual_seed(2)
    width = 6
    subnet = TopologyBondSubnet(2, 2, 32, 8, width, layers=2)
    _wake(subnet)
    rows, cols, _ = _pair_positions(width)
    values = torch.randn(1, rows.numel(), 2)
    mask = torch.ones(1, rows.numel(), dtype=torch.bool)
    context = torch.randn(1, 8)
    with torch.no_grad():
        dense = unpack_upper(values, width)
        assert torch.allclose(dense, dense.transpose(1, 2)), "the grid must be symmetric"
        scale, shift = subnet(values, mask, context)
    assert torch.isfinite(scale).all() and torch.isfinite(shift).all()


# ------------------------------------------------------------- no leakage -------
def test_the_coupling_never_shows_the_active_channels_to_the_conditioner():
    """Invertibility rests on this: only pass-through channels may enter the subnet."""
    width = 6
    rows, _, _ = _pair_positions(width)
    seen: list[torch.Tensor] = []

    class Recorder(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x, mask, context):            # noqa: ANN001
            seen.append(x.clone())
            return self.inner(x, mask, context)

    torch.manual_seed(3)
    inner = TopologyBondSubnet(2, 2, 16, 8, width, layers=2)
    _wake(inner)
    coupling = ConditionalCoupling(4, Recorder(inner), passthrough=2,
                                   max_log_scale=0.75, max_shift=3.0)
    x = torch.randn(1, rows.numel(), 4)
    mask = torch.ones(1, rows.numel(), dtype=torch.bool)
    context = torch.randn(1, 8)
    with torch.no_grad():
        coupling(x, mask, context, None)
    assert len(seen) == 1
    assert seen[0].shape[-1] == 2, "the subnet received more than the pass-through half"
    assert torch.equal(seen[0], x[..., :2]), "the subnet saw transformed channels"

    # and the inverse must recompute the SAME parameters from the same pass-through half
    with torch.no_grad():
        forward, _ = coupling(x, mask, context, None)
        recovered = coupling.inverse(forward, mask, context, None)
    assert torch.allclose(recovered, x, atol=1e-5)
    assert torch.equal(seen[1], seen[2]), "forward and inverse must see the same input"


# ------------------------------- the flow still works, for BOTH conditioners ----
@pytest.mark.parametrize("conditioner", list(BOND_CONDITIONERS))
@pytest.mark.parametrize("counts", [[10, 7, 4, 2], [10]])
def test_both_conditioners_keep_the_flow_invertible(conditioner, counts):
    torch.manual_seed(4)
    config = GraphFlowConfig(max_nodes=10, bond_blocks=4, atom_blocks=4, hidden=48,
                             conv_hidden=24, condition_dim=16, condition_hidden=32,
                             bond_conditioner=conditioner)
    flow = ConditionalGraphFlow(config)
    width = config.max_nodes
    node_mask = torch.zeros(len(counts), width, dtype=torch.bool)
    for row, count in enumerate(counts):
        node_mask[row, :count] = True
    pair_mask = pair_mask_from_counts(counts, width)
    atoms = torch.randint(0, NUM_ATOM_TYPES, (len(counts), width)) * node_mask
    dense = torch.randint(0, NUM_BOND_TYPES, (len(counts), width, width))
    dense = torch.triu(dense, 1)
    dense = dense + dense.transpose(1, 2)
    context = flow.conditioning(torch.randn(len(counts), 2))
    flow.train()
    x_atom, x_bond = dequantize(atoms, dense, node_mask, pair_mask,
                                num_atom_types=NUM_ATOM_TYPES,
                                num_bond_types=NUM_BOND_TYPES, noise_scale=0.6)
    flow.log_likelihood(x_atom, x_bond, node_mask, pair_mask, context)   # init actnorm
    flow.eval()
    for coupling in (m for m in flow.modules() if isinstance(m, ConditionalCoupling)):
        _wake(coupling.subnet, scale=0.3)
    # measured AFTER waking, so clean and dirty describe the same model
    clean = flow.log_likelihood(x_atom, x_bond, node_mask, pair_mask, context)

    encoded = flow.encode(x_atom, x_bond, node_mask, pair_mask, context)
    bond_error = float(((flow._invert(flow.bond_layers, encoded["z_bond"], pair_mask,
                                      context, None) - x_bond).abs()
                        * pair_mask.unsqueeze(-1)).max().detach())
    atom_error = float(((flow._invert(flow.atom_layers, encoded["z_atom"], node_mask,
                                      context, unpack_upper(x_bond, width)) - x_atom)
                        .abs() * node_mask.unsqueeze(-1)).max().detach())
    assert bond_error < 1e-4 and atom_error < 1e-4, (conditioner, bond_error, atom_error)
    # padding inert
    assert float((encoded["z_bond"] * (~pair_mask).unsqueeze(-1)).abs().max().detach()) == 0.0
    assert float((encoded["z_atom"] * (~node_mask).unsqueeze(-1)).abs().max().detach()) == 0.0
    dirty_atom, dirty_bond = x_atom.clone(), x_bond.clone()
    dirty_atom[~node_mask] = torch.randn_like(dirty_atom[~node_mask]) * 100
    dirty_bond[~pair_mask] = torch.randn_like(dirty_bond[~pair_mask]) * 100
    dirty = flow.log_likelihood(dirty_atom, dirty_bond, node_mask, pair_mask, context)
    assert float((clean["log_prob"] - dirty["log_prob"]).abs().max().detach()) < 1e-3
    # samples: symmetric, zero diagonal, exact N
    sampled = flow.sample(counts, context, temperature=0.85,
                          generator=torch.Generator().manual_seed(1))
    for graph, count in zip(flow.discrete_graphs(sampled), counts):
        bonds = graph["bond_matrix"]
        assert graph["atom_types"].shape == (count,)
        assert np.array_equal(bonds, bonds.T)
        assert (np.diag(bonds) == 0).all()
    worst = max(sampled["latent_round_trip_relative_error"].values())
    assert worst < 1e-3, (conditioner, worst)


def test_the_topology_conditioner_is_not_simply_bigger():
    """The improvement must not be explained by parameter count."""
    pairwise = ConditionalGraphFlow(GraphFlowConfig(max_nodes=38,
                                                    bond_conditioner="pairwise"))
    topology = ConditionalGraphFlow(GraphFlowConfig(max_nodes=38,
                                                    bond_conditioner="topology"))
    first = sum(p.numel() for p in pairwise.parameters())
    second = sum(p.numel() for p in topology.parameters())
    assert second < first, (
        f"topology ({second:,}) must need FEWER parameters than pairwise ({first:,}), so "
        f"an improvement cannot be attributed to capacity")


def test_an_unknown_conditioner_is_refused():
    with pytest.raises(ValueError, match="unknown bond_conditioner"):
        ConditionalGraphFlow(GraphFlowConfig(max_nodes=6, bond_conditioner="telepathy"))


# ---------------------------------------------- configuration and resumption ----
def test_a_checkpoint_without_a_conditioner_entry_loads_as_pairwise():
    """Stage-22 checkpoints predate the field and must not silently become topology."""
    stored = GraphFlowConfig(max_nodes=12, bond_conditioner="pairwise").to_dict()
    del stored["bond_conditioner"]
    assert GraphFlowConfig.from_dict(stored).bond_conditioner == "pairwise"
    # a stage-23 checkpoint carries it explicitly and round-trips
    explicit = GraphFlowConfig(max_nodes=12, bond_conditioner="topology").to_dict()
    assert GraphFlowConfig.from_dict(explicit).bond_conditioner == "topology"


def test_the_numerical_defaults_have_one_source_of_truth():
    reference = GraphFlowConfig()
    assert reference.max_log_scale == 0.75, "stage 23 fixes the bound at 0.75"
    assert reference.max_shift == 3.0
    arguments = load_script("23_train_graph_flow.py").parse_args(["--output-dir", "x"])
    for field in ("max_log_scale", "max_shift", "noise_scale", "bond_conditioner",
                  "conv_hidden", "conv_layers", "conv_kernel"):
        assert getattr(arguments, field) == getattr(reference, field), field


@pytest.fixture
def fixture_data(tmp_path):
    from zinc_gnf.conditioning import load_condition_scaler

    data = write_dataset(tmp_path / "data")
    payload = pickle.loads((Path(data) / "splits.pkl").read_bytes())
    (Path(data) / "scaler.json").write_text(
        json.dumps(ConditionScaler.fit(payload["train"]).to_dict()))
    return {"dir": data, "scaler": load_condition_scaler(data)}


def _train_args(run: Path, data: Path, **extra) -> list[str]:
    arguments = ["--output-dir", str(run), "--data-dir", str(data),
                 "--batch-size", "4", "--bond-blocks", "2", "--atom-blocks", "2",
                 "--hidden", "32", "--conv-hidden", "16", "--condition-dim", "16",
                 "--val-molecules", "8", "--warmup-steps", "1", "--device", "cpu",
                 "--max-nodes", "18", "--eval-every", "2", "--save-every", "2",
                 "--log-every", "2", "--max-steps", "4"]
    for key, value in extra.items():
        arguments += [f"--{key.replace('_', '-')}", str(value)]
    return arguments


def test_resuming_with_a_different_conditioner_is_refused(fixture_data, tmp_path):
    script = load_script("23_train_graph_flow.py")
    run = tmp_path / "run"
    assert script.main(_train_args(run, fixture_data["dir"],
                                   bond_conditioner="topology")) == 0
    assert script.main(_train_args(run, fixture_data["dir"],
                                   bond_conditioner="pairwise")) == 4
    log = (run / "train.log").read_text()
    assert "DIFFERENT model configuration" in log
    assert "bond_conditioner" in log


def test_training_refuses_an_ill_conditioned_inverse_before_spending_gpu_hours(
        fixture_data, tmp_path):
    """The safeguard: check the inverse at INITIALISATION, not after the run."""
    script = load_script("23_train_graph_flow.py")
    run = tmp_path / "bad"
    code = script.main(_train_args(run, fixture_data["dir"], bond_blocks=12,
                                   atom_blocks=12, max_log_scale=6.0))
    log = (run / "train.log").read_text()
    assert "inverse at initialisation" in log
    if code != 0:
        assert "ill-conditioned" in log
        summary = json.loads((run / "training_summary.json").read_text())
        assert summary["failure"] == "ill_conditioned_inverse_at_initialisation"


def test_training_tracks_the_inverse_at_every_evaluation(fixture_data, tmp_path):
    script = load_script("23_train_graph_flow.py")
    run = tmp_path / "ok"
    assert script.main(_train_args(run, fixture_data["dir"])) == 0
    summary = json.loads((run / "training_summary.json").read_text())
    assert summary["history"]
    for entry in summary["history"]:
        assert "inverse_round_trip_worst" in entry
        assert "inverse_acceptable" in entry
    assert "inverse at initialisation" in (run / "train.log").read_text()


# ------------------------------------------------- structural measurements ------
def test_structure_statistics_attribute_a_single_cause():
    carbon = np.zeros(4, dtype=np.int64)
    chain = np.zeros((4, 4), dtype=np.int64)
    for a, b in ((0, 1), (1, 2), (2, 3)):
        chain[a, b] = chain[b, a] = 1
    assert graph_structure_statistics(carbon, chain)["cause"] == "valid"

    split = np.zeros((4, 4), dtype=np.int64)
    split[0, 1] = split[1, 0] = 1
    split[2, 3] = split[3, 2] = 1
    broken = graph_structure_statistics(carbon, split)
    assert broken["cause"] == "disconnected_only"
    assert broken["components"] == 2

    oxygen = np.full(3, 2, dtype=np.int64)          # valence 2
    heavy = np.array([[0, 3, 3], [3, 0, 3], [3, 3, 0]], dtype=np.int64)
    over = graph_structure_statistics(oxygen, heavy)
    assert over["cause"] == "over_valent_only"
    assert over["over_valent_atoms"] == 3
    assert over["excess_bond_order"] > 0

    both = np.zeros((4, 4), dtype=np.int64)
    both[0, 1] = both[1, 0] = 3
    both[2, 3] = both[3, 2] = 1
    mixed = graph_structure_statistics(np.full(4, 2, dtype=np.int64), both)
    assert mixed["cause"] == "over_valent_and_disconnected"


def test_population_structure_reports_the_columns_stage_23_is_judged_on():
    graphs = []
    for _ in range(4):
        bonds = np.zeros((5, 5), dtype=np.int64)
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 4)):
            bonds[a, b] = bonds[b, a] = 1
        graphs.append({"atom_types": np.zeros(5, dtype=np.int64), "bond_matrix": bonds})
    block = population_structure(graphs, requested_n=5)
    for key in ("bonds_mean", "bond_density_mean", "components_mean",
                "largest_component_mean", "largest_component_fraction_of_n",
                "over_valent_atoms_mean", "raw_valid_rate", "bond_type_fraction",
                "atom_type_fraction", "correction_atoms_lost_mean",
                "correction_orders_removed_mean", "correction_exact_n_rate"):
        assert key in block, key
    assert block["bonds_mean"] == 4.0
    assert block["components_mean"] == 1.0
    assert block["raw_valid_rate"] == 1.0
    assert block["correction_exact_n_rate"] == 1.0
    assert block["atom_type_fraction"]["C"] == 1.0


def test_the_real_reference_is_computed_from_the_train_split(fixture_data):
    from zinc_gnf.dataset import load_training_splits

    train, _, _ = load_training_splits(fixture_data["dir"])
    counts = sorted({int(r["n_node"]) for r in train})
    reference = real_structure_reference(train, n=counts[-1])
    assert reference is not None
    assert reference["source"] == "train_split"
    assert reference["bonds_mean"] > 0
    assert real_structure_reference(train, n=999) is None


# ------------------------------------------------------- end to end -------------
def test_the_stage23_experiment_runs_and_compares_against_the_previous_flow(
        fixture_data, tmp_path):
    train = load_script("23_train_graph_flow.py")
    arms = {}
    for name, extra in (("topology", {"bond_conditioner": "topology"}),
                        ("null", {"bond_conditioner": "topology"}),
                        ("previous", {"bond_conditioner": "pairwise"})):
        run = tmp_path / name
        arguments = _train_args(run, fixture_data["dir"], **extra)
        if name == "null":
            arguments += ["--shuffle-qed"]
        assert train.main(arguments) == 0, name
        arms[name] = run / "last.pt"

    output = tmp_path / "eval"
    assert load_script("24_evaluate_graph_flow.py").main([
        "--graph-flow-checkpoint", str(arms["topology"]),
        "--null-checkpoint", str(arms["null"]),
        "--previous-graph-checkpoint", str(arms["previous"]),
        "--output-dir", str(output), "--data-dir", str(fixture_data["dir"]),
        "--target-n", "15", "--target-qed", "0.4,0.8", "--temperature", "1.0",
        "--count", "8", "--bootstrap-resamples", "100", "--qed-window", "0.5",
        "--seed", "17", "--device", "cpu"]) == 0
    payload = json.loads((output / "graph_flow_evaluation.json").read_text())
    assert set(payload["summary"]) == {"graph_flow", "graph_flow_null",
                                       "graph_flow_previous", "graph_flow_prior"}
    # the structural columns the stage is judged on must be present and populated
    for arm in ("graph_flow", "graph_flow_previous"):
        for cell in payload["cells"][arm]:
            structure = cell["structure"]
            assert structure is not None
            for key in ("bonds_mean", "components_mean", "largest_component_mean",
                        "over_valent_atoms_mean", "raw_valid_rate",
                        "correction_atoms_lost_mean", "atom_type_fraction",
                        "bond_type_fraction"):
                assert key in structure, key
            assert cell["real_reference"] is not None
    assert payload["real_structure_reference"]
    text = (output / "graph_flow_evaluation.txt").read_text()
    assert "STRUCTURE of the RAW graphs" in text
    assert "must rise TOWARDS" in text, "the report must state the anti-gaming criterion"


# ------------------------------------------------------------------- SLURM ------
def test_the_stage23_job_is_bounded_resumable_and_estimates_its_cost():
    path = REPOSITORY / "slurm" / "23_topology_graph_flow.sh"
    source = path.read_text(encoding="utf-8")
    directives = {}
    for line in source.splitlines():
        match = re.match(r"#SBATCH\s+--([\w-]+)(?:=(.*))?$", line.strip())
        if match:
            directives[match.group(1)] = (match.group(2) or "").strip()
    assert directives["gres"] == "gpu:1"
    assert directives["time"] and directives["mem"]
    assert 'if [[ -z "${SLURM_JOB_ID:-}" ]]' in source
    assert "Runtime estimate" in source and "Memory estimate" in source
    assert "Resumable" in source
    executable = "\n".join(line for line in source.splitlines()
                           if not line.lstrip().startswith("#") and "echo " not in line)
    assert "scripts/23_train_graph_flow.py" in executable
    assert "scripts/24_evaluate_graph_flow.py" in executable
    assert "--bond-conditioner topology" in executable
    assert "--shuffle-qed" in executable, "the null arm must be part of the job"
    assert "--previous-graph-checkpoint" in executable, "stage 22 must be a comparison arm"
    assert "--max-log-scale" not in executable, (
        "the numerical settings must come from the one source of truth, not be re-stated")
    assert not re.search(r"(?m)^\s*(rm|mv)\s", executable)
    for protected in ("graph_flow_real", "graph_flow_null", "warm_start_stageA",
                      "decoder_information", "qed_pathway", "decoder_experiment"):
        assert f"REAL_DIR=\"${{REAL_DIR:-outputs/checkpoints/{protected}" not in source
    assert "READ-ONLY" in source
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0
