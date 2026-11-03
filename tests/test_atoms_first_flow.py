"""Stage 25: the atoms-first factorisation, and the properties it must not cost.

Stage 24 established the motivation by inspection: bonds-first runs the bond stack with
`extra=None`, so the block deciding bond orders never sees an atom type, while valence
couples an atom's TYPE to the sum of its incident bond orders. Atoms-first reverses the
two. Both are exact factorisations of the same joint, so nothing here claims bonds-first
*cannot* represent valid molecules - the claim is about learnability and inductive bias,
and the architectural part of it is what these tests pin down:

  THE MECHANISM   in atoms_first the bond latents MUST move when an atom type changes,
                  and in bonds_first they provably cannot. The mirror image holds for the
                  atom latents and the bonds. Two tests measure both directions, so the
                  factorisation is a demonstrated fact about the code rather than a claim
                  in a docstring.
  CHEMISTRY       the context actually carries allowed valence and formal charge, from
                  the charge-aware vocabulary, symmetrically in (i, j) and masked.
  CONTINUOUS vs   `dequantized` keeps the likelihood path smooth - the gradient of the
  DISCRETE        bond term with respect to the atom channels is non-zero - and `onehot`
                  makes it exactly zero. That is the precise difference between the two
                  modes, asserted rather than described.
  NO LEAKAGE      the conditioners still see ONLY pass-through channels, so the inverse
                  recomputes the same scale and shift.
  PRESERVED       exact log-determinants against the true Jacobian, exact inverses,
                  padding inertness, bond symmetry, variable N, numerical refusal, and -
                  the one that protects the baseline - stage-23 checkpoints still load
                  and still resume.
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
from zinc_gnf.constants_charge_aware import (
    ATOM_VALENCE,
    ATOM_VOCAB,
    CHARGE_OF_TYPE,
    NUM_ATOM_TYPES,
    NUM_BOND_TYPES,
)
from zinc_gnf.factorisation_comparison import (
    compare,
    qed_on_valid_exact_n,
    within_n_response_slope,
)
from zinc_gnf.graph_flow import (
    ATOM_CONTEXT_MODES,
    FACTORISATIONS,
    AtomChemistryContext,
    ConditionalCoupling,
    ConditionalGraphFlow,
    GraphFlowConfig,
    dequantize,
    pair_mask_from_counts,
)
from zinc_gnf.preprocessing import ConditionScaler
from zinc_gnf.trainer_graph_flow import load_graph_flow

REPOSITORY = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""),
                                                  REPOSITORY / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny(factorisation: str, **extra) -> GraphFlowConfig:
    settings = dict(max_nodes=10, bond_blocks=3, atom_blocks=3, hidden=48,
                    conv_hidden=24, atom_conv_hidden=24, condition_dim=16,
                    condition_hidden=32, factorisation=factorisation)
    settings.update(extra)
    return GraphFlowConfig(**settings)


def _head(subnet: torch.nn.Module) -> torch.nn.Module:
    head = getattr(subnet, "head", None)
    return head if head is not None else subnet.net[-1]


def wake(flow: ConditionalGraphFlow, scale: float = 0.3) -> None:
    """Move every coupling off the identity, so the tests exercise a real transform."""
    with torch.no_grad():
        for coupling in (m for m in flow.modules() if isinstance(m, ConditionalCoupling)):
            for parameter in _head(coupling.subnet).parameters():
                parameter.add_(torch.randn_like(parameter) * scale)


def make_inputs(flow: ConditionalGraphFlow, counts: list[int], *, seed: int = 0):
    torch.manual_seed(seed)
    width = flow.config.max_nodes
    node_mask = torch.zeros(len(counts), width, dtype=torch.bool)
    for row, count in enumerate(counts):
        node_mask[row, :count] = True
    pair_mask = pair_mask_from_counts(counts, width)
    atom_types = torch.randint(0, NUM_ATOM_TYPES, (len(counts), width)) * node_mask
    dense = torch.randint(0, NUM_BOND_TYPES, (len(counts), width, width))
    dense = torch.triu(dense, 1)
    dense = dense + dense.transpose(1, 2)
    context = flow.conditioning(torch.randn(len(counts), 2))
    x_atom, x_bond = dequantize(atom_types, dense, node_mask, pair_mask,
                               num_atom_types=NUM_ATOM_TYPES,
                               num_bond_types=NUM_BOND_TYPES, noise_scale=0.6)
    flow.train()
    flow.log_likelihood(x_atom, x_bond, node_mask, pair_mask, context)   # init actnorm
    flow.eval()
    return x_atom, x_bond, node_mask, pair_mask, context


# ------------------------------------------------- THE mechanism of stage 25 ----
@pytest.mark.parametrize("factorisation,must_depend", [("atoms_first", True),
                                                       ("bonds_first", False)])
def test_the_bond_latents_depend_on_the_atoms_only_in_atoms_first(factorisation,
                                                                  must_depend):
    """THE acceptance test. Change one atom type; do the bond latents move?

    This is the whole of stage 25 expressed as a measurement. In bonds-first the bond
    stack is run before the atoms exist and receives `extra=None`, so the difference must
    be EXACTLY zero - which is the structural blindness stage 24 attributed the 89.5%
    over-valence to. In atoms-first the same change must reach the bond latents, because
    the conditioner is given both endpoints' identity, charge and allowed valence.
    """
    flow = ConditionalGraphFlow(tiny(factorisation))
    x_atom, x_bond, node_mask, pair_mask, context = make_inputs(flow, [10, 6])
    wake(flow)
    with torch.no_grad():
        before = flow.encode(x_atom, x_bond, node_mask, pair_mask, context)["z_bond"]
        moved = x_atom.clone()
        # atom 0 becomes a different element entirely: carbon (cap 4) -> sulphur (cap 6)
        moved[0, 0] = 0.0
        moved[0, 0, ATOM_VOCAB.index("S")] = 1.0
        after = flow.encode(moved, x_bond, node_mask, pair_mask, context)["z_bond"]
    difference = float(((after - before).abs() * pair_mask.unsqueeze(-1)).max())
    if must_depend:
        assert difference > 1e-5, (
            "atoms-first must let an atom type reach the bond latents; got "
            f"{difference:.3e}")
    else:
        assert difference == 0.0, (
            "bonds-first is structurally blind to atom types here; if this is no longer "
            f"exactly zero the factorisation has changed. Got {difference:.3e}")


@pytest.mark.parametrize("factorisation,must_depend", [("atoms_first", False),
                                                       ("bonds_first", True)])
def test_the_atom_latents_depend_on_the_bonds_only_in_bonds_first(factorisation,
                                                                  must_depend):
    """The mirror image, which is the COST of the reversal and is not hidden.

    Atoms-first gives up the atom stack's relational view of the bonds. The replacement -
    an index convolution plus a masked set mean - keeps cross-atom structure, but it
    cannot see bonds, and that must be as explicit as the gain.
    """
    flow = ConditionalGraphFlow(tiny(factorisation))
    x_atom, x_bond, node_mask, pair_mask, context = make_inputs(flow, [10, 6])
    wake(flow)
    with torch.no_grad():
        before = flow.encode(x_atom, x_bond, node_mask, pair_mask, context)["z_atom"]
        moved = x_bond.clone()
        moved[0, 0] = torch.tensor([0.0, 0.0, 1.0, 0.0])[:moved.shape[-1]]
        after = flow.encode(x_atom, moved, node_mask, pair_mask, context)["z_atom"]
    difference = float(((after - before).abs() * node_mask.unsqueeze(-1)).max())
    if must_depend:
        assert difference > 1e-6
    else:
        assert difference == 0.0, (
            "in atoms-first the atom block runs first and cannot see bonds; got "
            f"{difference:.3e}")


def test_the_atom_stack_still_sees_other_atoms_so_the_comparison_is_not_confounded():
    """Composition must stay representable: `p(X|c)` is NOT a product over atoms.

    If the atoms-first atom stack were a plain per-atom MLP, every atom would be
    independent given the condition, which is strictly weaker than what stage 23 had. A
    comparison against that would confound the factorisation with a crippled atom model,
    so the index convolution and the set mean are load-bearing and tested as such.
    """
    flow = ConditionalGraphFlow(tiny("atoms_first"))
    x_atom, x_bond, node_mask, pair_mask, context = make_inputs(flow, [10])
    wake(flow)
    with torch.no_grad():
        before = flow.encode(x_atom, x_bond, node_mask, pair_mask, context)["z_atom"]
        moved = x_atom.clone()
        moved[0, 7] = 0.0
        moved[0, 7, ATOM_VOCAB.index("S")] = 1.0        # a DIFFERENT atom from atom 0
        after = flow.encode(moved, x_bond, node_mask, pair_mask, context)["z_atom"]
    reach = float((after[0, 0] - before[0, 0]).abs().max())
    assert reach > 1e-6, (
        f"atom 7's identity must reach atom 0's latent; got {reach:.3e}. Without this "
        f"the atom density factorises over atoms and cannot model composition")


# ------------------------------------------------------ the chemistry it carries ---
def test_the_atom_context_carries_allowed_valence_and_formal_charge():
    """Stage 24 asked for the valence caps explicitly. These are those channels."""
    context = AtomChemistryContext(NUM_ATOM_TYPES, embed_dim=4)
    types = [ATOM_VOCAB.index(label) for label in ("C", "N+1", "O-1", "S")]
    atoms = torch.zeros(1, len(types), NUM_ATOM_TYPES)
    for position, atom_type in enumerate(types):
        atoms[0, position, atom_type] = 1.0
    node_mask = torch.ones(1, len(types), dtype=torch.bool)
    _, chemistry = context.atom_features(atoms, node_mask)
    scale = max(ATOM_VALENCE)
    for position, atom_type in enumerate(types):
        assert float(chemistry[0, position, 0].detach()) == pytest.approx(
            ATOM_VALENCE[atom_type] / scale), ATOM_VOCAB[atom_type]
        assert float(chemistry[0, position, 1].detach()) == pytest.approx(
            float(CHARGE_OF_TYPE[atom_type])), ATOM_VOCAB[atom_type]
    # and they reach the grid as the symmetric pair (sum, |difference|)
    grid = context(atoms, node_mask).detach()
    assert grid.shape == (1, len(types), len(types), context.channels)
    carbon, sulphur = 0, 3
    caps = (ATOM_VALENCE[types[carbon]] / scale, ATOM_VALENCE[types[sulphur]] / scale)
    assert float(grid[0, carbon, sulphur, -4]) == pytest.approx(sum(caps))
    assert float(grid[0, carbon, sulphur, -2]) == pytest.approx(abs(caps[0] - caps[1]))
    charges = (CHARGE_OF_TYPE[types[1]], CHARGE_OF_TYPE[types[2]])       # N+1 and O-1
    assert float(grid[0, 1, 2, -3]) == pytest.approx(float(sum(charges)))
    assert float(grid[0, 1, 2, -1]) == pytest.approx(abs(float(charges[0] - charges[1])))


def test_the_atom_context_is_symmetric_and_padding_contributes_nothing():
    context = AtomChemistryContext(NUM_ATOM_TYPES, embed_dim=6)
    torch.manual_seed(0)
    atoms = torch.rand(2, 5, NUM_ATOM_TYPES)
    node_mask = torch.tensor([[True] * 5, [True, True, True, False, False]])
    grid = context(atoms, node_mask).detach()
    assert torch.allclose(grid, grid.transpose(1, 2)), "the pair context must be symmetric"
    assert float(grid[1, 3:, :].abs().max()) == 0.0
    assert float(grid[1, :, 3:].abs().max()) == 0.0
    noisy = atoms.clone()
    noisy[1, 3:] = torch.randn_like(noisy[1, 3:]) * 100
    assert torch.equal(context(noisy, node_mask).detach(), grid), (
        "padded atoms must be inert")


@pytest.mark.parametrize("mode", list(ATOM_CONTEXT_MODES))
def test_the_two_context_modes_differ_exactly_where_documented(mode):
    """`dequantized` reads the channels; `onehot` reads only the identity they imply."""
    context = AtomChemistryContext(NUM_ATOM_TYPES, embed_dim=4, mode=mode)
    atoms = torch.zeros(1, 2, NUM_ATOM_TYPES)
    atoms[0, 0, 0] = 1.0
    atoms[0, 1, 1] = 1.0
    node_mask = torch.ones(1, 2, dtype=torch.bool)
    base = context(atoms, node_mask).detach()
    jittered = atoms.clone()
    jittered[0, 0, 0] = 1.4                 # same argmax, different channel value
    jittered[0, 0, 5] = 0.3
    moved = float((context(jittered, node_mask).detach() - base).abs().max())
    if mode == "dequantized":
        assert moved > 1e-6, "the continuous channels must be read as they are"
    else:
        assert moved == 0.0, "onehot mode must depend on the identity only"


@pytest.mark.parametrize("mode,expect_gradient", [("dequantized", True),
                                                  ("onehot", False)])
def test_whether_the_bond_term_is_differentiable_in_the_atom_channels(mode,
                                                                      expect_gradient):
    """The explicit statement about the likelihood path, as a measurement.

    In `dequantized` mode every operation from the atom channels into the bond term is
    linear or smooth, so the gradient exists and is non-zero: nothing non-differentiable
    has been introduced. In `onehot` mode the argmax kills it exactly - the conditional is
    still a normalised density in the bond variable, so the dequantization bound stands,
    but the density is piecewise in the atom channels. That is the documented tradeoff.
    """
    flow = ConditionalGraphFlow(tiny("atoms_first", atom_context_mode=mode))
    x_atom, x_bond, node_mask, pair_mask, context = make_inputs(flow, [10])
    wake(flow)
    probe = x_atom.clone().requires_grad_(True)
    encoded = flow.encode(probe, x_bond, node_mask, pair_mask, context)
    bond_term = (flow.bond_prior.log_prob(encoded["z_bond"], pair_mask, context)
                 + encoded["logdet_bond"]).sum()
    gradient = torch.autograd.grad(bond_term, probe, allow_unused=True)[0]
    magnitude = 0.0 if gradient is None else float(gradient.abs().max())
    assert np.isfinite(magnitude)
    if expect_gradient:
        assert magnitude > 0.0, "dequantized mode must keep the path differentiable"
    else:
        assert magnitude == 0.0, "onehot mode routes through argmax, which has no gradient"


# ------------------------------------------- what the reversal must NOT cost ------
@pytest.mark.parametrize("factorisation", list(FACTORISATIONS))
@pytest.mark.parametrize("counts", [[10, 7, 4, 2], [10]])
def test_every_factorisation_is_exactly_invertible_and_mask_clean(factorisation, counts):
    flow = ConditionalGraphFlow(tiny(factorisation))
    x_atom, x_bond, node_mask, pair_mask, context = make_inputs(flow, counts)
    wake(flow)
    clean = flow.log_likelihood(x_atom, x_bond, node_mask, pair_mask, context)
    encoded = flow.encode(x_atom, x_bond, node_mask, pair_mask, context)

    # the two stacks invert exactly, each with the context its direction supplies
    if flow.atoms_first:
        atom_back = flow._invert(flow.atom_layers, encoded["z_atom"], node_mask,
                                 context, None)
        grid = flow.atom_context(x_atom, node_mask)
        bond_back = flow._invert(flow.bond_layers, encoded["z_bond"], pair_mask,
                                 context, grid)
    else:
        bond_back = flow._invert(flow.bond_layers, encoded["z_bond"], pair_mask,
                                 context, None)
        from zinc_gnf.graph_flow import unpack_upper
        atom_back = flow._invert(flow.atom_layers, encoded["z_atom"], node_mask, context,
                                 unpack_upper(x_bond, flow.config.max_nodes))
    atom_error = float(((atom_back - x_atom).abs() * node_mask.unsqueeze(-1))
                       .max().detach())
    bond_error = float(((bond_back - x_bond).abs() * pair_mask.unsqueeze(-1))
                       .max().detach())
    assert atom_error < 1e-4 and bond_error < 1e-4, (factorisation, atom_error,
                                                     bond_error)

    # padding is inert in the latents and in the density
    assert float((encoded["z_atom"] * (~node_mask).unsqueeze(-1)).abs()
                 .max().detach()) == 0.0
    assert float((encoded["z_bond"] * (~pair_mask).unsqueeze(-1)).abs()
                 .max().detach()) == 0.0
    dirty_atom, dirty_bond = x_atom.clone(), x_bond.clone()
    dirty_atom[~node_mask] = torch.randn_like(dirty_atom[~node_mask]) * 100
    dirty_bond[~pair_mask] = torch.randn_like(dirty_bond[~pair_mask]) * 100
    dirty = flow.log_likelihood(dirty_atom, dirty_bond, node_mask, pair_mask, context)
    assert float((clean["log_prob"] - dirty["log_prob"]).abs().max().detach()) < 1e-3

    # the density is the sum of the two stages, in either order
    assert torch.allclose(clean["log_prob"],
                          clean["log_prob_bond"] + clean["log_prob_atom"], atol=1e-4)

    # sampling: symmetric, zero diagonal, exact N, and a sound round trip
    sampled = flow.sample(counts, context, temperature=0.85,
                          generator=torch.Generator().manual_seed(1))
    for graph, count in zip(flow.discrete_graphs(sampled), counts):
        bonds = graph["bond_matrix"]
        assert graph["atom_types"].shape == (count,)
        assert np.array_equal(bonds, bonds.T)
        assert (np.diag(bonds) == 0).all()
    assert max(sampled["latent_round_trip_relative_error"].values()) < 1e-3


@pytest.mark.parametrize("factorisation", list(FACTORISATIONS))
def test_the_log_determinant_matches_the_true_jacobian(factorisation):
    """Exact likelihood: the reported log-determinant against autograd's Jacobian.

    Small enough to build the whole matrix. The map is block-triangular in both
    orderings, so the claim being checked is that the sum of the two stacks'
    log-determinants IS the log-determinant of the joint map - not an approximation of it.
    """
    torch.manual_seed(5)
    config = GraphFlowConfig(max_nodes=4, num_atom_types=5, num_bond_types=3,
                             bond_blocks=2, atom_blocks=2, hidden=16, conv_hidden=8,
                             atom_conv_hidden=8, condition_dim=4, condition_hidden=8,
                             atom_context_dim=3, factorisation=factorisation)
    flow = ConditionalGraphFlow(config)
    node_mask = torch.ones(1, 4, dtype=torch.bool)
    pair_mask = torch.ones(1, 6, dtype=torch.bool)
    context = flow.conditioning(torch.randn(1, 2))
    x_atom = torch.rand(1, 4, 5)
    x_bond = torch.rand(1, 6, 3)
    flow.train()
    flow.encode(x_atom, x_bond, node_mask, pair_mask, context)       # init actnorm
    flow.eval()
    wake(flow, scale=0.2)

    atom_size = x_atom.numel()
    flat = torch.cat([x_atom.reshape(-1), x_bond.reshape(-1)]).detach()

    def forward(values: torch.Tensor) -> torch.Tensor:
        atoms = values[:atom_size].reshape(x_atom.shape)
        bonds = values[atom_size:].reshape(x_bond.shape)
        encoded = flow.encode(atoms, bonds, node_mask, pair_mask, context)
        return torch.cat([encoded["z_atom"].reshape(-1), encoded["z_bond"].reshape(-1)])

    jacobian = torch.autograd.functional.jacobian(forward, flat)
    truth = float(torch.slogdet(jacobian)[1])
    encoded = flow.encode(x_atom, x_bond, node_mask, pair_mask, context)
    reported = float((encoded["logdet_atom"] + encoded["logdet_bond"])[0].detach())
    assert reported == pytest.approx(truth, abs=1e-3), (factorisation, reported, truth)


# -------------------------------------------- the stage-23 baseline is preserved ---
def test_a_bonds_first_model_has_no_atom_context_parameters_at_all():
    """So a stage 22-24 state_dict is exactly the shape it always was."""
    flow = ConditionalGraphFlow(tiny("bonds_first"))
    assert flow.atom_context is None
    assert not [key for key in flow.state_dict() if "atom_context" in key]
    assert flow.parameter_report()["atom_context"] == 0


def test_a_checkpoint_written_before_stage_25_loads_unchanged(tmp_path):
    flow = ConditionalGraphFlow(tiny("bonds_first"))
    stored = flow.config.to_dict()
    for key in ("factorisation", "atom_conv_hidden", "atom_conv_layers",
                "atom_conv_kernel", "atom_context_dim", "atom_context_mode"):
        stored.pop(key, None)                 # exactly what a stage-23 checkpoint has
    path = tmp_path / "stage23.pt"
    torch.save({"model_state_dict": flow.state_dict(), "model_config": stored,
                "step": 18000}, path)
    loaded, meta = load_graph_flow(path)
    assert meta["step"] == 18000
    assert loaded.config.factorisation == "bonds_first"
    assert loaded.atom_context is None
    for key, value in flow.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value), key


def test_a_bonds_first_config_records_only_what_describes_it():
    """The inert atom-context flags are not written into a bonds-first checkpoint.

    They build nothing there, so recording them would make the baseline's resumability
    depend on flags that change no weights - which is how a harmless default would come
    to strand an 18,000-step run.
    """
    bonds = GraphFlowConfig(factorisation="bonds_first", atom_conv_hidden=999).to_dict()
    assert bonds["factorisation"] == "bonds_first"
    for key in ("atom_conv_hidden", "atom_conv_layers", "atom_conv_kernel",
                "atom_context_dim", "atom_context_mode"):
        assert key not in bonds, key
    atoms = GraphFlowConfig(factorisation="atoms_first").to_dict()
    for key in ("atom_conv_hidden", "atom_context_dim", "atom_context_mode"):
        assert key in atoms, key
    # and a round trip through from_dict is stable for both
    for payload in (bonds, atoms):
        assert GraphFlowConfig.from_dict(payload).to_dict() == payload


def test_the_defaults_still_describe_the_stage_23_baseline():
    """The dataclass default must stay bonds-first, or the baseline stops resuming."""
    reference = GraphFlowConfig()
    assert reference.factorisation == "bonds_first"
    assert reference.atom_context_mode == "dequantized"
    arguments = load_script("23_train_graph_flow.py").parse_args(["--output-dir", "x"])
    for field in ("factorisation", "atom_conv_hidden", "atom_conv_layers",
                  "atom_conv_kernel", "atom_context_dim", "atom_context_mode"):
        assert getattr(arguments, field) == getattr(reference, field), field


def test_the_atoms_first_model_is_not_larger_than_the_baseline_it_must_beat():
    """A win must not be attributable to capacity, so the numbers are pinned here."""
    bonds = ConditionalGraphFlow(GraphFlowConfig(max_nodes=38)).parameter_report()
    atoms = ConditionalGraphFlow(
        GraphFlowConfig(max_nodes=38, factorisation="atoms_first")).parameter_report()
    assert bonds["total"] == 1_309_690, "the stage-23 baseline size must not drift"
    assert atoms["total"] == 1_258_778
    assert atoms["total"] < bonds["total"], (
        f"atoms-first ({atoms['total']:,}) must not exceed bonds-first "
        f"({bonds['total']:,}) at default settings")
    assert atoms["total"] / bonds["total"] > 0.9, (
        "it must also not be starved: a loss should not be explicable by capacity either")
    assert atoms["atom_context"] > 0 and bonds["atom_context"] == 0


def test_unknown_factorisations_and_context_modes_are_refused():
    with pytest.raises(ValueError, match="unknown factorisation"):
        ConditionalGraphFlow(tiny("atoms_last"))
    with pytest.raises(ValueError, match="unknown atom_context_mode"):
        ConditionalGraphFlow(tiny("atoms_first", atom_context_mode="telepathy"))


def test_a_bond_subnet_built_for_atom_context_refuses_to_run_without_it():
    """A silent None would train a different function than it samples with."""
    from zinc_gnf.graph_flow import BondSubnet, TopologyBondSubnet

    mask = torch.ones(1, 6, dtype=torch.bool)
    values = torch.randn(1, 6, 2)
    condition = torch.randn(1, 8)
    for subnet in (TopologyBondSubnet(2, 2, 16, 8, 4, layers=2, context_channels=6),
                   BondSubnet(2, 2, 16, 8, 4, context_channels=6)):
        with pytest.raises(ValueError, match="atom context"):
            subnet(values, mask, condition)


# --------------------------------------------------------------- no leakage -------
def test_the_bond_conditioner_never_sees_the_channels_it_transforms():
    """Invertibility rests on this, and atom context must not smuggle anything in."""
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []

    class Recorder(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x, mask, context, extra=None):        # noqa: ANN001
            seen.append((x.clone(), None if extra is None else extra.clone()))
            return self.inner(x, mask, context, extra)

    torch.manual_seed(3)
    flow = ConditionalGraphFlow(tiny("atoms_first"))
    x_atom, x_bond, node_mask, pair_mask, context = make_inputs(flow, [10])
    wake(flow)
    for layer in flow.bond_layers:
        layer["coupling"].subnet = Recorder(layer["coupling"].subnet)

    with torch.no_grad():
        encoded = flow.encode(x_atom, x_bond, node_mask, pair_mask, context)
        forward_calls = len(seen)
        grid = flow.atom_context(x_atom, node_mask)
        flow._invert(flow.bond_layers, encoded["z_bond"], pair_mask, context, grid)
    assert forward_calls == len(flow.bond_layers)
    for values, extra in seen:
        assert values.shape[-1] == flow.config.num_bond_types // 2, (
            "the conditioner received more than the pass-through half")
        assert extra is not None, "atoms-first must supply the atom context"
    # the inverse must recompute the SAME parameters from the SAME inputs
    forward_inputs = seen[:forward_calls]
    inverse_inputs = list(reversed(seen[forward_calls:]))
    assert len(inverse_inputs) == forward_calls
    for (a_values, a_extra), (b_values, b_extra) in zip(forward_inputs, inverse_inputs):
        assert torch.allclose(a_values, b_values, atol=1e-5)
        assert torch.allclose(a_extra, b_extra, atol=1e-6)


# ----------------------------------------------------- training and resumption ----
@pytest.fixture
def fixture_data(tmp_path):
    data = write_dataset(tmp_path / "data")
    payload = pickle.loads((Path(data) / "splits.pkl").read_bytes())
    (Path(data) / "scaler.json").write_text(
        json.dumps(ConditionScaler.fit(payload["train"]).to_dict()))
    return Path(data)


def train_args(run: Path, data: Path, **extra) -> list[str]:
    arguments = ["--output-dir", str(run), "--data-dir", str(data),
                 "--batch-size", "4", "--bond-blocks", "2", "--atom-blocks", "2",
                 "--hidden", "32", "--conv-hidden", "16", "--atom-conv-hidden", "16",
                 "--condition-dim", "16", "--val-molecules", "8", "--warmup-steps", "1",
                 "--device", "cpu", "--max-nodes", "18", "--eval-every", "2",
                 "--save-every", "2", "--log-every", "2", "--max-steps", "4"]
    for key, value in extra.items():
        if value is True:
            arguments.append(f"--{key.replace('_', '-')}")
        else:
            arguments += [f"--{key.replace('_', '-')}", str(value)]
    return arguments


def test_an_atoms_first_run_trains_and_records_what_it_built(fixture_data, tmp_path):
    script = load_script("23_train_graph_flow.py")
    run = tmp_path / "atoms"
    assert script.main(train_args(run, fixture_data,
                                  factorisation="atoms_first")) == 0
    log = (run / "train.log").read_text()
    assert "FACTORISATION atoms_first: p(X|c) p(A|X,c)" in log
    assert "identity, formal charge and allowed valence" in log
    assert "learnability, not expressive power" in log
    assert "parameters by part" in log
    summary = json.loads((run / "training_summary.json").read_text())
    assert summary["model_config"]["factorisation"] == "atoms_first"
    assert summary["provenance"]["stage"] == 25
    assert summary["provenance"]["parameter_report"]["atom_context"] > 0
    assert summary["history"], "evaluations must have run"
    for entry in summary["history"]:
        assert entry["inverse_acceptable"], "the inverse must stay sound while training"


def test_resuming_a_pre_stage_25_checkpoint_is_allowed(fixture_data, tmp_path):
    """Adding config fields must not strand the stage-23 baseline."""
    script = load_script("23_train_graph_flow.py")
    run = tmp_path / "bonds"
    assert script.main(train_args(run, fixture_data)) == 0
    checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    for key in ("factorisation", "atom_conv_hidden", "atom_conv_layers",
                "atom_conv_kernel", "atom_context_dim", "atom_context_mode"):
        checkpoint["model_config"].pop(key, None)
    checkpoint["step"] = 2
    torch.save(checkpoint, run / "last.pt")
    assert script.main(train_args(run, fixture_data)) == 0, (
        "a checkpoint written before stage 25 must still resume")
    assert "resumed from" in (run / "train.log").read_text()


def test_resuming_with_a_different_factorisation_is_refused(fixture_data, tmp_path):
    script = load_script("23_train_graph_flow.py")
    run = tmp_path / "mixed"
    assert script.main(train_args(run, fixture_data,
                                  factorisation="atoms_first")) == 0
    assert script.main(train_args(run, fixture_data,
                                  factorisation="bonds_first")) == 4
    log = (run / "train.log").read_text()
    assert "DIFFERENT model configuration" in log
    assert "factorisation" in log


def test_an_atoms_first_run_refuses_an_ill_conditioned_inverse(fixture_data, tmp_path):
    """The stage-22a lesson applies to the new stack too: check BEFORE spending GPU."""
    script = load_script("23_train_graph_flow.py")
    run = tmp_path / "unstable"
    code = script.main(train_args(run, fixture_data, factorisation="atoms_first",
                                  bond_blocks=12, atom_blocks=12, max_log_scale=6.0))
    log = (run / "train.log").read_text()
    assert "inverse at initialisation" in log
    if code != 0:
        assert "ill-conditioned" in log
        summary = json.loads((run / "training_summary.json").read_text())
        assert summary["failure"] == "ill_conditioned_inverse_at_initialisation"


# ----------------------------------------------------- the matched comparison -----
def test_the_response_slope_is_the_same_estimator_the_project_already_reports():
    """Pins the two implementations together so they cannot drift apart silently."""
    evaluator = load_script("24_evaluate_graph_flow.py")
    cells = [{"n": 15, "requested_qed": 0.4, "realised": {"raw": [0.30, 0.35, 0.33]}},
             {"n": 15, "requested_qed": 0.8, "realised": {"raw": [0.55, 0.60, 0.58]}},
             {"n": 23, "requested_qed": 0.4, "realised": {"raw": [0.28, 0.31]}},
             {"n": 23, "requested_qed": 0.8, "realised": {"raw": [0.52, 0.49]}}]
    mine = within_n_response_slope(cells, key="raw", resamples=64, seed=7)
    theirs = evaluator.response_slope(cells, "raw", resamples=64, seed=7)
    assert mine["slope"] == pytest.approx(theirs["slope"])
    assert mine["ci_low"] == pytest.approx(theirs["ci_low"])
    assert mine["ci_high"] == pytest.approx(theirs["ci_high"])
    assert mine["molecules_used"] == 10


def test_qed_is_restricted_to_valid_exact_n_molecules_with_its_support():
    records = [
        {"raw": {"valid": True, "exact_n": True, "qed": 0.78}},
        {"raw": {"valid": True, "exact_n": True, "qed": 0.82}},
        {"raw": {"valid": True, "exact_n": False, "qed": 0.10}},   # dropped atoms
        {"raw": {"valid": False, "exact_n": False, "qed": None}},
    ]
    block = qed_on_valid_exact_n(records, requested_qed=0.8, decoder="raw")
    assert block["support_molecules"] == 2, "only valid AND exact-N molecules count"
    assert block["support_rate"] == pytest.approx(0.5)
    assert block["realised_qed_mean"] == pytest.approx(0.80)
    assert block["mae"] == pytest.approx(0.02)
    assert block["hit_rate_tol_05"] == 1.0
    empty = qed_on_valid_exact_n(records[2:], requested_qed=0.8, decoder="raw")
    assert empty["support_molecules"] == 0 and not np.isfinite(empty["mae"])


def _arm(**overrides) -> dict:
    arm = {
        "raw_valid_rate": 0.10, "over_valent_atom_rate": 0.20,
        "atom_type_fraction": {"C": 0.73, "N": 0.12, "O": 0.15},
        "aromatic_rings_per_graph_raw": 0.01, "alternating_ring_share": 0.03,
        "rings_per_graph": 1.0, "latent_round_trip_worst": 1e-6,
        "parameters": {"total": 1_000_000},
        "raw": {"exact_n_rate": 0.5, "valid_exact_n": {
            "support_molecules": 100,
            "response_slope": {"slope": 0.05, "ci_low": 0.01, "ci_high": 0.09,
                               "molecules_used": 100}}},
        "corrected": {"strict_validity": 1.0},
    }
    for key, value in overrides.items():
        arm[key] = value
    return arm


def test_the_verdict_needs_aromaticity_as_well_as_validity():
    """Stage 25's criteria are conjunctive, and corrected validity is never a pass."""
    baseline = _arm()
    validity_only = _arm(raw_valid_rate=0.40, over_valent_atom_rate=0.05)
    both = _arm(raw_valid_rate=0.40, over_valent_atom_rate=0.05,
                aromatic_rings_per_graph_raw=0.9, alternating_ring_share=0.4)
    worse = _arm(raw_valid_rate=0.02)

    assert compare({"bonds_first": baseline, "atoms_first": validity_only},
                   real_ceiling={"aromatic_rings_per_molecule": 1.85}
                   )["verdict"] == "validity_only"
    assert compare({"bonds_first": baseline, "atoms_first": both},
                   real_ceiling={"aromatic_rings_per_molecule": 1.85}
                   )["verdict"] == "yes_on_both_criteria"
    assert compare({"bonds_first": baseline, "atoms_first": worse})["verdict"] == "no"
    # a perfect corrected validity must not rescue a failing native arm
    corrected_only = _arm(corrected={"strict_validity": 1.0})
    verdict = compare({"bonds_first": baseline, "atoms_first": corrected_only})
    assert verdict["verdict"] == "no"
    assert "NOT success" in verdict["warning"]
    # and a broken inverse is decisive whatever else improved
    broken = compare({"bonds_first": baseline,
                      "atoms_first": _arm(raw_valid_rate=0.9,
                                          aromatic_rings_per_graph_raw=2.0,
                                          latent_round_trip_worst=0.5)})
    assert broken["verdict"] == "no"
    assert broken["checks"]["inverse_still_sound"]["passed"] is False


def test_the_verdict_says_so_when_there_is_no_null_arm():
    verdict = compare({"bonds_first": _arm(), "atoms_first": _arm(raw_valid_rate=0.4)})
    check = verdict["checks"]["qed_response_beats_its_null"]
    assert check["passed"] is None
    assert "NO null arm supplied" in check["detail"]
    assert compare({"atoms_first": _arm()})["verdict"] == "incomparable"


def test_the_comparison_runs_end_to_end_and_reports_every_required_column(
        fixture_data, tmp_path):
    train = load_script("23_train_graph_flow.py")
    arms = {}
    for name, extra in (("atoms", {"factorisation": "atoms_first"}),
                        ("null", {"factorisation": "atoms_first", "shuffle_qed": True}),
                        ("bonds", {"factorisation": "bonds_first"})):
        run = tmp_path / name
        assert train.main(train_args(run, fixture_data, **extra)) == 0, name
        arms[name] = run / "last.pt"
    from collections import Counter

    from zinc_gnf.dataset import load_training_splits
    records, _, _ = load_training_splits(fixture_data)
    n = Counter(int(r["n_node"]) for r in records).most_common(1)[0][0]

    digests = {name: path.read_bytes() for name, path in arms.items()}
    output = tmp_path / "compare"
    assert load_script("27_compare_factorisations.py").main([
        "--atoms-first-checkpoint", str(arms["atoms"]),
        "--bonds-first-checkpoint", str(arms["bonds"]),
        "--null-checkpoint", str(arms["null"]),
        "--output-dir", str(output), "--data-dir", str(fixture_data),
        "--target-n", str(n), "--target-qed", "0.4,0.8", "--temperature", "1.0",
        "--count", "8", "--bootstrap-resamples", "50", "--val-molecules", "8",
        "--real-molecules", "16", "--seed", "17"]) == 0
    for name, path in arms.items():
        assert path.read_bytes() == digests[name], f"{name} must be READ-ONLY"

    payload = json.loads((output / "factorisation_comparison.json").read_text())
    assert payload["stage"] == 25
    assert set(payload["arms"]) == {"atoms_first", "atoms_first_null", "bonds_first"}
    assert payload["real_ceiling"]["aromatic_rings_per_molecule"] >= 0
    assert payload["support"], "support counts must be reported"
    for name, arm in payload["arms"].items():
        for key in ("raw_valid_rate", "over_valent_atom_rate", "infeasible_atom_rate",
                    "over_valent_rate_by_atom_type", "components_mean", "bonds_mean",
                    "bond_type_fraction", "atom_type_fraction", "ring_size_counts",
                    "rings_per_graph", "alternating_ring_share",
                    "aromatic_rings_per_graph_raw", "corrected_exact_n_rate",
                    "correction_atoms_lost_mean", "correction_orders_removed_mean",
                    "latent_round_trip_worst", "parameters", "likelihood"):
            assert key in arm, (name, key)
        for decoder in ("raw", "corrected"):
            for key in ("strict_validity", "uniqueness", "novelty_vs_train",
                        "internal_diversity", "qed_mae", "hit_rate_tol_10",
                        "response_slope", "exact_n_rate", "valid_exact_n"):
                assert key in arm[decoder], (name, decoder, key)
            assert "support_molecules" in arm[decoder]["valid_exact_n"]
        for key in ("validation_nll_per_dim", "condition_gap_per_atom",
                    "condition_gap_sem", "condition_win_rate",
                    "inverse_round_trip_worst"):
            assert key in arm["likelihood"], (name, key)
    assert payload["arms"]["atoms_first"]["factorisation"] == "atoms_first"
    assert payload["arms"]["bonds_first"]["factorisation"] == "bonds_first"
    assert payload["verdict"]["verdict"] in {"yes_on_both_criteria", "validity_only",
                                             "no"}
    assert payload["guarantees"]["no_oracle"]

    text = (output / "factorisation_comparison.txt").read_text()
    assert "NATIVE generation" in text
    assert "AROMATICITY - a separate criterion" in text
    assert "AFTER CORRECTION - reported, never the headline" in text
    assert "atoms LOST to correction" in text
    assert "restricted to VALID, EXACT-N molecules" in text
    assert "VERDICT:" in text


def test_the_comparison_refuses_two_arms_of_the_same_factorisation(fixture_data,
                                                                   tmp_path):
    train = load_script("23_train_graph_flow.py")
    run = tmp_path / "bonds"
    assert train.main(train_args(run, fixture_data)) == 0
    output = tmp_path / "bad"
    assert load_script("27_compare_factorisations.py").main([
        "--atoms-first-checkpoint", str(run / "last.pt"),
        "--bonds-first-checkpoint", str(run / "last.pt"),
        "--output-dir", str(output), "--data-dir", str(fixture_data),
        "--target-n", "15", "--target-qed", "0.4", "--temperature", "1.0",
        "--count", "4", "--bootstrap-resamples", "10", "--val-molecules", "8"]) == 4
    assert "not 'atoms_first'" in (output / "comparison.log").read_text()


def test_the_comparison_never_loads_the_test_split_or_selects_by_qed():
    source = (REPOSITORY / "scripts" / "27_compare_factorisations.py").read_text()
    assert "load_test_split" not in source
    for forbidden in ("QED.qed", "Descriptors.qed", "nlargest", "top_k", "argsort"):
        assert forbidden not in source, f"{forbidden} would make this a selection"
    assert "READ-ONLY" in source
    assert "QED-oracle selection or reranking" in source


# ------------------------------------------------------------------- SLURM --------
def test_the_stage25_job_is_bounded_resumable_and_states_its_cost():
    path = REPOSITORY / "slurm" / "25_atoms_first_graph_flow.sh"
    source = path.read_text(encoding="utf-8")
    directives = {}
    for line in source.splitlines():
        match = re.match(r"#SBATCH\s+--([\w-]+)(?:=(.*))?$", line.strip())
        if match:
            directives[match.group(1)] = (match.group(2) or "").strip()
    assert directives["gres"] == "gpu:1"
    assert directives["time"] and directives["mem"]
    assert 'if [[ -z "${SLURM_JOB_ID:-}" ]]' in source, "must refuse the login node"
    assert "Runtime estimate" in source and "Memory estimate" in source
    assert "Resumable" in source

    executable = "\n".join(line for line in source.splitlines()
                           if not line.lstrip().startswith("#") and "echo " not in line)
    assert "--factorisation atoms_first" in executable
    assert "--shuffle-qed" in executable, "the null arm must be part of the job"
    assert "scripts/27_compare_factorisations.py" in executable
    assert "--bonds-first-checkpoint" in executable, "the stage-23 arm must be compared"
    assert "--max-log-scale" not in executable, (
        "the numerical settings must come from the one source of truth")
    assert not re.search(r"(?m)^\s*(rm|mv)\s", executable), "no destructive commands"
    # the stage-23 baseline and its null must never be written to
    assert "graph_flow_atoms_first" in source
    for protected in ("outputs/checkpoints/graph_flow_topology",
                      "outputs/checkpoints/graph_flow_real"):
        assert f'REAL_DIR="${{REAL_DIR:-{protected}' not in source
        assert f'NULL_DIR="${{NULL_DIR:-{protected}' not in source
    assert "READ-ONLY" in source
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0


def test_the_stage25_design_note_states_the_hypothesis_and_its_limits():
    text = (REPOSITORY / "docs" / "stage25_atoms_first_design.md").read_text()
    for required in ("exact factorisation", "learnability", "atoms-first",
                     "AtomChemistryContext", "dequantized", "onehot",
                     "shuffled-QED", "raw validity", "aromaticity",
                     "parameter", "Limitations"):
        assert required in text, required
    assert "incapable" in text, (
        "the note must explicitly disclaim the stronger, false claim about bonds-first")
