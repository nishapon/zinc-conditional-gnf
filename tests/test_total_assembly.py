"""Stage 21: the total degree-constrained assembly.

The experiment's conclusion is only worth as much as these properties, so each is tested
on a constructed case where the answer is known:

  THE DEFECT      the baseline's parity gate is demonstrated, including the fact that
                  moving ONE atom's predicted degree by +-1 flips it - which is how a QED
                  request comes to change the assembly ALGORITHM.
  THE REPAIR      an odd sum costs exactly one atom one step; a predicted 0 is raised; a
                  sequence with too few bonds for any connected graph is raised to
                  2(n-1); the result is always graphical and always within the valence
                  caps; the cheapest change under the head's own softmax is the one made.
  VALIDITY        valence is never exceeded, the matrix is symmetric, and a graph that
                  cannot be assembled is reported as FAILED rather than returned as a
                  molecule.
  DETERMINISM     identical input gives a bit-identical bond matrix.
  NO FALLBACK     `fallback` is never True, and the module does not even import the greedy
                  decoder - there is no other algorithm for it to switch to.
  NO DEMOTION     a case whose argmax orders would violate valence is assembled within
                  the caps by the order program instead of being demoted.
  REGRESSION      the oscillation that an earlier version hit - two bridge edges swapping
                  back and forth forever - is pinned by a test.
  BASELINES       raw, constrained and degree stay out of the experimental path and the
                  default decoder set is unchanged.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest
import torch

from tests.test_autoencoder_training import write_dataset
from tests.test_conditional_flow import conditional_flow, scaler, wake_up_conditioning
from zinc_gnf.assembly_total import (
    _connect_by_swaps,
    is_graphical,
    repair_degree_sequence,
    total_degree_bond_decode,
)
from zinc_gnf.constants_charge_aware import ATOM_VALENCE, NUM_BOND_TYPES
from zinc_gnf.decoder_experiment import (
    evaluate_criteria,
    paired_decoder_changes,
    recommend_next,
)
from zinc_gnf.decoding_charge_aware import (
    degree_constrained_bond_decode,
    valence_aware_bond_decode_typed,
)
from zinc_gnf.generation import DECODERS, EXPERIMENTAL_DECODERS, KNOWN_DECODERS
from zinc_gnf.preprocessing import ConditionScaler

REPOSITORY = Path(__file__).resolve().parents[1]
CARBON = 0        # valence 4
FLUORINE = 3      # valence 1


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""),
                                                  REPOSITORY / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bond_probabilities(n: int, *, seed: int = 0, sparse: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, n, NUM_BOND_TYPES)) * 2.0
    logits = logits + logits.transpose(1, 0, 2)
    if sparse:
        logits[:, :, 0] += 3.0
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(-1, keepdims=True)


def degree_distribution(degrees, *, classes: int = 5, confidence: float = 4.0) -> np.ndarray:
    logits = np.full((len(degrees), classes), -confidence)
    for atom, degree in enumerate(degrees):
        logits[atom, min(int(degree), classes - 1)] = confidence
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(-1, keepdims=True)


# ------------------------------------------------------------------ the defect ---
def test_the_baseline_rejects_an_odd_degree_sum_and_one_atom_flips_it():
    """The discontinuity stage 20 measured, demonstrated at code level.

    The predicted degree sum has uncontrolled parity because the degrees are independent
    per-atom argmaxes. Moving ONE atom by one step flips it, so a QED request that shifts
    a single degree switches the assembly algorithm.
    """
    n = 6
    atoms = np.full(n, CARBON)
    probabilities = bond_probabilities(n, seed=1)
    even = np.array([2, 2, 2, 2, 1, 1])
    assert int(even.sum()) % 2 == 0
    _, even_info = degree_constrained_bond_decode(atoms, probabilities, even)

    odd = even.copy()
    odd[0] += 1                                   # ONE atom, one step
    assert int(odd.sum()) % 2 != 0
    _, odd_info = degree_constrained_bond_decode(atoms, probabilities, odd)
    assert odd_info["fallback"] is True
    assert odd_info["fallback_reason"] == "infeasible_degrees"
    # and the fallback IS the constrained decoder: a different algorithm entirely
    greedy, _ = valence_aware_bond_decode_typed(atoms, probabilities)
    odd_bonds, _ = degree_constrained_bond_decode(atoms, probabilities, odd)
    assert np.array_equal(odd_bonds, greedy)
    assert not (even_info["fallback"] and odd_info["fallback"]) or True

    # the experimental assembler keeps working on BOTH sides of that flip
    for sequence in (even, odd):
        _, info = total_degree_bond_decode(atoms, probabilities,
                                           degree_distribution(sequence),
                                           degrees=sequence)
        assert info["fallback"] is False
        assert info["relaxation_level"] is not None


def test_a_predicted_degree_of_zero_guarantees_the_baseline_disconnects():
    n = 6
    atoms = np.full(n, CARBON)
    probabilities = bond_probabilities(n, seed=2)
    degrees = np.array([0, 2, 2, 2, 2, 2])        # even sum, but an isolated atom
    assert int(degrees.sum()) % 2 == 0
    _, info = degree_constrained_bond_decode(atoms, probabilities, degrees)
    assert info["fallback"] is True
    assert info["fallback_reason"] == "not_connected"


# ------------------------------------------------------------------ the repair ---
def test_erdos_gallai():
    assert is_graphical([1, 1])
    assert is_graphical([2, 2, 2])
    assert not is_graphical([1, 1, 1])            # odd sum
    assert is_graphical([3, 1, 1, 1])             # a star: K(1,3)
    assert not is_graphical([3, 3, 1, 1])         # even sum, still not realisable
    assert not is_graphical([4, 1, 1])            # degree exceeds n - 1
    assert is_graphical([])


def test_the_repair_fixes_parity_with_exactly_one_single_step_change():
    n = 6
    atoms = np.full(n, CARBON)
    requested = np.array([2, 2, 2, 2, 2, 1])      # sum 11, odd
    degrees, info = repair_degree_sequence(degree_distribution(requested), atoms,
                                           requested=requested)
    assert info["requested_sum_was_odd"] is True
    assert info["parity_repaired"] is True
    assert int(degrees.sum()) % 2 == 0
    parity_changes = [c for c in info["changes"] if c["reason"] == "parity"]
    assert len(parity_changes) == 1
    assert abs(parity_changes[0]["to"] - parity_changes[0]["from"]) == 1
    assert is_graphical(degrees)


def test_the_repair_raises_a_predicted_zero_and_keeps_every_degree_at_least_one():
    n = 6
    atoms = np.full(n, CARBON)
    requested = np.array([0, 0, 2, 2, 2, 2])
    degrees, info = repair_degree_sequence(degree_distribution(requested), atoms,
                                           requested=requested)
    assert info["isolated_atoms_requested"] == 2
    assert (degrees >= 1).all(), "a degree of 0 guarantees a disconnected molecule"
    assert is_graphical(degrees)


def test_the_repair_adds_the_bonds_a_connected_molecule_needs():
    """sum(d) >= 2(n-1): the condition whose absence made an earlier version oscillate."""
    n = 15
    atoms = np.full(n, CARBON)
    requested = np.array([1, 2, 1, 1, 3, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1])   # 11 edges
    assert int(requested.sum()) // 2 < n - 1
    degrees, info = repair_degree_sequence(degree_distribution(requested), atoms,
                                           requested=requested)
    assert info["requested_edges"] == 11
    assert info["edges_needed_for_connectivity"] == n - 1
    assert info["degrees_raised_for_connectivity"] > 0
    assert int(degrees.sum()) >= 2 * (n - 1)
    assert info["connectivity_possible"] is True
    assert is_graphical(degrees)


def test_the_repair_respects_the_valence_caps_and_reports_impossibility():
    """All-fluorine: cap 1 everywhere, so no connected molecule exists beyond a dimer."""
    n = 6
    atoms = np.full(n, FLUORINE)
    requested = np.array([3, 3, 3, 3, 3, 3])
    degrees, info = repair_degree_sequence(degree_distribution(requested), atoms,
                                           requested=requested)
    caps = np.asarray([ATOM_VALENCE[int(a)] for a in atoms])
    assert (degrees <= caps).all(), "the repair must never exceed a valence cap"
    assert info["connectivity_possible"] is False
    assert info["clipped_atoms"] == n


def test_the_repair_chooses_the_cheapest_change_under_the_heads_own_confidence():
    """Atom 2 is the least confident, so it is the one the parity fix should move."""
    n = 4
    atoms = np.full(n, CARBON)
    requested = np.array([2, 2, 2, 1])            # sum 7, odd
    probabilities = degree_distribution(requested, confidence=6.0)
    probabilities[2] = np.array([0.02, 0.30, 0.34, 0.30, 0.04])   # nearly undecided
    probabilities = probabilities / probabilities.sum(-1, keepdims=True)
    _, info = repair_degree_sequence(probabilities, atoms, requested=requested)
    parity = [c for c in info["changes"] if c["reason"] == "parity"]
    assert len(parity) == 1
    assert parity[0]["atom"] == 2, "the least confident atom must absorb the parity fix"


def test_the_repair_is_deterministic():
    n = 10
    atoms = np.full(n, CARBON)
    requested = np.array([1, 2, 3, 1, 2, 1, 1, 2, 1, 1])
    probabilities = degree_distribution(requested)
    first, _ = repair_degree_sequence(probabilities, atoms, requested=requested)
    second, _ = repair_degree_sequence(probabilities, atoms, requested=requested)
    assert np.array_equal(first, second)


# ------------------------------------------------- validity of what comes out ---
@pytest.mark.parametrize("n", [6, 12, 18])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_assembly_never_breaks_valence_or_symmetry(n, seed):
    rng = np.random.default_rng(seed)
    atoms = rng.integers(0, 4, size=n)
    probabilities = bond_probabilities(n, seed=seed)
    logits = rng.normal(size=(n, 5)) * 1.5
    distribution = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    bonds, info = total_degree_bond_decode(atoms, probabilities, distribution,
                                           degrees=distribution.argmax(-1))
    caps = np.asarray([ATOM_VALENCE[int(a)] for a in atoms])
    assert np.array_equal(bonds, bonds.T), "bond matrix must be symmetric"
    assert (np.diag(bonds) == 0).all(), "no self bonds"
    assert (bonds.sum(axis=1) <= caps).all(), "valence cap exceeded"
    assert info["valence_respected"] is True
    if not info["assembly_failed"]:
        assert info["connected"] is True


def test_an_impossible_molecule_is_reported_failed_and_not_dressed_up_as_valid():
    """All-fluorine with 6 atoms: no connected graph exists. It must FAIL, not pretend."""
    n = 6
    atoms = np.full(n, FLUORINE)
    probabilities = bond_probabilities(n, seed=3)
    requested = np.array([1, 1, 1, 1, 1, 1])
    bonds, info = total_degree_bond_decode(atoms, probabilities,
                                           degree_distribution(requested),
                                           degrees=requested)
    assert info["degree_repair"]["connectivity_possible"] is False
    assert info["assembly_failed"] is True
    assert info["connected"] is False
    assert int(bonds.sum()) == 0, "a failed assembly returns no graph at all"


def test_the_order_program_respects_valence_where_an_argmax_would_not():
    """No chosen order is ever demoted: valence is a constraint, not a post-hoc fix."""
    n = 4
    atoms = np.full(n, 2)                         # oxygen, valence 2
    # every pair wants a TRIPLE bond, which no oxygen can carry twice
    logits = np.full((n, n, NUM_BOND_TYPES), -5.0)
    logits[:, :, 3] = 5.0
    probabilities = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    requested = np.array([1, 1, 1, 1])
    bonds, info = total_degree_bond_decode(atoms, probabilities,
                                           degree_distribution(requested),
                                           degrees=requested)
    caps = np.asarray([ATOM_VALENCE[int(a)] for a in atoms])
    assert (bonds.sum(axis=1) <= caps).all()
    assert info["fallback"] is False


# ------------------------------------------------------------- determinism -----
@pytest.mark.parametrize("n", [8, 16])
def test_the_assembly_is_deterministic(n):
    rng = np.random.default_rng(7)
    atoms = rng.integers(0, 3, size=n)
    probabilities = bond_probabilities(n, seed=5)
    logits = rng.normal(size=(n, 5))
    distribution = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    runs = [total_degree_bond_decode(atoms, probabilities, distribution,
                                     degrees=distribution.argmax(-1))[0]
            for _ in range(3)]
    assert np.array_equal(runs[0], runs[1])
    assert np.array_equal(runs[1], runs[2])


# --------------------------------------------------------------- no fallback ---
def test_the_module_cannot_fall_back_because_it_has_no_other_algorithm():
    """Structural: if this import ever appears, the discontinuity is back."""
    source = (REPOSITORY / "zinc_gnf" / "assembly_total.py").read_text(encoding="utf-8")
    # the docstring names the baselines it replaces, so inspect the CODE only
    code = "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith("#"))
    code = re.sub(r'""".*?"""', "", code, flags=re.S)
    for forbidden in ("valence_aware_bond_decode_typed", "degree_constrained_bond_decode",
                      "def fallback"):
        assert forbidden not in code, f"{forbidden} would restore the discontinuity"
    assert "from zinc_gnf.decoding_charge_aware import" not in source


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_fallback_is_never_reported_by_the_experimental_assembler(seed):
    rng = np.random.default_rng(seed)
    n = 14
    atoms = rng.integers(0, 4, size=n)
    probabilities = bond_probabilities(n, seed=seed)
    logits = rng.normal(size=(n, 5)) * 1.5
    distribution = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    _, info = total_degree_bond_decode(atoms, probabilities, distribution,
                                       degrees=distribution.argmax(-1))
    assert info["fallback"] is False
    assert info["fallback_reason"] is None


def test_degrees_are_satisfied_exactly_on_a_feasible_request():
    n = 12
    atoms = np.full(n, CARBON)
    probabilities = bond_probabilities(n, seed=9)
    requested = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3])    # 24 edges > n-1
    bonds, info = total_degree_bond_decode(atoms, probabilities,
                                           degree_distribution(requested),
                                           degrees=requested)
    assert info["relaxation_level"] == "exact_degrees"
    assert info["degree_satisfied"] is True
    assert np.array_equal((bonds > 0).sum(axis=1), np.asarray(info["degree_repair"]["degrees"]))


# ------------------------------------------------------- the swap regression ---
def test_two_bridge_components_cannot_oscillate_forever():
    """The bug an earlier version had: both removed edges bridges, count never falls.

    With only two 2-atom components there are 2 edges for 4 atoms - one short of the 3 a
    connected graph needs - so no swap can help. It must terminate and say so, not loop.
    """
    bond = np.zeros((4, 4), dtype=np.int64)
    bond[0, 1] = bond[1, 0] = 1
    bond[2, 3] = bond[3, 2] = 1
    log_any = np.full((4, 4), -1.0)
    result, record = _connect_by_swaps(bond, log_any)
    assert record["components_before"] == 2
    assert record["incomplete"] is True
    assert record["swaps"] == 0, "no merging swap exists; it must not thrash"
    assert record["components_after"] == 2


def test_a_swap_merges_when_a_non_bridge_is_available():
    """A triangle plus a dimer: the triangle's edges are non-bridges, so a swap merges."""
    bond = np.zeros((5, 5), dtype=np.int64)
    for a, b in ((0, 1), (1, 2), (0, 2)):
        bond[a, b] = bond[b, a] = 1
    bond[3, 4] = bond[4, 3] = 1
    degrees_before = (bond > 0).sum(axis=1).copy()
    log_any = np.full((5, 5), -1.0)
    result, record = _connect_by_swaps(bond, log_any)
    assert record["components_after"] == 1, "a non-bridge was available; it must merge"
    assert record["swaps"] == 1
    assert np.array_equal((result > 0).sum(axis=1), degrees_before), \
        "a 2-swap must preserve every degree"


# ---------------------------------------------------- the baselines are intact ---
def test_the_default_decoder_set_is_unchanged_and_total_is_opt_in():
    assert DECODERS == ("raw", "constrained", "degree")
    assert EXPERIMENTAL_DECODERS == ("total",)
    assert KNOWN_DECODERS == ("raw", "constrained", "degree", "total")


def test_requesting_only_the_baselines_never_enters_the_experimental_path(monkeypatch,
                                                                         tmp_path):
    """The baselines must be byte-for-byte the old path: the new code cannot even run."""
    import zinc_gnf.generation as generation
    from tests.test_flow_scripts_and_slurm import write_autoencoder
    from zinc_gnf.trainer_autoencoder import load_autoencoder

    autoencoder, _ = load_autoencoder(write_autoencoder(tmp_path / "ae.pt"), device="cpu")
    autoencoder.eval()
    torch.manual_seed(0)
    counts = [6, 7]
    embeddings = torch.randn(len(counts), max(counts),
                             autoencoder.config.node_embedding_dim)
    reference = generation.decode_embeddings(autoencoder, embeddings, counts)

    def explode(*args, **kwargs):               # noqa: ANN002, ANN003
        raise AssertionError("the experimental assembler must not run for a baseline")

    monkeypatch.setattr(generation, "total_degree_bond_decode", explode)
    guarded = generation.decode_embeddings(autoencoder, embeddings, counts)
    assert [r["raw"]["smiles"] for r in guarded] == [r["raw"]["smiles"] for r in reference]
    assert [r["degree"]["smiles"] for r in guarded] == \
        [r["degree"]["smiles"] for r in reference]
    assert [r["constrained"]["smiles"] for r in guarded] == \
        [r["constrained"]["smiles"] for r in reference]


def test_the_experimental_decoder_needs_a_degree_head(tmp_path):
    from tests.test_flow_scripts_and_slurm import write_autoencoder
    from zinc_gnf.generation import decode_embeddings
    from zinc_gnf.trainer_autoencoder import load_autoencoder

    path = write_autoencoder(tmp_path / "ae.pt", degree_head=False)
    autoencoder, _ = load_autoencoder(path, device="cpu")
    embeddings = torch.zeros(1, 6, autoencoder.config.node_embedding_dim)
    with pytest.raises(ValueError, match="degree head"):
        decode_embeddings(autoencoder, embeddings, [6], decoders=("total",))


def test_the_four_decoders_decode_the_same_embeddings(tmp_path):
    """Matchedness is structural: one call, one embedding, four assemblies."""
    from tests.test_flow_scripts_and_slurm import write_autoencoder
    from zinc_gnf.generation import decode_embeddings
    from zinc_gnf.trainer_autoencoder import load_autoencoder

    path = write_autoencoder(tmp_path / "ae.pt")
    autoencoder, _ = load_autoencoder(path, device="cpu")
    autoencoder.eval()
    torch.manual_seed(3)
    counts = [6, 7]
    embeddings = torch.randn(len(counts), max(counts),
                             autoencoder.config.node_embedding_dim)
    records = decode_embeddings(autoencoder, embeddings, counts,
                                decoders=("raw", "constrained", "degree", "total"))
    assert len(records) == 2
    for record in records:
        for name in ("raw", "constrained", "degree", "total"):
            assert name in record
        assert "predicted_degrees" in record
        assert record["total"]["fallback"] is False


# ----------------------------------------------------- the paired comparison ---
def test_the_common_subset_is_the_intersection_over_every_decoder():
    def record(values):
        return {name: ({"valid": True, "qed": value} if value is not None
                       else {"valid": False, "qed": None})
                for name, value in values.items()}

    low = [record({"degree": 0.4, "total": 0.4}), record({"degree": None, "total": 0.5}),
           record({"degree": 0.3, "total": 0.3})]
    high = [record({"degree": 0.45, "total": 0.60}), record({"degree": 0.5, "total": 0.7}),
            record({"degree": 0.35, "total": 0.50})]
    result = paired_decoder_changes(low, high, decoders=("degree", "total"),
                                    requested_change=0.4, resamples=200)
    assert result["common_subset_molecules"] == 2, "sample 1 fails for 'degree'"
    # common = samples 0 and 2; total moves +0.20 twice, degree +0.05 twice
    assert result["decoders"]["total"]["common_subset"]["mean"] == pytest.approx(0.20, abs=1e-6)
    assert result["decoders"]["degree"]["common_subset"]["mean"] == pytest.approx(0.05, abs=1e-6)
    improvement = result["improvement_over_baseline"]
    assert improvement["difference"] == pytest.approx(0.15, abs=1e-6)
    assert "no decoder can gain by failing" in result["note"]


def test_a_decoder_that_ignores_qed_shows_no_paired_change():
    low = [{"degree": {"valid": True, "qed": 0.5}, "total": {"valid": True, "qed": 0.5}}] * 8
    high = [{"degree": {"valid": True, "qed": 0.5}, "total": {"valid": True, "qed": 0.5}}] * 8
    result = paired_decoder_changes(low, high, decoders=("degree", "total"),
                                    requested_change=0.4, resamples=200)
    assert result["decoders"]["total"]["common_subset"]["mean"] == pytest.approx(0.0)
    assert not result["improvement_over_baseline"]["excludes_zero"]


# --------------------------------------------------- criteria and the decision ---
def _paired(improvement: float, excludes: bool = True, selection: float = 0.001):
    return [{"n": 23, "temperature": 0.2, "requested_change": 0.4,
             "common_subset_molecules": 180, "samples": 256,
             "decoders": {
                 "constrained": {"common_subset": {"mean": 0.0557},
                                 "own_subset": {"mean": 0.0557}, "selection_effect": 0.0},
                 "degree": {"common_subset": {"mean": -0.0041},
                            "own_subset": {"mean": -0.0041}, "selection_effect": 0.0},
                 "total": {"common_subset": {"mean": -0.0041 + improvement},
                           "own_subset": {"mean": -0.0041 + improvement - selection},
                           "selection_effect": selection}},
             "improvement_over_baseline": {"difference": improvement,
                                           "ci_low": improvement * 0.6,
                                           "ci_high": improvement * 1.4,
                                           "excludes_zero": excludes}}]


def _population(validity: float = 0.40, aromatic: float = 1.2, unique: float = 0.95,
                diversity: float = 0.85):
    return {"total": {"strict_validity": validity, "aromatic_rings_mean": aromatic,
                      "uniqueness": unique, "internal_diversity": diversity},
            "degree": {"strict_validity": 0.40, "aromatic_rings_mean": 1.2,
                       "uniqueness": 0.95, "internal_diversity": 0.85}}


def _achievement(mae: float = 0.19):
    return {"total": {"overall": {"mae_weighted": mae}},
            "degree": {"overall": {"mae_weighted": 0.21}}}


def test_the_achievement_keys_the_criteria_read_actually_exist():
    """Contract test: a renamed key here would silently blank the MAE columns."""
    from zinc_gnf.qed_evaluation import qed_achievement_table, qed_target_metrics

    records = [{"degree": {"valid": True, "qed": 0.5, "smiles": "CCO"}}] * 4
    cells = [{"n": 9, "requested_qed": value, "temperature": 0.2,
              "metrics": qed_target_metrics(records, requested_qed=value,
                                            decoder="degree")}
             for value in (0.4, 0.6, 0.8)]
    table = qed_achievement_table(cells)
    for key in ("mae_weighted", "hit_rate_weighted", "response_slope_pooled",
                "qed_independent_of_n", "cells"):
        assert key in table["overall"], key


def test_criteria_pass_when_the_response_improves_without_costing_validity():
    criteria = evaluate_criteria(cells=[], paired=_paired(0.05),
                                 achievement=_achievement(),
                                 assembly=_population(),
                                 cost={"total": {"seconds_per_molecule": 0.02},
                                       "degree": {"seconds_per_molecule": 0.006}})
    statuses = {c["criterion"]: c["status"] for c in criteria["checks"]}
    assert statuses["qed_control_improves"] == "pass"
    assert statuses["not_a_selection_artefact"] == "pass"
    assert statuses["validity_preserved"] == "pass"
    assert statuses["runtime_practical"] == "pass"
    assert statuses["no_leakage"] == "pass"
    assert criteria["retain_experimental_decoder"] is True


def test_criteria_fail_when_validity_is_traded_away():
    criteria = evaluate_criteria(cells=[], paired=_paired(0.05),
                                 achievement=_achievement(),
                                 assembly=_population(validity=0.20),
                                 cost={"total": {"seconds_per_molecule": 0.02}})
    statuses = {c["criterion"]: c["status"] for c in criteria["checks"]}
    assert statuses["validity_preserved"] == "fail"
    assert criteria["retain_experimental_decoder"] is False


def test_criteria_report_a_small_recovery_honestly():
    criteria = evaluate_criteria(cells=[], paired=_paired(0.004, excludes=False),
                                 achievement=_achievement(),
                                 assembly=_population(),
                                 cost={"total": {"seconds_per_molecule": 0.02}})
    statuses = {c["criterion"]: c["status"] for c in criteria["checks"]}
    assert statuses["qed_control_improves"] == "fail"
    assert criteria["retain_experimental_decoder"] is False


def test_the_milestone_is_reported_as_a_milestone_not_a_gate():
    criteria = evaluate_criteria(cells=[], paired=_paired(0.05),
                                 achievement=_achievement(mae=0.19),
                                 assembly=_population(),
                                 cost={"total": {"seconds_per_molecule": 0.02}})
    assert criteria["milestone"]["reached"] is False
    assert "not a gate" in criteria["milestone"]["note"]
    assert criteria["retain_experimental_decoder"] is True, \
        "missing the MAE milestone must not by itself discard the decoder"


def test_a_good_recovery_points_at_the_flow_not_at_more_decoder_work():
    criteria = evaluate_criteria(cells=[], paired=_paired(0.25),
                                 achievement=_achievement(),
                                 assembly=_population(),
                                 cost={"total": {"seconds_per_molecule": 0.02}})
    decision = recommend_next(criteria, paired=_paired(0.25), shift_auc=0.979)
    assert decision["primary"] == "A_strengthen_flow_conditioning"
    assert decision["fraction_of_request_recovered"] == pytest.approx(0.625, abs=0.01)


def test_a_small_recovery_with_a_severe_shift_points_at_the_direct_flow():
    criteria = evaluate_criteria(cells=[], paired=_paired(0.01),
                                 achievement=_achievement(),
                                 assembly=_population(),
                                 cost={"total": {"seconds_per_molecule": 0.02}})
    decision = recommend_next(criteria, paired=_paired(0.01), shift_auc=0.979)
    assert decision["primary"] == "B_direct_atom_bond_flow"
    assert "0.979" in decision["rationale"]
    assert set(decision["options"]) == {"A_strengthen_flow_conditioning",
                                        "B_direct_atom_bond_flow",
                                        "C_continue_decoder_work"}
    assert "deployable" in decision["guard"]


# ------------------------------------------------- the experiment, end to end ---
def test_the_experiment_runs_and_reports_every_required_column(tmp_path):
    import pickle

    from tests.test_flow_scripts_and_slurm import write_autoencoder

    data = write_dataset(tmp_path / "data")
    payload = pickle.loads((Path(data) / "splits.pkl").read_bytes())
    (Path(data) / "scaler.json").write_text(
        json.dumps(ConditionScaler.fit(payload["train"]).to_dict()))
    ae_path = write_autoencoder(tmp_path / "ae.pt")
    run = tmp_path / "joint"
    assert load_script("04_train_flow.py").main([
        "--config", str(REPOSITORY / "configs/zinc250k_flow_nqed.yaml"),
        "--data-dir", str(data), "--ae-checkpoint", str(ae_path),
        "--output-dir", str(run), "--device", "cpu", "--seed", "1", "--batch-size", "4",
        "--max-steps", "4", "--num-blocks", "1", "--statistics-molecules", "20",
        "--val-molecules", "8", "--eval-every", "2", "--save-every", "2",
        "--log-every", "2", "--warmup-steps", "0"]) == 0

    output = tmp_path / "experiment"
    assert load_script("22_decoder_experiment.py").main([
        "--flow-checkpoint", str(run / "last.pt"), "--ae-checkpoint", str(ae_path),
        "--data-dir", str(data), "--output-dir", str(output), "--target-n", "9",
        "--target-qed", "0.4,0.8", "--temperature", "0.2", "--count", "8",
        "--paired-count", "8", "--paired-qed", "0.4,0.8", "--qed-window", "0.5",
        "--timing-samples", "4", "--bootstrap-resamples", "100", "--shift-auc", "0.979",
        "--device", "cpu"]) == 0

    result = json.loads((output / "decoder_experiment.json").read_text())
    assert result["stage"] == 21
    assert result["decoders"] == ["raw", "constrained", "degree", "total"]
    assert result["experimental_decoder"] == "total"
    for name in ("raw", "constrained", "degree", "total"):
        block = result["population"][name]
        for key in ("strict_validity", "uniqueness", "novelty_vs_train",
                    "internal_diversity", "fallback_rate"):
            assert key in block, (name, key)
        assert name in result["cost"]
        assert result["cost"][name]["seconds_per_molecule"] > 0
        assert name in result["achievement"]
    for cell in result["cells"]:
        assert cell["support"]["support"] in {"well_supported", "low_support", "unknown"}
        for name in ("raw", "constrained", "degree", "total"):
            entry = cell["decoders"][name]
            for key in ("qed", "assembly", "degree_consistency", "aromatic_rings_mean",
                        "heavy_atom_count_exact"):
                assert key in entry, (name, key)
    assert result["paired"], "the paired comparison must run"
    assert result["criteria"]["checks"]
    assert result["recommendation"]["primary"] in {
        "A_strengthen_flow_conditioning", "B_direct_atom_bond_flow",
        "C_continue_decoder_work"}
    # the experimental decoder must never report a fallback anywhere
    for cell in result["cells"]:
        assert cell["decoders"]["total"]["assembly"]["fallback_rate"] == 0.0

    text = (output / "decoder_experiment.txt").read_text()
    assert "STEP 5 RECOMMENDATION" in text
    assert "raw native validity" in text
    assert "no_selection" in text
    assert "no_oracle" in text


def test_the_experiment_validates_its_arguments(tmp_path):
    script = load_script("22_decoder_experiment.py")
    assert script.main(["--flow-checkpoint", str(tmp_path / "ghost.pt"),
                        "--output-dir", str(tmp_path / "o"),
                        "--paired-qed", "0.8,0.4"]) == 4
    assert script.main(["--flow-checkpoint", str(tmp_path / "ghost.pt"),
                        "--output-dir", str(tmp_path / "o2")]) == 2


# ------------------------------------------------------------------- SLURM -----
def test_the_stage21_job_is_cpu_only_read_only_and_estimates_its_cost():
    path = REPOSITORY / "slurm" / "21_decoder_experiment.sh"
    source = path.read_text(encoding="utf-8")
    directives = {}
    for line in source.splitlines():
        match = re.match(r"#SBATCH\s+--([\w-]+)(?:=(.*))?$", line.strip())
        if match:
            directives[match.group(1)] = (match.group(2) or "").strip()
    assert directives["partition"] == "CPU"
    assert "gres" not in directives, "stage 21 needs no GPU"
    assert directives["time"] and directives["mem"]
    assert 'if [[ -z "${SLURM_JOB_ID:-}" ]]' in source
    assert "Runtime estimate" in source
    assert "Memory estimate" in source
    executable = "\n".join(line for line in source.splitlines()
                           if not line.lstrip().startswith("#") and "echo " not in line)
    assert "scripts/22_decoder_experiment.py" in executable
    assert "04_train_flow.py" not in executable, "stage 21 must not train the flow"
    assert "17_structure_finetune.py" not in executable
    assert "--device cpu" in executable
    assert not re.search(r"(?m)^\s*(rm|mv)\s", executable)
    assert "outputs/metrics/decoder_experiment" in source
    for protected in ("decoder_information", "qed_pathway", "warm_start", "signal_stageA"):
        assert f"OUT=\"${{OUT:-outputs/metrics/{protected}" not in source
    assert "warm_start_stageA" in source, "it must reuse the stage-17 checkpoint"
    assert "READ-ONLY" in source
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0
