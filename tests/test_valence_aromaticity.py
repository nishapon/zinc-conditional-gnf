"""Stage 24: the valence and aromaticity diagnostics, and the two mechanisms they test.

Stage 23 produced the right number of bonds and the wrong molecules: at N=23 the generated
bond count is 24.45 against 24.65 real with 1.20 components, yet 89.5% of graphs carry an
over-valent atom and aromatic rings sit at 0.008 per corrected molecule. Correct global
marginals, wrong local correlations.

Two mechanisms explain that, and both are asserted here on constructed graphs rather than
argued:

  H-VALENCE   the flow generates bonds FIRST and conditions atoms on them, so the bond
              block cannot respect valence - it does not know the atom types yet. The
              prediction is that the generated bonds are usually REALISABLE and the atom
              choice is what fails. `best_case_atom_assignment` is the instrument, and
              `test_best_case_relabelling_separates_realisable_bonds_from_impossible_ones`
              pins down both of its verdicts.
  H-AROMATIC  `valency_correction` removes one order from the LARGEST-order bond, which in
              a Kekule aromatic ring is a DOUBLE bond.
              `test_the_corrector_deletes_the_ring_double_bond_and_destroys_alternation`
              shows it choosing the ring bond over an equally large non-ring bond.

The ring machinery is also checked for the thing that would quietly invalidate the whole
audit: that `all_single` rings are not counted as failed benzenes, and that real molecules
do keep their aromaticity through our own serialise/Kekulise/reconstruct path - the ceiling
every generated number is read against.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import re
import subprocess

import numpy as np
import pytest

from tests.test_autoencoder_model import make_records
from tests.test_autoencoder_training import write_dataset
from zinc_gnf.constants_charge_aware import ATOM_VALENCE, ATOM_VOCAB
from zinc_gnf.graph_flow_generation import valency_correction
from zinc_gnf.preprocessing import ConditionScaler
from zinc_gnf.valence_diagnostics import (
    MAX_VALENCE,
    argmax_shift,
    aromatic_rings_of,
    best_case_atom_assignment,
    correction_damage,
    joint_atom_bond_distribution,
    real_reference_profile,
    ring_bond_pattern,
    ring_profile,
    valence_profile,
)

REPOSITORY = Path(__file__).resolve().parents[1]
CARBON = ATOM_VOCAB.index("C")
NITROGEN = ATOM_VOCAB.index("N")
BENZENE_ORDERS = [1, 2, 1, 2, 1, 2]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""),
                                                  REPOSITORY / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cycle(orders, *, substituent: tuple[int, int] | None = None) -> np.ndarray:
    """A ring carrying `orders` round it, optionally with one extra bonded atom."""
    size = len(orders)
    width = size + (1 if substituent else 0)
    bonds = np.zeros((width, width), dtype=np.int64)
    for position, order in enumerate(orders):
        a, b = position, (position + 1) % size
        bonds[a, b] = bonds[b, a] = int(order)
    if substituent:
        atom, order = substituent
        bonds[atom, size] = bonds[size, atom] = int(order)
    return bonds


def star(spokes: int, order: int = 1) -> np.ndarray:
    bonds = np.zeros((spokes + 1, spokes + 1), dtype=np.int64)
    for other in range(1, spokes + 1):
        bonds[0, other] = bonds[other, 0] = int(order)
    return bonds


def graph(bonds: np.ndarray, atom_type: int = CARBON) -> dict:
    return {"atom_types": np.full(bonds.shape[0], atom_type, dtype=np.int64),
            "bond_matrix": bonds}


# ------------------------------------------------------------------ valence ----
def test_valence_profile_names_the_element_the_charge_and_the_cap_it_broke():
    """Stage 24 asks for over-valence BY element, formal charge and allowed valence."""
    carbon_star = graph(star(5))                      # C with bond-order sum 5, cap 4
    nitrogen = {"atom_types": np.array([NITROGEN, CARBON, CARBON, CARBON, CARBON],
                                       dtype=np.int64),
                "bond_matrix": star(4)}               # N with sum 4, cap 3
    profile = valence_profile([carbon_star, nitrogen])

    assert profile["atoms"] == 11
    assert profile["over_valent_atoms"] == 2
    assert profile["excess_histogram"] == {1: 2}, "both are over by exactly one order"

    carbon = profile["by_atom_type"]["C"]
    assert carbon["element"] == "C" and carbon["formal_charge"] == 0
    assert carbon["allowed_valence"] == 4
    assert carbon["over_valent"] == 1 and carbon["atoms"] == 10
    assert carbon["mean_excess_when_over"] == 1.0
    assert profile["by_atom_type"]["N"]["allowed_valence"] == 3
    assert profile["by_element"]["N"]["over_valent_rate"] == 1.0
    assert profile["bond_order_sum_histogram"]["C"][5] == 1, "the sum histogram is kept"


def test_a_charged_atom_type_reports_its_charge_and_its_charge_adjusted_cap():
    """The dataset is charge-aware; 'N+1' must not be read as neutral nitrogen."""
    plus = ATOM_VOCAB.index("N+1")
    assert ATOM_VALENCE[plus] == 4, "N+ carries four bonds, unlike neutral N"
    entry = valence_profile([{"atom_types": np.full(5, plus, dtype=np.int64),
                              "bond_matrix": star(4)}])["by_atom_type"]["N+1"]
    assert entry["element"] == "N" and entry["formal_charge"] == 1
    assert entry["allowed_valence"] == 4
    assert entry["over_valent"] == 0, "sum 4 is legal for N+, and must not be flagged"
    minus = ATOM_VOCAB.index("O-1")
    negative = valence_profile([{"atom_types": np.full(2, minus, dtype=np.int64),
                                 "bond_matrix": star(1)}])["by_atom_type"]["O-1"]
    assert negative["formal_charge"] == -1 and negative["allowed_valence"] == 1


def test_an_atom_beyond_every_cap_in_the_vocabulary_is_counted_separately():
    """The number that decides whether the BONDS or the ATOM CHOICE is wrong.

    A bond-order sum above the largest cap in the vocabulary cannot be repaired by any
    relabelling, so it is evidence against H-VALENCE. A sum of 5 on a carbon can be.
    """
    assert MAX_VALENCE == 6, "sulphur is the ceiling; the diagnostic is written round it"
    repairable = valence_profile([graph(star(5))])
    assert repairable["over_valent_atoms"] == 1
    assert repairable["infeasible_for_every_type"] == 0

    impossible = valence_profile([graph(star(7))])
    assert impossible["infeasible_for_every_type"] == 1
    assert impossible["infeasible_rate"] == pytest.approx(1 / 8)


def test_best_case_relabelling_separates_realisable_bonds_from_impossible_ones():
    """The decisive counterfactual: keep the bonds, relabel the atoms as kindly as possible.

    A high feasible rate beside a low raw validity means the generated BOND pattern is
    chemically realisable and the atom choice is the failure - exactly what a bonds-first
    factorisation predicts. The relabelled distribution is checked too, because a 'fix'
    that turns the molecule into phosphorus is not a fix.
    """
    realisable = best_case_atom_assignment([graph(star(5))])
    assert realisable["valence_feasible_rate"] == 1.0
    assert realisable["atoms_relabelled"] == 1, "only the centre needed changing"
    distribution = realisable["best_case_atom_distribution"]
    assert distribution["P"] == pytest.approx(1 / 6), "cap 5 is the smallest that fits"
    assert distribution["C"] == pytest.approx(5 / 6), "legal atoms keep their type"

    impossible = best_case_atom_assignment([graph(star(7))])
    assert impossible["valence_feasible_rate"] == 0.0
    assert impossible["best_case_atom_distribution"]["<infeasible>"] > 0

    mixed = best_case_atom_assignment([graph(star(5)), graph(star(7))])
    assert mixed["valence_feasible_rate"] == 0.5
    assert [row["valence_feasible"] for row in mixed["per_graph"]] == [True, False]


def test_a_valid_molecule_is_left_completely_alone():
    """No false positives: the diagnostics must be silent on a legal graph."""
    profile = valence_profile([graph(cycle(BENZENE_ORDERS))])
    assert profile["over_valent_atoms"] == 0
    assert profile["infeasible_for_every_type"] == 0
    best = best_case_atom_assignment([graph(cycle(BENZENE_ORDERS))])
    assert best["atoms_relabelled"] == 0
    assert best["best_case_atom_distribution"] == {"C": 1.0}


def test_the_joint_atom_and_bond_order_distribution_is_what_the_factorisation_must_match():
    """Stage 23 got the MARGINALS right; this is the joint it is being judged on."""
    joint = joint_atom_bond_distribution([graph(cycle(BENZENE_ORDERS))])
    assert joint["pairs"] == {"C|3": 1.0}, "every benzene carbon carries 1 + 2"
    assert joint["atoms"] == 6
    two = joint_atom_bond_distribution([graph(cycle(BENZENE_ORDERS)), graph(star(5))])
    assert two["pairs"]["C|5"] == pytest.approx(1 / 12), "the over-valent centre appears"


def test_argmax_sharpening_is_measured_against_the_soft_scores():
    """If argmax concentrates far beyond the soft scores, DISCRETISATION is part of the
    marginal error rather than the density - a decoding fix, not an architecture one."""
    values = np.zeros((3, 4))
    values[:, 0] = 2.0                 # a mild preference for class 0 ...
    values[2, :] = -5.0                # ... and a padded row that must be ignored
    report = argmax_shift(values, np.array([True, True, False]), labels=list("abcd"))
    assert report["rows"] == 2, "the masked row must not be counted"
    assert report["argmax_distribution"] == {"a": 1.0}, "argmax is absolute"
    assert report["soft_distribution"]["a"] < 0.95, "the soft score is not"
    assert report["max_absolute_difference"] > 0.05
    assert argmax_shift(values, np.zeros(3, dtype=bool), labels=list("abcd"))["rows"] == 0


# -------------------------------------------------------------- aromaticity ----
def test_ring_patterns_distinguish_benzene_from_cyclohexane():
    """A Kekule 6-ring is aromatic only when the alternation is exactly right, so an
    `all_single` ring is a saturated ring and NOT a failed benzene. Conflating the two
    would make 'the model builds rings' look like 'the model builds aromatic rings'."""
    members = list(range(6))
    assert ring_bond_pattern(cycle(BENZENE_ORDERS), members)["pattern"] == "alternating"
    assert ring_bond_pattern(cycle([1] * 6), members)["pattern"] == "all_single"
    assert ring_bond_pattern(cycle([1, 1, 2, 1, 1, 1]),
                             members)["pattern"] == "mixed_with_double"
    # a 5-ring cannot alternate at all: odd length
    five = ring_bond_pattern(cycle([1, 2, 1, 2, 1]), list(range(5)))
    assert five["size"] == 5 and five["pattern"] == "mixed_with_double"
    assert ring_bond_pattern(cycle(BENZENE_ORDERS), members)["orders"] == BENZENE_ORDERS
    # and a set of atoms that is not a cycle is reported as such, not silently patterned
    chain = np.zeros((4, 4), dtype=np.int64)
    for a, b in ((0, 1), (1, 2), (2, 3)):
        chain[a, b] = chain[b, a] = 1
    assert ring_bond_pattern(chain, [0, 1, 2, 3])["pattern"] == "not_a_simple_cycle"


def test_rings_are_found_without_rdkit_because_most_graphs_cannot_be_sanitized():
    """At N=23 about 92% of generated graphs are unsanitizable, and that is exactly the
    population in question - so ring counting must not go through RDKit."""
    over_valent = graph(cycle(BENZENE_ORDERS, substituent=(0, 2)))
    assert aromatic_rings_of(over_valent["atom_types"],
                             over_valent["bond_matrix"]) is None, "RDKit refuses it"
    profile = ring_profile([over_valent])
    assert profile["rings_total"] == 1, "the ring is still counted"
    assert profile["graphs_with_a_ring_rate"] == 1.0
    assert profile["pattern_by_size"][6] == {"alternating": 1}


def test_ring_profile_separates_ring_sizes_and_patterns_over_a_population():
    profile = ring_profile([graph(cycle(BENZENE_ORDERS)), graph(cycle([1] * 6)),
                            graph(cycle([1] * 5))])
    assert profile["ring_size_counts"] == {5: 1, 6: 2}
    assert profile["rings_per_graph"] == 1.0
    assert profile["pattern_by_size"][6] == {"all_single": 1, "alternating": 1}
    assert profile["pattern_by_size"][5] == {"all_single": 1}
    assert profile["ring_atom_distribution"]["C"] == 1.0
    assert profile["population"] == "generated"
    empty = ring_profile([graph(np.zeros((4, 4), dtype=np.int64))])
    assert empty["rings_total"] == 0 and empty["graphs_with_a_ring"] == 0


def test_the_corrector_deletes_the_ring_double_bond_and_destroys_alternation():
    """H-AROMATIC, demonstrated rather than argued.

    One benzene with a double-bonded substituent on atom 0, which makes atom 0 over-valent
    (1 + 2 + 2 = 5 against cap 4). Two bonds tie for largest order: the RING double bond
    and the substituent double bond. Removing the substituent's order would leave an intact
    aromatic ring; `valency_correction` takes the ring bond instead, and the alternation -
    the only thing RDKit perceives aromaticity from in a Kekule representation - is gone.

    With 89.5% of graphs over-valent at N=23, almost every molecule passes through this.
    """
    before = cycle(BENZENE_ORDERS, substituent=(0, 2))
    atoms = np.full(7, CARBON, dtype=np.int64)
    assert before.sum(axis=1)[0] == 5 > ATOM_VALENCE[CARBON]
    assert before[0, 5] == 2 and before[0, 6] == 2, "the tie this test is about"

    _, after, info = valency_correction(atoms, before)
    assert info["bond_orders_removed"] == 1
    assert after[0, 6] == 2, "the SUBSTITUENT double bond survived ..."
    assert after[0, 5] == 1, "... and the RING double bond was taken instead"

    report = correction_damage([{"atom_types": atoms, "bond_matrix": before}])
    assert report["alternating_rings_before"] == 1
    assert report["alternating_rings_after"] == 0
    assert report["alternating_rings_destroyed"] == 1
    assert report["sanitizable_before"] == 0, "it is the alternation count that sees this"
    assert report["atoms_lost_mean"] == 0.0, "no atom was dropped, only aromaticity"


def test_correction_leaves_an_already_valid_aromatic_molecule_untouched():
    """The other half of the control: the damage measure must not fire on a clean ring."""
    atoms = np.full(6, CARBON, dtype=np.int64)
    report = correction_damage([{"atom_types": atoms,
                                 "bond_matrix": cycle(BENZENE_ORDERS)}])
    assert report["sanitizable_before"] == report["sanitizable_after"] == 1
    assert report["aromatic_rings_before_total"] == 1
    assert report["aromatic_rings_after_total"] == 1
    assert report["aromatic_lost_to_correction"] == 0
    assert report["alternating_rings_destroyed"] == 0
    assert report["aromatic_rings_before_per_graph"] == 1.0


def test_real_molecules_keep_their_aromaticity_through_our_own_round_trip():
    """The CEILING. If real molecules lost aromaticity in our serialisation, Kekulisation
    and reconstruction, the representation would be the bug and no model could keep it -
    so this number must be established before any generated aromaticity is interpreted."""
    reference = real_reference_profile(make_records())
    assert reference["molecules"] == 16
    assert reference["reconstructable_rate"] == 1.0, (
        "every real molecule must survive graph_to_mol_charge_aware")
    assert reference["aromatic_rings_per_molecule"] > 0.0
    assert reference["molecules_with_an_aromatic_ring"] >= 7
    assert reference["valence"]["over_valent_atoms"] == 0, "real data is valence-legal"
    assert reference["rings"]["population"] == "real_train"
    patterns = reference["rings"]["pattern_distribution_5_and_6"]
    assert patterns.get("alternating", 0) > 0, (
        "real aromatic rings must appear as alternating Kekule rings")
    assert reference["source"] == "train_split"
    assert real_reference_profile(make_records(), n=999)["molecules"] == 0


# ------------------------------------------------------------- the audit job ----
@pytest.fixture
def fixture_data(tmp_path):
    data = write_dataset(tmp_path / "data")
    payload = pickle.loads((Path(data) / "splits.pkl").read_bytes())
    (Path(data) / "scaler.json").write_text(
        json.dumps(ConditionScaler.fit(payload["train"]).to_dict()))
    return Path(data)


@pytest.fixture
def trained_flow(fixture_data, tmp_path):
    run = tmp_path / "flow"
    assert load_script("23_train_graph_flow.py").main([
        "--output-dir", str(run), "--data-dir", str(fixture_data),
        "--batch-size", "4", "--bond-blocks", "2", "--atom-blocks", "2",
        "--hidden", "32", "--conv-hidden", "16", "--condition-dim", "16",
        "--val-molecules", "8", "--warmup-steps", "1", "--device", "cpu",
        "--max-nodes", "18", "--eval-every", "2", "--save-every", "2",
        "--log-every", "2", "--max-steps", "4"]) == 0
    return run / "last.pt"


def _common_n(data: Path) -> int:
    from collections import Counter

    from zinc_gnf.dataset import load_training_splits

    train, _, _ = load_training_splits(data)
    return Counter(int(r["n_node"]) for r in train).most_common(1)[0][0]


def test_the_audit_runs_end_to_end_and_reports_every_requested_measurement(
        fixture_data, trained_flow, tmp_path):
    output = tmp_path / "audit"
    n = _common_n(fixture_data)
    digest = hashlib.sha256(trained_flow.read_bytes()).hexdigest()
    assert load_script("26_valence_aromaticity_audit.py").main([
        "--checkpoint", str(trained_flow), "--output-dir", str(output),
        "--data-dir", str(fixture_data), "--target-n", str(n),
        "--target-qed", "0.4,0.8", "--temperature", "1.0,0.85",
        "--count", "8", "--real-molecules", "16", "--seed", "17"]) == 0
    assert hashlib.sha256(trained_flow.read_bytes()).hexdigest() == digest, (
        "the checkpoint must be opened READ-ONLY")

    payload = json.loads((output / "valence_aromaticity_audit.json").read_text())
    assert payload["stage"] == 24
    assert payload["guarantees"]["read_only"]
    assert payload["guarantees"]["no_property_oracle"]
    assert len(payload["cells"]) == 4, "2 QED x 2 temperatures at one N"
    assert payload["real_reference"]["all"]["aromatic_rings_per_molecule"] > 0, (
        "the real ceiling must be reported, and reported first")

    for cell in payload["cells"]:
        for key in ("structure", "valence", "joint_atom_bond", "rings",
                    "best_case_relabelling", "correction_damage",
                    "argmax_shift_atoms", "argmax_shift_bonds"):
            assert cell[key], key
        # A: by element, by charge, by cap, with the excess and the sum histograms
        for key in ("by_element", "by_atom_type", "excess_histogram",
                    "bond_order_sum_histogram", "infeasible_for_every_type"):
            assert key in cell["valence"], key
        # C: both oracle arms, present because the fixture has donors at this N
        assert cell["oracle_atoms_generated_bonds"]["valence"]["atoms"] > 0
        assert cell["generated_atoms_oracle_bonds"]["valence"]["atoms"] > 0
        assert "REAL atom types" in cell["oracle_atoms_generated_bonds"]["note"]
        # B: before AND after correction, on the same graphs
        damage = cell["correction_damage"]
        assert "aromatic_rings_before_per_graph" in damage
        assert "aromatic_rings_after_per_graph" in damage
        assert "alternating_rings_destroyed" in damage
        assert cell["best_case_relabelling"]["valence_feasible_rate"] >= 0.0
        assert "per_graph" not in cell["best_case_relabelling"], "payload stays small"

    # the grid is what the stage asked for: N, requested QED and temperature
    assert {(c["requested_qed"], c["temperature"]) for c in payload["cells"]} == {
        (0.4, 1.0), (0.8, 1.0), (0.4, 0.85), (0.8, 0.85)}

    log = (output / "audit.log").read_text()
    assert "ceiling" in log
    assert "COUNTERFACTUAL best-case relabelling" in log
    assert "infeasible for ANY atom type" in log
    assert "alternating rings destroyed" in log


def test_the_audit_never_opens_the_test_split_or_scores_a_molecule_by_qed():
    """E's constraint, asserted on the source: no oracle QED reranking, no test data."""
    source = (REPOSITORY / "scripts" / "26_valence_aromaticity_audit.py").read_text()
    assert "load_test_split" not in source
    for forbidden in ("QED.qed", "Descriptors.qed", "qed_from_smiles", "molecular_profile",
                      "argsort", "nlargest", "top_k", "rerank"):
        assert forbidden not in source, f"{forbidden} would make this a selection, not an audit"
    assert "READ-ONLY" in source


def test_the_audit_validates_its_arguments_and_its_inputs(fixture_data, trained_flow,
                                                          tmp_path):
    script = load_script("26_valence_aromaticity_audit.py")
    assert script.main(["--checkpoint", str(trained_flow),
                        "--output-dir", str(tmp_path / "a"),
                        "--data-dir", str(fixture_data), "--target-n", ""]) == 4
    assert script.main(["--checkpoint", str(tmp_path / "absent.pt"),
                        "--output-dir", str(tmp_path / "b"),
                        "--data-dir", str(fixture_data)]) == 2
    assert script.main(["--checkpoint", str(trained_flow),
                        "--output-dir", str(tmp_path / "c"),
                        "--data-dir", str(tmp_path / "no-such-data")]) == 2


# ------------------------------------------------------------------- SLURM ------
def test_the_stage24_job_is_cpu_only_read_only_and_states_its_cost():
    path = REPOSITORY / "slurm" / "24_valence_aromaticity.sh"
    source = path.read_text(encoding="utf-8")
    directives = {}
    for line in source.splitlines():
        match = re.match(r"#SBATCH\s+--([\w-]+)(?:=(.*))?$", line.strip())
        if match:
            directives[match.group(1)] = (match.group(2) or "").strip()
    assert "gres" not in directives, "a read-only audit must not reserve a GPU"
    assert directives["partition"] == "CPU"
    assert directives["time"] and directives["mem"]
    assert 'if [[ -z "${SLURM_JOB_ID:-}" ]]' in source, "must refuse the login node"
    assert "Runtime estimate" in source and "Memory estimate" in source
    assert "READ-ONLY" in source

    executable = "\n".join(line for line in source.splitlines()
                           if not line.lstrip().startswith("#") and "echo " not in line)
    assert "scripts/26_valence_aromaticity_audit.py" in executable
    assert "23_train_graph_flow.py" not in executable, "stage 24 trains nothing"
    assert not re.search(r"(?m)^\s*(rm|mv)\s", executable), "no destructive commands"
    assert 'OUT="${OUT:-outputs/metrics/graph_flow_topology/valence_audit}"' in source
    for written in re.findall(r"mkdir -p (.*)", executable):
        for target in written.split():
            assert target in ('outputs/logs', '"${OUT}"'), target
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0


def test_the_stage24_documentation_states_the_remedy_and_its_tradeoff():
    """The recommendation has to be written down with what it costs, not just asserted."""
    text = (REPOSITORY / "docs" / "stage24_valence_aromaticity.md").read_text()
    for required in ("H-VALENCE", "H-AROMATIC", "exact likelihood", "atoms first",
                     "shuffled-QED", "raw validity", "Remaining uncertainty"):
        assert required in text, required
