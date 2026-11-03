"""Stage 16: the decoder-aligned structural objective and its fine-tune experiment.

These tests check the things the experiment's conclusion depends on:
  * the differentiable statistics agree with the discrete quantities they stand in for,
    measured on hand-built graphs where the answer is known;
  * the gradient actually reaches the flow (otherwise the term is decorative);
  * the targets come from TRAINING graphs only;
  * lambda = 0 reproduces pure maximum likelihood exactly, so the control arm is a real
    control;
  * the source checkpoint is never modified.
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
from zinc_gnf.constants_charge_aware import NUM_BOND_TYPES
from zinc_gnf.flow import node_mask_from_counts
from zinc_gnf.preprocessing import ConditionScaler
from zinc_gnf.structure_objective import (
    DEFAULT_WEIGHTS,
    TERMS,
    StructureTargets,
    structure_loss,
    structure_statistics,
)

REPOSITORY = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""),
                                                  REPOSITORY / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def logits_from_bonds(bonds: np.ndarray, *, sharpness: float = 30.0) -> torch.Tensor:
    """Near-one-hot bond logits for a known bond matrix, so the soft statistics are
    comparable with the discrete ones."""
    n = bonds.shape[0]
    logits = torch.zeros(1, n, n, NUM_BOND_TYPES)
    for i in range(n):
        for j in range(n):
            logits[0, i, j, int(bonds[i, j])] = sharpness
    return logits


def ring(n: int = 6, *, alternating: bool = True) -> np.ndarray:
    """A single ring; with `alternating` it is a Kekule aromatic ring."""
    bonds = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        j = (i + 1) % n
        order = 2 if (alternating and i % 2 == 0) else 1
        bonds[i, j] = bonds[j, i] = order
    return bonds


def chain(n: int = 5) -> np.ndarray:
    bonds = np.zeros((n, n), dtype=np.int64)
    for i in range(n - 1):
        bonds[i, i + 1] = bonds[i + 1, i] = 1
    return bonds


# ------------------------------------------- the statistics mean what they claim ---
def test_the_soft_statistics_match_a_known_benzene_ring():
    bonds = ring(6, alternating=True)
    mask = torch.ones(1, 6, dtype=torch.bool)
    atom_logits = torch.zeros(1, 6, 1)                       # a single atom type
    statistics = structure_statistics(logits_from_bonds(bonds), atom_logits, mask)
    # 6 bonds over 6 atoms, cycle rank 1, three of six bonds are double, none isolated
    assert float(statistics["bonds_per_atom"]) == pytest.approx(1.0, abs=1e-3)
    assert float(statistics["cycle_rank"]) == pytest.approx(1.0, abs=1e-2)
    assert float(statistics["double_bond_fraction"]) == pytest.approx(0.5, abs=1e-2)
    assert float(statistics["isolated_atom_rate"]) == pytest.approx(0.0, abs=1e-3)


def test_the_soft_statistics_match_a_known_chain():
    bonds = chain(5)
    mask = torch.ones(1, 5, dtype=torch.bool)
    statistics = structure_statistics(logits_from_bonds(bonds), torch.zeros(1, 5, 1), mask)
    assert float(statistics["bonds_per_atom"]) == pytest.approx(4 / 5, abs=1e-3)
    assert float(statistics["cycle_rank"]) == pytest.approx(0.0, abs=1e-2)   # a tree
    assert float(statistics["double_bond_fraction"]) == pytest.approx(0.0, abs=1e-3)


def test_an_empty_graph_is_all_isolated_atoms_which_is_the_failure_we_see():
    """The flow's samples look like this: everything 'no bond'."""
    n = 8
    mask = torch.ones(1, n, dtype=torch.bool)
    statistics = structure_statistics(logits_from_bonds(np.zeros((n, n), dtype=np.int64)),
                                      torch.zeros(1, n, 1), mask)
    assert float(statistics["bonds_per_atom"]) == pytest.approx(0.0, abs=1e-3)
    assert float(statistics["isolated_atom_rate"]) == pytest.approx(1.0, abs=1e-2)
    assert float(statistics["cycle_rank"]) == pytest.approx(1.0 - n, abs=1e-2)


def test_padding_is_excluded_from_every_statistic():
    bonds = np.zeros((4, 4), dtype=np.int64)
    bonds[0, 1] = bonds[1, 0] = 1
    logits = logits_from_bonds(bonds)
    mask = torch.tensor([[True, True, False, False]])
    statistics = structure_statistics(logits, torch.zeros(1, 4, 1), mask)
    assert float(statistics["bonds_per_atom"]) == pytest.approx(0.5, abs=1e-3)
    assert float(statistics["isolated_atom_rate"]) == pytest.approx(0.0, abs=1e-3)


def test_valence_overflow_fires_only_when_valence_is_exceeded():
    from zinc_gnf.constants_charge_aware import ATOM_VALENCE

    carbon = int(np.argmax(np.asarray(ATOM_VALENCE) == 4))
    n = 6
    mask = torch.ones(1, n, dtype=torch.bool)
    atom_logits = torch.full((1, n, len(ATOM_VALENCE)), -30.0)
    atom_logits[..., carbon] = 30.0
    fine = structure_statistics(logits_from_bonds(ring(n)), atom_logits, mask)
    assert float(fine["valence_overflow"]) == pytest.approx(0.0, abs=1e-2)
    # every pair triple-bonded: wildly over any valence cap
    everything = np.full((n, n), 3, dtype=np.int64)
    np.fill_diagonal(everything, 0)
    broken = structure_statistics(logits_from_bonds(everything), atom_logits, mask)
    assert float(broken["valence_overflow"]) > 5.0


# ------------------------------------------------------------------ targets -------
def test_targets_are_measured_from_training_graphs(tmp_path):
    from zinc_gnf.dataset import load_training_splits

    data = write_dataset(tmp_path / "data")
    train, val, _ = load_training_splits(data)
    targets = StructureTargets.from_records(train)
    assert targets.molecules == len(train)
    assert 0.0 < targets.bonds_per_atom < 3.0
    assert targets.isolated_atom_rate == pytest.approx(0.0, abs=1e-6)  # real molecules
    assert targets.valence_overflow == 0.0
    assert 0.0 <= targets.double_bond_fraction <= 1.0
    # round trip through the checkpoint representation
    assert StructureTargets.from_dict(targets.to_dict()).to_dict() == targets.to_dict()
    # and the validation records are a DIFFERENT population, so nothing is shared
    assert StructureTargets.from_records(val).molecules == len(val)


def test_targets_reject_an_empty_population():
    with pytest.raises(ValueError):
        StructureTargets.from_records([])


def test_known_targets_for_hand_built_molecules():
    records = [{"bond_matrix": ring(6, alternating=True)}, {"bond_matrix": chain(5)}]
    targets = StructureTargets.from_records(records)
    assert targets.bonds_per_atom == pytest.approx((1.0 + 4 / 5) / 2)
    assert targets.cycle_rank == pytest.approx((1 + 0) / 2)
    assert targets.double_bond_fraction == pytest.approx((0.5 + 0.0) / 2)


# -------------------------------------------------------------------- loss --------
def test_the_loss_is_zero_at_the_target_and_grows_away_from_it():
    targets = StructureTargets(bonds_per_atom=1.0, cycle_rank=2.0, double_bond_fraction=0.3,
                               isolated_atom_rate=0.0, valence_overflow=0.0, molecules=10)
    perfect = {name: torch.tensor([float(getattr(targets, name))]) for name in TERMS}
    value, report = structure_loss(perfect, targets)
    assert float(value) == pytest.approx(0.0, abs=1e-9)
    assert report["structure_loss"] == pytest.approx(0.0, abs=1e-9)
    worse = dict(perfect)
    worse["bonds_per_atom"] = torch.tensor([0.0])
    assert float(structure_loss(worse, targets)[0]) > 0.5
    assert report["bonds_per_atom_target"] == pytest.approx(1.0)


def test_the_loss_reports_every_term_even_at_zero_weight():
    targets = StructureTargets(1.0, 2.0, 0.3, 0.0, 0.0, 10)
    statistics = {name: torch.tensor([0.0]) for name in TERMS}
    weights = {"bonds_per_atom": 1.0}
    _, report = structure_loss(statistics, targets, weights=weights)
    for name in TERMS:
        assert f"{name}_value" in report and f"{name}_squared_error" in report
    with pytest.raises(ValueError, match="unknown structure terms"):
        structure_loss(statistics, targets, weights={"nonsense": 1.0})
    with pytest.raises(ValueError, match="every structure weight is zero"):
        structure_loss(statistics, targets, weights={"bonds_per_atom": 0.0})


def test_the_gradient_reaches_the_flow_parameters():
    """If this fails the structural term is decorative."""
    from tests.test_conditional_flow import conditional_flow, wake_up_conditioning
    from tests.test_flow_scripts_and_slurm import write_autoencoder
    import tempfile

    from zinc_gnf.generation import generation_batch
    from zinc_gnf.trainer_autoencoder import load_autoencoder

    with tempfile.TemporaryDirectory() as directory:
        autoencoder, _ = load_autoencoder(write_autoencoder(Path(directory) / "ae.pt"),
                                          device="cpu")
    autoencoder.eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)

    flow = wake_up_conditioning(conditional_flow(features=("n", "qed"), dimension=64))
    counts = [5, 6]
    mask = node_mask_from_counts(counts)
    condition = flow.conditioning(torch.zeros(2, 2), None, 2, "cpu")
    mu, sigma = flow.prior.parameters_for(2, condition, "cpu")
    noise = torch.randn(2, mask.shape[1], 64, generator=torch.Generator().manual_seed(0))
    z = (mu.unsqueeze(1) + sigma.unsqueeze(1) * noise) * mask.unsqueeze(-1)
    sampled = flow.inverse(z, mask, condition)
    outputs = autoencoder.decode(sampled, generation_batch(counts), teacher_forcing=False)
    statistics = structure_statistics(outputs["bond_logits"], outputs["atom_logits"], mask)
    targets = StructureTargets(1.1, 2.0, 0.3, 0.0, 0.0, 10)
    loss, _ = structure_loss(statistics, targets)
    loss.backward()
    touched = [name for name, parameter in flow.named_parameters()
               if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0]
    assert touched, "no flow parameter received a gradient from the structural term"
    assert any("prior" in name for name in touched), "the conditional prior must be reachable"


# ------------------------------------------------- the fine-tune, end to end ------
@pytest.fixture
def joint_checkpoint(tmp_path):
    import pickle

    from tests.test_flow_scripts_and_slurm import write_autoencoder

    data = write_dataset(tmp_path / "data")
    payload = pickle.loads((Path(data) / "splits.pkl").read_bytes())
    (Path(data) / "scaler.json").write_text(
        json.dumps(ConditionScaler.fit(payload["train"]).to_dict()))
    autoencoder = write_autoencoder(tmp_path / "ae.pt")
    run = tmp_path / "source"
    assert load_script("04_train_flow.py").main([
        "--config", str(REPOSITORY / "configs/zinc250k_flow_nqed.yaml"),
        "--data-dir", str(data), "--ae-checkpoint", str(autoencoder),
        "--output-dir", str(run), "--device", "cpu", "--seed", "1", "--batch-size", "4",
        "--max-steps", "4", "--num-blocks", "1", "--statistics-molecules", "20",
        "--val-molecules", "8", "--eval-every", "2", "--save-every", "2",
        "--log-every", "2", "--warmup-steps", "0"]) == 0
    return data, autoencoder, run / "last.pt"


def finetune(joint_checkpoint, output, *, lam, steps=2, extra=()):
    data, autoencoder, source = joint_checkpoint
    return load_script("17_structure_finetune.py").main([
        "--flow-checkpoint", str(source), "--ae-checkpoint", str(autoencoder),
        "--data-dir", str(data), "--output-dir", str(output),
        "--lambda-structure", str(lam), "--max-steps", str(steps),
        "--batch-size", "4", "--sample-batch", "3", "--val-molecules", "8",
        "--eval-every", "1", "--save-every", "1", "--log-every", "1",
        "--target-molecules", "20", "--device", "cpu", "--seed", "1", *extra])


def test_the_structural_arm_runs_and_logs_both_terms(joint_checkpoint, tmp_path):
    output = tmp_path / "arm"
    assert finetune(joint_checkpoint, output, lam=1.0) == 0
    configuration = json.loads((output / "config.json").read_text())
    assert configuration["arm"] == "structure"
    assert configuration["structure_targets"]["molecules"] == 20
    assert configuration["structure_weights"] == DEFAULT_WEIGHTS
    log = (output / "train.log").read_text()
    assert "no longer pure maximum likelihood" in log
    assert "bonds/atom" in log
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    train_rows = [r for r in records if r["kind"] == "train"]
    assert train_rows and "structure_loss" in train_rows[0]
    assert any(r["kind"] == "val" and "val_nll_per_node" in r for r in records)
    assert (output / "last.pt").exists()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["completed"] and summary["lambda_structure"] == 1.0


def test_lambda_zero_is_exactly_pure_likelihood(joint_checkpoint, tmp_path):
    """The control arm must contain no structural term at all."""
    output = tmp_path / "control"
    assert finetune(joint_checkpoint, output, lam=0.0) == 0
    assert json.loads((output / "config.json").read_text())["arm"] == "control_pure_mle"
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    for row in (r for r in records if r["kind"] == "train"):
        assert "structure_loss" not in row
        assert row["loss"] == pytest.approx(row["nll"], rel=1e-9)
    log = (output / "train.log").read_text()
    assert "bonds/atom=" not in log, "no structural term may be logged per step"
    assert "lambda_structure=0.0" in log


def test_the_source_checkpoint_is_never_modified(joint_checkpoint, tmp_path):
    _, _, source = joint_checkpoint
    before = source.read_bytes()
    assert finetune(joint_checkpoint, tmp_path / "arm2", lam=0.5) == 0
    assert source.read_bytes() == before, "the 50k checkpoint must be read-only"


def test_the_finetune_is_resumable_and_deterministic(joint_checkpoint, tmp_path):
    output = tmp_path / "resume"
    assert finetune(joint_checkpoint, output, lam=1.0, steps=2) == 0
    assert finetune(joint_checkpoint, output, lam=1.0, steps=4) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["final_step"] == 4
    assert "RESUMED" in (output / "train.log").read_text()


def test_bad_arguments_and_wrong_flow_kinds_are_refused(joint_checkpoint, tmp_path):
    data, autoencoder, source = joint_checkpoint
    script = load_script("17_structure_finetune.py")
    assert script.main(["--flow-checkpoint", str(source), "--output-dir", str(tmp_path / "x"),
                        "--lambda-structure", "-1", "--data-dir", str(data)]) == 4
    assert script.main(["--flow-checkpoint", str(tmp_path / "ghost.pt"),
                        "--output-dir", str(tmp_path / "y"), "--lambda-structure", "1",
                        "--data-dir", str(data)]) == 2
    n_only = tmp_path / "n_only"
    from tests.test_conditional_flow import run_training as run_n_only
    assert run_n_only(data, autoencoder, n_only) == 0
    assert script.main(["--flow-checkpoint", str(n_only / "last.pt"),
                        "--ae-checkpoint", str(autoencoder), "--data-dir", str(data),
                        "--output-dir", str(tmp_path / "z"), "--lambda-structure", "1",
                        "--max-steps", "1", "--device", "cpu"]) == 4


# ------------------------------------------------------------------ SLURM ---------
def test_the_experiment_job_runs_both_arms_and_keeps_them_comparable():
    path = REPOSITORY / "slurm" / "16_structure_experiment.sh"
    source = path.read_text(encoding="utf-8")
    directives = {}
    for line in source.splitlines():
        match = re.match(r"#SBATCH\s+--([\w-]+)(?:=(.*))?$", line.strip())
        if match:
            directives[match.group(1)] = (match.group(2) or "").strip()
    assert directives["partition"] == "STUD" and "gres" in directives
    assert 'if [[ -z "${SLURM_JOB_ID:-}" ]]' in source
    executable = "\n".join(line for line in source.splitlines()
                           if not line.lstrip().startswith("#") and "echo " not in line)
    assert "scripts/17_structure_finetune.py" in executable
    assert "scripts/13_generate_nqed.py" in executable
    assert "run_arm control 0.0" in executable, "the control must be run, not optional"
    assert 'run_arm arm "${LAMBDA}"' in executable
    assert 'local out="outputs/checkpoints/structure_${name}"' in source
    assert "outputs/samples/structure_${name}" in source
    assert "flow_nqed/seed_1/best.pt" in executable                    # read-only source
    # a word-boundary check: "run_arm (" contains "rm " as a substring
    assert not re.search(r"(?m)^\s*(rm|mv)\s", executable)
    # the two arms must differ ONLY in lambda: every other flag is shared in run_arm
    assert executable.count("--lambda-structure") == 1
    assert "GO / NO-GO" in source
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0
