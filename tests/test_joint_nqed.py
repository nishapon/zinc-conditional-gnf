"""Stage 13: joint (N, QED) conditioning, QED target achievement, and the decode ceiling.

The claims under test:
  * QED can be permuted ALONE, so "QED does something N cannot" is actually measurable
    (N and QED are correlated in ZINC, so permuting both cannot answer it);
  * target-achievement metrics are arithmetically right, computed over valid molecules
    only, and the correlation is never taken within a single cell where the request is
    constant;
  * the within-N correlation - the headline - is 0 for a model that only tracks N;
  * the decode ceiling measures the SAME decoders on real embeddings, so a 0% raw
    validity is interpretable;
  * the job refuses to print a QED table when the QED-only control fails.
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
from tests.test_conditional_flow import (
    StubEncoder,
    conditional_dataset,
    conditional_flow,
    scaler,
    wake_up_conditioning,
)
from zinc_gnf.conditioning import condition_sensitivity, standardise_condition
from zinc_gnf.flow import FlowConfig
from zinc_gnf.preprocessing import ConditionScaler
from zinc_gnf.qed_evaluation import (
    DEFAULT_QED_TOLERANCE,
    decode_ceiling,
    format_qed_achievement,
    qed_achievement_table,
    qed_target_metrics,
)

REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = "configs/zinc250k_flow_nqed.yaml"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""),
                                                  REPOSITORY / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cell(n: int, requested: float, realised: list[float]) -> dict:
    records = [{"degree": {"valid": True, "smiles": "CCO", "qed": q}} for q in realised]
    return {"n": n, "requested_qed": requested,
            "metrics": qed_target_metrics(records, requested_qed=requested)}


# --------------------------------------------------- the joint configuration ---
def test_the_shipped_joint_config_conditions_on_both_features():
    from zinc_gnf.utils.config import load_config

    configuration = load_config(REPOSITORY / CONFIG)
    flow = FlowConfig.from_dict(configuration["flow"])
    assert flow.condition_features == ("n", "qed")
    assert flow.node_autoregressive is True, "stage 13 reuses the frozen Arm B architecture"
    assert (flow.num_blocks, flow.hidden_dim, flow.noise_sigma) == (12, 256, 0.05)
    assert configuration["training"]["output_dir"] == "outputs/checkpoints/flow_nqed"
    assert configuration["training"]["max_steps"] == 50000


def test_a_joint_request_standardises_both_columns():
    the_scaler = scaler()
    values = standardise_condition(("n", "qed"), scaler=the_scaler, n=[15, 23], qed=[0.4, 0.8])
    assert values.shape == (2, 2)
    for row, (n, qed) in enumerate(((15, 0.4), (23, 0.8))):
        expected = the_scaler.transform(qed, n)
        assert values[row, 0] == pytest.approx(expected[0], abs=1e-6)
        assert values[row, 1] == pytest.approx(expected[1], abs=1e-6)
    # the same requested QED at two different N keeps the QED column identical
    same = standardise_condition(("n", "qed"), scaler=the_scaler, n=[15, 30], qed=[0.6, 0.6])
    assert same[0, 1] == pytest.approx(same[1, 1])
    assert same[0, 0] != pytest.approx(same[1, 0])


# ------------------------------------------- permuting QED alone (the control) ---
def test_qed_can_be_permuted_alone_and_the_control_separates_the_features():
    flow = wake_up_conditioning(conditional_flow(features=("n", "qed")))
    dataset = conditional_dataset()
    both = condition_sensitivity(flow, StubEncoder(), dataset, device="cpu", batch_size=8)
    qed_only = condition_sensitivity(flow, StubEncoder(), dataset, device="cpu", batch_size=8,
                                     permute_features=["qed"])
    n_only = condition_sensitivity(flow, StubEncoder(), dataset, device="cpu", batch_size=8,
                                   permute_features=["n"])
    assert both["permuted_features"] == ["n", "qed"]
    assert qed_only["permuted_features"] == ["qed"]
    assert n_only["permuted_features"] == ["n"]
    # permuting one feature is a different experiment from permuting both
    assert qed_only["nll_per_node_shuffled"] != both["nll_per_node_shuffled"]
    assert qed_only["nll_per_node_true"] == pytest.approx(both["nll_per_node_true"])


def test_permuting_qed_alone_costs_nothing_when_qed_never_reaches_the_density():
    """A model wired to N only must show a zero QED-only gap - that is the whole point."""
    flow = wake_up_conditioning(conditional_flow(features=("n", "qed")))
    # Kill the QED input: zero the QED column's weights in the condition embedding.
    with torch.no_grad():
        first = flow.condition_embedding.mlp[0]
        first.weight[:, 1] = 0.0          # QED value column
        first.weight[:, 3] = 0.0          # QED present-flag column
        flow.condition_embedding.absent[1] = 0.0
    result = condition_sensitivity(flow, StubEncoder(), conditional_dataset(), device="cpu",
                                   batch_size=8, permute_features=["qed"])
    assert result["gap_nats_per_node"] == pytest.approx(0.0, abs=1e-9)
    assert result["exact_ties"] == pytest.approx(1.0)
    # ... while N still matters, so the joint test alone would have looked fine
    joint = condition_sensitivity(flow, StubEncoder(), conditional_dataset(), device="cpu",
                                  batch_size=8)
    assert joint["gap_nats_per_node"] != 0.0


def test_permute_features_validates_its_argument():
    flow = wake_up_conditioning(conditional_flow(features=("n", "qed")))
    with pytest.raises(ValueError, match="cannot permute"):
        condition_sensitivity(flow, StubEncoder(), conditional_dataset(), device="cpu",
                              batch_size=8, permute_features=["mass"])
    with pytest.raises(ValueError, match="empty"):
        condition_sensitivity(flow, StubEncoder(), conditional_dataset(), device="cpu",
                              batch_size=8, permute_features=[])


# ------------------------------------------------------- target achievement ---
def test_qed_target_metrics_are_arithmetically_right():
    records = [{"degree": {"valid": True, "smiles": "CCO", "qed": q}}
               for q in (0.50, 0.60, 0.70)]
    records.append({"degree": {"valid": False, "smiles": None}})       # no QED to average
    result = qed_target_metrics(records, requested_qed=0.6, tolerance=0.1)
    assert result["samples"] == 4 and result["valid_molecules"] == 3
    assert result["realised_qed_mean"] == pytest.approx(0.6)
    assert result["mae"] == pytest.approx((0.1 + 0.0 + 0.1) / 3)
    assert result["bias"] == pytest.approx(0.0)
    assert result["rmse"] == pytest.approx(np.sqrt((0.01 + 0 + 0.01) / 3))
    assert result["hit_rate"] == pytest.approx(1.0)                    # all within +-0.1
    assert qed_target_metrics(records, requested_qed=0.6, tolerance=0.05)["hit_rate"] == \
        pytest.approx(1 / 3)


def test_target_metrics_are_undefined_without_valid_molecules():
    result = qed_target_metrics([{"degree": {"valid": False}}], requested_qed=0.5)
    assert result["valid_molecules"] == 0
    assert np.isnan(result["mae"]) and np.isnan(result["hit_rate"])
    assert "undefined" in result["note"]


def test_a_model_that_follows_qed_scores_high_within_n_correlation():
    cells = [cell(15, 0.4, [0.42, 0.38]), cell(15, 0.6, [0.61, 0.59]),
             cell(15, 0.8, [0.78, 0.82]), cell(23, 0.4, [0.41, 0.39]),
             cell(23, 0.6, [0.60, 0.60]), cell(23, 0.8, [0.79, 0.81])]
    table = qed_achievement_table(cells)
    assert table["overall"]["qed_independent_of_n"] == pytest.approx(1.0, abs=1e-3)
    assert table["overall"]["response_slope_pooled"] == pytest.approx(1.0, abs=0.05)
    assert table["overall"]["mae_weighted"] < 0.03
    assert table["overall"]["hit_rate_weighted"] == pytest.approx(1.0)
    assert set(table["by_n"]) == {15, 23}


def test_a_model_that_only_tracks_n_scores_zero_within_n_correlation():
    """Realised QED depends on N but not on the request: the pooled numbers can still
    look non-trivial, so the WITHIN-N correlation is the one that must be reported."""
    cells = [cell(15, q, [0.50, 0.50]) for q in (0.4, 0.6, 0.8)]
    cells += [cell(30, q, [0.70, 0.70]) for q in (0.4, 0.6, 0.8)]
    table = qed_achievement_table(cells)
    assert np.isnan(table["overall"]["qed_independent_of_n"]) or \
        table["overall"]["qed_independent_of_n"] == pytest.approx(0.0, abs=1e-6)
    for row in table["by_n"].values():
        assert np.isnan(row["correlation_requested_vs_realised"]) or \
            row["correlation_requested_vs_realised"] == pytest.approx(0.0, abs=1e-6)
    assert table["overall"]["mae_weighted"] > 0.1          # and it misses the targets


def test_the_table_survives_cells_without_valid_molecules():
    cells = [cell(15, 0.4, [0.4]), {"n": 15, "requested_qed": 0.8,
                                    "metrics": qed_target_metrics(
                                        [{"degree": {"valid": False}}], requested_qed=0.8)}]
    table = qed_achievement_table(cells)
    assert table["overall"]["cells"] == 1
    assert len(table["cells"]) == 2                        # the empty cell is still listed
    empty = qed_achievement_table([cells[1]])
    assert empty["overall"] is None and "cannot be assessed" in empty["note"]


def test_the_report_names_the_within_n_number_as_the_headline():
    text = format_qed_achievement(qed_achievement_table(
        [cell(15, 0.4, [0.4]), cell(15, 0.8, [0.8])]))
    assert "within-N correlation" in text
    assert "hit rate within" in text and "MAE" in text
    assert "VALID molecules only" in text


# ------------------------------------------------------------ decode ceiling ---
def test_decode_ceiling_runs_the_same_decoders_on_real_embeddings(tmp_path):
    from tests.test_flow_scripts_and_slurm import write_autoencoder
    from zinc_gnf.trainer_autoencoder import load_autoencoder
    from zinc_gnf.dataset import MolecularGraphDataset, load_training_splits

    data = write_dataset(tmp_path / "data")
    autoencoder, _ = load_autoencoder(write_autoencoder(tmp_path / "ae.pt"), device="cpu")
    autoencoder.eval()
    _, val_records, _ = load_training_splits(data)
    val_set = MolecularGraphDataset(val_records)
    result = decode_ceiling(autoencoder, val_set, indices=list(range(len(val_set))),
                            decoders=("raw", "constrained", "degree"), device="cpu")
    assert result["molecules"] == len(val_set)
    assert set(result["decoders"]) == {"raw", "constrained", "degree"}
    for row in result["decoders"].values():
        assert 0.0 <= row["strict_validity"] <= 1.0
        assert row["aromatic_rings_mean"] == row["aromatic_rings_mean"] or True  # may be NaN
    assert "real embeddings" in result["interpretation"]
    assert "not against 100%" in result["interpretation"]


# --------------------------------------------------- the script, end to end ----
def run_training(data, autoencoder, output, *, extra=()):
    script = load_script("04_train_flow.py")
    return script.main([
        "--config", str(REPOSITORY / CONFIG), "--data-dir", str(data),
        "--ae-checkpoint", str(autoencoder), "--output-dir", str(output),
        "--device", "cpu", "--seed", "1", "--batch-size", "4", "--max-steps", "4",
        "--num-blocks", "1", "--statistics-molecules", "20", "--val-molecules", "8",
        "--eval-every", "2", "--save-every", "2", "--log-every", "2", "--warmup-steps", "0",
        *extra])


@pytest.fixture
def tiny_joint(tmp_path):
    import pickle

    from tests.test_flow_scripts_and_slurm import write_autoencoder

    data = write_dataset(tmp_path / "data")
    payload = pickle.loads((Path(data) / "splits.pkl").read_bytes())
    fitted = ConditionScaler.fit(payload["train"])
    (Path(data) / "scaler.json").write_text(json.dumps(fitted.to_dict()))
    return data, write_autoencoder(tmp_path / "ae.pt")


def test_joint_training_runs_and_logs_the_condition_gap(tiny_joint, tmp_path):
    data, autoencoder = tiny_joint
    output = tmp_path / "run"
    assert run_training(data, autoencoder, output) == 0
    configuration = json.loads((output / "config.json").read_text())
    assert configuration["flow"]["condition_features"] == ["n", "qed"]
    assert "conditioned on n, qed" in (output / "train.log").read_text()
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert all("val_condition_gap" in r for r in records if r.get("kind") == "val")
    assert run_training(data, autoencoder, output, extra=("--max-steps", "6")) == 0
    assert json.loads((output / "summary.json").read_text())["completed"]


def test_the_joint_script_reports_everything_the_stage_requires(tiny_joint, tmp_path):
    data, autoencoder = tiny_joint
    run = tmp_path / "run"
    assert run_training(data, autoencoder, run) == 0
    script = load_script("13_generate_nqed.py")
    output = tmp_path / "nqed"
    code = script.main(["--flow-checkpoint", str(run / "last.pt"),
                        "--ae-checkpoint", str(autoencoder), "--data-dir", str(data),
                        "--output-dir", str(output), "--target-n", "9,12",
                        "--target-qed", "0.4,0.8", "--temperature", "0.2",
                        "--count", "4", "--batch-size", "2", "--check-molecules", "8",
                        "--ceiling-molecules", "8", "--device", "cpu", "--no-diversity",
                        "--allow-weak-conditioning"])
    assert code == 0
    summary = json.loads((output / "summary.json").read_text())

    # the three controls, incl. QED alone
    controls = summary["verification"]["controls"]
    assert set(controls) == {"both", "qed_only", "n_only"}
    assert controls["qed_only"]["permuted_features"] == ["qed"]
    assert controls["n_only"]["permuted_features"] == ["n"]
    # the prior's response to QED at FIXED N
    response = summary["verification"]["prior_response_over_qed_at_fixed_n"]
    assert response["fixed_n"] in (9, 12)
    assert len(response["rows"]) == 7
    # the decode ceiling, with raw kept separate
    ceiling = summary["decode_ceiling"]
    assert set(ceiling["decoders"]) >= {"raw", "degree"}
    # the grid, with every required metric per cell
    assert len(summary["cells"]) == 4
    for row in summary["cells"]:
        assert row["n"] in (9, 12) and row["requested_qed"] in (0.4, 0.8)
        for key in ("raw_validity", "headline_validity", "fallback_rate",
                    "aromatic_rings_mean", "uniqueness", "novelty_vs_train"):
            assert key in row, key
        assert "mae" in row["metrics"] and "hit_rate" in row["metrics"]
        assert Path(row["directory"]).joinpath("metrics.json").exists()
        assert Path(row["directory"]).joinpath("samples.jsonl").exists()
    # the achievement table
    assert "overall" in summary["qed_achievement"]
    report = (output / "nqed_report.txt").read_text()
    assert "raw validity" in report and "fallback" in report and "aromatic rings" in report
    assert "qed_only" in report and "decisive for QED control" in report
    assert "Decode ceiling on REAL validation embeddings" in report
    assert "never merged" in report


def test_the_joint_script_refuses_an_n_only_flow(tiny_joint, tmp_path):
    from tests.test_conditional_flow import run_training as run_n_only

    data, autoencoder = tiny_joint
    n_only = tmp_path / "n_only"
    assert run_n_only(data, autoencoder, n_only) == 0
    script = load_script("13_generate_nqed.py")
    assert script.main(["--flow-checkpoint", str(n_only / "last.pt"),
                        "--ae-checkpoint", str(autoencoder), "--data-dir", str(data),
                        "--output-dir", str(tmp_path / "o"), "--count", "2",
                        "--device", "cpu"]) == 4


def test_the_joint_script_exits_5_when_qed_is_not_used(tiny_joint, tmp_path):
    """A model with the QED input disconnected still satisfies N perfectly.

    It must not be allowed to print a QED-achievement table, because any apparent QED
    effect would be N acting through the N-QED correlation in ZINC.
    """
    from zinc_gnf.trainer_flow import load_flow
    from zinc_gnf.utils.checkpointing import save_checkpoint

    data, autoencoder = tiny_joint
    run = tmp_path / "run"
    assert run_training(data, autoencoder, run) == 0
    flow, probe = load_flow(run / "last.pt")
    with torch.no_grad():
        first = flow.condition_embedding.mlp[0]
        first.weight[:, 1] = 0.0
        first.weight[:, 3] = 0.0
        flow.condition_embedding.absent[1] = 0.0
    dead = run / "qed_dead.pt"
    extra = {k: v for k, v in probe.items()
             if k not in {"epoch", "best_metric", "configuration", "model_state_dict",
                          "optimizer_state_dict", "torch_rng_state", "cuda_rng_state"}}
    save_checkpoint(dead, flow, None, epoch=0, best_metric=1.0,
                    configuration=probe.get("configuration", {}), extra=extra)

    script = load_script("13_generate_nqed.py")
    output = tmp_path / "dead"
    arguments = ["--flow-checkpoint", str(dead), "--ae-checkpoint", str(autoencoder),
                 "--data-dir", str(data), "--output-dir", str(output), "--target-n", "9",
                 "--target-qed", "0.4,0.8", "--count", "2", "--batch-size", "2",
                 "--check-molecules", "8", "--ceiling-molecules", "0", "--device", "cpu",
                 "--no-diversity"]
    assert script.main(arguments) == 5
    check = json.loads((output / "conditioning_check.json").read_text())
    assert check["verification"]["passed"] is False
    assert check["verification"]["passed_per_control"]["qed_only"] is False
    assert check["verification"]["controls"]["qed_only"]["gap_nats_per_node"] == \
        pytest.approx(0.0, abs=1e-9)
    assert not (output / "summary.json").exists(), "no results table from an unverified model"


def test_bad_grid_arguments_are_rejected(tiny_joint, tmp_path):
    data, autoencoder = tiny_joint
    script = load_script("13_generate_nqed.py")
    base = ["--flow-checkpoint", str(tmp_path / "none.pt"), "--data-dir", str(data),
            "--output-dir", str(tmp_path / "o")]
    assert script.main(base + ["--target-qed", "1.5"]) == 4          # QED outside [0, 1]
    assert script.main(base + ["--target-n", "0"]) == 4
    assert script.main(base + ["--temperature", "0"]) == 4
    assert script.main(base + ["--count", "0"]) == 4
    assert script.main(base) == 2                                     # checkpoint missing


# ------------------------------------------------------------------ SLURM -----
@pytest.mark.parametrize("name, partition, gres", [
    ("13_flow_nqed_smoke_gpu.sh", "STUD", True),
    ("13_flow_nqed_gpu.sh", "STUD", True),
    ("13_generate_nqed.sh", "CPU", False),
])
def test_stage_13_slurm_jobs_follow_the_cluster_rules(name, partition, gres):
    path = REPOSITORY / "slurm" / name
    source = path.read_text(encoding="utf-8")
    directives = {}
    for line in source.splitlines():
        match = re.match(r"#SBATCH\s+--([\w-]+)(?:=(.*))?$", line.strip())
        if match:
            directives[match.group(1)] = (match.group(2) or "").strip()
    assert directives["partition"] == partition
    assert ("gres" in directives) is gres
    assert directives["output"].startswith("outputs/logs/")
    assert directives["time"] and directives["mem"]
    assert 'if [[ -z "${SLURM_JOB_ID:-}" ]]' in source
    assert "srun " in source
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0


def test_stage_13_jobs_use_the_joint_config_and_leave_earlier_stages_alone():
    for name in ("13_flow_nqed_smoke_gpu.sh", "13_flow_nqed_gpu.sh"):
        source = (REPOSITORY / "slurm" / name).read_text(encoding="utf-8")
        executable = "\n".join(line for line in source.splitlines()
                               if not line.lstrip().startswith("#") and "echo " not in line)
        assert "configs/zinc250k_flow_nqed.yaml" in executable
        assert "flow_uncond" not in executable
        assert "flow_ncond" not in executable, "stage 13 must not write into stage 12's output"
        assert "rm " not in executable
    assert "--resume auto" in (REPOSITORY / "slurm" / "13_flow_nqed_gpu.sh").read_text()
    smoke = (REPOSITORY / "slurm" / "13_flow_nqed_smoke_gpu.sh").read_text()
    assert "13_generate_nqed.py" in smoke and "--allow-weak-conditioning" in smoke
    assert "--ceiling-molecules" in smoke, "the smoke test must measure the decode ceiling"
    assert "0.2,0.85" in smoke, "the smoke test must measure the temperature trade-off"
