"""Stage 12: N-conditioned training, sampling and the conditioning verification.

The point of these tests is the claim the whole stage rests on: that the condition
actually enters the density, that a requested N is standardised exactly the way training
standardised it, and that the verification would NOTICE a model which ignores its
conditioning. A run that completes proves nothing, so the ignored-conditioning case is
constructed explicitly (zeroing the conditioning pathways) and the check must fail on it.
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
from zinc_gnf.conditioning import (
    batch_condition,
    condition_columns,
    condition_embedding_for,
    condition_sensitivity,
    format_conditioning_check,
    latent_spread_by_n,
    load_condition_scaler,
    prior_response,
    standardise_condition,
)
from zinc_gnf.dataset import MolecularGraphDataset, make_batch
from zinc_gnf.flow import CONDITION_FEATURES, FlowConfig, NodeFlow, node_mask_from_counts
from zinc_gnf.generation import generate_molecules, requested_n_report
from zinc_gnf.preprocessing import CONDITION_NAMES, ConditionScaler

REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = "configs/zinc250k_flow_ncond.yaml"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""),
                                                  REPOSITORY / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scaler() -> ConditionScaler:
    return ConditionScaler(mean=(3.0, 0.75), std=(0.2, 0.1))


def conditional_flow(features=("n",), seed: int = 0, dimension: int = 8) -> NodeFlow:
    flow = NodeFlow(FlowConfig(node_embedding_dim=dimension, num_blocks=2, hidden_dim=16,
                               attention_dim=16, num_heads=2, node_autoregressive=True,
                               condition_features=features, condition_embedding_dim=8,
                               condition_hidden_dim=8), init_seed=seed)
    generator = torch.Generator().manual_seed(seed + 1)
    flow.set_data_statistics(torch.randn(dimension, generator=generator),
                            torch.rand(dimension, generator=generator) + 0.5)
    flow.eval()
    return flow


def wake_up_conditioning(flow: NodeFlow, scale: float = 0.5) -> NodeFlow:
    """A freshly built conditional flow is the UNCONDITIONAL map on purpose.

    Every FiLM head and the prior's mu/sigma heads are zero-initialised, so at step 0 the
    condition cannot change the density - which is correct behaviour, and also means a
    test of "conditioning works" on an untrained model would be testing nothing. This
    gives those heads non-zero weights, the way training would.
    """
    generator = torch.Generator().manual_seed(99)
    for name, parameter in flow.named_parameters():
        if any(key in name for key in ("film", "mu_head", "sigma_head")):
            with torch.no_grad():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * scale)
    return flow


class StubEncoder:
    """Returns a deterministic embedding per molecule, like a frozen encoder."""

    def encode(self, batch):  # noqa: ANN001
        torch.manual_seed(int(batch.n_node.sum()))
        values = torch.randn(batch.batch_size, batch.max_nodes, 8)
        return values * batch.node_mask.unsqueeze(-1)


def conditional_dataset(size: int = 24, dimension: int = 8) -> MolecularGraphDataset:
    """A tiny dataset whose records carry condition vectors, like the real one."""
    rng = np.random.default_rng(3)
    the_scaler = scaler()
    records = []
    for index in range(size):
        n = int(5 + index % 6)
        bonds = np.zeros((n, n), dtype=np.int64)
        for position in range(n - 1):
            bonds[position, position + 1] = bonds[position + 1, position] = 1
        qed = float(0.3 + 0.5 * rng.random())
        records.append({"atom_types": np.zeros(n, dtype=np.int64), "bond_matrix": bonds,
                        "smiles": f"C{index}", "qed": qed, "row_id": index,
                        "condition": the_scaler.transform(qed, n)})
    return MolecularGraphDataset(records)


# ------------------------------------------------- feature / column mapping ---
def test_feature_columns_match_the_dataset_layout():
    assert CONDITION_FEATURES == ("n", "qed")
    assert CONDITION_NAMES == ("log1p_n_node", "qed")
    assert condition_columns(("n",)) == [0]
    assert condition_columns(("qed",)) == [1]
    assert condition_columns(("n", "qed")) == [0, 1]
    with pytest.raises(ValueError):
        condition_columns(("mass",))


def test_requested_n_is_standardised_exactly_as_training_standardised_it():
    the_scaler = scaler()
    for n in (9, 23, 38):
        expected = the_scaler.transform(0.0, n)[0]
        got = standardise_condition(("n",), scaler=the_scaler, n=[n])
        assert got.shape == (1, 1)
        assert got[0, 0] == pytest.approx(expected, abs=1e-6)
    # ... and the dataset's own vector for a molecule agrees with the sampling path
    dataset = conditional_dataset()
    item = dataset[0]
    assert item["condition"][0] == pytest.approx(
        standardise_condition(("n",), scaler=scaler(), n=[item["n_node"]])[0, 0], abs=1e-6)


def test_n_only_models_never_need_a_qed_target_and_joint_models_do():
    assert standardise_condition(("n",), scaler=scaler(), n=[10]).shape == (1, 1)
    with pytest.raises(ValueError):
        standardise_condition(("n", "qed"), scaler=scaler(), n=[10])
    both = standardise_condition(("n", "qed"), scaler=scaler(), n=[10, 20], qed=0.8)
    assert both.shape == (2, 2)
    assert both[0, 1] == pytest.approx(both[1, 1])          # same QED target, same column


def test_batch_condition_selects_the_right_column_and_requires_vectors():
    dataset = conditional_dataset()
    batch = make_batch(dataset, [0, 1, 2])
    values, present = batch_condition(batch, ("n",))
    assert values.shape == (3, 1) and torch.equal(present, torch.ones_like(present))
    assert torch.allclose(values[:, 0], batch.condition[:, 0])
    both, _ = batch_condition(batch, ("n", "qed"))
    assert torch.allclose(both, batch.condition[:, :2])
    plain = write_dataset  # noqa: F841 - the fixture dataset below has no conditions
    bare = MolecularGraphDataset([{"atom_types": np.zeros(3, dtype=np.int64),
                                   "bond_matrix": np.zeros((3, 3), dtype=np.int64),
                                   "smiles": "CCC", "qed": 0.5, "row_id": 0}])
    with pytest.raises(ValueError, match="no condition vectors"):
        batch_condition(make_batch(bare, [0]), ("n",))


def test_scaler_is_loaded_from_the_dataset_directory(tmp_path):
    (tmp_path / "scaler.json").write_text(json.dumps(scaler().to_dict()))
    loaded = load_condition_scaler(tmp_path)
    assert loaded.mean == pytest.approx(scaler().mean)
    assert loaded.std == pytest.approx(scaler().std)
    (tmp_path / "scaler.json").write_text(json.dumps({**scaler().to_dict(), "fit_on": "all"}))
    with pytest.raises(ValueError, match="train"):
        load_condition_scaler(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_condition_scaler(tmp_path / "elsewhere")


# ------------------------------------------- the condition enters the density ---
def test_the_condition_changes_the_density_once_the_heads_are_non_zero():
    flow = wake_up_conditioning(conditional_flow())
    mask = node_mask_from_counts([6, 4])
    x = torch.randn(2, mask.shape[1], 8) * mask.unsqueeze(-1)
    low = flow.conditioning(torch.tensor([[-1.5], [-1.5]]), None, 2, "cpu")
    high = flow.conditioning(torch.tensor([[1.5], [1.5]]), None, 2, "cpu")
    assert not torch.allclose(flow.log_prob(x, mask, low)["log_prob"],
                              flow.log_prob(x, mask, high)["log_prob"])
    # and the prior itself responds
    mu_low, sigma_low = flow.prior.parameters_for(2, low, "cpu")
    mu_high, sigma_high = flow.prior.parameters_for(2, high, "cpu")
    assert not torch.allclose(mu_low, mu_high)
    assert not torch.allclose(sigma_low, sigma_high)


def test_an_untrained_conditional_flow_is_the_unconditional_map():
    """Zero-initialised heads: this is intended, and is why tests must wake them up."""
    flow = conditional_flow()
    mask = node_mask_from_counts([5])
    x = torch.randn(1, mask.shape[1], 8) * mask.unsqueeze(-1)
    low = flow.conditioning(torch.tensor([[-2.0]]), None, 1, "cpu")
    high = flow.conditioning(torch.tensor([[2.0]]), None, 1, "cpu")
    assert torch.allclose(flow.log_prob(x, mask, low)["log_prob"],
                          flow.log_prob(x, mask, high)["log_prob"], atol=1e-6)


def test_condition_embedding_shape_errors_are_caught_early():
    flow = conditional_flow()
    with pytest.raises(ValueError, match="required"):
        condition_embedding_for(flow, None, None, 2, "cpu")
    with pytest.raises(ValueError, match=r"\[B, 1\]"):
        condition_embedding_for(flow, torch.zeros(2, 2), None, 2, "cpu")
    with pytest.raises(ValueError, match="rows"):
        condition_embedding_for(flow, torch.zeros(3, 1), None, 2, "cpu")
    plain = NodeFlow(FlowConfig(node_embedding_dim=8, num_blocks=1, hidden_dim=8,
                                attention_dim=8, num_heads=2))
    with pytest.raises(ValueError, match="unconditional"):
        condition_embedding_for(plain, torch.zeros(2, 1), None, 2, "cpu")
    assert condition_embedding_for(plain, None, None, 2, "cpu") is None


# --------------------------------------------------------- the verification ---
def test_the_check_detects_a_model_that_uses_its_conditioning():
    flow = wake_up_conditioning(conditional_flow())
    result = condition_sensitivity(flow, StubEncoder(), conditional_dataset(), device="cpu",
                                   batch_size=8)
    assert result["gap_nats_per_node"] != 0.0
    assert result["exact_ties"] < 1.0
    assert result["molecules"] == 24 and result["nodes"] > 0


def test_the_check_fails_on_a_model_that_ignores_its_conditioning():
    """The decisive test: zero heads = conditioning cannot act, and the gap must be 0."""
    flow = conditional_flow()                      # zero-initialised heads
    result = condition_sensitivity(flow, StubEncoder(), conditional_dataset(), device="cpu",
                                   batch_size=8)
    assert result["gap_nats_per_node"] == pytest.approx(0.0, abs=1e-9)
    assert result["exact_ties"] == pytest.approx(1.0)
    assert result["molecules_better_with_true_condition"] == 0.0


def test_the_check_refuses_an_unconditional_flow():
    plain = NodeFlow(FlowConfig(node_embedding_dim=8, num_blocks=1, hidden_dim=8,
                                attention_dim=8, num_heads=2))
    plain.set_data_statistics(torch.zeros(8), torch.ones(8))
    for call in (condition_sensitivity, latent_spread_by_n):
        with pytest.raises(ValueError, match="conditional"):
            if call is latent_spread_by_n:
                call(plain, StubEncoder(), conditional_dataset(), n_values=[5], device="cpu")
            else:
                call(plain, StubEncoder(), conditional_dataset(), device="cpu")


def test_prior_response_is_flat_for_a_dead_prior_and_varies_for_a_live_one():
    dead = prior_response(conditional_flow(), scaler=scaler(), n_values=[10, 20, 30])
    assert dead["mu_spread_across_n"] == pytest.approx(0.0, abs=1e-9)
    assert dead["sigma_spread_across_n"] == pytest.approx(0.0, abs=1e-9)
    live = prior_response(wake_up_conditioning(conditional_flow()), scaler=scaler(),
                          n_values=[10, 20, 30])
    assert live["sigma_spread_across_n"] > 0
    assert [row["n"] for row in live["rows"]] == [10, 20, 30]


def test_latent_spread_by_n_reports_per_n_and_withholds_thin_buckets():
    flow = wake_up_conditioning(conditional_flow())
    result = latent_spread_by_n(flow, StubEncoder(), conditional_dataset(), n_values=[5, 7],
                                device="cpu", batch_size=8, min_nodes=1000)
    assert [row["n"] for row in result["rows"]] == [5, 7]
    assert all(row["zscore_rms"] is None for row in result["rows"])       # too few nodes
    generous = latent_spread_by_n(flow, StubEncoder(), conditional_dataset(), n_values=[5, 7],
                                  device="cpu", batch_size=8, min_nodes=1)
    assert all(row["zscore_rms"] > 0 for row in generous["rows"])
    assert "no calibration is installed" in generous["note"]


def test_the_report_says_atom_count_is_not_evidence():
    flow = wake_up_conditioning(conditional_flow())
    text = format_conditioning_check(
        condition_sensitivity(flow, StubEncoder(), conditional_dataset(), device="cpu",
                              batch_size=8),
        prior_response(flow, scaler=scaler(), n_values=[5, 7]))
    assert "permuted" in text.lower()
    assert "NOT" in text and "atom count" in text
    assert "satisfied by" in text


# -------------------------------------------------------------- sampling ------
def test_conditional_sampling_requires_and_uses_the_condition():
    flow = wake_up_conditioning(conditional_flow())

    class Decoder:
        config = type("C", (), {"degree_head": False})()

    from zinc_gnf.generation import NodeCountSampler

    counts = NodeCountSampler([5, 6, 7])
    with pytest.raises(ValueError, match="condition_values is required"):
        generate_molecules(flow, Decoder(), count=2, node_counts=counts, decoders=(),
                           target_n=6)
    bad = np.zeros((3, 1), dtype=np.float32)
    with pytest.raises(ValueError, match=r"\[2, 1\]"):
        generate_molecules(flow, Decoder(), count=2, node_counts=counts, decoders=(),
                           condition_values=bad, target_n=6)


def test_unconditional_sampling_rejects_condition_values():
    plain = NodeFlow(FlowConfig(node_embedding_dim=8, num_blocks=1, hidden_dim=8,
                                attention_dim=8, num_heads=2))
    plain.set_data_statistics(torch.zeros(8), torch.ones(8))
    from zinc_gnf.generation import NodeCountSampler
    with pytest.raises(ValueError, match="unconditional"):
        generate_molecules(plain, object(), count=1, node_counts=NodeCountSampler([5]),
                           decoders=(), condition_values=np.zeros((1, 1), dtype=np.float32))


def test_requested_n_report_separates_plumbing_from_chemistry():
    records = [
        {"n_node": 6, "degree": {"valid": True, "smiles": "c1ccccc1"}},      # 6 heavy atoms
        {"n_node": 6, "degree": {"valid": True, "smiles": "CCO"}},           # 3: a decoder bug
        {"n_node": 6, "degree": {"valid": False, "smiles": None}},
    ]
    report = requested_n_report(records, 6, decoders=("degree",))
    assert report["before_decoding"] == pytest.approx(1.0)
    after = report["after_decoding"]["degree"]
    assert after["valid_molecules"] == 2
    assert after["match_among_valid"] == pytest.approx(0.5)
    assert after["match_over_all_samples"] == pytest.approx(1 / 3)
    assert "structural" in report["note"]
    assert "permuted-condition gap" in report["note"]


# ------------------------------------------------------ training end to end ---
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
def tiny_conditional(tmp_path):
    """Tiny dataset + frozen autoencoder + the train-only scaler.json the real data has."""
    import pickle

    from tests.test_flow_scripts_and_slurm import write_autoencoder

    data = write_dataset(tmp_path / "data")
    payload = pickle.loads((Path(data) / "splits.pkl").read_bytes())
    fitted = ConditionScaler.fit(payload["train"])          # TRAIN records only, as in production
    (Path(data) / "scaler.json").write_text(json.dumps(fitted.to_dict()))
    return data, write_autoencoder(tmp_path / "ae.pt")


def test_the_shipped_conditional_config_is_n_conditioned_arm_b():
    from zinc_gnf.utils.config import load_config
    configuration = load_config(REPOSITORY / CONFIG)
    flow = FlowConfig.from_dict(configuration["flow"])
    assert flow.condition_features == ("n",)
    assert flow.conditional and flow.condition_dim == flow.condition_embedding_dim
    assert flow.node_autoregressive is True, "stage 12 reuses the frozen Arm B architecture"
    # unchanged from the frozen unconditional run
    assert (flow.num_blocks, flow.hidden_dim, flow.noise_sigma) == (12, 256, 0.05)
    assert configuration["training"]["output_dir"] == "outputs/checkpoints/flow_ncond"


def test_conditional_training_runs_resumes_and_records_the_condition_gap(tiny_conditional,
                                                                        tmp_path):
    data, autoencoder = tiny_conditional
    output = tmp_path / "run"
    assert run_training(data, autoencoder, output) == 0
    configuration = json.loads((output / "config.json").read_text())
    assert configuration["flow"]["condition_features"] == ["n"]
    log = (output / "train.log").read_text()
    assert "conditioned on n" in log
    assert "PERMUTED" in log
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    validations = [r for r in records if r.get("kind") == "val"]
    assert validations, "no validation record"
    for record in validations:
        assert "val_condition_gap" in record
        assert "val_condition_win_rate" in record
        assert np.isfinite(record["val_condition_gap"])
    # resume continues rather than restarting
    assert run_training(data, autoencoder, output, extra=("--max-steps", "6")) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["final_step"] == 6 and summary["completed"]


def test_training_refuses_a_dataset_without_condition_vectors(tiny_conditional, tmp_path):
    """A conditional run must die at startup, not silently train unconditioned."""
    import pickle

    data, autoencoder = tiny_conditional
    payload = pickle.loads((Path(data) / "splits.pkl").read_bytes())
    for split in ("train", "val", "test"):
        for record in payload[split]:
            record.pop("condition", None)
    stripped = tmp_path / "nocond"
    stripped.mkdir()
    (stripped / "splits.pkl").write_bytes(pickle.dumps(payload))
    assert run_training(stripped, autoencoder, tmp_path / "run2") == 4


def test_the_condition_features_flag_overrides_the_config(tiny_conditional, tmp_path):
    script = load_script("04_train_flow.py")
    data, autoencoder = tiny_conditional
    arguments = script.parse_args([
        "--config", str(REPOSITORY / CONFIG), "--data-dir", str(data),
        "--ae-checkpoint", str(autoencoder), "--output-dir", str(tmp_path / "x"),
        "--condition-features", "n,qed"])
    flow_config, _ = script.build_configs(arguments)
    assert flow_config.condition_features == ("n", "qed")
    arguments.condition_features = ""
    flow_config, _ = script.build_configs(arguments)
    assert flow_config.condition_features == () and not flow_config.conditional


# --------------------------------------------- the generation script, end to end ---
def test_the_generation_script_verifies_conditioning_then_samples(tiny_conditional, tmp_path):
    data, autoencoder = tiny_conditional
    run = tmp_path / "run"
    assert run_training(data, autoencoder, run) == 0
    script = load_script("12_generate_ncond.py")
    output = tmp_path / "ncond"
    arguments = ["--flow-checkpoint", str(run / "last.pt"), "--ae-checkpoint", str(autoencoder),
                 "--data-dir", str(data), "--output-dir", str(output), "--count", "4",
                 "--target-n", "9,12", "--batch-size", "2", "--check-molecules", "8",
                 "--temperature", "0.2", "--device", "cpu", "--no-diversity"]
    # Four training steps on 20 molecules may or may not clear the verification
    # thresholds; what must hold is that the exit code FOLLOWS the verdict, so a model
    # with unproven conditioning cannot quietly produce a results table.
    code = script.main(arguments)
    check = json.loads((output / "conditioning_check.json").read_text())
    assert code == (0 if check["passed"] else 5)
    assert "gap_nats_per_node" in check["sensitivity"]
    assert (output / "conditioning_check.txt").exists()

    assert script.main(arguments + ["--allow-weak-conditioning"]) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert set(summary["by_target_n"]) == {"N9", "N12"}
    for key, row in summary["by_target_n"].items():
        assert row["delivered_before_decoding"] == pytest.approx(1.0)
        assert "degree" in row["delivered_after_decoding"]
        assert row["requested_n"] == int(key[1:])
    for name in ("N9", "N12"):
        directory = output / name
        metrics = json.loads((directory / "metrics.json").read_text())
        assert metrics["samples"] == 4
        assert metrics["requested_n"] == int(name[1:])
        assert metrics["condition"]["features"] == ["n"]
        assert metrics["condition"]["scaler"]["fit_on"] == "train_only"
        assert len((directory / "samples.jsonl").read_text().splitlines()) == 4
        records = [json.loads(line) for line in
                   (directory / "samples.jsonl").read_text().splitlines()]
        assert all(r["n_node"] == int(name[1:]) for r in records)
        assert all(len(r["condition"]) == 1 for r in records)
    report = (output / "ncond_report.txt").read_text()
    assert "raw validity" in report and "fallback" in report
    assert "structural" in report


def test_the_generation_script_refuses_the_wrong_kind_of_flow(tiny_conditional, tmp_path):
    from tests.test_flow_scripts_and_slurm import run_training as run_unconditional

    data, autoencoder = tiny_conditional
    plain = tmp_path / "plain"
    assert run_unconditional(tmp_path, data, autoencoder, output=plain) == 0
    script = load_script("12_generate_ncond.py")
    assert script.main(["--flow-checkpoint", str(plain / "last.pt"),
                        "--ae-checkpoint", str(autoencoder), "--data-dir", str(data),
                        "--output-dir", str(tmp_path / "o"), "--count", "2",
                        "--device", "cpu"]) == 4


def test_default_target_n_comes_from_the_training_distribution():
    script = load_script("12_generate_ncond.py")
    counts = [9] * 10 + [15] * 50 + [23] * 100 + [30] * 40 + [38] * 5
    targets = script.default_target_n(counts)
    assert targets == sorted(set(targets))
    assert min(targets) >= min(counts) and max(targets) <= max(counts)
    assert 23 in targets                                   # the mode must be represented


# ------------------------------------------------------------------ SLURM -----
@pytest.mark.parametrize("name, partition, gres", [
    ("12_flow_ncond_smoke_gpu.sh", "STUD", True),
    ("12_flow_ncond_gpu.sh", "STUD", True),
    ("12_generate_ncond.sh", "CPU", False),
])
def test_slurm_jobs_follow_the_cluster_rules(name, partition, gres):
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
    assert 'if [[ -z "${SLURM_JOB_ID:-}" ]]' in source, "must refuse to run on the login node"
    assert "srun " in source
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0


def test_the_training_jobs_use_the_conditional_config_and_do_not_touch_frozen_outputs():
    for name in ("12_flow_ncond_smoke_gpu.sh", "12_flow_ncond_gpu.sh"):
        source = (REPOSITORY / "slurm" / name).read_text(encoding="utf-8")
        executable = "\n".join(line for line in source.splitlines()
                               if not line.lstrip().startswith("#") and "echo " not in line)
        assert "configs/zinc250k_flow_ncond.yaml" in executable
        assert "scripts/04_train_flow.py" in executable
        assert "flow_uncond" not in executable, "must not write into the frozen unconditional run"
        assert "rm " not in executable
    full = (REPOSITORY / "slurm" / "12_flow_ncond_gpu.sh").read_text(encoding="utf-8")
    assert "--resume auto" in full, "the full run must be resumable"
    smoke = (REPOSITORY / "slurm" / "12_flow_ncond_smoke_gpu.sh").read_text(encoding="utf-8")
    assert "--max-steps" in smoke
    assert "12_generate_ncond.py" in smoke, "the smoke test must also check the conditioning"
    assert "--allow-weak-conditioning" in smoke


def test_the_generation_script_exits_5_when_the_conditioning_is_dead(tiny_conditional, tmp_path):
    """The decisive end-to-end guard: a checkpoint whose conditioning pathways are zero.

    Such a model still produces molecules with exactly the requested atom count, because
    N sets how many node vectors are drawn. The job must refuse it anyway.
    """
    from zinc_gnf.trainer_flow import load_flow
    from zinc_gnf.utils.checkpointing import save_checkpoint

    data, autoencoder = tiny_conditional
    run = tmp_path / "run"
    assert run_training(data, autoencoder, run) == 0

    flow, probe = load_flow(run / "last.pt")
    with torch.no_grad():
        for name, parameter in flow.named_parameters():
            if any(key in name for key in ("film", "mu_head", "sigma_head",
                                           "condition_embedding")):
                parameter.zero_()
    dead = run / "dead.pt"
    extra = {k: v for k, v in probe.items()
             if k not in {"epoch", "best_metric", "configuration", "model_state_dict",
                          "optimizer_state_dict", "torch_rng_state", "cuda_rng_state"}}
    save_checkpoint(dead, flow, None, epoch=0, best_metric=1.0,
                    configuration=probe.get("configuration", {}), extra=extra)

    script = load_script("12_generate_ncond.py")
    output = tmp_path / "dead_out"
    arguments = ["--flow-checkpoint", str(dead), "--ae-checkpoint", str(autoencoder),
                 "--data-dir", str(data), "--output-dir", str(output), "--count", "4",
                 "--target-n", "9", "--batch-size", "2", "--check-molecules", "8",
                 "--device", "cpu", "--no-diversity"]
    assert script.main(arguments) == 5
    check = json.loads((output / "conditioning_check.json").read_text())
    assert check["passed"] is False
    assert check["sensitivity"]["gap_nats_per_node"] == pytest.approx(0.0, abs=1e-9)
    assert check["sensitivity"]["exact_ties"] == pytest.approx(1.0)
    assert check["prior_response"]["sigma_spread_across_n"] == pytest.approx(0.0, abs=1e-9) \
        or len(check["prior_response"]["rows"]) == 1
    assert not (output / "N9").exists(), "no generation numbers from an unverified model"
    # ... and the requested atom count would have been delivered anyway, which is the point
    assert script.main(arguments + ["--allow-weak-conditioning"]) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["by_target_n"]["N9"]["delivered_before_decoding"] == pytest.approx(1.0)
    assert summary["verification"]["passed"] is False
