"""Stage 18: the conditional-signal measurement, the NaN mechanism, and stage B.

What these tests establish:
  * the delivered/required displacement measurement is correct on constructed cases where
    the answer is known (a prior that ignores QED must deliver exactly zero);
  * non-finite samples are COUNTED and never silently turned into a displacement, and the
    overflow mechanism behind the stage-17 FiLM NaNs is reproduced on purpose;
  * the audit's verdict no longer treats a non-finite variant as structural recovery;
  * stage B unfreezes exactly the named blocks' couplings at the lower learning rate and
    nothing else.
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
from zinc_gnf.conditional_signal import (
    delivered_displacement,
    format_signal_report,
    required_displacement,
    signal_ratio,
)
from zinc_gnf.flow import node_mask_from_counts
from zinc_gnf.preprocessing import ConditionScaler

REPOSITORY = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""),
                                                  REPOSITORY / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ready_flow(dimension: int = 8) -> object:
    flow = conditional_flow(features=("n", "qed"), dimension=dimension)
    flow.set_data_statistics(torch.zeros(dimension), torch.full((dimension,), 0.1))
    return flow


# ----------------------------------------------- the measurement is correct ----
def test_a_prior_that_ignores_qed_delivers_exactly_zero_displacement():
    """A freshly built conditional flow is neutral, so the measured move must be 0."""
    flow = ready_flow()
    result = delivered_displacement(flow, scaler=scaler(), n=6, qed_low=0.4, qed_high=0.8,
                                    count=16, device="cpu")
    assert result["usable_molecules"] == 16
    assert result["embedding_mean_shift_norm"] == pytest.approx(0.0, abs=1e-6)
    assert result["embedding_paired_shift_per_node"] == pytest.approx(0.0, abs=1e-6)
    assert result["shift_over_spread"] == pytest.approx(0.0, abs=1e-6)


def test_a_live_conditional_prior_delivers_a_non_zero_displacement():
    flow = wake_up_conditioning(ready_flow(), scale=0.3)
    result = delivered_displacement(flow, scaler=scaler(), n=6, qed_low=0.4, qed_high=0.8,
                                    count=32, device="cpu")
    assert result["embedding_mean_shift_norm"] > 0
    assert result["per_condition"]["low"]["prior_sigma_mean"] > 0
    assert result["per_condition"]["low"]["finite"]["nonfinite_rate"] == 0.0


def test_the_same_noise_is_used_for_both_conditions():
    """Otherwise the 'displacement' would mostly be sampling variance."""
    flow = ready_flow()
    first = delivered_displacement(flow, scaler=scaler(), n=6, qed_low=0.4, qed_high=0.8,
                                   count=16, device="cpu", seed=5)
    second = delivered_displacement(flow, scaler=scaler(), n=6, qed_low=0.4, qed_high=0.8,
                                    count=16, device="cpu", seed=5)
    assert first["embedding_mean_shift_norm"] == pytest.approx(
        second["embedding_mean_shift_norm"])
    # a neutral flow with shared noise gives EXACTLY zero; with independent noise it
    # could not
    assert first["embedding_paired_shift_per_node"] == pytest.approx(0.0, abs=1e-6)


def test_required_displacement_uses_training_molecules_at_that_exact_n(tmp_path):
    from tests.test_flow_scripts_and_slurm import write_autoencoder
    from zinc_gnf.dataset import MolecularGraphDataset, load_training_splits
    from zinc_gnf.trainer_autoencoder import load_autoencoder

    data = write_dataset(tmp_path / "data")
    autoencoder, _ = load_autoencoder(write_autoencoder(tmp_path / "ae.pt"), device="cpu")
    autoencoder.eval()
    train, _, _ = load_training_splits(data)
    dataset = MolecularGraphDataset(train)
    counts = np.asarray(dataset.n_node)
    qed = np.asarray(dataset.qed, dtype=np.float64)
    n = int(np.bincount(counts).argmax())
    low, high = float(np.min(qed[counts == n])), float(np.max(qed[counts == n]))
    result = required_displacement(autoencoder, dataset, n=n, qed_low=low, qed_high=high,
                                   window=0.05, device="cpu")
    assert result["low_molecules"] >= 1 and result["high_molecules"] >= 1
    assert result["embedding_mean_shift_norm"] >= 0.0
    # an unsupported target is reported as such, not as a zero requirement
    empty = required_displacement(autoencoder, dataset, n=n, qed_low=0.999, qed_high=1.0,
                                  device="cpu")
    assert empty["low_molecules"] == 0
    assert "not supported" in empty["note"]


def test_the_ratio_names_the_failure_mode():
    delivered = {"usable_molecules": 10, "embedding_mean_shift_norm": 0.01,
                 "embedding_shift_vector": [1.0, 0.0]}
    required = {"embedding_mean_shift_norm": 1.0, "embedding_shift_vector": [1.0, 0.0]}
    capacity = signal_ratio(delivered, required)
    assert capacity["ratio"] == pytest.approx(0.01)
    assert capacity["cosine"] == pytest.approx(1.0)
    assert "capacity" in capacity["verdict"]

    wrong_way = signal_ratio({**delivered, "embedding_mean_shift_norm": 1.0,
                              "embedding_shift_vector": [0.0, 1.0]}, required)
    assert wrong_way["cosine"] == pytest.approx(0.0)
    assert "DIRECTION" in wrong_way["verdict"]

    unmeasurable = signal_ratio({"usable_molecules": 0}, required)
    assert np.isnan(unmeasurable["ratio"])
    assert "not measurable" in unmeasurable["verdict"]


def test_the_report_explains_what_the_ratio_means():
    delivered = delivered_displacement(ready_flow(), scaler=scaler(), n=6, qed_low=0.4,
                                       qed_high=0.8, count=8, device="cpu")
    required = {"embedding_mean_shift_norm": 1.0, "embedding_shift_vector": [0.0] * 8,
                "low_molecules": 5, "high_molecules": 7}
    text = format_signal_report([{"n": 6, "delivered": delivered, "required": required,
                                  "ratio": signal_ratio(delivered, required)}])
    assert "delivered / required" in text
    assert "non-finite" in text
    assert "sampling spread" in text


# ------------------------------------------------- the NaN mechanism ----------
def test_non_finite_samples_are_counted_and_never_become_a_displacement():
    """The stage-17 FiLM NaNs must be reported as a failure, not as a huge move."""
    flow = ready_flow()
    original_inverse = flow.inverse

    def broken(z, mask, condition=None):       # noqa: ANN001
        out = original_inverse(z, mask, condition)
        return out * float("inf")

    flow.inverse = broken                      # type: ignore[method-assign]
    try:
        result = delivered_displacement(flow, scaler=scaler(), n=6, qed_low=0.4,
                                        qed_high=0.8, count=8, device="cpu")
    finally:
        flow.inverse = original_inverse        # type: ignore[method-assign]
    assert result["usable_molecules"] == 0
    assert result["per_condition"]["low"]["finite"]["nonfinite_rate"] == 1.0
    assert "non-finite" in result["note"]
    assert "embedding_mean_shift_norm" not in result
    assert np.isnan(signal_ratio(result, {"embedding_mean_shift_norm": 1.0,
                                          "embedding_shift_vector": [0.0] * 8})["ratio"])


def test_the_node_autoregressive_inverse_can_overflow_when_translations_are_large():
    """The documented mechanism: the log-scale is tanh-clamped but the TRANSLATION head
    is not, and the inverse recursion feeds recovered nodes back into it."""
    flow = ready_flow()
    block = flow.blocks[0]
    assert block.node_coupling is not None
    with torch.no_grad():
        # a large positive translation bias makes the recursion expansive
        block.node_coupling.network_translation = None       # (no such attribute: use heads)
        for name, parameter in block.node_coupling.named_parameters():
            if "translation_head" in name:
                parameter.fill_(60.0)
    mask = node_mask_from_counts([30])
    z = torch.randn(1, mask.shape[1], 8) * mask.unsqueeze(-1)
    x = flow.inverse(z, mask, flow.conditioning(torch.zeros(1, 2), None, 1, "cpu"))
    assert not torch.isfinite(x).all(), ("the recursion should overflow here; if this ever "
                                         "passes, the translation head has become bounded")


# ------------------------------------------------------- stage B wiring -------
@pytest.fixture
def stage_a_checkpoint(tmp_path):
    import pickle

    from tests.test_flow_scripts_and_slurm import write_autoencoder

    data = write_dataset(tmp_path / "data")
    payload = pickle.loads((Path(data) / "splits.pkl").read_bytes())
    (Path(data) / "scaler.json").write_text(
        json.dumps(ConditionScaler.fit(payload["train"]).to_dict()))
    autoencoder = write_autoencoder(tmp_path / "ae.pt")
    run = tmp_path / "joint"
    assert load_script("04_train_flow.py").main([
        "--config", str(REPOSITORY / "configs/zinc250k_flow_nqed.yaml"),
        "--data-dir", str(data), "--ae-checkpoint", str(autoencoder),
        "--output-dir", str(run), "--device", "cpu", "--seed", "1", "--batch-size", "4",
        "--max-steps", "4", "--num-blocks", "3", "--statistics-molecules", "20",
        "--val-molecules", "8", "--eval-every", "2", "--save-every", "2",
        "--log-every", "2", "--warmup-steps", "0"]) == 0
    return data, autoencoder, run / "last.pt"


def test_stage_b_unfreezes_exactly_the_named_blocks(stage_a_checkpoint, tmp_path):
    data, autoencoder, source = stage_a_checkpoint
    output = tmp_path / "stageb"
    assert load_script("17_structure_finetune.py").main([
        "--flow-checkpoint", str(source), "--ae-checkpoint", str(autoencoder),
        "--data-dir", str(data), "--output-dir", str(output), "--lambda-structure", "0",
        "--freeze-backbone", "--unfreeze-blocks", "1", "--backbone-lr", "1e-5",
        "--max-steps", "2", "--batch-size", "4", "--val-molecules", "8",
        "--eval-every", "1", "--save-every", "1", "--log-every", "1",
        "--target-molecules", "20", "--device", "cpu", "--seed", "1"]) == 0
    configuration = json.loads((output / "config.json").read_text())
    assert configuration["unfreeze_blocks"] == 1
    assert configuration["backbone_lr"] == 1e-5
    unfrozen = configuration["unfrozen_tensors"]
    assert unfrozen, "stage B must unfreeze something"
    # the LAST block only (it is applied first when sampling), couplings only
    assert all(name.startswith("blocks.2.") for name in unfrozen), unfrozen
    assert all(("couplings" in name) or ("node_coupling" in name) for name in unfrozen)
    assert not any("act_norm" in name or "linear" in name for name in unfrozen)
    assert "STAGE B" in (output / "train.log").read_text()

    # ActNorm and the invertible linears really did not move
    from zinc_gnf.trainer_flow import load_flow

    before, _ = load_flow(source)
    after, _ = load_flow(output / "last.pt")
    before_state, after_state = before.state_dict(), after.state_dict()
    for name in before_state:
        if "act_norm" in name or ".linear." in name:
            assert torch.equal(before_state[name], after_state[name]), name


def test_stage_b_requires_the_frozen_backbone_flag(stage_a_checkpoint, tmp_path):
    data, autoencoder, source = stage_a_checkpoint
    assert load_script("17_structure_finetune.py").main([
        "--flow-checkpoint", str(source), "--ae-checkpoint", str(autoencoder),
        "--data-dir", str(data), "--output-dir", str(tmp_path / "bad"),
        "--lambda-structure", "0", "--unfreeze-blocks", "1", "--max-steps", "1",
        "--device", "cpu"]) == 4


def test_the_signal_audit_script_runs_and_decides(stage_a_checkpoint, tmp_path):
    data, autoencoder, source = stage_a_checkpoint
    output = tmp_path / "signal"
    assert load_script("19_conditional_signal_audit.py").main([
        "--flow-checkpoint", str(source), "--ae-checkpoint", str(autoencoder),
        "--data-dir", str(data), "--output-dir", str(output), "--label", "test",
        "--target-n", "9", "--qed-pair", "0.4,0.8", "--count", "16",
        "--temperature", "0.2", "--real-molecules", "32", "--device", "cpu"]) == 0
    payload = json.loads((output / "signal.json").read_text())
    assert payload["label"] == "test"
    assert payload["results"] and "ratio" in payload["results"][0]
    assert "stage_b_justified" in payload["decision"]
    assert "reading" in payload["decision"]
    assert (output / "signal.txt").exists()


def test_the_signal_audit_refuses_a_flow_without_qed(stage_a_checkpoint, tmp_path):
    from tests.test_conditional_flow import run_training as run_n_only

    data, autoencoder, _ = stage_a_checkpoint
    n_only = tmp_path / "n_only"
    assert run_n_only(data, autoencoder, n_only) == 0
    assert load_script("19_conditional_signal_audit.py").main([
        "--flow-checkpoint", str(n_only / "last.pt"), "--ae-checkpoint", str(autoencoder),
        "--data-dir", str(data), "--output-dir", str(tmp_path / "s2"),
        "--device", "cpu"]) == 4


# ------------------------------------------------------------------ SLURM ------
def test_the_stage18_job_gates_stage_b_on_the_audit():
    path = REPOSITORY / "slurm" / "18_signal_and_stageb.sh"
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
    assert "scripts/19_conditional_signal_audit.py" in executable
    assert "scripts/17_structure_finetune.py" in executable
    assert "scripts/13_generate_nqed.py" in executable
    # stage B must be skipped unless the audit justifies it
    assert 'if [[ "${JUSTIFIED}" != "True" ]]; then' in executable
    assert "--unfreeze-blocks" in executable and "--backbone-lr" in executable
    assert "--lambda-structure 0" in executable
    assert "structure_objective" not in executable
    assert not re.search(r"(?m)^\s*(rm|mv)\s", executable)
    for fragment in ("outputs/metrics/signal_stageA", "outputs/checkpoints/stageb_arm",
                     "outputs/checkpoints/stageb_control", "outputs/samples/stageb_"):
        assert fragment in source
    assert "GO / NO-GO" in source
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0
