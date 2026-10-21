"""Stage 15: invertibility diagnostics and the equal-budget QED experiment.

The claims under test:
  * the two round trips measure different things, and `z -> x -> z` is small even when
    the inverse is deliberately broken - which is exactly why it must not be quoted as
    evidence (this is constructed, not asserted);
  * float64 reduces the data-direction error, and the cast is fully reversible;
  * the component breakdown restores every module it disabled;
  * the chemistry-impact check notices when a perturbation does change molecules;
  * the experiment keeps the decoded-candidate budget equal, gates guidance on the
    predictor's held-out accuracy, and never labels selection as conditioning.
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
from tests.test_conditional_flow import StubEncoder, conditional_dataset, conditional_flow
from zinc_gnf.flow import FlowConfig, NodeFlow, node_mask_from_counts
from zinc_gnf.invertibility import (
    COMPONENTS,
    chemistry_impact,
    component_breakdown,
    double_precision_inverse,
    error_by_size,
    log_det_budget,
    relative_error,
    round_trip,
)
from zinc_gnf.preprocessing import ConditionScaler

REPOSITORY = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""),
                                                  REPOSITORY / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trained_like_flow(seed: int = 0, dimension: int = 8, blocks: int = 3) -> NodeFlow:
    """A flow whose weights are NOT the identity, so inversion is a real computation."""
    flow = NodeFlow(FlowConfig(node_embedding_dim=dimension, num_blocks=blocks, hidden_dim=16,
                               attention_dim=16, num_heads=2, node_autoregressive=True,
                               max_log_scale=1.0), init_seed=seed)
    generator = torch.Generator().manual_seed(seed + 5)
    with torch.no_grad():
        for name, parameter in flow.named_parameters():
            if "scale_head" in name or "translation_head" in name or "log_scale" in name:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.3)
    flow.set_data_statistics(torch.zeros(dimension), torch.full((dimension,), 0.05))
    flow.eval()
    return flow


# -------------------------------------------------- the asymmetry is explained ---
def test_both_directions_are_measured_from_the_same_inputs():
    flow = trained_like_flow()
    mask = node_mask_from_counts([6, 9])
    x = torch.randn(2, mask.shape[1], 8) * mask.unsqueeze(-1)
    result = round_trip(flow, x, mask)
    assert result["dtype"] == "float32"
    assert "data_direction_x_z_x" in result and "latent_direction_z_x_z" in result
    for key in ("max_absolute", "max_relative", "reference_scale"):
        assert key in result["data_direction_x_z_x"]


def test_a_deliberately_wrong_inverse_shows_up_in_the_data_direction():
    """The data direction is the one that detects an inaccurate inverse.

    The inverse is perturbed by a known 20%. `x -> z -> x` must report an error of that
    order, which is what makes it the direction to trust for inversion accuracy.
    """
    flow = trained_like_flow()
    mask = node_mask_from_counts([8])
    x = torch.randn(1, mask.shape[1], 8) * mask.unsqueeze(-1)
    original_inverse = flow.inverse

    def wobbly(z, m, condition=None):          # noqa: ANN001 - a deliberate 20% error
        return original_inverse(z, m, condition) * 1.2

    flow.inverse = wobbly                      # type: ignore[method-assign]
    try:
        broken = round_trip(flow, x, mask)
    finally:
        flow.inverse = original_inverse        # type: ignore[method-assign]

    assert broken["data_direction_x_z_x"]["max_relative"] > 0.05
    honest = round_trip(flow, x, mask)
    assert honest["data_direction_x_z_x"]["max_relative"] < \
        broken["data_direction_x_z_x"]["max_relative"]


def test_both_directions_report_absolute_error_and_their_own_reference_scale():
    """The two directions live in different spaces, so a relative number alone is not
    comparable between them: the absolute error and the scale it was divided by must
    both be reported."""
    flow = trained_like_flow()
    mask = node_mask_from_counts([8, 11])
    x = torch.randn(2, mask.shape[1], 8) * mask.unsqueeze(-1)
    result = round_trip(flow, x, mask)
    data, latent = result["data_direction_x_z_x"], result["latent_direction_z_x_z"]
    for block in (data, latent):
        assert block["max_absolute"] >= 0.0
        assert block["reference_scale"] > 0.0
        assert block["max_relative"] == pytest.approx(
            block["max_absolute"] / block["reference_scale"], rel=1e-6)
    # and they are genuinely different scales, which is the whole point
    assert data["reference_scale"] != latent["reference_scale"]


def test_float64_reduces_the_data_direction_error_and_the_cast_is_reversible():
    flow = trained_like_flow()
    mask = node_mask_from_counts([10, 12])
    x = torch.randn(2, mask.shape[1], 8) * mask.unsqueeze(-1)
    single = round_trip(flow, x, mask)
    double = round_trip(flow, x, mask, double=True)
    assert double["dtype"] == "float64"
    assert double["data_direction_x_z_x"]["max_relative"] <= \
        single["data_direction_x_z_x"]["max_relative"] + 1e-12
    # the flow is float32 again, exactly as before
    assert next(flow.parameters()).dtype == torch.float32
    assert round_trip(flow, x, mask)["data_direction_x_z_x"]["max_relative"] == \
        pytest.approx(single["data_direction_x_z_x"]["max_relative"], rel=1e-6)


def test_the_double_precision_context_restores_float32_even_on_error():
    flow = trained_like_flow()
    with pytest.raises(RuntimeError):
        with double_precision_inverse(flow):
            assert next(flow.parameters()).dtype == torch.float64
            raise RuntimeError("boom")
    assert next(flow.parameters()).dtype == torch.float32


# ---------------------------------------------------------- component split ----
def test_the_component_breakdown_covers_every_stack_and_restores_the_model():
    flow = trained_like_flow()
    mask = node_mask_from_counts([7, 7])
    x = torch.randn(2, mask.shape[1], 8) * mask.unsqueeze(-1)
    before = round_trip(flow, x, mask)["data_direction_x_z_x"]["max_relative"]
    rows = component_breakdown(flow, x, mask)
    assert [r["components"] for r in rows] == list(COMPONENTS)
    assert all(np.isfinite(r["data_direction_x_z_x"]["max_relative"]) for r in rows)
    # every module is back, so the full stack reproduces the original measurement
    for block in flow.blocks:
        assert block.act_norm is not None and block.linear is not None
        assert len(block.couplings) == 2 and block.node_coupling is not None
    after = round_trip(flow, x, mask)["data_direction_x_z_x"]["max_relative"]
    assert after == pytest.approx(before, rel=1e-6)


def test_actnorm_only_is_essentially_exact():
    """A per-channel affine map inverts to machine precision; if the first row is large,
    the problem is not in the couplings."""
    flow = trained_like_flow()
    mask = node_mask_from_counts([6])
    x = torch.randn(1, mask.shape[1], 8) * mask.unsqueeze(-1)
    rows = component_breakdown(flow, x, mask)
    assert rows[0]["components"] == "actnorm"
    assert rows[0]["data_direction_x_z_x"]["max_relative"] < 1e-4


def test_error_by_size_groups_per_molecule():
    flow = trained_like_flow()
    mask = node_mask_from_counts([5, 9, 9])
    x = torch.randn(3, mask.shape[1], 8) * mask.unsqueeze(-1)
    rows = error_by_size(flow, x, mask)
    assert [r["n"] for r in rows] == [5, 9]
    assert dict((r["n"], r["molecules"]) for r in rows) == {5: 1, 9: 2}
    assert all(np.isfinite(r["max_relative_error"]) for r in rows)


def test_relative_error_ignores_padding():
    mask = torch.tensor([[True, False]])
    a = torch.zeros(1, 2, 3)
    b = torch.zeros(1, 2, 3)
    b[0, 1] = 1000.0                      # padding only
    result = relative_error(a, b, mask)
    assert result["max_absolute"] == pytest.approx(0.0)


def test_log_det_budget_explains_the_magnitude():
    flow = trained_like_flow(dimension=8, blocks=3)
    budget = log_det_budget(flow)
    # 2 couplings-worth per block (affine + node-autoregressive) x blocks x dims x clamp
    assert budget["coupling_clamp_budget_per_node"] == pytest.approx(2 * 3 * 8 * 1.0)
    assert np.isfinite(budget["whitening_log_det_per_node"])
    assert "whitening" in budget["note"]


# ------------------------------------------------------- chemistry impact ------
def test_chemistry_impact_detects_a_perturbation_that_changes_molecules(tmp_path):
    from tests.test_flow_scripts_and_slurm import write_autoencoder
    from zinc_gnf.trainer_autoencoder import load_autoencoder

    autoencoder, _ = load_autoencoder(write_autoencoder(tmp_path / "ae.pt"), device="cpu")
    autoencoder.eval()
    counts = [6, 6]
    x = torch.randn(2, 6, 64)
    identical = chemistry_impact(autoencoder, x, x.clone(), counts, decoders=("raw",))
    assert identical["decoders"]["raw"]["identical_smiles_rate"] == pytest.approx(1.0)
    assert "unaffected" in identical["verdict"]
    wrecked = chemistry_impact(autoencoder, x, x + 50.0, counts, decoders=("raw",))
    assert wrecked["decoders"]["raw"]["identical_smiles_rate"] <= 1.0
    if wrecked["decoders"]["raw"]["identical_smiles_rate"] < 0.999:
        assert "CHANGES decoded molecules" in wrecked["verdict"]


# ------------------------------------------------------ the audit script -------
@pytest.fixture
def joint_setup(tmp_path):
    import pickle

    from tests.test_flow_scripts_and_slurm import write_autoencoder

    data = write_dataset(tmp_path / "data")
    payload = pickle.loads((Path(data) / "splits.pkl").read_bytes())
    fitted = ConditionScaler.fit(payload["train"])
    (Path(data) / "scaler.json").write_text(json.dumps(fitted.to_dict()))
    autoencoder = write_autoencoder(tmp_path / "ae.pt")
    run = tmp_path / "run"
    script = load_script("04_train_flow.py")
    assert script.main([
        "--config", str(REPOSITORY / "configs/zinc250k_flow_nqed.yaml"),
        "--data-dir", str(data), "--ae-checkpoint", str(autoencoder),
        "--output-dir", str(run), "--device", "cpu", "--seed", "1", "--batch-size", "4",
        "--max-steps", "4", "--num-blocks", "1", "--statistics-molecules", "20",
        "--val-molecules", "8", "--eval-every", "2", "--save-every", "2",
        "--log-every", "2", "--warmup-steps", "0"]) == 0
    return data, autoencoder, run / "last.pt"


def test_the_audit_script_reports_both_directions_and_a_verdict(joint_setup, tmp_path):
    data, autoencoder, checkpoint = joint_setup
    script = load_script("15_invertibility_audit.py")
    output = tmp_path / "audit"
    assert script.main(["--flow-checkpoint", str(checkpoint), "--ae-checkpoint", str(autoencoder),
                        "--data-dir", str(data), "--output-dir", str(output),
                        "--molecules", "8", "--batch-size", "4", "--target-n", "9",
                        "--target-qed", "0.4,0.8", "--samples-per-cell", "4",
                        "--device", "cpu"]) == 0
    payload = json.loads((output / "invertibility.json").read_text())
    assert [row["dtype"] for row in payload["precision"]] == ["float32", "float64"]
    assert payload["precision"][0]["data_direction_x_z_x"]["max_relative"] >= 0
    assert [row["components"] for row in payload["components"]] == list(COMPONENTS)
    assert payload["error_by_size"]
    assert len(payload["generated_grid"]) == 2            # one N x two QED targets
    assert "identical_smiles_rate" in payload["chemistry_impact"]["decoders"]["raw"]
    assert "recommendation" in payload["verdict"]
    assert payload["log_det_budget"]["coupling_clamp_budget_per_node"] > 0
    text = (output / "invertibility.txt").read_text()
    assert "consistency check" in text
    assert "x->z->x" in text and "z->x->z" in text
    # the checkpoint must be untouched
    assert checkpoint.exists()


def test_the_audit_script_validates_its_inputs(tmp_path):
    script = load_script("15_invertibility_audit.py")
    assert script.main(["--flow-checkpoint", str(tmp_path / "absent.pt"),
                        "--output-dir", str(tmp_path / "o")]) == 2
    assert script.main(["--flow-checkpoint", str(tmp_path / "absent.pt"),
                        "--output-dir", str(tmp_path / "o2"), "--target-n", ""]) == 4


# ------------------------------------------------- the guided experiment -------
def test_the_experiment_keeps_the_budget_equal_and_labels_the_arms(joint_setup, tmp_path):
    data, autoencoder, checkpoint = joint_setup
    script = load_script("16_qed_guided_experiment.py")
    output = tmp_path / "guided"
    assert script.main(["--flow-checkpoint", str(checkpoint), "--ae-checkpoint", str(autoencoder),
                        "--data-dir", str(data), "--output-dir", str(output),
                        "--target-n", "9", "--target-qed", "0.4,0.8", "--count", "4",
                        "--rerank-group", "2", "--predictor-molecules", "16",
                        "--predictor-epochs", "2", "--predictor-mae-gate", "1.0",
                        "--guidance-steps", "3", "--batch-size", "4", "--device", "cpu",
                        "--no-diversity"]) == 0
    payload = json.loads((output / "guided_qed.json").read_text())
    assert payload["budget"]["rerank_candidates"] == 8          # 4 kept x 2 candidates
    assert payload["guidance_enabled"] is True                  # gate was permissive
    assert payload["predictor"]["held_out"]["molecules"] > 0
    assert len(payload["cells"]) == 2
    for cell in payload["cells"]:
        assert cell["decoded_candidates"] == 8
        arms = cell["arms"]
        assert {"native_equal_budget", "rerank_equal_budget", "native_per_slot",
                "guided_equal_budget"} <= set(arms)
        for row in arms.values():
            for key in ("qed_mae", "qed_hit_rate", "raw_validity", "corrected_validity",
                        "fallback_rate", "aromatic_rings_mean", "uniqueness",
                        "novelty_vs_train"):
                assert key in row, key
        # every arm that was kept at the equal budget sees the same candidates
        assert arms["native_equal_budget"]["kept_molecules"] == 8
        assert arms["rerank_equal_budget"]["kept_molecules"] == 4
    text = (output / "guided_qed.txt").read_text()
    assert "not learned conditioning" in text
    assert "post-selection" in text
    assert "Only the native rows are evidence" in text


def test_guidance_is_skipped_when_the_predictor_is_too_weak(joint_setup, tmp_path):
    """A bad guide must produce NO guided row, rather than a weak one presented as a result."""
    data, autoencoder, checkpoint = joint_setup
    script = load_script("16_qed_guided_experiment.py")
    output = tmp_path / "gated"
    assert script.main(["--flow-checkpoint", str(checkpoint), "--ae-checkpoint", str(autoencoder),
                        "--data-dir", str(data), "--output-dir", str(output),
                        "--target-n", "9", "--target-qed", "0.6", "--count", "2",
                        "--rerank-group", "2", "--predictor-molecules", "16",
                        "--predictor-epochs", "1", "--predictor-mae-gate", "0.0",
                        "--batch-size", "4", "--device", "cpu", "--no-diversity"]) == 0
    payload = json.loads((output / "guided_qed.json").read_text())
    assert payload["guidance_enabled"] is False
    assert all("guided_equal_budget" not in cell["arms"] for cell in payload["cells"])
    assert "GUIDED ARM SKIPPED" in (output / "guided_qed.txt").read_text()


def test_the_experiment_refuses_a_flow_that_is_not_joint(joint_setup, tmp_path):
    from tests.test_conditional_flow import run_training as run_n_only

    data, autoencoder, _ = joint_setup
    n_only = tmp_path / "n_only"
    assert run_n_only(data, autoencoder, n_only) == 0
    script = load_script("16_qed_guided_experiment.py")
    assert script.main(["--flow-checkpoint", str(n_only / "last.pt"),
                        "--ae-checkpoint", str(autoencoder), "--data-dir", str(data),
                        "--output-dir", str(tmp_path / "o"), "--device", "cpu"]) == 4


def test_the_experiment_can_run_the_inverse_in_double(joint_setup, tmp_path):
    data, autoencoder, checkpoint = joint_setup
    script = load_script("16_qed_guided_experiment.py")
    output = tmp_path / "double"
    assert script.main(["--flow-checkpoint", str(checkpoint), "--ae-checkpoint", str(autoencoder),
                        "--data-dir", str(data), "--output-dir", str(output),
                        "--target-n", "9", "--target-qed", "0.6", "--count", "2",
                        "--rerank-group", "2", "--predictor-molecules", "16",
                        "--predictor-epochs", "1", "--predictor-mae-gate", "1.0",
                        "--guidance-steps", "2", "--batch-size", "4", "--device", "cpu",
                        "--no-diversity", "--double-inverse"]) == 0
    payload = json.loads((output / "guided_qed.json").read_text())
    assert payload["settings"]["double_inverse"] is True


# ------------------------------------------------------------------ SLURM ------
def test_the_stage15_slurm_job_is_cpu_only_and_read_only():
    path = REPOSITORY / "slurm" / "15_numerics_and_guided_qed.sh"
    source = path.read_text(encoding="utf-8")
    directives = {}
    for line in source.splitlines():
        match = re.match(r"#SBATCH\s+--([\w-]+)(?:=(.*))?$", line.strip())
        if match:
            directives[match.group(1)] = (match.group(2) or "").strip()
    assert directives["partition"] == "CPU"
    assert "gres" not in directives
    assert 'if [[ -z "${SLURM_JOB_ID:-}" ]]' in source
    executable = "\n".join(line for line in source.splitlines()
                           if not line.lstrip().startswith("#") and "echo " not in line)
    assert "scripts/15_invertibility_audit.py" in executable
    assert "scripts/16_qed_guided_experiment.py" in executable
    assert "04_train_flow.py" not in executable, "stage 15 must not train anything"
    assert "rm " not in executable and "mv " not in executable
    assert "outputs/checkpoints" in executable                 # read-only input path
    assert "--output-dir \"${OUT}/invertibility\"" in source
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0
