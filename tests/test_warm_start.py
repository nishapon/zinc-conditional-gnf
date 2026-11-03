"""Stage 17: warm-starting a conditional flow from the unconditional checkpoint.

The experiment is only meaningful if the transfer is exact, so that is what these tests
establish - with reproducible initialisation, and by construction rather than by
assertion of intent:
  * the two state dictionaries differ in exactly the four expected groups;
  * a warm-started conditional model is BIT-EXACTLY the unconditional model, for every
    condition, on latents, log-determinant, log-likelihood, inverse and samples;
  * a partial or mismatched transfer is refused instead of silently half-loading;
  * freezing the backbone leaves only the adapters trainable, and training cannot move a
    frozen tensor;
  * the evaluation-time ablation restores every weight it touched.
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
from zinc_gnf.flow import FlowConfig, NodeFlow, node_mask_from_counts
from zinc_gnf.preprocessing import ConditionScaler
from zinc_gnf.warm_start import (
    CONDITIONAL_ONLY,
    adapter_parameters,
    backbone_parameters,
    equivalence_report,
    freeze_backbone,
    neutralised,
    transfer_report,
    warm_start,
)

REPOSITORY = Path(__file__).resolve().parents[1]
DIMENSION = 8


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""),
                                                  REPOSITORY / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def base_config(**overrides) -> FlowConfig:
    values = dict(node_embedding_dim=DIMENSION, num_blocks=2, hidden_dim=16, attention_dim=16,
                  num_heads=2, node_autoregressive=True, condition_embedding_dim=8,
                  condition_hidden_dim=8)
    values.update(overrides)
    return FlowConfig(**values)


def trained_unconditional(seed: int = 0) -> NodeFlow:
    """An unconditional flow with NON-trivial weights, like a real checkpoint."""
    flow = NodeFlow(base_config(condition_features=()), init_seed=seed)
    generator = torch.Generator().manual_seed(seed + 11)
    with torch.no_grad():
        for name, parameter in flow.named_parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.2)
        # a realistic learned prior: close to, but not exactly, N(0, 1)
        flow.prior.mu.copy_(torch.full((DIMENSION,), 0.03))
        flow.prior.pre_sigma.copy_(torch.full((DIMENSION,), 0.5))
        flow.set_data_statistics(torch.randn(DIMENSION, generator=generator),
                                 torch.rand(DIMENSION, generator=generator) + 0.4)
    flow.eval()
    return flow


def fresh_conditional(seed: int = 3) -> NodeFlow:
    # ConditionEmbedding and prior.hidden are initialised through the GLOBAL torch RNG
    # (nn.init without a generator), so reproducible construction needs it seeded too -
    # which is exactly what scripts/18_warm_start_audit.py does via --seed.
    torch.manual_seed(seed)
    flow = NodeFlow(base_config(condition_features=("n", "qed")), init_seed=seed)
    flow.eval()
    return flow


# ----------------------------------------------------- 1. compatibility --------
def test_the_state_dicts_differ_in_exactly_the_four_expected_groups():
    source = trained_unconditional().state_dict()
    report = transfer_report(source, fresh_conditional())
    assert report["compatible"], report["verdict"]
    assert not report["shape_mismatch"] and not report["unexpected"]
    conditional = [entry["name"] for entry in report["conditional_only"]]
    assert any(".film." in name for name in conditional)
    assert any(name.startswith("condition_embedding.") for name in conditional)
    assert any("prior.mu_head." in name for name in conditional)
    assert any("prior.sigma_head." in name for name in conditional)
    assert {entry["name"] for entry in report["unconditional_only"]} == \
        {"prior.mu", "prior.pre_sigma"}
    # and the backbone really is the bulk of the model
    assert report["transferable_count"] > len(conditional)
    for entry in report["transferable"]:
        assert ".film." not in entry["name"]
    assert any(entry["name"] == "data_mean" for entry in report["transferable"])
    assert any(entry["name"] == "data_std" for entry in report["transferable"])


def test_an_architecture_mismatch_is_refused():
    source = trained_unconditional().state_dict()
    wider = NodeFlow(base_config(condition_features=("n", "qed"), hidden_dim=32), init_seed=3)
    report = transfer_report(source, wider)
    assert not report["compatible"]
    assert report["shape_mismatch"]
    with pytest.raises(ValueError, match="NOT COMPATIBLE"):
        warm_start(wider, source)


def test_a_conditional_source_is_refused():
    source = fresh_conditional(seed=5).state_dict()
    with pytest.raises(ValueError, match="not an unconditional NodePrior"):
        warm_start(fresh_conditional(), source)


def test_a_partial_source_is_refused_rather_than_half_loaded():
    """A missing backbone tensor is caught by the compatibility check, before any load."""
    source = trained_unconditional().state_dict()
    del source["blocks.0.act_norm.log_scale"]
    report = transfer_report(source, fresh_conditional())
    assert not report["compatible"]
    assert any(entry["name"] == "blocks.0.act_norm.log_scale" for entry in report["unexpected"])
    with pytest.raises(ValueError, match="NOT COMPATIBLE"):
        warm_start(fresh_conditional(), source)


# ------------------------------------------- 2. functional equivalence ---------
def test_the_warm_started_model_is_bit_exactly_the_unconditional_one():
    unconditional = trained_unconditional()
    target = fresh_conditional()
    transfer = warm_start(target, unconditional.state_dict())
    assert transfer["neutral"] and transfer["actions"]

    mask = node_mask_from_counts([5, 7, 4])
    x = torch.randn(3, mask.shape[1], DIMENSION) * mask.unsqueeze(-1)
    conditions = torch.tensor([[-2.0, -2.0], [0.0, 0.0], [1.5, 2.5], [3.0, -1.0]])
    report = equivalence_report(unconditional, target, x, mask, conditions)
    assert report["bit_exact"], report["worst"]
    assert report["equivalent"]
    assert len(report["rows"]) == 4
    for row in report["rows"]:
        for key, value in row.items():
            if key != "condition":
                assert value == 0.0, (key, value)


def test_the_prior_reproduces_the_unconditional_one_for_every_condition():
    unconditional = trained_unconditional()
    target = fresh_conditional()
    warm_start(target, unconditional.state_dict())
    reference = unconditional.prior.parameters_for(1, None, "cpu")
    for condition in (torch.zeros(1, 2), torch.tensor([[5.0, -5.0]]), torch.tensor([[-3.0, 2.0]])):
        embedding = target.conditioning(condition, None, 1, "cpu")
        mu, sigma = target.prior.parameters_for(1, embedding, "cpu")
        assert torch.equal(mu, reference[0])
        assert torch.equal(sigma, reference[1])


def test_equivalence_detects_a_transfer_that_is_not_neutral():
    """With neutral=False the conditional pathways keep their random initialisation, so
    the model is NOT the unconditional one - the check must say so."""
    unconditional = trained_unconditional()
    target = fresh_conditional()
    warm_start(target, unconditional.state_dict(), neutral=False)
    mask = node_mask_from_counts([6])
    x = torch.randn(1, mask.shape[1], DIMENSION) * mask.unsqueeze(-1)
    report = equivalence_report(unconditional, target, x, mask, torch.tensor([[1.0, 1.0]]))
    assert not report["bit_exact"]
    assert "NOT EQUIVALENT" in report["verdict"] or not report["equivalent"]


def test_warm_start_is_deterministic():
    source = trained_unconditional().state_dict()
    first, second = fresh_conditional(seed=3), fresh_conditional(seed=3)
    warm_start(first, source)
    warm_start(second, source)
    for (name, a), (_, b) in zip(first.state_dict().items(), second.state_dict().items()):
        assert torch.equal(a, b), name


# ------------------------------------------------- 3. the parameter split ------
def test_freezing_the_backbone_leaves_only_the_adapters_trainable():
    target = fresh_conditional()
    warm_start(target, trained_unconditional().state_dict())
    split = freeze_backbone(target)
    assert split["trainable_parameters"] > 0
    assert split["frozen_parameters"] > split["trainable_parameters"]
    assert 0.0 < split["trainable_fraction"] < 0.5
    for name, parameter in target.named_parameters():
        expected = any(pattern in name for pattern in CONDITIONAL_ONLY)
        assert parameter.requires_grad is expected, name
    assert {name for name, _ in adapter_parameters(target)}.isdisjoint(
        {name for name, _ in backbone_parameters(target)})


def test_a_frozen_backbone_cannot_be_moved_by_a_gradient_step():
    target = fresh_conditional()
    warm_start(target, trained_unconditional().state_dict())
    freeze_backbone(target)
    before = {name: parameter.detach().clone()
              for name, parameter in target.named_parameters()}
    mask = node_mask_from_counts([6, 6])
    x = torch.randn(2, mask.shape[1], DIMENSION) * mask.unsqueeze(-1)
    embedding = target.conditioning(torch.zeros(2, 2), None, 2, "cpu")
    loss = target.negative_log_likelihood(x, mask, embedding)["loss"]
    optimiser = torch.optim.Adam([p for p in target.parameters() if p.requires_grad], lr=0.1)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    moved = [name for name, parameter in target.named_parameters()
             if not torch.equal(parameter.detach(), before[name])]
    assert moved, "the adapters must be able to move"
    for name in moved:
        assert any(pattern in name for pattern in CONDITIONAL_ONLY), name


# ------------------------------------------------------- 4. the ablation -------
def test_neutralising_film_restores_the_unconditional_coupling_behaviour():
    target = fresh_conditional()
    warm_start(target, trained_unconditional().state_dict())
    # give FiLM real weights, as training would
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, parameter in target.named_parameters():
            if ".film." in name:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.4)
    mask = node_mask_from_counts([6])
    x = torch.randn(1, mask.shape[1], DIMENSION) * mask.unsqueeze(-1)
    embedding = target.conditioning(torch.tensor([[1.0, -1.0]]), None, 1, "cpu")
    active = target(x, mask, embedding)[0]
    with neutralised(target, film=True):
        off = target(x, mask, embedding)[0]
    assert not torch.allclose(active, off)
    # ... and everything is restored afterwards
    assert torch.equal(target(x, mask, embedding)[0], active)


def test_neutralising_the_prior_removes_its_condition_dependence_only():
    target = fresh_conditional()
    warm_start(target, trained_unconditional().state_dict())
    generator = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for name, parameter in target.named_parameters():
            if "prior.mu_head" in name or "prior.sigma_head" in name:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.3)
    low = target.conditioning(torch.tensor([[-2.0, -2.0]]), None, 1, "cpu")
    high = target.conditioning(torch.tensor([[2.0, 2.0]]), None, 1, "cpu")
    assert not torch.allclose(target.prior.parameters_for(1, low, "cpu")[0],
                              target.prior.parameters_for(1, high, "cpu")[0])
    reference = torch.tensor([[0.0, 0.0]])
    with neutralised(target, prior=True, reference_condition=reference):
        assert torch.allclose(target.prior.parameters_for(1, low, "cpu")[0],
                              target.prior.parameters_for(1, high, "cpu")[0])
        assert torch.allclose(target.prior.parameters_for(1, low, "cpu")[1],
                              target.prior.parameters_for(1, high, "cpu")[1])
    assert not torch.allclose(target.prior.parameters_for(1, low, "cpu")[0],
                              target.prior.parameters_for(1, high, "cpu")[0])


def test_the_ablation_restores_weights_even_on_an_exception():
    target = fresh_conditional()
    warm_start(target, trained_unconditional().state_dict())
    before = {name: p.detach().clone() for name, p in target.named_parameters()}
    with pytest.raises(RuntimeError):
        with neutralised(target, film=True, prior=True,
                         reference_condition=torch.zeros(1, 2)):
            raise RuntimeError("boom")
    for name, parameter in target.named_parameters():
        assert torch.equal(parameter.detach(), before[name]), name


# --------------------------------------------------- the audit, end to end -----
@pytest.fixture
def two_checkpoints(tmp_path):
    """A tiny unconditional run and a tiny joint run, trained the normal way."""
    import pickle

    from tests.test_flow_scripts_and_slurm import write_autoencoder

    data = write_dataset(tmp_path / "data")
    payload = pickle.loads((Path(data) / "splits.pkl").read_bytes())
    (Path(data) / "scaler.json").write_text(
        json.dumps(ConditionScaler.fit(payload["train"]).to_dict()))
    autoencoder = write_autoencoder(tmp_path / "ae.pt")
    script = load_script("04_train_flow.py")
    common = ["--data-dir", str(data), "--ae-checkpoint", str(autoencoder),
              "--device", "cpu", "--seed", "1", "--batch-size", "4", "--max-steps", "4",
              "--num-blocks", "1", "--statistics-molecules", "20", "--val-molecules", "8",
              "--eval-every", "2", "--save-every", "2", "--log-every", "2",
              "--warmup-steps", "0"]
    uncond = tmp_path / "uncond"
    assert script.main(["--config", str(REPOSITORY / "configs/zinc250k_flow_uncond.yaml"),
                        "--output-dir", str(uncond), "--node-autoregressive", *common]) == 0
    cond = tmp_path / "cond"
    assert script.main(["--config", str(REPOSITORY / "configs/zinc250k_flow_nqed.yaml"),
                        "--output-dir", str(cond), *common]) == 0
    return data, autoencoder, uncond / "last.pt", cond / "last.pt"


def test_the_audit_script_proves_equivalence_and_writes_a_new_checkpoint(two_checkpoints,
                                                                        tmp_path):
    data, autoencoder, uncond, cond = two_checkpoints
    script = load_script("18_warm_start_audit.py")
    output = tmp_path / "audit"
    assert script.main(["--unconditional-checkpoint", str(uncond),
                        "--conditional-checkpoint", str(cond),
                        "--ae-checkpoint", str(autoencoder), "--data-dir", str(data),
                        "--output-dir", str(output), "--molecules", "8",
                        "--ablation-molecules", "4", "--ablation-n", "9",
                        "--ablation-qed", "0.6", "--device", "cpu"]) == 0
    payload = json.loads((output / "warm_start.json").read_text())
    assert payload["transfer"]["compatible"] is True
    assert payload["equivalence"]["bit_exact"] is True
    assert payload["parameter_split"]["trainable_parameters"] > 0
    assert [row["variant"] for row in payload["component_ablation"]] == \
        ["as_trained", "film_neutralised", "prior_condition_frozen", "both_neutralised"]
    for row in payload["component_ablation"]:
        for key in ("raw_validity", "aromatic_rings_mean", "bonds_per_atom",
                    "isolated_atom_rate", "fallback_rate"):
            assert key in row, key
    produced = output / "warm_start.pt"
    assert produced.exists()
    text = (output / "warm_start.txt").read_text()
    assert "equivalence at initialisation" in text
    assert "bit-exact" in text

    # the new checkpoint loads as a conditional flow and carries its provenance
    from zinc_gnf.trainer_flow import load_flow

    restored, probe = load_flow(produced)
    assert restored.config.condition_features == ("n", "qed")
    assert probe["warm_start"]["source"] == str(uncond)
    assert probe["warm_start"]["equivalence"]["bit_exact"] is True
    # and a second run refuses to overwrite it
    assert script.main(["--unconditional-checkpoint", str(uncond),
                        "--ae-checkpoint", str(autoencoder), "--data-dir", str(data),
                        "--output-dir", str(output), "--molecules", "8",
                        "--ablation-molecules", "0", "--device", "cpu"]) == 4


def test_the_audit_refuses_a_conditional_source(two_checkpoints, tmp_path):
    data, autoencoder, _, cond = two_checkpoints
    script = load_script("18_warm_start_audit.py")
    assert script.main(["--unconditional-checkpoint", str(cond),
                        "--ae-checkpoint", str(autoencoder), "--data-dir", str(data),
                        "--output-dir", str(tmp_path / "a2"), "--device", "cpu"]) == 4


def test_stage_a_trains_only_adapters_end_to_end(two_checkpoints, tmp_path):
    data, autoencoder, uncond, _ = two_checkpoints
    audit = load_script("18_warm_start_audit.py")
    output = tmp_path / "audit"
    assert audit.main(["--unconditional-checkpoint", str(uncond),
                       "--ae-checkpoint", str(autoencoder), "--data-dir", str(data),
                       "--output-dir", str(output), "--molecules", "8",
                       "--ablation-molecules", "0", "--device", "cpu"]) == 0

    finetune = load_script("17_structure_finetune.py")
    run = tmp_path / "stageA"
    assert finetune.main(["--flow-checkpoint", str(output / "warm_start.pt"),
                          "--ae-checkpoint", str(autoencoder), "--data-dir", str(data),
                          "--output-dir", str(run), "--lambda-structure", "0",
                          "--freeze-backbone", "--max-steps", "2", "--batch-size", "4",
                          "--val-molecules", "8", "--eval-every", "1", "--save-every", "1",
                          "--log-every", "1", "--target-molecules", "20",
                          "--device", "cpu", "--seed", "1"]) == 0
    configuration = json.loads((run / "config.json").read_text())
    assert configuration["freeze_backbone"] is True
    assert configuration["parameter_split"]["trainable_parameters"] > 0
    assert configuration["warm_start"]["equivalence"]["bit_exact"] is True
    assert "BACKBONE FROZEN" in (run / "train.log").read_text()

    # the backbone of the fine-tuned model must equal the warm start's backbone exactly
    from zinc_gnf.trainer_flow import load_flow

    before, _ = load_flow(output / "warm_start.pt")
    after, _ = load_flow(run / "last.pt")
    before_state, after_state = before.state_dict(), after.state_dict()
    for name in before_state:
        if any(pattern in name for pattern in CONDITIONAL_ONLY):
            continue
        assert torch.equal(before_state[name], after_state[name]), name


# ------------------------------------------------------------------ SLURM ------
def test_the_warm_start_job_keeps_every_old_experiment():
    path = REPOSITORY / "slurm" / "17_warm_start_experiment.sh"
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
    assert "scripts/18_warm_start_audit.py" in executable
    assert "scripts/17_structure_finetune.py" in executable
    assert "scripts/13_generate_nqed.py" in executable
    assert "scripts/05_generate.py" in executable, "the QED-blind baseline must be evaluated"
    assert "--freeze-backbone" in executable
    assert "--lambda-structure 0" in executable
    assert "structure_objective" not in executable, "the failed structural term must not return"
    assert not re.search(r"(?m)^\s*(rm|mv)\s", executable), "no existing output may be removed"
    # new output locations only
    for path_fragment in ("outputs/checkpoints/warm_start", "outputs/samples/warm_start",
                          "outputs/metrics/warm_start"):
        assert path_fragment in source
    assert "GO / NO-GO" in source and "NO-GO" in source
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0
