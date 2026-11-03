"""Stage 19: the decoder-information audit.

Each instrument is tested on a constructed case where the answer is known, because the
audit's verdict is only worth as much as the measurements behind it:
  * the probe recovers a QED signal that exists and reports honest error when it does not;
  * the out-of-distribution statistic separates in-distribution from shifted inputs;
  * the paired intervention returns exactly zero change for a model that ignores QED, and
    pairs molecules valid under BOTH conditions;
  * the transfer control detects a decoder that destroys a QED difference it was given;
  * support counts state their own denominator.
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
from zinc_gnf.decoder_information import (
    EmbeddingQedProbe,
    decoder_qed_transfer,
    fit_embedding_probe,
    ood_statistics,
    paired_intervention,
    support_counts,
)
from zinc_gnf.preprocessing import ConditionScaler

REPOSITORY = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""),
                                                  REPOSITORY / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def real_setup(tmp_path):
    """A tiny dataset, the frozen autoencoder, and a joint flow trained the normal way."""
    import pickle

    from tests.test_flow_scripts_and_slurm import write_autoencoder
    from zinc_gnf.dataset import MolecularGraphDataset, load_training_splits
    from zinc_gnf.trainer_autoencoder import load_autoencoder

    data = write_dataset(tmp_path / "data")
    payload = pickle.loads((Path(data) / "splits.pkl").read_bytes())
    (Path(data) / "scaler.json").write_text(
        json.dumps(ConditionScaler.fit(payload["train"]).to_dict()))
    ae_path = write_autoencoder(tmp_path / "ae.pt")
    autoencoder, _ = load_autoencoder(ae_path, device="cpu")
    autoencoder.eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)
    train, val, _ = load_training_splits(data)
    return {"data": data, "ae_path": ae_path, "autoencoder": autoencoder,
            "train": MolecularGraphDataset(train), "val": MolecularGraphDataset(val)}


# ----------------------------------------------------------------- the probe ---
def test_the_probe_reports_its_own_held_out_error(real_setup):
    probe, report = fit_embedding_probe(
        real_setup["autoencoder"], real_setup["train"], real_setup["val"],
        train_indices=range(len(real_setup["train"])),
        val_indices=range(len(real_setup["val"])), device="cpu", epochs=3)
    assert isinstance(probe, EmbeddingQedProbe)
    assert report["validation"]["molecules"] == len(real_setup["val"])
    assert np.isfinite(report["validation"]["mae"])
    assert "EXTRAPOLATION" in report["note"]
    assert len(report["history"]) == 3


def test_the_probe_can_learn_a_signal_that_exists():
    """A sanity check on the read-out itself: a linear QED signal must be learnable."""
    torch.manual_seed(0)
    dimension, molecules = 8, 256
    direction = torch.randn(dimension)
    direction = direction / direction.norm()
    targets = torch.rand(molecules)
    x = torch.randn(molecules, 6, dimension) * 0.1 + (targets[:, None, None] * direction)
    mask = torch.ones(molecules, 6, dtype=torch.bool)
    probe = EmbeddingQedProbe(dimension)
    optimiser = torch.optim.Adam(probe.parameters(), lr=1e-2)
    for _ in range(300):
        loss = torch.nn.functional.mse_loss(probe(x, mask), targets)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    with torch.no_grad():
        error = float((probe(x, mask) - targets).abs().mean())
    assert error < 0.05, "the read-out cannot detect a signal it should see"


def test_the_probe_is_not_used_for_guidance_anywhere():
    source = (REPOSITORY / "zinc_gnf" / "decoder_information.py").read_text()
    assert "diagnostic" in source
    assert "guide" in source.lower()
    audit = (REPOSITORY / "scripts" / "20_decoder_information_audit.py").read_text()
    assert "never used as ground" in audit or "never used to guide" in audit
    assert "guide_latents" not in audit, "the audit must not perform guidance"


# ------------------------------------------------------------------- the OOD ---
def test_the_ood_statistic_separates_matched_from_shifted_inputs():
    torch.manual_seed(1)
    reference = torch.randn(512, 8)
    matched = ood_statistics(reference, torch.randn(256, 8))
    shifted = ood_statistics(reference, torch.randn(256, 8) * 4.0 + 6.0)
    assert matched["generated_over_real_mean"] == pytest.approx(1.0, abs=0.25)
    assert shifted["generated_over_real_mean"] > 2.0
    assert shifted["fraction_of_real_below_a_generated_point"] > \
        matched["fraction_of_real_below_a_generated_point"]
    assert "never fitted on" in shifted["note"]


# --------------------------------------------------- the paired intervention ---
def test_a_model_that_ignores_qed_shows_zero_change_at_every_stage(real_setup):
    """A neutral conditional flow must give exactly 0 displacement and 0 QED change."""
    flow = conditional_flow(features=("n", "qed"), dimension=64)
    flow.set_data_statistics(torch.zeros(64), torch.full((64,), 0.1))
    result = paired_intervention(flow, real_setup["autoencoder"], scaler=scaler(), n=6,
                                 qed_low=0.4, qed_high=0.8, count=8, temperature=1.0,
                                 device="cpu", decoders=("raw",))
    assert result["usable_molecules"] == 8
    assert result["embedding"]["per_node_norm"] == pytest.approx(0.0, abs=1e-6)
    assert result["bond_logits"]["per_pair_l2"] == pytest.approx(0.0, abs=1e-5)
    assert result["bond_logits"]["argmax_change_rate"] == pytest.approx(0.0)
    paired = result["decoders"]["raw"]["paired"]
    assert paired["qed_change_mean"] == pytest.approx(0.0, abs=1e-9) or \
        paired["paired_molecules"] == 0


def test_the_intervention_follows_every_stage_and_pairs_the_molecules(real_setup):
    flow = wake_up_conditioning(conditional_flow(features=("n", "qed"), dimension=64),
                                scale=0.2)
    flow.set_data_statistics(torch.zeros(64), torch.full((64,), 0.1))
    result = paired_intervention(flow, real_setup["autoencoder"], scaler=scaler(), n=6,
                                 qed_low=0.4, qed_high=0.8, count=8, temperature=1.0,
                                 device="cpu", decoders=("raw", "degree"))
    for key in ("latent", "embedding", "bond_logits", "structure", "decoders"):
        assert key in result, key
    assert result["embedding"]["per_node_norm"] > 0
    for name in ("raw", "degree"):
        block = result["decoders"][name]
        for key in ("low", "high", "aromatic_rings_change", "rings_change",
                    "unpaired_qed_change", "paired"):
            assert key in block, (name, key)
        paired = block["paired"]
        # paired molecules can never exceed those valid under either condition
        assert paired["paired_fraction"] <= min(paired["valid_low"], paired["valid_high"]) + 1e-9
        assert "qed_change_sem" in paired


def test_non_finite_samples_are_reported_and_nothing_is_measured(real_setup):
    flow = conditional_flow(features=("n", "qed"), dimension=64)
    flow.set_data_statistics(torch.zeros(64), torch.full((64,), 0.1))
    original = flow.inverse

    def broken(z, mask, condition=None):       # noqa: ANN001
        return original(z, mask, condition) * float("nan")

    flow.inverse = broken                      # type: ignore[method-assign]
    try:
        result = paired_intervention(flow, real_setup["autoencoder"], scaler=scaler(), n=6,
                                     qed_low=0.4, qed_high=0.8, count=4, temperature=1.0,
                                     device="cpu", decoders=("raw",))
    finally:
        flow.inverse = original                # type: ignore[method-assign]
    assert result["nonfinite_rate"] == pytest.approx(1.0)
    assert "note" in result and "non-finite" in result["note"]
    assert "embedding" not in result


# ------------------------------------------------------- the transfer control --
def test_the_transfer_control_runs_on_real_molecules(real_setup):
    dataset = real_setup["train"]
    counts = np.asarray(dataset.n_node)
    qed = np.asarray(dataset.qed, dtype=np.float64)
    n = int(np.bincount(counts).argmax())
    at_n = qed[counts == n]
    result = decoder_qed_transfer(real_setup["autoencoder"], dataset, n=n,
                                  qed_low=float(at_n.min()), qed_high=float(at_n.max()),
                                  window=0.05, max_molecules=8, device="cpu",
                                  decoders=("raw",))
    assert result["groups"]["low"]["molecules_available"] >= 1
    assert "verdict" in result
    if "preservation" in result:
        assert set(result["preservation"]) == {"raw"}
        assert "decoded_qed_difference" in result


def test_the_transfer_control_detects_a_decoder_that_destroys_the_difference(real_setup):
    """A decoder whose output ignores its input must show ~zero preservation."""

    class ConstantDecoder:
        config = real_setup["autoencoder"].config

        def __init__(self, inner):
            self.inner = inner
            self.training = False

        def encode(self, batch):
            return self.inner.encode(batch)

        def eval(self):
            return self

        def train(self, mode: bool = True):
            return self

        def decode(self, embeddings, batch, teacher_forcing=False):  # noqa: ANN001
            # the same output whatever the embedding: all QED information destroyed
            return self.inner.decode(torch.zeros_like(embeddings), batch,
                                     teacher_forcing=teacher_forcing)

    dataset = real_setup["train"]
    counts = np.asarray(dataset.n_node)
    qed = np.asarray(dataset.qed, dtype=np.float64)
    n = int(np.bincount(counts).argmax())
    at_n = qed[counts == n]
    if float(at_n.max() - at_n.min()) < 1e-6:
        pytest.skip("the fixture has no QED spread at this N")
    result = decoder_qed_transfer(ConstantDecoder(real_setup["autoencoder"]), dataset, n=n,
                                 qed_low=float(at_n.min()), qed_high=float(at_n.max()),
                                 window=0.5, max_molecules=8, device="cpu",
                                 decoders=("raw",))
    if "preservation" in result:
        value = result["preservation"]["raw"]
        assert not np.isfinite(value) or abs(value) < 0.2
        assert "loses most" in result["verdict"] or "preserves" not in result["verdict"]


# ------------------------------------------------------------------ support ----
def test_support_counts_state_their_denominator(real_setup):
    dataset = real_setup["train"]
    rows = support_counts(dataset, n_values=[int(np.bincount(np.asarray(dataset.n_node))
                                                  .argmax())],
                          qed_values=[0.4, 0.8])
    row = rows[0]
    assert row["molecules_at_this_n"] >= 1
    assert set(row["by_qed"]) == {"0.4", "0.8"}
    for value in row["by_qed"].values():
        assert value["molecules_within_window"] <= row["molecules_at_this_n"]
    assert row["minimum_over_qed_values"] == min(
        value["molecules_within_window"] for value in row["by_qed"].values())


# ------------------------------------------------- the audit script, end to end -
def test_the_audit_runs_and_names_a_hypothesis(real_setup, tmp_path):
    data, ae_path = real_setup["data"], real_setup["ae_path"]
    run = tmp_path / "joint"
    assert load_script("04_train_flow.py").main([
        "--config", str(REPOSITORY / "configs/zinc250k_flow_nqed.yaml"),
        "--data-dir", str(data), "--ae-checkpoint", str(ae_path),
        "--output-dir", str(run), "--device", "cpu", "--seed", "1", "--batch-size", "4",
        "--max-steps", "4", "--num-blocks", "1", "--statistics-molecules", "20",
        "--val-molecules", "8", "--eval-every", "2", "--save-every", "2",
        "--log-every", "2", "--warmup-steps", "0"]) == 0

    output = tmp_path / "audit"
    assert load_script("20_decoder_information_audit.py").main([
        "--flow-checkpoint", str(run / "last.pt"), "--ae-checkpoint", str(ae_path),
        "--data-dir", str(data), "--output-dir", str(output), "--target-n", "9",
        "--qed-pair", "0.4,0.8", "--count", "8", "--temperature", "0.2",
        "--probe-train-molecules", "20", "--probe-val-molecules", "8",
        "--probe-epochs", "2", "--transfer-molecules", "8", "--device", "cpu"]) == 0
    payload = json.loads((output / "decoder_information.json").read_text())
    assert payload["verdict"]["hypothesis"] in {"H1-decoder", "H1-decoding", "H2-direction",
                                                "H3-constraints"}
    assert "probe_held_out_mae" in payload["verdict"]
    assert payload["probe"]["validation"]["molecules"] > 0
    assert payload["support"] and "minimum_over_qed_values" in payload["support"][0]
    assert "minimum" in payload["support_definition"]
    assert payload["decoder_qed_transfer"]
    assert payload["interventions"]
    assert all("records" not in row for row in payload["interventions"])
    text = (output / "decoder_information.txt").read_text()
    assert "VERDICT" in text
    assert "does the decoder preserve a QED difference it is GIVEN" in text
    assert "held-out MAE" in text
    assert "minimum" in text


def test_the_audit_refuses_a_flow_without_qed(real_setup, tmp_path):
    from tests.test_conditional_flow import run_training as run_n_only

    data, ae_path = real_setup["data"], real_setup["ae_path"]
    n_only = tmp_path / "n_only"
    assert run_n_only(data, ae_path, n_only) == 0
    assert load_script("20_decoder_information_audit.py").main([
        "--flow-checkpoint", str(n_only / "last.pt"), "--ae-checkpoint", str(ae_path),
        "--data-dir", str(data), "--output-dir", str(tmp_path / "o"),
        "--device", "cpu"]) == 4


def test_the_audit_validates_its_arguments(tmp_path):
    script = load_script("20_decoder_information_audit.py")
    assert script.main(["--flow-checkpoint", str(tmp_path / "ghost.pt"),
                        "--output-dir", str(tmp_path / "o"), "--qed-pair", "0.8,0.4"]) == 4
    assert script.main(["--flow-checkpoint", str(tmp_path / "ghost.pt"),
                        "--output-dir", str(tmp_path / "o2")]) == 2


# ------------------------------------------------------------------- SLURM -----
def test_the_stage19_job_is_cpu_only_read_only_and_estimates_runtime():
    path = REPOSITORY / "slurm" / "19_decoder_information.sh"
    source = path.read_text(encoding="utf-8")
    directives = {}
    for line in source.splitlines():
        match = re.match(r"#SBATCH\s+--([\w-]+)(?:=(.*))?$", line.strip())
        if match:
            directives[match.group(1)] = (match.group(2) or "").strip()
    assert directives["partition"] == "CPU"
    assert "gres" not in directives
    assert directives["time"] and directives["mem"]
    assert 'if [[ -z "${SLURM_JOB_ID:-}" ]]' in source
    assert "Runtime estimate" in source, "the job must state its expected runtime"
    executable = "\n".join(line for line in source.splitlines()
                           if not line.lstrip().startswith("#") and "echo " not in line)
    assert "scripts/20_decoder_information_audit.py" in executable
    assert "04_train_flow.py" not in executable, "stage 19 must not train the flow"
    assert "--device cpu" in executable
    assert not re.search(r"(?m)^\s*(rm|mv)\s", executable)
    assert "outputs/metrics/decoder_information" in source
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0
