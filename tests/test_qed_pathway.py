"""Stage 20: the QED pathway audit.

Every instrument is tested on a constructed case where the answer is known, and so is
every CLAIM the audit's reasoning rests on:

  * `pooled_features` is bit-compatible with `EmbeddingQedProbe.forward`, so carrying the
    stage-19 probe into stage 20 is an identity and not a re-implementation;
  * the ridge read-out's paired prediction difference is exactly `w . (f_high - f_low)`;
  * a paired-mean bootstrap interval is narrower than a single-prediction error - the
    methodological point that invalidates "MAE as a detection threshold";
  * the measured GAIN recovers 1.0 for a perfect read-out and the compression factor for a
    compressed one, so an observed change can be converted into an implied real one;
  * a change buried in the null is reported `not_resolved`, never as absence of signal;
  * discriminability separates two clouds whose Mahalanobis means MATCH - the case stage
    19's aggregate statistic could not see;
  * the degree decoder returns exactly the constrained decoder's bonds on a fallback,
    which is the identity the whole attenuation decomposition rests on;
  * the strata contributions sum to the measured gap;
  * E2 and E3 can both be `supported` at once, which the stage-19 ladder could not express;
  * an inadequate instrument yields `needs_further_test`, never evidence about the model.
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
from zinc_gnf.decoder_contrast import (
    decode_with_graphs,
    decoder_contrast,
    summarise_attenuation,
    transfer_on_split,
)
from zinc_gnf.decoder_information import EmbeddingQedProbe
from zinc_gnf.decoding_charge_aware import (
    degree_constrained_bond_decode,
    valence_aware_bond_decode_typed,
)
from zinc_gnf.preprocessing import ConditionScaler
from zinc_gnf.qed_evidence import (
    assess_evidence,
    format_report,
    recommend_intervention,
)
from zinc_gnf.qed_probes import (
    MlpProbeReadout,
    WithinNRidgeProbe,
    distribution_discriminability,
    interpret_generated_change,
    null_contrast,
    paired_bootstrap_ci,
    paired_difference_ci,
    pooled_features,
    within_n_reference_contrast,
)

REPOSITORY = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""),
                                                  REPOSITORY / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def real_setup(tmp_path):
    """A tiny dataset plus the frozen autoencoder, exactly as the stage-19 tests build it."""
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


# ------------------------------------------------------- the pooling identity ---
def test_pooled_features_is_the_stage19_probe_pooling():
    """If this drifts, carrying the stage-19 probe into stage 20 stops being an identity."""
    torch.manual_seed(0)
    dimension = 12
    x = torch.randn(7, 9, dimension)
    mask = torch.zeros(7, 9, dtype=torch.bool)
    for row, count in enumerate([9, 8, 6, 5, 4, 3, 2]):
        mask[row, :count] = True
    probe = EmbeddingQedProbe(dimension)
    with torch.no_grad():
        through_forward = probe(x, mask)
        through_pooling = probe.net(pooled_features(x, mask)).squeeze(-1)
    assert torch.allclose(through_forward, through_pooling, atol=0.0, rtol=0.0)


def test_the_mlp_readout_reproduces_the_probe_exactly():
    torch.manual_seed(1)
    dimension = 8
    x = torch.randn(16, 5, dimension)
    mask = torch.ones(16, 5, dtype=torch.bool)
    probe = EmbeddingQedProbe(dimension)
    readout = MlpProbeReadout(probe)
    features = pooled_features(x, mask).to(torch.float64).numpy()
    with torch.no_grad():
        expected = probe(x, mask).numpy()
    assert np.allclose(readout.contrast(features), expected, atol=1e-6)


# ------------------------------------------------------------- the ridge probe ---
def test_the_ridge_readout_learns_a_within_n_signal_and_ignores_the_n_offset():
    """QED = a per-N offset plus a linear within-N term. Only the second is learnable."""
    rng = np.random.default_rng(0)
    dimension, molecules = 6, 400
    direction = rng.normal(size=dimension)
    direction /= np.linalg.norm(direction)
    n = rng.choice([10, 20, 30], size=molecules)
    # the within-N term must dominate the feature noise on its own axis, or no read-out
    # could recover it and the test would be measuring the fixture rather than the code
    within = rng.normal(scale=0.2, size=molecules)
    features = rng.normal(scale=0.05, size=(molecules, dimension)) + within[:, None] * direction
    offsets = {10: 0.3, 20: 0.5, 30: 0.7}
    qed = np.asarray([offsets[int(v)] for v in n]) + within
    probe = WithinNRidgeProbe.fit(features, qed, n, seed=0)
    predicted = probe.contrast(features)
    assert float(np.corrcoef(predicted, within)[0, 1]) > 0.9
    # the per-N offsets must be recovered from TRAIN, not learned from the features
    for value, expected in offsets.items():
        assert probe.offsets[value] == pytest.approx(expected, abs=0.05)
    assert probe.alpha in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    assert probe.selection, "alpha must be chosen on an inner split of the training data"


def test_the_ridge_paired_difference_is_exactly_a_linear_functional():
    """The claim in the docstring: a paired prediction difference is w . (f_hi - f_lo)."""
    rng = np.random.default_rng(2)
    features = rng.normal(size=(60, 5))
    qed = rng.uniform(0.2, 0.9, size=60)
    n = rng.choice([8, 9], size=60)
    probe = WithinNRidgeProbe.fit(features, qed, n, seed=1)
    low, high = rng.normal(size=(20, 5)), rng.normal(size=(20, 5))
    observed = probe.contrast(high) - probe.contrast(low)
    expected = ((high - low) / probe.feature_std) @ probe.weights
    assert np.allclose(observed, expected, atol=1e-12)


# ---------------------------------------------------------------- bootstraps ---
def test_a_paired_mean_is_resolved_far_below_the_single_prediction_error():
    """The stage-19 methodological error, demonstrated.

    Per-molecule noise of 0.09 - the stage-19 held-out MAE - with a true paired shift of
    only 0.02 still yields an interval that excludes zero over 400 pairs. A single
    prediction's error is therefore not the detection threshold for a paired mean.
    """
    rng = np.random.default_rng(3)
    values = 0.02 + rng.normal(scale=0.09, size=400)
    block = paired_bootstrap_ci(values, seed=5)
    assert block["excludes_zero"], "a real paired shift well below the MAE must be resolved"
    assert block["ci_high"] - block["ci_low"] < 0.09
    assert block["mean"] == pytest.approx(0.02, abs=0.015)


def test_the_bootstrap_reports_no_signal_when_there_is_none():
    rng = np.random.default_rng(4)
    block = paired_bootstrap_ci(rng.normal(scale=0.09, size=400), seed=5)
    assert not block["excludes_zero"]


def test_the_paired_difference_keeps_the_pairing():
    """Two decoders measured on the same molecules: the correlation must not be discarded."""
    rng = np.random.default_rng(6)
    shared = rng.normal(scale=0.5, size=300)
    first = shared + 0.05
    second = shared
    block = paired_difference_ci(first, second, seed=7)
    assert block["difference"] == pytest.approx(0.05, abs=1e-9)
    assert block["excludes_zero"], "a constant offset on paired data must be detected"
    assert block["ci_high"] - block["ci_low"] < 0.01


def test_paired_arrays_must_match_in_length():
    with pytest.raises(ValueError):
        paired_difference_ci([1.0, 2.0], [1.0])


# ------------------------------------------------------------ gain and null ---
def _synthetic_population(seed: int = 8, molecules: int = 600):
    rng = np.random.default_rng(seed)
    direction = np.zeros(4)
    direction[0] = 1.0
    qed = rng.uniform(0.3, 0.9, size=molecules)
    features = rng.normal(scale=0.05, size=(molecules, 4)) + qed[:, None] * direction
    n = np.full(molecules, 15, dtype=np.int64)
    return features, qed, n


class _PerfectReadout:
    """Reads QED off the feature it was planted in. Gain must come out at 1.0."""

    name = "perfect"

    def contrast(self, features):
        return np.asarray(features, dtype=np.float64)[:, 0]

    def absolute(self, features, n=None):
        return self.contrast(features)


class _CompressedReadout(_PerfectReadout):
    """Right order, a quarter of the range: gain must come out at 0.25."""

    name = "compressed"

    def contrast(self, features):
        return 0.25 * np.asarray(features, dtype=np.float64)[:, 0]


class _BlindReadout(_PerfectReadout):
    name = "blind"

    def contrast(self, features):
        return np.zeros(np.asarray(features).shape[0], dtype=np.float64)


def test_the_gain_recovers_one_for_a_perfect_readout():
    features, qed, n = _synthetic_population()
    block = within_n_reference_contrast(_PerfectReadout(), features, qed, n, target_n=15,
                                       qed_low=0.4, qed_high=0.8, window=0.05,
                                       resamples=400, seed=9)
    assert block["pairs"] > 10
    assert block["gain"] == pytest.approx(1.0, abs=0.1)
    assert block["gain_excludes_zero"]


def test_the_gain_recovers_the_compression_factor():
    """This is what turns an observed +0.05 into an implied real difference."""
    features, qed, n = _synthetic_population()
    block = within_n_reference_contrast(_CompressedReadout(), features, qed, n, target_n=15,
                                        qed_low=0.4, qed_high=0.8, window=0.05,
                                        resamples=400, seed=9)
    assert block["gain"] == pytest.approx(0.25, abs=0.05)


def test_a_blind_readout_has_no_gain_and_cannot_be_used_as_evidence():
    features, qed, n = _synthetic_population()
    block = within_n_reference_contrast(_BlindReadout(), features, qed, n, target_n=15,
                                        qed_low=0.4, qed_high=0.8, window=0.05,
                                        resamples=400, seed=9)
    assert block["gain"] == pytest.approx(0.0, abs=1e-9)
    assert not block.get("gain_excludes_zero", False)


def test_the_null_is_centred_on_zero_for_same_band_pairs():
    features, qed, n = _synthetic_population()
    block = null_contrast(_PerfectReadout(), features, qed, n, target_n=15, qed_centre=0.8,
                          window=0.05, resamples=400, seed=9)
    assert block["pairs"] > 5
    assert abs(block["predicted_difference"]["mean"]) < 0.05
    assert np.isfinite(block["null_sd"])


def test_a_change_inside_the_null_is_not_resolved_and_says_so():
    features, qed, n = _synthetic_population()
    reference = within_n_reference_contrast(_CompressedReadout(), features, qed, n,
                                            target_n=15, qed_low=0.4, qed_high=0.8,
                                            window=0.05, resamples=400, seed=9)
    null = null_contrast(_CompressedReadout(), features, qed, n, target_n=15,
                         qed_centre=0.8, window=0.05, resamples=400, seed=9)
    rng = np.random.default_rng(11)
    buried = rng.normal(scale=max(null["null_sd"], 1e-6), size=12)
    reading = interpret_generated_change(buried, null=null, reference=reference,
                                         requested=0.4, resamples=400, seed=9)
    assert reading["status"] == "not_resolved"
    assert "not evidence that the embeddings carry no QED" in reading["note"]


def test_a_resolved_change_is_divided_by_the_gain_to_imply_a_real_difference():
    features, qed, n = _synthetic_population()
    reference = within_n_reference_contrast(_CompressedReadout(), features, qed, n,
                                            target_n=15, qed_low=0.4, qed_high=0.8,
                                            window=0.05, resamples=400, seed=9)
    null = null_contrast(_CompressedReadout(), features, qed, n, target_n=15,
                         qed_centre=0.8, window=0.05, resamples=400, seed=9)
    observed = np.full(300, 0.05)
    reading = interpret_generated_change(observed, null=null, reference=reference,
                                         requested=0.4, resamples=400, seed=9)
    assert reading["status"] == "resolved"
    # gain ~0.25, so +0.05 reported implies ~+0.20 of real QED, i.e. half the request
    assert reading["implied_real_qed_difference"] == pytest.approx(0.2, abs=0.05)
    assert reading["implied_fraction_of_request"] == pytest.approx(0.5, abs=0.15)


# --------------------------------------------------------- distribution shift ---
def test_discriminability_separates_clouds_whose_mahalanobis_means_match():
    """The case stage 19's aggregate statistic could not see.

    A tight cluster placed at the real cloud's own mean Mahalanobis radius has a
    generated/real ratio near 1 while being almost perfectly separable.
    """
    from zinc_gnf.decoder_information import ood_statistics

    rng = np.random.default_rng(12)
    dimension = 8
    real = rng.normal(size=(600, dimension))
    radius = float(np.linalg.norm(real, axis=1).mean())
    centre = np.zeros(dimension)
    centre[0] = radius
    generated = rng.normal(scale=0.05, size=(600, dimension)) + centre
    statistics = ood_statistics(torch.from_numpy(real), torch.from_numpy(generated))
    assert statistics["generated_over_real_mean"] == pytest.approx(1.0, abs=0.2)
    block = distribution_discriminability(real, generated, seed=13)
    assert block["held_out_auc"] > 0.9, "a separable shift must be visible as an AUC"


def test_discriminability_is_near_chance_for_the_same_distribution():
    rng = np.random.default_rng(14)
    block = distribution_discriminability(rng.normal(size=(500, 6)),
                                          rng.normal(size=(500, 6)), seed=15)
    assert abs(block["held_out_auc"] - 0.5) < 0.12


# ------------------------------------------------- THE identity the audit needs ---
def test_the_degree_decoder_falls_back_to_exactly_the_constrained_decoder():
    """The whole attenuation decomposition rests on this being an identity.

    On every fallback path the degree decoder returns
    `valence_aware_bond_decode_typed(...)`, so on a molecule where it fell back under both
    QED requests its response CANNOT differ from the constrained decoder's.
    """
    rng = np.random.default_rng(16)
    atoms = np.array([0, 0, 0, 0], dtype=np.int64)
    probabilities = rng.random((4, 4, 4))
    probabilities = 0.5 * (probabilities + probabilities.transpose(1, 0, 2))
    probabilities /= probabilities.sum(-1, keepdims=True)
    # an odd degree sum can never be realised by any graph -> guaranteed fallback
    infeasible = np.array([1, 1, 1, 2], dtype=np.int64)
    degree_bonds, info = degree_constrained_bond_decode(atoms, probabilities, infeasible)
    constrained_bonds, _ = valence_aware_bond_decode_typed(atoms, probabilities)
    assert info["fallback"] is True
    assert info["fallback_reason"] == "infeasible_degrees"
    assert np.array_equal(degree_bonds, constrained_bonds)


def test_decode_with_graphs_keeps_the_matrices_and_the_project_validity(real_setup):
    from zinc_gnf.generation import decode_embeddings

    torch.manual_seed(0)
    autoencoder = real_setup["autoencoder"]
    counts = [6, 6, 7]
    embeddings = torch.randn(len(counts), max(counts),
                             autoencoder.config.node_embedding_dim)
    records = decode_with_graphs(autoencoder, embeddings, counts,
                                 decoders=("raw", "constrained", "degree"))
    reference = decode_embeddings(autoencoder, embeddings, counts,
                                  decoders=("raw", "constrained", "degree"))
    assert len(records) == len(counts)
    for record, expected, count in zip(records, reference, counts):
        for name in ("raw", "constrained", "degree"):
            matrix = record[name]["bond_matrix"]
            assert matrix.shape == (count, count)
            assert np.array_equal(matrix, matrix.T), "bond matrices must be symmetric"
            # validity must mean exactly what it means in generation.decode_embeddings
            assert bool(record[name]["valid"]) == bool(expected[name]["valid"])
            assert record[name]["smiles"] == expected[name]["smiles"]


def test_decode_with_graphs_rejects_an_unknown_decoder(real_setup):
    with pytest.raises(ValueError):
        decode_with_graphs(real_setup["autoencoder"],
                           torch.zeros(1, 6, real_setup["autoencoder"]
                                       .config.node_embedding_dim), [6],
                           decoders=("telepathy",))


# ------------------------------------------------------- the decoder contrast ---
def test_a_flow_that_ignores_qed_produces_no_decoder_gap(real_setup):
    flow = conditional_flow(features=("n", "qed"), dimension=64)
    flow.set_data_statistics(torch.zeros(64), torch.full((64,), 0.1))
    row = decoder_contrast(flow, real_setup["autoencoder"], scaler=scaler(), n=6,
                           qed_low=0.4, qed_high=0.8, count=12, temperature=1.0,
                           device="cpu", seed=3, resamples=200)
    common = row["common_subset"]
    if not common.get("molecules"):
        pytest.skip("the tiny fixture produced no commonly valid molecule")
    assert common["constrained"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert common["degree"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert common["gap_constrained_minus_degree"]["difference"] == pytest.approx(0.0, abs=1e-9)
    assert not common["gap_constrained_minus_degree"]["excludes_zero"]


def test_the_contrast_uses_one_common_subset_and_its_strata_add_up(real_setup):
    flow = wake_up_conditioning(conditional_flow(features=("n", "qed"), dimension=64),
                                scale=0.3)
    flow.set_data_statistics(torch.zeros(64), torch.full((64,), 0.1))
    row = decoder_contrast(flow, real_setup["autoencoder"], scaler=scaler(), n=6,
                           qed_low=0.4, qed_high=0.8, count=24, temperature=1.0,
                           device="cpu", seed=5, resamples=200)
    common = row["common_subset"]
    if not common.get("molecules"):
        pytest.skip("the tiny fixture produced no commonly valid molecule")
    # the common subset can never exceed either decoder's own valid-in-both population
    for name in ("constrained", "degree"):
        assert common["molecules"] <= row["own_subset"][name]["pairs"]
    attribution = row["attribution"]
    assert attribution["fallback_identity_holds"], \
        "on a fallback the degree decoder IS the constrained decoder"
    assert attribution["fallback_identity_max_deviation"] == pytest.approx(0.0, abs=1e-9)
    # law of total expectation over the three strata
    gap = common["gap_constrained_minus_degree"]["difference"]
    assert attribution["sum_of_contributions"] == pytest.approx(gap, abs=1e-9)
    shares = sum(block["share_of_common_subset"]
                 for block in attribution["strata"].values())
    assert shares == pytest.approx(1.0, abs=1e-9)
    # the mechanism columns the attribution names must all be present
    assert "change_rate" in attribution["predicted_degrees"]
    for name in ("constrained", "degree"):
        assert "bond_set_change_rate" in attribution["graph_response"][name]
        assert "bond_order_change_rate" in attribution["graph_response"][name]
    assert "raw" in row["validity"], "native raw validity must stay separately reported"
    assert "criterion" in row["examples"]


def test_non_finite_samples_yield_no_contrast(real_setup):
    flow = conditional_flow(features=("n", "qed"), dimension=64)
    flow.set_data_statistics(torch.zeros(64), torch.full((64,), 0.1))
    original = flow.inverse

    def broken(z, mask, condition=None):       # noqa: ANN001
        return original(z, mask, condition) * float("nan")

    flow.inverse = broken                      # type: ignore[method-assign]
    try:
        row = decoder_contrast(flow, real_setup["autoencoder"], scaler=scaler(), n=6,
                               qed_low=0.4, qed_high=0.8, count=4, temperature=1.0,
                               device="cpu", seed=3, resamples=100)
    finally:
        flow.inverse = original                # type: ignore[method-assign]
    assert row["nonfinite_rate"] == pytest.approx(1.0)
    assert "common_subset" not in row
    assert "non-finite" in row["note"]


# -------------------------------------------------------- the transfer control ---
def test_transfer_runs_on_the_validation_split_with_an_interval(real_setup):
    dataset = real_setup["val"]
    counts = np.asarray(dataset.n_node)
    qed = np.asarray(dataset.qed, dtype=np.float64)
    n = int(np.bincount(counts).argmax())
    at_n = qed[counts == n]
    if at_n.size < 2 or float(at_n.max() - at_n.min()) < 1e-6:
        pytest.skip("the fixture has no QED spread at this N in validation")
    row = transfer_on_split(real_setup["autoencoder"], dataset, n=n,
                            qed_low=float(at_n.min()), qed_high=float(at_n.max()),
                            window=0.5, max_molecules=8, device="cpu",
                            decoders=("raw",), split_name="validation")
    assert row["split"] == "validation"
    assert row["groups"]["low"]["molecules_available"] >= 1
    if "preservation" in row:
        block = row["preservation"]["raw"]
        assert "ci_low" in block and "ci_high" in block
        assert block["low_molecules"] >= 0 and block["high_molecules"] >= 0
        assert "aromatic_rings_mean" in row["groups"]["high"]["decoders"]["raw"]
        assert "true_qed_mean_of_survivors" in row["groups"]["high"]["decoders"]["raw"]


def test_transfer_detects_a_decoder_that_destroys_the_difference(real_setup):
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
            return self.inner.decode(torch.zeros_like(embeddings), batch,
                                     teacher_forcing=teacher_forcing)

    dataset = real_setup["train"]
    counts = np.asarray(dataset.n_node)
    qed = np.asarray(dataset.qed, dtype=np.float64)
    n = int(np.bincount(counts).argmax())
    at_n = qed[counts == n]
    if float(at_n.max() - at_n.min()) < 1e-6:
        pytest.skip("the fixture has no QED spread at this N")
    row = transfer_on_split(ConstantDecoder(real_setup["autoencoder"]), dataset, n=n,
                            qed_low=float(at_n.min()), qed_high=float(at_n.max()),
                            window=0.5, max_molecules=8, device="cpu", decoders=("raw",),
                            split_name="train")
    if "preservation" in row:
        ratio = row["preservation"]["raw"]["ratio"]
        assert not np.isfinite(ratio) or abs(ratio) < 0.3


# ------------------------------------------------------- the four-axis logic ----
def _instrument(name: str, gain: float, usable: bool = True) -> dict:
    return {"name": name, "validation": {"mae": 0.08, "correlation": 0.6},
            "calibration": {"slope_of_truth_on_prediction": 0.9},
            "reference_contrasts": [{"n": 23, "gain": gain,
                                     "gain_excludes_zero": usable,
                                     "gain_ci_low": gain * 0.8,
                                     "gain_ci_high": gain * 1.2}],
            "null_contrasts": [{"n": 23, "null_sd": 0.01}]}


def _transfer(ratio: float, low: float, high: float, split: str = "validation") -> dict:
    return {"split": split, "n": 23,
            "groups": {"high": {"decoders": {"degree": {"aromatic_rings_mean": 1.2,
                                                        "validity": 0.4}}}},
            "preservation": {"degree": {"ratio": ratio, "ci_low": low, "ci_high": high,
                                        "low_molecules": 60, "high_molecules": 200,
                                        "true_difference": 0.4,
                                        "measured_difference": 0.4 * ratio}}}


def _attenuation(attenuating: bool) -> dict:
    cell = {"n": 23, "common_molecules": 180, "constrained_change": 0.07,
            "degree_change": 0.005 if attenuating else 0.068,
            "gap": 0.065 if attenuating else 0.002,
            "gap_ci": [0.03, 0.10] if attenuating else [-0.01, 0.015],
            "gap_excludes_zero": attenuating, "attenuating": attenuating,
            "channels": ["integer_programming_constraints"] if attenuating else [],
            "degree_prediction_change_rate": 0.02,
            "degree_fallback_rate_high": 0.3, "fallback_identity_holds": True}
    return {"by_temperature": {"0.2": {"cells": [cell], "cells_measured": 1,
                                       "attenuating_cells": int(attenuating),
                                       "channels_implicated":
                                       {"integer_programming_constraints": 1}
                                       if attenuating else {}}}}


def _reading(instrument: str, fraction: float, status: str = "resolved") -> dict:
    return {"instrument": instrument, "n": 23, "temperature": 0.2, "status": status,
            "observed": {"mean": 0.05, "ci_low": 0.03, "ci_high": 0.07, "pairs": 300},
            "implied_real_qed_difference": 0.4 * fraction,
            "implied_fraction_of_request": fraction}


def test_e2_and_e3_can_both_be_supported_at_once():
    """The stage-19 ladder could not express this; the whole point of the rewrite."""
    evidence = assess_evidence(
        readouts=[_instrument("ridge", 0.5)],
        transfer=[_transfer(0.2, 0.1, 0.35)],
        attenuation=_attenuation(True),
        shift={"held_out_auc": 0.55},
        generated_readings=[_reading("ridge", 0.2)])
    supported = set(evidence["summary"]["supported"])
    assert {"E1_generated_direction", "E2_decoder_transfer",
            "E3_degree_attenuation"} <= supported
    assert evidence["findings"]["E4_distribution_shift"]["status"] == "not_supported"


def test_an_inadequate_instrument_is_never_read_as_evidence_about_the_model():
    """A read-out with no demonstrated within-N gain must not produce a conclusion."""
    evidence = assess_evidence(
        readouts=[_instrument("blind", 0.02, usable=False)],
        transfer=[_transfer(0.95, 0.9, 1.0)],
        attenuation=_attenuation(False),
        shift={"held_out_auc": 0.55},
        generated_readings=[_reading("blind", 0.05)])
    finding = evidence["findings"]["E1_generated_direction"]
    assert finding["status"] == "needs_further_test"
    assert "instruments have not been shown to detect" in finding["reasoning"]
    recommendation = recommend_intervention(evidence)
    assert recommendation["primary"] != "C_representation"
    assert recommendation["confidence"] == "low"
    assert any("unmeasured axis is not a cleared axis" in item
               for item in recommendation["preconditions"])


def test_an_unresolved_change_is_inconclusive_not_a_refutation():
    evidence = assess_evidence(
        readouts=[_instrument("ridge", 0.5)],
        transfer=[_transfer(0.95, 0.9, 1.0)],
        attenuation=_attenuation(False),
        shift={"held_out_auc": 0.55},
        generated_readings=[_reading("ridge", 0.2, status="not_resolved")])
    finding = evidence["findings"]["E1_generated_direction"]
    assert finding["status"] == "inconclusive"
    assert "not the same as the change being zero" in finding["reasoning"]


def test_a_strong_embedding_response_clears_the_flow():
    evidence = assess_evidence(
        readouts=[_instrument("ridge", 0.5)],
        transfer=[_transfer(0.95, 0.9, 1.0)],
        attenuation=_attenuation(True),
        shift={"held_out_auc": 0.55},
        generated_readings=[_reading("ridge", 0.95)])
    assert evidence["findings"]["E1_generated_direction"]["status"] == "not_supported"
    assert evidence["findings"]["E2_decoder_transfer"]["status"] == "not_supported"
    recommendation = recommend_intervention(evidence)
    assert recommendation["primary"] == "B_decoder"


def test_a_broken_fallback_identity_invalidates_the_attribution():
    attenuation = _attenuation(True)
    attenuation["by_temperature"]["0.2"]["cells"][0]["fallback_identity_holds"] = False
    evidence = assess_evidence(readouts=[_instrument("ridge", 0.5)],
                               transfer=[_transfer(0.95, 0.9, 1.0)],
                               attenuation=attenuation, shift={"held_out_auc": 0.55},
                               generated_readings=[_reading("ridge", 0.2)])
    finding = evidence["findings"]["E3_degree_attenuation"]
    assert finding["status"] == "needs_further_test"
    assert "precondition" in finding["uncertainty"]


def test_the_recommendation_prefers_b_when_both_mechanisms_hold():
    evidence = assess_evidence(readouts=[_instrument("ridge", 0.5)],
                               transfer=[_transfer(0.95, 0.9, 1.0)],
                               attenuation=_attenuation(True),
                               shift={"held_out_auc": 0.55},
                               generated_readings=[_reading("ridge", 0.2)])
    recommendation = recommend_intervention(evidence)
    assert recommendation["primary"] == "B_decoder"
    assert recommendation["confidence"] == "high"
    assert "not exclusive" in recommendation["rationale"]
    assert set(recommendation["options"]) == {"A_flow", "B_decoder", "C_representation"}
    for option in recommendation["options"].values():
        for key in ("expected_benefit", "risks", "compute_cost", "reversible"):
            assert key in option
    assert "changes no architecture" in recommendation["not_implemented_note"]


def test_c_is_only_recommended_when_the_representation_is_implicated():
    evidence = assess_evidence(readouts=[_instrument("ridge", 0.5)],
                               transfer=[_transfer(0.2, 0.1, 0.35)],
                               attenuation=_attenuation(False),
                               shift={"held_out_auc": 0.92},
                               generated_readings=[_reading("ridge", 0.2)])
    assert evidence["findings"]["E4_distribution_shift"]["status"] == "supported"
    assert recommend_intervention(evidence)["primary"] == "C_representation"


def test_the_report_states_that_mae_is_not_the_threshold():
    evidence = assess_evidence(readouts=[_instrument("ridge", 0.5)],
                               transfer=[_transfer(0.9, 0.85, 0.95)],
                               attenuation=_attenuation(True),
                               shift={"held_out_auc": 0.55},
                               generated_readings=[_reading("ridge", 0.2)])
    text = format_report({
        "flow": {"checkpoint": "x.pt", "step": 1, "sha256": "ab"},
        "target_n": [23], "qed_pair": [0.4, 0.8], "temperatures": [0.2], "count": 384,
        "transfer": [_transfer(0.9, 0.85, 0.95)],
        "readouts": [_instrument("ridge", 0.5)],
        "generated_readings": [_reading("ridge", 0.2)],
        "attenuation": _attenuation(True),
        "support": [{"split": "train", "n": 23, "molecules_at_this_n": 17010,
                     "by_qed": {"0.4": {"molecules_within_window": 236},
                                "0.8": {"molecules_within_window": 6336}},
                     "minimum_over_qed_values": 236}],
        "evidence": evidence,
        "recommendation": recommend_intervention(evidence)})
    assert "is NOT that threshold" in text
    assert "RECOMMENDATION" in text
    assert "four independent axes" in text
    assert "No axis suppresses another" in text
    assert "exactly zero to the gap" in text


def test_the_attenuation_summary_keeps_temperatures_apart():
    rows = [
        {"n": 23, "temperature": 0.2, "common_subset": {
            "molecules": 100, "constrained": {"mean": 0.07}, "degree": {"mean": 0.005},
            "gap_constrained_minus_degree": {"difference": 0.065, "ci_low": 0.03,
                                             "ci_high": 0.1, "excludes_zero": True},
            "selection_effect": {"degree": 0.001}},
         "attribution": {"strata": {"solved_both": {"difference": 0.07,
                                                    "excludes_zero": True}},
                         "predicted_degrees": {"change_rate": 0.02},
                         "fallback_rate": {"high": 0.3},
                         "graph_response": {"constrained": {"bond_set_change_rate": 0.8},
                                            "degree": {"bond_set_change_rate": 0.05}},
                         "fallback_identity_holds": True}},
        {"n": 23, "temperature": 1.0, "common_subset": {
            "molecules": 100, "constrained": {"mean": 0.03}, "degree": {"mean": 0.028},
            "gap_constrained_minus_degree": {"difference": 0.002, "ci_low": -0.01,
                                             "ci_high": 0.015, "excludes_zero": False},
            "selection_effect": {"degree": 0.0}},
         "attribution": {"strata": {}, "predicted_degrees": {"change_rate": 0.4},
                         "fallback_rate": {"high": 0.2}, "graph_response": {},
                         "fallback_identity_holds": True}},
    ]
    summary = summarise_attenuation(rows)
    assert set(summary["by_temperature"]) == {"0.2", "1"}
    assert summary["by_temperature"]["0.2"]["attenuating_cells"] == 1
    assert summary["by_temperature"]["1"]["attenuating_cells"] == 0
    channels = summary["by_temperature"]["0.2"]["channels_implicated"]
    assert "integer_programming_constraints" in channels
    assert "bond_set_frozen_by_degree_sequence" in channels
    assert "not universal across" in summary["note"]


# ------------------------------------------------- the audit script, end to end -
def test_the_audit_runs_and_grades_four_axes(real_setup, tmp_path):
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
    assert load_script("21_qed_pathway_audit.py").main([
        "--flow-checkpoint", str(run / "last.pt"), "--ae-checkpoint", str(ae_path),
        "--data-dir", str(data), "--output-dir", str(output), "--target-n", "9",
        "--qed-pair", "0.4,0.8", "--count", "8", "--temperature", "0.2",
        "--qed-window", "0.5", "--probe-train-molecules", "20",
        "--probe-val-molecules", "8", "--probe-epochs", "2", "--transfer-molecules", "8",
        "--reference-pairs", "8", "--bootstrap-resamples", "100",
        "--device", "cpu"]) == 0

    payload = json.loads((output / "qed_pathway.json").read_text())
    assert payload["stage"] == 20
    findings = payload["evidence"]["findings"]
    assert set(findings) == {"E1_generated_direction", "E2_decoder_transfer",
                             "E3_degree_attenuation", "E4_distribution_shift"}
    for finding in findings.values():
        assert finding["status"] in {"supported", "not_supported", "inconclusive",
                                     "needs_further_test"}
        assert finding["reasoning"] and finding["uncertainty"]
        assert finding["what_would_settle_it"]
    assert payload["recommendation"]["primary"] in {"A_flow", "B_decoder",
                                                    "C_representation"}
    assert payload["recommendation"]["confidence"] in {"low", "moderate", "high"}
    # both read-outs must be present, each with its own gain and null measurements
    assert {entry["name"] for entry in payload["readouts"]} == {"stage19_mlp",
                                                                "within_n_ridge"}
    for entry in payload["readouts"]:
        assert entry["reference_contrasts"] and entry["null_contrasts"]
        assert "by_n" in entry["stratified"] and "by_qed_band" in entry["stratified"]
        assert "slope_of_truth_on_prediction" in entry["calibration"] \
            or "note" in entry["calibration"]
    # transfer must cover BOTH splits, validation included
    assert {row["split"] for row in payload["transfer"]} == {"train", "validation"}
    assert {row["split"] for row in payload["support"]} == {"train", "validation"}
    text = (output / "qed_pathway.txt").read_text()
    assert "RECOMMENDATION" in text
    assert "is NOT that threshold" in text
    assert "changes no architecture" in text


def test_the_audit_refuses_a_flow_without_qed(real_setup, tmp_path):
    from tests.test_conditional_flow import run_training as run_n_only

    data, ae_path = real_setup["data"], real_setup["ae_path"]
    n_only = tmp_path / "n_only"
    assert run_n_only(data, ae_path, n_only) == 0
    assert load_script("21_qed_pathway_audit.py").main([
        "--flow-checkpoint", str(n_only / "last.pt"), "--ae-checkpoint", str(ae_path),
        "--data-dir", str(data), "--output-dir", str(tmp_path / "o"),
        "--device", "cpu"]) == 4


def test_the_audit_validates_its_arguments(tmp_path):
    script = load_script("21_qed_pathway_audit.py")
    assert script.main(["--flow-checkpoint", str(tmp_path / "ghost.pt"),
                        "--output-dir", str(tmp_path / "o"), "--qed-pair", "0.8,0.4"]) == 4
    assert script.main(["--flow-checkpoint", str(tmp_path / "ghost.pt"),
                        "--output-dir", str(tmp_path / "o2")]) == 2


# ------------------------------------------------------------------- SLURM -----
def test_the_stage20_job_is_cpu_only_read_only_and_estimates_its_cost():
    path = REPOSITORY / "slurm" / "20_qed_pathway_audit.sh"
    source = path.read_text(encoding="utf-8")
    directives = {}
    for line in source.splitlines():
        match = re.match(r"#SBATCH\s+--([\w-]+)(?:=(.*))?$", line.strip())
        if match:
            directives[match.group(1)] = (match.group(2) or "").strip()
    assert directives["partition"] == "CPU"
    assert "gres" not in directives, "stage 20 needs no GPU"
    assert directives["time"] and directives["mem"]
    assert 'if [[ -z "${SLURM_JOB_ID:-}" ]]' in source, "must refuse the login node"
    assert "Runtime estimate" in source
    assert "Memory estimate" in source
    executable = "\n".join(line for line in source.splitlines()
                           if not line.lstrip().startswith("#") and "echo " not in line)
    assert "scripts/21_qed_pathway_audit.py" in executable
    assert "04_train_flow.py" not in executable, "stage 20 must not train the flow"
    assert "17_structure_finetune.py" not in executable
    assert "--device cpu" in executable
    assert not re.search(r"(?m)^\s*(rm|mv)\s", executable), "must not delete or move anything"
    assert "outputs/metrics/qed_pathway" in source
    # it must not write into any stage 17-19 output directory
    for protected in ("decoder_information", "warm_start", "signal_stageA", "stageb_arm"):
        assert f"OUT={protected}" not in source
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0


def test_the_stage20_job_reads_the_existing_stage17_checkpoint():
    source = (REPOSITORY / "slurm" / "20_qed_pathway_audit.sh").read_text(encoding="utf-8")
    assert "warm_start_stageA" in source, "stage 20 reuses the stage-17 warm start"
    assert "READ-ONLY" in source
