#!/usr/bin/env python3
"""ACP falsification gate: can a LATENT TRANSLATION at a FIXED condition move RDKit QED?

READ-ONLY. No training, no checkpoint written, no existing output touched. The test split
is never loaded. Models are opened in eval() with gradients disabled.

WHAT THIS DECIDES
-----------------
The anchored-conditional-prior (ACP) proposal rests on one premise: that adding a fixed
vector to the latent -- which is the ONLY thing a conditional prior mean can do, since
`NodePrior` broadcasts `mu` of shape [B, D] across nodes -- changes the decoded RDKit QED.
This gate tests that premise directly and is designed to REJECT quickly.

Everything happens at ONE fixed reference condition c_ref = (N, QED=0.6):

    z        = f(x, c_ref)                 forward map, same function for every molecule
    z_shift  = z + delta * d_real * w      uniform shift on REAL nodes only
    x_shift  = f^-1(z_shift, c_ref)        SAME frozen map, SAME condition
    molecule = decode(x_shift)             frozen decoder + assembler
    measure    RDKit QED

Fixing c_ref is the methodological core. If each molecule were mapped with its own
QED-conditioned transform, a group difference would conflate real molecular differences
with condition-dependent changes in the map itself. With c_ref fixed, f is one function,
so differences in z reflect differences in x alone.

Because the inverse condition is held fixed, this isolates the PRIOR/translation channel
from the FiLM/map channel. That is exactly the channel ACP would modify, and exactly the
channel the job-366922 geometry found to be weak (prior-route projection -0.0089 at N=23)
and sometimes wrong-signed. A negative result here falsifies prior-only intervention.

DIRECTIONS COMPARED (the "does it beat just moving the existing prior?" control)
-------------------------------------------------------------------------------
  real_qed      normalised pooled difference-of-means between TRAIN low- and high-QED
                bands at fixed c_ref. Its norm d_real is the train-measured real
                displacement and sets the magnitude unit. No hyperparameter, no tuning.
  prior_delta   normalised mu(N, 0.8) - mu(N, 0.4) from the model's OWN prior. Shifting
                along this is "simply moving the existing prior", so real_qed must beat
                it to justify a learned anchor.
  random        fixed-seed isotropic unit vector. Tests whether a shift of this SIZE
                moves QED regardless of direction.

Direction and magnitude are fitted on TRAIN ONLY and never tuned on the evaluated
samples.

TWO ARMS
--------
  real        held-out VALIDATION molecules whose TRUE QED is in the low band. Starting
              point is on-manifold. Selection is on a label, not an outcome.
  generated   the model's own samples at c_ref, swept in BOTH directions. No post-hoc
              selection on decoded QED, so no outcome conditioning. The two-sided sweep
              also gives a monotonicity test.

WHAT A UNIFORM SHIFT PROVABLY CANNOT DO
---------------------------------------
A prior mean shift is identical on every node. It can move the pooled mean and nothing
else. The report therefore also measures the part of the real low->high change a mean
shift cannot reach: the per-node second-moment difference between the two bands. If that
dominates, prior-mean intervention is capped no matter how well the direction is chosen.

CONTROLS AND SANITY CHECKS
--------------------------
  * delta = 0 on real molecules must reconstruct the original molecule (SMILES and QED).
  * delta = 0 on generated samples is the unshifted baseline every cell is compared to.
  * forward/inverse round-trip error is reported on the same latents actually used.
  * the model's NATIVE conditional response (generate at c=(N,0.4) vs (N,0.8) with its
    own conditioning, no shift) is measured as the incumbent to beat.
  * molecular size is reported, because QED is size-dependent and a QED change bought by
    changing atom count would not be the effect we want.

Exit codes: 0 ok; 2 missing input; 3 provenance/vocabulary mismatch; 4 bad arguments;
1 unexpected. The gate never "passes" by raising: the verdict is data in the report.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from zinc_gnf.conditioning import (  # noqa: E402
    condition_embedding_for,
    load_condition_scaler,
    standardise_condition,
)
from zinc_gnf.dataset import (  # noqa: E402
    MolecularGraphDataset,
    iter_eval_batches,
    load_training_splits,
)
from zinc_gnf.trainer_autoencoder import ResumeMismatchError, load_autoencoder  # noqa: E402
from zinc_gnf.trainer_flow import file_sha256, load_flow  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402

DIRECTIONS = ("real_qed", "prior_delta", "random")
RANDOM_DIRECTION_SEED = 20261109
GENERATION_SEED = 20261110
BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261111
# Below this many train molecules in a band the fitted direction is not trustworthy.
MIN_BAND_FOR_FIT = 50
MIN_BAND_WARN = 150
# Below this many validation molecules the real arm is reported but flagged.
MIN_REAL_ARM = 15


# ------------------------------------------------------------------------ helpers ---
def bootstrap_ci(values, *, resamples: int = BOOTSTRAP, seed: int = BOOTSTRAP_SEED):
    array = np.asarray([v for v in np.ravel(values) if v is not None and np.isfinite(v)],
                       dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": None, "ci95": [None, None], "excludes_zero": False}
    if array.size == 1:
        return {"n": 1, "mean": float(array[0]), "ci95": [None, None],
                "excludes_zero": False}
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, array.size, size=(resamples, array.size))].mean(axis=1)
    low, high = float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
    return {"n": int(array.size), "mean": float(array.mean()),
            "sd": float(array.std(ddof=1)), "ci95": [low, high],
            "excludes_zero": bool(low > 0.0 or high < 0.0)}


def descriptors(mol) -> dict[str, float]:
    from rdkit.Chem import QED, rdMolDescriptors
    return {
        "qed": float(QED.qed(mol)),
        "aromatic_rings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "rings": float(rdMolDescriptors.CalcNumRings(mol)),
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
    }


def reference_condition(flow, scaler, n: int, qed: float, count: int, device):
    """c for (N, qed) repeated `count` times, standardised with the TRAIN-only scaler."""
    raw = standardise_condition(flow.config.condition_features, scaler=scaler,
                                n=[n] * count, qed=[qed] * count)
    values = torch.from_numpy(raw).to(device)
    return condition_embedding_for(flow, values, torch.ones_like(values), count, device)


@torch.no_grad()
def pooled_latents(flow, autoencoder, dataset, indices, *, n, qed_ref, scaler, device,
                   batch_size):
    """Mean-pooled z = f(x, c_ref) over REAL nodes, plus per-node second moments.

    The SAME c_ref for every molecule, so f is one fixed function and differences in z
    come only from differences in x. Pooling matches the object a prior mean can address:
    `NodePrior` produces mu of shape [B, D] and broadcasts it over nodes.
    """
    pooled, second, counts = [], [], 0
    for batch in iter_eval_batches(dataset, batch_size, list(indices)):
        batch = batch.to(device)
        cond = reference_condition(flow, scaler, n, qed_ref, batch.batch_size, device)
        x = autoencoder.encode(batch)
        z, _ = flow(x, batch.node_mask, cond)
        mask = batch.node_mask.unsqueeze(-1).to(z.dtype)
        denom = batch.node_mask.sum(-1, keepdim=True).to(z.dtype)
        mean = (z * mask).sum(1) / denom                       # [B, D]
        # per-node dispersion within each molecule, the part a uniform shift cannot touch
        var = (((z - mean.unsqueeze(1)) ** 2) * mask).sum(1) / denom
        pooled.append(mean.detach().cpu().numpy().astype(np.float64))
        second.append(var.detach().cpu().numpy().astype(np.float64))
        counts += batch.batch_size
    if not pooled:
        return None, None
    return np.concatenate(pooled), np.concatenate(second)


@torch.no_grad()
def fit_direction(flow, autoencoder, train_set, *, n, qed_ref, qed_low, qed_high,
                  halfwidth, scaler, device, batch_size, max_per_band):
    """TRAIN-only within-N QED direction at fixed c_ref, and its magnitude unit."""
    out: dict[str, Any] = {"n": int(n), "qed_ref": float(qed_ref),
                           "qed_low": float(qed_low), "qed_high": float(qed_high)}
    bands = {}
    for label, centre in (("low", qed_low), ("high", qed_high)):
        eligible = np.flatnonzero((train_set.n_node == n)
                                 & (np.abs(train_set.qed - centre) <= halfwidth))
        out[f"train_molecules_{label}"] = int(eligible.size)
        if eligible.size < MIN_BAND_FOR_FIT:
            out["usable"] = False
            out["note"] = (f"only {eligible.size} TRAIN molecules at N={n}, QED~{centre} "
                           f"(need {MIN_BAND_FOR_FIT}); direction not fitted")
            return out, None, None
        rng = np.random.default_rng(RANDOM_DIRECTION_SEED + n)
        chosen = rng.permutation(eligible)[:max_per_band].tolist()
        mean, var = pooled_latents(flow, autoencoder, train_set, chosen, n=n,
                                   qed_ref=qed_ref, scaler=scaler, device=device,
                                   batch_size=batch_size)
        bands[label] = {"pooled": mean, "pernode_var": var}
    low, high = bands["low"]["pooled"], bands["high"]["pooled"]
    displacement = high.mean(0) - low.mean(0)
    d_real = float(np.linalg.norm(displacement))
    out["usable"] = True
    out["low_band_warning"] = bool(out["train_molecules_low"] < MIN_BAND_WARN)
    out["high_band_warning"] = bool(out["train_molecules_high"] < MIN_BAND_WARN)
    out["d_real_pooled_displacement"] = d_real
    # within-band dispersion of the pooled mean: the scale at which the shift is resolvable
    spread = 0.5 * (float(np.linalg.norm(low.std(0))) + float(np.linalg.norm(high.std(0))))
    out["within_band_pooled_spread"] = spread
    out["displacement_over_spread"] = d_real / max(spread, 1e-12)
    # the part a UNIFORM shift provably cannot address
    var_low = bands["low"]["pernode_var"].mean(0)
    var_high = bands["high"]["pernode_var"].mean(0)
    out["pernode_variance_change_l2"] = float(np.linalg.norm(var_high - var_low))
    out["pernode_variance_mean_low"] = float(var_low.mean())
    out["pernode_variance_mean_high"] = float(var_high.mean())
    out["uniform_shift_limitation"] = (
        "a prior mean shift is identical on every node, so it can move the pooled mean "
        "only. pernode_variance_change_l2 is the component of the real low->high change "
        "it provably cannot reproduce; compare it with d_real.")
    if d_real < 1e-9:
        out["usable"] = False
        out["note"] = "train low and high bands have the same pooled latent mean"
        return out, None, None
    return out, displacement / d_real, d_real


@torch.no_grad()
def build_directions(flow, scaler, *, n, fitted, d_real, qed_low, qed_high, device, dim):
    """The three candidate directions, all unit vectors, plus their mutual cosines."""
    vectors: dict[str, np.ndarray] = {"real_qed": fitted}
    cond_low = reference_condition(flow, scaler, n, qed_low, 1, device)
    cond_high = reference_condition(flow, scaler, n, qed_high, 1, device)
    mu_low, _ = flow.prior.parameters_for(1, cond_low, device)
    mu_high, _ = flow.prior.parameters_for(1, cond_high, device)
    prior_delta = (mu_high - mu_low).squeeze(0).detach().cpu().numpy().astype(np.float64)
    prior_norm = float(np.linalg.norm(prior_delta))
    vectors["prior_delta"] = (prior_delta / prior_norm if prior_norm > 1e-12
                              else np.zeros(dim))
    generator = np.random.default_rng(RANDOM_DIRECTION_SEED + 1000 + n)
    noise = generator.normal(size=dim)
    vectors["random"] = noise / np.linalg.norm(noise)
    info = {
        "prior_delta_raw_norm": prior_norm,
        "prior_delta_norm_over_d_real": prior_norm / max(d_real, 1e-12),
        "cosines": {
            "real_qed_vs_prior_delta": float(vectors["real_qed"] @ vectors["prior_delta"]),
            "real_qed_vs_random": float(vectors["real_qed"] @ vectors["random"]),
            "prior_delta_vs_random": float(vectors["prior_delta"] @ vectors["random"]),
        },
        "note": ("prior_delta_norm_over_d_real < 1 means the model's own prior moves LESS "
                 "than real molecules of these two QED bands differ."),
    }
    return vectors, info


@torch.no_grad()
def decode_and_measure(autoencoder, embeddings, counts, decoders):
    """Decode and score; returns one record per molecule per decoder."""
    from rdkit import Chem, rdBase
    from zinc_gnf.generation import decode_embeddings

    records = decode_embeddings(autoencoder, embeddings, counts, decoders=tuple(decoders))
    out: dict[str, list[dict[str, Any]]] = {d: [] for d in decoders}
    for name in decoders:
        for record in records:
            entry = record.get(name) or {}
            row: dict[str, Any] = {
                "valid": bool(entry.get("valid")),
                "fallback": bool(entry.get("fallback", False)),
                "smiles": entry.get("smiles"),
            }
            if entry.get("valid") and entry.get("smiles"):
                with rdBase.BlockLogs():
                    mol = Chem.MolFromSmiles(entry["smiles"])
                if mol is not None:
                    row.update(descriptors(mol))
            out[name].append(row)
    return out


def summarise_cell(rows, *, baseline=None) -> dict[str, Any]:
    """Validity, QED, aromaticity, size, uniqueness; paired deltas against a baseline."""
    total = len(rows)
    valid = [r for r in rows if r.get("valid") and "qed" in r]
    block: dict[str, Any] = {
        "attempts": total,
        "valid": len(valid),
        "validity": len(valid) / max(1, total),
        "fallback_rate": sum(r["fallback"] for r in rows) / max(1, total),
        "unique_smiles": len({r["smiles"] for r in valid}) if valid else 0,
        "uniqueness_over_valid": (len({r["smiles"] for r in valid}) / len(valid)
                                  if valid else None),
    }
    for key in ("qed", "aromatic_rings", "rings", "heavy_atoms"):
        series = [r[key] for r in valid]
        block[f"{key}_mean"] = float(np.mean(series)) if series else None
    if baseline is not None:
        # PAIRED by sample index: the same eps / same molecule at shift 0 vs shift delta.
        pairs = [(b, r) for b, r in zip(baseline, rows)
                 if b.get("valid") and "qed" in b and r.get("valid") and "qed" in r]
        deltas = [r["qed"] - b["qed"] for b, r in pairs]
        size_deltas = [r["heavy_atoms"] - b["heavy_atoms"] for b, r in pairs]
        block["paired_jointly_valid"] = len(pairs)
        block["paired_qed_delta"] = bootstrap_ci(deltas)
        block["paired_heavy_atom_delta"] = bootstrap_ci(size_deltas)
        block["paired_note"] = ("paired on jointly-valid samples at identical sample "
                                "index, so the shift is the only difference")
    return block


# --------------------------------------------------------------------------- arms ---
@torch.no_grad()
def real_arm(flow, autoencoder, val_set, *, n, qed_ref, qed_low, halfwidth, scaler,
             device, batch_size, max_molecules, directions, d_real, deltas, decoders):
    """Held-out REAL low-QED molecules: on-manifold starting points."""
    eligible = np.flatnonzero((val_set.n_node == n)
                             & (np.abs(val_set.qed - qed_low) <= halfwidth))
    result: dict[str, Any] = {
        "validation_molecules_available": int(eligible.size),
        "support_sufficient": bool(eligible.size >= MIN_REAL_ARM),
    }
    if eligible.size == 0:
        result["note"] = f"NO validation support at N={n}, QED~{qed_low}"
        return result
    if eligible.size < MIN_REAL_ARM:
        result["note"] = (f"only {eligible.size} validation molecules (<{MIN_REAL_ARM}); "
                          f"results reported but UNDERPOWERED")
    chosen = eligible[:max_molecules].tolist()

    latents, masks, counts, truths = [], [], [], []
    for batch in iter_eval_batches(val_set, batch_size, chosen):
        batch = batch.to(device)
        cond = reference_condition(flow, scaler, n, qed_ref, batch.batch_size, device)
        x = autoencoder.encode(batch)
        z, _ = flow(x, batch.node_mask, cond)
        latents.append(z)
        masks.append(batch.node_mask)
        counts.extend(int(v) for v in batch.n_node.tolist())
        truths.extend(float(v) for v in batch.qed.detach().cpu())
    result["true_qed_mean"] = float(np.mean(truths))

    per_direction: dict[str, Any] = {}
    zero_shift_reference: dict[str, list] | None = None
    for name, vector in directions.items():
        w = torch.from_numpy(vector).to(device=device, dtype=latents[0].dtype)
        cells: dict[str, Any] = {}
        for delta in deltas:
            rows: dict[str, list] = {d: [] for d in decoders}
            offset = 0
            for z, mask in zip(latents, masks):
                cond = reference_condition(flow, scaler, n, qed_ref, z.shape[0], device)
                shifted = z + float(delta) * float(d_real) * w
                shifted = shifted * mask.unsqueeze(-1).to(shifted.dtype)
                x_back = flow.inverse(shifted, mask, cond)
                chunk = counts[offset:offset + z.shape[0]]
                offset += z.shape[0]
                measured = decode_and_measure(autoencoder, x_back, chunk, decoders)
                for d in decoders:
                    rows[d].extend(measured[d])
            if abs(delta) < 1e-12:
                zero_shift_reference = rows
            cells[f"{delta:+.2f}"] = {
                d: summarise_cell(rows[d],
                                  baseline=(zero_shift_reference or {}).get(d))
                for d in decoders}
        per_direction[name] = cells
    result["by_direction"] = per_direction

    # zero-shift reconstruction sanity: delta=0 must return the original molecule
    if zero_shift_reference is not None:
        sanity = {}
        for d in decoders:
            rows = zero_shift_reference[d]
            valid = [r for r in rows if r.get("valid") and "qed" in r]
            sanity[d] = {
                "validity": len(valid) / max(1, len(rows)),
                "qed_mean": float(np.mean([r["qed"] for r in valid])) if valid else None,
                "qed_mae_vs_true": (
                    float(np.mean([abs(r["qed"] - t) for r, t in zip(rows, truths)
                                   if r.get("valid") and "qed" in r])) if valid else None),
                "note": ("delta=0 is x -> f -> f^-1 -> decode. A large MAE here means the "
                         "pipeline itself is lossy and every shifted number inherits it."),
            }
        result["zero_shift_reconstruction"] = sanity
    return result


@torch.no_grad()
def generated_arm(flow, autoencoder, *, n, qed_ref, scaler, device, samples, temperature,
                  directions, d_real, deltas, decoders, eps_seed):
    """The model's OWN samples at c_ref, swept in BOTH directions.

    No post-hoc selection on decoded QED, so no outcome conditioning. The two-sided sweep
    is what makes non-monotonicity visible.
    """
    dim = int(flow.config.node_embedding_dim)
    mask = torch.ones((samples, n), dtype=torch.bool, device=device)
    generator = torch.Generator(device=device).manual_seed(int(eps_seed))
    eps = torch.randn((samples, n, dim), device=device, generator=generator)
    cond = reference_condition(flow, scaler, n, qed_ref, samples, device)
    mu, sigma = flow.prior.parameters_for(samples, cond, device)
    z0 = (mu.unsqueeze(1) + float(temperature) * sigma.unsqueeze(1) * eps) \
        * mask.unsqueeze(-1)

    # round-trip on the latents actually used
    x0 = flow.inverse(z0, mask, cond)
    z_round, _ = flow(x0, mask, cond)
    scale = float(z0.abs().max().clamp(min=1e-30))
    result: dict[str, Any] = {
        "samples": int(samples), "temperature": float(temperature),
        "round_trip_max_absolute": float((z_round - z0).abs().max()),
        "round_trip_reference_scale": scale,
        "round_trip_max_relative": float((z_round - z0).abs().max()) / scale,
    }

    per_direction: dict[str, Any] = {}
    zero_rows = decode_and_measure(autoencoder, x0, [n] * samples, decoders)
    result["unshifted_control"] = {d: summarise_cell(zero_rows[d]) for d in decoders}
    for name, vector in directions.items():
        w = torch.from_numpy(vector).to(device=device, dtype=z0.dtype)
        cells: dict[str, Any] = {}
        for delta in deltas:
            shifted = (z0 + float(delta) * float(d_real) * w) * mask.unsqueeze(-1)
            x_back = flow.inverse(shifted, mask, cond)
            rows = decode_and_measure(autoencoder, x_back, [n] * samples, decoders)
            cells[f"{delta:+.2f}"] = {
                d: summarise_cell(rows[d], baseline=zero_rows[d]) for d in decoders}
        per_direction[name] = cells
    result["by_direction"] = per_direction
    return result


@torch.no_grad()
def native_response(flow, autoencoder, *, n, qed_low, qed_high, scaler, device, samples,
                    temperature, decoders, eps_seed):
    """The incumbent: the model's OWN conditional response with no shift at all.

    Generated with its real conditioning at (N, qed_low) and (N, qed_high), common eps.
    Any shift-based gain must beat THIS, not zero.
    """
    dim = int(flow.config.node_embedding_dim)
    mask = torch.ones((samples, n), dtype=torch.bool, device=device)
    generator = torch.Generator(device=device).manual_seed(int(eps_seed))
    eps = torch.randn((samples, n, dim), device=device, generator=generator)
    out: dict[str, Any] = {}
    rows_by_q: dict[str, dict[str, list]] = {}
    for label, q in (("low", qed_low), ("high", qed_high)):
        cond = reference_condition(flow, scaler, n, q, samples, device)
        mu, sigma = flow.prior.parameters_for(samples, cond, device)
        z = (mu.unsqueeze(1) + float(temperature) * sigma.unsqueeze(1) * eps) \
            * mask.unsqueeze(-1)
        x = flow.inverse(z, mask, cond)
        rows_by_q[label] = decode_and_measure(autoencoder, x, [n] * samples, decoders)
        out[label] = {d: summarise_cell(rows_by_q[label][d]) for d in decoders}
    out["paired_response"] = {
        d: summarise_cell(rows_by_q["high"][d], baseline=rows_by_q["low"][d])["paired_qed_delta"]
        for d in decoders}
    out["note"] = ("the model's native conditional response at common eps. This is the "
                   "number any latent-shift intervention must beat.")
    return out


# --------------------------------------------------------------------------- main ---
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flow-checkpoint", required=True)
    p.add_argument("--ae-checkpoint", default=None)
    p.add_argument("--data-dir", default="data/processed/zinc250k_charge_aware_moflow_fixed")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target-n", default="15,23,30")
    p.add_argument("--qed-reference", type=float, default=0.6)
    p.add_argument("--qed-low", type=float, default=0.4)
    p.add_argument("--qed-high", type=float, default=0.8)
    p.add_argument("--qed-halfwidth", type=float, default=0.05)
    p.add_argument("--shift-deltas", default="-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5")
    p.add_argument("--real-molecules", type=int, default=32)
    p.add_argument("--generated-samples", type=int, default=32)
    p.add_argument("--train-per-band", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--decoders", default="degree,constrained")
    p.add_argument("--directions", default="real_qed,prior_delta,random")
    p.add_argument("--eps-seed", type=int, default=GENERATION_SEED)
    p.add_argument("--allow-ae-mismatch", action="store_true")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        targets = [int(v) for v in args.target_n.split(",") if v.strip()]
        deltas = [float(v) for v in args.shift_deltas.split(",") if v.strip()]
        decoders = [d.strip() for d in args.decoders.split(",") if d.strip()]
        wanted = [d.strip() for d in args.directions.split(",") if d.strip()]
        if not targets or not deltas or not decoders:
            raise ValueError("need at least one N, one delta and one decoder")
        if not any(abs(d) < 1e-12 for d in deltas):
            raise ValueError("shift-deltas must include 0.0 (the sanity/control cell)")
        unknown = [d for d in wanted if d not in DIRECTIONS]
        if unknown:
            raise ValueError(f"unknown directions {unknown}; known {list(DIRECTIONS)}")
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4

    started = time.perf_counter()
    try:
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        device = torch.device(args.device)
        scaler = load_condition_scaler(args.data_dir)
        train_records, val_records, fingerprint = load_training_splits(args.data_dir)
        train_set = MolecularGraphDataset(train_records)
        val_set = MolecularGraphDataset(val_records)
        del train_records, val_records

        flow, probe = load_flow(args.flow_checkpoint, device=device)
        flow.eval().requires_grad_(False)
        if "qed" not in flow.config.condition_features:
            raise ValueError("this gate needs a QED-conditional flow")
        recorded = probe.get("autoencoder") or {}
        ae_path = Path(args.ae_checkpoint or recorded.get("checkpoint", ""))
        if not ae_path.is_file():
            raise FileNotFoundError(f"autoencoder not found: {ae_path}")
        digest = file_sha256(ae_path)
        match = bool(recorded.get("sha256") == digest)
        if not match and not args.allow_ae_mismatch:
            raise ResumeMismatchError(
                f"autoencoder SHA256 mismatch: flow recorded "
                f"{str(recorded.get('sha256'))[:16]}, {ae_path} is {digest[:16]}")
        autoencoder, _ = load_autoencoder(ae_path, device=device)
        autoencoder.eval().requires_grad_(False)
        print(f"flow {args.flow_checkpoint} step={probe.get('step')}")
        print(f"autoencoder provenance match: {match} ({digest[:16]})", flush=True)

        payload: dict[str, Any] = {
            "purpose": ("ACP falsification gate: does a uniform latent translation at a "
                        "FIXED reference condition change decoded RDKit QED?"),
            "flow": {"checkpoint": str(args.flow_checkpoint),
                     "sha256": file_sha256(args.flow_checkpoint),
                     "step": probe.get("step"),
                     "config": flow.config.to_dict()},
            "autoencoder": {"checkpoint": str(ae_path), "sha256_actual": digest,
                            "sha256_recorded_in_flow": recorded.get("sha256"),
                            "sha256_match": match,
                            "mismatch_override_used": bool(not match)},
            "data_dir": args.data_dir, "data_fingerprint": fingerprint,
            "qed_reference": args.qed_reference, "qed_low": args.qed_low,
            "qed_high": args.qed_high, "shift_deltas": deltas,
            "temperature": args.temperature, "decoders": decoders,
            "protocol": (
                "one fixed reference condition c_ref=(N, qed_reference) for BOTH the "
                "forward map used to fit the direction and the inverse map used to "
                "generate, so no group difference is confounded by a condition-dependent "
                "transform. Direction and magnitude fitted on TRAIN only."),
            "by_n": {},
        }

        dim = int(flow.config.node_embedding_dim)
        for n in targets:
            print(f"\n=== N={n} ===", flush=True)
            fit, unit, d_real = fit_direction(
                flow, autoencoder, train_set, n=n, qed_ref=args.qed_reference,
                qed_low=args.qed_low, qed_high=args.qed_high,
                halfwidth=args.qed_halfwidth, scaler=scaler, device=device,
                batch_size=args.batch_size, max_per_band=args.train_per_band)
            block: dict[str, Any] = {"direction_fit": fit}
            if not fit.get("usable"):
                print(f"  SKIPPED: {fit.get('note')}", flush=True)
                payload["by_n"][str(n)] = block
                continue
            print(f"  train bands: low={fit['train_molecules_low']} "
                  f"high={fit['train_molecules_high']}  d_real={d_real:.4f}  "
                  f"d_real/spread={fit['displacement_over_spread']:.3f}  "
                  f"pernode_var_change={fit['pernode_variance_change_l2']:.4f}", flush=True)

            vectors, info = build_directions(
                flow, scaler, n=n, fitted=unit, d_real=d_real, qed_low=args.qed_low,
                qed_high=args.qed_high, device=device, dim=dim)
            block["directions"] = info
            selected = {k: v for k, v in vectors.items() if k in wanted}
            print(f"  prior_delta/d_real={info['prior_delta_norm_over_d_real']:.4f}  "
                  f"cos(real_qed, prior_delta)="
                  f"{info['cosines']['real_qed_vs_prior_delta']:+.3f}", flush=True)

            block["native_response"] = native_response(
                flow, autoencoder, n=n, qed_low=args.qed_low, qed_high=args.qed_high,
                scaler=scaler, device=device, samples=args.generated_samples,
                temperature=args.temperature, decoders=decoders, eps_seed=args.eps_seed)
            block["real_arm"] = real_arm(
                flow, autoencoder, val_set, n=n, qed_ref=args.qed_reference,
                qed_low=args.qed_low, halfwidth=args.qed_halfwidth, scaler=scaler,
                device=device, batch_size=args.batch_size,
                max_molecules=args.real_molecules, directions=selected, d_real=d_real,
                deltas=deltas, decoders=decoders)
            block["generated_arm"] = generated_arm(
                flow, autoencoder, n=n, qed_ref=args.qed_reference, scaler=scaler,
                device=device, samples=args.generated_samples,
                temperature=args.temperature, directions=selected, d_real=d_real,
                deltas=deltas, decoders=decoders, eps_seed=args.eps_seed)
            payload["by_n"][str(n)] = block

            head = decoders[0]
            gen = block["generated_arm"]["by_direction"].get("real_qed", {})
            cell = gen.get("+1.00", {}).get(head, {})
            delta = (cell.get("paired_qed_delta") or {}).get("mean")
            print(f"  GENERATED arm, real_qed direction, delta=+1.00 [{head}]: "
                  f"paired dQED={delta if delta is None else f'{delta:+.4f}'}  "
                  f"validity={cell.get('validity')}", flush=True)

        payload["elapsed_seconds"] = time.perf_counter() - started
        payload["verdict"] = verdict(payload, decoders[0])
        atomic_write_json(payload, out_dir / "acp_gate.json")
        (out_dir / "acp_gate.txt").write_text(format_gate(payload, decoders))
        print(f"\nwrote {out_dir / 'acp_gate.json'}")
        print(f"wrote {out_dir / 'acp_gate.txt'}")
        print("\n" + payload["verdict"]["summary"])
        return 0
    except FileNotFoundError as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 2
    except ResumeMismatchError as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 3
    except Exception:  # noqa: BLE001
        print("FATAL: unexpected\n" + traceback.format_exc(), file=sys.stderr, flush=True)
        return 1


def verdict(payload: dict[str, Any], head: str) -> dict[str, Any]:
    """Mechanical GO / NO-GO. The criteria are fixed here, not chosen after seeing data."""
    per_n: dict[str, Any] = {}
    for n, block in payload["by_n"].items():
        if "generated_arm" not in block:
            per_n[n] = {"status": "not_evaluated"}
            continue
        gen = block["generated_arm"]
        cell = gen["by_direction"].get("real_qed", {}).get("+1.00", {}).get(head, {})
        base = gen["unshifted_control"][head]
        paired = cell.get("paired_qed_delta") or {}
        native = (block.get("native_response", {}).get("paired_response", {})
                  .get(head) or {})
        random_cell = (gen["by_direction"].get("random", {}).get("+1.00", {})
                       .get(head, {}).get("paired_qed_delta") or {})
        moved = bool(paired.get("excludes_zero")) and (paired.get("mean") or 0) > 0
        kept_validity = (cell.get("validity") is not None
                         and base.get("validity") is not None
                         and cell["validity"] >= 0.8 * base["validity"])
        beats_random = (paired.get("mean") or 0) > abs(random_cell.get("mean") or 0)
        beats_native = (paired.get("mean") or 0) > (native.get("mean") or 0)
        kept_diversity = (cell.get("uniqueness_over_valid") is None
                          or cell["uniqueness_over_valid"] >= 0.5)
        per_n[n] = {
            "status": "GO" if (moved and kept_validity and beats_random
                               and kept_diversity) else "NO-GO",
            "paired_qed_delta": paired.get("mean"),
            "paired_ci95": paired.get("ci95"),
            "excludes_zero": paired.get("excludes_zero"),
            "validity_shifted": cell.get("validity"),
            "validity_unshifted": base.get("validity"),
            "validity_retained": kept_validity,
            "uniqueness": cell.get("uniqueness_over_valid"),
            "beats_random_direction": beats_random,
            "beats_native_conditioning": beats_native,
            "native_paired_delta": native.get("mean"),
        }
    decided = [v for v in per_n.values() if v.get("status") in ("GO", "NO-GO")]
    gos = [v for v in decided if v["status"] == "GO"]
    overall = ("GO" if decided and len(gos) == len(decided)
               else "PARTIAL" if gos else "NO-GO")
    return {
        "overall": overall,
        "per_n": per_n,
        "criteria": (
            "GO at an N requires, on the GENERATED arm with the real_qed direction at "
            "delta=+1.00 and the headline decoder: a positive paired QED change whose 95% "
            "bootstrap CI excludes zero; validity at least 80% of the unshifted control; "
            "uniqueness among valid at least 0.5; and a larger effect than the random "
            "direction. 'beats_native_conditioning' is reported separately because an "
            "intervention that does not beat the model's own conditioning is not useful "
            "even if it is statistically non-zero."),
        "summary": (
            f"VERDICT: {overall}. " + "; ".join(
                f"N={n}: {v.get('status')}"
                + (f" (dQED={v['paired_qed_delta']:+.4f})"
                   if v.get("paired_qed_delta") is not None else "")
                for n, v in per_n.items())),
    }


def format_gate(payload: dict[str, Any], decoders: list[str]) -> str:
    head = decoders[0]
    lines = ["ACP falsification gate", "=" * 78, "",
             f"flow: {payload['flow']['checkpoint']} (step {payload['flow']['step']})",
             f"autoencoder: {payload['autoencoder']['checkpoint']}",
             f"  provenance match: {payload['autoencoder']['sha256_match']}"
             + ("  *** OVERRIDE USED ***"
                if payload['autoencoder']['mismatch_override_used'] else ""),
             f"reference condition: QED={payload['qed_reference']} (fixed for forward AND "
             f"inverse)", f"headline decoder: {head}", "",
             payload["protocol"], ""]
    for n, block in payload["by_n"].items():
        lines += [f"--- N={n} ---"]
        fit = block.get("direction_fit", {})
        if not fit.get("usable"):
            lines += [f"  SKIPPED: {fit.get('note')}", ""]
            continue
        lines += [f"  train bands: low={fit['train_molecules_low']} "
                  f"high={fit['train_molecules_high']}"
                  + ("  *** LOW-BAND SUPPORT WARNING ***" if fit.get("low_band_warning")
                     else ""),
                  f"  d_real={fit['d_real_pooled_displacement']:.4f}  "
                  f"within-band spread={fit['within_band_pooled_spread']:.4f}  "
                  f"ratio={fit['displacement_over_spread']:.3f}",
                  f"  per-node variance change a uniform shift CANNOT reach: "
                  f"{fit['pernode_variance_change_l2']:.4f}"]
        info = block.get("directions", {})
        if info:
            lines.append(f"  prior_delta/d_real="
                         f"{info['prior_delta_norm_over_d_real']:.4f}  "
                         f"cos(real_qed,prior_delta)="
                         f"{info['cosines']['real_qed_vs_prior_delta']:+.3f}  "
                         f"cos(real_qed,random)="
                         f"{info['cosines']['real_qed_vs_random']:+.3f}")
        native = (block.get("native_response", {}).get("paired_response", {}).get(head)
                  or {})
        if native.get("mean") is not None:
            lines.append(f"  NATIVE conditional response (incumbent): "
                         f"{native['mean']:+.4f} "
                         f"CI[{native['ci95'][0]:+.4f},{native['ci95'][1]:+.4f}]")
        real = block.get("real_arm", {})
        if real.get("zero_shift_reconstruction", {}).get(head):
            s = real["zero_shift_reconstruction"][head]
            mae = s.get("qed_mae_vs_true")
            mae_text = "n/a" if mae is None else f"{mae:.4f}"
            lines.append(f"  zero-shift reconstruction [{head}]: validity "
                         f"{s['validity']:.3f}  QED MAE vs true {mae_text}")
        gen = block.get("generated_arm", {})
        if gen:
            lines.append(f"  round-trip (generated latents): "
                         f"{gen['round_trip_max_relative']:.2e} relative")
        for arm_name, arm in (("REAL", real), ("GENERATED", gen)):
            if not arm.get("by_direction"):
                continue
            lines.append(f"  {arm_name} arm  (paired dQED vs delta=0, [{head}])")
            if arm_name == "REAL" and not arm.get("support_sufficient", True):
                lines.append(f"    *** {arm.get('note')} ***")
            for direction, cells in arm["by_direction"].items():
                parts = []
                for key in sorted(cells, key=lambda k: float(k)):
                    cell = cells[key][head]
                    paired = cell.get("paired_qed_delta") or {}
                    mean = paired.get("mean")
                    parts.append(f"{key}:"
                                 + ("  n/a" if mean is None else f"{mean:+.4f}")
                                 + f"(v{cell['validity']:.2f})")
                lines.append(f"    {direction:<12} " + "  ".join(parts))
                # monotonicity across the sweep
                values = [(float(k), (cells[k][head].get("paired_qed_delta") or {}).get("mean"))
                          for k in sorted(cells, key=lambda k: float(k))]
                usable = [(d, v) for d, v in values if v is not None]
                if len(usable) >= 3:
                    diffs = [b[1] - a[1] for a, b in zip(usable, usable[1:])]
                    mono = all(x >= -1e-9 for x in diffs) or all(x <= 1e-9 for x in diffs)
                    lines.append(f"      monotonic across sweep: {mono}")
        lines.append("")
    v = payload.get("verdict", {})
    lines += ["", "VERDICT", "-" * 78, v.get("summary", ""), "", v.get("criteria", "")]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
