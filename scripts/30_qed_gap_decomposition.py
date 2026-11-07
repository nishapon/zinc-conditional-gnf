#!/usr/bin/env python3
"""Why conditional-likelihood discrimination does not become generative QED control.

READ-ONLY. No training, no checkpoint is written, no existing output is touched. The
test split is never loaded. Every model is opened in eval() with gradients disabled.

Three measurements, in the order that makes them interpretable:

SECTION A - exact three-way decomposition of the likelihood gap
---------------------------------------------------------------
For a real embedding x, a true condition c_t and an alternative condition c_a, write
z_t = f(x, c_t), z_a = f(x, c_a) and let p_t, p_a be the conditional Gaussian priors.
The gap decomposes by telescoping:

    D_z       = log p_t(z_t)            - log p_t(z_a)
    D_prior   = log p_t(z_a)            - log p_a(z_a)
    D_logdet  = log|det J_f(x, c_t)|    - log|det J_f(x, c_a)|
    ------------------------------------------------------------
    D_total   = log p(x|c_t) - log p(x|c_a) = D_z + D_prior + D_logdet

The identity is EXACT (the +-log p_t(z_a) terms cancel), so the script asserts it
numerically and refuses to report if it does not hold. Section A is computed in float64
so the residual is at round-off rather than float32 noise.

HOW TO READ IT - and how NOT to. The three terms are SIGNED and may cancel, so they are
reported as signed nats/node with bootstrap intervals, never as percentages of the total.

  D_z        measures how much the latent MOVES with the condition (the FiLM channel),
             scored in the true prior's metric. A large D_z is NOT by itself a failure:
             a conditional translation of z that the prior mean tracks is precisely how
             correct conditioning works. D_z says the condition reaches the map; it says
             nothing about whether f^-1(mu(c), c) moves usefully. Section B decides that.
  D_prior    measures the (mu, sigma) dependence on c at a FIXED latent.
  D_logdet   measures the condition-dependent volume change. This term cannot move the
             location of a low-temperature sample, so it is the one channel that buys
             discrimination with no possible generative benefit.

SECTION B - inverse-sampling geometry, with the two routes separated
--------------------------------------------------------------------
At fixed N and FIXED Gaussian eps, a sample is z(c) = mu(c) + T sigma(c) eps and
x_gen(c) = f^-1(z(c), c). The condition therefore reaches the sample by two routes:

    prior route   z(c) changes          -> measured by  f^-1(z(c_hi), c_lo) vs f^-1(z(c_lo), c_lo)
    map route     f^-1(., c) changes    -> measured by  f^-1(z(c_lo), c_hi) vs f^-1(z(c_lo), c_lo)
    full                                  f^-1(z(c_hi), c_hi) vs f^-1(z(c_lo), c_lo)

The 2x2 cross-evaluation attributes the realised embedding displacement to the two
routes and shows whether they reinforce or cancel. The displacement is then compared
against the REQUIRED displacement between real encoded embeddings at the same N and the
same two QED bands, and projected onto a within-N QED-predictive direction fitted on
TRAIN embeddings only. Forward/inverse round-trip error is reported on the same latents,
so numerical inversion is measured rather than assumed.

SECTION C - does the movement survive the decoder and the assembler?
--------------------------------------------------------------------
The same generated embeddings are decoded with the constrained and the degree decoder,
and ten RDKit descriptors are correlated against the requested QED. A working
conditioner produces a chemically coherent program (QED co-moves with TPSA, HBA, rings);
an incoherent or single-descriptor response localises the signal loss. A decoder CONTROL
runs the identical measurement on REAL embeddings from the low/high QED bands, which
bounds how much of any loss is attributable to the decoder rather than to the flow.

Exit codes: 0 ok; 2 missing input; 3 autoencoder/vocabulary mismatch; 4 bad arguments;
5 the decomposition identity failed; 1 unexpected.
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
    batch_condition,
    condition_embedding_for,
    load_condition_scaler,
    standardise_condition,
)
from zinc_gnf.dataset import (  # noqa: E402
    MolecularGraphDataset,
    iter_eval_batches,
    load_training_splits,
    seeded_subset_indices,
)
from zinc_gnf.flow import node_mask_from_counts  # noqa: E402
from zinc_gnf.trainer_autoencoder import ResumeMismatchError, load_autoencoder  # noqa: E402
from zinc_gnf.trainer_flow import (  # noqa: E402
    VALIDATION_NOISE_SEED,
    VALIDATION_SUBSET_SEED,
    file_sha256,
    load_flow,
)
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402

# The identity is exact in exact arithmetic; in float64 the residual is round-off on
# sums of order 1e4, so this bound is ~1e-9 relative.
IDENTITY_TOLERANCE = 1e-5
BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261107
PROBE_SEED = 20261108


# --------------------------------------------------------------------------- utils ---
def bootstrap_ci(values: np.ndarray, *, level: float = 0.95,
                 resamples: int = BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Percentile CI for the MEAN of per-molecule values."""
    array = np.asarray([v for v in np.ravel(values) if np.isfinite(v)], dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": float("nan"), "ci95": [float("nan"), float("nan")],
                "excludes_zero": False}
    if array.size == 1:
        return {"n": 1, "mean": float(array[0]), "ci95": [float("nan"), float("nan")],
                "excludes_zero": False}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, array.size, size=(resamples, array.size))
    means = array[draws].mean(axis=1)
    low, high = float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
    return {"n": int(array.size), "mean": float(array.mean()),
            "sd": float(array.std(ddof=1)), "ci95": [low, high],
            "excludes_zero": bool(low > 0.0 or high < 0.0)}


def masked_norm(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-molecule L2 norm over real nodes -> [B]."""
    weighted = x * mask.unsqueeze(-1).to(x.dtype)
    return weighted.flatten(1).norm(dim=1)


def per_node_rms(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-molecule RMS per node-dimension, so sizes are comparable across N."""
    weighted = x * mask.unsqueeze(-1).to(x.dtype)
    counts = mask.sum(-1).clamp(min=1).to(x.dtype) * x.shape[-1]
    return (weighted.pow(2).flatten(1).sum(dim=1) / counts).sqrt()


# ----------------------------------------------------------------- section A ---
@torch.no_grad()
def decompose(flow, autoencoder, dataset, indices, *, device, batch_size, protocol,
              qed_offset, scaler, noise_seed=VALIDATION_NOISE_SEED,
              shuffle_seed=31337) -> dict[str, Any]:
    """Exact three-way decomposition of the QED-only conditional likelihood gap.

    `protocol='permute-qed'` reproduces the existing QED-only permutation control (each
    molecule keeps its true N and receives ANOTHER molecule's QED). `protocol='offset'`
    shifts standardised QED by a fixed amount, which is the protocol the generation grid
    corresponds to. Both are provided because the two answer different questions and the
    published control numbers use the permutation one.
    """
    features = flow.config.condition_features
    if "qed" not in features:
        raise ValueError("this flow does not condition on QED")
    qed_col = features.index("qed")
    rng = np.random.default_rng(shuffle_seed)
    generator = torch.Generator(device=device).manual_seed(int(noise_seed))

    rows: list[dict[str, float]] = []
    worst_residual = 0.0
    for batch in iter_eval_batches(dataset, batch_size, indices):
        batch = batch.to(device)
        if batch.batch_size < 2:
            continue
        clean = autoencoder.encode(batch)
        # The smoothing noise is drawn in float32 from the same seed as the recorded
        # controls (clean is float32 here, so randn uses float32), then the arithmetic is
        # promoted to float64 because the flow has been cast. That is deliberate: the
        # NOISE VALUES match the float32 control stream so the total gap is comparable,
        # while the decomposition arithmetic is float64 so the identity holds at
        # round-off. Expect agreement with the recorded gap to ~1e-6 relative, not exactly.
        x = flow.add_smoothing_noise(clean, batch.node_mask, generator=generator).double()
        mask = batch.node_mask
        values, present = batch_condition(batch, features)
        values = values.double()
        present = present.double()

        alt = values.clone()
        if protocol == "permute-qed":
            order = rng.permutation(batch.batch_size)
            if np.array_equal(order, np.arange(batch.batch_size)):
                order = np.roll(order, 1)
            alt[:, qed_col] = values[torch.as_tensor(order.copy(), device=device), qed_col]
        elif protocol == "offset":
            alt[:, qed_col] = values[:, qed_col] + float(qed_offset)
        else:
            raise ValueError("protocol must be 'permute-qed' or 'offset'")

        cond_t = condition_embedding_for(flow, values, present, batch.batch_size, device)
        cond_a = condition_embedding_for(flow, alt, present, batch.batch_size, device)

        z_t, logdet_t = flow(x, mask, cond_t)
        z_a, logdet_a = flow(x, mask, cond_a)
        lp_t_zt = flow.prior.log_prob(z_t, mask, cond_t)
        lp_t_za = flow.prior.log_prob(z_a, mask, cond_t)
        lp_a_za = flow.prior.log_prob(z_a, mask, cond_a)

        d_z = lp_t_zt - lp_t_za
        d_prior = lp_t_za - lp_a_za
        d_logdet = logdet_t - logdet_a
        d_total = (lp_t_zt + logdet_t) - (lp_a_za + logdet_a)
        residual = (d_total - (d_z + d_prior + d_logdet)).abs()
        worst_residual = max(worst_residual, float(residual.max()))

        nodes = mask.sum(-1).clamp(min=1).double()
        separation = (alt[:, qed_col] - values[:, qed_col]).abs()
        for i in range(batch.batch_size):
            rows.append({
                "n": int(batch.n_node[i]),
                "qed_true": float(batch.qed[i]),
                "qed_separation_std": float(separation[i]),
                "d_z": float(d_z[i] / nodes[i]),
                "d_prior": float(d_prior[i] / nodes[i]),
                "d_logdet": float(d_logdet[i] / nodes[i]),
                "d_total": float(d_total[i] / nodes[i]),
                "residual": float(residual[i] / nodes[i]),
            })

    if not rows:
        raise ValueError("no molecules were scored")
    if worst_residual > IDENTITY_TOLERANCE:
        raise FloatingPointError(
            f"the three-way decomposition did not reconstruct the total: worst absolute "
            f"residual {worst_residual:.3e} exceeds {IDENTITY_TOLERANCE:.1e}. The "
            f"decomposition is reported only when the identity holds.")

    def summarise(subset: list[dict[str, float]]) -> dict[str, Any]:
        if not subset:
            return {"molecules": 0}
        out: dict[str, Any] = {"molecules": len(subset)}
        for key in ("d_total", "d_z", "d_prior", "d_logdet"):
            out[key] = bootstrap_ci(np.array([r[key] for r in subset]))
        out["mean_qed_separation_std"] = float(np.mean([r["qed_separation_std"] for r in subset]))
        # the three signed terms may cancel; this records how much cancellation occurs
        magnitude = float(np.mean([abs(r["d_z"]) + abs(r["d_prior"]) + abs(r["d_logdet"])
                                   for r in subset]))
        out["mean_absolute_term_sum"] = magnitude
        out["net_over_absolute"] = (abs(out["d_total"]["mean"]) / magnitude
                                    if magnitude > 0 else float("nan"))
        return out

    result: dict[str, Any] = {
        "protocol": protocol,
        "qed_offset_standardised": float(qed_offset) if protocol == "offset" else None,
        "identity_worst_absolute_residual": worst_residual,
        "identity_tolerance": IDENTITY_TOLERANCE,
        "identity_verified": True,
        "dtype": "float64",
        "noise_seed": int(noise_seed),
        "overall": summarise(rows),
        "by_n": {},
        "interpretation": (
            "signed nats/node with 95% bootstrap CIs over molecules. The terms may "
            "cancel, so percentages of the total are NOT reported. A large d_z means the "
            "condition reaches the forward map; it does not by itself imply a generative "
            "failure, because a conditional translation of z that the prior mean tracks "
            "is correct conditioning. d_logdet is the only term that cannot move a "
            "low-temperature sample's location. Section B decides what actually moves."),
    }
    for target in sorted({r["n"] for r in rows}):
        subset = [r for r in rows if r["n"] == target]
        if len(subset) >= 5:
            result["by_n"][str(target)] = summarise(subset)
    return result


# ----------------------------------------------------------------- section B ---
@torch.no_grad()
def geometry(flow, autoencoder, dataset, *, device, scaler, target_n, qed_values,
             samples, temperature, eps_seed, probe_direction=None,
             real_bands=None) -> dict[str, Any]:
    """Route-separated inverse-sampling geometry at fixed N with COMMON Gaussian eps."""
    features = flow.config.condition_features
    dim = int(flow.config.node_embedding_dim)
    mask = torch.ones((samples, target_n), dtype=torch.bool, device=device)
    generator = torch.Generator(device=device).manual_seed(int(eps_seed))
    eps = torch.randn((samples, target_n, dim), device=device, generator=generator)

    conditions, mus, sigmas, zs, gens = {}, {}, {}, {}, {}
    for q in qed_values:
        raw = standardise_condition(features, scaler=scaler, n=[target_n] * samples,
                                    qed=[q] * samples)
        values = torch.from_numpy(raw).to(device)
        cond = condition_embedding_for(flow, values, torch.ones_like(values), samples, device)
        mu, sigma = flow.prior.parameters_for(samples, cond, device)
        z = (mu.unsqueeze(1) + float(temperature) * sigma.unsqueeze(1) * eps) * mask.unsqueeze(-1)
        conditions[q], mus[q], sigmas[q] = cond, mu, sigma
        zs[q] = z
        gens[q] = flow.inverse(z, mask, cond)

    low, high = min(qed_values), max(qed_values)
    # 2x2 cross-evaluation: which route moves the sample?
    prior_route = flow.inverse(zs[high], mask, conditions[low])   # z moved, map fixed
    map_route = flow.inverse(zs[low], mask, conditions[high])     # map moved, z fixed

    full_delta = gens[high] - gens[low]
    prior_delta = prior_route - gens[low]
    map_delta = map_route - gens[low]

    def stat(delta: torch.Tensor) -> dict[str, Any]:
        return {"per_node_rms_mean": float(per_node_rms(delta, mask).mean()),
                "l2_mean": float(masked_norm(delta, mask).mean())}

    # additivity of the two routes (non-linear map, so this is a measurement not a law)
    residual_route = full_delta - (prior_delta + map_delta)
    cos_routes = float(torch.nn.functional.cosine_similarity(
        prior_delta.flatten(1), map_delta.flatten(1), dim=1).mean())

    # round-trip on the SAME latents actually used for sampling
    z_round, _ = flow(gens[low], mask, conditions[low])
    round_trip = float((z_round - zs[low]).abs().max())
    scale = float(zs[low].abs().max().clamp(min=1e-30))

    result: dict[str, Any] = {
        "n": int(target_n), "samples": int(samples), "temperature": float(temperature),
        "eps_seed": int(eps_seed), "qed_low": float(low), "qed_high": float(high),
        "prior": {
            "mu_per_dim_abs_delta_mean": float((mus[high] - mus[low]).abs().mean()),
            "mu_l2_delta": float((mus[high] - mus[low]).norm(dim=-1).mean()),
            "sigma_per_dim_abs_delta_mean": float((sigmas[high] - sigmas[low]).abs().mean()),
            "sigma_mean_low": float(sigmas[low].mean()),
            "sigma_mean_high": float(sigmas[high].mean()),
            # the mean shift expressed in units of the ACTUAL sampling spread
            "mu_delta_over_sampling_std": float(
                (mus[high] - mus[low]).norm(dim=-1).mean()
                / (float(temperature) * sigmas[low].norm(dim=-1).mean()).clamp(min=1e-30)),
        },
        "latent_delta": stat(zs[high] - zs[low]),
        "embedding_delta_full": stat(full_delta),
        "embedding_delta_prior_route": stat(prior_delta),
        "embedding_delta_map_route": stat(map_delta),
        "embedding_delta_route_residual": stat(residual_route),
        "route_cosine_prior_vs_map": cos_routes,
        "round_trip_z_x_z_max_absolute": round_trip,
        "round_trip_reference_scale": scale,
        "round_trip_max_relative": round_trip / scale,
    }

    # projection onto a within-N QED-predictive direction fitted on TRAIN embeddings
    if probe_direction is not None:
        w = probe_direction / max(float(np.linalg.norm(probe_direction)), 1e-12)
        w_t = torch.from_numpy(w).to(device=device, dtype=full_delta.dtype)
        def project(delta: torch.Tensor) -> float:
            pooled = (delta * mask.unsqueeze(-1).to(delta.dtype)).sum(1) \
                / mask.sum(-1, keepdim=True).to(delta.dtype)
            return float((pooled @ w_t).mean())
        result["projection_on_real_qed_direction"] = {
            "full": project(full_delta),
            "prior_route": project(prior_delta),
            "map_route": project(map_delta),
            "note": ("signed projection of the mean-pooled embedding displacement onto a "
                     "within-N QED-predictive direction fitted on TRAIN embeddings only. "
                     "Positive means the sample moved the way real higher-QED molecules "
                     "differ from lower-QED ones at this N."),
        }

    # required displacement between REAL encoded embeddings at the same N and bands
    if real_bands is not None:
        low_mean, high_mean, spread = real_bands
        required = float(np.linalg.norm(high_mean - low_mean))
        pooled_full = ((full_delta * mask.unsqueeze(-1).to(full_delta.dtype)).sum(1)
                       / mask.sum(-1, keepdim=True).to(full_delta.dtype)).mean(0)
        delivered_vec = pooled_full.detach().cpu().numpy().astype(np.float64)
        delivered = float(np.linalg.norm(delivered_vec))
        required_vec = (high_mean - low_mean).astype(np.float64)
        denom = max(np.linalg.norm(delivered_vec) * np.linalg.norm(required_vec), 1e-30)
        result["real_reference"] = {
            "required_l2": required,
            "delivered_l2": delivered,
            "delivered_over_required": delivered / max(required, 1e-30),
            "cosine_delivered_required": float(delivered_vec @ required_vec / denom),
            "within_band_spread_l2": float(spread),
            "delivered_over_within_band_spread": delivered / max(float(spread), 1e-30),
            "note": ("required = distance between mean REAL encoded embeddings of the two "
                     "QED bands at this N; spread = within-band dispersion, the scale at "
                     "which a displacement is distinguishable at all."),
        }
    return result


@torch.no_grad()
def real_band_statistics(autoencoder, dataset, *, device, target_n, qed_low, qed_high,
                         halfwidth, batch_size, max_molecules) -> tuple | None:
    """Mean pooled REAL embedding of each QED band at fixed N, plus within-band spread."""
    pooled: dict[str, list[np.ndarray]] = {"low": [], "high": []}
    for label, centre in (("low", qed_low), ("high", qed_high)):
        eligible = np.flatnonzero((dataset.n_node == target_n)
                                 & (np.abs(dataset.qed - centre) <= halfwidth))
        if eligible.size == 0:
            return None
        chosen = eligible[:max_molecules].tolist()
        for batch in iter_eval_batches(dataset, batch_size, chosen):
            batch = batch.to(device)
            nodes = autoencoder.encode(batch)
            m = batch.node_mask.unsqueeze(-1).to(nodes.dtype)
            mean = (nodes * m).sum(1) / batch.node_mask.sum(-1, keepdim=True).to(nodes.dtype)
            pooled[label].append(mean.detach().cpu().numpy().astype(np.float64))
    low = np.concatenate(pooled["low"])
    high = np.concatenate(pooled["high"])
    if low.shape[0] < 2 or high.shape[0] < 2:
        return None
    spread = 0.5 * (np.linalg.norm(low.std(0)) + np.linalg.norm(high.std(0)))
    return low.mean(0), high.mean(0), spread


@torch.no_grad()
def fit_within_n_qed_direction(autoencoder, train_set, *, device, target_n, batch_size,
                               max_molecules, alpha=1.0) -> np.ndarray | None:
    """Ridge direction predicting within-N centred QED from pooled TRAIN embeddings.

    TRAIN ONLY. This is an instrument for reading displacements, never a generator.
    """
    eligible = np.flatnonzero(train_set.n_node == target_n)
    if eligible.size < 50:
        return None
    rng = np.random.default_rng(PROBE_SEED)
    chosen = rng.permutation(eligible)[:max_molecules].tolist()
    feats, qeds = [], []
    for batch in iter_eval_batches(train_set, batch_size, chosen):
        batch = batch.to(device)
        nodes = autoencoder.encode(batch)
        m = batch.node_mask.unsqueeze(-1).to(nodes.dtype)
        mean = (nodes * m).sum(1) / batch.node_mask.sum(-1, keepdim=True).to(nodes.dtype)
        feats.append(mean.detach().cpu().numpy().astype(np.float64))
        qeds.append(batch.qed.detach().cpu().numpy().astype(np.float64))
    x = np.concatenate(feats)
    y = np.concatenate(qeds)
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean()
    gram = x.T @ x + alpha * np.eye(x.shape[1])
    try:
        return np.linalg.solve(gram, x.T @ y)
    except np.linalg.LinAlgError:
        return None


# ----------------------------------------------------------------- section C ---
def descriptors(mol) -> dict[str, float]:
    from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors
    return {
        "qed": float(QED.qed(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "mw": float(Descriptors.MolWt(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "hba": float(rdMolDescriptors.CalcNumHBA(mol)),
        "hbd": float(rdMolDescriptors.CalcNumHBD(mol)),
        "rings": float(rdMolDescriptors.CalcNumRings(mol)),
        "aromatic_rings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "frac_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
    }


DESCRIPTOR_KEYS = ("qed", "logp", "mw", "tpsa", "hba", "hbd", "rings",
                   "aromatic_rings", "frac_csp3", "heavy_atoms")


@torch.no_grad()
def descriptor_coherence(flow, autoencoder, *, device, scaler, target_n, qed_values,
                         samples, temperature, eps_seed, decoders) -> dict[str, Any]:
    """Decode the SAME generated embeddings and correlate descriptors with requested QED."""
    from rdkit import Chem, rdBase
    from zinc_gnf.generation import decode_embeddings

    features = flow.config.condition_features
    dim = int(flow.config.node_embedding_dim)
    mask = torch.ones((samples, target_n), dtype=torch.bool, device=device)
    generator = torch.Generator(device=device).manual_seed(int(eps_seed))
    eps = torch.randn((samples, target_n, dim), device=device, generator=generator)

    per_decoder: dict[str, list[dict[str, Any]]] = {d: [] for d in decoders}
    for q in qed_values:
        raw = standardise_condition(features, scaler=scaler, n=[target_n] * samples,
                                    qed=[q] * samples)
        values = torch.from_numpy(raw).to(device)
        cond = condition_embedding_for(flow, values, torch.ones_like(values), samples, device)
        mu, sigma = flow.prior.parameters_for(samples, cond, device)
        z = (mu.unsqueeze(1) + float(temperature) * sigma.unsqueeze(1) * eps) * mask.unsqueeze(-1)
        x_gen = flow.inverse(z, mask, cond)
        records = decode_embeddings(autoencoder, x_gen, [target_n] * samples,
                                    decoders=tuple(decoders))
        for name in decoders:
            for record in records:
                entry = record.get(name) or {}
                row: dict[str, Any] = {"requested_qed": float(q),
                                       "valid": bool(entry.get("valid")),
                                       "fallback": bool(entry.get("fallback", False))}
                smiles = entry.get("smiles")
                if entry.get("valid") and smiles:
                    with rdBase.BlockLogs():
                        mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        row.update(descriptors(mol))
                per_decoder[name].append(row)

    out: dict[str, Any] = {"n": int(target_n), "qed_values": [float(q) for q in qed_values],
                           "samples_per_cell": int(samples), "decoders": {}}
    for name, rows in per_decoder.items():
        valid = [r for r in rows if r.get("valid") and "qed" in r]
        block: dict[str, Any] = {
            "attempts": len(rows),
            "valid": len(valid),
            "strict_validity_over_attempts": len(valid) / max(1, len(rows)),
            "fallback_rate": sum(r["fallback"] for r in rows) / max(1, len(rows)),
        }
        if len(valid) >= 6:
            requested = np.array([r["requested_qed"] for r in valid], dtype=np.float64)
            correlations: dict[str, Any] = {}
            for key in DESCRIPTOR_KEYS:
                series = np.array([r[key] for r in valid], dtype=np.float64)
                if series.std() < 1e-12 or requested.std() < 1e-12:
                    correlations[key] = None
                    continue
                order_r = np.argsort(np.argsort(requested)).astype(np.float64)
                order_s = np.argsort(np.argsort(series)).astype(np.float64)
                correlations[key] = {
                    "pearson": float(np.corrcoef(requested, series)[0, 1]),
                    "spearman": float(np.corrcoef(order_r, order_s)[0, 1]),
                    "mean": float(series.mean()),
                }
            block["descriptor_correlations"] = correlations
            # cell means, which is what the paired experiment reports
            block["cell_means"] = {
                f"{q:.2f}": {
                    "molecules": int(sum(1 for r in valid if r["requested_qed"] == q)),
                    "qed": float(np.mean([r["qed"] for r in valid if r["requested_qed"] == q]))
                    if any(r["requested_qed"] == q for r in valid) else None,
                    "aromatic_rings": float(np.mean(
                        [r["aromatic_rings"] for r in valid if r["requested_qed"] == q]))
                    if any(r["requested_qed"] == q for r in valid) else None,
                }
                for q in qed_values}
            block["note"] = ("a working conditioner produces a chemically COHERENT program: "
                             "QED co-moving with TPSA / HBA / rings. A near-zero response "
                             "across every descriptor means the signal never reached "
                             "chemistry; a single dominant nuisance descriptor (e.g. "
                             "heavy_atoms) means it was absorbed by the wrong feature.")
        out["decoders"][name] = block
    return out


@torch.no_grad()
def decoder_control(autoencoder, dataset, *, device, target_n, qed_low, qed_high,
                    halfwidth, batch_size, max_molecules, decoders) -> dict[str, Any]:
    """Does the FROZEN decoder preserve a QED gap it is GIVEN, on REAL embeddings?

    This bounds how much of any loss in section C is attributable to the decoder and the
    assembler rather than to the flow: real embeddings are on-manifold by construction.
    """
    from rdkit import Chem, rdBase
    from zinc_gnf.generation import decode_embeddings

    out: dict[str, Any] = {"n": int(target_n), "decoders": {}}
    bands: dict[str, dict[str, Any]] = {}
    for label, centre in (("low", qed_low), ("high", qed_high)):
        eligible = np.flatnonzero((dataset.n_node == target_n)
                                 & (np.abs(dataset.qed - centre) <= halfwidth))
        if eligible.size == 0:
            out["note"] = f"no validation support at N={target_n}, QED~{centre}"
            return out
        chosen = eligible[:max_molecules].tolist()
        true_qed, decoded = [], {d: [] for d in decoders}
        for batch in iter_eval_batches(dataset, batch_size, chosen):
            batch = batch.to(device)
            nodes = autoencoder.encode(batch)
            counts = [int(v) for v in batch.n_node.tolist()]
            records = decode_embeddings(autoencoder, nodes, counts, decoders=tuple(decoders))
            true_qed.extend(float(v) for v in batch.qed.detach().cpu())
            for name in decoders:
                for record in records:
                    entry = record.get(name) or {}
                    if entry.get("valid") and entry.get("smiles"):
                        with rdBase.BlockLogs():
                            mol = Chem.MolFromSmiles(entry["smiles"])
                        decoded[name].append(float(descriptors(mol)["qed"])
                                             if mol is not None else float("nan"))
                    else:
                        decoded[name].append(float("nan"))
        bands[label] = {"true_qed_mean": float(np.mean(true_qed)),
                        "molecules": len(true_qed),
                        "decoded": {d: decoded[d] for d in decoders}}

    true_gap = bands["high"]["true_qed_mean"] - bands["low"]["true_qed_mean"]
    for name in decoders:
        hi = np.array(bands["high"]["decoded"][name], dtype=np.float64)
        lo = np.array(bands["low"]["decoded"][name], dtype=np.float64)
        hi, lo = hi[np.isfinite(hi)], lo[np.isfinite(lo)]
        if hi.size < 2 or lo.size < 2:
            out["decoders"][name] = {"note": "too few valid reconstructions"}
            continue
        decoded_gap = float(hi.mean() - lo.mean())
        out["decoders"][name] = {
            "true_qed_gap": float(true_gap),
            "decoded_qed_gap": decoded_gap,
            "preserved_fraction": decoded_gap / true_gap if abs(true_gap) > 1e-9 else float("nan"),
            "valid_low": int(lo.size), "valid_high": int(hi.size),
            "note": ("preserved_fraction near 1 means the decoder and assembler transmit a "
                     "QED difference they are GIVEN on real embeddings, so a loss observed "
                     "on GENERATED embeddings cannot be blamed on the decoder alone."),
        }
    return out


# --------------------------------------------------------------------------- main ---
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flow-checkpoint", required=True, action="append",
                   help="repeatable: label=path, or just path")
    p.add_argument("--ae-checkpoint", default=None)
    p.add_argument("--data-dir", default="data/processed/zinc250k_charge_aware_moflow_fixed")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target-n", default="15,23,30")
    p.add_argument("--qed-values", default="0.4,0.6,0.8")
    p.add_argument("--decomposition-molecules", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--samples-per-cell", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--qed-offset", type=float, default=1.0,
                   help="standardised-QED offset for the 'offset' protocol")
    p.add_argument("--qed-halfwidth", type=float, default=0.05)
    p.add_argument("--real-molecules-per-band", type=int, default=128)
    p.add_argument("--probe-molecules", type=int, default=2000)
    p.add_argument("--eps-seed", type=int, default=20261107)
    p.add_argument("--decoders", default="constrained,degree")
    p.add_argument("--skip-descriptors", action="store_true")
    p.add_argument("--allow-ae-mismatch", action="store_true")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        targets = [int(v) for v in args.target_n.split(",") if v.strip()]
        qed_values = [float(v) for v in args.qed_values.split(",") if v.strip()]
        decoders = [d.strip() for d in args.decoders.split(",") if d.strip()]
        if len(qed_values) < 2 or not targets:
            raise ValueError("need at least two QED values and one N")
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

        models: list[tuple[str, Path]] = []
        for item in args.flow_checkpoint:
            label, _, path = item.partition("=")
            models.append((label, Path(path)) if path else (Path(item).stem, Path(item)))

        # one autoencoder, taken from the FIRST checkpoint's provenance unless overridden
        first_flow, first_probe = load_flow(models[0][1], device=device)
        recorded = first_probe.get("autoencoder") or {}
        ae_path = Path(args.ae_checkpoint or recorded.get("checkpoint", ""))
        if not ae_path.is_file():
            raise FileNotFoundError(f"autoencoder not found: {ae_path}; pass --ae-checkpoint")
        digest = file_sha256(ae_path)
        if recorded.get("sha256") and digest != recorded["sha256"] and not args.allow_ae_mismatch:
            raise ResumeMismatchError(
                f"autoencoder mismatch: checkpoint recorded {str(recorded['sha256'])[:16]}, "
                f"{ae_path} is {digest[:16]} (use --allow-ae-mismatch to override)")
        autoencoder, _ = load_autoencoder(ae_path, device=device)
        autoencoder.eval().requires_grad_(False)

        # the SAME subset seed the trainer's validation and condition_sensitivity use, so
        # the total gap here is comparable with the recorded control numbers
        indices = seeded_subset_indices(len(val_set), args.decomposition_molecules,
                                        VALIDATION_SUBSET_SEED)

        # instruments that do not depend on the flow: fitted ONCE on train/val
        probe_dirs, real_bands = {}, {}
        for n in targets:
            probe_dirs[n] = fit_within_n_qed_direction(
                autoencoder, train_set, device=device, target_n=n,
                batch_size=args.batch_size, max_molecules=args.probe_molecules)
            real_bands[n] = real_band_statistics(
                autoencoder, val_set, device=device, target_n=n,
                qed_low=min(qed_values), qed_high=max(qed_values),
                halfwidth=args.qed_halfwidth, batch_size=args.batch_size,
                max_molecules=args.real_molecules_per_band)

        payload: dict[str, Any] = {
            "purpose": ("read-only diagnostic: exact three-way likelihood-gap "
                        "decomposition, route-separated inverse-sampling geometry, and "
                        "decoder/descriptor coherence"),
            "data_dir": args.data_dir, "data_fingerprint": fingerprint,
            "autoencoder": {"checkpoint": str(ae_path), "sha256": digest},
            "validation_molecules": len(val_set),
            "decomposition_molecules": len(indices),
            "targets_n": targets, "qed_values": qed_values,
            "temperature": args.temperature, "eps_seed": args.eps_seed,
            "models": {},
        }

        for label, path in models:
            flow, probe = (first_flow, first_probe) if path == models[0][1] \
                else load_flow(path, device=device)
            flow.eval().requires_grad_(False)
            if "qed" not in flow.config.condition_features:
                payload["models"][label] = {"skipped": "flow does not condition on QED"}
                continue
            print(f"[{label}] {path} step={probe.get('step')} "
                  f"features={list(flow.config.condition_features)}", flush=True)

            block: dict[str, Any] = {
                "checkpoint": str(path), "sha256": file_sha256(path),
                "step": probe.get("step"),
                "flow_config": flow.config.to_dict(),
                "decomposition": {}, "geometry": {}, "descriptors": {}, "decoder_control": {},
            }

            # --- Section A, in float64 ---
            doubled = flow.double()
            for protocol in ("permute-qed", "offset"):
                block["decomposition"][protocol] = decompose(
                    doubled, autoencoder, val_set, indices, device=device,
                    batch_size=args.batch_size, protocol=protocol,
                    qed_offset=args.qed_offset, scaler=scaler)
                mean = block["decomposition"][protocol]["overall"]["d_total"]["mean"]
                print(f"  [{label}] {protocol}: d_total={mean:+.4f} nats/node "
                      f"(identity residual "
                      f"{block['decomposition'][protocol]['identity_worst_absolute_residual']:.2e})",
                      flush=True)
            flow = doubled.float()

            # --- Section B ---
            for n in targets:
                block["geometry"][str(n)] = geometry(
                    flow, autoencoder, val_set, device=device, scaler=scaler, target_n=n,
                    qed_values=qed_values, samples=args.samples_per_cell,
                    temperature=args.temperature, eps_seed=args.eps_seed,
                    probe_direction=probe_dirs.get(n), real_bands=real_bands.get(n))
                g = block["geometry"][str(n)]
                print(f"  [{label}] N={n}: mu_delta/sampling_std="
                      f"{g['prior']['mu_delta_over_sampling_std']:.4f} "
                      f"emb_delta(full/prior/map)="
                      f"{g['embedding_delta_full']['per_node_rms_mean']:.4f}/"
                      f"{g['embedding_delta_prior_route']['per_node_rms_mean']:.4f}/"
                      f"{g['embedding_delta_map_route']['per_node_rms_mean']:.4f}",
                      flush=True)

            # --- Section C ---
            if not args.skip_descriptors:
                for n in targets:
                    block["descriptors"][str(n)] = descriptor_coherence(
                        flow, autoencoder, device=device, scaler=scaler, target_n=n,
                        qed_values=qed_values, samples=args.samples_per_cell,
                        temperature=args.temperature, eps_seed=args.eps_seed,
                        decoders=decoders)
                    block["decoder_control"][str(n)] = decoder_control(
                        autoencoder, val_set, device=device, target_n=n,
                        qed_low=min(qed_values), qed_high=max(qed_values),
                        halfwidth=args.qed_halfwidth, batch_size=args.batch_size,
                        max_molecules=args.real_molecules_per_band, decoders=decoders)
            payload["models"][label] = block

        payload["elapsed_seconds"] = time.perf_counter() - started
        atomic_write_json(payload, out_dir / "qed_gap_decomposition.json")
        (out_dir / "qed_gap_decomposition.txt").write_text(format_report(payload))
        print(f"\nwrote {out_dir / 'qed_gap_decomposition.json'}")
        print(f"wrote {out_dir / 'qed_gap_decomposition.txt'}")
        return 0
    except FileNotFoundError as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 2
    except ResumeMismatchError as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 3
    except FloatingPointError as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 5
    except Exception:  # noqa: BLE001
        print("FATAL: unexpected error\n" + traceback.format_exc(), file=sys.stderr, flush=True)
        return 1


def format_report(payload: dict[str, Any]) -> str:
    lines = ["QED likelihood-gap decomposition and inverse-sampling geometry",
             "=" * 78, "",
             f"data: {payload['data_dir']}",
             f"autoencoder: {payload['autoencoder']['checkpoint']} "
             f"(sha {str(payload['autoencoder']['sha256'])[:16]})",
             f"decomposition molecules: {payload['decomposition_molecules']}  "
             f"temperature: {payload['temperature']}", ""]
    for label, block in payload["models"].items():
        if "skipped" in block:
            lines += [f"--- {label}: SKIPPED ({block['skipped']}) ---", ""]
            continue
        lines += [f"--- {label}  (step {block.get('step')}) ---", ""]
        for protocol, result in block["decomposition"].items():
            o = result["overall"]
            lines.append(f"  decomposition [{protocol}]  identity residual "
                         f"{result['identity_worst_absolute_residual']:.2e} "
                         f"(tol {result['identity_tolerance']:.0e}) VERIFIED")
            lines.append(f"    {'term':<12}{'mean':>10}{'95% CI':>26}")
            for key, name in (("d_total", "d_total"), ("d_z", "d_z"),
                              ("d_prior", "d_prior"), ("d_logdet", "d_logdet")):
                e = o[key]
                lines.append(f"    {name:<12}{e['mean']:>+10.4f}"
                             f"   [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]")
            lines.append(f"    sum of |terms| = {o['mean_absolute_term_sum']:.4f}; "
                         f"|net|/sum|terms| = {o['net_over_absolute']:.4f} "
                         f"(low = heavy cancellation)")
            for n, e in sorted(result["by_n"].items(), key=lambda kv: int(kv[0])):
                lines.append(f"      N={n:<4} d_total={e['d_total']['mean']:+.4f} "
                             f"d_z={e['d_z']['mean']:+.4f} "
                             f"d_prior={e['d_prior']['mean']:+.4f} "
                             f"d_logdet={e['d_logdet']['mean']:+.4f}")
            lines.append("")
        lines.append("  geometry (common eps, fixed N)")
        for n, g in sorted(block["geometry"].items(), key=lambda kv: int(kv[0])):
            lines.append(f"    N={n}: mu_delta/sampling_std="
                         f"{g['prior']['mu_delta_over_sampling_std']:.4f}  "
                         f"round_trip_rel={g['round_trip_max_relative']:.2e}")
            lines.append(f"      embedding delta per-node RMS: full="
                         f"{g['embedding_delta_full']['per_node_rms_mean']:.4f} "
                         f"prior_route={g['embedding_delta_prior_route']['per_node_rms_mean']:.4f} "
                         f"map_route={g['embedding_delta_map_route']['per_node_rms_mean']:.4f} "
                         f"cos(routes)={g['route_cosine_prior_vs_map']:+.3f}")
            if "real_reference" in g:
                r = g["real_reference"]
                lines.append(f"      vs real: delivered/required="
                             f"{r['delivered_over_required']:.4f} "
                             f"cos={r['cosine_delivered_required']:+.3f} "
                             f"delivered/within-band-spread="
                             f"{r['delivered_over_within_band_spread']:.4f}")
            if "projection_on_real_qed_direction" in g:
                p = g["projection_on_real_qed_direction"]
                lines.append(f"      projection on real QED direction: full={p['full']:+.4f} "
                             f"prior={p['prior_route']:+.4f} map={p['map_route']:+.4f}")
        lines.append("")
        if block.get("descriptors"):
            lines.append("  descriptor coherence (generated)")
            for n, d in sorted(block["descriptors"].items(), key=lambda kv: int(kv[0])):
                for name, b in d["decoders"].items():
                    corr = b.get("descriptor_correlations") or {}
                    shown = ", ".join(
                        f"{k}={corr[k]['spearman']:+.2f}" for k in DESCRIPTOR_KEYS
                        if corr.get(k) is not None)
                    lines.append(f"    N={n} [{name}] valid="
                                 f"{b['strict_validity_over_attempts']:.3f} "
                                 f"fallback={b['fallback_rate']:.3f}")
                    lines.append(f"      spearman vs requested QED: {shown}")
            lines.append("")
        if block.get("decoder_control"):
            lines.append("  decoder control (REAL embeddings: does it transmit a given gap?)")
            for n, d in sorted(block["decoder_control"].items(), key=lambda kv: int(kv[0])):
                for name, b in (d.get("decoders") or {}).items():
                    if "preserved_fraction" in b:
                        lines.append(f"    N={n} [{name}] true_gap={b['true_qed_gap']:+.4f} "
                                     f"decoded_gap={b['decoded_qed_gap']:+.4f} "
                                     f"preserved={b['preserved_fraction']:.3f}")
            lines.append("")
    lines += ["", "Reading guide", "-" * 78,
              "d_z large is NOT a failure on its own: a conditional translation of z that",
              "the prior mean tracks is correct conditioning. d_logdet is the only term",
              "that cannot move a low-temperature sample's location. The decisive numbers",
              "are the geometry block's mu_delta/sampling_std, the prior/map route split,",
              "and delivered/required against the real embedding populations."]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
