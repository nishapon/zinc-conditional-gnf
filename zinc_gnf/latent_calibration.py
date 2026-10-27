"""Post-hoc latent recalibration of a trained flow's prior.

WHY THIS EXISTS
---------------
A normalizing flow is trained to push the data through `f` so that `z = f(x)` looks
like its prior. Nothing guarantees it got there. If the latents the trained flow
actually produces are, say, 1.5x wider per dimension than the prior it is sampled
from, then every generated sample is drawn from the WRONG region of latent space,
`f^-1` is evaluated off the manifold it was fitted on, and the decoded graphs
degrade - even though the flow itself is perfectly fine.

The teammates measured exactly this failure on QM9 (their
`molecular_generation/inspect_latent_calibration.py` and `latent_stats.py`): the
learned sigma(N) underestimated the empirical latent spread by about 1.5x, and
widening sigma at sampling time alone recovered uniqueness at N=5. That diagnostic
is per-dimension, which matters: a single pooled std divided by the mean of the
per-dimension sigmas is inflated whenever sigma varies across dimensions, even for a
perfectly calibrated prior.

WHAT THIS DOES
--------------
1. `latent_statistics` measures, per latent dimension and streaming (no big buffer),
   the mean and std of `z = f(x + smoothing noise)` over REAL nodes of TRAINING
   molecules, plus the teammates' calibration diagnostics against the model's own
   prior: `zscore_rms` (1.0 = calibrated, 1.5 = data 1.5x wider), kurtosis,
   `abs_ratio` and bias.

2. `calibrate_prior` installs those statistics as the prior
   (`NodePrior.set_calibration`), so BOTH `log_prob` and `sample` use them.

This is not a heuristic sampling trick. For a diagonal Gaussian prior family, the
maximum-likelihood parameters of the observed latents are exactly their empirical
per-dimension mean and std, and the trained prior is a member of that same family.
So on the measurement data the calibrated log-likelihood is >= the trained one, with
equality only if the flow was already calibrated: the change cannot make the model
worse by its own training objective, and it is a no-op for a converged flow. Unlike a
generation-time `sigma_scale`, the model stays a normalized density that can be
scored, and the improvement (or its absence) is a measurable number, not a guess.

It applies to an EXISTING checkpoint. No retraining, no architecture change, and the
frozen autoencoder is untouched.

Scope and honesty: this fixes a prior/latent mismatch. It does not by itself teach
the flow inter-node dependence, and on the teammates' own ZINC run calibration plus a
low sampling temperature still left atom composition and aromaticity unmatched. The
report below is written so the effect can be read off, including when it is zero.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from zinc_gnf.dataset import MolecularGraphDataset, iter_eval_batches
from zinc_gnf.flow import NodeFlow

# Fixed seeds, so a calibration run is reproducible and independent of the
# training/validation noise streams.
CALIBRATION_NOISE_SEED = 90210
CALIBRATION_SUBSET_SEED = 54321

GAUSSIAN_ABS_RATIO = 0.7978845608028654       # E|u| / sqrt(E u^2) for a Gaussian
GAUSSIAN_KURTOSIS = 3.0


@torch.no_grad()
def latent_statistics(flow: NodeFlow, autoencoder, dataset: MolecularGraphDataset, *,
                      indices: Sequence[int] | None = None,
                      device: torch.device | str = "cpu",
                      batch_size: int = 256,
                      noise_seed: int = CALIBRATION_NOISE_SEED) -> dict[str, Any]:
    """Per-dimension statistics of `z = f(x + noise)` on real nodes.

    `x` is the frozen encoder's embedding and the noise is the flow's own smoothing
    noise, so the measured density is the one the flow was trained on - measuring
    noise-free embeddings would describe a different (singular) target.

    Accumulation is in float64 over first, second, absolute and fourth moments, so
    the result does not depend on batch size or molecule order.
    """
    was_training = flow.training
    flow.eval()
    dimension = int(flow.config.node_embedding_dim)
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(int(noise_seed))
    # The unconditional prior is one constant (mu, sigma) pair, so the z-scores
    # u = (z - mu) / sigma can be formed inside the loop and their moments accumulated
    # exactly - no raw-moment expansion, and the tail statistics stay correct even if
    # training moved mu away from zero.
    prior_mu, prior_sigma = flow.prior.trained_parameters_for(1, None, device)
    prior_mu = prior_mu[0].detach().to(torch.float64)
    prior_sigma = prior_sigma[0].detach().to(torch.float64)

    zero = lambda: torch.zeros(dimension, dtype=torch.float64, device=device)  # noqa: E731
    s1, s2 = zero(), zero()                       # moments of z
    u1, u2, ua, u4 = zero(), zero(), zero(), zero()   # moments of u
    nodes = 0
    molecules = 0
    for batch in iter_eval_batches(dataset, int(batch_size), indices):
        batch = batch.to(device)
        embeddings = autoencoder.encode(batch)
        embeddings = flow.add_smoothing_noise(embeddings, batch.node_mask, generator=generator)
        z, _ = flow(embeddings, batch.node_mask)
        keep = batch.node_mask.reshape(-1)
        flat = z.reshape(-1, dimension).to(torch.float64)[keep]
        s1 += flat.sum(0)
        s2 += (flat * flat).sum(0)
        u = (flat - prior_mu) / prior_sigma
        u1 += u.sum(0)
        u2 += (u * u).sum(0)
        ua += u.abs().sum(0)
        u4 += (u ** 4).sum(0)
        nodes += int(keep.sum())
        molecules += int(batch.batch_size)
    if nodes < 2:
        raise ValueError("need at least two real nodes to measure latent statistics")
    if was_training:
        flow.train()

    count = float(nodes)
    mean = s1 / count
    std = (s2 / count - mean * mean).clamp(min=0.0).sqrt()

    second_u = (u2 / count).clamp(min=0.0)
    ratio = second_u.sqrt()               # the likelihood-optimal sigma, in sigma units
    bias = u1 / count
    tail = {
        "abs_ratio_mean": float(((ua / count) / ratio.clamp(min=1e-12)).mean()),
        "kurtosis_mean": float(((u4 / count) / second_u.clamp(min=1e-12) ** 2).mean()),
        "kurtosis_median": float(((u4 / count) / second_u.clamp(min=1e-12) ** 2).median()),
    }
    return {
        "molecules": molecules,
        "nodes": nodes,
        "mean": mean.to(torch.float32).cpu(),
        "std": std.to(torch.float32).cpu(),
        "prior_mu": prior_mu.to(torch.float32).cpu(),
        "prior_sigma": prior_sigma.to(torch.float32).cpu(),
        "noise_seed": int(noise_seed),
        "summary": {
            "latent_std_mean": float(std.mean()),
            "latent_std_min": float(std.min()),
            "latent_std_max": float(std.max()),
            "latent_mean_abs_mean": float(mean.abs().mean()),
            "prior_sigma_mean": float(prior_sigma.mean()),
            "prior_sigma_min": float(prior_sigma.min()),
            "prior_sigma_max": float(prior_sigma.max()),
            # 1.0 = calibrated; 1.5 = latents 1.5x wider than the prior assumes.
            "zscore_rms": float(ratio.pow(2).mean().sqrt()),
            "zscore_rms_dim_median": float(ratio.median()),
            "zscore_rms_dim_min": float(ratio.min()),
            "zscore_rms_dim_max": float(ratio.max()),
            "fraction_dims_wider_than_1_25": float((ratio > 1.25).to(torch.float64).mean()),
            "fraction_dims_narrower_than_0_8": float((ratio < 0.8).to(torch.float64).mean()),
            "bias_abs_mean": float(bias.abs().mean()),
            "prior_mu_abs_mean": float(prior_mu.abs().mean()),
            **tail,
        },
    }


def calibrate_prior(flow: NodeFlow, statistics: dict[str, Any], *,
                    std_floor: float = 1e-3) -> dict[str, Any]:
    """Install measured latent statistics as the flow's prior. Returns a note."""
    floored = flow.prior.set_calibration(statistics["mean"], statistics["std"],
                                         std_floor=std_floor)
    return {
        "calibrated": True,
        "std_floor": float(std_floor),
        "floored_dimensions": floored,
        "measured_on": {"molecules": statistics["molecules"], "nodes": statistics["nodes"]},
        "noise_seed": statistics["noise_seed"],
    }


@torch.no_grad()
def dataset_negative_log_likelihood(flow: NodeFlow, autoencoder,
                                    dataset: MolecularGraphDataset, *,
                                    indices: Sequence[int] | None = None,
                                    device: torch.device | str = "cpu",
                                    batch_size: int = 256,
                                    noise_seed: int = CALIBRATION_NOISE_SEED) -> dict[str, float]:
    """Per-node NLL in nats over `indices`, with a fixed smoothing-noise stream.

    Called once before and once after calibration with the SAME seed, so the two
    numbers differ only by the prior.
    """
    was_training = flow.training
    flow.eval()
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(int(noise_seed))
    total_log_prob = 0.0
    total_log_prob_z = 0.0
    nodes = 0
    molecules = 0
    for batch in iter_eval_batches(dataset, int(batch_size), indices):
        batch = batch.to(device)
        embeddings = autoencoder.encode(batch)
        embeddings = flow.add_smoothing_noise(embeddings, batch.node_mask, generator=generator)
        parts = flow.log_prob(embeddings, batch.node_mask)
        total_log_prob += float(parts["log_prob"].sum())
        total_log_prob_z += float(parts["log_prob_z"].sum())
        nodes += int(batch.node_mask.sum())
        molecules += int(batch.batch_size)
    if was_training:
        flow.train()
    if nodes == 0:
        raise ValueError("no real nodes to score")
    return {
        "nll_per_node": -total_log_prob / nodes,
        "log_prob_z_per_node": total_log_prob_z / nodes,
        "molecules": molecules,
        "nodes": nodes,
    }


def format_calibration(statistics: dict[str, Any], *, title: str = "Latent calibration") -> str:
    """Readable report of the measurement (the ZINC analogue of the team's Figure 6)."""
    s = statistics["summary"]
    lines = [title, "=" * len(title), ""]
    lines.append(f"measured on {statistics['molecules']:,} training molecules "
                 f"({statistics['nodes']:,} real nodes), smoothing-noise seed "
                 f"{statistics['noise_seed']}")
    lines.append("")
    lines.append("prior the flow was trained with:")
    lines.append(f"  sigma  mean {s['prior_sigma_mean']:.4f}  "
                 f"min {s['prior_sigma_min']:.4f}  max {s['prior_sigma_max']:.4f}")
    lines.append("latents the flow actually produces:")
    lines.append(f"  std    mean {s['latent_std_mean']:.4f}  "
                 f"min {s['latent_std_min']:.4f}  max {s['latent_std_max']:.4f}")
    lines.append(f"  |mean| mean {s['latent_mean_abs_mean']:.4f}")
    lines.append("")
    lines.append("calibration (1.0 = prior matches the latents, 1.5 = latents 1.5x wider):")
    lines.append(f"  zscore_rms (all dims)      {s['zscore_rms']:.4f}")
    lines.append(f"  per-dimension median       {s['zscore_rms_dim_median']:.4f}  "
                 f"[min {s['zscore_rms_dim_min']:.4f}, max {s['zscore_rms_dim_max']:.4f}]")
    lines.append(f"  dims wider than 1.25       {s['fraction_dims_wider_than_1_25'] * 100:.1f}%")
    lines.append(f"  dims narrower than 0.80    {s['fraction_dims_narrower_than_0_8'] * 100:.1f}%")
    lines.append(f"  |bias| mean (sigma units)  {s['bias_abs_mean']:.4f}")
    lines.append(f"  kurtosis mean / median     {s['kurtosis_mean']:.3f} / "
                 f"{s['kurtosis_median']:.3f}  (Gaussian {GAUSSIAN_KURTOSIS:.1f}; "
                 f"larger = heavier tails)")
    lines.append(f"  E|u|/rms(u)                {s['abs_ratio_mean']:.4f}  "
                 f"(Gaussian {GAUSSIAN_ABS_RATIO:.4f}; smaller = heavier tails)")
    lines.append("")
    lines.append("A calibrated prior cannot fix heavy tails (kurtosis >> 3) or inter-node "
                 "dependence; it only matches the per-dimension location and spread.")
    return "\n".join(lines)


__all__ = [
    "CALIBRATION_NOISE_SEED",
    "CALIBRATION_SUBSET_SEED",
    "GAUSSIAN_ABS_RATIO",
    "GAUSSIAN_KURTOSIS",
    "calibrate_prior",
    "dataset_negative_log_likelihood",
    "format_calibration",
    "latent_statistics",
]
