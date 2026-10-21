"""Conditioning: wiring the dataset's condition vector into the flow, and PROVING it matters.

The architecture has been conditional since `zinc_gnf/flow.py` was written (FiLM in every
coupling, a learned `mu(c), sigma(c)` prior with zero-initialised heads, and a
`ConditionEmbedding` with per-feature present flags). What was missing is the plumbing:
the trainer never read `MolecularBatch.condition`, and the sampler refused conditional
flows. This module is that plumbing, plus the measurements that decide whether the
conditioning is real.

Feature order is fixed and shared by the data and the model. The preprocessing stores
`condition = [standardised log1p(N), standardised QED]`, standardised with a TRAIN-ONLY
scaler (`<data_dir>/scaler.json`); `flow.CONDITION_FEATURES = ("n", "qed")` names the same
two columns in the same order. `condition_columns` is the single place that mapping lives,
so an N-only model takes column 0 and an N+QED model takes both - and a future QED-only
model would take column 1 without any other change.

WHY THE VERIFICATION FUNCTIONS EXIST
------------------------------------
A conditional run that merely finishes, or that produces molecules with the requested atom
count, is NOT evidence that conditioning was learned. In this pipeline N also sets how many
node vectors are drawn, and the decoder emits exactly one atom per node vector, so the
atom count is satisfied by CONSTRUCTION even if `mu(N), sigma(N)` and every FiLM layer were
dead. Two cheap, decisive measurements are provided instead:

  `condition_sensitivity`  the same validation molecules scored twice with the same
      smoothing noise: once with their own conditions, once with the conditions permuted
      across molecules (same marginal distribution of conditions, wrong pairing). A model
      that ignores its conditioning gets the identical likelihood; a model that uses it is
      strictly worse under the permutation. The gap, in nats per node, is the evidence.

  `prior_response`  `mu(c), sigma(c)` evaluated along a sweep of requested N. If the
      conditional prior is dead, the curve is flat and the spread is 0 - visible before a
      single molecule is decoded.

`latent_spread_by_n` additionally reports, per N, the empirical latent std against the
model's own `sigma(N)` (the conditional analogue of stage 10's `zscore_rms`). It is a
DIAGNOSTIC only: no calibration is installed here, and none should be unless this number
is far from 1.0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from zinc_gnf.dataset import MolecularBatch, MolecularGraphDataset, iter_eval_batches
from zinc_gnf.flow import CONDITION_FEATURES, NodeFlow
from zinc_gnf.preprocessing import CONDITION_NAMES, ConditionScaler

# The dataset's condition vector column for each model-side feature name.
CONDITION_COLUMN: dict[str, int] = {"n": 0, "qed": 1}
assert tuple(CONDITION_COLUMN) == CONDITION_FEATURES, "feature names must match flow.py"
assert CONDITION_NAMES == ("log1p_n_node", "qed"), "dataset column order changed"

CONDITION_SHUFFLE_SEED = 31337


def condition_columns(features: Sequence[str]) -> list[int]:
    """Dataset condition-vector columns for the model's `condition_features`."""
    unknown = [f for f in features if f not in CONDITION_COLUMN]
    if unknown:
        raise ValueError(f"unknown condition features {unknown}; known: {list(CONDITION_COLUMN)}")
    return [CONDITION_COLUMN[f] for f in features]


def batch_condition(batch: MolecularBatch, features: Sequence[str]
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """`(values, present)` for one batch: each molecule's OWN standardised conditions.

    Every feature is present during training - this is plain conditional maximum
    likelihood, not classifier-free guidance - but the present flags are passed
    explicitly so that dropping a feature later needs no change here.
    """
    features = tuple(features)
    if not features:
        raise ValueError("batch_condition called for an unconditional model")
    if batch.condition is None:
        raise ValueError(
            "this dataset carries no condition vectors, so a conditional flow cannot be "
            "trained on it; re-run the preprocessing (it writes the train-only scaler and "
            "a condition vector per molecule)")
    columns = condition_columns(features)
    values = batch.condition[:, columns].to(torch.float32)
    if not torch.isfinite(values).all():
        raise ValueError("non-finite condition values in the batch (a NaN QED label?)")
    return values, torch.ones_like(values)


def load_condition_scaler(data_dir: str | Path) -> ConditionScaler:
    """The TRAIN-ONLY scaler the preprocessing fitted, from `<data_dir>/scaler.json`."""
    path = Path(data_dir) / "scaler.json"
    if not path.exists():
        raise FileNotFoundError(
            f"condition scaler not found: {path}. Sampling a conditional flow needs the same "
            f"train-only standardisation the model was trained with.")
    payload = json.loads(path.read_text())
    names = tuple(payload.get("names", ()))
    if names != CONDITION_NAMES:
        raise ValueError(f"{path}: scaler names {names} != {CONDITION_NAMES}")
    if payload.get("fit_on") != "train_only":
        raise ValueError(f"{path}: scaler was not fitted on the training split only")
    mean, std = payload["mean"], payload["std"]
    return ConditionScaler(mean=(float(mean[0]), float(mean[1])),
                           std=(float(std[0]), float(std[1])), names=CONDITION_NAMES)


def standardise_condition(features: Sequence[str], *, scaler: ConditionScaler,
                          n: Sequence[int] | int, qed: Sequence[float] | float | None = None
                          ) -> np.ndarray:
    """Requested targets -> `[B, len(features)]` standardised values the model expects.

    The SAME scaler the training data used, so a requested N means at sampling time
    exactly what it meant at training time. QED is required if 'qed' is a feature; N is
    always required, because the scaler standardises the two jointly and N also fixes how
    many node vectors are drawn.
    """
    features = tuple(features)
    counts = np.atleast_1d(np.asarray(n, dtype=np.int64))
    if (counts < 1).any():
        raise ValueError("every requested N must be >= 1")
    if "qed" in features:
        if qed is None:
            raise ValueError("this model conditions on QED; a qed target is required")
        qed_values = np.atleast_1d(np.asarray(qed, dtype=np.float64))
        if qed_values.size == 1 and counts.size > 1:
            qed_values = np.full(counts.shape, float(qed_values[0]))
        if qed_values.shape != counts.shape:
            raise ValueError("n and qed targets must have the same length")
        if not np.isfinite(qed_values).all():
            raise ValueError("qed targets must be finite")
    else:
        # QED is not a feature, so its column is never read; 0.0 keeps `transform` happy.
        qed_values = np.zeros(counts.shape, dtype=np.float64)
    columns = condition_columns(features)
    rows = np.stack([scaler.transform(float(q), int(c)) for c, q in zip(counts, qed_values)])
    return rows[:, columns].astype(np.float32)


def condition_embedding_for(flow: NodeFlow, values: torch.Tensor | None,
                            present: torch.Tensor | None, batch_size: int,
                            device: torch.device | str) -> torch.Tensor | None:
    """`flow.conditioning` with the shape check a caller would otherwise forget."""
    if not flow.config.conditional:
        if values is not None:
            raise ValueError("this flow is unconditional but condition values were given")
        return None
    if values is None:
        raise ValueError("this flow is conditional; condition values are required")
    width = len(flow.config.condition_features)
    if values.dim() != 2 or values.shape[1] != width:
        raise ValueError(f"condition values must be [B, {width}], got {tuple(values.shape)}")
    if values.shape[0] != batch_size:
        raise ValueError(f"condition values have {values.shape[0]} rows for a batch of "
                         f"{batch_size}")
    return flow.conditioning(values.to(device), None if present is None else present.to(device),
                             batch_size, device)


# ---------------------------------------------------------------------------
# Does the conditioning do anything? (Requirement: a finished run is not evidence.)
# ---------------------------------------------------------------------------
@torch.no_grad()
def condition_sensitivity(flow: NodeFlow, autoencoder, dataset: MolecularGraphDataset, *,
                          indices: Sequence[int] | None = None,
                          device: torch.device | str = "cpu",
                          batch_size: int = 128,
                          noise_seed: int = 777,
                          shuffle_seed: int = CONDITION_SHUFFLE_SEED) -> dict[str, Any]:
    """Likelihood with the TRUE conditions vs with the conditions permuted across molecules.

    Identical molecules, identical smoothing noise, identical weights: the only difference
    is which condition vector each molecule is scored against. The permutation keeps the
    marginal distribution of the conditions exactly, so a model that ignores conditioning
    scores the two identically (gap 0) and a model that uses it is worse under the
    permutation (gap > 0). The gap is reported in nats per node, with the per-molecule
    win rate, which does not depend on a few outliers.
    """
    if not flow.config.conditional:
        raise ValueError("condition_sensitivity is only meaningful for a conditional flow")
    features = flow.config.condition_features
    was_training = flow.training
    flow.eval()
    device = torch.device(device)
    true_noise = torch.Generator(device=device).manual_seed(int(noise_seed))
    shuffle_rng = np.random.default_rng(int(shuffle_seed))
    totals = {"true": 0.0, "shuffled": 0.0}
    nodes = molecules = wins = ties = 0
    per_molecule_gap: list[float] = []
    for batch in iter_eval_batches(dataset, int(batch_size), indices):
        batch = batch.to(device)
        if batch.batch_size < 2:
            continue                     # a permutation of one molecule is the identity
        embeddings = autoencoder.encode(batch)
        embeddings = flow.add_smoothing_noise(embeddings, batch.node_mask, generator=true_noise)
        values, present = batch_condition(batch, features)
        order = shuffle_rng.permutation(batch.batch_size)
        # Guarantee a genuine derangement-like permutation: if the draw happens to be the
        # identity the "control" would be no control at all.
        if np.array_equal(order, np.arange(batch.batch_size)):
            order = np.roll(order, 1)
        permuted = values[torch.as_tensor(order.copy(), device=values.device)]
        counts = batch.node_mask.sum(-1).to(torch.float64)
        results = {}
        for name, condition_values in (("true", values), ("shuffled", permuted)):
            embedding = condition_embedding_for(flow, condition_values, present,
                                                batch.batch_size, device)
            parts = flow.log_prob(embeddings, batch.node_mask, embedding)
            results[name] = parts["log_prob"].to(torch.float64)
            totals[name] += float(parts["log_prob"].sum())
        gap = ((results["true"] - results["shuffled"]) / counts).cpu().numpy()
        per_molecule_gap.extend(float(g) for g in gap)
        wins += int((gap > 0).sum())
        ties += int((gap == 0).sum())
        nodes += int(batch.node_mask.sum())
        molecules += batch.batch_size
    if was_training:
        flow.train()
    if molecules == 0:
        raise ValueError("need at least two molecules in a batch to permute the conditions")
    true_nll = -totals["true"] / nodes
    shuffled_nll = -totals["shuffled"] / nodes
    gaps = np.asarray(per_molecule_gap, dtype=np.float64)
    return {
        "molecules": molecules,
        "nodes": nodes,
        "features": list(features),
        "nll_per_node_true": true_nll,
        "nll_per_node_shuffled": shuffled_nll,
        # > 0 means the true conditions explain the data better than permuted ones.
        "gap_nats_per_node": shuffled_nll - true_nll,
        "per_molecule_gap_mean": float(gaps.mean()),
        "per_molecule_gap_median": float(np.median(gaps)),
        "molecules_better_with_true_condition": wins / molecules,
        "exact_ties": ties / molecules,
        "noise_seed": int(noise_seed),
        "shuffle_seed": int(shuffle_seed),
        "interpretation": ("gap > 0 and a win rate well above 0.5 mean the conditioning is "
                           "used; a gap of exactly 0 with ties = 1.0 means it is ignored"),
    }


@torch.no_grad()
def prior_response(flow: NodeFlow, *, scaler: ConditionScaler, n_values: Sequence[int],
                   qed: float | None = None, device: torch.device | str = "cpu") -> dict[str, Any]:
    """`mu(c), sigma(c)` along a sweep of requested N. A flat curve = dead conditioning."""
    if not flow.config.conditional:
        raise ValueError("prior_response is only meaningful for a conditional flow")
    features = flow.config.condition_features
    device = torch.device(device)
    values = torch.from_numpy(standardise_condition(features, scaler=scaler,
                                                    n=list(n_values), qed=qed)).to(device)
    embedding = condition_embedding_for(flow, values, torch.ones_like(values),
                                        len(list(n_values)), device)
    mu, sigma = flow.prior.parameters_for(values.shape[0], embedding, device)
    mu, sigma = mu.detach().cpu(), sigma.detach().cpu()
    rows = []
    for position, count in enumerate(n_values):
        rows.append({"n": int(count),
                     "mu_mean": float(mu[position].mean()),
                     "mu_norm": float(mu[position].norm()),
                     "sigma_mean": float(sigma[position].mean()),
                     "sigma_min": float(sigma[position].min()),
                     "sigma_max": float(sigma[position].max())})
    return {
        "rows": rows,
        # Spread ACROSS the requested N values, per latent dimension, then averaged: this
        # is the number that is 0 for a prior that ignores N.
        "mu_spread_across_n": float(mu.std(dim=0).mean()) if mu.shape[0] > 1 else float("nan"),
        "sigma_spread_across_n": float(sigma.std(dim=0).mean()) if sigma.shape[0] > 1
        else float("nan"),
        "mu_max_abs_difference": float((mu.max(dim=0).values - mu.min(dim=0).values).max())
        if mu.shape[0] > 1 else float("nan"),
        "qed_target": qed,
    }


@torch.no_grad()
def latent_spread_by_n(flow: NodeFlow, autoencoder, dataset: MolecularGraphDataset, *,
                       n_values: Sequence[int], indices: Sequence[int] | None = None,
                       device: torch.device | str = "cpu", batch_size: int = 128,
                       noise_seed: int = 777, min_nodes: int = 200) -> dict[str, Any]:
    """Per-N empirical latent spread against the model's own sigma(N). Diagnostic only.

    `zscore_rms` here is the conditional analogue of stage 10's statistic: 1.0 means the
    conditional prior's width matches the latents the flow produces for molecules of that
    size, 1.5 would mean the latents are 1.5x wider than sigma(N) assumes. Nothing is
    installed; this only tells us whether a conditional calibration is NEEDED.
    """
    if not flow.config.conditional:
        raise ValueError("latent_spread_by_n is only meaningful for a conditional flow")
    features = flow.config.condition_features
    wanted = {int(n) for n in n_values}
    was_training = flow.training
    flow.eval()
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(int(noise_seed))
    dimension = int(flow.config.node_embedding_dim)
    sums: dict[int, torch.Tensor] = {}
    squares: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {}
    sigmas: dict[int, torch.Tensor] = {}
    for batch in iter_eval_batches(dataset, int(batch_size), indices):
        batch = batch.to(device)
        keep_molecule = torch.as_tensor([int(n) in wanted for n in batch.n_node.tolist()],
                                        device=device)
        if not bool(keep_molecule.any()):
            continue
        embeddings = autoencoder.encode(batch)
        embeddings = flow.add_smoothing_noise(embeddings, batch.node_mask, generator=generator)
        values, present = batch_condition(batch, features)
        embedding = condition_embedding_for(flow, values, present, batch.batch_size, device)
        z, _ = flow(embeddings, batch.node_mask, embedding)
        mu, sigma = flow.prior.parameters_for(batch.batch_size, embedding, device)
        u = (z - mu.unsqueeze(1)) / sigma.unsqueeze(1)
        for position, count in enumerate(batch.n_node.tolist()):
            count = int(count)
            if count not in wanted:
                continue
            rows = u[position][batch.node_mask[position]].to(torch.float64)
            sums[count] = sums.get(count, torch.zeros(dimension, dtype=torch.float64,
                                                      device=device)) + rows.sum(0)
            squares[count] = squares.get(count, torch.zeros(dimension, dtype=torch.float64,
                                                            device=device)) + (rows ** 2).sum(0)
            counts[count] = counts.get(count, 0) + int(rows.shape[0])
            sigmas.setdefault(count, sigma[position].detach().to(torch.float64))
    if was_training:
        flow.train()
    rows = []
    for count in sorted(counts):
        if counts[count] < min_nodes:
            rows.append({"n": count, "nodes": counts[count], "zscore_rms": None,
                         "note": f"fewer than {min_nodes} nodes; not reported"})
            continue
        second = squares[count] / counts[count]
        rows.append({"n": count, "nodes": counts[count],
                     "zscore_rms": float(second.mean().clamp(min=0.0).sqrt()),
                     "bias_abs_mean": float((sums[count] / counts[count]).abs().mean()),
                     "sigma_mean": float(sigmas[count].mean())})
    reported = [r["zscore_rms"] for r in rows if r.get("zscore_rms") is not None]
    return {
        "rows": rows,
        "zscore_rms_mean": float(np.mean(reported)) if reported else float("nan"),
        "zscore_rms_max": float(np.max(reported)) if reported else float("nan"),
        "noise_seed": int(noise_seed),
        "note": ("diagnostic only: no calibration is installed for a conditional prior. "
                 "A value near 1.0 means none is needed."),
    }


def format_conditioning_check(sensitivity: dict[str, Any], response: dict[str, Any],
                              spread: dict[str, Any] | None = None) -> str:
    """Readable verdict on whether the conditioning is real."""
    lines = ["Conditioning check", "==================", ""]
    lines.append(f"features: {', '.join(sensitivity['features'])}")
    lines.append("")
    lines.append("1. Likelihood with TRUE vs PERMUTED conditions "
                 f"({sensitivity['molecules']:,} validation molecules, "
                 f"{sensitivity['nodes']:,} nodes)")
    lines.append(f"   NLL/node, true conditions      {sensitivity['nll_per_node_true']:.4f}")
    lines.append(f"   NLL/node, permuted conditions  {sensitivity['nll_per_node_shuffled']:.4f}")
    lines.append(f"   gap (positive = conditioning used)  {sensitivity['gap_nats_per_node']:+.4f} nats/node")
    lines.append(f"   molecules better with their own condition  "
                 f"{sensitivity['molecules_better_with_true_condition'] * 100:.1f}%")
    lines.append(f"   exact ties (identical likelihood)          "
                 f"{sensitivity['exact_ties'] * 100:.1f}%")
    lines.append("")
    lines.append("2. Conditional prior response to the requested N")
    lines.append(f"   mu spread across N     {response['mu_spread_across_n']:.4f}")
    lines.append(f"   sigma spread across N  {response['sigma_spread_across_n']:.4f}")
    for row in response["rows"]:
        lines.append(f"     N={row['n']:>3}  |mu|={row['mu_norm']:.3f}  "
                     f"sigma_mean={row['sigma_mean']:.3f}")
    if spread is not None:
        lines.append("")
        lines.append("3. Per-N latent spread against the model's own sigma(N) (diagnostic)")
        for row in spread["rows"]:
            if row.get("zscore_rms") is None:
                lines.append(f"     N={row['n']:>3}  {row.get('note', 'not reported')}")
            else:
                lines.append(f"     N={row['n']:>3}  zscore_rms={row['zscore_rms']:.3f}  "
                             f"({row['nodes']:,} nodes)")
        lines.append(f"   {spread['note']}")
    lines.append("")
    lines.append("A finished training run, or a molecule with the requested atom count, is NOT "
                 "evidence of conditioning: N also sets how many node vectors are drawn and the "
                 "decoder emits one atom per vector, so the atom count is satisfied by "
                 "construction. Section 1 is the evidence.")
    return "\n".join(lines)


__all__ = [
    "CONDITION_COLUMN",
    "CONDITION_SHUFFLE_SEED",
    "batch_condition",
    "condition_columns",
    "condition_embedding_for",
    "condition_sensitivity",
    "format_conditioning_check",
    "latent_spread_by_n",
    "load_condition_scaler",
    "prior_response",
    "standardise_condition",
]
