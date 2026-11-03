"""Is the conditional signal big enough to be SEEN, in the space the decoder reads?

THE PUZZLE STAGE 17 LEFT US
---------------------------
Across three models the QED-only likelihood gap is essentially constant (+0.0741 warm
start, +0.0768 original, +0.0867 control) while the sampled QED response differs by 4.5x
(slope 0.0315 / 0.0901 / 0.1409). So QED demonstrably enters the DENSITY in all three,
and yet at N=23 the warm start returns 0.5633 / 0.5611 / 0.5704 for requests 0.4 / 0.6 /
0.8. A likelihood gap is a statement about scoring; generation is a statement about where
the samples land. Those are different quantities, and this module measures the second.

WHAT IS MEASURED
----------------
Sampling conditions on c in exactly two places: the prior `N(mu(c), sigma(c)^2 T^2)` and
FiLM inside `f^-1`. So the conditional effect on a sample is bounded by how far the
generated EMBEDDINGS move when c changes, and that has to be compared with two
references:

  delivered   || E[x | c_high] - E[x | c_low] ||  with the SAME noise draws, so the
              difference is the conditioning and not sampling variance. Measured in the
              autoencoder's embedding space, because that is what the decoder reads.

  required    || mean real embedding at (N, QED_high) - mean at (N, QED_low) ||, from
              TRAINING molecules: how far apart the two populations actually are.

  resolvable  the within-population spread of real embeddings at fixed (N, QED), i.e.
              the scale at which a displacement is distinguishable at all.

`delivered / required` is the number that decides the next experiment. If conditioning
delivers a few percent of the displacement the data demands, no amount of prior/FiLM
tuning closes it and the capacity has to come from somewhere else. If it delivers most of
it but in the wrong DIRECTION (cosine against the required direction is reported too),
that is a different failure with a different fix.

Everything here is measurement on existing checkpoints: no training, no RDKit, no
sampling temperature tricks, and non-finite samples are counted rather than silently
turned into zeros.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from zinc_gnf.conditioning import condition_embedding_for, standardise_condition
from zinc_gnf.flow import NodeFlow, node_mask_from_counts


def _finite_report(x: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    """Non-finite accounting, per molecule. Never let a NaN become a silent zero."""
    real = x * mask.unsqueeze(-1).to(x.dtype)
    finite_nodes = torch.isfinite(real).all(dim=-1) | ~mask
    finite_molecules = finite_nodes.all(dim=-1)
    return {
        "molecules": int(x.shape[0]),
        "nonfinite_molecules": int((~finite_molecules).sum()),
        "nonfinite_rate": float((~finite_molecules).to(torch.float64).mean()),
        "finite_mask": finite_molecules,
    }


@torch.no_grad()
def delivered_displacement(flow: NodeFlow, *, scaler, n: int, qed_low: float, qed_high: float,
                           count: int = 256, temperature: float = 1.0,
                           device: torch.device | str = "cpu",
                           seed: int = 17) -> dict[str, Any]:
    """How far do the generated embeddings move when ONLY the QED request changes?

    The same noise draws are reused for both conditions, so the difference is the
    conditional effect and not sampling noise. Reported in latent space (where the prior
    acts) and in embedding space (where the decoder reads), plus the non-finite rate,
    because a NaN sample is a failure and not a large displacement.
    """
    features = flow.config.condition_features
    if "qed" not in features:
        raise ValueError(f"this flow does not condition on QED (features {list(features)})")
    device = torch.device(device)
    counts = [int(n)] * int(count)
    mask = node_mask_from_counts(counts, device=device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    noise = torch.randn((count, mask.shape[1], flow.config.node_embedding_dim),
                        device=device, generator=generator)

    rows: dict[str, Any] = {}
    latents: dict[str, torch.Tensor] = {}
    embeddings: dict[str, torch.Tensor] = {}
    for label, qed in (("low", qed_low), ("high", qed_high)):
        values = torch.from_numpy(standardise_condition(
            features, scaler=scaler, n=counts, qed=[qed] * count)).to(device)
        condition = condition_embedding_for(flow, values, None, count, device)
        mu, sigma = flow.prior.parameters_for(count, condition, device)
        z = (mu.unsqueeze(1) + float(temperature) * sigma.unsqueeze(1) * noise)
        z = z * mask.unsqueeze(-1).to(z.dtype)
        x = flow.inverse(z, mask, condition)
        latents[label] = z
        embeddings[label] = x
        rows[label] = {
            "requested_qed": float(qed),
            "prior_mu_norm": float(mu[0].norm()),
            "prior_sigma_mean": float(sigma[0].mean()),
            "finite": _finite_report(x, mask),
        }

    finite = rows["low"]["finite"]["finite_mask"] & rows["high"]["finite"]["finite_mask"]
    for label in ("low", "high"):
        rows[label]["finite"].pop("finite_mask")
    if int(finite.sum()) == 0:
        return {"n": int(n), "qed_low": float(qed_low), "qed_high": float(qed_high),
                "temperature": float(temperature), "per_condition": rows,
                "usable_molecules": 0,
                "note": "every sample was non-finite; no displacement can be measured"}

    keep = finite
    real = mask[keep].unsqueeze(-1).to(torch.float64)
    nodes = real.sum().clamp(min=1.0)

    def pooled_mean(tensor: torch.Tensor) -> torch.Tensor:
        selected = tensor[keep].to(torch.float64) * real
        return selected.sum((0, 1)) / nodes

    latent_shift = pooled_mean(latents["high"]) - pooled_mean(latents["low"])
    embedding_shift = pooled_mean(embeddings["high"]) - pooled_mean(embeddings["low"])
    # Per-molecule paired difference: the conditional effect at fixed noise.
    paired = ((embeddings["high"] - embeddings["low"])[keep].to(torch.float64) * real)
    paired_norm = paired.norm(dim=-1).sum() / nodes
    spread = (embeddings["low"][keep].to(torch.float64) * real)
    spread_mean = spread.sum((0, 1)) / nodes
    within = ((spread - spread_mean) * real).norm(dim=-1).sum() / nodes
    return {
        "n": int(n), "qed_low": float(qed_low), "qed_high": float(qed_high),
        "temperature": float(temperature),
        "usable_molecules": int(keep.sum()),
        "per_condition": rows,
        "latent_mean_shift_norm": float(latent_shift.norm()),
        "embedding_mean_shift_norm": float(embedding_shift.norm()),
        "embedding_paired_shift_per_node": float(paired_norm),
        "embedding_within_condition_spread_per_node": float(within),
        # < 1 means the conditional move is smaller than the model's own sampling spread,
        # i.e. invisible in practice however real it is in the likelihood.
        "shift_over_spread": float(paired_norm / within.clamp(min=1e-12)),
        "embedding_shift_vector": [float(v) for v in embedding_shift],
    }


@torch.no_grad()
def required_displacement(autoencoder, dataset, *, n: int, qed_low: float, qed_high: float,
                          window: float = 0.05, max_molecules: int = 512,
                          device: torch.device | str = "cpu") -> dict[str, Any]:
    """How far apart are the REAL embedding populations at the two QED values?

    Training molecules only, at exactly this N, within +-`window` of each requested QED.
    This is the displacement the data demands; `delivered_displacement` is what the model
    produces.
    """
    device = torch.device(device)
    counts = np.asarray(dataset.n_node)
    qed = np.asarray(dataset.qed, dtype=np.float64)
    result: dict[str, Any] = {"n": int(n), "qed_low": float(qed_low),
                              "qed_high": float(qed_high), "window": float(window)}
    means: dict[str, torch.Tensor] = {}
    spreads: dict[str, float] = {}
    for label, target in (("low", qed_low), ("high", qed_high)):
        selected = np.flatnonzero((counts == int(n)) & (np.abs(qed - target) <= window))
        result[f"{label}_molecules"] = int(selected.size)
        if selected.size == 0:
            continue
        chosen = selected[:max_molecules]
        from zinc_gnf.dataset import iter_eval_batches

        pooled, total = None, 0.0
        vectors: list[torch.Tensor] = []
        for batch in iter_eval_batches(dataset, 64, [int(i) for i in chosen]):
            batch = batch.to(device)
            nodes = autoencoder.encode(batch)
            real = batch.node_mask.unsqueeze(-1).to(torch.float64)
            flat = (nodes.to(torch.float64) * real)
            vectors.append(flat[batch.node_mask])
            pooled = flat.sum((0, 1)) if pooled is None else pooled + flat.sum((0, 1))
            total += float(real.sum())
        means[label] = pooled / max(total, 1.0)
        stacked = torch.cat(vectors)
        spreads[label] = float((stacked - means[label]).norm(dim=-1).mean())
    if len(means) == 2:
        result["embedding_mean_shift_norm"] = float((means["high"] - means["low"]).norm())
        result["embedding_shift_vector"] = [float(v) for v in (means["high"] - means["low"])]
        result["within_condition_spread_per_node"] = float(np.mean(list(spreads.values())))
        result["shift_over_spread"] = (result["embedding_mean_shift_norm"]
                                       / max(result["within_condition_spread_per_node"], 1e-12))
    else:
        result["note"] = "one or both QED windows are empty at this N; the target is not supported"
    return result


def signal_ratio(delivered: dict[str, Any], required: dict[str, Any]) -> dict[str, Any]:
    """delivered / required, plus the direction agreement. The decision number."""
    if delivered.get("usable_molecules", 0) == 0 or "embedding_mean_shift_norm" not in required:
        return {"ratio": float("nan"), "cosine": float("nan"),
                "verdict": "not measurable (non-finite samples, or no real molecules at "
                           "this (N, QED))"}
    produced = float(delivered["embedding_mean_shift_norm"])
    needed = float(required["embedding_mean_shift_norm"])
    a = np.asarray(delivered["embedding_shift_vector"], dtype=np.float64)
    b = np.asarray(required["embedding_shift_vector"], dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine = float(a @ b / denominator) if denominator > 0 else float("nan")
    ratio = produced / max(needed, 1e-12)
    if ratio < 0.2:
        verdict = ("the conditional move is under a fifth of what the data demands: prior "
                   "and FiLM are not delivering the displacement, so capacity - not tuning "
                   "- is the binding constraint")
    elif cosine < 0.3:
        verdict = ("the magnitude is there but the DIRECTION disagrees with the data: the "
                   "conditioning moves samples somewhere else, which tuning the same "
                   "mechanism will not fix")
    else:
        verdict = ("displacement and direction are broadly right, so the loss of control "
                   "happens after the embeddings - look at the decoder's resolution")
    return {"delivered_norm": produced, "required_norm": needed, "ratio": ratio,
            "cosine": cosine, "verdict": verdict}


def format_signal_report(rows: Sequence[dict[str, Any]]) -> str:
    lines = ["Conditional signal audit", "========================", "",
             "delivered = how far generated embeddings move when only the QED request "
             "changes (same noise)",
             "required  = how far the REAL embedding populations at those two QED values "
             "are apart (training data)", "",
             "|   N | QED low -> high | real molecules | delivered | required | ratio | "
             "cosine | shift/spread | non-finite |",
             "|----:|----------------:|---------------:|----------:|---------:|------:|"
             "-------:|-------------:|-----------:|"]
    for row in rows:
        delivered, required, ratio = row["delivered"], row["required"], row["ratio"]
        nonfinite = max(delivered["per_condition"]["low"]["finite"]["nonfinite_rate"],
                        delivered["per_condition"]["high"]["finite"]["nonfinite_rate"])
        support = min(required.get("low_molecules", 0), required.get("high_molecules", 0))

        def show(value: Any, template: str = "{:.4f}") -> str:
            return ("N/A" if value is None or (isinstance(value, float) and not np.isfinite(value))
                    else template.format(float(value)))

        lines.append(
            f"| {row['n']:>3} | {delivered['qed_low']:.2f} -> {delivered['qed_high']:.2f} | "
            f"{support:>14,} | {show(ratio.get('delivered_norm'))} | "
            f"{show(ratio.get('required_norm'))} | {show(ratio.get('ratio'), '{:.3f}')} | "
            f"{show(ratio.get('cosine'), '{:+.3f}')} | "
            f"{show(delivered.get('shift_over_spread'), '{:.3f}')} | "
            f"{nonfinite * 100:.1f}% |")
    lines += ["", "ratio = delivered / required. shift/spread < 1 means the conditional "
              "move is smaller than the model's own sampling spread at fixed condition, "
              "so it cannot be seen in individual samples however real it is in the "
              "likelihood."]
    return "\n".join(lines)


__all__ = [
    "delivered_displacement",
    "format_signal_report",
    "required_displacement",
    "signal_ratio",
]
