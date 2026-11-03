"""Latent QED guidance: MoFlow's property mechanism, adapted, and honestly labelled.

VERIFIED SOURCE, not a paper citation: MoFlow's `mflow/optimize_property.py` does NOT
reach a property target by conditioning. It trains `MoFlowProp.propNN`, a small MLP on
the flow's LATENT vector, to regress the property (MSE), then takes normalised gradient
steps in latent space, `z <- z + lr * g / |g|`, decodes every visited latent, checks
validity and sorts the survivors by the property measured with RDKit. Their starting
point is an ENCODED REAL MOLECULE, and the result is reported as property optimisation.

This module does the same thing for a *target* rather than a maximum, starting from our
conditional prior instead of a real molecule:

    z ~ p(z | N, QED)                      (what the model already does)
    z <- z - lr * d/dz (q_hat(z) - q*)^2   (a few normalised steps; q_hat is the predictor)
    x = f^-1(z), decode, measure QED with RDKit

Three things make this honest rather than a dressed-up filter:

1. The predictor is fitted on TRAIN latents only, and its own accuracy is reported, so a
   guided result cannot be read without knowing how good the guide is.
2. Guided sampling is always reported against unguided sampling at the SAME TOTAL NUMBER
   OF CANDIDATES. Both `select_best` (reranking k candidates by RDKit QED) and guidance
   consume a budget; comparing k=1 unguided with k=20 guided would be meaningless.
3. Nothing here is native conditioning, and `METHOD_LABELS` spells that out for every
   row a report prints. Reranking by RDKit QED is a SELECTION method: it cannot be
   presented as the flow having learned QED.

The flow is never trained or modified. Only the small predictor is fitted.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

METHOD_LABELS = {
    "native": "native conditional sampling (the model's own QED control)",
    "rerank": "SELECTION: k native candidates per slot, best RDKit QED kept "
              "(not learned conditioning)",
    "guided": "GUIDED: latent gradient steps towards the target using a latent->QED "
              "predictor (not learned conditioning)",
}
PREDICTOR_SEED = 20261021


class LatentQedPredictor(nn.Module):
    """Mean-pooled latent -> QED. Small on purpose: it is a guide, not a model of QED.

    Pooling over nodes makes the prediction permutation-invariant and size-robust, which
    matters because the latents are per-node and N varies.
    """

    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(z.dtype)
        counts = weights.sum(dim=1).clamp(min=1.0)
        mean = (z * weights).sum(dim=1) / counts
        spread = (((z - mean.unsqueeze(1)) ** 2) * weights).sum(dim=1) / counts
        features = torch.cat([mean, spread.sqrt()], dim=-1)
        return self.net(features).squeeze(-1)


def fit_latent_predictor(flow, autoencoder, dataset, *, indices: Sequence[int],
                         device: torch.device | str = "cpu", batch_size: int = 128,
                         epochs: int = 8, learning_rate: float = 1e-3,
                         validation_fraction: float = 0.2,
                         seed: int = PREDICTOR_SEED) -> tuple[LatentQedPredictor, dict[str, Any]]:
    """Fit `q_hat(z)` on TRAIN latents. Returns the predictor and its honest accuracy.

    The flow is frozen and used only in the forward direction. The held-out part of the
    fitting set gives the predictor's own MAE and correlation, which bound how much any
    guidance driven by it can be trusted.
    """
    from zinc_gnf.conditioning import batch_condition, condition_embedding_for
    from zinc_gnf.dataset import iter_eval_batches

    device = torch.device(device)
    flow.eval()
    order = [int(i) for i in indices]
    rng = np.random.default_rng(int(seed))
    rng.shuffle(order)
    split = int(len(order) * (1.0 - validation_fraction))
    parts: dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
    with torch.no_grad():
        for name, subset in (("fit", order[:split]), ("held_out", order[split:])):
            collected = []
            for batch in iter_eval_batches(dataset, batch_size, subset):
                batch = batch.to(device)
                embeddings = autoencoder.encode(batch)
                condition = None
                if flow.config.conditional:
                    values, present = batch_condition(batch, flow.config.condition_features)
                    condition = condition_embedding_for(flow, values, present,
                                                        batch.batch_size, device)
                z, _ = flow(embeddings, batch.node_mask, condition)
                collected.append((z.detach(), batch.node_mask.detach(), batch.qed.detach()))
            parts[name] = collected
    if not parts["fit"]:
        raise ValueError("no molecules to fit the latent QED predictor on")

    predictor = LatentQedPredictor(int(flow.config.node_embedding_dim)).to(device)
    optimiser = torch.optim.Adam(predictor.parameters(), lr=learning_rate)
    torch.manual_seed(int(seed))
    history = []
    for epoch in range(int(epochs)):
        predictor.train()
        losses = []
        for z, mask, qed in parts["fit"]:
            prediction = predictor(z, mask)
            loss = nn.functional.mse_loss(prediction, qed.to(prediction.dtype))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch + 1, "train_mse": float(np.mean(losses))})
    predictor.eval()

    def evaluate(name: str) -> dict[str, Any]:
        if not parts[name]:
            return {"molecules": 0, "mae": float("nan"), "correlation": float("nan")}
        predictions, truths = [], []
        with torch.no_grad():
            for z, mask, qed in parts[name]:
                predictions.append(predictor(z, mask).cpu().numpy())
                truths.append(qed.cpu().numpy())
        p = np.concatenate(predictions).astype(np.float64)
        t = np.concatenate(truths).astype(np.float64)
        correlation = (float(np.corrcoef(p, t)[0, 1])
                       if p.size > 1 and p.std() > 0 and t.std() > 0 else float("nan"))
        return {"molecules": int(p.size), "mae": float(np.abs(p - t).mean()),
                "rmse": float(np.sqrt(((p - t) ** 2).mean())), "correlation": correlation,
                "predicted_mean": float(p.mean()), "true_mean": float(t.mean())}

    report = {
        "molecules_fitted": sum(int(q.shape[0]) for _, _, q in parts["fit"]),
        "epochs": int(epochs), "history": history,
        "fit": evaluate("fit"), "held_out": evaluate("held_out"),
        "seed": int(seed),
        "note": ("this is the GUIDE's accuracy, not a model result. Guidance cannot be "
                 "trusted beyond the held-out MAE reported here."),
    }
    return predictor, report


def guide_latents(predictor: LatentQedPredictor, z: torch.Tensor, mask: torch.Tensor, *,
                  target: float, steps: int = 10, step_size: float = 0.1) -> dict[str, Any]:
    """Normalised gradient steps on `(q_hat(z) - target)^2`, MoFlow's update rule.

    MoFlow ascends the property; a TARGET needs a squared-error descent instead, so a
    request of 0.4 is not turned into "as high as possible". The step is normalised by
    the gradient norm, exactly as in their code, which makes the step size a distance in
    latent space rather than a function of the predictor's scale.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    current = z.detach().clone()
    weights = mask.unsqueeze(-1).to(z.dtype)
    trace = []
    for _ in range(int(steps)):
        current.requires_grad_(True)
        prediction = predictor(current, mask)
        loss = ((prediction - float(target)) ** 2).sum()
        gradient = torch.autograd.grad(loss, current)[0]
        norm = gradient.flatten(1).norm(dim=1).clamp(min=1e-12).view(-1, 1, 1)
        with torch.no_grad():
            current = (current - float(step_size) * gradient / norm) * weights
        trace.append({"predicted_mean": float(prediction.detach().mean()),
                      "gradient_norm_mean": float(norm.mean())})
        current = current.detach()
    with torch.no_grad():
        final = predictor(current, mask)
    return {"z": current, "predicted_qed": final.detach(), "trace": trace,
            "steps": int(steps), "step_size": float(step_size), "target": float(target)}


def select_best(records: Sequence[dict[str, Any]], *, target: float, group: int,
                decoder: str = "degree") -> list[dict[str, Any]]:
    """Reranking baseline: from each consecutive group of `group` candidates keep the one
    whose RDKit QED is closest to the target (an invalid candidate can never win).

    This is SELECTION, not conditioning, and it spends `group` candidates per kept
    molecule - which is why any comparison must fix the TOTAL candidate budget.
    """
    if group < 1:
        raise ValueError("group must be >= 1")
    kept: list[dict[str, Any]] = []
    for start in range(0, len(records), group):
        chunk = records[start:start + group]
        scored = [(abs(float((r.get(decoder) or {}).get("qed", np.nan)) - float(target)), i)
                  for i, r in enumerate(chunk)
                  if (r.get(decoder) or {}).get("valid")
                  and (r.get(decoder) or {}).get("qed") is not None]
        if not scored:
            kept.append(dict(chunk[0], selection={"group": len(chunk), "chosen": None,
                                                  "reason": "no valid candidate in group"}))
            continue
        scored.sort()
        index = scored[0][1]
        kept.append(dict(chunk[index], selection={"group": len(chunk), "chosen": index,
                                                  "candidates_valid": len(scored)}))
    return kept


def equal_budget_note(count: int, *, group: int, steps: int) -> dict[str, Any]:
    """The budget accounting a comparison must state to be meaningful."""
    return {
        "kept_molecules": int(count),
        "native_candidates": int(count),
        "rerank_candidates": int(count * group),
        "guided_candidates": int(count),
        "guided_extra_flow_inverses": 0,
        "guided_extra_predictor_passes": int(count * steps),
        "note": ("reranking spends `group` decoded candidates per kept molecule; guidance "
                 "spends one decode plus `steps` cheap predictor passes. A fair table "
                 "fixes the number of DECODED candidates, which is what costs time, and "
                 "reports native sampling at the same total."),
    }


__all__ = [
    "METHOD_LABELS",
    "PREDICTOR_SEED",
    "LatentQedPredictor",
    "equal_budget_note",
    "fit_latent_predictor",
    "guide_latents",
    "select_best",
]
