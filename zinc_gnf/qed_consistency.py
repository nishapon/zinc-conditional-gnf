"""Optional conditional QED-consistency auxiliary objective (TRAINING ONLY).

WHAT THIS IS
------------
A training-time auxiliary loss that teaches the flow's *generation path* to realise the
requested QED. It adds NO parameters to `NodeFlow` and does not change the density: the
surrogate it needs lives in the trainer and is saved under a separate checkpoint key, so
a flow trained with this objective is a byte-for-byte drop-in for every existing
generation and evaluation script (`load_flow` rebuilds only the `NodeFlow`). With the
weight at 0 (the default everywhere) training is identical to the baseline.

WHY IT TARGETS THE OBSERVED FAILURE
-----------------------------------
The joint (N, QED) flow carries QED in its LIKELIHOOD (permutation gap ~ +0.077
nats/node, win rate ~68%) yet its sampled QED barely follows the request (128D seed-3
paired slope +0.007, negative at N=23). Maximum likelihood scores a FORWARD pass on real
embeddings; it never scores WHERE the inverse map lands, so nothing in the objective ties
"the condition c I asked for" to "the QED of what I sampled at c". This loss supplies
exactly that term:

    z ~ N(mu(c), (T*sigma(c))^2)          reparameterised, so gradients reach mu, sigma
    x_gen = f^{-1}(z, c)                   the inverse used at generation time
    q_hat = g_phi( pool(x_gen) )           a QED read-out of the generated embeddings
    L_cons = MSE( q_hat, qed(c) )          qed(c) is the standardised QED already in c

The gradient of `L_cons` w.r.t. the flow reaches it through BOTH the conditional prior
`mu(c), sigma(c)` and the inverse coupling / FiLM layers, i.e. every place conditioning
can move a sample. Minimising it makes "change the requested QED" move the generated
embeddings in the direction a QED read-out recognises -- the missing response.

THE SURROGATE, AND THE OFF-MANIFOLD RISK (stated and mitigated)
---------------------------------------------------------------
`g_phi` is a small MLP on the SAME pooled features as the stage-19/20 probe
(`zinc_gnf.qed_probes.pooled_features`), regressed on the molecule's own standardised QED
using REAL (clean, un-noised) encoder embeddings only. Two properties make the gradient
trustworthy and keep it from simply pushing samples off-manifold to fool the reader:

  * the surrogate is calibrated ONLY on real embeddings, where the AE reconstructs QED at
    MAE ~0.005, so it is a faithful QED reader exactly on the manifold the flow should
    stay on;
  * inside the consistency term the surrogate's parameters are held constant (the caller
    sets `requires_grad=False` for the duration), so the FLOW is moved toward the reader,
    never the reader toward the flow. The surrogate is updated by a SEPARATE optimiser on
    real embeddings only.

Because `L_cons` is small, warmed up, and added on top of an unchanged NLL term that
keeps the forward map honest, pulling samples toward where a real-calibrated reader is
confident pulls them toward the real embedding region, not away from it. Everything here
is logged separately from the NLL so the trade-off is visible, and the unchanged
`val_condition_gap` continues to report whether conditioning is alive.

NO DATA LEAKAGE
---------------
The surrogate sees TRAIN embeddings only and is never used in evaluation or generation.
It is a disposable training aid, not part of the generative model.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from zinc_gnf.flow import NodeFlow


def pooled_mean_std(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean and std over the REAL nodes -> `[B, 2*D]`.

    Deliberately identical to `zinc_gnf.qed_probes.pooled_features` (asserted in the
    tests), so the surrogate reads embeddings exactly the way the stage-20 probe does.
    """
    weights = mask.unsqueeze(-1).to(x.dtype)
    counts = weights.sum(dim=1).clamp(min=1.0)
    mean = (x * weights).sum(dim=1) / counts
    spread = (((x - mean.unsqueeze(1)) ** 2) * weights).sum(dim=1) / counts
    return torch.cat([mean, spread.sqrt()], dim=-1)


class EmbeddingQedSurrogate(nn.Module):
    """Small MLP: pooled node embeddings -> predicted STANDARDISED QED (a scalar per graph).

    Width-invariant through the masked pooling, so it handles any molecule size. Trained
    only on real embeddings; see the module docstring.
    """

    def __init__(self, node_embedding_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * int(node_embedding_dim), int(hidden_dim)), nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)), nn.ReLU(),
            nn.Linear(int(hidden_dim), 1))
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.net(pooled_mean_std(x, mask)).squeeze(-1)       # [B]


def consistency_weight(step: int, *, weight: float, warmup: int, ramp: int) -> float:
    """The schedule for the auxiliary weight.

    0 until `warmup` steps have passed (so the flow settles into a good density and the
    surrogate calibrates first), then a linear ramp to `weight` over `ramp` steps, then
    constant. A clean gradient pathway with no sudden shock to the NLL objective.
    """
    if weight <= 0 or step < warmup:
        return 0.0
    if ramp <= 0:
        return float(weight)
    return float(weight) * min(1.0, (step - warmup + 1) / ramp)


def surrogate_regression_loss(surrogate: EmbeddingQedSurrogate, x_real_clean: torch.Tensor,
                              mask: torch.Tensor, qed_std: torch.Tensor) -> torch.Tensor:
    """MSE of the surrogate's standardised-QED prediction on REAL (clean) embeddings.

    `x_real_clean` must be detached from the flow graph by the caller: calibrating the
    surrogate must never send gradients into the frozen encoder or the flow.
    """
    return F.mse_loss(surrogate(x_real_clean, mask), qed_std)


def conditional_consistency_loss(flow: NodeFlow, surrogate: EmbeddingQedSurrogate, *,
                                 condition: torch.Tensor, mask: torch.Tensor,
                                 qed_std_target: torch.Tensor, temperature: float = 1.0,
                                 generator: torch.Generator | None = None
                                 ) -> tuple[torch.Tensor, dict[str, Any]]:
    """Sample at `condition`, invert, read QED, and penalise the gap to `qed_std_target`.

    The reparameterised draw mirrors `NodePrior.sample` exactly (same mu + T*sigma*eps,
    same masking), so this is the generation path with gradients kept. The surrogate's
    parameters must be frozen by the caller (`requires_grad=False`) for the duration, so
    the gradient updates the FLOW only; it reaches the flow through `mu(c)`, `sigma(c)`
    and the inverse coupling / FiLM layers.
    """
    if condition is None:
        raise ValueError("conditional_consistency_loss needs a conditioning embedding")
    batch, width = mask.shape
    dim = int(flow.config.node_embedding_dim)
    mu, sigma = flow.prior.parameters_for(batch, condition, mask.device)
    eps = torch.randn((batch, width, dim), device=mask.device, dtype=mu.dtype,
                      generator=generator)
    z = mu.unsqueeze(1) + float(temperature) * sigma.unsqueeze(1) * eps
    z = z * mask.unsqueeze(-1).to(z.dtype)
    x_gen = flow.inverse(z, mask, condition)
    prediction = surrogate(x_gen, mask)
    loss = F.mse_loss(prediction, qed_std_target)
    diagnostics = {
        "qed_consistency_mse": float(loss.detach()),
        "qed_consistency_pred_mean": float(prediction.detach().mean()),
        "qed_consistency_pred_std": float(prediction.detach().std()) if batch > 1 else 0.0,
        "qed_consistency_target_mean": float(qed_std_target.detach().mean()),
    }
    return loss, diagnostics


__all__ = [
    "EmbeddingQedSurrogate",
    "conditional_consistency_loss",
    "consistency_weight",
    "pooled_mean_std",
    "surrogate_regression_loss",
]
