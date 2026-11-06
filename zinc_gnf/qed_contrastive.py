"""N-matched QED-contrastive conditional-likelihood objective (FORWARD-ONLY, training only).

This REPLACES the sampling-based `zinc_gnf.qed_consistency` objective, which was
measured to diverge on the cluster (jobs 359287 / 359294 / 360343). The root cause and
the reason this design is numerically different are documented below, because the whole
point of the redesign is the gradient path, not the loss value.

-------------------------------------------------------------------------------
1. WHY THE SAMPLING OBJECTIVE DIVERGED (evidence-grounded)
-------------------------------------------------------------------------------
The old objective backpropagated through `flow.inverse`. Three operations exist in the
INVERSE path and in NO part of the NLL path, which is exactly why the NLL stayed finite
while the generation-path gradient blew up:

  (a) `InvertiblePointwiseLinear.inverse` calls `torch.linalg.inv(W)` (flow.py:247).
      The forward only ever needs `x @ W.T` and `log|det| = sum(log_s)`. A matrix can
      therefore drift to near-singular with NO penalty in the NLL: the forward matmul is
      well behaved and the log-determinant only sees the SUM of `log_s`, not its spread.
      The gradient of a matrix inverse is  d/dW (W^-1) = -W^-T (.) W^-T, i.e. it scales
      with the SQUARE of ||W^-1||. With 12 blocks each contributing one such inverse,
      a modest per-block ill-conditioning becomes catastrophic multiplicatively.

  (b) `NodeAutoregressiveCoupling.inverse` is SEQUENTIAL (flow.py:92-95): node i is
      recovered from the already-recovered nodes, re-running the full conditioner inside
      the loop. For N nodes that is a chain of N dependent Jacobians per block. Each
      carries a factor up to exp(max_log_scale) = e^1. With N up to 38 and 12 blocks the
      backward pass multiplies O(N x blocks) ~ 456 such factors in series. float32
      overflows at e^89, so this product is not merely large, it is unrepresentable.
      The forward of the same module (flow.py:75) is ONE parallel pass - no chain.

  (c) The old surrogate pooled `spread.sqrt()`, whose derivative 1/(2*sqrt(v)) is
      unbounded as v -> 0, and symmetry-equivalent atoms give near-identical embeddings
      (v ~ 0) by construction in this encoder.

The decisive evidence that this is STRUCTURAL and not a step-size problem: lowering the
auxiliary weight 0.5 -> 0.25 and the temperature 1.0 -> 0.2 moved the failure step by
12 steps only (5,397 -> 5,409). A loss-magnitude problem would scale with the weight; a
conditioning problem does not. Job 360343's NLL-only fallback then rejected nearly every
auxiliary update, which is the same statement measured a third way: the inverse-path
gradient is essentially never finite once the flow has drifted, so the objective was
silently absent rather than stabilised.

-------------------------------------------------------------------------------
2. THE NEW OBJECTIVE
-------------------------------------------------------------------------------
For a real embedding x (train split, with the same smoothing noise the NLL uses) whose
true standardised QED is q and whose atom count is N, build K candidate conditions that
share the molecule's OWN N and differ only in QED:

    c_0 = (N, q)                        <- the true candidate, label 0
    c_k = (N, clamp(q + delta_k))       <- N-matched negatives, k = 1..K-1

Score every candidate by the model's own conditional log-likelihood, normalised per node
so molecules of different size are comparable, and apply a cross-entropy with label 0:

    s_k      = log p(x | c_k) / n_nodes
    logits   = (s_k - s_0) * logit_scale            (exact shift-invariant centring)
    L_contr  = -log softmax(logits)[0]
             = logsumexp_k(logits)                  (since logits[0] = 0)

In words: *among QED values that a molecule of this size could have, the model's density
must prefer the one it actually has.*

GRADIENT PATH. `log p(x | c_k)` is computed by `flow.log_prob`, i.e. the FORWARD map -
the identical computational path the NLL has already traversed stably for 50,000 steps.
There is no `linalg.inv`, no sequential autoregressive inverse, no unbounded sqrt. The
backward graph depth equals the NLL's.

GRADIENT MAGNITUDE IS PROVABLY BOUNDED RELATIVE TO THE NLL. The derivative of a softmax
cross-entropy with respect to its logits lies in [-1, 1] per entry. Hence

    || dL_contr / d(theta) ||  <=  K * max_k || d s_k / d(theta) ||
                               =   K * max_k || d log p(x|c_k)/d(theta) || / n_nodes,

and the right-hand side is (up to the constant K) exactly the per-node NLL gradient that
training already tolerates. This is the strongest stability statement available short of
running it: the new term cannot produce a gradient of a different ORDER than the NLL.

WHY THE LOG-DETERMINANT NO LONGER SWAMPS THE SIGNAL. `log|det J|` is ~350 per node here
and dominates the NLL, so the QED-relevant prior term is a tiny fraction of the NLL
gradient. In the softmax, the component of each s_k that is common to all candidates
CANCELS (softmax is shift-invariant, and the centring above makes the cancellation
exact in float32 rather than relying on catastrophic subtraction of two ~350 values).
What remains is purely the *relative* conditional fit across QED. This is the concrete
reason to expect a different outcome from merely having a positive permutation gap.

WHY N-MATCHING IS ESSENTIAL. N and QED are strongly dependent in ZINC250K (mean QED
rises to ~0.79 at N~20-21 then falls to ~0.46 by N=35). Negatives that change N would let
the model win by reading N alone - which it already does, and which is exactly why the
existing permutation gap (+0.077 nats/node) coexists with a ~0 within-N response slope.
Every candidate here shares the molecule's own N, so the only way to reduce the loss is
genuine *within-N* QED discrimination: the precise contrast the evaluation measures.

-------------------------------------------------------------------------------
3. WHY THIS SHOULD MOVE GENERATION, AND THE HONEST LIMIT
-------------------------------------------------------------------------------
At the evaluation temperature T = 0.2 a sample is z = mu(c) + 0.2*sigma(c)*eps, so it
lies within a fifth of a prior standard deviation of mu(c), and the generated molecule is
essentially `decode(f^-1(mu(c), c))`. Controllability at low temperature is therefore
governed by how much `mu(c)` (and the FiLM-shaped map around it) moves with QED.

The degenerate shortcut is blocked. If the flow simply shifted the latent by a
condition-dependent amount, z_k = g(x) + h(c_k), and the prior learned mu(c_k) = h(c_k),
then z_k - mu(c_k) = g(x) is IDENTICAL for every candidate, a pure shift contributes
nothing to log|det J|, so all s_k are equal and the loss sits at its maximum log K. The
objective actively penalises the shortcut that would otherwise produce an impressive
likelihood score with no generative control. (This is also why the latent must be
recomputed per candidate and not shared: sharing a latent computed under c_0 would
reward exactly that shortcut.)

`prior_mu_gap` is logged every time the term is applied: the mean L2 distance between
mu(c_0) and mu(c_k). It is the mechanistic quantity, and it is reported so the claim
"the prior is separating" is measured rather than assumed.

LIMITATION, STATED PLAINLY. This objective makes the conditional DENSITY discriminative
in QED at fixed N. That is necessary for conditional generation but not sufficient for
the RDKit-measured QED of the final molecule to track the request, because the frozen
decoder and the degree-constrained assembly sit between the embedding and the measured
QED, and this project has already measured that this pathway can attenuate signal
(stages 20-21). Nothing here may be reported as improved QED control until it is
measured with actual RDKit QED on generated molecules. There is also a diversity risk:
the loss can be reduced by sharpening sigma(c) as well as by separating mu(c), so prior
sigma, uniqueness and diversity must be read alongside any QED gain.

-------------------------------------------------------------------------------
4. COST, PARAMETERS, COMPATIBILITY
-------------------------------------------------------------------------------
K forward passes on a configurable sub-batch, optionally every `every` steps; no inverse
pass at all, so this is strictly cheaper than the old objective. It introduces NO new
parameters and NO new checkpoint keys, so a checkpoint trained with it is bit-compatible
with `load_flow` and every existing generation/evaluation script. Train split only; the
candidate clamp range is derived from TRAIN conditions only. No leakage.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F

from zinc_gnf.conditioning import condition_embedding_for
from zinc_gnf.flow import NodeFlow

# Finite stand-in for "exclude this candidate from the softmax". exp(-1e4) underflows to
# exactly 0 in float32 without producing a NaN, and log_softmax/logsumexp handle it
# stably. -inf is deliberately NOT used: it poisons gradients if a row is fully masked.
MASK_LOGIT = -1.0e4
NEGATIVE_STRATEGIES = ("offset", "inbatch")


def candidate_conditions(values: torch.Tensor, qed_column: int, offsets: Sequence[float],
                         *, qed_min: float, qed_max: float) -> torch.Tensor:
    """`[B, F]` standardised conditions -> `[B, K, F]` candidates differing ONLY in QED.

    Every candidate keeps the molecule's own N column, which is what makes the contrast
    N-matched. Candidate QED is clamped to the TRAIN-split standardised range so the
    prior is never interrogated about a QED value no molecule can have.
    `offsets[0]` must be 0.0, so candidate 0 is the true condition and the label is 0.
    """
    if not offsets or float(offsets[0]) != 0.0:
        raise ValueError("offsets[0] must be exactly 0.0 (the true candidate carries label 0)")
    if qed_max < qed_min:
        raise ValueError("qed_max must be >= qed_min")
    batch = values.shape[0]
    deltas = torch.as_tensor([float(o) for o in offsets], dtype=values.dtype,
                             device=values.device)
    candidates = values.unsqueeze(1).repeat(1, deltas.numel(), 1)       # [B, K, F]
    shifted = values[:, qed_column].unsqueeze(1) + deltas.unsqueeze(0)  # [B, K]
    candidates[:, :, qed_column] = shifted.clamp(min=float(qed_min), max=float(qed_max))
    if batch and candidates.shape[1] != deltas.numel():
        raise AssertionError("candidate construction produced the wrong K")
    return candidates


def inbatch_candidate_conditions(values: torch.Tensor, qed_column: int, k: int, *,
                                 generator: torch.Generator | None = None) -> torch.Tensor:
    """Negatives taken from the QED values of OTHER molecules in the batch.

    Closer to a proper InfoNCE bound on I(x; QED | N) because negatives follow the
    empirical QED marginal, and in-range by construction. Each candidate still keeps the
    molecule's own N column.
    """
    if k < 2:
        raise ValueError("inbatch negatives need k >= 2")
    batch = values.shape[0]
    if batch < 2:
        raise ValueError("inbatch negatives need at least two molecules in the batch")
    candidates = values.unsqueeze(1).repeat(1, k, 1)
    index = torch.arange(batch, device=values.device)
    for position in range(1, k):
        # a fixed non-zero roll per candidate slot guarantees a different molecule
        shift = torch.randint(1, batch, (1,), generator=generator,
                              device="cpu" if generator is not None else values.device)
        rolled = (index + int(shift)) % batch
        candidates[:, position, qed_column] = values[rolled, qed_column]
    return candidates


def contrastive_weight(step: int, *, weight: float, warmup: int, ramp: int) -> float:
    """0 until `warmup`, then a linear ramp to `weight` over `ramp` steps, then constant."""
    if weight <= 0 or step < warmup:
        return 0.0
    if ramp <= 0:
        return float(weight)
    return float(weight) * min(1.0, (step - warmup + 1) / ramp)


def qed_contrastive_loss(flow: NodeFlow, embeddings: torch.Tensor, mask: torch.Tensor,
                         values: torch.Tensor, *, qed_column: int,
                         offsets: Sequence[float] = (0.0, -1.0, 1.0),
                         qed_min: float, qed_max: float,
                         min_separation: float = 0.25, logit_scale: float = 1.0,
                         negatives: str = "offset",
                         generator: torch.Generator | None = None
                         ) -> tuple[torch.Tensor, dict[str, Any]]:
    """The objective. Returns `(loss, diagnostics)`; forward-only, no inverse pass.

    `embeddings` must be the SAME (smoothing-noised) real training embeddings the NLL
    scored, and `values` their standardised condition rows. Nothing here samples the
    prior or inverts the flow.
    """
    if negatives not in NEGATIVE_STRATEGIES:
        raise ValueError(f"negatives must be one of {NEGATIVE_STRATEGIES}, got {negatives!r}")
    if not flow.config.conditional or "qed" not in flow.config.condition_features:
        raise ValueError("the QED-contrastive objective needs a flow conditioned on 'qed'")
    if min_separation <= 0:
        raise ValueError("min_separation must be > 0")
    batch = int(embeddings.shape[0])
    device = embeddings.device

    if negatives == "offset":
        candidates = candidate_conditions(values, qed_column, offsets,
                                          qed_min=qed_min, qed_max=qed_max)
    else:
        candidates = inbatch_candidate_conditions(values, qed_column, len(offsets),
                                                  generator=generator)
    k = int(candidates.shape[1])

    # A candidate too close to the truth makes the label ambiguous rather than wrong, so
    # it is excluded from that molecule's softmax instead of injecting label noise.
    true_qed = values[:, qed_column].unsqueeze(1)                       # [B, 1]
    separation = (candidates[:, :, qed_column] - true_qed).abs()        # [B, K]
    valid = separation >= float(min_separation)
    valid[:, 0] = True                                                  # the truth always counts
    has_negative = valid[:, 1:].any(dim=1)                              # [B]

    counts = mask.sum(-1).clamp(min=1).to(embeddings.dtype)             # [B]
    present = torch.ones_like(values)
    scores = []
    for position in range(k):
        condition = condition_embedding_for(flow, candidates[:, position, :], present,
                                            batch, device)
        log_prob = flow.log_prob(embeddings, mask, condition)["log_prob"]
        scores.append(log_prob / counts)                                # per node
    stacked = torch.stack(scores, dim=1)                                # [B, K]

    # Centre on the TRUE candidate. Mathematically a no-op for the softmax (shift
    # invariance) but it removes the float32 cancellation of two ~350-magnitude numbers,
    # so the ~0.08-magnitude signal is not lost to rounding.
    logits = (stacked - stacked[:, :1]) * float(logit_scale)
    logits = torch.where(valid, logits, torch.full_like(logits, MASK_LOGIT))
    # logits[:, 0] == 0 by construction, so the cross-entropy against label 0 is logsumexp.
    per_molecule = torch.logsumexp(logits, dim=1)
    usable = has_negative.to(per_molecule.dtype)
    denominator = usable.sum().clamp(min=1.0)
    loss = (per_molecule * usable).sum() / denominator

    with torch.no_grad():
        probabilities = torch.softmax(logits, dim=1)
        chosen = logits.argmax(dim=1)
        correct = ((chosen == 0) & has_negative).sum()
        # The mechanistic quantity: does the conditional prior MEAN separate across QED?
        mu_gap = torch.zeros((), device=device, dtype=embeddings.dtype)
        gap_terms = 0
        mu_true, sigma_true = flow.prior.parameters_for(
            batch, condition_embedding_for(flow, candidates[:, 0, :], present, batch, device),
            device)
        for position in range(1, k):
            mu_other, _ = flow.prior.parameters_for(
                batch,
                condition_embedding_for(flow, candidates[:, position, :], present, batch, device),
                device)
            step_gap = (mu_other - mu_true).norm(dim=-1)                # [B]
            mu_gap = mu_gap + (step_gap * valid[:, position].to(step_gap.dtype)).sum()
            gap_terms += int(valid[:, position].sum())
        diagnostics = {
            "qed_contrast_loss": float(loss.detach()),
            "qed_contrast_chance_loss": float(torch.log(torch.tensor(float(k)))),
            "qed_contrast_accuracy": float(correct) / max(1, int(has_negative.sum())),
            "qed_contrast_true_probability": float(
                (probabilities[:, 0] * usable).sum() / denominator),
            "qed_contrast_usable_fraction": float(has_negative.to(torch.float32).mean()),
            "qed_contrast_molecules": int(has_negative.sum()),
            "qed_contrast_candidates": k,
            "prior_mu_gap": (float(mu_gap) / gap_terms) if gap_terms else 0.0,
            "prior_sigma_mean_true": float(sigma_true.mean()),
            "qed_contrast_logit_spread": float(
                (stacked.max(dim=1).values - stacked.min(dim=1).values).mean()),
        }
    return loss, diagnostics


@torch.no_grad()
def invertible_linear_conditioning(flow: NodeFlow) -> dict[str, Any]:
    """Read-only conditioning report for every `InvertiblePointwiseLinear`.

    This is the diagnostic for root cause (a): the NLL never constrains the CONDITIONING
    of these matrices (it only sees `sum(log_s)`), yet `inverse` must invert them and the
    gradient of that inverse scales with ||W^-1||^2. A large condition number here is
    direct evidence that backpropagating through `flow.inverse` was unsound, and it is
    worth watching even for sampling, where the inverse is still evaluated.

    Changes nothing; safe to call at validation time.
    """
    numbers: list[float] = []
    log_s_extremes: list[float] = []
    for block in flow.blocks:
        linear = getattr(block, "linear", None)
        if linear is None:
            continue
        log_s_extremes.append(float(linear.log_s.detach().abs().max()))
        try:
            weight = linear.weight().detach().to(torch.float64)
            numbers.append(float(torch.linalg.cond(weight)))
        except Exception:  # noqa: BLE001 - a failed SVD must not stop training
            numbers.append(float("nan"))
    if not numbers:
        return {}
    finite = [v for v in numbers if v == v]
    return {
        "linear_condition_number_max": max(finite) if finite else float("nan"),
        "linear_condition_number_mean": (sum(finite) / len(finite)) if finite else float("nan"),
        "linear_max_abs_log_s": max(log_s_extremes) if log_s_extremes else float("nan"),
        "linear_layers": len(numbers),
    }


__all__ = [
    "MASK_LOGIT",
    "NEGATIVE_STRATEGIES",
    "candidate_conditions",
    "contrastive_weight",
    "inbatch_candidate_conditions",
    "invertible_linear_conditioning",
    "qed_contrastive_loss",
]
