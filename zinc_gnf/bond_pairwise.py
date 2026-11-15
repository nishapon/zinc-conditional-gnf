"""Bond-aware pairwise pathway for the affine coupling conditioner (experiment C).

WHAT PAIR-SPECIFIC INFORMATION THE CODE ALREADY HAS
---------------------------------------------------
`MaskedSelfAttention.forward` (flow.py:270) already computes a learned pair score:

    logits_ij = (W_q p_i) . (W_k p_j) / sqrt(head_dim),  p = input_projection(source)
    attended_i = sum_j softmax_j(logits_ij) * (W_v p_j)

So "a learned pairwise score between nodes, fed into the conditioner" is NOT new, and a
change that only adds another content-based pair score would largely duplicate it. Three
properties of the pathway below are outside what that attention can express, and the
first is the one this experiment is actually betting on.

1. UNNORMALISED AGGREGATION CARRIES DEGREE; SOFTMAX CANNOT.
   `softmax` over j makes `attended_i` a convex combination of value vectors: the weights
   sum to 1 no matter how many neighbours are "selected". An attention head can say WHICH
   nodes matter and in what proportion, but not HOW MANY matter. The gate here is a
   per-pair independent sigmoid, so

       degree_i = sum_j b_ij,     b_ij = sigmoid(l_ij) in (0, 1)

   is a soft neighbour COUNT, and it is handed to the conditioner as an explicit feature.
   Valence - degree together with bond order - is exactly the constraint that decides
   whether a decoded molecule parses, and raw validity is what collapses at N=30 (18-53%
   at N=15, 4-10% at N=23, 0-1% at N=30). A conditioner that cannot represent "this atom
   already has four neighbours" cannot avoid the error that dominates those numbers. That
   is the mechanism-level reason to expect something here that the correlated prior did
   not deliver: the prior changed the latent's covariance, not the conditioner's access
   to local structure.

2. EXACTLY SYMMETRIC. `l_ij == l_ji` holds by construction (see `_pair_logits`), while
   `Q K^T` is asymmetric. A bond is a symmetric relation, so the asymmetric form spends
   capacity representing a distinction that does not exist in the target.

3. SUPERVISABLE. The score has a ground-truth target - the molecule's bond matrix, which
   `MolecularBatch.bond_types` already carries into every training step - whereas the
   attention logits are only ever shaped by the likelihood. Arm `pairwise_supervised`
   tests whether that target is what matters; arm `pairwise` runs the identical
   architecture on likelihood alone, so architecture and training signal are separable.

A secondary point: `l_ij` includes `w . |h_i - h_j|`, which is not a bilinear form in
(h_i, h_j) and therefore not representable by any single dot-product head.

WHY THIS PRESERVES THE FLOW MATHEMATICS
---------------------------------------
The scorer reads ONLY:

    * `source` - the coupling's pass-through half, which the coupling leaves untouched
      and which is therefore bit-identical in the forward and inverse directions;
    * `condition` - the (N, QED) embedding, supplied identically in both directions;
    * `mask` - from which the pair mask is derived.

The true bond matrix is NEVER an input. It is used only as a target for an auxiliary
loss. So `scale` and `translation` remain functions of the pass-through half alone, the
map stays a bijection in `x`, and `log|det J| = sum of scale over real nodes and channels`
is unchanged. This is the distinction the design hinges on: conditioning the coupling ON
bonds would turn the model into `p(embeddings | N, QED, A)` and require a bond matrix at
sampling time, which `NodeFlow.sample` has no way to supply.

Padding: the pair mask is `mask_i & mask_j & (i != j)`, the gate is multiplied by it
before any aggregation, and rows of padded nodes are zeroed on the way out. A padded
column therefore contributes neither to `degree_i` nor to any message.

CLASS IMBALANCE
---------------
A ZINC molecule with N heavy atoms and R rings has N-1+R bonds among N(N-1)/2 pairs. At
three ring systems that is a positive rate of 0.152 at N=15, 0.099 at N=23 and 0.074 at
N=30, so "no bond anywhere" scores 84.8%, 90.1% and 92.6% accuracy while being useless,
and the imbalance gets WORSE exactly where validity is worst. Two things stop the trivial
predictor from looking like success: the loss carries
`pos_weight = negatives/positives` measured per batch, and the reported metrics are
precision, recall and F1 for the positive class alongside `trivial_accuracy` (the accuracy
of the all-negative predictor on that same batch), so plain accuracy can never be read on
its own. `bond_metrics` also reports the predicted and true mean degree, which is the
quantity the conditioner actually consumes.

INITIALISATION
--------------
`pair_to_hidden` is zero-initialised, so at step 0 the pathway adds exactly zero to the
conditioner's pre-activation and the augmented flow reproduces the parent checkpoint's
density bit for bit. One consequence must be stated plainly: with that weight at zero,
d(loss)/d(scorer parameters) through the LIKELIHOOD is zero at step 0, because it carries
a factor of `pair_to_hidden.weight`. The weight itself has a nonzero gradient
(d(hidden)/d(W) = pair_features), so W leaves zero on the first step and the scorer
receives likelihood gradient from step 1 onward. The bond supervision, by contrast,
reaches the scorer at step 0 directly. `tests/test_bond_pairwise.py` asserts both halves
of this, because "the new pathway never learns anything" and "the new pathway is
identity-initialised" would otherwise be indistinguishable from the loss curve.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn
import torch.nn.functional as F

# Width of the per-node projection the pair scorer works in. The dominant memory cost of
# this pathway is a [B, N, N, BOND_PAIR_DIM] tensor per coupling (the |h_i - h_j| term),
# so this is the knob to turn when memory is tight.
BOND_PAIR_DIM = 16
# Cap on the positive-class reweighting, so a batch that happens to contain very few
# bonded pairs cannot produce an enormous gradient spike.
POS_WEIGHT_CAP = 50.0
# Threshold at which the reported precision/recall/F1 are computed. Diagnostic only; the
# conditioner consumes the continuous gate, never a thresholded one.
BOND_DECISION_THRESHOLD = 0.5


def pair_mask_from_node_mask(mask: torch.Tensor) -> torch.Tensor:
    """`[B, N, N]` true for real ordered pairs `i != j`, from `[B, N]` alone.

    Derived from the node mask rather than taken from the batch, because the conditioner
    must be able to build it at generation time, where only `mask` exists.
    """
    both = mask.unsqueeze(-1) & mask.unsqueeze(1)
    width = mask.shape[1]
    off_diagonal = ~torch.eye(width, dtype=torch.bool, device=mask.device).unsqueeze(0)
    return both & off_diagonal


class BondPairwiseConditioner(nn.Module):
    """Soft bond scores from the pass-through half, aggregated degree-sensitively.

    Returns `(pair_features, logits)`:

        pair_features [B, N, P + 2]   per-node summary to add into the conditioner
        logits        [B, N, N]       symmetric, masked, the auxiliary loss's input
    """

    def __init__(self, in_dim: int, *, pair_dim: int = BOND_PAIR_DIM,
                 condition_dim: int = 0) -> None:
        super().__init__()
        self.pair_dim = int(pair_dim)
        self.pair_projection = nn.Linear(in_dim, self.pair_dim)
        # three additive contributions to the pair logit, all symmetric in (i, j)
        self.sum_head = nn.Linear(self.pair_dim, 1)                  # w . (u_i + u_j)
        self.product_weight = nn.Parameter(torch.zeros(self.pair_dim))  # w . (h_i * h_j)
        self.absolute_head = nn.Linear(self.pair_dim, 1, bias=False)  # w . |h_i - h_j|
        self.condition_bias = (nn.Linear(condition_dim, 1) if condition_dim else None)
        self.value = nn.Linear(self.pair_dim, self.pair_dim)
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.xavier_uniform_(self.pair_projection.weight)
        nn.init.zeros_(self.pair_projection.bias)
        nn.init.xavier_uniform_(self.sum_head.weight)
        nn.init.zeros_(self.sum_head.bias)
        nn.init.xavier_uniform_(self.absolute_head.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.zeros_(self.value.bias)
        with torch.no_grad():
            self.product_weight.normal_(std=0.1)
        if self.condition_bias is not None:
            nn.init.zeros_(self.condition_bias.weight)
            nn.init.zeros_(self.condition_bias.bias)
        # Set by the arm trainer. When true AND the module is training, the logits of the
        # most recent forward are kept so the auxiliary loss can read them without
        # threading a pair tensor through every layer signature.
        self.publish_logits = False
        self.last_bond_logits: torch.Tensor | None = None

    @property
    def output_dim(self) -> int:
        """Width of `pair_features`: the message, plus two degree scalars."""
        return self.pair_dim + 2

    def _pair_logits(self, hidden: torch.Tensor, pair_mask: torch.Tensor,
                     condition: torch.Tensor | None) -> torch.Tensor:
        """Symmetric `[B, N, N]` logits. Every term is invariant under swapping i and j."""
        projected = self.sum_head(hidden)                              # [B, N, 1]
        logits = projected + projected.transpose(1, 2)                 # w.(u_i + u_j)
        weighted = hidden * self.product_weight                        # [B, N, P]
        logits = logits + torch.einsum("bip,bjp->bij", weighted, hidden)
        difference = (hidden.unsqueeze(2) - hidden.unsqueeze(1)).abs()  # [B, N, N, P]
        logits = logits + self.absolute_head(difference).squeeze(-1)
        logits = logits + self.bias
        if self.condition_bias is not None:
            if condition is None:
                raise ValueError("this bond conditioner is conditional but no condition "
                                 "was given")
            logits = logits + self.condition_bias(condition).unsqueeze(-1)
        # padded pairs and the diagonal are driven to a hard zero gate, so they cannot
        # contribute to any degree or message
        return logits.masked_fill(~pair_mask, float("-inf"))

    def forward(self, source: torch.Tensor, mask: torch.Tensor,
                condition: torch.Tensor | None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        floats = mask.unsqueeze(-1).to(source.dtype)
        hidden = self.pair_projection(source) * floats
        pair_mask = pair_mask_from_node_mask(mask)
        logits = self._pair_logits(hidden, pair_mask, condition)
        gate = torch.sigmoid(logits) * pair_mask.to(source.dtype)      # exact 0 on -inf
        degree = gate.sum(-1)                                          # [B, N] soft count
        values = self.value(hidden) * floats
        message = torch.einsum("bij,bjp->bip", gate, values)
        # dividing by (1 + degree) keeps the message bounded while `degree` itself is
        # passed through untouched, so the count survives as its own feature
        message = message / (1.0 + degree.unsqueeze(-1))
        neighbours = mask.sum(-1, keepdim=True).to(source.dtype).clamp(min=2.0) - 1.0
        features = torch.cat([message,
                              torch.log1p(degree).unsqueeze(-1),
                              (degree / neighbours).unsqueeze(-1)], dim=-1)
        features = features * floats
        if self.publish_logits and self.training:
            self.last_bond_logits = logits
        return features, logits

    def clear_published(self) -> None:
        self.last_bond_logits = None


# --------------------------------------------------------------- supervision ---
def bond_targets(bond_types: torch.Tensor) -> torch.Tensor:
    """`1.0` where a bond of any order exists. Binary, because the gate is one score."""
    return (bond_types > 0).to(torch.float32)


def upper_triangle_mask(pair_mask: torch.Tensor) -> torch.Tensor:
    """Each unordered real pair exactly once, so no pair is counted twice."""
    width = pair_mask.shape[-1]
    upper = torch.triu(torch.ones((width, width), dtype=torch.bool,
                                   device=pair_mask.device), diagonal=1)
    return pair_mask & upper.unsqueeze(0)


def bond_loss_and_metrics(logits: Sequence[torch.Tensor], bond_types: torch.Tensor,
                          node_mask: torch.Tensor, *,
                          pos_weight_cap: float = POS_WEIGHT_CAP,
                          threshold: float = BOND_DECISION_THRESHOLD
                          ) -> dict[str, Any]:
    """Positive-reweighted BCE over unordered real pairs, averaged over layers.

    `logits` is one `[B, N, N]` tensor per bond-aware coupling. Each is scored against the
    same target, and the loss is their mean, so adding blocks does not change the weight
    of this term relative to the likelihood.
    """
    if len(logits) == 0:
        raise ValueError("no bond logits were published; the bond-aware pathway did not "
                         "run, so there is nothing to supervise")
    pair_mask = pair_mask_from_node_mask(node_mask)
    counted = upper_triangle_mask(pair_mask)
    target = bond_targets(bond_types)
    weights = counted.to(target.dtype)
    total = weights.sum().clamp(min=1.0)
    positives = (target * weights).sum()
    negatives = (total - positives).clamp(min=0.0)
    # the whole point: a positive pair costs negatives/positives times as much as a
    # negative one, so "no bond anywhere" is not a cheap solution
    pos_weight = (negatives / positives.clamp(min=1.0)).clamp(max=float(pos_weight_cap))

    losses = []
    for tensor in logits:
        if tensor.shape != target.shape:
            raise ValueError(f"bond logits {tuple(tensor.shape)} do not match the target "
                             f"{tuple(target.shape)}")
        # -inf on padded pairs would make BCE non-finite, so score a finite surrogate
        # there and then zero it with the weights
        safe = torch.where(counted, tensor, torch.zeros_like(tensor))
        element = F.binary_cross_entropy_with_logits(
            safe, target, reduction="none", pos_weight=pos_weight)
        losses.append((element * weights).sum() / total)
    loss = torch.stack(losses).mean()

    with torch.no_grad():
        mean_logits = torch.stack([torch.where(counted, t, torch.full_like(t, -30.0))
                                   for t in logits]).mean(0)
        predicted = (torch.sigmoid(mean_logits) >= threshold) & counted
        actual = (target > 0.5) & counted
        true_positive = float((predicted & actual).sum())
        false_positive = float((predicted & ~actual).sum())
        false_negative = float((~predicted & actual).sum())
        correct = float(((predicted == actual) & counted).sum())
        pairs = float(total)
        positive_pairs = float(positives)
        precision = true_positive / max(true_positive + false_positive, 1.0)
        recall = true_positive / max(true_positive + false_negative, 1.0)
        f1 = (2 * precision * recall / max(precision + recall, 1e-12)
              if (precision + recall) > 0 else 0.0)
        gate = torch.sigmoid(mean_logits) * counted.to(target.dtype)
        # degree over the FULL symmetric matrix, which is what the conditioner sees
        symmetric_gate = gate + gate.transpose(1, 2)
        nodes = node_mask.sum().clamp(min=1).to(target.dtype)
        metrics = {
            "bond_loss": float(loss.detach()),
            "bond_pos_weight": float(pos_weight),
            "bond_pos_rate": positive_pairs / max(pairs, 1.0),
            "bond_accuracy": correct / max(pairs, 1.0),
            # the number plain accuracy has to beat to mean anything
            "bond_trivial_accuracy": (pairs - positive_pairs) / max(pairs, 1.0),
            "bond_precision": precision,
            "bond_recall": recall,
            "bond_f1": f1,
            "bond_predicted_degree": float((symmetric_gate.sum(-1)
                                            * node_mask).sum() / nodes),
            "bond_true_degree": float(((target * pair_mask.to(target.dtype)).sum(-1)
                                       * node_mask).sum() / nodes),
            "bond_layers": len(logits),
        }
    return {"loss": loss, "metrics": metrics}


def collect_published_logits(flow, *, batch_size: int, width: int
                             ) -> list[torch.Tensor]:
    """Gather this step's logits from every bond-aware conditioner, with a freshness check.

    The alternative was threading a pair tensor through `NodeFlow.forward`, `FlowBlock`,
    `AffineCoupling` and `CouplingNetwork`. That would have changed four signatures in a
    file the cluster has diverged on, for a tensor only the auxiliary loss ever reads. The
    conditioners instead keep their most recent logits, and the shape check below makes a
    stale tensor impossible to use by accident: the trainer clears after every step, so a
    missing or wrongly shaped entry raises instead of silently supervising the previous
    batch.
    """
    found = []
    for name, module in flow.named_modules():
        if not isinstance(module, BondPairwiseConditioner):
            continue
        logits = module.last_bond_logits
        if logits is None:
            raise RuntimeError(
                f"{name} published no bond logits for this step; the auxiliary loss would "
                f"otherwise be computed from a stale or absent forward pass")
        if tuple(logits.shape) != (batch_size, width, width):
            raise RuntimeError(
                f"{name} published logits of shape {tuple(logits.shape)}, expected "
                f"{(batch_size, width, width)}; this is a stale tensor from an earlier "
                f"batch, not this step's")
        found.append(logits)
    return found


def clear_published_logits(flow) -> int:
    cleared = 0
    for module in flow.modules():
        if isinstance(module, BondPairwiseConditioner):
            module.clear_published()
            cleared += 1
    return cleared


def set_publishing(flow, enabled: bool) -> int:
    count = 0
    for module in flow.modules():
        if isinstance(module, BondPairwiseConditioner):
            module.publish_logits = bool(enabled)
            count += 1
    return count


def bond_pathway_report(flow) -> dict[str, Any]:
    """What the pathway looks like in a built flow. Written into the arm report."""
    conditioners = [(name, module) for name, module in flow.named_modules()
                    if isinstance(module, BondPairwiseConditioner)]
    gate_parameters = 0
    bridge_parameters = 0
    for _, module in conditioners:
        gate_parameters += sum(p.numel() for p in module.parameters())
    for name, module in flow.named_modules():
        if name.endswith("pair_to_hidden"):
            bridge_parameters += sum(p.numel() for p in module.parameters())
    bridges = [module for name, module in flow.named_modules()
               if name.endswith("pair_to_hidden")]
    return {
        "conditioners": len(conditioners),
        "names": [name for name, _ in conditioners][:4],
        "pair_dim": conditioners[0][1].pair_dim if conditioners else None,
        "gate_parameters": gate_parameters,
        "bridge_parameters": bridge_parameters,
        "new_parameters": gate_parameters + bridge_parameters,
        "bridge_is_zero": all(float(m.weight.abs().max()) == 0.0 for m in bridges)
                          if bridges else None,
        "note": ("bridge_is_zero true means the pathway currently adds exactly nothing "
                 "to the conditioner, i.e. the flow still reproduces the parent density"),
    }


__all__ = [
    "BOND_DECISION_THRESHOLD",
    "BOND_PAIR_DIM",
    "POS_WEIGHT_CAP",
    "BondPairwiseConditioner",
    "bond_loss_and_metrics",
    "bond_pathway_report",
    "bond_targets",
    "clear_published_logits",
    "collect_published_logits",
    "pair_mask_from_node_mask",
    "set_publishing",
    "upper_triangle_mask",
]
