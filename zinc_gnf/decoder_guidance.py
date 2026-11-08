"""Decoder-aware guidance in embedding space (INFERENCE-TIME ONLY, Stage 1).

THIS IS GUIDANCE, NOT INTRINSIC CONDITIONAL GENERATION. Every number produced with it
must be labelled as post-hoc guidance. It is a MECHANISM TEST: does a decoder-aware
local step move the ACTUAL decoded RDKit QED and the RAW chemical validity at all?
Nothing here trains the flow, and the distillation that would make it intrinsic is
deliberately not implemented (Stage 2 is not approved).

WHY GRADIENTS NEVER TOUCH `flow.inverse`
----------------------------------------
The QED-consistency v1 experiment went non-finite because it backpropagated through the
inverse map. Four sites, all present ONLY in the inverse and therefore never constrained
by the NLL:

  flow.py:247      `torch.linalg.inv(self.weight())`. The forward needs only `x @ W.T`
                   and `sum(log_s)` (line 243), so W can drift near-singular with zero
                   NLL penalty, and the matrix-inverse gradient scales as ||W^-1||^2,
                   compounding over 12 blocks.
  flow.py:503-520  the autoregressive inverse re-runs `_scale_and_translation` INSIDE a
                   loop over nodes, each iteration depending on the previous through
                   `torch.cat`. Backward chains O(N x blocks) ~ 456 Jacobians, each up to
                   exp(max_log_scale) = e^1; float32 overflows at e^89.
  flow.py:324/351/489  only the SCALE is tanh-clamped; `translation_head(hidden)` is
                   unbounded and compounds additively across blocks.
  (v1 surrogate)   pooled `spread.sqrt()`, derivative 1/(2*sqrt(v)) -> infinity as v -> 0,
                   and symmetry-equivalent atoms give v ~ 0 by construction.

This module avoids all four STRUCTURALLY rather than mitigating them: `f^-1` is evaluated
once under `no_grad`, and the only backward pass is through the frozen decoder's logit
heads (`decoder.py:70-77,102`), which are plain MLPs on node embeddings. Backward depth
is ~3, not ~456. There is no matrix inverse, no sequential chain, no unbounded exp, and
pooling uses a masked MEAN with no square root.

THE OBJECTIVE
-------------
    x_{k+1} = Pi_rho( x_k - eta * grad_x [ lambda_q * L_qed + lambda_m * L_manifold ] )

    L_qed      = ( q_hat(phi(x)) - q_target )^2        TARGET MATCHING, both directions
    L_manifold = weighted squared error between differentiable decoder statistics of
                 phi(x) and the statistics MEASURED on real molecules at the same N

`L_qed` is a squared error to the REQUESTED value, so guidance moves QED down for a
target of 0.4 and up for 0.8. Maximising predicted QED would have been wrong for every
condition except the highest, which is the correction this version implements.

`Pi_rho` projects onto a trust region of radius rho around the unguided sample, with rho
measured from the real within-N embedding dispersion. The step is `grad_x` evaluated AT
the current point, so it is local and non-linear in x and is re-derived every iteration:
this is the property a single fixed latent direction lacked, which is why that approach
failed even on real N=30 embeddings.

FEATURES, AND WHY FIRST MOMENTS ALONE ARE NOT ENOUGH
----------------------------------------------------
Mean-pooled atom / bond / degree probabilities are first moments. They cannot express
ring structure at all: a chain and a fused-ring system can share a bond-class histogram.
Since aromatic-ring count is both a QED component (AROM) and one of our measured
deficits, a first-moment feature set is expected to be blind to exactly what matters.

The justified addition is derived, not fitted. For a connected graph the cyclomatic
number (count of independent rings) is

    rings = E - N + 1,

and the expected edge count is available twice, differentiably, from the two heads:

    E_bond   = sum_{i<j} ( 1 - P(no bond)_{ij} )
    E_degree = (1/2) * sum_i E[deg_i],    E[deg_i] = sum_d d * P(deg_i = d)

so `structural` features add E_bond, E_degree, both implied ring counts, their
disagreement |E_bond - E_degree| (an internal-consistency signal that the assembler
needs, since `degree_constrained_bond_decode` falls back when the degree sequence is
infeasible) and N itself. `FEATURE_SETS` exposes both so the question is answered
EMPIRICALLY by held-out fit quality rather than asserted.

THE MANIFOLD TERM IS AN EMPIRICAL HYPOTHESIS
--------------------------------------------
Matching mean statistics does not guarantee chemically valid graphs, and it has three
specific failure modes which this implementation addresses explicitly but does not claim
to have solved:

  sparse bond tensors   real molecules occupy ~N of N(N-1)/2 pairs, so P(no bond) is
                        ~0.9; the generated value was measured at 0.994 against 0.904
                        real. Pushing the MEAN toward the real value adds edge mass but
                        says nothing about WHERE edges go, so a dense blob is a real
                        risk. Mitigated by targeting the expected EDGE COUNT and RING
                        COUNT (not only the class histogram), both of which a blob
                        violates, and detected by raw validity and uniqueness.
  degree consistency    the two heads can disagree. The consistency term penalises
                        |E_bond - E_degree| directly, which is the structural identity
                        sum_i deg_i = 2E that the assembler relies on.
  size dependence       every target is measured PER N and every extensive quantity is
                        divided by N, so N=15 and N=30 contribute comparably.

Whether it helps is decided by measured raw validity, corrected validity, aromaticity
and diversity -- not by the loss going down.

SURROGATE EXPLOITATION
----------------------
Cannot be prevented a priori. Mitigations: supervision is real (train embeddings ->
RDKit QED), the surrogate is frozen during guidance, the trust region caps displacement
at the real dispersion, and the default surrogate is RIDGE -- linear in the features, so
it has no capacity to carve out adversarial basins, while its gradient
J_phi(x)^T w remains point-dependent through J_phi. The detector is the metric itself:
surrogate-predicted QED and actual decoded RDKit QED are always reported side by side,
so exploitation appears as the two diverging.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

FEATURE_SETS = ("moments", "structural")
RIDGE_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
RIDGE_SEED = 20261112
# Objective component weights. Fixed, documented, and NOT swept: the brief forbids broad
# searches and every extensive quantity is already normalised by N.
MANIFOLD_WEIGHTS = {
    "bond_histogram": 1.0,
    "degree_histogram": 1.0,
    "edges_per_node": 1.0,
    "rings_per_node": 1.0,
    "head_consistency": 1.0,
}


# ------------------------------------------------------------------- features ---
def _masked_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Masked mean over dim 1. No square root anywhere (v1's NaN source)."""
    total = weights.sum(dim=1).clamp(min=1.0)
    return (values * weights).sum(dim=1) / total


def decoder_statistics(outputs: dict[str, torch.Tensor], node_mask: torch.Tensor,
                       pair_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Differentiable per-molecule statistics of the frozen decoder's logits.

    Every quantity is a function of the logits only, so it is differentiable in the node
    embeddings through the decoder heads and nothing else.
    """
    node_weights = node_mask.unsqueeze(-1).to(outputs["atom_logits"].dtype)
    counts = node_mask.sum(-1).clamp(min=1).to(node_weights.dtype)          # [B]

    atom_probabilities = torch.softmax(outputs["atom_logits"], dim=-1)      # [B,N,17]
    atom_histogram = _masked_mean(atom_probabilities, node_weights)         # [B,17]

    bond_probabilities = torch.softmax(outputs["bond_logits"], dim=-1)      # [B,N,N,4]
    pair_weights = pair_mask.unsqueeze(-1).to(bond_probabilities.dtype)
    pair_total = pair_weights.sum(dim=(1, 2)).clamp(min=1.0)                # [B,1]
    bond_histogram = ((bond_probabilities * pair_weights).sum(dim=(1, 2))
                      / pair_total)                                         # [B,4]

    # expected edge count from the bond head. pair_mask excludes the diagonal and counts
    # each unordered pair twice, hence the factor 1/2.
    edge_mass = ((1.0 - bond_probabilities[..., 0]) * pair_mask.to(bond_probabilities.dtype)
                 ).sum(dim=(1, 2)) * 0.5                                    # [B]

    statistics: dict[str, torch.Tensor] = {
        "atom_histogram": atom_histogram,
        "bond_histogram": bond_histogram,
        "edges": edge_mass,
        "edges_per_node": edge_mass / counts,
        # cyclomatic number of a connected graph: E - N + 1
        "rings_from_bonds": edge_mass - counts + 1.0,
        "n_node": counts,
    }
    if "degree_logits" in outputs:
        degree_probabilities = torch.softmax(outputs["degree_logits"], dim=-1)  # [B,N,7]
        degree_histogram = _masked_mean(degree_probabilities, node_weights)     # [B,7]
        levels = torch.arange(degree_probabilities.shape[-1],
                              device=degree_probabilities.device,
                              dtype=degree_probabilities.dtype)
        expected_degree = (degree_probabilities * levels).sum(-1)               # [B,N]
        degree_sum = (expected_degree * node_mask.to(expected_degree.dtype)).sum(-1)
        edges_from_degree = degree_sum * 0.5                                    # [B]
        statistics.update({
            "degree_histogram": degree_histogram,
            "edges_from_degree": edges_from_degree,
            "rings_from_degree": edges_from_degree - counts + 1.0,
            # the structural identity sum_i deg_i = 2E, as a penalty-ready residual
            "head_disagreement": (edge_mass - edges_from_degree).abs() / counts,
        })
    return statistics


def feature_vector(statistics: dict[str, torch.Tensor], *,
                   feature_set: str = "structural") -> torch.Tensor:
    """`[B, F]` differentiable features. `moments` = first moments only."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
    parts = [statistics["atom_histogram"], statistics["bond_histogram"]]
    if "degree_histogram" in statistics:
        parts.append(statistics["degree_histogram"])
    if feature_set == "structural":
        counts = statistics["n_node"]
        extra = [statistics["edges_per_node"].unsqueeze(-1),
                 (statistics["rings_from_bonds"] / counts).unsqueeze(-1),
                 counts.unsqueeze(-1)]
        if "rings_from_degree" in statistics:
            extra.extend([(statistics["edges_from_degree"] / counts).unsqueeze(-1),
                          (statistics["rings_from_degree"] / counts).unsqueeze(-1),
                          statistics["head_disagreement"].unsqueeze(-1)])
        parts.extend(extra)
    return torch.cat(parts, dim=-1)


def feature_names(num_atom_types: int, num_bond_types: int, max_degree_classes: int,
                  *, feature_set: str = "structural") -> list[str]:
    names = [f"atom_{i}" for i in range(num_atom_types)]
    names += [f"bond_{i}" for i in range(num_bond_types)]
    names += [f"degree_{i}" for i in range(max_degree_classes)]
    if feature_set == "structural":
        names += ["edges_per_node", "rings_per_node", "n_node",
                  "edges_from_degree_per_node", "rings_from_degree_per_node",
                  "head_disagreement"]
    return names


@torch.no_grad()
def embedding_features(autoencoder, embeddings: torch.Tensor, node_mask: torch.Tensor,
                       pair_mask: torch.Tensor, *, feature_set: str = "structural"
                       ) -> torch.Tensor:
    """Features for already-computed embeddings, no gradient (fitting / evaluation)."""
    outputs = autoencoder.decoder(embeddings, node_mask, pair_mask, atom_types=None)
    return feature_vector(decoder_statistics(outputs, node_mask, pair_mask),
                          feature_set=feature_set)


def features_with_grad(autoencoder, embeddings: torch.Tensor, node_mask: torch.Tensor,
                       pair_mask: torch.Tensor, *, feature_set: str = "structural"):
    """Features WITH the graph retained, for guidance. Decoder params stay frozen.

    The caller supplies `embeddings` with `requires_grad_(True)`; the decoder's own
    parameters have `requires_grad=False`, so the only gradient produced is d/d(embeddings).
    """
    outputs = autoencoder.decoder(embeddings, node_mask, pair_mask, atom_types=None)
    statistics = decoder_statistics(outputs, node_mask, pair_mask)
    return feature_vector(statistics, feature_set=feature_set), statistics


# ------------------------------------------------------------------ surrogate ---
class RidgeReadout:
    """Closed-form ridge on the differentiable features. Deterministic, no SGD, no sweep.

    Linear on purpose. A linear readout has no capacity to carve adversarial basins, so
    it is the least exploitable surrogate available, while its gradient
    J_phi(x)^T w stays point-dependent through the decoder Jacobian -- so guidance is
    still local and non-linear in x, which is what the fixed-direction approach lacked.
    `alpha` is chosen on an INNER split of TRAIN and never on evaluation data.
    """

    def __init__(self, weights: np.ndarray, bias: float, mean: np.ndarray,
                 scale: np.ndarray, alpha: float, target: str) -> None:
        self.weights, self.bias = weights, float(bias)
        self.mean, self.scale = mean, scale
        self.alpha, self.target = float(alpha), str(target)

    @classmethod
    def fit(cls, features: np.ndarray, targets: np.ndarray, *, target: str,
            alphas: Sequence[float] = RIDGE_ALPHAS, seed: int = RIDGE_SEED
            ) -> tuple["RidgeReadout", dict[str, Any]]:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64).reshape(-1)
        if x.shape[0] != y.shape[0] or x.shape[0] < 20:
            raise ValueError("need at least 20 aligned samples to fit a readout")
        mean = x.mean(0)
        scale = x.std(0)
        scale[scale < 1e-8] = 1.0
        xs = (x - mean) / scale
        rng = np.random.default_rng(seed)
        order = rng.permutation(xs.shape[0])
        cut = int(0.8 * xs.shape[0])
        inner_train, inner_val = order[:cut], order[cut:]
        best = None
        for alpha in alphas:
            gram = xs[inner_train].T @ xs[inner_train] + alpha * np.eye(xs.shape[1])
            w = np.linalg.solve(gram, xs[inner_train].T @ (y[inner_train] - y[inner_train].mean()))
            b = y[inner_train].mean()
            error = float(np.abs(xs[inner_val] @ w + b - y[inner_val]).mean())
            if best is None or error < best[0]:
                best = (error, alpha, w, b)
        _, alpha, _, _ = best
        gram = xs.T @ xs + alpha * np.eye(xs.shape[1])
        weights = np.linalg.solve(gram, xs.T @ (y - y.mean()))
        readout = cls(weights, float(y.mean()), mean, scale, alpha, target)
        return readout, {"alpha_selected": float(alpha),
                         "inner_validation_mae": float(best[0]),
                         "train_samples": int(x.shape[0]),
                         "features": int(x.shape[1])}

    def predict_numpy(self, features: np.ndarray) -> np.ndarray:
        x = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        return x @ self.weights + self.bias

    def torch_parameters(self, device, dtype):
        return (torch.as_tensor(self.weights, device=device, dtype=dtype),
                torch.as_tensor(self.mean, device=device, dtype=dtype),
                torch.as_tensor(self.scale, device=device, dtype=dtype))

    def predict_torch(self, features: torch.Tensor) -> torch.Tensor:
        w, mean, scale = self.torch_parameters(features.device, features.dtype)
        return ((features - mean) / scale) @ w + self.bias

    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target, "alpha": self.alpha, "bias": self.bias,
                "weights": [float(v) for v in self.weights],
                "feature_mean": [float(v) for v in self.mean],
                "feature_scale": [float(v) for v in self.scale]}


def readout_quality(readout: RidgeReadout, features: np.ndarray, truths: np.ndarray,
                    *, groups: np.ndarray | None = None) -> dict[str, Any]:
    """Held-out MAE / correlation, overall and per group (per N)."""
    predicted = readout.predict_numpy(features)
    truth = np.asarray(truths, dtype=np.float64).reshape(-1)

    def block(p, t) -> dict[str, Any]:
        if p.size < 3:
            return {"samples": int(p.size), "mae": None, "correlation": None}
        correlation = (float(np.corrcoef(p, t)[0, 1])
                       if p.std() > 1e-9 and t.std() > 1e-9 else None)
        return {"samples": int(p.size), "mae": float(np.abs(p - t).mean()),
                "bias": float((p - t).mean()), "correlation": correlation,
                "truth_std": float(t.std()), "prediction_std": float(p.std())}

    result: dict[str, Any] = {"target": readout.target, "overall": block(predicted, truth)}
    if groups is not None:
        per_group = {}
        for value in sorted(set(int(v) for v in groups)):
            keep = np.asarray(groups) == value
            per_group[str(value)] = block(predicted[keep], truth[keep])
        result["per_n"] = per_group
    return result


# ------------------------------------------------------------------ objective ---
def manifold_loss(statistics: dict[str, torch.Tensor], targets: dict[str, torch.Tensor],
                  weights: dict[str, float] = MANIFOLD_WEIGHTS
                  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Squared error to statistics MEASURED on real molecules at the same N.

    Extensive quantities are per-node, so molecular size does not dominate. Includes the
    head-consistency residual, which is the structural identity sum_i deg_i = 2E that the
    degree-constrained assembler needs in order not to fall back.
    """
    parts: dict[str, torch.Tensor] = {}
    if "bond_histogram" in targets:
        parts["bond_histogram"] = ((statistics["bond_histogram"]
                                    - targets["bond_histogram"]) ** 2).sum(-1)
    if "degree_histogram" in targets and "degree_histogram" in statistics:
        parts["degree_histogram"] = ((statistics["degree_histogram"]
                                      - targets["degree_histogram"]) ** 2).sum(-1)
    if "edges_per_node" in targets:
        parts["edges_per_node"] = (statistics["edges_per_node"]
                                   - targets["edges_per_node"]) ** 2
    if "rings_per_node" in targets:
        implied = statistics["rings_from_bonds"] / statistics["n_node"]
        parts["rings_per_node"] = (implied - targets["rings_per_node"]) ** 2
    if "head_disagreement" in statistics:
        parts["head_consistency"] = statistics["head_disagreement"] ** 2
    total = None
    for name, value in parts.items():
        term = float(weights.get(name, 1.0)) * value
        total = term if total is None else total + term
    if total is None:
        raise ValueError("no manifold targets were supplied")
    return total, parts


def guidance_loss(readout: RidgeReadout | None, features: torch.Tensor,
                  statistics: dict[str, torch.Tensor],
                  targets: dict[str, torch.Tensor] | None, *, qed_target: float,
                  lambda_qed: float, lambda_manifold: float
                  ) -> tuple[torch.Tensor, dict[str, Any]]:
    """`lambda_qed * (q_hat - q_target)^2 + lambda_manifold * L_manifold`, per molecule.

    TARGET MATCHING, so the gradient points DOWN in QED for a low target and UP for a
    high one. Maximising predicted QED would be wrong for every condition but the
    highest.
    """
    total = torch.zeros(features.shape[0], device=features.device, dtype=features.dtype)
    diagnostics: dict[str, Any] = {}
    if lambda_qed != 0.0:
        if readout is None:
            raise ValueError("lambda_qed != 0 requires a fitted QED readout")
        predicted = readout.predict_torch(features)
        qed_term = (predicted - float(qed_target)) ** 2
        total = total + float(lambda_qed) * qed_term
        diagnostics["surrogate_qed"] = predicted
        diagnostics["qed_term"] = qed_term
    if lambda_manifold != 0.0:
        if targets is None:
            raise ValueError("lambda_manifold != 0 requires manifold targets")
        manifold, parts = manifold_loss(statistics, targets)
        total = total + float(lambda_manifold) * manifold
        diagnostics["manifold_term"] = manifold
        diagnostics["manifold_parts"] = parts
    return total, diagnostics


# ------------------------------------------------------------------- guidance ---
def guide(autoencoder, embeddings: torch.Tensor, node_mask: torch.Tensor,
          pair_mask: torch.Tensor, *, readout: RidgeReadout | None,
          targets: dict[str, torch.Tensor] | None, qed_target: float,
          lambda_qed: float, lambda_manifold: float, steps: int, step_size: float,
          trust_radius: float, feature_set: str = "structural",
          direction: str = "gradient", generator: torch.Generator | None = None
          ) -> tuple[torch.Tensor, dict[str, Any]]:
    """Bounded guidance. Returns `(guided_embeddings, report)`.

    `direction='random'` replaces the gradient with a fixed-norm isotropic step, which is
    the matched-norm control: it answers whether any displacement of this size changes
    the outcome.

    Safeguards: the decoder and the readout are frozen; `f^-1` is never involved; the
    gradient is clipped per molecule; the displacement from `x_0` is projected onto a
    ball of radius `trust_radius`; padding is re-zeroed every step; a non-finite step
    REVERTS to the last finite state and is counted.
    """
    if steps < 0:
        raise ValueError("steps must be >= 0")
    start = (embeddings * node_mask.unsqueeze(-1).to(embeddings.dtype)).detach()
    current = start.clone()
    batch = start.shape[0]
    node_weights = node_mask.unsqueeze(-1).to(start.dtype)
    nonfinite_steps = 0
    history: list[dict[str, float]] = []

    for _ in range(int(steps)):
        if direction == "random":
            if generator is None:
                raise ValueError("direction='random' needs a generator")
            update = torch.randn(current.shape, device=current.device,
                                 dtype=current.dtype, generator=generator)
        else:
            probe = current.clone().requires_grad_(True)
            features, statistics = features_with_grad(
                autoencoder, probe, node_mask, pair_mask, feature_set=feature_set)
            loss, diagnostics = guidance_loss(
                readout, features, statistics, targets, qed_target=qed_target,
                lambda_qed=lambda_qed, lambda_manifold=lambda_manifold)
            objective = loss.sum()
            if not torch.isfinite(objective):
                nonfinite_steps += 1
                continue
            (gradient,) = torch.autograd.grad(objective, probe, retain_graph=False,
                                              allow_unused=True)
            # allow_unused covers the degenerate lambda_qed = lambda_manifold = 0 call,
            # where the objective is constant and there is nothing to descend.
            update = -(torch.zeros_like(probe) if gradient is None else gradient)
            history.append({
                "loss": float(loss.detach().mean()),
                "surrogate_qed": float(diagnostics["surrogate_qed"].detach().mean())
                if "surrogate_qed" in diagnostics else float("nan"),
                "manifold": float(diagnostics["manifold_term"].detach().mean())
                if "manifold_term" in diagnostics else float("nan"),
            })

        update = update * node_weights
        if not torch.isfinite(update).all():
            nonfinite_steps += 1
            continue
        # per-molecule normalisation, so one molecule cannot set the step for the batch
        norms = update.flatten(1).norm(dim=1).clamp(min=1e-12).view(batch, 1, 1)
        scaled = update / norms * float(step_size) * float(trust_radius)
        proposal = current + scaled
        # trust region around the UNGUIDED sample
        displacement = (proposal - start) * node_weights
        magnitude = displacement.flatten(1).norm(dim=1).view(batch, 1, 1)
        factor = torch.clamp(float(trust_radius) / magnitude.clamp(min=1e-12), max=1.0)
        proposal = start + displacement * factor
        proposal = proposal * node_weights
        if not torch.isfinite(proposal).all():
            nonfinite_steps += 1
            continue
        current = proposal.detach()

    displacement = ((current - start) * node_weights).flatten(1).norm(dim=1)
    report = {
        "steps_requested": int(steps),
        "nonfinite_steps": int(nonfinite_steps),
        "nonfinite_rate": nonfinite_steps / max(1, int(steps)),
        "trust_radius": float(trust_radius),
        "step_size": float(step_size),
        "direction": direction,
        "lambda_qed": float(lambda_qed),
        "lambda_manifold": float(lambda_manifold),
        "displacement_mean": float(displacement.mean()),
        "displacement_max": float(displacement.max()),
        "trust_region_saturated_fraction": float(
            (displacement >= 0.999 * float(trust_radius)).to(torch.float64).mean()),
        "history": history,
        "fallback_behaviour": ("a non-finite loss, gradient or proposal reverts to the "
                              "last finite state and is counted in nonfinite_steps; the "
                              "guidance is never silently continued on a bad state"),
    }
    return current, report


__all__ = [
    "FEATURE_SETS",
    "MANIFOLD_WEIGHTS",
    "RidgeReadout",
    "decoder_statistics",
    "embedding_features",
    "feature_names",
    "feature_vector",
    "features_with_grad",
    "guidance_loss",
    "guide",
    "manifold_loss",
    "readout_quality",
]
