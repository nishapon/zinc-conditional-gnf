"""Exchangeable conditional prior with a shared graph-level component. EXACT density.

MOTIVATION FROM THE CODE
------------------------
`zinc_gnf/flow.py:606` documents `NodePrior` as a "Per-node i.i.d. Gaussian prior", and
`NodePrior.log_prob` (flow.py:732) sums an independent per-dimension Gaussian over nodes
with no cross-node term. So every bit of coordination between a molecule's node vectors
has to be manufactured by the coupling blocks; the prior itself cannot express "these N
vectors belong to the same molecule". That is the gap this addresses.

WHY NOT A HIERARCHICAL LATENT
-----------------------------
The obvious construction is a global latent u with

    p(z | c) = integral p(u | c) * prod_i p(z_i | u, c) du

and it must be rejected here: that integral has no closed form, so the model's exact
log-likelihood would be replaced by a bound and the flow would stop being an
exact-likelihood model. Swapping the training objective for an ELBO while still calling
the result a normalizing flow would make every NLL number in the project
non-comparable with the ones already reported.

THE EXACT ALTERNATIVE
---------------------
Marginalise the shared component ANALYTICALLY by choosing it Gaussian and linear. Draw
one scalar per feature dimension, shared across all nodes of a molecule:

    g_d ~ N(0, 1)                                   one draw per molecule, per dimension
    z_{i,d} = mu_d(c) + sigma_d(c) * ( eps_{i,d} + a_d * g_d ),   eps_{i,d} ~ N(0, 1)

Marginally, for a molecule with N real nodes, the N-vector z_{:,d} is Gaussian with

    mean    = mu_d(c) * 1
    cov     = sigma_d(c)^2 * ( I_N + a_d^2 * 1 1^T )

a rank-one-plus-diagonal covariance. Both the determinant and the inverse are closed
form, by the matrix determinant lemma and the Woodbury identity:

    log|cov| = N * log sigma_d^2 + log(1 + N * a_d^2)
    cov^-1   = sigma_d^-2 * ( I_N - (a_d^2 / (1 + N a_d^2)) * 1 1^T )

so with r_i = (z_{i,d} - mu_d) / sigma_d,

    log p(z_{:,d}) = -0.5 * [ N log(2 pi) + N log sigma_d^2 + log(1 + N a_d^2)
                              + sum_i r_i^2 - (a_d^2 / (1 + N a_d^2)) * (sum_i r_i)^2 ]

summed over d. Exact, O(N*D), no integration and no bound. Verified against a brute-force
dense-covariance multivariate normal to better than 1e-15 for several (N, sigma, a).

PROPERTIES THAT MATTER HERE
---------------------------
* EXCHANGEABLE. The covariance depends on node count only, never on node order, so the
  prior stays permutation-invariant like the graphs it models.
* N-AWARE BY CONSTRUCTION. The `log(1 + N a_d^2)` term and the shrinkage coefficient both
  depend on N, so the prior's shape genuinely varies with molecule size rather than only
  its sample count.
* LOCAL DIVERSITY PRESERVED. The per-node `eps` term is untouched; `a_d` adds a shared
  offset on top of it. Node vectors become correlated, not identical, and the
  within-molecule variance stays sigma_d^2.
* STRICT GENERALISATION. At a_d = 0 the density and the sampler reduce EXACTLY to the
  existing i.i.d. prior, so `from_node_prior` can warm-start from a trained checkpoint
  with the new parameter at zero and the model is initially unchanged. That makes the
  unchanged-training control a genuine control: same checkpoint, same steps, same budget,
  differing only in whether `a` is allowed to move.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

# a_d is parameterised through a bounded tanh so a single bad step cannot send the
# shared component to infinity; the bound is generous (a=2 means the shared part
# dominates the per-node part by 4x in variance).
MAX_COUPLING = 2.0


class CoupledNodePrior(nn.Module):
    """`NodePrior` plus a shared per-dimension component. Drop-in, exact, exchangeable.

    Reuses the wrapped prior's `mu(c), sigma(c)` unchanged, so the conditioning pathway
    and its zero-initialised heads are exactly the ones already trained.
    """

    def __init__(self, base: nn.Module, *, init_coupling: float = 0.0) -> None:
        super().__init__()
        if not hasattr(base, "parameters_for"):
            raise TypeError("base must expose parameters_for(batch, condition, device)")
        if not 0.0 <= init_coupling < MAX_COUPLING:
            raise ValueError(f"init_coupling must lie in [0, {MAX_COUPLING})")
        self.base = base
        self.dim = int(base.dim)
        # atanh of the initial value, so pre_coupling = 0 gives coupling = 0 exactly
        ratio = init_coupling / MAX_COUPLING
        start = 0.5 * math.log((1.0 + ratio) / (1.0 - ratio)) if ratio > 0 else 0.0
        self.pre_coupling = nn.Parameter(torch.full((self.dim,), float(start)))

    # -- parameters ----------------------------------------------------
    @property
    def conditional(self) -> bool:
        return bool(self.base.conditional)

    def coupling(self) -> torch.Tensor:
        """`a_d >= 0`, bounded by MAX_COUPLING. Zero reproduces the i.i.d. prior."""
        return MAX_COUPLING * torch.tanh(self.pre_coupling).abs()

    def parameters_for(self, batch_size: int, condition: torch.Tensor | None,
                       device: torch.device | str):
        return self.base.parameters_for(batch_size, condition, device)

    # -- density -------------------------------------------------------
    def log_prob(self, z: torch.Tensor, mask: torch.Tensor,
                 condition: torch.Tensor | None) -> torch.Tensor:
        """Exact log density of the rank-one-plus-diagonal Gaussian. `[B]`.

        Padding is excluded by masking the residuals before both the sum of squares and
        the shared-component sum, and by using the REAL node count in the determinant
        and shrinkage terms. A padded position must not contribute to either, or the
        density would depend on the padding width.
        """
        mu, sigma = self.parameters_for(z.shape[0], condition, z.device)
        mu, sigma = mu.unsqueeze(1), sigma.unsqueeze(1)
        weights = mask.to(z.dtype).unsqueeze(-1)
        residual = ((z - mu) / sigma) * weights                      # [B, N, D]
        counts = mask.to(z.dtype).sum(-1).clamp(min=1.0).unsqueeze(-1)   # [B, 1]

        coupling = self.coupling().view(1, -1)                       # [1, D]
        squared = coupling ** 2
        denominator = 1.0 + counts * squared                         # [B, D]
        shrinkage = squared / denominator

        sum_squares = (residual ** 2).sum(dim=1)                     # [B, D]
        sum_residual = residual.sum(dim=1)                           # [B, D]
        log_sigma = torch.log(sigma.squeeze(1))                      # [B, D]

        per_dimension = -0.5 * (
            counts * math.log(2 * math.pi)
            + 2.0 * counts * log_sigma
            + torch.log(denominator)
            + sum_squares
            - shrinkage * sum_residual ** 2
        )
        return per_dimension.sum(-1)                                 # [B]

    # -- sampling ------------------------------------------------------
    @torch.no_grad()
    def sample(self, mask: torch.Tensor, condition: torch.Tensor | None, *,
               temperature: float = 1.0,
               generator: torch.Generator | None = None) -> torch.Tensor:
        """`z = mu + T * sigma * (eps + a * g)`, one `g` per molecule per dimension.

        Temperature scales the whole deviation, shared component included, so the
        sampler stays consistent with the density it is scored by at T = 1.
        """
        batch, width = mask.shape
        device = mask.device
        mu, sigma = self.parameters_for(batch, condition, device)
        noise = torch.randn((batch, width, self.dim), device=device, dtype=mu.dtype,
                            generator=generator)
        shared = torch.randn((batch, 1, self.dim), device=device, dtype=mu.dtype,
                             generator=generator)
        deviation = noise + self.coupling().view(1, 1, -1) * shared
        z = mu.unsqueeze(1) + float(temperature) * sigma.unsqueeze(1) * deviation
        return z * mask.unsqueeze(-1).to(z.dtype)

    # -- construction / reporting --------------------------------------
    @classmethod
    def from_node_prior(cls, base: nn.Module, *, init_coupling: float = 0.0
                        ) -> "CoupledNodePrior":
        """Wrap a trained prior. At `init_coupling = 0` the model is unchanged."""
        return cls(base, init_coupling=init_coupling)

    def summary(self) -> dict[str, Any]:
        with torch.no_grad():
            coupling = self.coupling()
        return {
            "max_coupling": MAX_COUPLING,
            "coupling_mean": float(coupling.mean()),
            "coupling_max": float(coupling.max()),
            "coupling_min": float(coupling.min()),
            "dimensions_above_0p05": int((coupling > 0.05).sum()),
            "reduces_to_iid_prior": bool(float(coupling.max()) == 0.0),
            "note": ("a_d = 0 in every dimension means the prior is exactly the "
                     "original i.i.d. prior, so any change in behaviour came from "
                     "continued training rather than from this parameter"),
        }

    def shared_variance_fraction(self) -> torch.Tensor:
        """`a^2 / (1 + a^2)`: how much of a node's variance is the shared component."""
        squared = self.coupling() ** 2
        return squared / (1.0 + squared)


def dense_log_prob(z: torch.Tensor, mask: torch.Tensor, mu: torch.Tensor,
                   sigma: torch.Tensor, coupling: torch.Tensor) -> torch.Tensor:
    """Brute-force reference: build the dense covariance and score it directly.

    Exists only so tests can confirm the closed form against an independent computation
    rather than against itself. O(N^3) per molecule per dimension, so test-only.
    """
    batch = z.shape[0]
    totals = []
    for index in range(batch):
        keep = mask[index].to(torch.bool)
        count = int(keep.sum())
        if count == 0:
            totals.append(torch.zeros((), dtype=z.dtype, device=z.device))
            continue
        total = torch.zeros((), dtype=torch.float64, device=z.device)
        for dimension in range(z.shape[-1]):
            values = z[index, keep, dimension].to(torch.float64)
            mean = mu[index, dimension].to(torch.float64)
            scale = sigma[index, dimension].to(torch.float64)
            a = coupling[dimension].to(torch.float64)
            identity = torch.eye(count, dtype=torch.float64, device=z.device)
            ones = torch.ones((count, count), dtype=torch.float64, device=z.device)
            covariance = scale ** 2 * (identity + a ** 2 * ones)
            residual = (values - mean).unsqueeze(-1)
            solved = torch.linalg.solve(covariance, residual)
            total = total - 0.5 * (
                count * math.log(2 * math.pi)
                + torch.logdet(covariance)
                + (residual.transpose(0, 1) @ solved).squeeze())
        totals.append(total.to(z.dtype))
    return torch.stack(totals)


__all__ = ["MAX_COUPLING", "CoupledNodePrior", "dense_log_prob"]
