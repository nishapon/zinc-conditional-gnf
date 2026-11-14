"""Exchangeable conditional prior with a shared graph-level component. EXACT density.

MOTIVATION FROM THE CODE
------------------------
`zinc_gnf/flow.py:606` documents `NodePrior` as a "Per-node i.i.d. Gaussian prior", and
`NodePrior.log_prob` (flow.py:732) sums an independent per-dimension Gaussian over nodes
with no cross-node term. Every bit of coordination between a molecule's node vectors must
therefore be manufactured by the coupling blocks; the prior cannot express "these N
vectors belong to the same molecule".

WHY NOT A HIERARCHICAL LATENT
-----------------------------
`p(z|c) = integral p(u|c) prod_i p(z_i|u,c) du` has no closed form, so exact likelihood
would be replaced by a bound and every NLL already reported would stop being comparable.
Rejected. The shared component is marginalised ANALYTICALLY instead.

THE PARAMETERISATION, AND TWO CORRECTIONS THAT CHANGED IT
---------------------------------------------------------
The quantity learned is the SHARED VARIANCE FRACTION rho = corr(z_i, z_j), i != j.

1. NOT the coupling amplitude `a`. With z_i = mu + sigma(eps_i + a*g) the density depends
   on a only through a^2, so d(log p)/da carries a factor 2a and is EXACTLY ZERO at a=0.
   A parameter initialised at zero could never move. Parameterising rho directly removes
   that factor: the numerical derivative d(log p)/d(rho) is -0.390 at rho = 1e-9, i.e.
   nonzero in the limit. `rho = RHO_MAX * sigmoid(pre_rho)` has a strictly positive
   Jacobian everywhere, so the gradient reaching `pre_rho` never vanishes by
   construction.

2. MARGINAL VARIANCE IS PRESERVED, BY EXPLICIT CHOICE. The amplitude form inflates it to
   sigma^2 (1 + a^2); only the pairwise difference variance stayed at 2 sigma^2. That is
   the wrong trade here, because the flow was trained against sigma(c) and rescaling the
   latent would interact with the whitening and the log-determinant, confounding the
   comparison with the baseline. So the shared and private parts are normalised to sum
   to one:

       z_{i,d} = mu_d(c) + sigma_d(c) * ( sqrt(1 - rho_d) * eps_{i,d}
                                          + sqrt(rho_d) * g_d )

   with eps_{i,d} ~ N(0,1) per node and g_d ~ N(0,1) once per molecule per dimension.
   Then Var(z_{i,d}) = sigma_d^2 exactly, for every rho, and Corr(z_i, z_j) = rho_d.
   Changing rho REDISTRIBUTES variance between shared and private; it does not add any.

THE EXACT DENSITY
-----------------
Marginally the N real nodes are Gaussian with
    mean = mu_d * 1,  cov = sigma_d^2 * ( tau_d * I_N + rho_d * 1 1^T ),  tau_d = 1 - rho_d
which is diagonal plus rank one, so by the matrix determinant lemma and Woodbury

    log|cov| = N log sigma_d^2 + (N-1) log tau_d + log(tau_d + N rho_d)
    cov^-1   = sigma_d^-2 * tau_d^-1 * ( I - (rho_d / (tau_d + N rho_d)) 1 1^T )

giving, with r_i = (z_{i,d} - mu_d) / sigma_d,

    log p(z_{:,d}) = -0.5 * [ N log(2 pi) + N log sigma_d^2
                              + (N-1) log tau_d + log(tau_d + N rho_d)
                              + tau_d^-1 * ( sum_i r_i^2
                                             - (rho_d/(tau_d + N rho_d)) (sum_i r_i)^2 ) ]

summed over d. Exact, O(N*D), no integration and no bound. Verified against a
brute-force dense-covariance solve for six (N, sigma, rho) settings including rho = 1e-9
and rho = 0.95: agreement to 1e-13 or better. As rho -> 0 it reduces to the i.i.d.
density (checked to 1.7e-12), though rho = 0 is not attainable by the sigmoid, which is
deliberate -- see the control design below.

CONTROL DESIGN
--------------
Because rho cannot be exactly zero, the i.i.d. baseline is the ORIGINAL `NodePrior`, not
this module at rho = 0. Three arms, all warm-started from the same checkpoint with the
same step budget:

    baseline   original NodePrior, continued training
    frozen     this prior at rho = INIT_RHO, `pre_rho.requires_grad = False`
    learned    this prior at rho = INIT_RHO, `pre_rho` trainable

`frozen` versus `baseline` isolates the effect of HAVING cross-node correlation at all;
`learned` versus `frozen` isolates the effect of LEARNING it. Without the frozen arm at
the identical initialisation the two effects are inseparable.

MASKING
-------
Padded positions contribute to neither the likelihood nor the shared-component
statistics: residuals are masked before both `sum_i r_i^2` and `sum_i r_i`, the node
count in the determinant and shrinkage terms is the REAL count, and `sample` re-zeroes
padding after drawing. A padded column therefore cannot change the density, which
matters because N is a conditioning variable here.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

# rho is a correlation, so it lives in (0, 1). The cap keeps the covariance comfortably
# positive definite and stops a single step driving every node of a molecule to the same
# vector, which would destroy local diversity.
RHO_MAX = 0.90
# Documented small nonzero initialisation. Small enough that the warm-started model is
# close to the trained one, large enough to be numerically well clear of the boundary.
INIT_RHO = 0.01


class CoupledNodePrior(nn.Module):
    """`NodePrior` plus an exchangeable shared component. Exact, variance-preserving.

    The wrapped prior's `mu(c), sigma(c)` are reused unchanged, so the conditioning
    pathway is exactly the one already trained.
    """

    def __init__(self, base: nn.Module, *, init_rho: float = INIT_RHO,
                 learnable: bool = True) -> None:
        super().__init__()
        if not hasattr(base, "parameters_for"):
            raise TypeError("base must expose parameters_for(batch, condition, device)")
        if not 0.0 < init_rho < RHO_MAX:
            raise ValueError(f"init_rho must lie in (0, {RHO_MAX}); rho = 0 is not "
                             f"representable and the i.i.d. baseline is the original "
                             f"NodePrior, not this module at zero")
        self.base = base
        self.dim = int(base.dim)
        ratio = init_rho / RHO_MAX
        start = math.log(ratio / (1.0 - ratio))          # logit, so sigmoid(start)=ratio
        self.pre_rho = nn.Parameter(torch.full((self.dim,), float(start)),
                                    requires_grad=bool(learnable))
        self.register_buffer("init_rho", torch.tensor(float(init_rho)))
        self.register_buffer("learnable", torch.tensor(bool(learnable)))

    # -- parameters ----------------------------------------------------
    @property
    def conditional(self) -> bool:
        return bool(self.base.conditional)

    def rho(self) -> torch.Tensor:
        """Shared variance fraction in (0, RHO_MAX). Strictly positive Jacobian."""
        return RHO_MAX * torch.sigmoid(self.pre_rho)

    def parameters_for(self, batch_size: int, condition: torch.Tensor | None,
                       device: torch.device | str):
        return self.base.parameters_for(batch_size, condition, device)

    def freeze_rho(self) -> None:
        self.pre_rho.requires_grad_(False)
        self.learnable.fill_(False)

    # -- density -------------------------------------------------------
    def log_prob(self, z: torch.Tensor, mask: torch.Tensor,
                 condition: torch.Tensor | None) -> torch.Tensor:
        """Exact log density of the diagonal-plus-rank-one Gaussian. Returns `[B]`."""
        mu, sigma = self.parameters_for(z.shape[0], condition, z.device)
        mu, sigma = mu.unsqueeze(1), sigma.unsqueeze(1)
        weights = mask.to(z.dtype).unsqueeze(-1)
        residual = ((z - mu) / sigma) * weights                       # [B, N, D]
        counts = mask.to(z.dtype).sum(-1).clamp(min=1.0).unsqueeze(-1)    # [B, 1]

        rho = self.rho().view(1, -1)                                  # [1, D]
        tau = 1.0 - rho
        denominator = tau + counts * rho                              # [B, D]

        sum_squares = (residual ** 2).sum(dim=1)                      # [B, D]
        sum_residual = residual.sum(dim=1)                            # [B, D]
        log_sigma = torch.log(sigma.squeeze(1))                       # [B, D]

        per_dimension = -0.5 * (
            counts * math.log(2 * math.pi)
            + 2.0 * counts * log_sigma
            + (counts - 1.0) * torch.log(tau)
            + torch.log(denominator)
            + (sum_squares - (rho / denominator) * sum_residual ** 2) / tau
        )
        return per_dimension.sum(-1)                                  # [B]

    # -- sampling ------------------------------------------------------
    @torch.no_grad()
    def sample(self, mask: torch.Tensor, condition: torch.Tensor | None, *,
               temperature: float = 1.0,
               generator: torch.Generator | None = None) -> torch.Tensor:
        """`z = mu + T*sigma*( sqrt(1-rho)*eps_i + sqrt(rho)*g )`, g once per molecule.

        Consistent with `log_prob` at T = 1 by construction: the two share the same
        tau/rho split, so the sampler draws from exactly the density it is scored by.
        """
        batch, width = mask.shape
        device = mask.device
        mu, sigma = self.parameters_for(batch, condition, device)
        rho = self.rho().view(1, 1, -1)
        private = torch.randn((batch, width, self.dim), device=device, dtype=mu.dtype,
                              generator=generator)
        shared = torch.randn((batch, 1, self.dim), device=device, dtype=mu.dtype,
                             generator=generator)
        deviation = torch.sqrt(1.0 - rho) * private + torch.sqrt(rho) * shared
        z = mu.unsqueeze(1) + float(temperature) * sigma.unsqueeze(1) * deviation
        # padding is re-zeroed, so a padded node carries neither likelihood nor shared
        # statistics downstream
        return z * mask.unsqueeze(-1).to(z.dtype)

    # -- construction / reporting --------------------------------------
    @classmethod
    def from_node_prior(cls, base: nn.Module, *, init_rho: float = INIT_RHO,
                        learnable: bool = True) -> "CoupledNodePrior":
        return cls(base, init_rho=init_rho, learnable=learnable)

    def summary(self) -> dict[str, Any]:
        # detach before any float() conversion so a diagnostic never
        # touches the autograd graph
        with torch.no_grad():
            rho = self.rho().detach()
        return {
            "rho_max": RHO_MAX,
            "init_rho": float(self.init_rho),
            "learnable": bool(self.learnable),
            "rho_mean": float(rho.mean()),
            "rho_max_observed": float(rho.max()),
            "rho_min_observed": float(rho.min()),
            "dimensions_above_init": int((rho > float(self.init_rho) * 1.5).sum()),
            "marginal_variance": "preserved by construction: Var(z_i) = sigma^2 "
                                 "for every rho",
            "moved_from_initialisation": bool(
                abs(float(rho.mean()) - float(self.init_rho)) > 1e-6),
        }


def dense_log_prob(z: torch.Tensor, mask: torch.Tensor, mu: torch.Tensor,
                   sigma: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """Brute-force reference: build the dense covariance and solve it. TEST ONLY.

    Independent of the closed form, so tests compare two different computations rather
    than one against itself. Honours the mask, so batched masked densities can be
    checked against it directly.
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
            correlation = rho[dimension].to(torch.float64)
            identity = torch.eye(count, dtype=torch.float64, device=z.device)
            ones = torch.ones((count, count), dtype=torch.float64, device=z.device)
            covariance = scale ** 2 * ((1.0 - correlation) * identity
                                       + correlation * ones)
            residual = (values - mean).unsqueeze(-1)
            solved = torch.linalg.solve(covariance, residual)
            total = total - 0.5 * (
                count * math.log(2 * math.pi)
                + torch.logdet(covariance)
                + (residual.transpose(0, 1) @ solved).squeeze())
        totals.append(total.to(z.dtype))
    return torch.stack(totals)


__all__ = ["INIT_RHO", "RHO_MAX", "CoupledNodePrior", "dense_log_prob"]
