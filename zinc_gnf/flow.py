"""Node-feature graph normalizing flow over the frozen autoencoder's embeddings.

The flow models the density of the autoencoder's per-atom node embeddings,
`[B, N, 64]`, with explicit padding masks. Generation draws node embeddings
from the prior, runs the flow backwards, and hands the result to the frozen
decoder, which predicts atom types, pairwise bond logits and per-atom degrees;
`decoding_charge_aware.degree_constrained_bond_decode` then assembles the
molecule.

Structure of one block (12 by default), all permutation-equivariant:

    ActNorm  ->  invertible per-node linear  ->  affine coupling (swap=False)
                                             ->  affine coupling (swap=True)

and around the stack:

    x  --(+ smoothing noise, train only)-->  --whitening-->  blocks  -->  z
    log p(x) = log p_prior(z | c) + log|det J|

Relation to the teammates' TF1 implementation (`molecular_generation/
molecular_flow.py`, `conditional_prior.py`, `n_conditioning.py`) and the
reasons for each deviation:

  * KEPT: half-channel-split affine coupling with both halves updated per
    block; ActNorm rather than BatchNorm (BatchNorm's forward and inverse use
    different statistics and are not exact inverses); `max_log_scale *
    tanh(s / max_log_scale)` clamping of the coupling scale; fully connected
    masked self-attention s/t networks that never see bond features (bonds do
    not exist yet at sampling time); a learned conditional prior
    `N(mu(c), sigma(c)^2)` with a softplus floor and zero-initialized heads;
    FiLM conditioning driven by the same conditioning embedding that feeds the
    prior.
  * CHANGED: ActNorm is initialized data-dependently (Glow-style) from the
    first batch instead of at zero, and every coupling network's output layer
    is zero-initialized, so the whole flow starts as the identity map on
    whitened data and `log|det J| = 0` at step 0. Their zero-initialized
    ActNorm never renormalized anything, which is why they needed
    `max_log_scale=0.25` (their own note: output norm reached 6.8e6 at
    iteration 0 under 2.0). With a real ActNorm the default here is 1.0.
  * CHANGED: one trunk per coupling emits both `s` and `t` (RealNVP/Glow
    practice) instead of two separate networks: half the parameters, same
    expressiveness.
  * ADDED: a fixed diagonal whitening of the embedding space and a learned
    invertible per-node linear layer (the per-node analogue of Glow's 1x1
    convolution) in every block. Both target a measured property of OUR
    latent space that QM9's did not have to contend with: the participation
    ratio is about 5 of 64 dimensions, so the half-split alone would mix
    channels that carry almost no variance.
  * ADDED: optional Gaussian smoothing noise (`noise_sigma`, in units of each
    dimension's training standard deviation). The encoder is deterministic and
    gives symmetry-equivalent atoms bit-identical embeddings (measured: 99.5%
    of symmetry classes), so the empirical node-embedding distribution is
    singular and maximum likelihood on it is ill-posed. The flow therefore
    models the smoothed density, the same reasoning that makes dequantization
    necessary for image flows.

Conditioning is wired in from the start but switched off by default:
`FlowConfig.condition_features = ()` gives an unconditional flow with a single
global learned prior and no FiLM parameters. Setting it to `("n",)` or
`("n", "qed")` turns on FiLM in every coupling network and makes the prior
conditional, with no other change to this module. The unconditional model
deliberately does NOT feed N into its prior (the teammates' "unconditional"
flow did), so that the later N-conditioned model has an honest baseline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F

CONDITION_FEATURES = ("n", "qed")       # order is fixed; a config names a prefix-free subset


# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------
@dataclass
class FlowConfig:
    """Everything that defines the density. Stored in every checkpoint."""

    node_embedding_dim: int = 64            # must equal the autoencoder's embedding width
    num_blocks: int = 12                    # [teammates] 12 coupling steps
    attention_dim: int = 64                 # width of the s/t attention trunk
    num_heads: int = 4                      # [teammates used 2 heads in the AE; 4 here]
    hidden_dim: int = 256                   # [teammates] s/t MLP hidden width
    max_log_scale: float = 1.0              # tanh clamp on the coupling log-scale
    actnorm: bool = True
    invertible_linear: bool = True
    prior_min_sigma: float = 0.1            # [teammates] softplus floor
    prior_init_sigma: float = 1.0           # [teammates] sigma at initialization
    prior_hidden_dim: int = 64              # [teammates] conditional prior MLP width
    condition_features: tuple[str, ...] = ()        # () = unconditional
    condition_embedding_dim: int = 64
    condition_hidden_dim: int = 64
    noise_sigma: float = 0.05               # smoothing noise, in per-dimension std units
    whiten: bool = True
    whiten_std_floor: float = 1e-2          # protects near-constant embedding dimensions

    def __post_init__(self) -> None:
        self.condition_features = tuple(self.condition_features)
        if self.node_embedding_dim % 2 != 0:
            raise ValueError("node_embedding_dim must be even (the coupling splits it in half)")
        if self.num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if self.attention_dim % self.num_heads != 0:
            raise ValueError("attention_dim must be divisible by num_heads")
        unknown = [f for f in self.condition_features if f not in CONDITION_FEATURES]
        if unknown:
            raise ValueError(f"unknown condition features {unknown}; known: {list(CONDITION_FEATURES)}")
        if len(set(self.condition_features)) != len(self.condition_features):
            raise ValueError("condition_features contains duplicates")
        if self.max_log_scale <= 0:
            raise ValueError("max_log_scale must be > 0")
        if self.noise_sigma < 0:
            raise ValueError("noise_sigma must be >= 0")

    @property
    def conditional(self) -> bool:
        return len(self.condition_features) > 0

    @property
    def condition_dim(self) -> int:
        return self.condition_embedding_dim if self.conditional else 0

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["condition_features"] = list(self.condition_features)
        return values

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "FlowConfig":
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown flow settings: {sorted(unknown)}")
        values = dict(values)
        if "condition_features" in values and values["condition_features"] is not None:
            values["condition_features"] = tuple(values["condition_features"])
        return cls(**values)


# ---------------------------------------------------------------------------
# Building blocks.
# ---------------------------------------------------------------------------
def _masked_mean_std(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean and std over the REAL nodes of `x [B, N, D]`."""
    weights = mask.to(x.dtype).unsqueeze(-1)
    count = weights.sum().clamp(min=1.0)
    mean = (x * weights).sum(dim=(0, 1)) / count
    variance = (((x - mean) ** 2) * weights).sum(dim=(0, 1)) / count
    return mean, variance.clamp(min=1e-12).sqrt()


class ActNorm(nn.Module):
    """Per-channel affine normalization, `y = x * exp(log_scale) + bias`.

    Initialized data-dependently from the first training batch so the output is
    zero-mean / unit-variance per channel over the real nodes (Glow). This is
    the fix the teammates wanted but could not make in shared code; it is what
    allows a looser `max_log_scale` than their 0.25.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def initialize(self, x: torch.Tensor, mask: torch.Tensor) -> None:
        mean, std = _masked_mean_std(x, mask)
        self.log_scale.copy_(-torch.log(std))
        self.bias.copy_(-mean / std)
        self.initialized.fill_(True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if bool(self.initialized) is False and self.training:
            self.initialize(x, mask)
        y = x * torch.exp(self.log_scale) + self.bias
        y = y * mask.unsqueeze(-1).to(y.dtype)
        log_det = self.log_scale.sum() * mask.sum(-1).to(y.dtype)       # [B]
        return y, log_det

    def inverse(self, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = (y - self.bias) * torch.exp(-self.log_scale)
        return x * mask.unsqueeze(-1).to(x.dtype)


class InvertiblePointwiseLinear(nn.Module):
    """A learned invertible `D x D` matrix applied to every node independently.

    LU-parameterized (`W = P L U`) so the log-determinant is the sum of
    `log|diag(U)|` and no matrix decomposition is needed per step. Initialized
    to a fixed random permutation (`L = U = I`), i.e. a channel shuffle with
    zero log-determinant, which keeps step 0 an exact identity on the
    distribution's scale while still breaking the fixed half-split.
    """

    def __init__(self, dim: int, generator: torch.Generator | None = None) -> None:
        super().__init__()
        self.dim = int(dim)
        permutation = torch.randperm(dim, generator=generator)
        self.register_buffer("permutation_matrix", torch.eye(dim)[permutation])
        self.register_buffer("lower_mask", torch.tril(torch.ones(dim, dim), diagonal=-1))
        self.register_buffer("upper_mask", torch.triu(torch.ones(dim, dim), diagonal=1))
        self.register_buffer("sign_s", torch.ones(dim))
        self.lower = nn.Parameter(torch.zeros(dim, dim))
        self.upper = nn.Parameter(torch.zeros(dim, dim))
        self.log_s = nn.Parameter(torch.zeros(dim))

    def weight(self) -> torch.Tensor:
        eye = torch.eye(self.dim, device=self.log_s.device, dtype=self.log_s.dtype)
        lower = eye + self.lower * self.lower_mask
        upper = self.upper * self.upper_mask + torch.diag(self.sign_s * torch.exp(self.log_s))
        return self.permutation_matrix @ lower @ upper

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = x @ self.weight().t()
        y = y * mask.unsqueeze(-1).to(y.dtype)
        log_det = self.log_s.sum() * mask.sum(-1).to(y.dtype)
        return y, log_det

    def inverse(self, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        inverse_weight = torch.linalg.inv(self.weight())
        x = y @ inverse_weight.t()
        return x * mask.unsqueeze(-1).to(x.dtype)


class MaskedSelfAttention(nn.Module):
    """Multi-head self-attention over the real nodes of a padded batch.

    Deliberately bond-free: at generation time no bonds exist yet, so the
    coupling networks may only ever look at node features. Padding columns are
    masked out of the attention logits and padding rows are zeroed.
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads, self.head_dim = int(num_heads), dim // int(num_heads)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        for layer in (self.query, self.key, self.value, self.output):
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, width, _ = x.shape
        shape = (batch, width, self.num_heads, self.head_dim)
        queries = self.query(x).view(shape).transpose(1, 2)       # [B, H, N, d]
        keys = self.key(x).view(shape).transpose(1, 2)
        values = self.value(x).view(shape).transpose(1, 2)
        logits = queries @ keys.transpose(-1, -2) / math.sqrt(self.head_dim)
        logits = logits.masked_fill(~mask[:, None, None, :], float("-inf"))
        attended = torch.softmax(logits, dim=-1) @ values
        merged = attended.transpose(1, 2).reshape(batch, width, -1)
        return self.output(merged) * mask.unsqueeze(-1).to(x.dtype)


class CouplingNetwork(nn.Module):
    """`(half of x, conditioning) -> (log-scale, translation)` for the other half.

    One trunk, two heads, output layer zero-initialized so the coupling is the
    identity at step 0. FiLM `(gamma, beta)` from the conditioning embedding
    modulates the hidden layer; its generator is zero-initialized too, so an
    untrained conditional model starts from the unconditional map.
    """

    def __init__(self, in_dim: int, out_dim: int, *, attention_dim: int, num_heads: int,
                 hidden_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(in_dim, attention_dim)
        self.attention = MaskedSelfAttention(attention_dim, num_heads)
        self.hidden = nn.Linear(in_dim + attention_dim, hidden_dim)
        self.film = nn.Linear(condition_dim, 2 * hidden_dim) if condition_dim else None
        self.scale_head = nn.Linear(hidden_dim, out_dim)
        self.translation_head = nn.Linear(hidden_dim, out_dim)
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        for head in (self.scale_head, self.translation_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                condition: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.input_projection(x) * mask.unsqueeze(-1).to(x.dtype)
        attended = self.attention(projected, mask)
        hidden = self.hidden(torch.cat([x, attended], dim=-1))
        if self.film is not None:
            if condition is None:
                raise ValueError("this coupling network is conditional but no condition was given")
            gamma, beta = self.film(condition).unsqueeze(1).chunk(2, dim=-1)
            hidden = hidden * (1.0 + gamma) + beta
        hidden = F.relu(hidden)
        zeros = mask.unsqueeze(-1).to(x.dtype)
        return self.scale_head(hidden) * zeros, self.translation_head(hidden) * zeros


class AffineCoupling(nn.Module):
    """Transforms one channel half, conditioned on the other half of every node."""

    def __init__(self, config: FlowConfig, *, swap: bool) -> None:
        super().__init__()
        self.swap = bool(swap)
        self.max_log_scale = float(config.max_log_scale)
        half = config.node_embedding_dim // 2
        self.network = CouplingNetwork(
            half, half, attention_dim=config.attention_dim, num_heads=config.num_heads,
            hidden_dim=config.hidden_dim, condition_dim=config.condition_dim)

    def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        first, second = x.chunk(2, dim=-1)
        return (second, first) if self.swap else (first, second)

    def _join(self, source: torch.Tensor, transformed: torch.Tensor) -> torch.Tensor:
        parts = (transformed, source) if self.swap else (source, transformed)
        return torch.cat(parts, dim=-1)

    def _scale_and_translation(self, source: torch.Tensor, mask: torch.Tensor,
                               condition: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        raw_scale, translation = self.network(source, mask, condition)
        bound = self.max_log_scale
        return bound * torch.tanh(raw_scale / bound), translation

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                condition: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        source, target = self._split(x)
        scale, translation = self._scale_and_translation(source, mask, condition)
        transformed = (target * torch.exp(scale) + translation) * mask.unsqueeze(-1).to(x.dtype)
        log_det = (scale * mask.unsqueeze(-1).to(x.dtype)).sum(dim=(1, 2))
        return self._join(source, transformed), log_det

    def inverse(self, y: torch.Tensor, mask: torch.Tensor,
                condition: torch.Tensor | None) -> torch.Tensor:
        source, target = self._split(y)
        scale, translation = self._scale_and_translation(source, mask, condition)
        original = ((target - translation) * torch.exp(-scale)) * mask.unsqueeze(-1).to(y.dtype)
        return self._join(source, original)


class FlowBlock(nn.Module):
    """ActNorm -> invertible linear -> two affine couplings."""

    def __init__(self, config: FlowConfig, generator: torch.Generator | None = None) -> None:
        super().__init__()
        dim = config.node_embedding_dim
        self.act_norm = ActNorm(dim) if config.actnorm else None
        self.linear = InvertiblePointwiseLinear(dim, generator) if config.invertible_linear else None
        self.couplings = nn.ModuleList([AffineCoupling(config, swap=False),
                                        AffineCoupling(config, swap=True)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                condition: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        log_det = x.new_zeros(x.shape[0])
        if self.act_norm is not None:
            x, term = self.act_norm(x, mask)
            log_det = log_det + term
        if self.linear is not None:
            x, term = self.linear(x, mask)
            log_det = log_det + term
        for coupling in self.couplings:
            x, term = coupling(x, mask, condition)
            log_det = log_det + term
        return x, log_det

    def inverse(self, y: torch.Tensor, mask: torch.Tensor,
                condition: torch.Tensor | None) -> torch.Tensor:
        for coupling in reversed(self.couplings):
            y = coupling.inverse(y, mask, condition)
        if self.linear is not None:
            y = self.linear.inverse(y, mask)
        if self.act_norm is not None:
            y = self.act_norm.inverse(y, mask)
        return y


# ---------------------------------------------------------------------------
# Conditioning and prior.
# ---------------------------------------------------------------------------
class ConditionEmbedding(nn.Module):
    """`(standardized feature values, present flags) -> [B, embedding_dim]`.

    Each feature arrives standardized (the dataset's train-only scaler already
    stores `log1p(N)` and QED that way) together with a present flag, and an
    absent feature contributes a learned constant instead of a value. That is
    what lets one trained architecture serve N-only and N+QED conditioning, and
    later classifier-free guidance, without touching the coupling layers.
    """

    def __init__(self, features: Sequence[str], embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.features = tuple(features)
        if not self.features:
            raise ValueError("ConditionEmbedding needs at least one feature")
        self.absent = nn.Parameter(torch.zeros(len(self.features)))
        self.mlp = nn.Sequential(
            nn.Linear(2 * len(self.features), hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim))
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.trunc_normal_(layer.bias, std=0.1, a=-0.2, b=0.2)

    def forward(self, values: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != len(self.features):
            raise ValueError(f"expected {len(self.features)} condition features, got {values.shape[-1]}")
        flags = present.to(values.dtype)
        filled = values * flags + self.absent * (1.0 - flags)
        return self.mlp(torch.cat([filled, flags], dim=-1))


class NodePrior(nn.Module):
    """Per-node i.i.d. Gaussian prior, optionally conditional.

    Unconditional: one learned `(mu, sigma)` pair shared by every node of every
    graph, so N enters generation only through how many node vectors are drawn.
    Conditional: `mu(c), sigma(c)` from a small MLP with zero-initialized heads
    and a softplus floor, so sigma starts at exactly `init_sigma` whatever the
    conditioning embedding does (the teammates' `ConditionalPriorNetwork`
    trick, kept because it is the right one).
    """

    def __init__(self, dim: int, *, condition_dim: int = 0, min_sigma: float = 0.1,
                 init_sigma: float = 1.0, hidden_dim: int = 64) -> None:
        super().__init__()
        self.dim, self.min_sigma = int(dim), float(min_sigma)
        if init_sigma <= min_sigma:
            raise ValueError("init_sigma must be greater than min_sigma")
        pre_sigma = math.log(math.expm1(init_sigma - min_sigma))
        self.conditional = condition_dim > 0
        if self.conditional:
            self.hidden = nn.Linear(condition_dim, hidden_dim)
            nn.init.xavier_normal_(self.hidden.weight)
            nn.init.zeros_(self.hidden.bias)
            self.mu_head = nn.Linear(hidden_dim, dim)
            self.sigma_head = nn.Linear(hidden_dim, dim)
            for head, bias in ((self.mu_head, 0.0), (self.sigma_head, pre_sigma)):
                nn.init.zeros_(head.weight)
                nn.init.constant_(head.bias, bias)
        else:
            self.mu = nn.Parameter(torch.zeros(dim))
            self.pre_sigma = nn.Parameter(torch.full((dim,), pre_sigma))

    def parameters_for(self, batch_size: int, condition: torch.Tensor | None,
                       device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.conditional:
            if condition is None:
                raise ValueError("conditional prior requires a conditioning embedding")
            hidden = F.relu(self.hidden(condition))
            mu, pre_sigma = self.mu_head(hidden), self.sigma_head(hidden)
        else:
            mu = self.mu.unsqueeze(0).expand(batch_size, -1)
            pre_sigma = self.pre_sigma.unsqueeze(0).expand(batch_size, -1)
        return mu, F.softplus(pre_sigma) + self.min_sigma

    def log_prob(self, z: torch.Tensor, mask: torch.Tensor,
                 condition: torch.Tensor | None) -> torch.Tensor:
        mu, sigma = self.parameters_for(z.shape[0], condition, z.device)
        mu, sigma = mu.unsqueeze(1), sigma.unsqueeze(1)
        per_dimension = -0.5 * (math.log(2 * math.pi) + 2 * torch.log(sigma)
                                + ((z - mu) / sigma) ** 2)
        return (per_dimension.sum(-1) * mask.to(z.dtype)).sum(-1)       # [B]

    @torch.no_grad()
    def sample(self, mask: torch.Tensor, condition: torch.Tensor | None, *,
               temperature: float = 1.0, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draws are data, never part of a gradient graph, hence `no_grad`."""
        batch, width = mask.shape
        device = mask.device
        mu, sigma = self.parameters_for(batch, condition, device)
        noise = torch.randn((batch, width, self.dim), device=device, dtype=mu.dtype,
                            generator=generator)
        z = mu.unsqueeze(1) + float(temperature) * sigma.unsqueeze(1) * noise
        return z * mask.unsqueeze(-1).to(z.dtype)


# ---------------------------------------------------------------------------
# The flow.
# ---------------------------------------------------------------------------
class NodeFlow(nn.Module):
    """Density model over padded node embeddings, with whitening and prior."""

    def __init__(self, config: FlowConfig | None = None, *, init_seed: int = 0) -> None:
        super().__init__()
        self.config = config or FlowConfig()
        c = self.config
        generator = torch.Generator().manual_seed(int(init_seed))
        self.blocks = nn.ModuleList([FlowBlock(c, generator) for _ in range(c.num_blocks)])
        self.condition_embedding = (
            ConditionEmbedding(c.condition_features, c.condition_embedding_dim, c.condition_hidden_dim)
            if c.conditional else None)
        self.prior = NodePrior(c.node_embedding_dim, condition_dim=c.condition_dim,
                               min_sigma=c.prior_min_sigma, init_sigma=c.prior_init_sigma,
                               hidden_dim=c.prior_hidden_dim)
        # Embedding statistics of the TRAINING split: the smoothing-noise scale and
        # the fixed whitening. Filled by `set_data_statistics` before training and
        # restored from the checkpoint afterwards, never recomputed at generation time.
        self.register_buffer("data_mean", torch.zeros(c.node_embedding_dim))
        self.register_buffer("data_std", torch.ones(c.node_embedding_dim))
        self.register_buffer("statistics_set", torch.zeros((), dtype=torch.bool))

    # -- statistics ----------------------------------------------------
    def set_data_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Store the training embeddings' mean and per-dimension std.

        A std of exactly zero is ACCEPTED: a 64-dimensional embedding from an encoder of
        ReLU MLPs can easily contain a dead unit that is constant over the whole training
        set, and refusing that would abort a GPU run at startup for a dimension that
        carries no information. `whiten_std_floor` handles those dimensions, both in the
        whitening and in the smoothing noise, and `floored_dimensions()` reports how many
        there are. Negative or non-finite statistics are still a bug and are rejected.
        """
        if mean.shape != self.data_mean.shape or std.shape != self.data_std.shape:
            raise ValueError("statistics must have shape [node_embedding_dim]")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std < 0).any():
            raise ValueError("embedding statistics must be finite with non-negative std")
        self.data_mean.copy_(mean.to(self.data_mean.device, self.data_mean.dtype))
        self.data_std.copy_(std.to(self.data_std.device, self.data_std.dtype))
        self.statistics_set.fill_(True)

    @property
    def effective_std(self) -> torch.Tensor:
        """Whitening divisor, floored so near-constant dimensions cannot explode."""
        return self.data_std.clamp(min=float(self.config.whiten_std_floor))

    def floored_dimensions(self) -> int:
        return int((self.data_std < float(self.config.whiten_std_floor)).sum())

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -- conditioning --------------------------------------------------
    def conditioning(self, values: torch.Tensor | None, present: torch.Tensor | None,
                     batch_size: int, device: torch.device | str) -> torch.Tensor | None:
        """Conditioning embedding, or None for the unconditional model."""
        if self.condition_embedding is None:
            return None
        if values is None:
            raise ValueError("this flow is conditional; condition values are required")
        if present is None:
            present = torch.ones_like(values)
        return self.condition_embedding(values.to(device), present.to(device))

    # -- whitening -----------------------------------------------------
    def whiten(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.config.whiten:
            return x, x.new_zeros(x.shape[0])
        std = self.effective_std
        y = (x - self.data_mean) / std
        y = y * mask.unsqueeze(-1).to(y.dtype)
        log_det = -torch.log(std).sum() * mask.sum(-1).to(y.dtype)
        return y, log_det

    def unwhiten(self, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not self.config.whiten:
            return y
        x = y * self.effective_std + self.data_mean
        return x * mask.unsqueeze(-1).to(x.dtype)

    def add_smoothing_noise(self, x: torch.Tensor, mask: torch.Tensor, *,
                            generator: torch.Generator | None = None) -> torch.Tensor:
        """`x + sigma * per-dimension std * eps` on real nodes only.

        The scale is the FLOORED std, the same one the whitening uses. Scaling by the raw
        std would give an exactly-constant dimension exactly zero noise, leaving the
        density a point mass in that direction with unbounded likelihood - the very
        degeneracy this noise exists to remove. After whitening the noise is `sigma` in
        every dimension, which is what makes the objective well posed.
        """
        sigma = float(self.config.noise_sigma)
        if sigma <= 0:
            return x
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        return x + sigma * self.effective_std * noise * mask.unsqueeze(-1).to(x.dtype)

    # -- forward / inverse ---------------------------------------------
    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                condition: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Data -> latent. Returns `(z, log|det J|)` with the log-det per graph."""
        if x.shape[:2] != mask.shape:
            raise ValueError(f"embeddings {tuple(x.shape)} do not match mask {tuple(mask.shape)}")
        x = x * mask.unsqueeze(-1).to(x.dtype)
        z, log_det = self.whiten(x, mask)
        for block in self.blocks:
            z, term = block(z, mask, condition)
            log_det = log_det + term
        return z, log_det

    def inverse(self, z: torch.Tensor, mask: torch.Tensor,
                condition: torch.Tensor | None = None) -> torch.Tensor:
        """Latent -> data (embeddings in the autoencoder's own space)."""
        z = z * mask.unsqueeze(-1).to(z.dtype)
        for block in reversed(self.blocks):
            z = block.inverse(z, mask, condition)
        return self.unwhiten(z, mask)

    def log_prob(self, x: torch.Tensor, mask: torch.Tensor,
                 condition: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        z, log_det = self.forward(x, mask, condition)
        log_prob_z = self.prior.log_prob(z, mask, condition)
        return {"log_prob": log_prob_z + log_det, "log_prob_z": log_prob_z,
                "log_det": log_det, "z": z}

    def negative_log_likelihood(self, x: torch.Tensor, mask: torch.Tensor,
                                condition: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Per-node NLL in nats (the training objective) plus its parts."""
        parts = self.log_prob(x, mask, condition)
        nodes = mask.sum().clamp(min=1).to(parts["log_prob"].dtype)
        mu, sigma = self.prior.parameters_for(x.shape[0], condition, x.device)
        return {
            "loss": -parts["log_prob"].sum() / nodes,
            "log_prob_z_per_node": parts["log_prob_z"].sum() / nodes,
            "log_det_per_node": parts["log_det"].sum() / nodes,
            "z_norm": (parts["z"].norm(dim=-1) * mask).sum() / nodes,
            "prior_sigma_mean": sigma.mean(),
            "n_nodes": nodes,
        }

    @torch.no_grad()
    def sample(self, mask: torch.Tensor, condition: torch.Tensor | None = None, *,
               temperature: float = 1.0, generator: torch.Generator | None = None,
               return_latent: bool = False):
        """Draw node embeddings for the graph sizes encoded in `mask`.

        `return_latent=True` also returns the prior draw, so a caller can verify that
        re-encoding the sample reproduces it (the sampling self-check in `generation.py`).
        """
        z = self.prior.sample(mask, condition, temperature=temperature, generator=generator)
        embeddings = self.inverse(z, mask, condition)
        return (embeddings, z) if return_latent else embeddings


def node_mask_from_counts(n_node: Sequence[int] | torch.Tensor, *, max_nodes: int | None = None,
                          device: torch.device | str = "cpu") -> torch.Tensor:
    """`[B, N]` boolean mask from a list of node counts."""
    counts = torch.as_tensor([int(n) for n in n_node], dtype=torch.long, device=device)
    if counts.numel() == 0:
        raise ValueError("need at least one node count")
    if int(counts.min()) < 1:
        raise ValueError("node counts must be >= 1")
    width = int(max_nodes or counts.max())
    if width < int(counts.max()):
        raise ValueError(f"max_nodes={width} is smaller than the largest count {int(counts.max())}")
    return torch.arange(width, device=device).unsqueeze(0) < counts.unsqueeze(1)


__all__ = [
    "ActNorm",
    "AffineCoupling",
    "CONDITION_FEATURES",
    "ConditionEmbedding",
    "CouplingNetwork",
    "FlowBlock",
    "FlowConfig",
    "InvertiblePointwiseLinear",
    "MaskedSelfAttention",
    "NodeFlow",
    "NodePrior",
    "node_mask_from_counts",
]
