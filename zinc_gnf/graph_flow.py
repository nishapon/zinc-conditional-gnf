"""Stage 22: a conditional normalizing flow over molecular graphs, directly.

WHY THIS EXISTS
---------------
Stages 17-21 all failed the same way. The conditional signal is present in the embedding
space (stage 18: delivered/required displacement 0.665, cosine +0.926) and the decoder
preserves QED differences on REAL embeddings (stage 20: 97.8-99.1% on held-out
molecules), yet realised QED does not follow the request. Stage 20 located the gap: a
real-versus-generated embedding classifier reaches AUC 0.979, so the frozen decoder is
asked to invert points it was never fitted on. Stage 21 then showed that fixing the
decoder's assembly discontinuity makes conditioning WORSE, because pinning the bond set
to the degree sequence everywhere removes the channel the signal was travelling through.

The common factor is a separately trained decoder standing between the conditional model
and the molecule. This module removes it: atoms and bonds are the flow's own variables.

REPRESENTATION
--------------
Exactly the dataset's tensors, dequantized (MoFlow/GraphNVP style):

    atoms   x_a = onehot(X) + s*u        [B, N, 17]        u ~ U[0,1)
    bonds   x_b = onehot(A_ij) + s*u     [B, P, 4]         strict UPPER TRIANGLE only

Modelling only the upper triangle makes bond symmetry STRUCTURAL rather than something
the model has to learn or something a symmetrisation step has to repair - and it is why
the density must count each pair once. There is no virtual atom type: N is given by the
condition, so exactly N atoms are generated and padded rows are excluded from the
likelihood entirely. Exact N is therefore satisfied by construction, which - as this
repository has documented since stage 12 - is structural and is NEVER evidence of learned
conditioning.

Dequantization gives a BOUND, not an exact likelihood:

    log P(G | c)  >=  E_u[ log p(x | c) ]  +  D * log s,     D = 17*N + 4*P_real

so the reported NLL is an upper bound on the discrete NLL. It is NOT comparable with the
embedding flow's continuous density: different representation, different base measure.

FACTORISATION - SELECTABLE SINCE STAGE 25
-----------------------------------------
    bonds_first   log p(G|c) = log p_bond(x_b | c)      + log p_atom(x_a | A, c)
    atoms_first   log p(G|c) = log p_atom(x_a | c)       + log p_bond(x_b | X, c)

Both are EXACT factorisations of the same joint. Neither is incapable of representing
valid molecules, and nothing here claims otherwise: the stage-25 hypothesis is about
learnability, inductive bias and sampling quality, not about expressive power.

What stage 24 measured is the motivation. Bonds-first runs the bond stack with
`extra=None`, so the block that decides bond orders never sees an atom type, while
valence is a constraint between an atom's type and the sum of its incident bond orders.
The whole burden of satisfying it fell on the atom stack, against a 73%-carbon marginal:
89.5% of graphs at N=23 carried an over-valent atom, and best-case relabelling showed
98.4% of those bond patterns were realisable by SOME atom assignment - the bonds were
fine, the atom choice was not.

Atoms-first reverses the two so the bond conditioner receives endpoint chemistry -
identity, formal charge and allowed valence - through `AtomChemistryContext`. The cost is
symmetric and is measured rather than hidden: bond errors now compound on atom errors
instead of the other way round, and the atom stack loses its relational view of the bonds
(replaced, so the comparison is not confounded, by `AtomSequenceSubnet`'s index
convolution and masked set mean).

The default is bonds_first, because the stage-23 checkpoint is the comparison baseline
and must stay loadable and resumable; atoms_first is requested explicitly.

CONDITIONING - THE PART THAT IS MEASURED, NOT ASSUMED
-----------------------------------------------------
No published molecular flow established native target-property conditional generation:
MoFlow optimises QED by latent gradient ascent with a separately trained regressor,
GraphAF and GraphDF use RL. So nothing licenses the assumption that attaching a QED
embedding yields control, and this module is built so the assumption can be falsified.

`c = [N, QED]` (train-only standardised) -> `h_c = MLP(c)`, a cINN-style unconstrained
conditioning network trained jointly (Ardizzone et al.). It enters in two places that are
deliberately separable:

  1. inside EVERY coupling, concatenated to the subnet input, so it modulates the
     transformation and reshapes p(G|c). Invertibility is untouched because c is known in
     both directions.
  2. in the prior, as mu(c) and sigma(c).

These are not interchangeable. A prior-only conditional flow merely shifts the latent
cloud: it can move average QED without reshaping anything - which is exactly the artefact
stage 21 produced (QED up, response slope negative). It is also the cheapest thing for an
optimiser to find. `neutralise_coupling_condition` therefore switches off (1) on a TRAINED
model with zero training and every weight restored, so the prior-only response and the
full response are reported separately.

Conditioning dropout replaces c with a learned null embedding with probability
`condition_dropout`, so one set of weights models p(G|c) and p(G). That gives an exact
within-model unconditional reference and penalises ignoring c. Sampling is NATIVE ONLY:
no guidance, no w>1 extrapolation between conditional and unconditional scores.

Chemistry is deliberately absent. A valence constraint is not an invertible map and
cannot live inside a flow without destroying the density. Raw argmax validity is the
architectural number; any correction is applied and reported separately, outside here.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from zinc_gnf.constants_charge_aware import (
    ATOM_VALENCE,
    CHARGE_OF_TYPE,
    MAX_NODES,
    NUM_ATOM_TYPES,
    NUM_BOND_TYPES,
)

LOG_SIGMA_CLAMP = 5.0
ACTNORM_EPSILON = 1e-6
BOND_CONDITIONERS = ("topology", "pairwise")
# Stage 25. Both are exact factorisations of the same joint - this is a claim about
# learnability and inductive bias, NOT about what a flow can represent.
FACTORISATIONS = ("bonds_first", "atoms_first")
ATOM_CONTEXT_MODES = ("dequantized", "onehot")


# --------------------------------------------------------- upper-triangle plumbing ---
def upper_triangle_indices(width: int, device: torch.device | str = "cpu"
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Row/column indices of the strict upper triangle, in a fixed order."""
    rows, cols = torch.triu_indices(width, width, offset=1, device=torch.device(device))
    return rows, cols


def pair_mask_from_counts(counts: Sequence[int] | torch.Tensor, width: int,
                          device: torch.device | str = "cpu") -> torch.Tensor:
    """[B, P] - a pair is real only when BOTH its atoms are real."""
    values = torch.as_tensor([int(c) for c in counts], dtype=torch.long,
                             device=torch.device(device))
    rows, cols = upper_triangle_indices(width, device)
    limit = values.unsqueeze(1)
    return (rows.unsqueeze(0) < limit) & (cols.unsqueeze(0) < limit)


def pack_upper(dense: torch.Tensor, width: int) -> torch.Tensor:
    """[B, N, N] -> [B, P], the strict upper triangle in index order."""
    rows, cols = upper_triangle_indices(width, dense.device)
    return dense[:, rows, cols]


def unpack_upper(values: torch.Tensor, width: int) -> torch.Tensor:
    """[B, P, C] -> [B, N, N, C], symmetric by construction, zero diagonal."""
    batch, _, channels = values.shape
    rows, cols = upper_triangle_indices(width, values.device)
    dense = values.new_zeros((batch, width, width, channels))
    dense[:, rows, cols] = values
    dense[:, cols, rows] = values
    return dense


# ------------------------------------------------------------------------ layers ---
class MaskedActNorm(nn.Module):
    """Per-channel affine normalisation, initialised from the first masked batch."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("initialised", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def initialise(self, x: torch.Tensor, mask: torch.Tensor) -> None:
        weights = mask.unsqueeze(-1).to(x.dtype)
        total = weights.sum().clamp(min=1.0)
        mean = (x * weights).sum(dim=(0, 1)) / total
        variance = (((x - mean) ** 2) * weights).sum(dim=(0, 1)) / total
        self.bias.copy_(-mean)
        self.log_scale.copy_(-0.5 * torch.log(variance + ACTNORM_EPSILON))
        self.initialised.fill_(True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training and not bool(self.initialised):
            self.initialise(x, mask)
        weights = mask.unsqueeze(-1).to(x.dtype)
        y = (x + self.bias) * torch.exp(self.log_scale) * weights
        logdet = self.log_scale.sum() * mask.to(x.dtype).sum(dim=1)
        return y, logdet

    def inverse(self, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(y.dtype)
        return (y * torch.exp(-self.log_scale) - self.bias) * weights


class MaskedInvertibleLinear(nn.Module):
    """An invertible mixing of the channel axis - Glow's 1x1 convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        weight = torch.linalg.qr(torch.randn(channels, channels))[0]
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor, mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = mask.unsqueeze(-1).to(x.dtype)
        y = (x @ self.weight) * weights
        logdet = torch.slogdet(self.weight)[1] * mask.to(x.dtype).sum(dim=1)
        return y, logdet

    def inverse(self, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(y.dtype)
        inverse = torch.linalg.inv(self.weight.to(torch.float64)).to(y.dtype)
        return (y @ inverse) * weights


def _mlp(inputs: int, hidden: int, outputs: int) -> nn.Sequential:
    net = nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(),
                        nn.Linear(hidden, hidden), nn.ReLU(),
                        nn.Linear(hidden, outputs))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)        # every coupling starts as the identity
    return net


class BondSubnet(nn.Module):
    """Pair-wise conditioner that is SYMMETRIC in (i, j) by construction.

    For pair (i, j) it sees its own pass-through channels plus two endpoint aggregates,
    `r_i + r_j` and `|r_i - r_j|`, where `r` sums a channel over the pairs incident on an
    atom. Both are symmetric functions of the unordered pair, so the model cannot learn an
    orientation - which is what keeps the bond tensor symmetric without a repair step.
    """

    def __init__(self, passthrough: int, outputs: int, hidden: int,
                 condition_dim: int, width: int, *, context_channels: int = 0) -> None:
        super().__init__()
        self.width = int(width)
        rows, cols = upper_triangle_indices(width)
        self.register_buffer("rows", rows)
        self.register_buffer("cols", cols)
        self.context_channels = int(context_channels)
        self.net = _mlp(3 * passthrough + condition_dim + self.context_channels,
                        hidden, 2 * outputs)
        self.outputs = int(outputs)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, context: torch.Tensor,
                extra: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = mask.unsqueeze(-1).to(x.dtype)
        masked = x * weights
        batch, _, channels = masked.shape
        totals = masked.new_zeros((batch, self.width, channels))
        totals.index_add_(1, self.rows, masked)
        totals.index_add_(1, self.cols, masked)
        left, right = totals[:, self.rows], totals[:, self.cols]
        features = [masked, left + right, (left - right).abs(),
                    context.unsqueeze(1).expand(-1, masked.shape[1], -1)]
        if self.context_channels:
            if extra is None:
                raise ValueError("this BondSubnet was built with atom context channels "
                                 "and must be given the per-pair atom context")
            # atoms-first: the per-pair atom context, gathered for this pair only
            features.append(extra[:, self.rows, self.cols] * weights)
        scale, shift = self.net(torch.cat(features, dim=-1)).chunk(2, dim=-1)
        return scale, shift


class AtomSubnet(nn.Module):
    """Atom conditioner with one relational message-passing step over the bonds.

    This is MoFlow's "graph conditional flow for atoms given bonds" in miniature: the
    message for atom i is `sum_c sum_j A_c(i,j) * W_c * x_pass(j)`, so an atom's
    transformation depends on what it is bonded to and with which bond order.
    """

    def __init__(self, passthrough: int, outputs: int, hidden: int, condition_dim: int,
                 num_bond_types: int) -> None:
        super().__init__()
        self.relational = nn.Parameter(
            torch.randn(num_bond_types, passthrough, hidden) * (1.0 / max(passthrough, 1)) ** 0.5)
        self.net = _mlp(passthrough + hidden + condition_dim, hidden, 2 * outputs)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, context: torch.Tensor,
                bonds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = mask.unsqueeze(-1).to(x.dtype)
        masked = x * weights
        # bonds: [B, N, N, C_bond]; projected: [B, C_bond, N, hidden]
        projected = torch.einsum("bnp,cph->bcnh", masked, self.relational)
        message = torch.einsum("bijc,bcjh->bih", bonds, projected)
        features = torch.cat([masked, message,
                              context.unsqueeze(1).expand(-1, masked.shape[1], -1)], dim=-1)
        scale, shift = self.net(features).chunk(2, dim=-1)
        return scale, shift


class AtomSequenceSubnet(nn.Module):
    """Stage 25: the atom conditioner for the ATOMS-FIRST factorisation.

    In bonds-first, `AtomSubnet` conditions an atom on the bonds around it. Atoms-first
    has no bonds yet, so that conditioner is unavailable - and replacing it with a plain
    per-atom MLP would make `p(X|c)` a product of INDEPENDENT per-atom densities, unable
    to represent composition (``exactly one sulphur, three nitrogens''). That is strictly
    weaker than what stage 23 had, and a comparison against it would confound the
    factorisation with a crippled atom model. So the atom stack keeps cross-atom
    structure through two masked, pass-through-only paths:

      LOCAL   a 1-D convolution along the atom INDEX axis. Atom order is canonical SMILES
              order - a DFS walk - so index-adjacent atoms are usually bonded. This is the
              same inductive bias, and the same stated non-equivariance tradeoff, that
              `TopologyBondSubnet` already relies on.
      GLOBAL  the masked MEAN over real atoms, broadcast back (a Deep-Sets aggregate), so
              every atom's transformation can depend on the whole composition.

    Invertibility is untouched: the conditioner reads ONLY the coupling's pass-through
    channels, which the coupling leaves unchanged, so the inverse recomputes the same
    scale and shift. Padded atoms are zeroed before and after every convolution and the
    mask is supplied as its own channel, so padding contributes nothing but zeros.
    """

    def __init__(self, passthrough: int, outputs: int, hidden: int, condition_dim: int,
                 *, layers: int = 3, kernel: int = 3) -> None:
        super().__init__()
        padding = int(kernel) // 2
        self.stem = nn.Conv1d(passthrough + 1, hidden, int(kernel), padding=padding)
        self.context = nn.Linear(condition_dim, hidden)
        self.blocks = nn.ModuleList([
            nn.Conv1d(hidden, hidden, int(kernel), padding=padding)
            for _ in range(max(0, int(layers) - 1))])
        self.pooled = nn.Conv1d(2 * hidden, hidden, 1)
        self.head = nn.Conv1d(hidden, 2 * outputs, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)      # every coupling starts as the identity

    def forward(self, x: torch.Tensor, mask: torch.Tensor, context: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = mask.unsqueeze(-1).to(x.dtype)
        lanes = torch.cat([x * weights, weights], dim=-1).transpose(1, 2)
        keep = weights.transpose(1, 2)
        hidden = torch.relu(self.stem(lanes)
                            + self.context(context).unsqueeze(-1)) * keep
        for block in self.blocks:
            hidden = torch.relu(block(hidden)) * keep
        counts = keep.sum(dim=-1, keepdim=True).clamp(min=1.0)
        pooled = (hidden * keep).sum(dim=-1, keepdim=True) / counts
        hidden = torch.relu(self.pooled(torch.cat(
            [hidden, pooled.expand_as(hidden)], dim=1))) * keep
        out = self.head(hidden).transpose(1, 2)
        scale, shift = out.chunk(2, dim=-1)
        return scale, shift


class AtomChemistryContext(nn.Module):
    """Stage 25: per-pair ATOM context for the bond stack. The point of atoms-first.

    Stage 24 established that the bonds-first bond stack ran with `extra=None` and so
    never saw an atom type, while valence is a constraint between an atom's TYPE and the
    sum of its incident bond orders. This module is what the reversed factorisation makes
    available: for every pair (i, j) it supplies features of both endpoints, including the
    two quantities the constraint is actually written in - the allowed valence and the
    formal charge of each endpoint, read from the charge-aware vocabulary.

    WHAT GOES IN, AND WHY IT IS CONTINUOUS
    --------------------------------------
    `mode="dequantized"` (the default) uses the dequantized atom channels exactly as the
    likelihood sees them. Every operation here is then linear or smooth, so NOTHING
    non-differentiable and nothing non-invertible enters the likelihood path, and training
    and sampling use the same quantity: the atom tensor the atom stack produced.

    `mode="onehot"` instead uses `argmax` of those channels. That is a legitimate
    conditional - `p(x_b | argmax(x_a), c)` is still a normalised density in `x_b` for
    each fixed `x_a`, so the joint is a valid density and the dequantization bound is
    unaffected - and it matches decoding, which argmaxes anyway. Its cost is stated rather
    than hidden: the density becomes piecewise in `x_a`, with discontinuities across
    argmax boundaries. It is offered as a LABELLED variant, not the default.

    The per-pair features are symmetric in (i, j) by construction - sums and absolute
    differences - so the context cannot teach the conditioner an orientation. Padded rows
    and columns are zeroed. The diagonal carries `2*f_i` and `0`, which is a well-defined
    per-atom channel and is left in place.
    """

    def __init__(self, num_atom_types: int, *, embed_dim: int,
                 mode: str = "dequantized") -> None:
        super().__init__()
        if mode not in ATOM_CONTEXT_MODES:
            raise ValueError(f"unknown atom_context_mode {mode!r}; "
                             f"known: {list(ATOM_CONTEXT_MODES)}")
        self.num_atom_types = int(num_atom_types)
        self.mode = str(mode)
        caps = torch.as_tensor(ATOM_VALENCE[:num_atom_types], dtype=torch.float32)
        charges = torch.as_tensor(CHARGE_OF_TYPE[:num_atom_types], dtype=torch.float32)
        # scaled to O(1) so the conditioner does not have to absorb the units
        self.register_buffer("caps", caps / caps.max().clamp(min=1.0))
        self.register_buffer("charges", charges)
        self.embed = nn.Linear(self.num_atom_types + 2, int(embed_dim))
        self.channels = 2 * int(embed_dim) + 4

    def atom_features(self, atoms: torch.Tensor, node_mask: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, N, 17+2] atom features and the [B, N, 2] raw chemistry pair."""
        weights = node_mask.unsqueeze(-1).to(atoms.dtype)
        values = atoms * weights
        if self.mode == "onehot":
            hard = F.one_hot(values.argmax(dim=-1), self.num_atom_types).to(atoms.dtype)
            values = hard * weights
        cap = (values * self.caps).sum(dim=-1, keepdim=True)
        charge = (values * self.charges).sum(dim=-1, keepdim=True)
        chemistry = torch.cat([cap, charge], dim=-1)
        return torch.cat([values, chemistry], dim=-1), chemistry

    def forward(self, atoms: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """-> [B, N, N, channels], symmetric, masked, zero outside the real atoms."""
        features, chemistry = self.atom_features(atoms, node_mask)
        weights = node_mask.unsqueeze(-1).to(atoms.dtype)
        embedded = self.embed(features) * weights

        def pairwise(values: torch.Tensor) -> torch.Tensor:
            left = values.unsqueeze(2)
            right = values.unsqueeze(1)
            return torch.cat([left + right, (left - right).abs()], dim=-1)

        grid = torch.cat([pairwise(embedded), pairwise(chemistry)], dim=-1)
        both_real = weights.unsqueeze(2) * weights.unsqueeze(1)
        return grid * both_real


class TopologyBondSubnet(nn.Module):
    """Stage 23: a masked 2-D convolutional conditioner over the adjacency grid.

    WHY A CONVOLUTION AND NOT MESSAGE PASSING
    -----------------------------------------
    `BondSubnet` gives a pair only `r_i + r_j` and `|r_i - r_j|`, and stage 22c proved it
    blind: a six-ring and two three-rings get byte-identical scales and shifts. The
    obvious repair - iterate the pair -> atom -> pair aggregation a few times - does NOT
    fix it. Both graphs are 2-regular and are the textbook counterexample to 1-WL: every
    atom has degree 2, every neighbour has degree 2, the colouring is stable immediately,
    so ANY sum-aggregation message-passing scheme over the atom graph assigns identical
    atom embeddings in both and any pair feature derived from them is identical too.

    A convolution over the N x N grid is not 1-WL-bounded, because it reads the adjacency
    pattern as a function of atom INDEX. Pair (0,1) sits at grid position (0,1) and, after
    a few 3x3 layers, its receptive field covers the positions where the two graphs
    actually differ - {(2,3), (5,0)} versus {(2,0), (5,3)}. That is precisely the
    mechanism MoFlow used (10 coupling layers of 3x3 convolutions at 512 hidden,
    82% validity without correction on ZINC250K).

    THE TRADEOFF, STATED. Using the index grid means the model is not permutation
    equivariant: its density is over CANONICALLY ORDERED graphs. That is already true of
    this project's representation - `chemistry_charge_aware` fixes the atom order to
    canonical SMILES order - and of MoFlow and GraphNVP, so it adds no new limitation. It
    does mean the receptive field's notion of "nearby" is index locality; canonical SMILES
    order is a DFS walk, so bonded atoms do tend to receive nearby indices, which is what
    makes the inductive bias useful rather than arbitrary.

    INVERTIBILITY IS UNTOUCHED. The conditioner reads ONLY the coupling's pass-through
    channels, which the coupling leaves unchanged, so the inverse recomputes exactly the
    same scale and shift. The transformed (active) channels never enter here - that is
    enforced by `ConditionalCoupling._split` and pinned by a test.

    SYMMETRY. The convolution output is averaged with its transpose, so a pair's
    parameters do not depend on which of (i,j) or (j,i) names it. Matrix symmetry itself
    does not rely on this: only the strict upper triangle is modelled and `unpack_upper`
    mirrors it. That realisation is what frees the conditioner to be arbitrary at all.

    MASKING. Padded rows and columns are zeroed before every convolution and the result
    is re-masked after it, so padding contributes nothing but zeros. A real pair near the
    padding boundary therefore sees fewer non-zero neighbours, exactly as zero-padding
    implies - the same convention MoFlow's virtual atoms produce.
    """

    def __init__(self, passthrough: int, outputs: int, hidden: int, condition_dim: int,
                 width: int, *, layers: int = 3, kernel: int = 3,
                 context_channels: int = 0) -> None:
        super().__init__()
        self.width = int(width)
        self.passthrough = int(passthrough)
        self.outputs = int(outputs)
        self.context_channels = int(context_channels)
        rows, cols = upper_triangle_indices(width)
        self.register_buffer("rows", rows)
        self.register_buffer("cols", cols)
        padding = int(kernel) // 2
        self.stem = nn.Conv2d(passthrough + 1 + self.context_channels, hidden,
                              int(kernel), padding=padding)
        self.context = nn.Linear(condition_dim, hidden)
        self.blocks = nn.ModuleList([
            nn.Conv2d(hidden, hidden, int(kernel), padding=padding)
            for _ in range(max(0, int(layers) - 1))])
        # Axial aggregation: the masked row and column means of the hidden map, broadcast
        # back. For pair (i,j) these are summaries of everything incident on atom i and on
        # atom j, which is the information stage 22's `r_i + r_j` carried - useful for
        # valence and degree, just insufficient on its own. Keeping it ALONGSIDE the
        # convolution means the conditioner has both endpoint-global and local structure,
        # rather than trading one for the other.
        # 1x1: the row and column means are already global along their axis, so a further
        # spatial mixing here would only add parameters. The local structure was already
        # mixed by the convolution stack above.
        self.axial = nn.Conv2d(3 * hidden, hidden, 1)
        self.head = nn.Conv2d(hidden, 2 * outputs, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)      # every coupling starts as the identity

    def forward(self, x: torch.Tensor, mask: torch.Tensor, context: torch.Tensor,
                extra: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = mask.unsqueeze(-1).to(x.dtype)
        dense = unpack_upper(x * weights, self.width).permute(0, 3, 1, 2)
        grid = unpack_upper(weights, self.width).permute(0, 3, 1, 2)
        channels = [dense, grid]
        if self.context_channels:
            if extra is None:
                raise ValueError("this TopologyBondSubnet was built with atom context "
                                 "channels and must be given the per-pair atom context")
            # atoms-first: the atom grid enters as extra INPUT channels of the same
            # convolution, so endpoint chemistry and local topology are read together
            channels.append(extra.permute(0, 3, 1, 2))
        features = torch.cat(channels, dim=1)
        hidden = torch.relu(self.stem(features)
                            + self.context(context).unsqueeze(-1).unsqueeze(-1)) * grid
        for block in self.blocks:
            hidden = torch.relu(block(hidden)) * grid
        counts = grid.sum(dim=3, keepdim=True).clamp(min=1.0)
        row = (hidden * grid).sum(dim=3, keepdim=True) / counts
        column = row.transpose(-1, -2)
        hidden = torch.relu(self.axial(torch.cat(
            [hidden, row.expand_as(hidden), column.expand_as(hidden)], dim=1))) * grid
        out = self.head(hidden)
        out = 0.5 * (out + out.transpose(-1, -2))
        gathered = out[:, :, self.rows, self.cols].permute(0, 2, 1)
        scale, shift = gathered.chunk(2, dim=-1)
        return scale, shift


class ConditionalCoupling(nn.Module):
    """Affine coupling whose scale and shift depend on the condition.

    The log-scale is tanh-bounded. This flow is ONE-SHOT, so the sequential float32
    overflow stage 18 traced in the node-autoregressive embedding flow cannot occur here -
    but the bound still keeps the inverse well conditioned, and the invertibility test
    pins it.
    """

    def __init__(self, channels: int, subnet: nn.Module, *, passthrough: int,
                 max_log_scale: float, max_shift: float = 10.0,
                 flip: bool = False) -> None:
        super().__init__()
        self.channels = int(channels)
        self.passthrough = int(passthrough)
        self.subnet = subnet
        self.max_log_scale = float(max_log_scale)
        self.max_shift = float(max_shift)
        self.flip = bool(flip)
        self.register_buffer("condition_enabled", torch.ones((), dtype=torch.bool))

    def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.flip:
            return x[..., -self.passthrough:], x[..., :-self.passthrough]
        return x[..., :self.passthrough], x[..., self.passthrough:]

    def _join(self, passthrough: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        if self.flip:
            return torch.cat([active, passthrough], dim=-1)
        return torch.cat([passthrough, active], dim=-1)

    def _scale_shift(self, passthrough: torch.Tensor, mask: torch.Tensor,
                     context: torch.Tensor, extra: torch.Tensor | None):
        if not bool(self.condition_enabled):
            context = torch.zeros_like(context)
        if extra is None:
            scale, shift = self.subnet(passthrough, mask, context)
        else:
            scale, shift = self.subnet(passthrough, mask, context, extra)
        # BOTH the scale and the SHIFT are bounded. Stage 18 traced a float32 failure in
        # the embedding flow to an unbounded translation head; this flow is one-shot so it
        # cannot overflow the same way, but a large shift still wrecks the INVERSE: it
        # inflates the intermediate magnitudes block over block, and since sampling IS the
        # inverse, the generated molecules stop coming from the trained density while the
        # training loss still looks healthy. The soft clamp is the identity for the shifts
        # this data needs and saturates smoothly beyond them.
        bounded_shift = self.max_shift * torch.tanh(shift / self.max_shift)
        return self.max_log_scale * torch.tanh(scale), bounded_shift

    def forward(self, x: torch.Tensor, mask: torch.Tensor, context: torch.Tensor,
                extra: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        passthrough, active = self._split(x)
        scale, shift = self._scale_shift(passthrough, mask, context, extra)
        weights = mask.unsqueeze(-1).to(x.dtype)
        transformed = (active * torch.exp(scale) + shift) * weights
        logdet = (scale * weights).sum(dim=(1, 2))
        return self._join(passthrough, transformed) * weights, logdet

    def inverse(self, y: torch.Tensor, mask: torch.Tensor, context: torch.Tensor,
                extra: torch.Tensor | None = None) -> torch.Tensor:
        passthrough, active = self._split(y)
        scale, shift = self._scale_shift(passthrough, mask, context, extra)
        weights = mask.unsqueeze(-1).to(y.dtype)
        recovered = ((active - shift) * torch.exp(-scale)) * weights
        return self._join(passthrough, recovered) * weights


class ConditionalPrior(nn.Module):
    """N(mu(c), sigma(c)^2) per channel, starting at the standard normal."""

    def __init__(self, channels: int, condition_dim: int, conditional: bool) -> None:
        super().__init__()
        self.base_mu = nn.Parameter(torch.zeros(channels))
        self.base_log_sigma = nn.Parameter(torch.zeros(channels))
        self.conditional = bool(conditional)
        if conditional:
            self.mu_head = nn.Linear(condition_dim, channels)
            self.log_sigma_head = nn.Linear(condition_dim, channels)
            for head in (self.mu_head, self.log_sigma_head):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def parameters_for(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.base_mu.unsqueeze(0).expand(context.shape[0], -1)
        log_sigma = self.base_log_sigma.unsqueeze(0).expand(context.shape[0], -1)
        if self.conditional:
            mu = mu + self.mu_head(context)
            log_sigma = log_sigma + self.log_sigma_head(context)
        return mu, log_sigma.clamp(-LOG_SIGMA_CLAMP, LOG_SIGMA_CLAMP)

    def log_prob(self, z: torch.Tensor, mask: torch.Tensor,
                 context: torch.Tensor) -> torch.Tensor:
        mu, log_sigma = self.parameters_for(context)
        mu = mu.unsqueeze(1)
        log_sigma = log_sigma.unsqueeze(1)
        weights = mask.unsqueeze(-1).to(z.dtype)
        normalised = (z - mu) * torch.exp(-log_sigma)
        density = -0.5 * (normalised ** 2) - log_sigma - 0.5 * float(np.log(2.0 * np.pi))
        return (density * weights).sum(dim=(1, 2))

    def sample(self, mask: torch.Tensor, context: torch.Tensor, *, temperature: float,
               channels: int, generator: torch.Generator | None = None) -> torch.Tensor:
        mu, log_sigma = self.parameters_for(context)
        shape = (mask.shape[0], mask.shape[1], channels)
        noise = torch.randn(shape, device=mask.device, dtype=mu.dtype, generator=generator)
        z = mu.unsqueeze(1) + float(temperature) * torch.exp(log_sigma).unsqueeze(1) * noise
        return z * mask.unsqueeze(-1).to(z.dtype)


# ------------------------------------------------------------------------- model ---
@dataclass
class GraphFlowConfig:
    max_nodes: int = MAX_NODES
    num_atom_types: int = NUM_ATOM_TYPES
    num_bond_types: int = NUM_BOND_TYPES
    bond_blocks: int = 6
    atom_blocks: int = 6
    hidden: int = 256
    # STAGE 25. "bonds_first" is stages 22-24: p(A|c) p(X|A,c). "atoms_first" is
    # p(X|c) p(A|X,c), which is what gives the bond stack endpoint chemistry.
    #
    # The DEFAULT STAYS bonds_first on purpose. The stage-23 18,000-step checkpoint is
    # the comparison baseline for stage 25; if the default flipped, re-submitting stage
    # 23's own driver would build a different architecture and `_resume` would refuse to
    # continue that checkpoint. The stage-25 driver passes --factorisation atoms_first
    # explicitly, and `from_dict` maps a checkpoint with no entry to bonds_first.
    factorisation: str = "bonds_first"
    # atoms_first only: the width of the atom stack's own conditioner, and of the
    # per-atom embedding that reaches the bond stack. `atom_context_mode` is the explicit
    # choice between the continuous dequantized channels (default; nothing
    # non-differentiable in the likelihood path) and hard argmax identities.
    # 128 is chosen so the atoms-first model lands at 1,258,778 parameters against the
    # stage-23 bonds-first baseline's 1,309,690 - 0.961x, strictly FEWER. The comparison
    # is then not confoundable by capacity in the direction that would flatter stage 25,
    # and it is close enough that a loss cannot be dismissed as starvation either.
    atom_conv_hidden: int = 128
    atom_conv_layers: int = 3
    atom_conv_kernel: int = 3
    atom_context_dim: int = 8
    atom_context_mode: str = "dequantized"
    # "topology" is the stage-23 masked 2-D convolution over the adjacency grid.
    # "pairwise" is the stage-22 conditioner, kept so stage-22 checkpoints stay loadable
    # and usable as a baseline; `from_dict` maps a checkpoint with no entry to it.
    bond_conditioner: str = "topology"
    # 48 channels puts the whole model at 1,309,690 parameters against the stage-22
    # pairwise model's 1,500,538 - strictly FEWER. That matters: it means any improvement
    # cannot be attributed to added capacity, only to the conditioner seeing topology.
    conv_hidden: int = 48
    conv_layers: int = 3
    conv_kernel: int = 3
    condition_features: tuple[str, ...] = ("n", "qed")
    condition_dim: int = 64
    condition_hidden: int = 128
    condition_dropout: float = 0.1
    conditional_prior: bool = True
    noise_scale: float = 0.6
    # The CUMULATIVE log-scale bounds how badly the inverse can amplify float32 rounding
    # error: a stack of K couplings each bounded by L has inverse condition number up to
    # exp(K*L), and float32 carries ~1e-7 relative precision. At L=2.0 and K=10 that is
    # exp(20) ~ 5e8 and the round trip is destroyed - measured, not assumed. L=0.75 with
    # K=6 gives exp(4.5) ~ 90, comfortably inside precision, and matches the 1.0 the
    # embedding flow already uses. `sample` reports the round-trip error so a badly
    # conditioned configuration is detected rather than silently emitting garbage.
    max_log_scale: float = 0.75
    # The shift is bounded to the scale of the data it moves. The dequantized variables
    # live in [0, 1 + noise_scale] and the latents are ~N(0, 1), so +-3 is ample. A large
    # bound here is actively harmful and was measured to be so: with +-10 the intermediate
    # magnitudes inflate block over block, and the inverse - which is what SAMPLING uses -
    # loses so much precision that the round-trip error reaches 0.18 relative. The atom
    # stack, with 17 channels, failed before the bond stack did.
    max_shift: float = 3.0

    def to_dict(self) -> dict[str, Any]:
        return {"max_nodes": self.max_nodes, "num_atom_types": self.num_atom_types,
                "num_bond_types": self.num_bond_types, "bond_blocks": self.bond_blocks,
                "atom_blocks": self.atom_blocks, "hidden": self.hidden,
                "factorisation": self.factorisation,
                # the atom-context settings describe nothing in a bonds-first model - no
                # such module is built - so they are omitted there. A bonds-first config
                # written now is then identical to a stage-23 one, which is what keeps the
                # baseline resumable whatever these (inert) flags were set to.
                **({"atom_conv_hidden": self.atom_conv_hidden,
                    "atom_conv_layers": self.atom_conv_layers,
                    "atom_conv_kernel": self.atom_conv_kernel,
                    "atom_context_dim": self.atom_context_dim,
                    "atom_context_mode": self.atom_context_mode}
                   if self.factorisation == "atoms_first" else {}),
                "bond_conditioner": self.bond_conditioner,
                "conv_hidden": self.conv_hidden, "conv_layers": self.conv_layers,
                "conv_kernel": self.conv_kernel,
                "condition_features": list(self.condition_features),
                "condition_dim": self.condition_dim,
                "condition_hidden": self.condition_hidden,
                "condition_dropout": self.condition_dropout,
                "conditional_prior": self.conditional_prior,
                "noise_scale": self.noise_scale, "max_log_scale": self.max_log_scale,
                "max_shift": self.max_shift}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GraphFlowConfig":
        """Rebuild a config, normalising anything a pre-stage-23 checkpoint lacks.

        A stage-22 checkpoint has no `bond_conditioner` entry and was trained with the
        pairwise conditioner, so it must map to "pairwise" rather than to the current
        default - otherwise loading it would silently build a different architecture.
        Stage 22-24 checkpoints likewise have no `factorisation` entry and were all
        bonds-first, which is also the dataclass default; it is pinned here explicitly so
        the mapping survives a later change of default. Normalising a stored config
        through this method is also what lets the trainer compare an OLD checkpoint's
        config with a current one without the new keys reading as differences.
        """
        data = dict(payload)
        data["condition_features"] = tuple(data.get("condition_features", ("n", "qed")))
        data.setdefault("bond_conditioner", "pairwise")
        data.setdefault("factorisation", "bonds_first")
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ConditionalGraphFlow(nn.Module):
    """p(atoms, bonds | N, QED) as an invertible map, with no decoder anywhere."""

    def __init__(self, config: GraphFlowConfig) -> None:
        super().__init__()
        self.config = config
        width = config.max_nodes
        self.condition_net = nn.Sequential(
            nn.Linear(len(config.condition_features), config.condition_hidden), nn.ReLU(),
            nn.Linear(config.condition_hidden, config.condition_dim), nn.Tanh())
        self.null_condition = nn.Parameter(torch.zeros(config.condition_dim))

        def stack(channels: int, blocks: int, make_subnet) -> nn.ModuleList:
            layers = nn.ModuleList()
            passthrough = max(1, channels // 2)
            for index in range(blocks):
                layers.append(nn.ModuleDict({
                    "actnorm": MaskedActNorm(channels),
                    "mixing": MaskedInvertibleLinear(channels),
                    "coupling": ConditionalCoupling(
                        channels,
                        make_subnet(passthrough, channels - passthrough),
                        passthrough=passthrough,
                        max_log_scale=config.max_log_scale,
                        max_shift=config.max_shift,
                        flip=bool(index % 2)),
                }))
            return layers

        if config.bond_conditioner not in BOND_CONDITIONERS:
            raise ValueError(f"unknown bond_conditioner {config.bond_conditioner!r}; "
                             f"known: {list(BOND_CONDITIONERS)}")
        if config.factorisation not in FACTORISATIONS:
            raise ValueError(f"unknown factorisation {config.factorisation!r}; "
                             f"known: {list(FACTORISATIONS)}")
        self.atoms_first = config.factorisation == "atoms_first"

        # Built ONLY for atoms-first, so a bonds-first state_dict is byte-for-byte the
        # shape it was in stages 22-24 and those checkpoints keep loading.
        self.atom_context = (
            AtomChemistryContext(config.num_atom_types,
                                 embed_dim=config.atom_context_dim,
                                 mode=config.atom_context_mode)
            if self.atoms_first else None)
        context_channels = self.atom_context.channels if self.atoms_first else 0

        def bond_subnet(passthrough: int, outputs: int) -> nn.Module:
            if config.bond_conditioner == "topology":
                return TopologyBondSubnet(passthrough, outputs, config.conv_hidden,
                                          config.condition_dim, width,
                                          layers=config.conv_layers,
                                          kernel=config.conv_kernel,
                                          context_channels=context_channels)
            return BondSubnet(passthrough, outputs, config.hidden, config.condition_dim,
                              width, context_channels=context_channels)

        def atom_subnet(passthrough: int, outputs: int) -> nn.Module:
            if self.atoms_first:
                # no bonds exist yet; cross-atom structure comes from the index
                # convolution and the masked set mean instead
                return AtomSequenceSubnet(passthrough, outputs, config.atom_conv_hidden,
                                          config.condition_dim,
                                          layers=config.atom_conv_layers,
                                          kernel=config.atom_conv_kernel)
            return AtomSubnet(passthrough, outputs, config.hidden, config.condition_dim,
                              config.num_bond_types)

        self.bond_layers = stack(config.num_bond_types, config.bond_blocks, bond_subnet)
        self.atom_layers = stack(config.num_atom_types, config.atom_blocks, atom_subnet)
        self.bond_prior = ConditionalPrior(config.num_bond_types, config.condition_dim,
                                           config.conditional_prior)
        self.atom_prior = ConditionalPrior(config.num_atom_types, config.condition_dim,
                                           config.conditional_prior)

    # -- conditioning ------------------------------------------------------
    def conditioning(self, values: torch.Tensor,
                     present: torch.Tensor | None = None) -> torch.Tensor:
        """h_c from the standardised condition, with a learned null for dropped rows."""
        context = self.condition_net(values)
        if present is None:
            return context
        keep = present.reshape(-1, 1).to(context.dtype)
        return keep * context + (1.0 - keep) * self.null_condition.unsqueeze(0)

    def dropout_mask(self, batch: int, device: torch.device | str,
                     generator: torch.Generator | None = None) -> torch.Tensor:
        probability = float(self.config.condition_dropout)
        if probability <= 0.0:
            return torch.ones(batch, device=device, dtype=torch.bool)
        draw = torch.rand(batch, device=device, generator=generator)
        return draw >= probability

    def null_context(self, batch: int, device: torch.device | str) -> torch.Tensor:
        return self.null_condition.unsqueeze(0).expand(batch, -1).to(device)

    def parameter_report(self) -> dict[str, Any]:
        """Where the capacity sits. Reported because the comparison depends on it.

        Stage 25 compares two factorisations, so the parameter counts must be on the
        record: if the atoms-first arm wins with FEWER parameters the result cannot be
        attributed to capacity, and if it loses with fewer the question is still open and
        a capacity-matched rerun is owed before concluding anything.
        """
        def count(module: nn.Module | None) -> int:
            return 0 if module is None else sum(p.numel() for p in module.parameters())

        return {
            "factorisation": self.config.factorisation,
            "total": sum(p.numel() for p in self.parameters()),
            "bond_stack": count(self.bond_layers),
            "atom_stack": count(self.atom_layers),
            "atom_context": count(self.atom_context),
            "conditioning": count(self.condition_net) + self.null_condition.numel(),
            "priors": count(self.bond_prior) + count(self.atom_prior),
        }

    # -- the two stacks ----------------------------------------------------
    def _run(self, layers: nn.ModuleList, x: torch.Tensor, mask: torch.Tensor,
             context: torch.Tensor, extra: torch.Tensor | None
             ) -> tuple[torch.Tensor, torch.Tensor]:
        logdet = x.new_zeros(x.shape[0])
        for layer in layers:
            x, step = layer["actnorm"](x, mask)
            logdet = logdet + step
            x, step = layer["mixing"](x, mask)
            logdet = logdet + step
            x, step = layer["coupling"](x, mask, context, extra)
            logdet = logdet + step
        return x, logdet

    def _invert(self, layers: nn.ModuleList, z: torch.Tensor, mask: torch.Tensor,
                context: torch.Tensor, extra: torch.Tensor | None) -> torch.Tensor:
        for layer in reversed(list(layers)):
            z = layer["coupling"].inverse(z, mask, context, extra)
            z = layer["mixing"].inverse(z, mask)
            z = layer["actnorm"].inverse(z, mask)
        return z

    def encode(self, atoms: torch.Tensor, bonds: torch.Tensor, node_mask: torch.Tensor,
               pair_mask: torch.Tensor, context: torch.Tensor) -> dict[str, torch.Tensor]:
        """(dequantized atoms, dequantized bonds) -> latents and log-determinants.

        The map is BLOCK-TRIANGULAR either way round - one block is transformed with the
        other held fixed as context - so the Jacobian's log-determinant is the sum of the
        two stacks' log-determinants in both factorisations. Which block is conditioned
        on which changes the model; it does not change that sum. A test checks it against
        the full autograd Jacobian for both.
        """
        if self.atoms_first:
            z_atom, logdet_atom = self._run(self.atom_layers, atoms, node_mask,
                                            context, None)
            grid = self.atom_context(atoms, node_mask)
            z_bond, logdet_bond = self._run(self.bond_layers, bonds, pair_mask,
                                            context, grid)
        else:
            z_bond, logdet_bond = self._run(self.bond_layers, bonds, pair_mask,
                                            context, None)
            dense = unpack_upper(bonds, self.config.max_nodes)
            z_atom, logdet_atom = self._run(self.atom_layers, atoms, node_mask,
                                            context, dense)
        return {"z_bond": z_bond, "z_atom": z_atom,
                "logdet_bond": logdet_bond, "logdet_atom": logdet_atom}

    def log_likelihood(self, atoms: torch.Tensor, bonds: torch.Tensor,
                       node_mask: torch.Tensor, pair_mask: torch.Tensor,
                       context: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-molecule continuous log-density of the dequantized graph."""
        encoded = self.encode(atoms, bonds, node_mask, pair_mask, context)
        log_bond = self.bond_prior.log_prob(encoded["z_bond"], pair_mask, context)
        log_atom = self.atom_prior.log_prob(encoded["z_atom"], node_mask, context)
        total = (log_bond + encoded["logdet_bond"] + log_atom + encoded["logdet_atom"])
        dimensions = (node_mask.to(total.dtype).sum(1) * self.config.num_atom_types
                      + pair_mask.to(total.dtype).sum(1) * self.config.num_bond_types)
        return {"log_prob": total, "log_prob_bond": log_bond + encoded["logdet_bond"],
                "log_prob_atom": log_atom + encoded["logdet_atom"],
                "dimensions": dimensions,
                # the dequantization bound: log P(discrete) >= log p(continuous) + D log s
                "discrete_lower_bound": total + dimensions * float(
                    np.log(self.config.noise_scale)),
                "z_bond": encoded["z_bond"], "z_atom": encoded["z_atom"]}

    # -- sampling ----------------------------------------------------------
    @torch.no_grad()
    def sample(self, counts: Sequence[int], context: torch.Tensor, *,
               temperature: float = 1.0, generator: torch.Generator | None = None,
               device: torch.device | str = "cpu") -> dict[str, Any]:
        """Native conditional sampling: one pass, no guidance, no selection."""
        device = torch.device(device)
        width = self.config.max_nodes
        sizes = [int(c) for c in counts]
        if any(c < 1 or c > width for c in sizes):
            raise ValueError(f"every requested N must lie in 1..{width}")
        node_mask = torch.zeros((len(sizes), width), dtype=torch.bool, device=device)
        for row, count in enumerate(sizes):
            node_mask[row, :count] = True
        pair_mask = pair_mask_from_counts(sizes, width, device)

        if self.atoms_first:
            # ATOMS FIRST. The bond stack is inverted with the atom context the atom
            # stack just produced - the same quantity `encode` reads at training time, so
            # the conditioning path is identical in both directions. Bond errors now
            # compound on atom errors instead of the other way round; that is inherent to
            # the factorisation and is measured, not hidden.
            z_atom = self.atom_prior.sample(node_mask, context, temperature=temperature,
                                            channels=self.config.num_atom_types,
                                            generator=generator)
            atoms = self._invert(self.atom_layers, z_atom, node_mask, context, None)
            grid = self.atom_context(atoms, node_mask)
            z_bond = self.bond_prior.sample(pair_mask, context, temperature=temperature,
                                            channels=self.config.num_bond_types,
                                            generator=generator)
            bonds = self._invert(self.bond_layers, z_bond, pair_mask, context, grid)
        else:
            z_bond = self.bond_prior.sample(pair_mask, context, temperature=temperature,
                                            channels=self.config.num_bond_types,
                                            generator=generator)
            bonds = self._invert(self.bond_layers, z_bond, pair_mask, context, None)
            dense = unpack_upper(bonds, width)
            z_atom = self.atom_prior.sample(node_mask, context, temperature=temperature,
                                            channels=self.config.num_atom_types,
                                            generator=generator)
            atoms = self._invert(self.atom_layers, z_atom, node_mask, context, dense)

        # Self-check: re-encoding the sampled graph must return the latents it came from.
        # Sampling IS the inverse, so a large error here means the molecules are not drawn
        # from the density that was trained - a failure that is otherwise invisible from
        # the loss curve. The baseline pipeline carries the same guard
        # (`generation.generate_molecules`), so the two are checked alike.
        recovered = self.encode(atoms, bonds, node_mask, pair_mask, context)
        def relative(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
            weights = mask.unsqueeze(-1).to(a.dtype)
            scale = max(1e-9, float((b * weights).abs().max()))
            return float(((a - b).abs() * weights).max()) / scale
        round_trip = {
            "bond": relative(recovered["z_bond"], z_bond, pair_mask),
            "atom": relative(recovered["z_atom"], z_atom, node_mask),
        }

        atom_types = atoms.argmax(dim=-1)
        bond_choice = bonds.argmax(dim=-1)
        return {"atom_types": atom_types, "bond_pairs": bond_choice,
                "atoms_continuous": atoms, "bonds_continuous": bonds,
                "node_mask": node_mask, "pair_mask": pair_mask, "counts": sizes,
                "latent_round_trip_relative_error": round_trip,
                "finite": (torch.isfinite(atoms).all(-1) | ~node_mask).all(-1)
                & (torch.isfinite(bonds).all(-1) | ~pair_mask).all(-1)}

    def discrete_graphs(self, sampled: dict[str, Any]) -> list[dict[str, np.ndarray]]:
        """argmax of the generated tensors -> discrete graphs. No chemistry applied."""
        width = self.config.max_nodes
        rows, cols = upper_triangle_indices(width, sampled["atom_types"].device)
        rows = rows.cpu().numpy()
        cols = cols.cpu().numpy()
        atom_types = sampled["atom_types"].cpu().numpy()
        pairs = sampled["bond_pairs"].cpu().numpy()
        pair_mask = sampled["pair_mask"].cpu().numpy()
        graphs = []
        for index, count in enumerate(sampled["counts"]):
            bonds = np.zeros((count, count), dtype=np.int64)
            keep = pair_mask[index]
            selected = np.flatnonzero(keep & (pairs[index] > 0))
            for position in selected:
                i, j = int(rows[position]), int(cols[position])
                bonds[i, j] = bonds[j, i] = int(pairs[index, position])
            graphs.append({"atom_types": atom_types[index, :count].copy(),
                           "bond_matrix": bonds,
                           "finite": bool(sampled["finite"][index])})
        return graphs


def dequantize(atom_types: torch.Tensor, bond_types: torch.Tensor,
               node_mask: torch.Tensor, pair_mask: torch.Tensor, *,
               num_atom_types: int, num_bond_types: int, noise_scale: float,
               generator: torch.Generator | None = None
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """One-hot + uniform noise, exactly MoFlow's `U[0, noise_scale]` scheme.

    Bond noise is drawn once per PAIR, not per ordered entry, because only the upper
    triangle is modelled - so the representation stays symmetric and each pair is counted
    once in the density.
    """
    width = node_mask.shape[1]
    atoms = F.one_hot(atom_types.clamp(min=0), num_atom_types).to(torch.float32)
    bonds = F.one_hot(pack_upper(bond_types, width).clamp(min=0),
                      num_bond_types).to(torch.float32)
    atom_noise = torch.rand(atoms.shape, device=atoms.device, generator=generator)
    bond_noise = torch.rand(bonds.shape, device=bonds.device, generator=generator)
    atoms = (atoms + float(noise_scale) * atom_noise) * node_mask.unsqueeze(-1)
    bonds = (bonds + float(noise_scale) * bond_noise) * pair_mask.unsqueeze(-1)
    return atoms, bonds


@contextmanager
def neutralise_coupling_condition(flow: ConditionalGraphFlow):
    """Switch off the condition INSIDE the couplings, leaving the conditional prior.

    Zero training, every buffer restored on exit - the same device stage 17's
    `warm_start.neutralised` used. What it isolates matters: a prior-only conditional flow
    shifts the latent cloud and can move average QED without reshaping p(G|c) at all,
    which is precisely the artefact stage 21's `total` decoder produced. Reporting the
    prior-only response beside the full response is how that is detected rather than
    celebrated.
    """
    couplings = [module for module in flow.modules()
                 if isinstance(module, ConditionalCoupling)]
    previous = [bool(module.condition_enabled) for module in couplings]
    try:
        for module in couplings:
            module.condition_enabled.fill_(False)
        yield flow
    finally:
        for module, state in zip(couplings, previous):
            module.condition_enabled.fill_(state)


__all__ = [
    "ATOM_CONTEXT_MODES",
    "BOND_CONDITIONERS",
    "FACTORISATIONS",
    "AtomChemistryContext",
    "AtomSequenceSubnet",
    "AtomSubnet",
    "BondSubnet",
    "ConditionalCoupling",
    "ConditionalGraphFlow",
    "ConditionalPrior",
    "GraphFlowConfig",
    "TopologyBondSubnet",
    "MaskedActNorm",
    "MaskedInvertibleLinear",
    "dequantize",
    "neutralise_coupling_condition",
    "pack_upper",
    "pair_mask_from_counts",
    "unpack_upper",
    "upper_triangle_indices",
]
