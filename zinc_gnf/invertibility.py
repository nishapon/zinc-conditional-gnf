"""Why x -> z -> x and z -> x -> z disagree, and what to do about it.

THE TWO NUMBERS ARE NOT COMPARABLE AS REPORTED, AND NEITHER BOUNDS THE OTHER.

Let F be the forward map as implemented in floating point and G the implemented inverse.
Both compositions are the identity in exact arithmetic, but they are measured in
DIFFERENT SPACES with different scales and different amplification:

    x -> z -> x    G(F(x)), measured in embedding space. The autoencoder's embeddings
                   have a small per-dimension spread, so a relative error divides by a
                   small number; and the trained map is strongly expansive
                   (log|det J|/node > 0), so rounding committed in F is mapped back
                   through a contractive G with its own conditioning.

    z -> x -> z    F(G(z)), measured in latent space, where the prior fixes the scale at
                   O(1). The second leg recomputes z from the very point the first leg
                   produced, so this composition is well conditioned precisely where the
                   code paths agree - it is a consistency check between the two
                   implementations, and a small value here does NOT establish that G(z)
                   is close to the true preimage.

So 1.4e-05 in the sampling direction and 4.9e-03 on real embeddings are not a
contradiction, and the small one must not be quoted as evidence that inversion is
accurate. Every measurement below therefore reports the ABSOLUTE error and the
REFERENCE SCALE next to the relative one, so the two directions can be compared
honestly, in float32 and float64, per component, and as a function of N - and then asks
the only question that decides the science: does the error change the DECODED MOLECULE?

The node-autoregressive coupling is the suspect for a size-dependent error. Its forward
pass is parallel, but its inverse must be sequential: node i is recovered using the
already-recovered nodes 1..i-1, so an error at node 1 propagates into every later node,
scaled by each step's exp(s). That predicts error growing with N, which `error_by_size`
tests directly.

The correction offered here is the smallest one that exists: run the INVERSE in float64
(`double_precision_inverse`). No weight is changed, no checkpoint is rewritten, nothing
is retrained - the parameters are cast for the duration of one inverse pass.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from zinc_gnf.flow import NodeFlow

# Component stacks, cheapest first: each adds one piece of a block.
COMPONENTS = ("actnorm", "actnorm+linear", "actnorm+linear+coupling", "full")


def relative_error(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    """Masked max / mean |a - b| scaled by the magnitude of `b`, plus the raw absolute max."""
    a, b = a.detach(), b.detach()
    weights = mask.unsqueeze(-1).to(a.dtype)
    difference = ((a - b) * weights).abs()
    scale = ((b * weights).abs().max()).clamp(min=1e-30)
    denominator = weights.sum().clamp(min=1.0)
    return {
        "max_absolute": float(difference.max()),
        "max_relative": float(difference.max() / scale),
        "mean_absolute": float(difference.sum() / (denominator * a.shape[-1])),
        "reference_scale": float(scale),
    }


@contextmanager
def double_precision_inverse(flow: NodeFlow) -> Iterator[NodeFlow]:
    """Run the flow in float64 inside the block, then restore float32 exactly.

    THE smallest possible correction: the weights are the trained weights, cast. Nothing
    is retrained and no checkpoint is written. The cast is reversible, and the original
    dtype is restored even if the body raises.
    """
    original = next(flow.parameters()).dtype
    try:
        flow.double()
        yield flow
    finally:
        flow.to(original)


@torch.no_grad()
def round_trip(flow: NodeFlow, x: torch.Tensor, mask: torch.Tensor,
               condition: torch.Tensor | None = None, *,
               double: bool = False) -> dict[str, Any]:
    """Both directions on the same inputs. `double` casts the whole flow to float64."""
    def measure(model: NodeFlow, data: torch.Tensor, cond: torch.Tensor | None) -> dict[str, Any]:
        z, log_det = model(data, mask, cond)
        back = model.inverse(z, mask, cond)
        data_direction = relative_error(back, data, mask)
        # Sampling direction, from the SAME latents, so the two numbers are comparable.
        reconstructed, _ = model(model.inverse(z, mask, cond), mask, cond)
        latent_direction = relative_error(reconstructed, z, mask)
        nodes = float(mask.sum().clamp(min=1))
        return {
            "data_direction_x_z_x": data_direction,
            "latent_direction_z_x_z": latent_direction,
            "log_det_per_node": float(log_det.sum() / nodes),
            "z_abs_max": float((z * mask.unsqueeze(-1)).abs().max()),
            "x_abs_max": float((data * mask.unsqueeze(-1)).abs().max()),
        }

    if not double:
        return {"dtype": "float32", **measure(flow, x, condition)}
    with double_precision_inverse(flow) as doubled:
        return {"dtype": "float64",
                **measure(doubled, x.double(),
                          None if condition is None else condition.double())}


@torch.no_grad()
def component_breakdown(flow: NodeFlow, x: torch.Tensor, mask: torch.Tensor,
                        condition: torch.Tensor | None = None) -> list[dict[str, Any]]:
    """Which part of a block loses the precision?

    Each stack is evaluated by DISABLING the later pieces rather than rebuilding a
    model, so the weights are exactly the trained ones. `x -> z -> x` is reported for
    every stack, so the jump between consecutive rows is that piece's contribution.
    """
    rows: list[dict[str, Any]] = []
    blocks = list(flow.blocks)
    saved = [(block.couplings if hasattr(block, "couplings") else None,
              getattr(block, "node_coupling", None),
              block.linear, block.act_norm) for block in blocks]
    try:
        for component in COMPONENTS:
            for block, (couplings, node_coupling, linear, act_norm) in zip(blocks, saved):
                block.act_norm = act_norm
                block.linear = linear if "linear" in component or component == "full" else None
                if couplings is not None:
                    keep = ("coupling" in component) or component == "full"
                    block.couplings = couplings if keep else torch.nn.ModuleList()
                block.node_coupling = node_coupling if component == "full" else None
            z, log_det = flow(x, mask, condition)
            back = flow.inverse(z, mask, condition)
            nodes = float(mask.sum().clamp(min=1))
            rows.append({"components": component,
                         "data_direction_x_z_x": relative_error(back, x, mask),
                         "log_det_per_node": float(log_det.sum() / nodes)})
    finally:
        for block, (couplings, node_coupling, linear, act_norm) in zip(blocks, saved):
            block.act_norm, block.linear = act_norm, linear
            if couplings is not None:
                block.couplings = couplings
            block.node_coupling = node_coupling
    return rows


@torch.no_grad()
def error_by_size(flow: NodeFlow, x: torch.Tensor, mask: torch.Tensor,
                  condition: torch.Tensor | None = None) -> list[dict[str, Any]]:
    """Round-trip error per molecule, grouped by N.

    A sequential inverse that accumulates error along the node order must show error
    growing with N. A per-node error that does not depend on N points elsewhere.
    """
    z, _ = flow(x, mask, condition)
    back = flow.inverse(z, mask, condition)
    per_molecule: dict[int, list[float]] = {}
    for index in range(x.shape[0]):
        rows = mask[index]
        count = int(rows.sum())
        if count == 0:
            continue
        difference = (back[index][rows] - x[index][rows]).abs().max()
        scale = x[index][rows].abs().max().clamp(min=1e-30)
        per_molecule.setdefault(count, []).append(float(difference / scale))
    return [{"n": n, "molecules": len(values),
             "max_relative_error": float(np.max(values)),
             "median_relative_error": float(np.median(values))}
            for n, values in sorted(per_molecule.items())]


@torch.no_grad()
def chemistry_impact(autoencoder, x: torch.Tensor, x_round_tripped: torch.Tensor,
                     counts: Sequence[int], *, decoders: Sequence[str] = ("raw", "degree")
                     ) -> dict[str, Any]:
    """Does the round-trip error change the DECODED MOLECULE?

    This is the question that decides whether the inversion error threatens the
    scientific validity of results already obtained: if the same embeddings, perturbed
    only by the round trip, decode to the same SMILES, then the error is below the
    decoder's resolution and every published number stands.
    """
    from zinc_gnf.generation import decode_embeddings

    original = decode_embeddings(autoencoder, x, counts, decoders=decoders)
    perturbed = decode_embeddings(autoencoder, x_round_tripped, counts, decoders=decoders)
    result: dict[str, Any] = {"molecules": len(original), "decoders": {}}
    for name in decoders:
        same_smiles = same_validity = 0
        for a, b in zip(original, perturbed):
            left, right = a.get(name) or {}, b.get(name) or {}
            same_validity += int(bool(left.get("valid")) == bool(right.get("valid")))
            same_smiles += int(left.get("smiles") == right.get("smiles"))
        result["decoders"][name] = {
            "identical_smiles_rate": same_smiles / max(1, len(original)),
            "identical_validity_rate": same_validity / max(1, len(original)),
        }
    result["verdict"] = (
        "the round-trip error is below the decoder's resolution: it does not change the "
        "molecules, so results already obtained are unaffected"
        if all(v["identical_smiles_rate"] > 0.999 for v in result["decoders"].values())
        else "the round-trip error CHANGES decoded molecules; corrective experiments must "
             "use the float64 inverse")
    return result


def log_det_budget(flow: NodeFlow) -> dict[str, Any]:
    """Is log|det J|/node large, or just large-looking?

    Two reference quantities make it interpretable:
      whitening      -sum(log std) is a fixed constant of the DATA (a gauge), not
                     evidence of expansion; subtract it before judging anything
      clamp budget   every coupling's log-scale is tanh-clamped to +-max_log_scale, so
                     the couplings cannot contribute more than
                     (couplings per block) x blocks x dims x max_log_scale per node
    """
    config = flow.config
    per_block = 1 + (1 if config.node_autoregressive else 0)
    budget = per_block * config.num_blocks * config.node_embedding_dim * config.max_log_scale
    whitening = float(torch.log(flow.effective_std).sum())
    return {
        "whitening_log_det_per_node": -whitening,
        "coupling_clamp_budget_per_node": float(budget),
        "actnorm_log_det_per_node": float(sum(
            float(block.act_norm.log_scale.detach().sum()) for block in flow.blocks
            if getattr(block, "act_norm", None) is not None)),
        "note": ("subtract the whitening term before judging the magnitude; what remains "
                 "must sit inside the clamp budget, and a value inside it is not evidence "
                 "of divergence. Only the x->z->x error says whether precision is lost."),
    }


__all__ = [
    "COMPONENTS",
    "chemistry_impact",
    "component_breakdown",
    "double_precision_inverse",
    "error_by_size",
    "log_det_budget",
    "relative_error",
    "round_trip",
]
