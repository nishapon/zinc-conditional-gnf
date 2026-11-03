"""Warm-starting a conditional flow from the unconditional Arm B checkpoint.

WHY THIS CAN BE MADE EXACT, NOT APPROXIMATE
-------------------------------------------
The two models differ in exactly four groups of parameters, and in nothing else:

    condition_embedding.*                       conditional only
    blocks.i.couplings.j.network.film.*         conditional only
    blocks.i.node_coupling.film.*               conditional only
    prior.hidden.* / prior.mu_head.* /
    prior.sigma_head.*                          conditional only
    prior.mu / prior.pre_sigma                  UNconditional only

Every other tensor - ActNorm scale and bias, the invertible linear layers, both affine
couplings' attention trunks, hidden layers and zero-initialised scale/translation heads,
the node-autoregressive coupling's trunk and start token, and the whitening buffers
`data_mean` / `data_std` - has the same name and the same shape in both, because
`condition_dim` only ever adds the separate FiLM layer (`CouplingNetwork.film`) and never
changes any other layer's width.

And the two conditional insertion points can both be made the IDENTITY:

* FiLM is applied as `hidden = hidden * (1 + gamma) + beta`. With `film.weight` and
  `film.bias` zero, gamma = beta = 0 and the expression is exactly `hidden` - multiplying
  by exactly 1.0 and adding exactly 0.0 are exact in IEEE-754, so this is bit-exact, not
  approximately neutral.
* The conditional prior computes `mu = mu_head(relu(hidden(c)))` and
  `sigma = softplus(sigma_head(...)) + min_sigma`, and the unconditional prior computes
  `softplus(pre_sigma) + min_sigma` from plain parameters. Zeroing both head WEIGHTS makes
  their outputs equal to their biases for every condition, so copying
  `prior.mu -> mu_head.bias` and `prior.pre_sigma -> sigma_head.bias` reproduces the
  unconditional prior exactly, for any c.

So a warm-started conditional model is, at initialisation, the unconditional model as a
function: same z, same log-determinant, same log-likelihood, same inverse and - given the
same RNG stream - the same samples. `equivalence_report` measures all five rather than
asserting it, and the audit script refuses to proceed if any of them is not at float
noise level.

WHY THIS IS SCIENTIFICALLY JUSTIFIED
------------------------------------
The conditional 50k run learned conditioning and cross-node structure simultaneously from
scratch and ended up worse at structure than the unconditional run (aromatic rings 0.35 vs
1.13, raw validity ~9% vs 14.35%). Warm-starting makes conditioning a PERTURBATION of a
generator that already has the structure, instead of a competing objective. It is the
standard zero-initialised-adapter construction, and here it is exactly function
preserving, which is what makes "did conditioning destroy structure?" a measurable
question rather than a guess.

WHAT IT CANNOT DO
-----------------
With the backbone frozen, the capacity available for conditioning is FiLM plus the prior -
the same capacity the from-scratch conditional model had. If that capacity is what limits
QED response (slope 0.0729 with a positive QED-only gap), a warm start will retain
structure but may not improve QED control much. The experiment is designed to separate
those two outcomes.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import torch

from zinc_gnf.flow import NodeFlow

# Parameters that exist only in the conditional model (so a missing one is expected).
CONDITIONAL_ONLY = (".film.", "condition_embedding.", "prior.hidden.",
                    "prior.mu_head.", "prior.sigma_head.")
# Parameters that exist only in the unconditional model (consumed by the prior transfer).
UNCONDITIONAL_ONLY = ("prior.mu", "prior.pre_sigma")
# Buffers that define the data space: these MUST match, or the two models are not
# describing the same density and no transfer is meaningful.
CRITICAL_BUFFERS = ("data_mean", "data_std")


def _is(name: str, patterns: Sequence[str]) -> bool:
    return any(pattern in name for pattern in patterns)


def transfer_report(source: dict[str, torch.Tensor], target: NodeFlow) -> dict[str, Any]:
    """Categorise every tensor of both models. No weight is written."""
    target_state = target.state_dict()
    transferable, shape_mismatch, conditional_only, unconditional_only, unexpected = \
        [], [], [], [], []
    for name, tensor in target_state.items():
        if name in source:
            if tuple(source[name].shape) == tuple(tensor.shape):
                transferable.append({"name": name, "shape": list(tensor.shape)})
            else:
                shape_mismatch.append({"name": name, "source": list(source[name].shape),
                                       "target": list(tensor.shape)})
        elif _is(name, CONDITIONAL_ONLY):
            conditional_only.append({"name": name, "shape": list(tensor.shape)})
        else:
            unexpected.append({"name": name, "shape": list(tensor.shape)})
    for name in source:
        if name not in target_state:
            (unconditional_only if _is(name, UNCONDITIONAL_ONLY) else unexpected).append(
                {"name": name, "shape": list(source[name].shape)})
    compatible = not shape_mismatch and not unexpected
    return {
        "transferable": transferable,
        "transferable_count": len(transferable),
        "conditional_only": conditional_only,
        "unconditional_only": unconditional_only,
        "shape_mismatch": shape_mismatch,
        "unexpected": unexpected,
        "compatible": bool(compatible),
        "verdict": ("every shared tensor matches in name and shape; the only differences are "
                    "the conditional insertion points and the unconditional prior's plain "
                    "parameters" if compatible else
                    "NOT COMPATIBLE: see shape_mismatch / unexpected - do not transfer"),
    }


@torch.no_grad()
def warm_start(target: NodeFlow, source: dict[str, torch.Tensor], *,
               neutral: bool = True) -> dict[str, Any]:
    """Load the unconditional backbone into a conditional flow, function-preserving.

    `neutral=True` (the only setting that preserves the function) zeroes every FiLM layer
    and makes the conditional prior reproduce the unconditional one for every condition.
    Raises if the two checkpoints are not compatible, so a silent partial load is
    impossible.
    """
    report = transfer_report(source, target)
    if not report["compatible"]:
        raise ValueError(report["verdict"])
    for name in CRITICAL_BUFFERS:
        if name not in source:
            raise ValueError(f"the unconditional checkpoint has no '{name}' buffer; its "
                             f"whitening is undefined and nothing can be transferred")

    missing, unexpected = target.load_state_dict(
        {k: v for k, v in source.items() if k in target.state_dict()}, strict=False)
    unexpected = list(unexpected)
    missing = list(missing)
    if unexpected:
        raise ValueError(f"unexpected keys while transferring: {unexpected[:5]}")
    not_conditional = [name for name in missing if not _is(name, CONDITIONAL_ONLY)]
    if not_conditional:
        raise ValueError("these non-conditional tensors were NOT transferred, so the "
                         f"backbone would be partly random: {not_conditional[:5]}")

    actions: list[str] = []
    if neutral:
        for name, module in target.named_modules():
            if name.endswith("film") and isinstance(module, torch.nn.Linear):
                module.weight.zero_()
                module.bias.zero_()
                actions.append(f"zeroed {name} (FiLM becomes the identity)")
        prior = target.prior
        if not prior.conditional:
            raise ValueError("the target flow has an unconditional prior; nothing to warm-start")
        if "prior.mu" not in source or "prior.pre_sigma" not in source:
            raise ValueError("the source prior is not an unconditional NodePrior")
        prior.mu_head.weight.zero_()
        prior.sigma_head.weight.zero_()
        prior.mu_head.bias.copy_(source["prior.mu"].to(prior.mu_head.bias.dtype))
        prior.sigma_head.bias.copy_(source["prior.pre_sigma"].to(prior.sigma_head.bias.dtype))
        actions.append("prior heads: weights zeroed, biases set to the unconditional "
                       "mu / pre_sigma, so mu(c), sigma(c) equal the unconditional prior "
                       "for every condition")
    return {**report, "missing_filled_by_initialisation": missing, "actions": actions,
            "neutral": bool(neutral)}


def backbone_parameters(flow: NodeFlow) -> list[tuple[str, torch.nn.Parameter]]:
    """Everything that is NOT a conditional insertion point."""
    return [(name, parameter) for name, parameter in flow.named_parameters()
            if not _is(name, CONDITIONAL_ONLY)]


def adapter_parameters(flow: NodeFlow) -> list[tuple[str, torch.nn.Parameter]]:
    """The conditional insertion points: FiLM, the condition embedding and the prior heads."""
    return [(name, parameter) for name, parameter in flow.named_parameters()
            if _is(name, CONDITIONAL_ONLY)]


def freeze_backbone(flow: NodeFlow) -> dict[str, Any]:
    """Stage A: train only the conditional adapters. Returns the parameter accounting."""
    frozen = trainable = 0
    for _, parameter in backbone_parameters(flow):
        parameter.requires_grad_(False)
        frozen += parameter.numel()
    for _, parameter in adapter_parameters(flow):
        parameter.requires_grad_(True)
        trainable += parameter.numel()
    if trainable == 0:
        raise ValueError("no adapter parameters are trainable; is this flow conditional?")
    return {"frozen_parameters": frozen, "trainable_parameters": trainable,
            "trainable_fraction": trainable / max(frozen + trainable, 1),
            "trainable_groups": sorted({name.rsplit(".", 1)[0]
                                        for name, _ in adapter_parameters(flow)})[:12]}


@torch.no_grad()
def equivalence_report(unconditional: NodeFlow, conditional: NodeFlow, x: torch.Tensor,
                       mask: torch.Tensor, condition_values: torch.Tensor, *,
                       seed: int = 1234) -> dict[str, Any]:
    """Is the warm-started conditional model the unconditional model as a function?

    Checked for EVERY supplied condition vector: latents, log-determinant, log-likelihood,
    the inverse, and the samples drawn from the same RNG stream. A warm start that is
    function preserving must be at float noise level on all five, for all conditions - if
    it is not, the transfer is wrong and no experiment built on it means anything.
    """
    unconditional.eval()
    conditional.eval()
    reference = unconditional.log_prob(x, mask, None)
    reference_inverse = unconditional.inverse(reference["z"], mask, None)
    generator = torch.Generator(device=x.device).manual_seed(int(seed))
    reference_sample = unconditional.sample(mask, None, temperature=1.0, generator=generator)

    rows = []
    for index in range(condition_values.shape[0]):
        values = condition_values[index: index + 1].expand(x.shape[0], -1)
        embedding = conditional.conditioning(values, None, x.shape[0], x.device)
        parts = conditional.log_prob(x, mask, embedding)
        inverse = conditional.inverse(parts["z"], mask, embedding)
        generator = torch.Generator(device=x.device).manual_seed(int(seed))
        sample = conditional.sample(mask, embedding, temperature=1.0, generator=generator)
        weights = mask.unsqueeze(-1).to(x.dtype)
        rows.append({
            "condition": [float(v) for v in condition_values[index]],
            "max_abs_latent_difference": float(((parts["z"] - reference["z"]) * weights).abs().max()),
            "max_abs_log_det_difference": float((parts["log_det"] - reference["log_det"]).abs().max()),
            "max_abs_log_prob_difference": float((parts["log_prob"] - reference["log_prob"]).abs().max()),
            "max_abs_inverse_difference": float(((inverse - reference_inverse) * weights).abs().max()),
            "max_abs_sample_difference": float(((sample - reference_sample) * weights).abs().max()),
        })
    worst = {key: max(row[key] for row in rows) for key in rows[0] if key != "condition"}
    exact = all(value == 0.0 for value in worst.values())
    return {"rows": rows, "worst": worst, "bit_exact": bool(exact),
            "equivalent": bool(max(worst.values()) < 1e-5),
            "verdict": ("bit-exact: the warm-started conditional model IS the unconditional "
                        "model at initialisation" if exact else
                        "equivalent to float noise" if max(worst.values()) < 1e-5 else
                        "NOT EQUIVALENT: the transfer changed the function; do not proceed")}


@contextmanager
def neutralised(flow: NodeFlow, *, film: bool = False, prior: bool = False,
                reference_condition: torch.Tensor | None = None) -> Iterator[NodeFlow]:
    """Temporarily switch off conditional pathways of a TRAINED conditional flow.

    This is the evaluation-time ablation that answers "which conditional component costs
    us molecular structure?" without training anything:

      film=True    every FiLM layer becomes the identity, so the couplings behave as in
                   an unconditional flow while the prior stays conditional
      prior=True   mu(c), sigma(c) are frozen at the value they take for
                   `reference_condition`, removing the prior's condition DEPENDENCE while
                   keeping its average location and scale

    Every tensor is restored on exit, including on an exception.
    """
    saved: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    try:
        with torch.no_grad():
            if film:
                for name, module in flow.named_modules():
                    if name.endswith("film") and isinstance(module, torch.nn.Linear):
                        saved.append((module.weight, module.weight.detach().clone()))
                        saved.append((module.bias, module.bias.detach().clone()))
                        module.weight.zero_()
                        module.bias.zero_()
            if prior:
                if reference_condition is None:
                    raise ValueError("prior neutralisation needs a reference condition")
                embedding = flow.conditioning(reference_condition, None,
                                              reference_condition.shape[0],
                                              reference_condition.device)
                mu, sigma = flow.prior.parameters_for(reference_condition.shape[0],
                                                      embedding, reference_condition.device)
                mu = mu.mean(0)
                pre_sigma = torch.log(torch.expm1((sigma.mean(0) - flow.prior.min_sigma)
                                                  .clamp(min=1e-6)))
                for parameter, value in ((flow.prior.mu_head.weight, None),
                                         (flow.prior.sigma_head.weight, None),
                                         (flow.prior.mu_head.bias, mu),
                                         (flow.prior.sigma_head.bias, pre_sigma)):
                    saved.append((parameter, parameter.detach().clone()))
                    if value is None:
                        parameter.zero_()
                    else:
                        parameter.copy_(value.to(parameter.dtype))
        yield flow
    finally:
        with torch.no_grad():
            for parameter, value in reversed(saved):
                parameter.copy_(value)


__all__ = [
    "CONDITIONAL_ONLY",
    "CRITICAL_BUFFERS",
    "UNCONDITIONAL_ONLY",
    "adapter_parameters",
    "backbone_parameters",
    "equivalence_report",
    "freeze_backbone",
    "neutralised",
    "transfer_report",
    "warm_start",
]
