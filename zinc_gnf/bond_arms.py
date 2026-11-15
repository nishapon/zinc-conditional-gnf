"""Experiment C: three matched arms for the bond-aware pairwise conditioner.

Kept in its own module so experiments A (balanced sampling) and B (correlated prior) stay
untouched and separately importable. The warm-start machinery is REUSED from
`zinc_gnf.experiment_arms`, because the hard parts -- translating the parent checkpoint so
`FlowTrainer._restore`'s strict checks pass honestly, and re-indexing Adam's moments by
verified parameter name -- are identical here.

    control              the existing conditioner, continued training only
    pairwise             the bond-aware pathway, LIKELIHOOD TRAINING ALONE
    pairwise_supervised  the same pathway, plus the bond-prediction loss

Why all three. `pairwise_supervised` changes two things relative to `control`: it adds a
pathway and it adds a training signal. With only those two arms, a gain could come from
either. `pairwise` holds the architecture fixed and removes only the supervision, so

    pairwise vs control             = what the architecture buys on likelihood alone
    pairwise_supervised vs pairwise = what the bond target buys on top

and `control` absorbs the fact that all three arms train for additional steps, so no gain
can be claimed against the parent checkpoint merely for having trained longer.

The two pairwise arms are architecturally IDENTICAL -- same parameters, same
initialisation, same seed. They differ only in `bond_loss_weight`, which is exactly zero
for `pairwise`. That is what makes the second comparison clean.

ONE THING TO WATCH IN ARM `pairwise`
------------------------------------
`pair_to_hidden` is zero-initialised, so at step 0 the likelihood gradient reaching the
bond scorer is zero (it carries a factor of that weight). The bridge weight itself has a
nonzero gradient and leaves zero immediately, after which the scorer trains normally. So
`pairwise` is slower to engage its pathway than `pairwise_supervised`, whose auxiliary
loss reaches the scorer at step 0. `bond_bridge_norm` is logged every window so this is
visible rather than inferred: if it stays at zero, the pathway never engaged and the arm's
result says nothing about the architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from zinc_gnf.bond_pairwise import (
    BOND_PAIR_DIM,
    bond_loss_and_metrics,
    bond_pathway_report,
    clear_published_logits,
    collect_published_logits,
    set_publishing,
)
from zinc_gnf.experiment_arms import ArmTrainer
from zinc_gnf.trainer_flow import ResumeMismatchError

BOND_ARMS = ("control", "pairwise", "pairwise_supervised")

# Flow-config keys experiment C is allowed to add on top of the parent. Anything else
# differing between the parent and an arm is refused by `verify_additive_architecture`.
BOND_ARCHITECTURE_KEYS = ("bond_aware_coupling", "bond_pair_dim")

# Weight on the auxiliary bond loss for the supervised arm. The likelihood term is a
# per-node NLL in nats (order 1-10 for this model) and the bond term is a reweighted BCE
# over pairs (order 0.1-1 at initialisation), so 0.1 keeps the auxiliary term a
# regulariser rather than the objective. Logged separately so the balance is auditable.
DEFAULT_BOND_LOSS_WEIGHT = 0.1


@dataclass
class BondArmConfig:
    """Everything that differs between the three arms of experiment C."""

    arm: str = "control"
    bond_pair_dim: int = BOND_PAIR_DIM
    bond_loss_weight: float = DEFAULT_BOND_LOSS_WEIGHT
    notes: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.arm not in BOND_ARMS:
            raise ValueError(f"arm must be one of {BOND_ARMS}, got {self.arm!r}")
        if self.bond_pair_dim < 1:
            raise ValueError("bond_pair_dim must be >= 1")
        if self.bond_loss_weight < 0:
            raise ValueError("bond_loss_weight must be >= 0")
        if self.arm == "pairwise_supervised" and self.bond_loss_weight == 0.0:
            raise ValueError(
                "arm 'pairwise_supervised' with bond_loss_weight 0 IS arm 'pairwise'; "
                "use that name so the two are not silently confused in the results")
        if self.arm == "pairwise" and self.bond_loss_weight != 0.0:
            raise ValueError(
                "arm 'pairwise' is the likelihood-only control and must have "
                "bond_loss_weight exactly 0; it is what isolates the architecture from "
                "the training signal")

    @property
    def uses_pairwise(self) -> bool:
        return self.arm in ("pairwise", "pairwise_supervised")

    @property
    def bond_supervised(self) -> bool:
        return self.arm == "pairwise_supervised" and self.bond_loss_weight > 0.0

    # -- duck-compatibility with experiment_arms.prepare_warm_start --------
    @property
    def uses_coupled_prior(self) -> bool:
        return False

    @property
    def uses_balanced_sampler(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {"experiment": "C_bond_pairwise",
                "arm": self.arm,
                "bond_pair_dim": self.bond_pair_dim,
                "bond_loss_weight": self.bond_loss_weight,
                "uses_pairwise": self.uses_pairwise,
                "bond_supervised": self.bond_supervised,
                "notes": self.notes}

    def flow_config_overrides(self) -> dict[str, Any]:
        """What this arm changes in the parent's flow config. Additive only."""
        if not self.uses_pairwise:
            return {}
        return {"bond_aware_coupling": True, "bond_pair_dim": self.bond_pair_dim}


class BondArmTrainer(ArmTrainer):
    """`FlowTrainer` plus the auxiliary bond loss. The architecture comes from the config.

    No `_prepare_flow` surgery is needed: `bond_aware_coupling` lives in `FlowConfig`, so
    `NodeFlow.__init__` builds the pathway and every new parameter is in
    `flow.parameters()` before the optimizer exists.
    """

    VERIFY_KEYS = ("arm", "bond_pair_dim", "bond_loss_weight")

    def __init__(self, *args, arm_config: BondArmConfig | None = None, **kwargs) -> None:
        bond_config = arm_config or BondArmConfig()
        bond_config.validate()
        super().__init__(*args, arm_config=bond_config, **kwargs)
        self.arm_report = {"arm": bond_config.arm, "experiment": "C_bond_pairwise"}

    # -- hook 1: verify the pathway is actually present ------------------
    def _prepare_flow(self, flow):
        report = bond_pathway_report(flow)
        expected = self.arm_config.uses_pairwise
        if expected and report["conditioners"] == 0:
            raise RuntimeError(
                f"arm {self.arm_config.arm} needs the bond-aware pathway but the flow was "
                f"built without it; flow_config.bond_aware_coupling is "
                f"{flow.config.bond_aware_coupling}")
        if not expected and report["conditioners"] > 0:
            raise RuntimeError(
                f"arm {self.arm_config.arm} is the unchanged control but the flow carries "
                f"{report['conditioners']} bond conditioners")
        if expected:
            set_publishing(flow, True)
            self.logger.info(
                f"arm {self.arm_config.arm}: {report['conditioners']} bond-aware "
                f"conditioners, pair_dim={report['pair_dim']}, "
                f"{report['new_parameters']:,} new parameters "
                f"({report['gate_parameters']:,} scorer + {report['bridge_parameters']:,} "
                f"bridge); bridge zero-initialised={report['bridge_is_zero']} so the flow "
                f"starts as the parent density")
            self.logger.info(
                f"bond supervision: {'ON' if self.arm_config.bond_supervised else 'OFF'} "
                f"(weight {self.arm_config.bond_loss_weight}); the likelihood term is "
                f"logged as nll_per_node and the auxiliary term as bond_loss, and model "
                f"selection still ranks on the validation likelihood alone")
        self.arm_report["pathway"] = report
        return flow

    # -- hook 2: the auxiliary loss --------------------------------------
    def _augment_loss(self, loss, *, flow, batch, parts):
        """Likelihood + `weight * bond BCE`, with both terms logged separately."""
        if not self.arm_config.uses_pairwise:
            return loss, {}
        width = int(batch.node_mask.shape[1])
        logits = collect_published_logits(flow, batch_size=batch.batch_size, width=width)
        result = bond_loss_and_metrics(logits, batch.bond_types, batch.node_mask)
        # cleared immediately, so a step that somehow skipped the forward pass raises in
        # `collect_published_logits` instead of supervising last step's batch
        clear_published_logits(flow)
        extras = dict(result["metrics"])
        extras["bond_bridge_norm"] = self._bridge_norm(flow)
        if self.arm_config.bond_supervised:
            total = loss + float(self.arm_config.bond_loss_weight) * result["loss"]
            extras["bond_loss_weighted"] = float(
                (float(self.arm_config.bond_loss_weight) * result["loss"]).detach())
            return total, extras
        # arm `pairwise`: the bond loss is MEASURED but not optimised, so the two pairwise
        # arms are comparable on the same diagnostic without the gradient
        extras["bond_loss_weighted"] = 0.0
        return loss, extras

    @staticmethod
    def _bridge_norm(flow) -> float:
        """Size of the zero-initialised bridge. Zero means the pathway is still inert."""
        total = 0.0
        for name, module in flow.named_modules():
            if name.endswith("pair_to_hidden") and hasattr(module, "weight"):
                total += float(module.weight.detach().norm()) ** 2
        return total ** 0.5


def bond_flow_config(parent_flow_config: dict[str, Any], arm: BondArmConfig):
    """The arm's `FlowConfig`: the parent's, plus this arm's additive overrides.

    Built from the PARENT's stored config rather than from a fresh default, so every
    setting the parent trained under is carried over exactly and the only differences are
    the declared additive keys.
    """
    from zinc_gnf.flow import FlowConfig

    values = dict(parent_flow_config)
    for key in BOND_ARCHITECTURE_KEYS:
        values.setdefault(key, FlowConfig.__dataclass_fields__[key].default)
    overrides = arm.flow_config_overrides()
    unknown = set(overrides) - set(BOND_ARCHITECTURE_KEYS)
    if unknown:
        raise ValueError(f"arm overrides touch undeclared keys: {sorted(unknown)}")
    values.update(overrides)
    return FlowConfig.from_dict(values)


def estimate_cost(flow, *, batch_size: int, max_nodes: int) -> dict[str, Any]:
    """Memory and parameter cost of the pathway, measured on the built flow.

    The dominant term is the `|h_i - h_j|` tensor, `[B, N, N, pair_dim]` per conditioner,
    which autograd keeps for the backward pass. Reported before scaling so the batch size
    or `bond_pair_dim` can be cut if it does not fit.
    """
    report = bond_pathway_report(flow)
    conditioners = max(report["conditioners"], 0)
    pair_dim = report["pair_dim"] or 0
    bytes_per_float = 4
    difference = batch_size * max_nodes * max_nodes * pair_dim * bytes_per_float
    matrices = batch_size * max_nodes * max_nodes * bytes_per_float
    per_conditioner = difference + 4 * matrices
    return {
        "conditioners": conditioners,
        "pair_dim": pair_dim,
        "batch_size": batch_size,
        "max_nodes": max_nodes,
        "new_parameters": report["new_parameters"],
        "total_parameters": sum(p.numel() for p in flow.parameters()),
        "parameter_increase_fraction": (
            report["new_parameters"] / max(sum(p.numel() for p in flow.parameters())
                                           - report["new_parameters"], 1)),
        "difference_tensor_mb_per_conditioner": difference / 1e6,
        "activation_mb_total": conditioners * per_conditioner / 1e6,
        "dominant_term": "|h_i - h_j| of shape [B, N, N, pair_dim], kept for backward",
        "levers": ("halving bond_pair_dim halves the dominant term; halving batch_size "
                   "halves it again; both leave the architecture intact"),
    }


__all__ = [
    "BOND_ARCHITECTURE_KEYS",
    "BOND_ARMS",
    "DEFAULT_BOND_LOSS_WEIGHT",
    "BondArmConfig",
    "BondArmTrainer",
    "bond_flow_config",
    "estimate_cost",
]
