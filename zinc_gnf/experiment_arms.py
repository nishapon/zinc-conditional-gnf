"""Isolated experiment arms: balanced sampling and the correlated prior.

ALL arm logic lives here so the training loop keeps working exactly as before. The only
change to `zinc_gnf/trainer_flow.py` is two hook points, both no-ops by default:

    flow = NodeFlow(...).to(self.device)
    self._prepare_flow(flow)              # <-- NEW, returns flow unchanged by default
    optimizer = torch.optim.Adam(flow.parameters(), ...)
    self.batcher = self._make_batcher()   # <-- CHANGED from the inline DeterministicBatcher

`_prepare_flow` deliberately sits BEFORE the optimizer is built, which is what puts the
new prior parameter into `flow.parameters()` and therefore into the optimizer.

FOUR ARMS, ONE CHECKPOINT, ONE BUDGET
-------------------------------------
    control         nothing changed; continued training only
    balanced        BalancedSampler at alpha; prior untouched
    prior_frozen    CoupledNodePrior at rho = init, pre_rho.requires_grad = False
    prior_learned   CoupledNodePrior at rho = init, pre_rho trainable

`control` exists because every arm trains for additional steps, so a gain must be shown
against continued training rather than against the original checkpoint.
`prior_frozen` exists because `prior_learned` changes two things at once -- it introduces
cross-node correlation AND learns it. Without the frozen arm at the identical
initialisation those two effects cannot be separated.

OPTIMIZER STATE ACROSS ARMS
---------------------------
`prior_learned` and `prior_frozen` add `prior.pre_rho` to `flow.parameters()`, so the
optimizer has one more tensor than the checkpoint's saved state and a plain
`load_state_dict` would fail on the index mismatch. `adapt_optimizer_state` restores the
saved Adam moments for every parameter that already existed and leaves the new one
unstarted, which Adam initialises lazily on its first step. Existing parameters therefore
resume identically in all four arms; only the genuinely new tensor differs, which is the
consistency the comparison needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from zinc_gnf.balanced_sampler import BalancedSampler, joint_distribution
from zinc_gnf.coupled_prior import INIT_RHO, CoupledNodePrior
from zinc_gnf.dataset import DeterministicBatcher
from zinc_gnf.trainer_flow import FlowTrainer

ARMS = ("control", "balanced", "prior_frozen", "prior_learned")

# Imbalance gate for arm `balanced`. If the joint N-QED distribution is already close to
# uniform over its populated cells there is nothing to rebalance, and the arm should not
# run. Measured, not assumed.
MIN_EFFECTIVE_CELL_FRACTION = 0.60      # effective_cells / cells below this = imbalanced
MIN_POPULATION_RATIO = 10.0             # largest/smallest populated cell


@dataclass
class ArmConfig:
    """Everything that differs between arms. One named intervention each."""

    arm: str = "control"
    balanced_alpha: float = 0.5
    min_cell_count: int = 50
    max_exposure_ratio: float = 20.0
    qed_bin_width: float = 0.1
    coupled_init_rho: float = INIT_RHO
    notes: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {self.arm!r}")
        if not 0.0 <= self.balanced_alpha <= 1.0:
            raise ValueError("balanced_alpha must lie in [0, 1]")
        if self.arm == "balanced" and self.balanced_alpha == 0.0:
            raise ValueError("arm 'balanced' with alpha 0 is the control; use 'control'")

    @property
    def uses_balanced_sampler(self) -> bool:
        return self.arm == "balanced"

    @property
    def uses_coupled_prior(self) -> bool:
        return self.arm in ("prior_frozen", "prior_learned")

    @property
    def rho_learnable(self) -> bool:
        return self.arm == "prior_learned"

    def to_dict(self) -> dict[str, Any]:
        return {"arm": self.arm, "balanced_alpha": self.balanced_alpha,
                "min_cell_count": self.min_cell_count,
                "max_exposure_ratio": self.max_exposure_ratio,
                "qed_bin_width": self.qed_bin_width,
                "coupled_init_rho": self.coupled_init_rho,
                "uses_balanced_sampler": self.uses_balanced_sampler,
                "uses_coupled_prior": self.uses_coupled_prior,
                "rho_learnable": self.rho_learnable,
                "notes": self.notes}


# ------------------------------------------------------------------ imbalance ---
def imbalance_report(n_node, qed, *, qed_bin_width: float = 0.1) -> dict[str, Any]:
    """Is there meaningful imbalance to rebalance? Gate for the `balanced` arm."""
    distribution = joint_distribution(n_node, qed, qed_bin_width=qed_bin_width)
    cells = max(1, distribution["cells"])
    effective = distribution.get("effective_cells") or 0.0
    fraction = effective / cells
    ratio = distribution.get("population_ratio")
    imbalanced = bool(fraction < MIN_EFFECTIVE_CELL_FRACTION
                      or (ratio is not None and ratio >= MIN_POPULATION_RATIO))
    return {
        "cells": cells,
        "effective_cells": effective,
        "effective_cell_fraction": fraction,
        "population_ratio": ratio,
        "largest_cell": distribution["largest_cell"],
        "smallest_cell": distribution["smallest_cell"],
        "cells_below_min_count": distribution["cells_below_50"],
        "thresholds": {"min_effective_cell_fraction": MIN_EFFECTIVE_CELL_FRACTION,
                       "min_population_ratio": MIN_POPULATION_RATIO},
        "meaningful_imbalance": imbalanced,
        "verdict": (
            "IMBALANCED: rebalancing has something to act on"
            if imbalanced else
            "NEARLY BALANCED ALREADY: the `balanced` arm should not run, because "
            "reweighting a distribution that is already close to uniform over its "
            "populated cells changes almost nothing and spends a GPU budget to "
            "demonstrate that."),
    }


# ------------------------------------------------------------ optimizer state ---
def adapt_optimizer_state(optimizer, saved: dict[str, Any] | None) -> dict[str, Any]:
    """Restore Adam state for pre-existing parameters; leave new ones unstarted.

    Adam's `state_dict` keys its per-parameter state by POSITION, so an optimizer with
    one extra tensor cannot load an older state directly. The extra indices are appended
    to the first parameter group with no state attached, which Adam creates lazily on
    the first step. Pre-existing parameters therefore resume bit-identically.
    """
    if not saved:
        return {"restored": False, "reason": "no saved optimizer state"}
    current = sum(len(group["params"]) for group in optimizer.param_groups)
    previous = sum(len(group["params"]) for group in saved.get("param_groups", []))
    if current == previous:
        optimizer.load_state_dict(saved)
        return {"restored": True, "parameters": current, "added": 0}
    if current < previous:
        raise ValueError(
            f"the optimizer has FEWER parameters ({current}) than the saved state "
            f"({previous}); refusing to guess which were dropped")
    patched = {"state": dict(saved.get("state", {})),
               "param_groups": [dict(group) for group in saved.get("param_groups", [])]}
    if not patched["param_groups"]:
        raise ValueError("saved optimizer state has no parameter groups")
    added = list(range(previous, current))
    patched["param_groups"][0] = dict(patched["param_groups"][0])
    patched["param_groups"][0]["params"] = (
        list(patched["param_groups"][0]["params"]) + added)
    optimizer.load_state_dict(patched)
    return {"restored": True, "parameters": current, "added": len(added),
            "note": ("new parameters were appended with no Adam moments; Adam "
                     "initialises them on the first step, so every pre-existing "
                     "parameter resumes identically")}


# ----------------------------------------------------------------- the trainer ---
class ArmTrainer(FlowTrainer):
    """`FlowTrainer` with one named intervention. Everything else is inherited."""

    def __init__(self, *args, arm_config: ArmConfig | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.arm_config = arm_config or ArmConfig()
        self.arm_config.validate()
        self.arm_report: dict[str, Any] = {"arm": self.arm_config.arm}

    # -- hook 1: wrap the prior BEFORE the optimizer is built ----------
    def _prepare_flow(self, flow):
        if not self.arm_config.uses_coupled_prior:
            self.arm_report["prior"] = {"type": "NodePrior", "changed": False}
            return flow
        flow.prior = CoupledNodePrior.from_node_prior(
            flow.prior, init_rho=self.arm_config.coupled_init_rho,
            learnable=self.arm_config.rho_learnable).to(self.device)
        names = [name for name, _ in flow.named_parameters()]
        if not any(name.endswith("pre_rho") for name in names):
            raise RuntimeError("pre_rho is not in flow.named_parameters(); it would "
                               "never reach the optimizer")
        self.arm_report["prior"] = {"type": "CoupledNodePrior", "changed": True,
                                     **flow.prior.summary()}
        self.logger.info(
            f"arm {self.arm_config.arm}: prior wrapped as CoupledNodePrior "
            f"(rho={self.arm_config.coupled_init_rho}, "
            f"learnable={self.arm_config.rho_learnable}); pre_rho registered")
        return flow

    # -- hook 2: choose the sampler ------------------------------------
    def _make_batcher(self):
        cfg = self.config
        if not self.arm_config.uses_balanced_sampler:
            self.arm_report["sampler"] = {"type": "DeterministicBatcher",
                                          "changed": False}
            return DeterministicBatcher(len(self.train_set), cfg.batch_size, cfg.seed)
        n_node = np.asarray(self.train_set.n_node)
        qed = np.asarray(self.train_set.qed)
        imbalance = imbalance_report(n_node, qed,
                                      qed_bin_width=self.arm_config.qed_bin_width)
        if not imbalance["meaningful_imbalance"]:
            raise RuntimeError(
                f"arm 'balanced' refused: {imbalance['verdict']} "
                f"(effective_cell_fraction={imbalance['effective_cell_fraction']:.3f}, "
                f"population_ratio={imbalance['population_ratio']})")
        sampler = BalancedSampler(
            n_node, qed, batch_size=cfg.batch_size, seed=cfg.seed,
            alpha=self.arm_config.balanced_alpha,
            min_cell_count=self.arm_config.min_cell_count,
            max_exposure_ratio=self.arm_config.max_exposure_ratio,
            qed_bin_width=self.arm_config.qed_bin_width)
        self.arm_report["sampler"] = {"type": "BalancedSampler", "changed": True,
                                      "imbalance": imbalance, **sampler.report()}
        weighting = sampler.report()["weighting"]
        self.logger.info(
            f"arm balanced: alpha={self.arm_config.balanced_alpha} over "
            f"{imbalance['cells']} cells; exposure ratio in "
            f"[{weighting['min_exposure_ratio_realised']:.3f}, "
            f"{weighting['max_exposure_ratio_realised']:.3f}]; effective sample "
            f"fraction {weighting['effective_sample_fraction']:.3f}")
        return sampler

    # -- optimizer-state-aware restore ---------------------------------
    def _restore(self, path, flow, optimizer):
        """Delegate, but adapt the optimizer state when the arm added a parameter."""
        if not self.arm_config.uses_coupled_prior:
            return super()._restore(path, flow, optimizer)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        saved = blob.get("optimizer_state_dict")
        step, history = super()._restore(path, flow, None)
        self.arm_report["optimizer_restore"] = adapt_optimizer_state(optimizer, saved)
        return step, history


# ----------------------------------------------------------- generation side ---
def load_arm_flow(checkpoint_path: str | Path, *, device: str | torch.device = "cpu"):
    """Load a checkpoint from ANY arm, wrapping the prior when the arm used one.

    Generation must not have to know which arm produced a checkpoint. The marker is the
    presence of a `prior.pre_rho` tensor in the state dict, so the wrapping happens
    before `load_state_dict` and the weights land in the right module.
    """
    from zinc_gnf.flow import FlowConfig, NodeFlow
    from zinc_gnf.trainer_flow import ResumeMismatchError, vocabulary_fingerprint

    probe = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if probe.get("vocabulary_fingerprint") != vocabulary_fingerprint():
        raise ResumeMismatchError("checkpoint uses a different atom vocabulary")
    flow = NodeFlow(FlowConfig.from_dict(probe["flow_config"]))
    state = probe.get("model_state_dict") or {}
    coupled = any(key.endswith("prior.pre_rho") for key in state)
    if coupled:
        flow.prior = CoupledNodePrior.from_node_prior(flow.prior, init_rho=INIT_RHO)
    flow.load_state_dict(state)
    flow.to(device).eval()
    if not bool(flow.statistics_set):
        raise ResumeMismatchError("checkpoint carries no embedding statistics")
    return flow, probe | {"coupled_prior": coupled}


__all__ = [
    "ARMS",
    "MIN_EFFECTIVE_CELL_FRACTION",
    "MIN_POPULATION_RATIO",
    "ArmConfig",
    "ArmTrainer",
    "adapt_optimizer_state",
    "imbalance_report",
    "load_arm_flow",
]
