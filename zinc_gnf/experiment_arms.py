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
from zinc_gnf.trainer_flow import FlowTrainer, ResumeMismatchError

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

    # -- arm identity is persisted and verified on resume --------------
    def _state(self, step: int) -> dict[str, Any]:
        state = super()._state(step)
        state["arm"] = self.arm_config.to_dict()
        return state

    def verify_arm(self, checkpoint: dict[str, Any]) -> None:
        """Reject resuming an interrupted run under a DIFFERENT arm."""
        saved = checkpoint.get("arm")
        if saved is None:
            raise ResumeMismatchError(
                "this checkpoint carries no arm block, so it was not written by an arm "
                "run; start a new experiment with prepare_warm_start instead of "
                "resuming")
        current = self.arm_config.to_dict()
        differing = [key for key in ("arm", "balanced_alpha", "coupled_init_rho",
                                      "min_cell_count", "max_exposure_ratio",
                                      "qed_bin_width")
                     if saved.get(key) != current.get(key)]
        if differing:
            raise ResumeMismatchError(
                f"cannot resume: the checkpoint's arm configuration differs in "
                f"{differing} (checkpoint "
                f"{ {k: saved.get(k) for k in differing} } vs now "
                f"{ {k: current.get(k) for k in differing} })")

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


# --------------------------------------------------- warm start translation ---
def parameter_order(flow_config_dict: dict[str, Any]) -> list[tuple[str, tuple]]:
    """`(name, shape)` in the exact order `flow.parameters()` yields them.

    Rebuilding the parent model from its own saved `flow_config` reconstructs the
    ordering its optimizer used, which is what makes a POSITIONAL Adam state
    interpretable by NAME. Nothing here depends on assuming an order.
    """
    from zinc_gnf.flow import FlowConfig, NodeFlow

    model = NodeFlow(FlowConfig.from_dict(flow_config_dict))
    return [(name, tuple(tensor.shape))
            for name, tensor in model.named_parameters()]


def map_optimizer_state(parent_blob: dict[str, Any], arm_flow, *,
                        learning_rate: float | None = None) -> dict[str, Any]:
    """Re-index the parent's Adam moments onto the arm's parameters BY NAME.

    Adam stores state positionally, so the parent's indices are meaningless against a
    model with an extra tensor. The parent's ordering is reconstructed from its own
    `flow_config`, turned into index -> name, then name -> the arm's index. Every parent
    name must exist in the arm with an identical shape or this raises; a new parameter is
    left with no moments, which Adam creates on its first step.
    """
    saved = parent_blob.get("optimizer_state_dict")
    if not saved:
        return {"optimizer_state_dict": None,
                "mapping": {"restored": False, "reason": "parent has no optimizer state"}}
    parent_order = parameter_order(parent_blob["flow_config"])
    arm_order = [(name, tuple(t.shape)) for name, t in arm_flow.named_parameters()]
    arm_index = {name: position for position, (name, _) in enumerate(arm_order)}
    arm_shape = dict(arm_order)

    groups = saved.get("param_groups") or []
    positional = [index for group in groups for index in group["params"]]
    if len(positional) != len(parent_order):
        raise ValueError(
            f"parent optimizer state covers {len(positional)} parameters but its "
            f"flow_config reconstructs {len(parent_order)}; refusing to guess the "
            f"mapping")

    remapped_state: dict[int, Any] = {}
    mapped, missing, reshaped = [], [], []
    for position, (name, shape) in enumerate(parent_order):
        if name not in arm_index:
            missing.append(name)
            continue
        if arm_shape[name] != shape:
            reshaped.append({"name": name, "parent": list(shape),
                             "arm": list(arm_shape[name])})
            continue
        entry = (saved.get("state") or {}).get(position)
        if entry is not None:
            remapped_state[arm_index[name]] = entry
        mapped.append(name)
    if missing or reshaped:
        raise ValueError(
            f"parameter-name mapping failed: missing from the arm {missing}; shape "
            f"mismatches {reshaped}")

    template = dict(groups[0]) if groups else {}
    template["params"] = list(range(len(arm_order)))
    # Adam's state_dict carries `lr` inside param_groups, so loading the parent's state
    # would silently reinstate the PARENT's learning rate. The experiment rate is written
    # in explicitly, after the moments are mapped.
    previous_lr = template.get("lr")
    if learning_rate is not None:
        template["lr"] = float(learning_rate)
    added = [name for name, _ in arm_order if name not in dict(parent_order)]
    return {
        "optimizer_state_dict": {"state": remapped_state, "param_groups": [template]},
        "mapping": {
            "restored": True,
            "parent_parameters": len(parent_order),
            "arm_parameters": len(arm_order),
            "moments_mapped": len(remapped_state),
            "names_mapped": len(mapped),
            "new_parameters_without_moments": added,
            "method": ("parent ordering reconstructed from its own flow_config, then "
                       "index -> name -> arm index, with shapes validated"),
            "learning_rate_in_param_groups": {
                "parent": previous_lr, "applied": template.get("lr"),
                "note": ("set explicitly after mapping, because Adam's state_dict "
                         "carries lr in param_groups and would otherwise reinstate the "
                         "parent's rate"),
            },
        },
    }


def prepare_warm_start(parent_path, target_path, *, arm_config: ArmConfig,
                       train_config, flow_config, sampler_state: dict[str, Any],
                       device: str = "cpu") -> dict[str, Any]:
    """Translate the common parent into an ARM-COMPATIBLE `last.pt`. PATH 1.

    `FlowTrainer._restore` is strict, and deliberately so: it compares the checkpoint's
    `seed`, `batch_size` and LEARNING RATE against the live config, restores the batcher
    state unconditionally, and inherits `best_loss` / `best_step` / `bad_evals`. A naive
    warm start therefore fails three ways -- the experiment learning rate is an override,
    a BalancedSampler cannot take a DeterministicBatcher state, and inherited
    best-checkpoint bookkeeping from a 50k run would stop a 3k run at lr 1e-5 ever
    writing `best.pt`.

    Rather than weaken those checks, this writes a checkpoint that satisfies them
    honestly:

      * `train_config` is the EXPERIMENT's, so the learning-rate override is recorded in
        the artefact and the strict comparison passes by agreement rather than by
        exemption;
      * `batcher` is the arm's OWN fresh sampler state, chosen deliberately;
      * `best_loss` / `best_step` / `bad_evals` / `history` are RESET, so selection and
        early stopping are experiment-local;
      * `step` is 0, so `max_steps = 3000` means exactly 3,000 additional optimizer steps
        and the warm-up is experiment-local;
      * parent weights are loaded into the base prior BEFORE it is wrapped, so no key
        remapping of model weights is needed at all;
      * Adam moments are re-indexed by verified parameter NAME;
      * RNG and noise-generator states are deliberately OMITTED, so every arm starts
        from `set_seed(cfg.seed)` and the four arms share identical randomness;
      * the parent is recorded separately under `parent`, never conflated with this
        run's own identity.

    Vocabulary, flow config, autoencoder and data fingerprints are carried over from the
    parent unchanged, so those compatibility checks remain fully active.
    """
    import torch as _torch
    from zinc_gnf.flow import NodeFlow
    from zinc_gnf.trainer_flow import file_sha256, vocabulary_fingerprint

    parent_path, target_path = Path(parent_path), Path(target_path)
    blob = _torch.load(parent_path, map_location="cpu", weights_only=False)
    if blob.get("vocabulary_fingerprint") != vocabulary_fingerprint():
        raise ResumeMismatchError("parent checkpoint uses a different atom vocabulary")
    if FlowConfigLike(blob["flow_config"]) != FlowConfigLike(flow_config.to_dict()):
        raise ResumeMismatchError(
            "parent flow configuration differs from the arm's; the architecture must be "
            "identical for the comparison to mean anything")

    flow = NodeFlow(flow_config, init_seed=train_config.seed)
    # weights FIRST, wrapping SECOND: no model-key remapping is ever required
    flow.load_state_dict(blob["model_state_dict"])
    if arm_config.uses_coupled_prior:
        flow.prior = CoupledNodePrior.from_node_prior(
            flow.prior, init_rho=arm_config.coupled_init_rho,
            learnable=arm_config.rho_learnable)
    flow.to(device)

    optimizer = map_optimizer_state(
        blob, flow, learning_rate=train_config.learning_rate)
    record = {
        "step": 0,
        "epoch": 0,
        "best_metric": float("inf"),
        "best_loss": None,
        "best_step": -1,
        "bad_evals": 0,
        "history": [],
        "batcher": sampler_state,
        "model_state_dict": flow.state_dict(),
        "optimizer_state_dict": optimizer["optimizer_state_dict"],
        "configuration": {"flow": flow_config.to_dict(),
                          "training": train_config.to_dict()},
        "flow_config": flow_config.to_dict(),
        "train_config": train_config.to_dict(),
        "vocabulary_fingerprint": blob.get("vocabulary_fingerprint"),
        "autoencoder": blob.get("autoencoder"),
        "data_fingerprint": blob.get("data_fingerprint"),
        "embedding_statistics": blob.get("embedding_statistics"),
        "arm": arm_config.to_dict(),
        "parent": {
            "checkpoint": str(parent_path),
            "sha256": file_sha256(parent_path),
            "step": blob.get("step"),
            "learning_rate": (blob.get("train_config") or {}).get("learning_rate"),
            "best_loss": blob.get("best_loss"),
            "note": ("the parent's identity, kept separate from this run's. Its "
                     "best_loss and step are NOT inherited."),
        },
        "warm_start": {
            "learning_rate_override": {
                "parent": (blob.get("train_config") or {}).get("learning_rate"),
                "experiment": train_config.learning_rate,
                "recorded_because": ("_restore compares learning_rate strictly; the "
                                     "override is written into train_config so the "
                                     "check passes by agreement, not by exemption"),
            },
            "optimizer_mapping": optimizer["mapping"],
            "reset": ["step", "best_loss", "best_step", "bad_evals", "history",
                      "batcher", "torch_rng_state", "noise_generator_state"],
            "prior_wrapped_after_loading_weights": arm_config.uses_coupled_prior,
        },
    }
    target_path.parent.mkdir(parents=True, exist_ok=True)
    _torch.save(record, target_path)
    return {"target": str(target_path), "parent": record["parent"],
            "warm_start": record["warm_start"]}


def FlowConfigLike(value):            # noqa: N802 - tiny normaliser, not a class
    """Compare flow configs as plain dicts without caring about key order."""
    return tuple(sorted((str(k), str(v)) for k, v in dict(value).items()))


# ------------------------------------------------- prior-agnostic latent draw ---
def draw_latents(flow, mask, condition, *, eps, shared, temperature: float = 1.0):
    """Build `z` from EXTERNALLY supplied noise, correctly for either prior.

    Evaluation needs the noise held fixed across checkpoints, so it cannot call
    `prior.sample`, which draws its own. Building `z` as `mu + T*sigma*eps` is correct
    for `NodePrior` but SILENTLY WRONG for `CoupledNodePrior`: it omits the shared
    graph-level component entirely, so a coupled checkpoint would be evaluated as though
    rho were zero and the intervention would look like it did nothing.

    This reproduces each prior's own sampler exactly while taking both noise tensors from
    the caller, so pairing by noise id survives:

        NodePrior          z = mu + T * sigma * eps
        CoupledNodePrior   z = mu + T * sigma * ( sqrt(1-rho)*eps + sqrt(rho)*shared )

    `eps` is `[B, N, D]` and `shared` is `[B, 1, D]`, both from the evaluation's own
    seeded stream.
    """
    mu, sigma = flow.prior.parameters_for(eps.shape[0], condition, eps.device)
    rho_fn = getattr(flow.prior, "rho", None)
    if callable(rho_fn):
        rho = rho_fn().view(1, 1, -1)
        deviation = torch.sqrt(1.0 - rho) * eps + torch.sqrt(rho) * shared
    else:
        deviation = eps
    z = mu.unsqueeze(1) + float(temperature) * sigma.unsqueeze(1) * deviation
    return z * mask.unsqueeze(-1).to(z.dtype)


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
    "draw_latents",
    "map_optimizer_state",
    "parameter_order",
    "prepare_warm_start",
    "imbalance_report",
    "load_arm_flow",
]
