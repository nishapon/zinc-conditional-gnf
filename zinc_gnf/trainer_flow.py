"""Flow training loop: frozen autoencoder, validation selection, exact resume.

Protocol (project decisions, unchanged from the autoencoder stage):
  * train on the TRAIN split only; validation is for selection and tuning;
    the TEST split is never loaded here (`load_training_splits` drops it);
  * the autoencoder is loaded from a checkpoint, put in `eval()` mode, has
    `requires_grad_(False)` on every parameter and is only ever called inside
    `torch.no_grad()`. Its file SHA-256 goes into the flow's config.json and
    every checkpoint, so a flow can always be traced to the exact encoder that
    produced its training data;
  * `best.pt` = best validation per-node NLL, `last.pt` = most recent state,
    from which training resumes exactly.

The density being learned is the SMOOTHED embedding density: before whitening,
`noise_sigma * per-dimension std` of Gaussian noise is added to every real
node (default 0.05, the value whose effect on the chosen checkpoint's degree
decoder is measured: 93.40% -> 92.20% exact). The deterministic encoder gives
symmetry-equivalent atoms identical embeddings, so the unsmoothed empirical
distribution is singular and its likelihood is unbounded. Validation uses a
FIXED noise seed, so the validation number is a deterministic function of the
model.

`pause_at_step` stops a run cleanly at an intermediate step while KEEPING
`max_steps` as the target: the summary records `paused: true` with
`completed: false`, so the cluster script's finished-seed check still sees an
unfinished run and resubmitting resumes from exactly that step. It exists so
the paused checkpoint can be sampled from (a mid-run generation check) before
committing GPU hours to the rest of the run. A pause point that has already
been passed is ignored, so leaving the flag set on the resubmission is safe.

Embedding statistics (mean and per-dimension std) are measured once on a
seeded subset of the training split, stored in the model as buffers, and
reused on resume and at generation time. They are never recomputed from
validation or test data.

Everything written goes under `output_dir`:
    config.json, metrics.jsonl, train.log, last.pt, best.pt, summary.json
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from zinc_gnf.conditioning import (
    CONDITION_SHUFFLE_SEED,
    batch_condition,
    condition_columns,
    condition_embedding_for,
)
from zinc_gnf.constants_charge_aware import vocabulary_fingerprint
from zinc_gnf.dataset import (
    DeterministicBatcher,
    MolecularBatch,
    MolecularGraphDataset,
    iter_eval_batches,
    load_training_splits,
    make_batch,
    seeded_subset_indices,
)
from zinc_gnf.flow import FlowConfig, NodeFlow
from zinc_gnf.trainer_autoencoder import ResumeMismatchError, load_autoencoder, resolve_device
from zinc_gnf.utils.checkpointing import load_checkpoint, save_checkpoint
from zinc_gnf.utils.logging import JobLogger
from zinc_gnf.utils.runtime import StopFlag, atomic_write_json, current_rss_mb, format_duration
from zinc_gnf.utils.seed import set_seed

VALIDATION_SUBSET_SEED = 12345      # same constant as the autoencoder trainer, on purpose
VALIDATION_NOISE_SEED = 777         # fixed, so validation loss is deterministic
STATISTICS_SEED = 4242              # subset used to measure embedding mean/std


@dataclass
class FlowTrainConfig:
    data_dir: str = "data/processed/zinc250k_charge_aware_moflow_fixed"
    output_dir: str = "outputs/checkpoints/flow_uncond"
    ae_checkpoint: str = "outputs/ablation/ae_degree/seed_3/last.pt"
    seed: int = 1
    device: str = "auto"                   # auto | cpu | cuda
    batch_size: int = 128
    max_steps: int = 50_000
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float | None = 5.0
    warmup_steps: int = 500                # linear LR warm-up; flows are fragile early on
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 1000
    val_molecules: int | None = 5000       # fixed seeded validation subset
    statistics_molecules: int = 20_000     # training molecules used for mean/std
    pause_at_step: int | None = None       # stop cleanly at this step WITHOUT finishing the run
    log_det_warn_threshold: float = 50.0   # |log|det J||/node above this is logged as a warning
    early_stopping_patience: int | None = 10       # in evaluations; None = off
    max_train_molecules: int | None = None         # smoke tests only
    ae_seen_only: bool = False             # restrict training to the AE's own 100k subset
    ae_subset_seed: int | None = 3         # seed that selected that subset (AE run seed)
    ae_subset_size: int = 100_000
    deterministic: bool = False
    # --- optional conditional QED-consistency auxiliary objective (training only) ------
    # 0.0 = OFF. Every existing config and checkpoint is byte-identical in behaviour when
    # this is 0. See zinc_gnf/qed_consistency.py. Requires 'qed' in condition_features.
    # The objective adds no parameters to NodeFlow and does not change the density, so a
    # checkpoint trained with it stays a drop-in for every generation/evaluation script.
    qed_consistency_weight: float = 0.0        # lambda on the auxiliary MSE term
    qed_consistency_warmup: int = 2000         # steps of pure NLL before the term switches on
    qed_consistency_ramp: int = 2000           # linear ramp of lambda after the warmup
    qed_consistency_subbatch: int = 32         # molecules used for the (sequential) inverse pass
    qed_consistency_temperature: float = 1.0   # prior temperature when sampling the aux term
    qed_surrogate_hidden: int = 128            # QED read-out MLP hidden width
    qed_surrogate_lr: float = 1e-3             # the read-out's own optimiser learning rate
    qed_consistency_every: int = 1             # apply the (sequential-inverse) aux term every N steps
    # --- N-matched QED-CONTRASTIVE objective (forward-only; the stable replacement) ----
    # 0.0 = OFF. Adds NO parameters and NO checkpoint keys, so checkpoints stay
    # bit-compatible with load_flow and every generation/evaluation script. Requires
    # 'qed' in condition_features. See zinc_gnf/qed_contrastive.py for the derivation,
    # the measured root cause of the sampling objective's divergence, and the limits.
    qed_contrast_weight: float = 0.0           # lambda on the contrastive cross-entropy
    qed_contrast_warmup: int = 2000            # steps of pure NLL before it switches on
    qed_contrast_ramp: int = 2000              # linear ramp of lambda after the warmup
    qed_contrast_offsets: tuple[float, ...] = (0.0, -1.0, 1.0)   # offsets[0] MUST be 0.0
    qed_contrast_subbatch: int = 32            # molecules scored per application
    qed_contrast_every: int = 1                # apply every N steps (cost control)
    qed_contrast_min_separation: float = 0.25  # standardised-QED gap for a usable negative
    qed_contrast_logit_scale: float = 1.0      # sharpness of the candidate softmax
    qed_contrast_negatives: str = "offset"     # 'offset' (deterministic) | 'inbatch'
    # Non-finite auxiliary updates are NOT silently skipped. A few transient skips are
    # tolerated; exceeding this rate after the grace window raises, because a silently
    # absent objective is the failure mode of the previous attempt (job 360343).
    qed_contrast_max_skip_rate: float = 0.02
    qed_contrast_skip_grace: int = 50          # applications before the rate is enforced

    def validate(self) -> None:
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
        for name in ("batch_size", "max_steps", "eval_every", "save_every", "log_every",
                     "statistics_molecules"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.pause_at_step is not None and not 1 <= self.pause_at_step <= self.max_steps:
            raise ValueError(f"pause_at_step must lie in 1..max_steps "
                             f"({self.pause_at_step} vs max_steps={self.max_steps})")
        if self.ae_seen_only and self.ae_subset_seed is None:
            raise ValueError("ae_seen_only needs ae_subset_seed (the autoencoder run's seed)")
        if self.qed_consistency_weight < 0:
            raise ValueError("qed_consistency_weight must be >= 0")
        for name in ("qed_consistency_warmup", "qed_consistency_ramp"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.qed_consistency_weight > 0:
            if self.qed_consistency_subbatch < 1:
                raise ValueError("qed_consistency_subbatch must be >= 1 when the objective is on")
            if self.qed_consistency_temperature <= 0:
                raise ValueError("qed_consistency_temperature must be > 0")
            if self.qed_surrogate_hidden < 1:
                raise ValueError("qed_surrogate_hidden must be >= 1")
            if self.qed_consistency_every < 1:
                raise ValueError("qed_consistency_every must be >= 1")
        # --- contrastive objective ---------------------------------------------------
        self.qed_contrast_offsets = tuple(float(o) for o in self.qed_contrast_offsets)
        if self.qed_contrast_weight < 0:
            raise ValueError("qed_contrast_weight must be >= 0")
        for name in ("qed_contrast_warmup", "qed_contrast_ramp", "qed_contrast_skip_grace"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.qed_contrast_weight > 0:
            if self.qed_consistency_weight > 0:
                raise ValueError(
                    "qed_contrast_weight and qed_consistency_weight must not both be > 0: "
                    "the contrastive objective REPLACES the sampling-based one (which was "
                    "measured to diverge). Enable exactly one.")
            if len(self.qed_contrast_offsets) < 2:
                raise ValueError("qed_contrast_offsets needs at least one negative")
            if self.qed_contrast_offsets[0] != 0.0:
                raise ValueError("qed_contrast_offsets[0] must be exactly 0.0 (the true candidate)")
            if self.qed_contrast_subbatch < 2:
                raise ValueError("qed_contrast_subbatch must be >= 2")
            if self.qed_contrast_every < 1:
                raise ValueError("qed_contrast_every must be >= 1")
            if self.qed_contrast_min_separation <= 0:
                raise ValueError("qed_contrast_min_separation must be > 0")
            if self.qed_contrast_logit_scale <= 0:
                raise ValueError("qed_contrast_logit_scale must be > 0")
            if self.qed_contrast_negatives not in ("offset", "inbatch"):
                raise ValueError("qed_contrast_negatives must be 'offset' or 'inbatch'")
            if not 0.0 <= self.qed_contrast_max_skip_rate <= 1.0:
                raise ValueError("qed_contrast_max_skip_rate must lie in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "FlowTrainConfig":
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown flow training settings: {sorted(unknown)}")
        return cls(**values)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def autoencoder_seen_indices(pool_size: int, seed: int, count: int) -> list[int]:
    """Indices of the training pool the autoencoder actually saw.

    `scripts/03b_reconstruction_experiment.py` picked its training molecules as
    `sorted(np.random.default_rng(seed).permutation(pool_size)[:count])`; this
    reproduces that exactly so the flow can be told which embeddings come from
    molecules the encoder was fitted on.
    """
    if count >= pool_size:
        return list(range(pool_size))
    order = np.random.default_rng(int(seed)).permutation(int(pool_size))[:int(count)]
    return sorted(int(i) for i in order)


@torch.no_grad()
def encode_batch(autoencoder, batch: MolecularBatch) -> torch.Tensor:
    """Node embeddings from the FROZEN encoder. Never tracks gradients."""
    return autoencoder.encode(batch)


@torch.no_grad()
def embedding_statistics_from_dataset(autoencoder, dataset: MolecularGraphDataset, *,
                                      indices: list[int], device: torch.device | str,
                                      batch_size: int = 256) -> dict[str, Any]:
    """Per-dimension mean / std of real-node embeddings (streaming, no big buffer)."""
    dim = autoencoder.config.node_embedding_dim
    total = torch.zeros(dim, dtype=torch.float64, device=device)
    total_square = torch.zeros(dim, dtype=torch.float64, device=device)
    count = 0
    for batch in iter_eval_batches(dataset, batch_size, indices):
        batch = batch.to(device)
        nodes = encode_batch(autoencoder, batch)
        mask = batch.node_mask.unsqueeze(-1).to(nodes.dtype)
        real = (nodes * mask).reshape(-1, dim).to(torch.float64)
        weights = mask.reshape(-1, 1).to(torch.float64)
        total += (real * weights).sum(0)
        total_square += (real * real * weights).sum(0)
        count += int(batch.node_mask.sum())
    if count < 2:
        raise ValueError("need at least two real nodes to measure embedding statistics")
    mean = total / count
    variance = (total_square / count - mean * mean).clamp(min=0.0)
    std = variance.sqrt()
    return {
        "nodes": count,
        "mean": mean.to(torch.float32).cpu(),
        "std": std.to(torch.float32).cpu(),
        "std_mean": float(std.mean()),
        "std_min": float(std.min()),
        "std_max": float(std.max()),
        "norm_of_mean": float(mean.norm()),
    }


class FlowTrainer:
    def __init__(self, flow_config: FlowConfig, train_config: FlowTrainConfig,
                 logger: JobLogger | None = None,
                 datasets: tuple[MolecularGraphDataset, MolecularGraphDataset, dict[str, Any]] | None = None,
                 autoencoder=None) -> None:
        """`datasets` and `autoencoder` are injection points for tests; normal runs
        read the dataset from `train_config.data_dir` and the encoder from
        `train_config.ae_checkpoint`."""
        train_config.validate()
        self.flow_config, self.config = flow_config, train_config
        self._preloaded = datasets
        self._autoencoder = autoencoder
        self.output_dir = Path(train_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or JobLogger("flow", self.output_dir, filename="train.log")
        self.device = resolve_device(train_config.device)

    # ------------------------------------------------------------------
    def _load_data(self) -> None:
        if self._preloaded is not None:
            self.train_set, self.val_set, fingerprint = self._preloaded
            self.ae_seen_mask_note = "preloaded datasets"
        else:
            train_records, val_records, fingerprint = load_training_splits(self.config.data_dir)
            pool_size = len(train_records)
            if self.config.ae_seen_only:
                keep = autoencoder_seen_indices(pool_size, self.config.ae_subset_seed,
                                                self.config.ae_subset_size)
                train_records = [train_records[i] for i in keep]
                self.ae_seen_mask_note = (f"restricted to the {len(keep):,} molecules the autoencoder "
                                          f"saw (rng seed {self.config.ae_subset_seed})")
            else:
                self.ae_seen_mask_note = f"all {pool_size:,} training molecules"
            if self.config.max_train_molecules:
                train_records = train_records[: self.config.max_train_molecules]
            self.train_set = MolecularGraphDataset(train_records)
            self.val_set = MolecularGraphDataset(val_records)
            del train_records, val_records
        fingerprint = dict(fingerprint)
        fingerprint["used"] = {"train": len(self.train_set), "val": len(self.val_set)}
        fingerprint["ae_seen_only"] = bool(self.config.ae_seen_only)
        self.data_fingerprint = fingerprint
        self.val_indices = seeded_subset_indices(len(self.val_set), self.config.val_molecules,
                                                 VALIDATION_SUBSET_SEED)

    def _load_autoencoder(self) -> None:
        if self._autoencoder is not None:
            self.autoencoder = self._autoencoder
            self.ae_fingerprint = {"checkpoint": "<injected>", "sha256": None,
                                   "model_config": self.autoencoder.config.to_dict()}
        else:
            path = Path(self.config.ae_checkpoint)
            if not path.exists():
                raise FileNotFoundError(f"autoencoder checkpoint not found: {path}")
            model, probe = load_autoencoder(path, device=self.device)
            self.autoencoder = model
            self.ae_fingerprint = {
                "checkpoint": str(path), "sha256": file_sha256(path),
                "model_config": model.config.to_dict(),
                "step": probe.get("step"), "data": probe.get("data_fingerprint"),
            }
        # FREEZE. The flow must never change the encoder that defines its data.
        self.autoencoder.to(self.device).eval()
        for parameter in self.autoencoder.parameters():
            parameter.requires_grad_(False)
        if self.autoencoder.config.node_embedding_dim != self.flow_config.node_embedding_dim:
            raise ResumeMismatchError(
                f"autoencoder embedding width {self.autoencoder.config.node_embedding_dim} != "
                f"flow node_embedding_dim {self.flow_config.node_embedding_dim}")
        if self.autoencoder.config.symmetry_noise_dim > 0:
            self.logger.warn("the autoencoder has symmetry-noise channels: its embeddings are "
                             "stochastic, so the flow's target density includes encoder noise")

    def _fit_statistics(self, flow: NodeFlow) -> None:
        indices = seeded_subset_indices(len(self.train_set), self.config.statistics_molecules,
                                        STATISTICS_SEED)
        statistics = embedding_statistics_from_dataset(
            self.autoencoder, self.train_set, indices=indices, device=self.device)
        flow.set_data_statistics(statistics["mean"], statistics["std"])
        self.statistics_summary = {k: v for k, v in statistics.items() if k not in ("mean", "std")}
        self.statistics_summary["molecules"] = len(indices)
        self.statistics_summary["floored_dimensions"] = flow.floored_dimensions()
        self.logger.info(
            f"embedding statistics on {len(indices):,} training molecules "
            f"({statistics['nodes']:,} nodes): per-dim std mean={statistics['std_mean']:.4f} "
            f"min={statistics['std_min']:.4f} max={statistics['std_max']:.4f}; "
            f"{flow.floored_dimensions()} of {self.flow_config.node_embedding_dim} dimensions below the "
            f"whitening floor {self.flow_config.whiten_std_floor}")

    # ------------------------------------------------------------------
    def _learning_rate(self, step: int) -> float:
        base = self.config.learning_rate
        warmup = self.config.warmup_steps
        if warmup and step < warmup:
            return base * (step + 1) / warmup
        return base

    def _condition(self, flow: NodeFlow, batch: MolecularBatch) -> torch.Tensor | None:
        """Conditioning embedding for one batch, or None for the unconditional model.

        Each molecule is conditioned on its OWN standardised (log1p(N), QED) columns, as
        written by the preprocessing with the train-only scaler: plain conditional maximum
        likelihood, no label smoothing or feature dropout at this stage.
        """
        if not flow.config.conditional:
            return None
        values, present = batch_condition(batch, flow.config.condition_features)
        return condition_embedding_for(flow, values, present, batch.batch_size, self.device)

    @torch.no_grad()
    def validate(self, flow: NodeFlow) -> dict[str, Any]:
        """Per-node NLL on the fixed validation subset with the fixed noise seed."""
        was_training = flow.training
        flow.eval()
        generator = torch.Generator(device=self.device).manual_seed(VALIDATION_NOISE_SEED)
        totals = {"log_prob": 0.0, "log_prob_z": 0.0, "log_det": 0.0, "z_norm": 0.0}
        nodes = 0
        molecules = 0
        # Conditional models get a PERMUTED-CONDITION control in the same pass: the same
        # molecules and the same smoothing noise, scored against another molecule's
        # condition vector. A model that ignores its conditioning scores both identically,
        # so this gap is the running evidence that the conditioning is alive. One extra
        # forward pass over the validation subset, no extra data.
        shuffle_rng = (np.random.default_rng(CONDITION_SHUFFLE_SEED)
                       if flow.config.conditional else None)
        shuffled_log_prob = 0.0
        condition_wins = 0
        condition_molecules = 0
        # Held-out ACCURACY OF THE QED SURROGATE on real (clean) embeddings. This is the
        # read-out the consistency term trusts; it is NOT the autoencoder's reconstruction
        # QED. A weak surrogate here means the auxiliary gradient is unreliable, so it is
        # reported every evaluation and must be inspected before trusting any QED gain.
        surrogate = getattr(self, "surrogate", None)
        qed_col = getattr(self, "_qed_col", None)
        surr_abs_error = 0.0
        surr_count = 0
        surr_pred: list[float] = []
        surr_true: list[float] = []
        for batch in iter_eval_batches(self.val_set, max(self.config.batch_size, 64), self.val_indices):
            batch = batch.to(self.device)
            clean = encode_batch(self.autoencoder, batch)
            embeddings = flow.add_smoothing_noise(clean, batch.node_mask, generator=generator)
            condition = self._condition(flow, batch)
            parts = flow.log_prob(embeddings, batch.node_mask, condition)
            if surrogate is not None and qed_col is not None:
                target = batch_condition(batch, flow.config.condition_features)[0][:, qed_col]
                prediction = surrogate(clean, batch.node_mask)
                surr_abs_error += float((prediction - target).abs().sum())
                surr_count += int(target.shape[0])
                surr_pred.extend(float(v) for v in prediction.detach().cpu())
                surr_true.extend(float(v) for v in target.detach().cpu())
            if shuffle_rng is not None and batch.batch_size > 1:
                values, present = batch_condition(batch, flow.config.condition_features)
                order = shuffle_rng.permutation(batch.batch_size)
                if np.array_equal(order, np.arange(batch.batch_size)):
                    order = np.roll(order, 1)
                permuted = values[torch.as_tensor(order.copy(), device=values.device)]
                control = flow.log_prob(
                    embeddings, batch.node_mask,
                    condition_embedding_for(flow, permuted, present, batch.batch_size, self.device))
                shuffled_log_prob += float(control["log_prob"].sum())
                condition_wins += int((parts["log_prob"] > control["log_prob"]).sum())
                condition_molecules += batch.batch_size
            totals["log_prob"] += float(parts["log_prob"].sum())
            totals["log_prob_z"] += float(parts["log_prob_z"].sum())
            totals["log_det"] += float(parts["log_det"].sum())
            totals["z_norm"] += float((parts["z"].norm(dim=-1) * batch.node_mask).sum())
            nodes += int(batch.node_mask.sum())
            molecules += batch.batch_size
        if was_training:
            flow.train()
        result = {
            "val_loss": -totals["log_prob"] / max(nodes, 1),
            "val_log_prob_z_per_node": totals["log_prob_z"] / max(nodes, 1),
            "val_log_det_per_node": totals["log_det"] / max(nodes, 1),
            "val_z_norm": totals["z_norm"] / max(nodes, 1),
            "val_molecules": molecules, "val_nodes": nodes,
        }
        if condition_molecules:
            shuffled_loss = -shuffled_log_prob / max(nodes, 1)
            result.update({
                "val_loss_shuffled_condition": shuffled_loss,
                # > 0 means the molecules' own conditions explain them better than
                # permuted ones; 0 means the conditioning is being ignored.
                "val_condition_gap": shuffled_loss - result["val_loss"],
                "val_condition_win_rate": condition_wins / condition_molecules,
            })
        # Root-cause diagnostic (a): the NLL only ever sees sum(log_s) of each invertible
        # linear, never its CONDITIONING, yet flow.inverse must invert these matrices and
        # the gradient of that inverse scales with ||W^-1||^2. Logged read-only.
        if self.flow_config.invertible_linear:
            try:
                from zinc_gnf.qed_contrastive import invertible_linear_conditioning
                result.update({f"val_{k}": v
                               for k, v in invertible_linear_conditioning(flow).items()})
            except Exception:  # noqa: BLE001 - a diagnostic must never stop training
                pass
        if surr_count:
            # MAE is in STANDARDISED QED units (the units the consistency term uses);
            # divide by the scaler's QED std to read it in raw QED. Correlation is unitless.
            result["val_surrogate_qed_mae_standardised"] = surr_abs_error / surr_count
            if len(surr_pred) > 2 and np.std(surr_pred) > 1e-8 and np.std(surr_true) > 1e-8:
                result["val_surrogate_qed_corr"] = float(np.corrcoef(surr_pred, surr_true)[0, 1])
            else:
                result["val_surrogate_qed_corr"] = float("nan")
        return result

    def _save(self, name: str, flow, optimizer, extra_state: dict[str, Any]) -> Path:
        path = self.output_dir / name
        save_checkpoint(path, flow, optimizer, epoch=self.batcher.epoch,
                        best_metric=self.best_loss if self.best_loss is not None else float("inf"),
                        configuration={"flow": self.flow_config.to_dict(),
                                       "training": self.config.to_dict()},
                        extra=extra_state)
        return path

    def _state(self, step: int) -> dict[str, Any]:
        state = {
            "step": step,
            "batcher": self.batcher.state_dict(),
            "best_loss": self.best_loss,
            "best_step": self.best_step,
            "bad_evals": self.bad_evals,
            # Contrastive-objective counters only (it has NO parameters), so the
            # checkpoint stays loadable by load_flow and by a baseline resume.
            "contrast_applied": int(getattr(self, "contrast_applied", 0)),
            "contrast_skipped": int(getattr(self, "contrast_skipped", 0)),
            "data_fingerprint": self.data_fingerprint,
            "autoencoder": self.ae_fingerprint,
            "embedding_statistics": self.statistics_summary,
            "vocabulary_fingerprint": vocabulary_fingerprint(),
            "flow_config": self.flow_config.to_dict(),
            "train_config": self.config.to_dict(),
            "numpy_rng_state": np.random.get_state(),
            # The smoothing noise comes from a dedicated generator, so its state must be
            # saved too: without it a resumed run would replay the noise stream from the
            # beginning and 3+3 steps would not equal 6.
            "noise_generator_state": (self.noise_generator.get_state().cpu()
                                      if getattr(self, "noise_generator", None) is not None else None),
        }
        # Training-only QED-consistency state, saved under its own keys so a checkpoint
        # stays loadable by load_flow (which rebuilds the NodeFlow only) and by any
        # baseline resume (which simply never reads these keys).
        if getattr(self, "surrogate", None) is not None:
            state["qed_surrogate_state"] = self.surrogate.state_dict()
            state["qed_surrogate_optimizer_state"] = self.surrogate_optimizer.state_dict()
            state["consistency_generator_state"] = (
                self.consistency_generator.get_state().cpu()
                if getattr(self, "consistency_generator", None) is not None else None)
        return state

    # ------------------------------------------------------------------
    def train(self, resume: str | Path | None = "auto", stop_flag: StopFlag | None = None) -> dict[str, Any]:
        cfg, log = self.config, self.logger
        set_seed(cfg.seed, deterministic=cfg.deterministic)
        features = self.flow_config.condition_features
        log.header(("Node-feature graph normalizing flow, conditioned on "
                    + ", ".join(features)) if features
                   else "Unconditional node-feature graph normalizing flow")
        log.info(f"device={self.device} seed={cfg.seed} batch_size={cfg.batch_size} "
                 f"max_steps={cfg.max_steps} lr={cfg.learning_rate} warmup={cfg.warmup_steps}")
        self._load_data()
        log.info(f"data {cfg.data_dir}: train={len(self.train_set):,} val={len(self.val_set):,} "
                 f"(test split NOT loaded) protocol={self.data_fingerprint.get('split_protocol')}")
        log.info(f"training pool: {self.ae_seen_mask_note}")
        log.info(f"validation subset: {len(self.val_indices):,} molecules "
                 f"(seed {VALIDATION_SUBSET_SEED}, noise seed {VALIDATION_NOISE_SEED})")
        self._load_autoencoder()
        log.info(f"frozen autoencoder: {self.ae_fingerprint['checkpoint']} "
                 f"sha256={str(self.ae_fingerprint['sha256'])[:16]} "
                 f"degree_head={self.autoencoder.config.degree_head} "
                 f"trainable_parameters={sum(p.requires_grad for p in self.autoencoder.parameters())}")

        if features:
            # Fail at startup, not at step 1 of a queued GPU job.
            for name, dataset in (("train", self.train_set), ("val", self.val_set)):
                if getattr(dataset, "condition", None) is None:
                    raise ValueError(
                        f"the {name} split carries no condition vectors, so a flow conditioned "
                        f"on {list(features)} cannot be trained on {cfg.data_dir}; the "
                        f"preprocessing writes them together with the train-only scaler")
            columns = condition_columns(features)
            counts = np.asarray(self.train_set.n_node)
            log.info(f"conditioning on {list(features)} = dataset condition columns {columns} "
                     f"(standardised with the TRAIN-ONLY scaler written by the preprocessing); "
                     f"training N ranges {int(counts.min())}-{int(counts.max())}, "
                     f"mean {counts.mean():.2f}")
            log.info("every evaluation also scores the same molecules against PERMUTED "
                     "conditions (val_condition_gap / val_condition_win_rate): a gap of 0 "
                     "means the conditioning is being ignored")

        flow = NodeFlow(self.flow_config, init_seed=cfg.seed).to(self.device)
        log.info(f"flow parameters: {flow.num_parameters():,} "
                 f"(blocks={self.flow_config.num_blocks} hidden={self.flow_config.hidden_dim} "
                 f"max_log_scale={self.flow_config.max_log_scale} "
                 f"conditional={self.flow_config.conditional})")
        optimizer = torch.optim.Adam(flow.parameters(), lr=cfg.learning_rate,
                                     weight_decay=cfg.weight_decay)
        self.batcher = DeterministicBatcher(len(self.train_set), cfg.batch_size, cfg.seed)
        self.best_loss: float | None = None
        self.best_step = -1
        self.bad_evals = 0
        self.statistics_summary: dict[str, Any] = {}
        self.noise_generator = torch.Generator(device=self.device).manual_seed(cfg.seed * 7919 + 13)

        # Optional conditional QED-consistency objective (training only). Built here so it
        # exists before any resume, so its state can be restored. When off, nothing below
        # runs and the loop is byte-identical to the baseline.
        self.surrogate = None
        self.surrogate_optimizer = None
        self.consistency_generator = None
        self._qed_col = None
        if cfg.qed_consistency_weight > 0 and "qed" not in self.flow_config.condition_features:
            raise ValueError("qed_consistency_weight > 0 requires 'qed' in the flow's "
                             f"condition_features (got {list(self.flow_config.condition_features)})")
        self.qed_consistency_on = (cfg.qed_consistency_weight > 0
                                   and "qed" in self.flow_config.condition_features)
        if self.qed_consistency_on:
            from zinc_gnf.qed_consistency import EmbeddingQedSurrogate
            self.surrogate = EmbeddingQedSurrogate(
                self.flow_config.node_embedding_dim,
                hidden_dim=cfg.qed_surrogate_hidden).to(self.device)
            self.surrogate_optimizer = torch.optim.Adam(self.surrogate.parameters(),
                                                        lr=cfg.qed_surrogate_lr)
            self.consistency_generator = torch.Generator(device=self.device).manual_seed(
                cfg.seed * 6271 + 5)
            self._qed_col = self.flow_config.condition_features.index("qed")
            log.info(
                f"QED-consistency objective ON (training only): weight={cfg.qed_consistency_weight} "
                f"warmup={cfg.qed_consistency_warmup} ramp={cfg.qed_consistency_ramp} "
                f"subbatch={cfg.qed_consistency_subbatch} T={cfg.qed_consistency_temperature} "
                f"surrogate_hidden={cfg.qed_surrogate_hidden} lr={cfg.qed_surrogate_lr}; the flow "
                f"density and checkpoint format are UNCHANGED (surrogate saved separately)")

        # N-matched QED-contrastive objective (forward-only). No parameters, no new
        # checkpoint keys; only counters, which are carried in the resume state.
        self.qed_contrast_on = (cfg.qed_contrast_weight > 0
                                and "qed" in self.flow_config.condition_features)
        if cfg.qed_contrast_weight > 0 and "qed" not in self.flow_config.condition_features:
            raise ValueError("qed_contrast_weight > 0 requires 'qed' in the flow's "
                             f"condition_features (got {list(self.flow_config.condition_features)})")
        self.contrast_applied = 0
        self.contrast_skipped = 0
        self._contrast_qed_col = None
        self._contrast_qed_range = (0.0, 0.0)
        self.contrast_generator = None
        if self.qed_contrast_on:
            self._contrast_qed_col = self.flow_config.condition_features.index("qed")
            # Candidate clamp range from the TRAIN split only (no validation/test access).
            column = np.asarray(self.train_set.condition, dtype=np.float64)[:, condition_columns(
                self.flow_config.condition_features)[self._contrast_qed_col]]
            self._contrast_qed_range = (float(np.min(column)), float(np.max(column)))
            self.contrast_generator = torch.Generator().manual_seed(cfg.seed * 2654435761 % 2**31)
            log.info(
                f"QED-CONTRASTIVE objective ON (forward-only, training only): "
                f"weight={cfg.qed_contrast_weight} warmup={cfg.qed_contrast_warmup} "
                f"ramp={cfg.qed_contrast_ramp} offsets={list(cfg.qed_contrast_offsets)} "
                f"subbatch={cfg.qed_contrast_subbatch} every={cfg.qed_contrast_every} "
                f"negatives={cfg.qed_contrast_negatives} "
                f"min_separation={cfg.qed_contrast_min_separation}")
            log.info(
                f"    standardised QED range on TRAIN: "
                f"[{self._contrast_qed_range[0]:.3f}, {self._contrast_qed_range[1]:.3f}]; "
                f"candidates are clamped to it and every candidate keeps the molecule's OWN N, "
                f"so the contrast is within-N. Chance loss = "
                f"{math.log(len(cfg.qed_contrast_offsets)):.4f}; a loss below it means the "
                f"density prefers the molecule's true QED over N-matched alternatives.")
            log.info(
                "    this objective backpropagates through the FORWARD map only - the same "
                "path the NLL uses - so it contains no linalg.inv and no sequential "
                "autoregressive inverse, which is why it is expected to be stable where the "
                "sampling-based objective was not. Non-finite updates are counted, not hidden.")

        step = 0
        history: list[dict[str, Any]] = []

        resume_path = self._resolve_resume(resume)
        if resume_path is not None:
            step, history = self._restore(resume_path, flow, optimizer)
            log.info(f"RESUMED from {resume_path} at step {step} (epoch {self.batcher.epoch}, "
                     f"offset {self.batcher.position}, best step {self.best_step}); "
                     f"embedding statistics restored from the checkpoint")
        else:
            self._fit_statistics(flow)
        if not bool(flow.statistics_set):
            raise RuntimeError("embedding statistics were not set; refusing to train")

        atomic_write_json({"flow": self.flow_config.to_dict(), "training": cfg.to_dict(),
                           "data": self.data_fingerprint, "autoencoder": self.ae_fingerprint,
                           "embedding_statistics": self.statistics_summary,
                           "vocabulary_fingerprint": vocabulary_fingerprint(),
                           "flow_parameters": flow.num_parameters(),
                           "resumed_from_step": step if resume_path else None,
                           "device": str(self.device)}, self.output_dir / "config.json")

        stop_flag = stop_flag or StopFlag()
        metrics_path = self.output_dir / "metrics.jsonl"
        started = time.perf_counter()
        step_started, window = time.perf_counter(), []
        stopped_early = False
        paused = False
        interrupted = None
        # A pause point already behind us must never stop the run again, otherwise
        # resubmitting the same command with PAUSE_AT still set would spin forever.
        pause_at = cfg.pause_at_step
        if pause_at is not None and step >= pause_at:
            log.info(f"pause point {pause_at} already passed (resumed at step {step}); "
                     f"continuing to max_steps={cfg.max_steps}")
            pause_at = None
        elif pause_at is not None:
            log.info(f"will PAUSE at step {pause_at} of {cfg.max_steps} (the run stays "
                     f"unfinished; resubmit to resume)")
        flow.train()

        while step < cfg.max_steps:
            for group in optimizer.param_groups:
                group["lr"] = self._learning_rate(step)
            indices = self.batcher.next_indices()
            batch = make_batch(self.train_set, indices).to(self.device)
            clean = encode_batch(self.autoencoder, batch)
            embeddings = flow.add_smoothing_noise(clean, batch.node_mask,
                                                  generator=self.noise_generator)
            condition = self._condition(flow, batch)
            parts = flow.negative_log_likelihood(embeddings, batch.node_mask, condition)
            nll_loss = parts["loss"]
            loss = nll_loss

            # Optional conditional QED-consistency term. Adds nll + lambda(step) * MSE,
            # where the MSE is on prior-sampled, inverted embeddings read by a frozen,
            # real-calibrated QED surrogate. Logged separately from the NLL below.
            aux = {}
            cons_weight = 0.0
            if self.qed_consistency_on:
                from zinc_gnf.qed_consistency import (conditional_consistency_loss,
                                                      consistency_weight)
                cons_weight = consistency_weight(step, weight=cfg.qed_consistency_weight,
                                                 warmup=cfg.qed_consistency_warmup,
                                                 ramp=cfg.qed_consistency_ramp)
                if cons_weight > 0.0 and step % cfg.qed_consistency_every == 0:
                    values, _present = batch_condition(batch, self.flow_config.condition_features)
                    sub = min(cfg.qed_consistency_subbatch, batch.batch_size)
                    # Freeze the surrogate IN-GRAPH so this term updates the flow only.
                    for parameter in self.surrogate.parameters():
                        parameter.requires_grad_(False)
                    consistency, aux = conditional_consistency_loss(
                        flow, self.surrogate, condition=condition[:sub],
                        mask=batch.node_mask[:sub], qed_std_target=values[:sub, self._qed_col],
                        temperature=cfg.qed_consistency_temperature,
                        generator=self.consistency_generator)
                    for parameter in self.surrogate.parameters():
                        parameter.requires_grad_(True)
                    loss = nll_loss + cons_weight * consistency

            # --- N-matched QED-contrastive term (forward-only) -----------------------
            contrast_diag: dict[str, Any] = {}
            contrast_weight_now = 0.0
            if self.qed_contrast_on:
                from zinc_gnf.qed_contrastive import contrastive_weight, qed_contrastive_loss
                contrast_weight_now = contrastive_weight(
                    step, weight=cfg.qed_contrast_weight, warmup=cfg.qed_contrast_warmup,
                    ramp=cfg.qed_contrast_ramp)
                if contrast_weight_now > 0.0 and step % cfg.qed_contrast_every == 0:
                    values_all, _ = batch_condition(batch, self.flow_config.condition_features)
                    sub = min(cfg.qed_contrast_subbatch, batch.batch_size)
                    contrast, contrast_diag = qed_contrastive_loss(
                        flow, embeddings[:sub], batch.node_mask[:sub], values_all[:sub],
                        qed_column=self._contrast_qed_col,
                        offsets=cfg.qed_contrast_offsets,
                        qed_min=self._contrast_qed_range[0],
                        qed_max=self._contrast_qed_range[1],
                        min_separation=cfg.qed_contrast_min_separation,
                        logit_scale=cfg.qed_contrast_logit_scale,
                        negatives=cfg.qed_contrast_negatives,
                        generator=self.contrast_generator)
                    self.contrast_applied += 1
                    if torch.isfinite(contrast):
                        loss = loss + contrast_weight_now * contrast
                    else:
                        # Counted and surfaced, never silently absorbed.
                        self.contrast_skipped += 1
                        contrast_diag["qed_contrast_nonfinite"] = True
                        log.warn(f"QED-contrastive loss was non-finite at step {step + 1} "
                                 f"(skipped {self.contrast_skipped}/{self.contrast_applied} "
                                 f"applications)")
                    rate = self.contrast_skipped / max(1, self.contrast_applied)
                    if self.contrast_applied >= cfg.qed_contrast_skip_grace \
                            and rate > cfg.qed_contrast_max_skip_rate:
                        self._save("last.pt", flow, optimizer,
                                   {**self._state(step), "history": history})
                        raise FloatingPointError(
                            f"the QED-contrastive objective was non-finite in {rate:.1%} of "
                            f"{self.contrast_applied} applications, above the permitted "
                            f"{cfg.qed_contrast_max_skip_rate:.1%}. Training is STOPPING rather "
                            f"than continuing with the auxiliary signal effectively absent "
                            f"(the failure mode of the previous sampling-based attempt). State "
                            f"saved to {self.output_dir / 'last.pt'}. Investigate before "
                            f"resuming; do not simply lower the weight.")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = None
            if cfg.grad_clip_norm:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(flow.parameters(), cfg.grad_clip_norm))
            if not torch.isfinite(loss):
                self._save("last.pt", flow, optimizer, {**self._state(step), "history": history})
                raise FloatingPointError(
                    f"non-finite loss at step {step + 1}: {float(loss.detach())}. The last finite "
                    f"state was saved to {self.output_dir / 'last.pt'}; lower max_log_scale or the "
                    f"learning rate before resuming.")
            optimizer.step()

            # Calibrate the QED surrogate on REAL (clean) embeddings only, with its own
            # optimiser and a separate graph, so the consistency term never moves it and
            # no gradient reaches the frozen encoder. Detach keeps the flow out of it too.
            surrogate_mse = None
            if self.qed_consistency_on:
                from zinc_gnf.qed_consistency import surrogate_regression_loss
                values_all, _ = batch_condition(batch, self.flow_config.condition_features)
                self.surrogate_optimizer.zero_grad(set_to_none=True)
                regression = surrogate_regression_loss(
                    self.surrogate, clean.detach(), batch.node_mask,
                    values_all[:, self._qed_col])
                regression.backward()
                self.surrogate_optimizer.step()
                surrogate_mse = float(regression.detach())

            step += 1

            entry = {"loss": float(nll_loss.detach()),
                     "log_det_per_node": float(parts["log_det_per_node"].detach()),
                     "log_prob_z_per_node": float(parts["log_prob_z_per_node"].detach()),
                     "z_norm": float(parts["z_norm"].detach()),
                     "prior_sigma_mean": float(parts["prior_sigma_mean"].detach())}
            if self.qed_consistency_on:
                entry["qed_cons_weight"] = float(cons_weight)
                entry["qed_consistency_mse"] = float(aux.get("qed_consistency_mse", 0.0))
                entry["qed_surrogate_mse"] = float(surrogate_mse if surrogate_mse is not None else 0.0)
            if self.qed_contrast_on:
                entry["qed_contrast_weight"] = float(contrast_weight_now)
                for key in ("qed_contrast_loss", "qed_contrast_accuracy",
                            "qed_contrast_true_probability", "prior_mu_gap",
                            "qed_contrast_usable_fraction", "qed_contrast_logit_spread"):
                    entry[key] = float(contrast_diag.get(key, 0.0))
            window.append(entry)
            if step % cfg.log_every == 0 or step == cfg.max_steps or step <= 5:
                mean = {k: float(np.mean([w[k] for w in window])) for k in window[0]}
                rate = len(window) / max(time.perf_counter() - step_started, 1e-9)
                log.info(f"step {step}/{cfg.max_steps} epoch={self.batcher.epoch} "
                         f"nll/node={mean['loss']:.4f} logdet/node={mean['log_det_per_node']:.4f} "
                         f"logp_z/node={mean['log_prob_z_per_node']:.4f} z_norm={mean['z_norm']:.3f} "
                         f"prior_sigma={mean['prior_sigma_mean']:.3f} lr={self._learning_rate(step):.2e} "
                         f"grad_norm={grad_norm if grad_norm is None else round(grad_norm, 3)} "
                         f"it/s={rate:.2f} elapsed={format_duration(time.perf_counter() - started)} "
                         f"rss_mb={current_rss_mb()}")
                if self.qed_consistency_on:
                    log.info(f"    QED-consistency: lambda={mean.get('qed_cons_weight', 0.0):.4f} "
                             f"cons_mse={mean.get('qed_consistency_mse', 0.0):.4f} "
                             f"surrogate_mse={mean.get('qed_surrogate_mse', 0.0):.4f} "
                             f"(aux term; NLL above is the density objective)")
                if self.qed_contrast_on:
                    chance = math.log(len(cfg.qed_contrast_offsets))
                    log.info(f"    QED-contrastive: lambda={mean.get('qed_contrast_weight', 0.0):.4f} "
                             f"loss={mean.get('qed_contrast_loss', 0.0):.4f} (chance {chance:.4f}) "
                             f"acc={mean.get('qed_contrast_accuracy', 0.0):.3f} "
                             f"p_true={mean.get('qed_contrast_true_probability', 0.0):.3f} "
                             f"prior_mu_gap={mean.get('prior_mu_gap', 0.0):.4f} "
                             f"usable={mean.get('qed_contrast_usable_fraction', 0.0):.3f} "
                             f"skipped={self.contrast_skipped}/{self.contrast_applied}")
                self._append(metrics_path, {"kind": "train", "step": step, **mean, "it_per_s": rate})
                # A log-determinant running away is the documented divergence signature of
                # this flow family (the teammates saw it before a NaN). Warn loudly while the
                # run is still finite, so the log says when it started.
                if abs(mean["log_det_per_node"]) > cfg.log_det_warn_threshold:
                    log.warn(f"log|det J|/node = {mean['log_det_per_node']:.1f} exceeds "
                             f"{cfg.log_det_warn_threshold} at step {step}: the flow is drifting "
                             f"into an expansive/contractive regime where float32 inversion "
                             f"degrades. Watch z_norm and the validation loss; if it keeps "
                             f"growing, lower max_log_scale or the learning rate.")
                window, step_started = [], time.perf_counter()

            reached_pause = pause_at is not None and step >= pause_at
            # Always validate and checkpoint at the pause step, so the paused last.pt /
            # best.pt are current and can be generated from immediately.
            do_eval = step % cfg.eval_every == 0 or step == cfg.max_steps or reached_pause
            if do_eval:
                metrics = self.validate(flow)
                improved = self.best_loss is None or metrics["val_loss"] < self.best_loss
                if improved:
                    self.best_loss, self.best_step, self.bad_evals = metrics["val_loss"], step, 0
                else:
                    self.bad_evals += 1
                condition_note = ""
                if "val_condition_gap" in metrics:
                    condition_note = (f"cond_gap={metrics['val_condition_gap']:+.4f} "
                                      f"cond_win={metrics['val_condition_win_rate']:.3f} ")
                surrogate_note = ""
                if "val_surrogate_qed_mae_standardised" in metrics:
                    surrogate_note = (
                        f"surr_qed_mae(std)={metrics['val_surrogate_qed_mae_standardised']:.4f} "
                        f"surr_qed_corr={metrics['val_surrogate_qed_corr']:.3f} ")
                log.info(f"VAL step {step}: nll/node={metrics['val_loss']:.4f} "
                         f"logp_z/node={metrics['val_log_prob_z_per_node']:.4f} "
                         f"logdet/node={metrics['val_log_det_per_node']:.4f} "
                         f"z_norm={metrics['val_z_norm']:.3f} "
                         f"{condition_note}{surrogate_note}"
                         f"molecules={metrics['val_molecules']:,} "
                         f"{'*BEST*' if improved else f'(no improvement x{self.bad_evals})'}")
                self._append(metrics_path, {"kind": "val", "step": step, "improved": improved, **metrics})
                history.append({"step": step, **metrics, "improved": improved})
                if improved:
                    self._save("best.pt", flow, optimizer, {**self._state(step), "val_metrics": metrics})
                patience = cfg.early_stopping_patience
                if patience is not None and self.bad_evals >= patience:
                    log.warn(f"early stopping: {self.bad_evals} evaluations without improvement")
                    stopped_early = True

            if step % cfg.save_every == 0 or do_eval:
                self._save("last.pt", flow, optimizer, {**self._state(step), "history": history})
            if reached_pause:
                paused = True
                log.info(f"PAUSED at step {step} of {cfg.max_steps} as requested. The run is "
                         f"NOT complete: optimizer, batcher, RNG and noise-stream state are in "
                         f"{self.output_dir / 'last.pt'}. Sample from it for a mid-run check, "
                         f"then resubmit the same job to resume at step {step + 1}.")
                break
            if stopped_early:
                break
            if stop_flag.requested:
                interrupted = stop_flag.signal_name
                log.warn(f"received {interrupted}: saving last.pt at step {step} and stopping "
                         f"(resubmit the same job to resume)")
                break

        self._save("last.pt", flow, optimizer, {**self._state(step), "history": history})
        result = {
            "completed": step >= cfg.max_steps and interrupted is None,
            "interrupted_by": interrupted, "early_stopped": stopped_early,
            # A pause is neither completion nor early stopping: the cluster script's
            # finished-seed check must still treat this seed as unfinished.
            "paused": paused, "paused_at_step": step if paused else None,
            "max_steps": cfg.max_steps,
            "final_step": step, "best_step": self.best_step, "best_val_loss": self.best_loss,
            "last_validation": history[-1] if history else None,
            "seconds": time.perf_counter() - started,
            "device": str(self.device),
            "flow_parameters": flow.num_parameters(),
            "data": self.data_fingerprint,
            "autoencoder": self.ae_fingerprint,
            "embedding_statistics": self.statistics_summary,
        }
        atomic_write_json(result, self.output_dir / "summary.json")
        log.info(f"finished step {step}; best step {self.best_step} "
                 f"(val nll/node {self.best_loss}); completed={result['completed']} "
                 f"paused={result['paused']}")
        return result

    # ------------------------------------------------------------------
    def _resolve_resume(self, resume: str | Path | None) -> Path | None:
        if resume in (None, "", "none"):
            return None
        if resume == "auto":
            candidate = self.output_dir / "last.pt"
            return candidate if candidate.exists() else None
        path = Path(resume)
        if not path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {path}")
        return path

    def _restore(self, path: Path, flow: NodeFlow, optimizer) -> tuple[int, list[dict[str, Any]]]:
        # Metadata is checked BEFORE any weights are loaded, so a mismatch is a clear
        # ResumeMismatchError rather than a state-dict shape error or a silent wrong resume.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        problems = []
        if checkpoint.get("vocabulary_fingerprint") != vocabulary_fingerprint():
            problems.append("atom vocabulary differs")
        try:
            saved_flow = FlowConfig.from_dict(checkpoint.get("flow_config", {})).to_dict()
        except (TypeError, ValueError):
            saved_flow = None
        if saved_flow != self.flow_config.to_dict():
            problems.append("flow configuration differs")
        saved_ae = checkpoint.get("autoencoder", {})
        if saved_ae.get("sha256") != self.ae_fingerprint.get("sha256"):
            problems.append(f"autoencoder checkpoint differs (checkpoint {str(saved_ae.get('sha256'))[:16]} "
                            f"vs now {str(self.ae_fingerprint.get('sha256'))[:16]})")
        saved_data = checkpoint.get("data_fingerprint", {})
        if saved_data.get("counts_digest") != self.data_fingerprint["counts_digest"] \
                or saved_data.get("used") != self.data_fingerprint["used"]:
            problems.append("training data differs")
        saved_train = checkpoint.get("train_config", {})
        for name in ("seed", "batch_size", "learning_rate"):
            if saved_train.get(name) != getattr(self.config, name):
                problems.append(f"{name}: checkpoint {saved_train.get(name)!r} vs now "
                                f"{getattr(self.config, name)!r}")
        if problems:
            raise ResumeMismatchError(f"cannot resume from {path}: " + "; ".join(problems))
        checkpoint = load_checkpoint(path, flow, optimizer, device=self.device)
        self.batcher.load_state_dict(checkpoint["batcher"])
        self.best_loss = checkpoint.get("best_loss")
        self.best_step = int(checkpoint.get("best_step", -1))
        self.bad_evals = int(checkpoint.get("bad_evals", 0))
        self.contrast_applied = int(checkpoint.get("contrast_applied", 0))
        self.contrast_skipped = int(checkpoint.get("contrast_skipped", 0))
        self.statistics_summary = dict(checkpoint.get("embedding_statistics") or {})
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if self.device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["cuda_rng_state"]])
        if checkpoint.get("numpy_rng_state") is not None:
            np.random.set_state(checkpoint["numpy_rng_state"])
        saved_noise_state = checkpoint.get("noise_generator_state")
        if saved_noise_state is not None and getattr(self, "noise_generator", None) is not None:
            self.noise_generator.set_state(saved_noise_state.to(torch.uint8).cpu())
        # Restore the training-only QED-consistency state when both the run and the
        # checkpoint have it; a baseline checkpoint simply carries none of these keys.
        if getattr(self, "surrogate", None) is not None \
                and checkpoint.get("qed_surrogate_state") is not None:
            self.surrogate.load_state_dict(checkpoint["qed_surrogate_state"])
            if checkpoint.get("qed_surrogate_optimizer_state") is not None:
                self.surrogate_optimizer.load_state_dict(
                    checkpoint["qed_surrogate_optimizer_state"])
            saved_consistency_state = checkpoint.get("consistency_generator_state")
            if saved_consistency_state is not None \
                    and getattr(self, "consistency_generator", None) is not None:
                self.consistency_generator.set_state(
                    saved_consistency_state.to(torch.uint8).cpu())
        return int(checkpoint["step"]), list(checkpoint.get("history", []))

    @staticmethod
    def _append(path: Path, record: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=float) + "\n")


def load_flow(checkpoint_path: str | Path, device: str | torch.device = "cpu") -> tuple[NodeFlow, dict[str, Any]]:
    """Rebuild a trained flow (with its embedding statistics) for generation."""
    probe = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if probe.get("vocabulary_fingerprint") != vocabulary_fingerprint():
        raise ResumeMismatchError("flow checkpoint was trained with a different atom vocabulary")
    flow = NodeFlow(FlowConfig.from_dict(probe["flow_config"]))
    load_checkpoint(checkpoint_path, flow, None, device=device)
    flow.to(device).eval()
    if not bool(flow.statistics_set):
        raise ResumeMismatchError("flow checkpoint carries no embedding statistics; refusing to sample")
    return flow, probe


__all__ = [
    "FlowTrainConfig",
    "FlowTrainer",
    "STATISTICS_SEED",
    "VALIDATION_NOISE_SEED",
    "VALIDATION_SUBSET_SEED",
    "autoencoder_seen_indices",
    "embedding_statistics_from_dataset",
    "encode_batch",
    "file_sha256",
    "load_flow",
]
