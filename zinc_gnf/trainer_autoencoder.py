"""Autoencoder training loop: validation-based model selection and exact resume.

Protocol (project decisions):
  * train on the TRAIN split only; the validation split is used for
    checkpoint selection and hyperparameter tuning; the TEST split is never
    loaded here (`load_training_splits` drops it); no train+val training;
  * `best.pt` = the checkpoint with the best validation selection key,
    `last.pt` = the most recent state, from which training resumes exactly.

Resume is exact: model, optimizer, batcher position (epoch, offset), RNG
states, best-so-far bookkeeping and history are all restored, and a test
checks that 3+3 steps equal 6 straight steps bit for bit on CPU.

Selection key (default `exact_molecule_constrained`): validation exact
molecule-match rate under constrained decoding (maximise), ties broken by
lower validation loss. `val_loss` selects on loss alone.

Everything written goes under `output_dir`:
    config.json, metrics.jsonl, train.log, last.pt, best.pt, summary.json
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import signal
import time
from typing import Any

import numpy as np
import torch

from zinc_gnf.autoencoder import (
    AutoencoderConfig,
    GraphAutoencoder,
    evaluate_reconstruction,
)
from zinc_gnf.constants_charge_aware import vocabulary_fingerprint
from zinc_gnf.dataset import (
    DeterministicBatcher,
    MolecularGraphDataset,
    load_training_splits,
    make_batch,
    seeded_subset_indices,
)
from zinc_gnf.utils.checkpointing import load_checkpoint, save_checkpoint
from zinc_gnf.utils.logging import JobLogger
from zinc_gnf.utils.runtime import StopFlag, atomic_write_json, current_rss_mb, format_duration
from zinc_gnf.utils.seed import set_seed

SELECTION_KEYS = ("exact_molecule_constrained", "val_loss")
VALIDATION_SUBSET_SEED = 12345       # fixed so runs with different training seeds are comparable


class ResumeMismatchError(RuntimeError):
    """The checkpoint does not belong to this data / model / configuration."""


@dataclass
class TrainConfig:
    data_dir: str = "data/processed/zinc250k_charge_aware_moflow_fixed"
    output_dir: str = "outputs/checkpoints/autoencoder"
    seed: int = 42
    device: str = "auto"                  # auto | cpu | cuda
    batch_size: int = 32                  # teammates' value
    max_steps: int = 50_000               # teammates' value (~7.9 epochs of 202k at batch 32)
    learning_rate: float = 1e-3           # teammates' Adam default
    weight_decay: float = 0.0
    grad_clip_norm: float | None = 5.0
    log_every: int = 100
    eval_every: int = 500
    save_every: int = 2000
    val_loss_molecules: int | None = 4000     # fixed seeded validation subset (loss/accuracy)
    val_recon_molecules: int | None = 1000    # subset used for the slower decode metrics
    selection_key: str = "exact_molecule_constrained"
    early_stopping_patience: int | None = None    # in evaluations; None = off
    max_train_molecules: int | None = None        # smoke tests only
    max_val_molecules: int | None = None          # smoke tests only
    deterministic: bool = False

    def validate(self) -> None:
        if self.selection_key not in SELECTION_KEYS:
            raise ValueError(f"selection_key must be one of {SELECTION_KEYS}")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
        for name in ("batch_size", "max_steps", "eval_every", "save_every", "log_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainConfig":
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown training settings: {sorted(unknown)}")
        return cls(**values)


def resolve_device(requested: str) -> torch.device:
    """`cuda` fails loudly when unavailable (a GPU job must never silently train on CPU)."""
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device 'cuda' requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def selection_key_value(metrics: dict[str, Any], mode: str) -> tuple[float, float]:
    """Larger tuple is better. Primary metric first, lower loss as tie-break."""
    loss = float(metrics["val_mean_loss"])
    if mode == "val_loss":
        return (-loss, -loss)
    exact = metrics.get("val_exact_molecule_constrained")
    exact = -1.0 if exact is None or math.isnan(exact) else float(exact)
    return (exact, -loss)


def validation_metrics(
    model: GraphAutoencoder, val: MolecularGraphDataset, config: TrainConfig,
    device: torch.device, loss_indices: list[int], recon_indices: list[int],
) -> dict[str, Any]:
    """Loss/accuracy on `loss_indices`; decode metrics on `recon_indices`."""
    basic = evaluate_reconstruction(model, val, indices=loss_indices, device=device,
                                    compute_ceiling=False, decode_molecules=False)
    decoded = evaluate_reconstruction(model, val, indices=recon_indices, device=device,
                                      compute_ceiling=True, decode_molecules=True)
    return {
        "val_mean_loss": basic["mean_loss"],
        "val_atom_accuracy": basic["atom_accuracy"],
        "val_bond_accuracy": basic["bond_accuracy"],
        "val_real_bond_accuracy": basic["real_bond_accuracy"],
        "val_loss_molecules": basic["molecules"],
        "val_recon_molecules": decoded["molecules"],
        "val_exact_molecule_constrained": decoded["constrained"]["exact_molecule_match"],
        "val_exact_graph_constrained": decoded["constrained"]["exact_graph_match"],
        "val_validity_constrained": decoded["constrained"]["validity"],
        "val_exact_molecule_raw": decoded["raw"]["exact_molecule_match"],
        "val_validity_raw": decoded["raw"]["validity"],
        "val_equivariant_ceiling": decoded.get("equivariant_exact_ceiling"),
        "val_atom_sequence_exact": decoded["atom_sequence_exact"],
        "decoded": decoded,
    }


class AutoencoderTrainer:
    def __init__(self, model_config: AutoencoderConfig, train_config: TrainConfig,
                 logger: JobLogger | None = None) -> None:
        train_config.validate()
        self.model_config, self.config = model_config, train_config
        self.output_dir = Path(train_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or JobLogger("autoencoder", self.output_dir, filename="train.log")
        self.device = resolve_device(train_config.device)

    # ------------------------------------------------------------------
    def _load_data(self) -> None:
        train_records, val_records, fingerprint = load_training_splits(self.config.data_dir)
        if self.config.max_train_molecules:
            train_records = train_records[: self.config.max_train_molecules]
        if self.config.max_val_molecules:
            val_records = val_records[: self.config.max_val_molecules]
        self.train_set = MolecularGraphDataset(train_records)
        self.val_set = MolecularGraphDataset(val_records)
        del train_records, val_records
        fingerprint = dict(fingerprint)
        fingerprint["used"] = {"train": len(self.train_set), "val": len(self.val_set)}
        self.data_fingerprint = fingerprint
        n_val = len(self.val_set)
        self.loss_indices = seeded_subset_indices(n_val, self.config.val_loss_molecules, VALIDATION_SUBSET_SEED)
        recon_pool = self.loss_indices
        count = self.config.val_recon_molecules
        self.recon_indices = recon_pool if count is None or count >= len(recon_pool) else recon_pool[:count]

    def _save(self, name: str, model, optimizer, extra_state: dict[str, Any]) -> Path:
        path = self.output_dir / name
        save_checkpoint(path, model, optimizer, epoch=self.batcher.epoch,
                        best_metric=self.best_key[0] if self.best_key else float("-inf"),
                        configuration={"autoencoder": self.model_config.to_dict(),
                                       "training": self.config.to_dict()},
                        extra=extra_state)
        return path

    def _state(self, step: int) -> dict[str, Any]:
        return {
            "step": step,
            "batcher": self.batcher.state_dict(),
            "best_key": list(self.best_key) if self.best_key else None,
            "best_step": self.best_step,
            "bad_evals": self.bad_evals,
            "data_fingerprint": self.data_fingerprint,
            "vocabulary_fingerprint": vocabulary_fingerprint(),
            "model_config": self.model_config.to_dict(),
            "train_config": self.config.to_dict(),
            "numpy_rng_state": np.random.get_state(),
        }

    # ------------------------------------------------------------------
    def train(self, resume: str | Path | None = "auto", stop_flag: StopFlag | None = None) -> dict[str, Any]:
        cfg, log = self.config, self.logger
        set_seed(cfg.seed, deterministic=cfg.deterministic)
        log.header("Charge-aware graph autoencoder training")
        log.info(f"device={self.device} seed={cfg.seed} batch_size={cfg.batch_size} "
                 f"max_steps={cfg.max_steps} lr={cfg.learning_rate} selection={cfg.selection_key}")
        self._load_data()
        log.info(f"data {cfg.data_dir}: train={len(self.train_set):,} val={len(self.val_set):,} "
                 f"(test split NOT loaded) split_protocol={self.data_fingerprint['split_protocol']}")
        log.info(f"validation subsets: loss={len(self.loss_indices):,} decode={len(self.recon_indices):,} "
                 f"(fixed seed {VALIDATION_SUBSET_SEED})")

        model = GraphAutoencoder(self.model_config).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                                     weight_decay=cfg.weight_decay)
        self.batcher = DeterministicBatcher(len(self.train_set), cfg.batch_size, cfg.seed)
        self.best_key: tuple[float, float] | None = None
        self.best_step = -1
        self.bad_evals = 0
        step = 0
        history: list[dict[str, Any]] = []
        log.info(f"model parameters: {model.num_parameters():,}")

        resume_path = self._resolve_resume(resume)
        if resume_path is not None:
            step, history = self._restore(resume_path, model, optimizer)
            log.info(f"RESUMED from {resume_path} at step {step} "
                     f"(epoch {self.batcher.epoch}, offset {self.batcher.position}, best step {self.best_step})")
        atomic_write_json({"autoencoder": self.model_config.to_dict(), "training": cfg.to_dict(),
                           "data": self.data_fingerprint,
                           "vocabulary_fingerprint": vocabulary_fingerprint(),
                           "resumed_from_step": step if resume_path else None,
                           "device": str(self.device)}, self.output_dir / "config.json")

        stop_flag = stop_flag or StopFlag()
        metrics_path = self.output_dir / "metrics.jsonl"
        started = time.perf_counter()
        step_started, window = time.perf_counter(), []
        stopped_early = False
        interrupted = None
        model.train()

        while step < cfg.max_steps:
            indices = self.batcher.next_indices()
            batch = make_batch(self.train_set, indices).to(self.device)
            outputs = model(batch, teacher_forcing=True)
            losses = model.reconstruction_loss(batch, outputs)
            optimizer.zero_grad(set_to_none=True)
            losses["mean_loss"].backward()
            grad_norm = None
            if cfg.grad_clip_norm:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm))
            if not torch.isfinite(losses["mean_loss"]):
                raise FloatingPointError(f"non-finite loss at step {step + 1}: {float(losses['mean_loss'].detach())}")
            optimizer.step()
            step += 1

            window.append({k: float(losses[k].detach()) for k in
                           ("mean_loss", "atom_accuracy", "bond_accuracy", "real_bond_accuracy")})
            if step % cfg.log_every == 0 or step == cfg.max_steps:
                mean = {k: float(np.mean([w[k] for w in window])) for k in window[0]}
                rate = len(window) / max(time.perf_counter() - step_started, 1e-9)
                log.info(f"step {step}/{cfg.max_steps} epoch={self.batcher.epoch} "
                         f"loss={mean['mean_loss']:.4f} atom_acc={mean['atom_accuracy']:.4f} "
                         f"bond_acc={mean['bond_accuracy']:.4f} real_bond_acc={mean['real_bond_accuracy']:.4f} "
                         f"grad_norm={grad_norm if grad_norm is None else round(grad_norm, 3)} "
                         f"it/s={rate:.2f} elapsed={format_duration(time.perf_counter() - started)} "
                         f"rss_mb={current_rss_mb()}")
                self._append(metrics_path, {"kind": "train", "step": step, **mean, "it_per_s": rate})
                window, step_started = [], time.perf_counter()

            do_eval = step % cfg.eval_every == 0 or step == cfg.max_steps
            if do_eval:
                model.eval()
                metrics = validation_metrics(model, self.val_set, cfg, self.device,
                                             self.loss_indices, self.recon_indices)
                model.train()
                key = selection_key_value(metrics, cfg.selection_key)
                improved = self.best_key is None or key > self.best_key
                if improved:
                    self.best_key, self.best_step, self.bad_evals = key, step, 0
                else:
                    self.bad_evals += 1
                summary = {k: v for k, v in metrics.items() if k != "decoded"}
                log.info(f"VAL step {step}: loss={metrics['val_mean_loss']:.4f} "
                         f"atom_acc={metrics['val_atom_accuracy']:.4f} bond_acc={metrics['val_bond_accuracy']:.4f} "
                         f"real_bond_acc={metrics['val_real_bond_accuracy']:.4f} | "
                         f"constrained: valid={metrics['val_validity_constrained']:.3f} "
                         f"exact_mol={metrics['val_exact_molecule_constrained']:.3f} | "
                         f"raw: valid={metrics['val_validity_raw']:.3f} exact_mol={metrics['val_exact_molecule_raw']:.3f} | "
                         f"ceiling={metrics['val_equivariant_ceiling']:.3f} "
                         f"{'*BEST*' if improved else ''}")
                self._append(metrics_path, {"kind": "val", "step": step, "improved": improved, **summary})
                history.append({"step": step, **summary, "improved": improved})
                if improved:
                    self._save("best.pt", model, optimizer, {**self._state(step), "val_metrics": summary})
                patience = cfg.early_stopping_patience
                if patience is not None and self.bad_evals >= patience:
                    log.warn(f"early stopping: {self.bad_evals} evaluations without improvement")
                    stopped_early = True

            if step % cfg.save_every == 0 or do_eval:
                self._save("last.pt", model, optimizer, {**self._state(step), "history": history})
            if stopped_early:
                break
            if stop_flag.requested:
                interrupted = stop_flag.signal_name
                log.warn(f"received {interrupted}: saving last.pt at step {step} and stopping (resume with --resume auto)")
                break

        self._save("last.pt", model, optimizer, {**self._state(step), "history": history})
        result = {
            "completed": step >= cfg.max_steps and interrupted is None,
            "interrupted_by": interrupted, "early_stopped": stopped_early,
            "final_step": step, "best_step": self.best_step,
            "best_selection_key": list(self.best_key) if self.best_key else None,
            "selection_key": cfg.selection_key,
            "last_validation": history[-1] if history else None,
            "best_validation": next((h for h in history if h["step"] == self.best_step), None),
            "seconds": time.perf_counter() - started,
            "device": str(self.device),
            "data": self.data_fingerprint,
        }
        atomic_write_json(result, self.output_dir / "summary.json")
        log.info(f"finished step {step}; best step {self.best_step}; completed={result['completed']}")
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

    def _restore(self, path: Path, model, optimizer) -> tuple[int, list[dict[str, Any]]]:
        # Validate the checkpoint's metadata BEFORE loading any weights, so a
        # mismatch is reported as a ResumeMismatchError and never as a raw
        # state-dict size error (or, worse, a silently wrong resume).
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        problems = []
        if checkpoint.get("vocabulary_fingerprint") != vocabulary_fingerprint():
            problems.append("atom vocabulary differs")
        if checkpoint.get("model_config") != self.model_config.to_dict():
            problems.append("autoencoder configuration differs")
        saved_data = checkpoint.get("data_fingerprint", {})
        if saved_data.get("counts_digest") != self.data_fingerprint["counts_digest"] \
                or saved_data.get("used") != self.data_fingerprint["used"]:
            problems.append("training data differs")
        saved_train = checkpoint.get("train_config", {})
        for name in ("seed", "batch_size", "learning_rate", "selection_key"):
            if saved_train.get(name) != getattr(self.config, name):
                problems.append(f"{name}: checkpoint {saved_train.get(name)!r} vs now {getattr(self.config, name)!r}")
        if problems:
            raise ResumeMismatchError(f"cannot resume from {path}: " + "; ".join(problems))
        checkpoint = load_checkpoint(path, model, optimizer, device=self.device)
        self.batcher.load_state_dict(checkpoint["batcher"])
        best = checkpoint.get("best_key")
        self.best_key = tuple(best) if best else None
        self.best_step = int(checkpoint.get("best_step", -1))
        self.bad_evals = int(checkpoint.get("bad_evals", 0))
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if self.device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["cuda_rng_state"]])
        if checkpoint.get("numpy_rng_state") is not None:
            np.random.set_state(checkpoint["numpy_rng_state"])
        return int(checkpoint["step"]), list(checkpoint.get("history", []))

    @staticmethod
    def _append(path: Path, record: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=float) + "\n")


def load_autoencoder(checkpoint_path: str | Path, device: str | torch.device = "cpu") -> tuple[GraphAutoencoder, dict[str, Any]]:
    """Rebuild a model from a checkpoint (e.g. `best.pt`) for evaluation or flow training."""
    probe = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if probe.get("vocabulary_fingerprint") != vocabulary_fingerprint():
        raise ResumeMismatchError("checkpoint was trained with a different atom vocabulary")
    model = GraphAutoencoder(AutoencoderConfig.from_dict(probe["model_config"]))
    load_checkpoint(checkpoint_path, model, None, device=device)
    model.to(device).eval()
    return model, probe


__all__ = [
    "AutoencoderTrainer",
    "ResumeMismatchError",
    "SELECTION_KEYS",
    "TrainConfig",
    "VALIDATION_SUBSET_SEED",
    "load_autoencoder",
    "resolve_device",
    "selection_key_value",
    "validation_metrics",
]
