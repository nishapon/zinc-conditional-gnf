"""Stage 22: bounded, resumable training of the direct conditional graph flow.

Two things make this trainer different from `trainer_flow`, and both exist because of
what stages 13-21 got wrong.

THE CONDITIONAL LIKELIHOOD GAP IS A FIRST-CLASS METRIC.
Every evaluation scores the same validation molecules twice: once under their true
`(N, QED)` and once under QED PERMUTED WITHIN N. The gap

    gap = mean[ log p(G | N, QED_true) - log p(G | N, QED_permuted) ]

is zero exactly when the density ignores QED. Permuting within N matters: N and QED are
correlated in ZINC250K, so a global permutation would let the model score the real
condition higher just by recognising N. The win rate - the fraction of molecules scored
higher under their true QED - is reported beside it because a mean gap can be carried by
a few outliers.

This is pre-registered as the undertraining discriminator: if the gap is still RISING at
the end of the budget the run is undertrained and nothing may be concluded; if it has
plateaued near zero, the architecture is not using the condition and a longer run will
not change that.

THE SHUFFLED-CONDITION ARM IS BUILT IN.
`shuffle_qed` permutes QED labels within N buckets in the TRAINING data, destroying the
association while leaving every marginal identical. That arm is the null for the sampled
QED response. Stages 13-21 each produced plausible molecules and no QED control; the only
defence is to make "is the condition being used at all?" a measured quantity that has its
own null to be compared against.

Training uses the TRAIN split only. Validation is used for model selection and
diagnostics. `load_training_splits` cannot return the test split, so it is never loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import torch

from zinc_gnf.dataset import DeterministicBatcher, MolecularGraphDataset, make_batch
from zinc_gnf.graph_flow import (
    ConditionalGraphFlow,
    GraphFlowConfig,
    dequantize,
    pair_mask_from_counts,
)

VALIDATION_SUBSET_SEED = 12345
PERMUTATION_SEED = 20261024


@dataclass
class GraphFlowTrainConfig:
    max_steps: int = 6000
    batch_size: int = 32
    learning_rate: float = 1e-3
    warmup_steps: int = 200
    grad_clip: float = 5.0
    weight_decay: float = 0.0
    eval_every: int = 500
    save_every: int = 500
    log_every: int = 100
    val_molecules: int = 1024
    seed: int = 17
    shuffle_qed: bool = False
    shuffle_seed: int = 99

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def shuffle_qed_within_n(records: Sequence[dict[str, Any]], *, seed: int,
                         scaler) -> list[dict[str, Any]]:
    """Permute QED within each N bucket: identical marginals, no association.

    This is the null arm. Permuting WITHIN N - not globally - is what makes it a null for
    QED specifically: N and QED are correlated in ZINC250K, so a global shuffle would
    also destroy the N-QED relationship and the control would be testing the wrong thing.
    """
    from zinc_gnf.conditioning import standardise_condition

    rows = [dict(record) for record in records]
    buckets: dict[int, list[int]] = {}
    for index, record in enumerate(rows):
        buckets.setdefault(int(record["n_node"]), []).append(index)
    generator = np.random.default_rng(int(seed))
    moved = 0
    for indices in buckets.values():
        if len(indices) < 2:
            continue
        values = [rows[i]["qed"] for i in indices]
        order = generator.permutation(len(values))
        for position, index in enumerate(indices):
            replacement = float(values[int(order[position])])
            moved += int(replacement != rows[index]["qed"])
            rows[index]["qed"] = replacement
    # the stored condition vector must agree with the (now permuted) label
    for record in rows:
        record["condition"] = standardise_condition(
            ("n", "qed"), scaler=scaler, n=[int(record["n_node"])],
            qed=[float(record["qed"])])[0]
    return rows


class GraphFlowTrainer:
    """Maximum-likelihood training of p(G | N, QED), checkpointed and resumable."""

    def __init__(self, *, model: ConditionalGraphFlow, config: GraphFlowTrainConfig,
                 train_set: MolecularGraphDataset, val_set: MolecularGraphDataset,
                 output_dir: Path, logger, device: torch.device | str = "cpu",
                 provenance: dict[str, Any] | None = None) -> None:
        self.model = model
        self.config = config
        self.train_set = train_set
        self.val_set = val_set
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.device = torch.device(device)
        self.provenance = provenance or {}
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=config.learning_rate,
                                          weight_decay=config.weight_decay)
        self.batcher = DeterministicBatcher(len(train_set), config.batch_size, config.seed)
        self.generator = torch.Generator(device=self.device).manual_seed(config.seed)
        subset = min(int(config.val_molecules), len(val_set))
        picker = torch.Generator().manual_seed(VALIDATION_SUBSET_SEED)
        self.val_indices = torch.randperm(len(val_set), generator=picker)[:subset].tolist()
        self.history: list[dict[str, Any]] = []
        self.best = float("inf")

    # -- the objective -----------------------------------------------------
    def _prepare(self, batch, *, dropout: bool):
        batch = batch.to(self.device)
        width = self.model.config.max_nodes
        if batch.max_nodes > width:
            raise ValueError(f"a molecule has {batch.max_nodes} atoms, above max_nodes "
                             f"{width}")
        counts = batch.n_node.tolist()
        node_mask = torch.zeros((batch.batch_size, width), dtype=torch.bool,
                               device=self.device)
        atoms = torch.zeros((batch.batch_size, width), dtype=torch.long,
                            device=self.device)
        bonds = torch.zeros((batch.batch_size, width, width), dtype=torch.long,
                            device=self.device)
        for row, count in enumerate(counts):
            node_mask[row, :count] = True
            atoms[row, :count] = batch.atom_types[row, :count]
            bonds[row, :count, :count] = batch.bond_types[row, :count, :count]
        pair_mask = pair_mask_from_counts(counts, width, self.device)
        x_atom, x_bond = dequantize(
            atoms, bonds, node_mask, pair_mask,
            num_atom_types=self.model.config.num_atom_types,
            num_bond_types=self.model.config.num_bond_types,
            noise_scale=self.model.config.noise_scale, generator=self.generator)
        if batch.condition is None:
            raise ValueError("the dataset carries no condition column")
        present = (self.model.dropout_mask(batch.batch_size, self.device, self.generator)
                   if dropout else None)
        context = self.model.conditioning(batch.condition, present)
        return x_atom, x_bond, node_mask, pair_mask, context, batch

    def _loss(self, batch, *, dropout: bool) -> dict[str, torch.Tensor]:
        x_atom, x_bond, node_mask, pair_mask, context, batch = self._prepare(
            batch, dropout=dropout)
        result = self.model.log_likelihood(x_atom, x_bond, node_mask, pair_mask, context)
        dimensions = result["dimensions"].clamp(min=1.0)
        # per-dimension so molecules of different size contribute comparably
        nll = -(result["log_prob"] / dimensions).mean()
        return {"loss": nll,
                "nll_per_dim": nll.detach(),
                "bound_per_dim": (-(result["discrete_lower_bound"] / dimensions)
                                  .mean().detach()),
                "logdet_bond": result["log_prob_bond"].detach().mean(),
                "logdet_atom": result["log_prob_atom"].detach().mean()}

    # -- the diagnostic ----------------------------------------------------
    @torch.no_grad()
    def condition_gap(self, indices: Sequence[int]) -> dict[str, Any]:
        """Score the same molecules under true and within-N-permuted QED."""
        self.model.eval()
        gaps: list[float] = []
        true_scores: list[float] = []
        null_scores: list[float] = []
        counts_by_n: dict[int, list[int]] = {}
        for position, index in enumerate(indices):
            counts_by_n.setdefault(int(self.val_set.n_node[int(index)]), []).append(position)
        rng = np.random.default_rng(PERMUTATION_SEED)
        permuted_of = {}
        for same in counts_by_n.values():
            order = rng.permutation(len(same))
            for slot, position in enumerate(same):
                permuted_of[position] = same[int(order[slot])]

        step = max(1, self.config.batch_size)
        for start in range(0, len(indices), step):
            chunk = [int(i) for i in indices[start:start + step]]
            batch = make_batch(self.val_set, chunk)
            x_atom, x_bond, node_mask, pair_mask, context, batch = self._prepare(
                batch, dropout=False)
            true = self.model.log_likelihood(x_atom, x_bond, node_mask, pair_mask,
                                             context)["log_prob"]
            # The permuted condition is taken as the STORED condition vector of a donor
            # molecule with the SAME N. Because the donor shares N, its N column is
            # identical and only the QED column changes - so no scaler is needed here and
            # the swap provably varies QED alone.
            donor = [int(indices[permuted_of[start + offset]]) for offset in range(len(chunk))]
            swapped = torch.stack([
                torch.as_tensor(self.val_set[int(d)]["condition"], dtype=torch.float32)
                for d in donor]).to(self.device)
            other = self.model.log_likelihood(
                x_atom, x_bond, node_mask, pair_mask,
                self.model.conditioning(swapped))["log_prob"]
            null = self.model.log_likelihood(
                x_atom, x_bond, node_mask, pair_mask,
                self.model.null_context(len(chunk), self.device))["log_prob"]
            dimensions = torch.as_tensor(
                [float(self.val_set.n_node[int(i)]) for i in chunk],
                device=self.device)
            gaps.extend(((true - other) / dimensions.clamp(min=1.0)).cpu().tolist())
            true_scores.extend((true / dimensions.clamp(min=1.0)).cpu().tolist())
            null_scores.extend((null / dimensions.clamp(min=1.0)).cpu().tolist())
        array = np.asarray(gaps, dtype=np.float64)
        return {
            "molecules": int(array.size),
            "condition_gap_per_atom": float(array.mean()) if array.size else float("nan"),
            "condition_gap_sem": (float(array.std(ddof=1) / np.sqrt(array.size))
                                  if array.size > 1 else float("nan")),
            "condition_win_rate": float((array > 0).mean()) if array.size else float("nan"),
            "conditional_logp_per_atom": float(np.mean(true_scores)) if true_scores
            else float("nan"),
            "null_logp_per_atom": float(np.mean(null_scores)) if null_scores
            else float("nan"),
            "note": ("QED is permuted WITHIN N, so the gap cannot be earned by "
                     "recognising N. A gap of 0 means the density ignores QED."),
        }

    @torch.no_grad()
    def evaluate(self) -> dict[str, Any]:
        self.model.eval()
        totals: dict[str, list[float]] = {}
        step = max(1, self.config.batch_size)
        for start in range(0, len(self.val_indices), step):
            chunk = self.val_indices[start:start + step]
            metrics = self._loss(make_batch(self.val_set, chunk), dropout=False)
            for key, value in metrics.items():
                if key == "loss":
                    continue
                totals.setdefault(key, []).append(float(value))
        summary = {key: float(np.mean(values)) for key, values in totals.items()}
        summary["molecules"] = len(self.val_indices)
        summary.update(self.condition_gap(self.val_indices))
        return summary

    # -- checkpoints -------------------------------------------------------
    def _write(self, name: str, step: int) -> Path:
        path = self.output / name
        payload = {
            "model_state_dict": self.model.state_dict(),
            "model_config": self.model.config.to_dict(),
            "train_config": self.config.to_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "batcher": self.batcher.state_dict(),
            "generator_state": self.generator.get_state(),
            "step": int(step),
            "best_validation": self.best,
            "history": self.history,
            "provenance": self.provenance,
        }
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        return path

    def _resume(self) -> int:
        path = self.output / "last.pt"
        if not path.exists():
            return 0
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.batcher.load_state_dict(checkpoint["batcher"])
        state = checkpoint.get("generator_state")
        if state is not None:
            self.generator.set_state(state.cpu() if hasattr(state, "cpu") else state)
        self.best = float(checkpoint.get("best_validation", float("inf")))
        self.history = list(checkpoint.get("history", []))
        step = int(checkpoint["step"])
        self.logger.info(f"resumed from {path} at step {step}")
        return step

    def _append(self, record: dict[str, Any]) -> None:
        with (self.output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=float) + "\n")

    # -- the loop ----------------------------------------------------------
    def train(self) -> dict[str, Any]:
        step = self._resume()
        if step >= self.config.max_steps:
            self.logger.info(f"already at step {step} >= {self.config.max_steps}; "
                             f"nothing to do")
            return {"step": step, "history": self.history, "completed": True}
        started = time.perf_counter()
        window: list[dict[str, float]] = []
        self.logger.info(f"training to step {self.config.max_steps} "
                         f"(batch {self.config.batch_size}, lr {self.config.learning_rate}, "
                         f"{self.batcher.batches_per_epoch} batches per epoch)")
        while step < self.config.max_steps:
            self.model.train()
            indices = self.batcher.next_indices()
            metrics = self._loss(make_batch(self.train_set, indices), dropout=True)
            loss = metrics["loss"]
            if not torch.isfinite(loss):
                self.logger.error(f"non-finite loss at step {step + 1}; stopping")
                self._write("last.pt", step)
                return {"step": step, "history": self.history, "completed": False,
                        "failure": "non_finite_loss"}
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                  self.config.grad_clip)
            scale = min(1.0, (step + 1) / max(1, self.config.warmup_steps))
            for group in self.optimizer.param_groups:
                group["lr"] = self.config.learning_rate * scale
            self.optimizer.step()
            step += 1
            window.append({k: float(v) for k, v in metrics.items() if k != "loss"})
            window[-1]["grad_norm"] = float(norm)

            if step % self.config.log_every == 0 or step == self.config.max_steps:
                mean = {k: float(np.mean([w[k] for w in window])) for k in window[0]}
                rate = step / max(time.perf_counter() - started, 1e-9)
                self.logger.info(
                    f"step {step}/{self.config.max_steps} "
                    f"nll/dim={mean['nll_per_dim']:.4f} "
                    f"bound/dim={mean['bound_per_dim']:.4f} "
                    f"grad={mean['grad_norm']:.2f} lr={self.optimizer.param_groups[0]['lr']:.2e} "
                    f"it/s={rate:.2f}")
                self._append({"kind": "train", "step": step, **mean})
                window = []
            if step % self.config.eval_every == 0 or step == self.config.max_steps:
                summary = self.evaluate()
                improved = summary["nll_per_dim"] < self.best
                if improved:
                    self.best = float(summary["nll_per_dim"])
                    self._write("best.pt", step)
                self.history.append({"step": step, "improved": improved, **summary})
                self._append({"kind": "val", "step": step, "improved": improved,
                              **summary})
                self.logger.info(
                    f"VAL step {step}: nll/dim={summary['nll_per_dim']:.4f} "
                    f"bound/dim={summary['bound_per_dim']:.4f} "
                    f"cond_gap/atom={summary['condition_gap_per_atom']:+.5f} "
                    f"+-{summary['condition_gap_sem']:.5f} "
                    f"win={summary['condition_win_rate']:.3f}"
                    + ("  *BEST*" if improved else ""))
            if step % self.config.save_every == 0 or step == self.config.max_steps:
                self._write("last.pt", step)
        self._write("last.pt", step)
        trend = self._gap_trend()
        self.logger.info(f"finished at step {step}; condition-gap trend: {trend['reading']}")
        return {"step": step, "history": self.history, "completed": True,
                "condition_gap_trend": trend}

    def _gap_trend(self) -> dict[str, Any]:
        """Is the conditional gap still rising? The pre-registered undertraining test."""
        usable = [h for h in self.history
                  if h.get("condition_gap_per_atom") is not None
                  and np.isfinite(h["condition_gap_per_atom"])]
        points = [(h["step"], float(h["condition_gap_per_atom"])) for h in usable]
        if len(points) < 3:
            return {"reading": "too few evaluations to judge the trend",
                    "status": "unknown", "points": points}
        half = len(points) // 2
        early = float(np.mean([v for _, v in points[:half]]))
        late = float(np.mean([v for _, v in points[half:]]))
        last = float(points[-1][1])
        sem = usable[-1].get("condition_gap_sem")
        sem = float(sem) if sem is not None and np.isfinite(sem) else float("nan")
        # "Resolved" means the final gap is positive by more than twice its own standard
        # error. A NEGATIVE gap - the model preferring the permuted label - is not weak
        # evidence of conditioning; it is no evidence, and is reported as such.
        resolved = bool(np.isfinite(sem) and last > 2.0 * sem)
        rising = late > early + abs(early) * 0.1 + 1e-6
        if rising and last > 0:
            status = "rising_undertrained"
            reading = (f"still RISING ({early:+.5f} -> {late:+.5f} per atom, final "
                       f"{last:+.5f} +-{sem:.5f}): the run is UNDERTRAINED and no "
                       "conclusion about the architecture is warranted. Extend the budget.")
        elif resolved:
            status = "plateaued_positive"
            reading = (f"plateaued at {last:+.5f} +-{sem:.5f} per atom, positive beyond "
                       "twice its standard error: the density DOES notice QED. Whether "
                       "SAMPLING follows it is a separate question - and stage 17 found "
                       "exactly that dissociation in embedding space.")
        else:
            status = "plateaued_zero"
            reading = (f"final gap {last:+.5f} +-{sem:.5f} per atom is not positive beyond "
                       "twice its standard error: the density is not demonstrably using "
                       "QED. A negative or within-noise gap is NO evidence of "
                       "conditioning, and a longer run at this size would not change it.")
        return {"reading": reading, "status": status, "early_mean": early,
                "late_mean": late, "final": last, "final_sem": sem,
                "resolved_positive": resolved, "points": points}


def load_graph_flow(path: str | Path, device: torch.device | str = "cpu"
                    ) -> tuple[ConditionalGraphFlow, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    model = ConditionalGraphFlow(GraphFlowConfig.from_dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    meta = {key: checkpoint.get(key) for key in
            ("step", "best_validation", "history", "provenance", "train_config")}
    return model, meta


__all__ = [
    "GraphFlowTrainConfig",
    "GraphFlowTrainer",
    "load_graph_flow",
    "shuffle_qed_within_n",
]
