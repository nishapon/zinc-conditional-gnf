"""Generated-QED relabeling: pool storage, the mixed objective, and the fine-tune loop.

THE IDEA, AND EXACTLY WHAT IT CAN AND CANNOT DO
-----------------------------------------------
Sample `x ~ p_theta(.|c_requested)`, decode it with the FROZEN autoencoder, measure the
ACTUAL RDKit QED of the decoded molecule, and fine-tune the flow on `(x, c_actual)`
instead of `(x, c_requested)`. This is hindsight relabeling. Its fixed point is

    p*(x | c) = p_bar_theta( x | q(x) = c ),     p_bar_theta(x) = int p_theta(x|c') pi(c') dc'

so in the limit E[q | c] = c for every c in the support of q under the model's own
aggregate output. That is a well-defined objective and it is FORWARD-ONLY: the gradient
goes through `flow.forward` and `prior.log_prob`, never through `flow.inverse`, so none
of the four sites that made the v1 QED-consistency run non-finite (`flow.py:247`
matrix inverse, `flow.py:503-520` sequential autoregressive inverse, the unbounded
translation heads) is in any backward path. Pool collection calls `flow.inverse` exactly
once per sample, under `no_grad`, and stores the result as data.

Two limits are structural, not implementation details:

  * relabeling injects NO NEW CHEMISTRY. It re-indexes the model's existing output, so it
    can only learn to address QED values the checkpoint already produces. Whatever the
    collected pool's QED support is, that is the ceiling.
  * the label is DECODER-INDEXED. `q` here is QED(RDKit(degree-constrained assembly of
    argmax(decoder(x)))). A different assembler gives a different label for the same
    embedding, so the trained conditional means "QED under the degree decoder" and
    nothing more. The evaluation therefore scores the response under raw, constrained
    AND degree separately: a response that appears only under the label decoder is a
    correction-dependent artefact, not intrinsic conditioning.

SMOOTHING NOISE ON THE RELABELED TERM
-------------------------------------
The relabeled term gets the SAME `add_smoothing_noise` treatment as the real term. This
is not cosmetic. The generated embeddings are outputs of `flow.inverse` applied to prior
draws, so they lie on (or very near) a set the model already assigns high density to;
maximising their likelihood without a noise floor drives the density toward a point mass
and the objective toward unbounded likelihood -- the exact degeneracy
`NodeFlow.add_smoothing_noise` exists to remove, and the reason the whitening uses a
floored std. Separate generators are used for the real and relabeled streams so neither
consumes the other's noise.

SELECTION BIAS, STATED UP FRONT
-------------------------------
Only samples that decode to a scorable molecule under the label decoder carry a label.
Raw validity is this project's known weak point, so the pool is a validity-selected
subsample of the model's output and `retained_fraction` is reported with every pool. The
fine-tune therefore sees a biased slice of the model's own distribution; that bias is a
property of the proposal, not a bug, and the real-NLL anchor exists to bound the damage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from zinc_gnf.conditioning import (
    CONDITION_SHUFFLE_SEED,
    batch_condition,
    condition_embedding_for,
    standardise_condition,
)
from zinc_gnf.dataset import iter_eval_batches, seeded_subset_indices
from zinc_gnf.flow import node_mask_from_counts
from zinc_gnf.trainer_flow import (
    VALIDATION_NOISE_SEED,
    VALIDATION_SUBSET_SEED,
    encode_batch,
)

# Dedicated, DISTINCT noise streams. The pool seed and the evaluation seed must never
# coincide: evaluating on the same prior draws the fine-tune trained on would be direct
# leakage and would manufacture a response out of memorisation.
POOL_NOISE_SEED = 20261109
EVAL_NOISE_SEED = 777_000_111
POOL_SPLIT_SEED = 20261110
RELABEL_NOISE_SEED = 20261111          # smoothing noise for the relabeled term
REAL_ANCHOR_NOISE_SEED = 20261112      # smoothing noise for the real anchor term

LABEL_DECODER = "degree"               # the assembler the labels are tied to
HOLDOUT_FRACTION = 0.15                # of the pool, split by generation cell and index


def assert_disjoint_seeds() -> None:
    """Fail loudly rather than silently leak if anyone edits the constants."""
    seeds = [POOL_NOISE_SEED, EVAL_NOISE_SEED, POOL_SPLIT_SEED, RELABEL_NOISE_SEED,
             REAL_ANCHOR_NOISE_SEED, VALIDATION_NOISE_SEED]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"relabeling seeds must all differ, got {seeds}")


# ------------------------------------------------------------------------ pool ---
@dataclass
class PoolConfig:
    target_n: tuple[int, ...] = (15, 23, 30)
    qed_requests: tuple[float, ...] = (0.4, 0.6, 0.8)
    samples_per_cell: int = 400
    temperature: float = 0.2
    noise_seed: int = POOL_NOISE_SEED
    label_decoder: str = LABEL_DECODER

    def to_dict(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in asdict(self).items()}


@dataclass
class RelabelPool:
    """Generated embeddings with their ACTUAL decoded QED. Tensors live on disk only.

    `embeddings` is a list of `[n_i, D]` float tensors, one per sample, exactly as
    `flow.inverse` produced them -- never re-encoded from SMILES, which would be a
    different experiment.
    """

    embeddings: list[torch.Tensor] = field(default_factory=list)
    n_node: list[int] = field(default_factory=list)
    requested_qed: list[float] = field(default_factory=list)
    actual_qed: list[float] = field(default_factory=list)
    smiles: list[str] = field(default_factory=list)
    valid: dict[str, list[bool]] = field(default_factory=dict)
    fallback: list[bool] = field(default_factory=list)
    split: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.embeddings)

    def indices_for(self, split: str) -> list[int]:
        return [i for i, s in enumerate(self.split) if s == split]

    def qed_support(self) -> dict[str, Any]:
        array = np.asarray(self.actual_qed, dtype=np.float64)
        if array.size == 0:
            return {"molecules": 0}
        return {
            "molecules": int(array.size),
            "mean": float(array.mean()), "std": float(array.std()),
            "min": float(array.min()), "max": float(array.max()),
            "quantiles": {f"p{q}": float(np.percentile(array, q))
                          for q in (5, 25, 50, 75, 95)},
            "iqr": float(np.percentile(array, 75) - np.percentile(array, 25)),
            "fraction_at_least_0p75": float((array >= 0.75).mean()),
        }

    def save(self, directory: str | Path) -> dict[str, Any]:
        """Tensors to `pool.pt`, everything else to `pool_manifest.json`.

        Kept OUT of git by `.gitignore`: this is model output, not source.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({"embeddings": [e.detach().cpu() for e in self.embeddings],
                    "n_node": self.n_node}, directory / "pool.pt")
        manifest = {
            "samples": len(self),
            "n_node": self.n_node,
            "requested_qed": self.requested_qed,
            "actual_qed": self.actual_qed,
            "smiles": self.smiles,
            "valid": self.valid,
            "fallback": self.fallback,
            "split": self.split,
            "metadata": self.metadata,
            "qed_support": self.qed_support(),
        }
        (directory / "pool_manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest

    @classmethod
    def load(cls, directory: str | Path) -> "RelabelPool":
        directory = Path(directory)
        tensors_path, manifest_path = directory / "pool.pt", directory / "pool_manifest.json"
        if not tensors_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"{directory} must contain pool.pt and pool_manifest.json; run "
                f"scripts/34_collect_relabel_pool.py first")
        blob = torch.load(tensors_path, map_location="cpu", weights_only=False)
        manifest = json.loads(manifest_path.read_text())
        pool = cls(embeddings=blob["embeddings"], n_node=manifest["n_node"],
                   requested_qed=manifest["requested_qed"],
                   actual_qed=manifest["actual_qed"], smiles=manifest["smiles"],
                   valid=manifest["valid"], fallback=manifest["fallback"],
                   split=manifest["split"], metadata=manifest["metadata"])
        if len(pool.embeddings) != manifest["samples"]:
            raise ValueError("pool.pt and pool_manifest.json disagree on the sample count")
        if blob["n_node"] != manifest["n_node"]:
            raise ValueError("pool.pt and pool_manifest.json disagree on the node counts")
        return pool


def assign_splits(n_node: Sequence[int], requested_qed: Sequence[float], *,
                  holdout_fraction: float = HOLDOUT_FRACTION,
                  seed: int = POOL_SPLIT_SEED) -> list[str]:
    """Stratified train/holdout split WITHIN each (N, request) cell.

    Stratified so the holdout covers every cell rather than whichever cells happened to
    be drawn last, and so a cell that decoded badly cannot silently vanish from one side.
    """
    cells: dict[tuple[int, float], list[int]] = {}
    for index, (n, qed) in enumerate(zip(n_node, requested_qed)):
        cells.setdefault((int(n), round(float(qed), 6)), []).append(index)
    split = ["train"] * len(n_node)
    generator = np.random.default_rng(seed)
    for key in sorted(cells):
        members = cells[key]
        order = generator.permutation(len(members))
        held = int(round(holdout_fraction * len(members)))
        for position in order[:held]:
            split[members[position]] = "holdout"
    return split


def batch_from_pool(pool: RelabelPool, indices: Sequence[int], *, device) -> dict[str, Any]:
    """Pad the selected pooled embeddings into one `[B, N, D]` batch plus its labels."""
    if not len(indices):
        raise ValueError("need at least one pooled sample")
    counts = [int(pool.n_node[i]) for i in indices]
    width = max(counts)
    dimension = pool.embeddings[indices[0]].shape[-1]
    embeddings = torch.zeros((len(indices), width, dimension), dtype=torch.float32)
    for row, index in enumerate(indices):
        tensor = pool.embeddings[index]
        embeddings[row, :tensor.shape[0]] = tensor
    mask = node_mask_from_counts(counts, max_nodes=width, device=device)
    return {
        "embeddings": embeddings.to(device),
        "node_mask": mask,
        "n_node": counts,
        "actual_qed": [float(pool.actual_qed[i]) for i in indices],
        "requested_qed": [float(pool.requested_qed[i]) for i in indices],
    }


def relabeled_condition(flow, scaler, *, n_node, qed, device) -> torch.Tensor:
    """Condition embedding built from the ACTUAL decoded QED, not the request."""
    values = torch.from_numpy(
        standardise_condition(flow.config.condition_features, scaler=scaler,
                              n=list(n_node), qed=list(qed))).to(device)
    return condition_embedding_for(flow, values, torch.ones_like(values),
                                   len(n_node), device)


# ------------------------------------------------------------------- objective ---
def mixed_loss(flow, *, real_batch, pool_batch, scaler, relabel_weight: float,
               real_generator=None, relabel_generator=None, device
               ) -> tuple[torch.Tensor, dict[str, float]]:
    """`(1-w) * NLL(real, c_real) + w * NLL(generated, c_actual)`, both per node.

    Forward-only: the gradient path is `flow.forward` plus `prior.log_prob`. Nothing
    here touches `flow.inverse`.
    """
    if not 0.0 <= relabel_weight <= 1.0:
        raise ValueError("relabel_weight must lie in [0, 1]")
    diagnostics: dict[str, float] = {}
    total = None

    if relabel_weight < 1.0:
        if real_batch is None:
            raise ValueError("relabel_weight < 1 requires a real batch anchor")
        # the caller supplies already-encoded real embeddings, so the frozen encoder is
        # never in the gradient path of the anchor either
        real_x = flow.add_smoothing_noise(real_batch["embeddings"],
                                          real_batch["node_mask"],
                                          generator=real_generator)
        parts = flow.negative_log_likelihood(real_x, real_batch["node_mask"],
                                             real_batch["condition"])
        total = (1.0 - relabel_weight) * parts["loss"]
        diagnostics["real_nll_per_node"] = float(parts["loss"].detach())
        diagnostics["real_log_det_per_node"] = float(parts["log_det_per_node"].detach())

    if relabel_weight > 0.0:
        if pool_batch is None:
            raise ValueError("relabel_weight > 0 requires a relabeled batch")
        condition = relabeled_condition(
            flow, scaler, n_node=pool_batch["n_node"],
            qed=pool_batch["actual_qed"], device=device)
        generated = flow.add_smoothing_noise(pool_batch["embeddings"],
                                             pool_batch["node_mask"],
                                             generator=relabel_generator)
        parts = flow.negative_log_likelihood(generated, pool_batch["node_mask"], condition)
        term = relabel_weight * parts["loss"]
        total = term if total is None else total + term
        diagnostics["relabel_nll_per_node"] = float(parts["loss"].detach())
        diagnostics["relabel_log_det_per_node"] = float(parts["log_det_per_node"].detach())

    if total is None:
        raise ValueError("relabel_weight produced an empty objective")
    diagnostics["loss"] = float(total.detach())
    return total, diagnostics


# ------------------------------------------------------------------ validation ---
@torch.no_grad()
def real_validation_nll(flow, autoencoder, val_set, *, scaler, device, batch_size=64,
                        val_molecules: int | None = 5000) -> dict[str, Any]:
    """Per-node NLL on the ORIGINAL validation subset with the ORIGINAL noise seed.

    Deliberately mirrors `FlowTrainer.validate` -- same subset seed, same noise seed,
    same permuted-condition control -- so the number is directly comparable with the
    original 50k run's logs rather than being a new, incomparable metric.
    """
    was_training = flow.training
    flow.eval()
    indices = seeded_subset_indices(len(val_set), val_molecules, VALIDATION_SUBSET_SEED)
    generator = torch.Generator(device=device).manual_seed(VALIDATION_NOISE_SEED)
    shuffle_rng = (np.random.default_rng(CONDITION_SHUFFLE_SEED)
                   if flow.config.conditional else None)
    log_prob = shuffled = 0.0
    nodes = molecules = wins = scored = 0
    for batch in iter_eval_batches(val_set, batch_size, indices):
        batch = batch.to(device)
        embeddings = encode_batch(autoencoder, batch)
        embeddings = flow.add_smoothing_noise(embeddings, batch.node_mask,
                                              generator=generator)
        values, present = batch_condition(batch, flow.config.condition_features)
        condition = condition_embedding_for(flow, values, present, batch.batch_size,
                                            device) if flow.config.conditional else None
        parts = flow.log_prob(embeddings, batch.node_mask, condition)
        log_prob += float(parts["log_prob"].sum())
        nodes += int(batch.node_mask.sum())
        molecules += batch.batch_size
        if shuffle_rng is not None and batch.batch_size > 1:
            order = shuffle_rng.permutation(batch.batch_size)
            if np.array_equal(order, np.arange(batch.batch_size)):
                order = np.roll(order, 1)
            permuted = values[torch.as_tensor(order.copy(), device=values.device)]
            control = flow.log_prob(
                embeddings, batch.node_mask,
                condition_embedding_for(flow, permuted, present, batch.batch_size, device))
            shuffled += float(control["log_prob"].sum())
            wins += int((parts["log_prob"] > control["log_prob"]).sum())
            scored += batch.batch_size
    if was_training:
        flow.train()
    result = {"val_loss": -log_prob / max(nodes, 1), "val_molecules": molecules,
              "val_nodes": nodes}
    if scored:
        shuffled_loss = -shuffled / max(nodes, 1)
        result.update({
            "val_loss_shuffled_condition": shuffled_loss,
            # > 0 means a molecule's own condition explains it better than a permuted
            # one; ~0 means the conditioning is being ignored.
            "val_condition_gap": shuffled_loss - result["val_loss"],
            "val_condition_win_rate": wins / scored,
        })
    return result


# ------------------------------------------------------------------- fine-tune ---
@dataclass
class FinetuneConfig:
    """Short and fixed. No search: every value here is set once and recorded."""

    max_steps: int = 400
    learning_rate: float = 1e-5        # 100x below the original 1e-3; this is a nudge
    warmup_steps: int = 50
    grad_clip_norm: float = 5.0
    weight_decay: float = 0.0
    relabel_weight: float = 0.5
    real_batch_size: int = 32
    pool_batch_size: int = 32
    eval_every: int = 100
    seed: int = 20261113

    def validate(self) -> None:
        if self.max_steps < 1 or self.eval_every < 1:
            raise ValueError("max_steps and eval_every must be >= 1")
        if self.warmup_steps < 0 or self.warmup_steps > self.max_steps:
            raise ValueError("warmup_steps must lie in 0..max_steps")
        if not 0.0 < self.learning_rate <= 1e-3:
            raise ValueError("learning_rate must be in (0, 1e-3]; this is a fine-tune")
        if not 0.0 <= self.relabel_weight <= 1.0:
            raise ValueError("relabel_weight must lie in [0, 1]")
        if self.real_batch_size < 1 or self.pool_batch_size < 1:
            raise ValueError("both batch sizes must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def learning_rate_at(config: FinetuneConfig, step: int) -> float:
    if config.warmup_steps and step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    return config.learning_rate


def holdout_relabel_nll(flow, pool: RelabelPool, *, scaler, device, batch_size=32
                        ) -> dict[str, Any]:
    """NLL of HELD-OUT relabeled pairs: did the fine-tune generalise or memorise?"""
    indices = pool.indices_for("holdout")
    if not indices:
        return {"samples": 0}
    was_training = flow.training
    flow.eval()
    total = 0.0
    nodes = 0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start:start + batch_size]
            batch = batch_from_pool(pool, chunk, device=device)
            condition = relabeled_condition(flow, scaler, n_node=batch["n_node"],
                                            qed=batch["actual_qed"], device=device)
            parts = flow.log_prob(batch["embeddings"], batch["node_mask"], condition)
            total += float(parts["log_prob"].sum())
            nodes += int(batch["node_mask"].sum())
    if was_training:
        flow.train()
    return {"samples": len(indices), "nll_per_node": -total / max(nodes, 1)}


__all__ = [
    "EVAL_NOISE_SEED",
    "HOLDOUT_FRACTION",
    "LABEL_DECODER",
    "POOL_NOISE_SEED",
    "POOL_SPLIT_SEED",
    "REAL_ANCHOR_NOISE_SEED",
    "RELABEL_NOISE_SEED",
    "FinetuneConfig",
    "PoolConfig",
    "RelabelPool",
    "assert_disjoint_seeds",
    "assign_splits",
    "batch_from_pool",
    "holdout_relabel_nll",
    "learning_rate_at",
    "mixed_loss",
    "real_validation_nll",
    "relabeled_condition",
]
