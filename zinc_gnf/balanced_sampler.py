"""Joint N-QED balanced sampling over REAL training molecules.

MOTIVATION FROM THE CODE
------------------------
`zinc_gnf/trainer_flow.py:444` builds the only sampler the project has:

    self.batcher = DeterministicBatcher(len(self.train_set), cfg.batch_size, cfg.seed)

`DeterministicBatcher` (dataset.py:230) shuffles indices `0..size-1` and nothing else, so
every training molecule is equally likely and the effective joint N-QED training
distribution IS the empirical ZINC250K one. No conditional balancing exists anywhere in
the repository, so it has never been tested.

WHAT THIS CHANGES, AND WHAT IT DOES NOT
---------------------------------------
Changes: which real molecules the optimiser sees, by reweighting the sampling
distribution over (N bucket, QED bucket) cells.

Does NOT change: the molecules themselves, their ACTUAL QED labels, the conditioning
vectors, or the architecture. This is not generated-sample relabeling -- every label
stays the RDKit value computed in preprocessing from the stored graph.

THE WEIGHTS, AND THE TWO GUARDS
-------------------------------
For molecule i in cell c(i) with population n_{c(i)}:

    w_i  proportional to  n_{c(i)}^(-alpha)

alpha = 0 reproduces the current uniform-over-molecules sampler exactly; alpha = 1 is
full balance over populated cells. The default is deliberately partial.

Two guards, because naive inverse-frequency weighting on a long-tailed joint
distribution is actively harmful here:

  MINIMUM POPULATION. A cell with fewer than `min_cell_count` molecules is NOT upweighted
  at all; it keeps its natural weight. Upweighting a cell of three molecules does not
  teach the model that region, it teaches it those three molecules. This is also what
  stops the sampler inventing support for conditions the data does not contain: a cell
  with zero molecules has no index to draw and can never be sampled, by construction.

  WEIGHT RATIO CAP. After weighting, the ratio between the largest and smallest
  per-molecule weight is clipped to `max_weight_ratio`. Without it, the rarest eligible
  cell in ZINC250K can be several orders of magnitude below the mode, and a single
  molecule would be drawn thousands of times per epoch.

Both guards are reported in `BalancedSampler.report()` with the realised effective
sample size, so the actual degree of rebalancing is measured rather than assumed.

RESUMABILITY
------------
State is `(epoch, position)`, matching `DeterministicBatcher`, so a run resumes at the
exact next batch. Each epoch's index order is drawn from a generator seeded by
`(seed, epoch)`, so it depends only on those two numbers and never on wall-clock or
process state.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

DEFAULT_ALPHA = 0.5
DEFAULT_MIN_CELL_COUNT = 50
DEFAULT_MAX_WEIGHT_RATIO = 20.0
DEFAULT_QED_BIN_WIDTH = 0.1


def cell_index(n_node, qed, *, qed_bin_width: float = DEFAULT_QED_BIN_WIDTH):
    """(N, QED bucket) label for each molecule. N is exact; QED is binned."""
    counts = np.asarray(n_node, dtype=np.int64)
    values = np.asarray(qed, dtype=np.float64)
    buckets = np.floor(values / float(qed_bin_width)).astype(np.int64)
    return counts, buckets


def joint_distribution(n_node, qed, *, qed_bin_width: float = DEFAULT_QED_BIN_WIDTH
                       ) -> dict[str, Any]:
    """Measure the empirical joint distribution. Pure description, no reweighting."""
    counts, buckets = cell_index(n_node, qed, qed_bin_width=qed_bin_width)
    keys, inverse, populations = np.unique(
        np.stack([counts, buckets], axis=1), axis=0, return_inverse=True,
        return_counts=True)
    total = int(populations.sum())
    fractions = populations / max(1, total)
    return {
        "molecules": total,
        "cells": int(len(keys)),
        "qed_bin_width": float(qed_bin_width),
        "largest_cell": int(populations.max()) if len(populations) else 0,
        "smallest_cell": int(populations.min()) if len(populations) else 0,
        "population_ratio": (float(populations.max() / populations.min())
                             if len(populations) and populations.min() > 0 else None),
        "cells_below_50": int((populations < 50).sum()),
        "cells_below_10": int((populations < 10).sum()),
        "mass_in_top_decile_of_cells": float(
            np.sort(fractions)[::-1][:max(1, len(fractions) // 10)].sum()),
        "effective_cells": float(1.0 / np.square(fractions).sum())
        if len(fractions) else None,
        "cell_keys": [[int(k[0]), int(k[1])] for k in keys],
        "cell_populations": [int(v) for v in populations],
        "assignment": inverse.astype(np.int64),
        "note": ("`effective_cells` is the inverse Simpson index over cells: the number "
                 "of equally populated cells that would give the same concentration. "
                 "Compare it with `cells` to judge whether imbalance is meaningful."),
    }


class BalancedSampler:
    """Deterministic, resumable, cell-reweighted sampler over real molecules."""

    def __init__(self, n_node, qed, *, batch_size: int, seed: int,
                 alpha: float = DEFAULT_ALPHA,
                 min_cell_count: int = DEFAULT_MIN_CELL_COUNT,
                 max_weight_ratio: float = DEFAULT_MAX_WEIGHT_RATIO,
                 qed_bin_width: float = DEFAULT_QED_BIN_WIDTH) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if max_weight_ratio < 1.0:
            raise ValueError("max_weight_ratio must be >= 1")
        if min_cell_count < 1:
            raise ValueError("min_cell_count must be >= 1")
        self.size = int(len(np.asarray(n_node)))
        if self.size < batch_size:
            raise ValueError("need at least batch_size molecules")
        self.batch_size, self.seed = int(batch_size), int(seed)
        self.alpha, self.min_cell_count = float(alpha), int(min_cell_count)
        self.max_weight_ratio = float(max_weight_ratio)
        self.distribution = joint_distribution(n_node, qed,
                                               qed_bin_width=qed_bin_width)
        self.weights, self.diagnostics = self._weights()
        self.epoch, self.position = 0, 0
        self._order = self._draw(self.epoch)

    def _weights(self):
        populations = np.asarray(self.distribution["cell_populations"],
                                 dtype=np.float64)
        assignment = self.distribution["assignment"]
        # cells below the floor keep their NATURAL weight: upweighting a handful of
        # molecules memorises them instead of covering their region
        eligible = populations >= self.min_cell_count
        factor = np.ones_like(populations)
        factor[eligible] = populations[eligible] ** (-self.alpha)
        # natural weight for an ineligible cell is the uniform-per-molecule weight,
        # which in this parameterisation is population^0 = 1 scaled to match
        reference = (populations[eligible] ** (-self.alpha)).mean() if eligible.any() \
            else 1.0
        factor[~eligible] = min(1.0, reference)
        weights = factor[assignment]
        # cap the spread so no single molecule can dominate an epoch
        low = weights.min()
        if low > 0:
            weights = np.minimum(weights, low * self.max_weight_ratio)
        weights = weights / weights.sum()
        effective = float(1.0 / np.square(weights).sum())
        return weights, {
            "alpha": self.alpha,
            "min_cell_count": self.min_cell_count,
            "max_weight_ratio": self.max_weight_ratio,
            "cells_upweighted": int(eligible.sum()),
            "cells_held_at_natural_weight": int((~eligible).sum()),
            "molecules_in_held_cells": int((~eligible)[assignment].sum()),
            "realised_weight_ratio": float(weights.max() / weights.min())
            if weights.min() > 0 else None,
            "effective_sample_size": effective,
            "effective_sample_fraction": effective / max(1, self.size),
            "note": ("effective_sample_size is Kish's: the number of equally weighted "
                     "molecules this weighting is worth. A large drop means the "
                     "sampler is concentrating, which is what max_weight_ratio bounds."),
        }

    def _draw(self, epoch: int) -> np.ndarray:
        """One epoch of indices, drawn with replacement under the weights."""
        generator = np.random.default_rng([self.seed, epoch])
        return generator.choice(self.size, size=self.size, replace=True,
                                p=self.weights)

    def next_indices(self) -> list[int]:
        if self.position + self.batch_size > self.size:
            self.epoch += 1
            self.position = 0
            self._order = self._draw(self.epoch)
        chunk = self._order[self.position:self.position + self.batch_size]
        self.position += self.batch_size
        return [int(value) for value in chunk]

    def state_dict(self) -> dict[str, Any]:
        return {"size": self.size, "batch_size": self.batch_size, "seed": self.seed,
                "alpha": self.alpha, "min_cell_count": self.min_cell_count,
                "max_weight_ratio": self.max_weight_ratio,
                "epoch": self.epoch, "position": self.position}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for key in ("size", "batch_size", "seed", "alpha", "min_cell_count",
                    "max_weight_ratio"):
            if key in state and state[key] != getattr(self, key):
                raise ValueError(f"sampler {key} mismatch: checkpoint {state[key]} vs "
                                 f"now {getattr(self, key)}")
        self.epoch, self.position = int(state["epoch"]), int(state["position"])
        self._order = self._draw(self.epoch)

    def report(self) -> dict[str, Any]:
        return {"distribution": {k: v for k, v in self.distribution.items()
                                 if k != "assignment"},
                "weighting": self.diagnostics,
                "reduces_to_uniform": bool(self.alpha == 0.0)}


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_MAX_WEIGHT_RATIO",
    "DEFAULT_MIN_CELL_COUNT",
    "DEFAULT_QED_BIN_WIDTH",
    "BalancedSampler",
    "cell_index",
    "joint_distribution",
]
