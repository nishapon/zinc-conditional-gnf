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

THE WEIGHTS ARE DEFINED PER CELL, THEN SPREAD UNIFORMLY INSIDE IT
----------------------------------------------------------------
The quantity being reweighted is the CELL probability mass, not an ad-hoc per-example
factor. Within a cell every molecule is equally likely, so the per-example probability is
just `q_c / n_c`. Writing it this way makes the guards checkable, which the earlier
per-example factor did not.

With natural cell mass `p_c = n_c / M`:

    eligible cells   (n_c >= min_cell_count):  q_c proportional to n_c^(1 - alpha)
    tiny cells       (n_c <  min_cell_count):  q_c = p_c   EXACTLY

The eligible masses are renormalised to fill whatever mass the tiny cells did not claim,
so the vector sums to one. `alpha = 0` returns `q_c = p_c` for every cell, i.e. the
current uniform-over-molecules sampler, bit for bit.

The reported quantity is the EXPOSURE RATIO `e_c = q_c / p_c`: how many times more often
a cell is seen than under natural sampling. By construction `e_c = 1` for every tiny
cell BEFORE clipping, which is the property that matters -- a cell of four molecules is
not upweighted at all, because upweighting four molecules memorises them rather than
covering their region. `exposure_ratio_by_cell` and
`max_exposure_ratio_among_tiny_cells` are both returned so this is verified against the
realised numbers instead of inferred from the construction.

EXPOSURE CLIP. `e_c` is clipped to `[1/max_exposure_ratio, max_exposure_ratio]` and the
masses renormalised. Without it the rarest eligible cell in ZINC250K sits orders of
magnitude below the mode and a handful of molecules would be drawn thousands of times per
epoch. Clipping only ever pulls ratios toward 1, so it cannot create a new outlier.

A cell with zero molecules has no index and can never be drawn, so the sampler cannot
invent support for conditions the data does not contain.

Kish's effective sample size is reported so the realised concentration is measured.

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
DEFAULT_MAX_EXPOSURE_RATIO = 20.0
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
                 max_exposure_ratio: float = DEFAULT_MAX_EXPOSURE_RATIO,
                 qed_bin_width: float = DEFAULT_QED_BIN_WIDTH) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if max_exposure_ratio < 1.0:
            raise ValueError("max_exposure_ratio must be >= 1")
        if min_cell_count < 1:
            raise ValueError("min_cell_count must be >= 1")
        self.size = int(len(np.asarray(n_node)))
        if self.size < batch_size:
            raise ValueError("need at least batch_size molecules")
        self.batch_size, self.seed = int(batch_size), int(seed)
        self.alpha, self.min_cell_count = float(alpha), int(min_cell_count)
        self.max_exposure_ratio = float(max_exposure_ratio)
        self.distribution = joint_distribution(n_node, qed,
                                               qed_bin_width=qed_bin_width)
        self.weights, self.diagnostics = self._weights()
        self.epoch, self.position = 0, 0
        self._order = self._draw(self.epoch)

    def _weights(self):
        """Per-CELL masses, then uniform inside each cell. Returns per-example weights."""
        populations = np.asarray(self.distribution["cell_populations"],
                                 dtype=np.float64)
        assignment = self.distribution["assignment"]
        total = populations.sum()
        natural = populations / total                      # p_c

        tiny = populations < self.min_cell_count
        mass = np.zeros_like(natural)
        mass[tiny] = natural[tiny]                         # q_c = p_c EXACTLY
        remaining = max(0.0, 1.0 - mass[tiny].sum())
        if (~tiny).any():
            shape = populations[~tiny] ** (1.0 - self.alpha)
            mass[~tiny] = remaining * shape / shape.sum()
        else:
            mass = natural.copy()

        exposure = np.divide(mass, natural, out=np.ones_like(mass),
                             where=natural > 0)            # e_c
        tiny_exposure_before_clip = (float(exposure[tiny].max()) if tiny.any()
                                     else None)
        # clipping only pulls ratios toward 1, so it cannot create a new outlier
        clipped = np.clip(exposure, 1.0 / self.max_exposure_ratio,
                          self.max_exposure_ratio)
        mass = natural * clipped
        mass = mass / mass.sum()
        exposure = np.divide(mass, natural, out=np.ones_like(mass),
                             where=natural > 0)

        weights = mass[assignment] / populations[assignment]   # uniform within cell
        weights = weights / weights.sum()
        effective = float(1.0 / np.square(weights).sum())
        return weights, {
            "alpha": self.alpha,
            "min_cell_count": self.min_cell_count,
            "max_exposure_ratio": self.max_exposure_ratio,
            "definition": ("weights are per-CELL masses q_c spread uniformly inside the "
                           "cell, so the per-example probability is q_c / n_c"),
            "cells_eligible": int((~tiny).sum()),
            "cells_tiny_held_at_natural_mass": int(tiny.sum()),
            "molecules_in_tiny_cells": int(populations[tiny].sum()),
            "natural_cell_probability": [float(v) for v in natural],
            "final_cell_probability": [float(v) for v in mass],
            "exposure_ratio_by_cell": [float(v) for v in exposure],
            "max_exposure_ratio_among_tiny_cells_before_clip":
                tiny_exposure_before_clip,
            "max_exposure_ratio_among_tiny_cells": (
                float(exposure[tiny].max()) if tiny.any() else None),
            "max_exposure_ratio_realised": float(exposure.max()),
            "min_exposure_ratio_realised": float(exposure.min()),
            "effective_sample_size": effective,
            "effective_sample_fraction": effective / max(1, self.size),
            "note": ("exposure_ratio is q_c / p_c, how many times more often a cell is "
                     "seen than under natural sampling. Tiny cells are 1.0 before "
                     "clipping by construction and the realised value is reported so "
                     "that is checked, not assumed. effective_sample_size is Kish's."),
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
                "max_exposure_ratio": self.max_exposure_ratio,
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
    "DEFAULT_MAX_EXPOSURE_RATIO",
    "DEFAULT_MIN_CELL_COUNT",
    "DEFAULT_QED_BIN_WIDTH",
    "BalancedSampler",
    "cell_index",
    "joint_distribution",
]
