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

EXPOSURE BOUNDS ARE SOLVED FOR, NOT CLIPPED AFTERWARDS
-----------------------------------------------------
An earlier version clipped the exposure ratios and then renormalised the whole vector to
sum to one. That does not work: the renormalisation scales every cell, so a clipped cell
immediately leaves its bound again. On populations (5000, 60) with alpha = 1 and a bound
of 5 it produced a realised ratio of 8.94.

The masses are now the solution of a one-dimensional monotone problem with the clipping
INSIDE it. Hold the tiny-cell mass fixed at natural, then find the scalar s with

    sum_{c in E} clip( s * t_c,  L * p_c,  U * p_c )  =  S_E,
    t_c = n_c^(1 - alpha),   L = 1 / max_exposure_ratio,   U = max_exposure_ratio

where S_E is the mass left over for the eligible cells. The left side is continuous and
non-decreasing in s, equals `L * S_E` at s = 0 and saturates at `U * S_E`, and since
L <= 1 <= U the target S_E is bracketed, so a solution exists and bisection finds it.
Setting q_c = p_c is itself feasible, which is why the natural allocation is always
available when the bounds include 1 -- in particular at alpha = 0 the solve returns the
natural probabilities exactly.

The result satisfies all three requirements simultaneously: the masses sum to one, tiny
cells keep their natural probability, and every eligible exposure ratio lies inside
[L, U]. Verified on the failing case (realised maximum exactly 5.0), on a three-cell
strong imbalance, with several tiny cells, with every cell tiny, and at alpha = 0.

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
# Narrow on purpose: ~1e7 x double precision, ~1e-6 of a bucket. It absorbs the
# representation error at a boundary without moving a value that is clearly inside a bin.
BOUNDARY_TOLERANCE = 1e-9


def cell_index(n_node, qed, *, qed_bin_width: float = DEFAULT_QED_BIN_WIDTH,
               boundary_tolerance: float = BOUNDARY_TOLERANCE):
    """(N, QED bucket) label for each molecule. N is exact; QED is binned.

    BINS ARE LEFT-CLOSED, RIGHT-OPEN: bucket k is [k*w, (k+1)*w), so a value sitting
    exactly on a boundary belongs to the HIGHER bucket. QED = 1.0 therefore lands in
    bucket 1/w, which is its own bucket, and QED = 0.0 lands in bucket 0.

    `floor(q / w)` alone is wrong in floating point: 0.60 / 0.1 evaluates to
    5.999999999999999, so floor gives 5 and a molecule at the 0.6 boundary is filed with
    the 0.5-0.6 bucket. The fix snaps q / w to an integer only when it is within
    `boundary_tolerance` of one. That tolerance is 1e-9 -- about 1e7 times double
    precision, so it absorbs the representation error, and about 1e-6 of a bucket, so it
    cannot move a value that is clearly inside one. A broader epsilon would.
    """
    counts = np.asarray(n_node, dtype=np.int64)
    values = np.asarray(qed, dtype=np.float64)
    scaled = values / float(qed_bin_width)
    nearest = np.round(scaled)
    on_boundary = np.abs(scaled - nearest) <= float(boundary_tolerance)
    buckets = np.where(on_boundary, nearest, np.floor(scaled)).astype(np.int64)
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


# Written into every state dict so a foreign sampler state is positively IDENTIFIED
# rather than inferred from which keys happen to be present. A `DeterministicBatcher`
# state is a strict SUBSET of this one's keys, with the same names and meanings, so no
# key-presence test can distinguish them.
STATE_KIND = "BalancedSampler"


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
        # retained as an attribute because it determines the CELLS, hence the weights;
        # a resume that changed it would sample a different distribution under the same
        # (epoch, position), so it has to be identifiable in the state
        self.qed_bin_width = float(qed_bin_width)
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

        bounds = (1.0 / self.max_exposure_ratio, self.max_exposure_ratio)
        mass, solve = self._solve_eligible_mass(natural, populations, tiny, bounds)
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
            "max_exposure_ratio_among_tiny_cells": (
                float(exposure[tiny].max()) if tiny.any() else None),
            "min_exposure_ratio_among_tiny_cells": (
                float(exposure[tiny].min()) if tiny.any() else None),
            "solve": solve,
            "max_exposure_ratio_realised": float(exposure.max()),
            "min_exposure_ratio_realised": float(exposure.min()),
            "effective_sample_size": effective,
            "effective_sample_fraction": effective / max(1, self.size),
            "note": ("exposure_ratio is q_c / p_c, how many times more often a cell is "
                     "seen than under natural sampling. Tiny cells are exactly 1.0, and "
                     "eligible cells lie inside [1/max_exposure_ratio, "
                     "max_exposure_ratio] because the bound is enforced inside the "
                     "solve rather than by clipping and renormalising. "
                     "effective_sample_size is Kish's."),
        }

    def _solve_eligible_mass(self, natural, populations, tiny, bounds):
        """Bisect for the scale `s` that makes the bounded eligible masses fill S_E.

        f(s) = sum_E clip(s * t_c, L p_c, U p_c) is continuous and non-decreasing, with
        f(0) = L * S_E and f(inf) = U * S_E. Because L <= 1 <= U the target S_E is
        bracketed, so a solution exists; q_c = p_c is itself feasible, which is why
        alpha = 0 returns the natural probabilities exactly.
        """
        low_ratio, high_ratio = bounds
        mass = np.zeros_like(natural)
        mass[tiny] = natural[tiny]
        eligible = ~tiny
        remaining = float(max(0.0, 1.0 - float(mass[tiny].sum())))
        if not eligible.any():
            total = float(mass.sum())
            return (mass / total if total > 0 else natural.copy()), {
                "eligible_cells": 0, "scale": None, "iterations": 0, "residual": 0.0,
                "note": ("every cell is below min_cell_count, so the natural "
                         "probabilities are the only feasible allocation"),
            }
        shape = populations[eligible] ** (1.0 - self.alpha)
        floor = low_ratio * natural[eligible]
        ceiling = high_ratio * natural[eligible]

        def filled(scale):
            return float(np.clip(scale * shape, floor, ceiling).sum())

        low, high = 0.0, float((ceiling / np.maximum(shape, 1e-300)).max()) * 1.000001
        iterations = 0
        for iterations in range(1, 201):
            middle = 0.5 * (low + high)
            if filled(middle) < remaining:
                low = middle
            else:
                high = middle
        scale = 0.5 * (low + high)
        mass[eligible] = np.clip(scale * shape, floor, ceiling)
        return mass, {
            "eligible_cells": int(eligible.sum()),
            "scale": float(scale),
            "iterations": iterations,
            "residual": float(abs(mass.sum() - 1.0)),
            "note": ("bisection on a monotone bounded allocation; the clip is inside "
                     "the solve, so the bounds hold in the final normalised masses "
                     "rather than being undone by a global renormalisation"),
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

    # Keys that must agree for `(epoch, position)` to mean the same thing, because each
    # one changes the weights and therefore the drawn order.
    IDENTIFYING = ("size", "batch_size", "seed", "alpha", "min_cell_count",
                   "max_exposure_ratio", "qed_bin_width")

    def state_dict(self) -> dict[str, Any]:
        return {"sampler": STATE_KIND,
                "size": self.size, "batch_size": self.batch_size, "seed": self.seed,
                "alpha": self.alpha, "min_cell_count": self.min_cell_count,
                "max_exposure_ratio": self.max_exposure_ratio,
                "qed_bin_width": self.qed_bin_width,
                "epoch": self.epoch, "position": self.position}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Resume only from a state this class wrote, with identical weighting.

        Nothing is mutated until every check has passed, so a rejected state leaves the
        sampler exactly as it was rather than half-loaded.

        The previous version checked `if key in state`, which could not reject a
        `DeterministicBatcher` state: that state carries `size`, `batch_size`, `seed`,
        `epoch` and `position` under the same names with the same meanings, and simply
        OMITS `alpha` and `min_cell_count`, so the presence guard skipped exactly the
        keys that distinguish the two samplers and the foreign state loaded silently. It
        also checked a key named `max_weight_ratio`, which `state_dict` has never
        written, so the exposure cap went unvalidated even between two balanced states.
        Both are fixed here: the kind is identified explicitly, and every identifying
        key must be PRESENT and equal.
        """
        kind = state.get("sampler")
        if kind != STATE_KIND:
            raise ValueError(
                f"this state was not written by a BalancedSampler (sampler tag "
                f"{kind!r}); a DeterministicBatcher state carries the same size, "
                f"batch_size, seed, epoch and position keys and would otherwise load "
                f"silently, resuming the balanced arm at a position belonging to a "
                f"uniform order. Refusing rather than guessing from which keys are "
                f"present.")
        absent = [key for key in (*self.IDENTIFYING, "epoch", "position")
                  if key not in state]
        if absent:
            raise ValueError(
                f"balanced sampler state is incomplete, missing {absent}; it cannot be "
                f"confirmed to describe this sampler's weighting")
        for key in self.IDENTIFYING:
            if state[key] != getattr(self, key):
                raise ValueError(f"sampler {key} mismatch: checkpoint {state[key]} vs "
                                 f"now {getattr(self, key)}")
        epoch, position = int(state["epoch"]), int(state["position"])
        if epoch < 0 or not 0 <= position <= self.size:
            raise ValueError(f"balanced sampler state is out of range: epoch {epoch}, "
                             f"position {position} against size {self.size}")
        order = self._draw(epoch)
        self.epoch, self.position, self._order = epoch, position, order

    def report(self) -> dict[str, Any]:
        return {"distribution": {k: v for k, v in self.distribution.items()
                                 if k != "assignment"},
                "weighting": self.diagnostics,
                "reduces_to_uniform": bool(self.alpha == 0.0)}


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_MAX_EXPOSURE_RATIO",
    "DEFAULT_MIN_CELL_COUNT",
    "BOUNDARY_TOLERANCE",
    "DEFAULT_QED_BIN_WIDTH",
    "STATE_KIND",
    "BalancedSampler",
    "cell_index",
    "joint_distribution",
]
