"""Stage 20: how well can we actually SEE a within-N QED change?

WHY THIS MODULE EXISTS
----------------------
Stage 19 concluded `H2-direction` because the stage-19 read-out predicted a change of
about +0.05-0.08 in the generated embeddings while its own held-out MAE was 0.0885. That
comparison is not valid, for two independent reasons:

  1. A held-out MAE is the error of a SINGLE absolute prediction. The quantity being
     tested is the MEAN of a PAIRED difference over hundreds of molecule pairs, whose
     standard error is smaller than the single-prediction error by roughly sqrt(pairs) and
     whose per-molecule biases cancel because the same molecule appears on both sides.
     Comparing a paired mean against a single-prediction MAE is a category error, and it
     is biased towards declaring "no signal".
  2. The read-out was fitted to ABSOLUTE QED across every N. Most of the QED variance in
     ZINC250K is between-N, so most of its capacity went on a contrast that a fixed-N
     intervention cannot produce. Its sensitivity to the within-N contrast we actually
     measure was never quantified.

So this module replaces "MAE as a threshold" with three measured quantities:

  GAIN   on REAL validation molecules at FIXED N, pair a low-QED with a high-QED molecule
         and ask what the read-out predicts. The slope of predicted against true
         difference is the instrument's gain on exactly the contrast the intervention
         produces. A read-out with gain 0.2 that predicts +0.05 on generated embeddings is
         reporting an implied real difference of +0.25, not a null result.
  NULL   pair REAL molecules at the same N and the SAME QED band, where the true
         difference is ~0. The spread of the predicted difference is the instrument's
         noise floor for a paired contrast. THIS, not the MAE, is the detection threshold.
  SHIFT  a two-sample discriminability (AUC) between real and generated pooled
         embeddings. A Mahalanobis mean ratio near 1 does not establish that generated
         embeddings are on-manifold; a classifier that separates them at AUC 0.95 shows
         they are distinguishable even when their mean distance matches, which bounds what
         any read-out fitted on real embeddings can claim about them.

A second instrument is fitted for comparison: `WithinNRidgeProbe`, closed-form ridge on
within-N centred QED. It is trained on TRAINING data only (alpha selected on an inner
split of train, never on validation). It is deliberately linear, so that a paired
prediction difference is exactly `w . (f_high - f_low)` - a linear functional of the
embedding displacement stage 18 measured, with nothing hidden in a non-linearity.

Nothing here guides generation, reranks samples, or is reported as achieved QED. Every
function is a measurement.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

RIDGE_SEED = 20261024
BOOTSTRAP_SEED = 20261025
RIDGE_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
BOOTSTRAP_RESAMPLES = 2000
DEFAULT_LEVEL = 0.95


# --------------------------------------------------------------------- pooling ---
def pooled_features(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean and spread over the REAL nodes: exactly `EmbeddingQedProbe`'s pooling.

    Reproduced rather than imported so both read-outs are scored on one identical feature
    matrix. It must stay identical to `EmbeddingQedProbe.forward`, which concatenates the
    masked mean with the masked standard deviation before applying its MLP - that is what
    makes `MlpProbeReadout` an exact re-expression of the stage-19 probe and not an
    approximation of it.
    """
    weights = mask.unsqueeze(-1).to(x.dtype)
    counts = weights.sum(dim=1).clamp(min=1.0)
    mean = (x * weights).sum(dim=1) / counts
    spread = (((x - mean.unsqueeze(1)) ** 2) * weights).sum(dim=1) / counts
    return torch.cat([mean, spread.sqrt()], dim=-1)


@torch.no_grad()
def encode_pooled(autoencoder, dataset, indices: Sequence[int], *,
                  device: torch.device | str = "cpu", batch_size: int = 128
                  ) -> dict[str, np.ndarray]:
    """Pooled embeddings, QED and N for REAL molecules. Pooling is width-invariant."""
    from zinc_gnf.dataset import iter_eval_batches

    features, qed, counts = [], [], []
    for batch in iter_eval_batches(dataset, batch_size, [int(i) for i in indices]):
        batch = batch.to(device)
        nodes = autoencoder.encode(batch)
        features.append(pooled_features(nodes, batch.node_mask).detach().cpu())
        qed.append(batch.qed.detach().cpu())
        counts.append(batch.n_node.detach().cpu())
    if not features:
        raise ValueError("no molecules were encoded")
    return {
        "features": torch.cat(features).to(torch.float64).numpy(),
        "qed": torch.cat(qed).to(torch.float64).numpy(),
        "n": torch.cat(counts).to(torch.int64).numpy(),
    }


# ------------------------------------------------------------------ bootstraps ---
def _percentile_interval(samples: np.ndarray, level: float) -> tuple[float, float]:
    tail = (1.0 - float(level)) / 2.0
    return (float(np.quantile(samples, tail)), float(np.quantile(samples, 1.0 - tail)))


def paired_bootstrap_ci(values: Sequence[float], *, resamples: int = BOOTSTRAP_RESAMPLES,
                        level: float = DEFAULT_LEVEL, seed: int = BOOTSTRAP_SEED
                        ) -> dict[str, Any]:
    """Percentile CI for the MEAN of paired differences, resampling PAIRS.

    The pair is the unit of resampling because the two conditions share their latent
    noise: a molecule's two measurements are not independent observations.
    """
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                       dtype=np.float64)
    result: dict[str, Any] = {"pairs": int(array.size), "resamples": int(resamples),
                              "level": float(level)}
    if array.size == 0:
        result.update({"mean": float("nan"), "sem": float("nan"),
                       "ci_low": float("nan"), "ci_high": float("nan"),
                       "excludes_zero": False})
        return result
    result["mean"] = float(array.mean())
    result["sd"] = float(array.std(ddof=1)) if array.size > 1 else float("nan")
    result["sem"] = (float(array.std(ddof=1) / np.sqrt(array.size)) if array.size > 1
                     else float("nan"))
    if array.size == 1:
        result.update({"ci_low": float("nan"), "ci_high": float("nan"),
                       "excludes_zero": False})
        return result
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, array.size, size=(int(resamples), array.size))
    means = array[draws].mean(axis=1)
    low, high = _percentile_interval(means, level)
    result.update({"ci_low": low, "ci_high": high,
                   "excludes_zero": bool(low > 0.0 or high < 0.0),
                   "fraction_positive": float((array > 0).mean())})
    return result


def paired_difference_ci(first: Sequence[float], second: Sequence[float], *,
                         resamples: int = BOOTSTRAP_RESAMPLES,
                         level: float = DEFAULT_LEVEL, seed: int = BOOTSTRAP_SEED
                         ) -> dict[str, Any]:
    """CI for mean(first) - mean(second) when both are measured on the SAME pairs.

    One index set is drawn and applied to both arrays, so the correlation between the two
    decoders' responses on the same molecule is preserved instead of being thrown away.
    """
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired_difference_ci needs two equally long paired arrays")
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    result: dict[str, Any] = {"pairs": int(a.size), "resamples": int(resamples),
                              "level": float(level)}
    if a.size == 0:
        result.update({"mean_first": float("nan"), "mean_second": float("nan"),
                       "difference": float("nan"), "ci_low": float("nan"),
                       "ci_high": float("nan"), "excludes_zero": False})
        return result
    result["mean_first"] = float(a.mean())
    result["mean_second"] = float(b.mean())
    result["difference"] = float(a.mean() - b.mean())
    if a.size == 1:
        result.update({"ci_low": float("nan"), "ci_high": float("nan"),
                       "excludes_zero": False})
        return result
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, a.size, size=(int(resamples), a.size))
    differences = a[draws].mean(axis=1) - b[draws].mean(axis=1)
    low, high = _percentile_interval(differences, level)
    result.update({"ci_low": low, "ci_high": high,
                   "excludes_zero": bool(low > 0.0 or high < 0.0)})
    return result


def group_ratio_bootstrap_ci(low_values: Sequence[float], low_truth: Sequence[float],
                             high_values: Sequence[float], high_truth: Sequence[float], *,
                             resamples: int = BOOTSTRAP_RESAMPLES,
                             level: float = DEFAULT_LEVEL, seed: int = BOOTSTRAP_SEED
                             ) -> dict[str, Any]:
    """CI for (mean(high_values) - mean(low_values)) / (mean(high_truth) - mean(low_truth)).

    The two QED groups are different molecules, so they are resampled INDEPENDENTLY; the
    measured and true values of one molecule are resampled TOGETHER, because they belong
    to the same molecule.
    """
    lv = np.asarray(low_values, dtype=np.float64)
    lt = np.asarray(low_truth, dtype=np.float64)
    hv = np.asarray(high_values, dtype=np.float64)
    ht = np.asarray(high_truth, dtype=np.float64)
    low_keep = np.isfinite(lv) & np.isfinite(lt)
    high_keep = np.isfinite(hv) & np.isfinite(ht)
    lv, lt, hv, ht = lv[low_keep], lt[low_keep], hv[high_keep], ht[high_keep]
    result: dict[str, Any] = {"low_molecules": int(lv.size), "high_molecules": int(hv.size),
                              "resamples": int(resamples), "level": float(level)}
    if lv.size == 0 or hv.size == 0:
        result.update({"true_difference": float("nan"), "measured_difference": float("nan"),
                       "ratio": float("nan"), "ci_low": float("nan"),
                       "ci_high": float("nan")})
        return result
    true_difference = float(ht.mean() - lt.mean())
    measured = float(hv.mean() - lv.mean())
    result["true_difference"] = true_difference
    result["measured_difference"] = measured
    result["ratio"] = (measured / true_difference) if true_difference else float("nan")
    if lv.size < 2 or hv.size < 2 or not true_difference:
        result.update({"ci_low": float("nan"), "ci_high": float("nan")})
        return result
    rng = np.random.default_rng(int(seed))
    low_draws = rng.integers(0, lv.size, size=(int(resamples), lv.size))
    high_draws = rng.integers(0, hv.size, size=(int(resamples), hv.size))
    numerator = hv[high_draws].mean(axis=1) - lv[low_draws].mean(axis=1)
    denominator = ht[high_draws].mean(axis=1) - lt[low_draws].mean(axis=1)
    usable = np.abs(denominator) > 1e-12
    if not usable.any():
        result.update({"ci_low": float("nan"), "ci_high": float("nan")})
        return result
    low, high = _percentile_interval(numerator[usable] / denominator[usable], level)
    result.update({"ci_low": low, "ci_high": high})
    return result


# ----------------------------------------------------------------- the read-outs ---
def _solve_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Closed-form ridge with an unpenalised intercept."""
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    centred = x - x_mean
    gram = centred.T @ centred + float(alpha) * np.eye(x.shape[1])
    weights = np.linalg.solve(gram, centred.T @ (y - y_mean))
    return weights, y_mean - float(x_mean @ weights)


class WithinNRidgeProbe:
    """Closed-form ridge read-out fitted to WITHIN-N centred QED.

    Three deliberate properties:
      * the target is `qed - mean(qed | N)` computed on TRAIN, so none of the model's
        capacity is spent on the between-N variation that a fixed-N intervention cannot
        produce - which is the contrast the stage-19 read-out was optimised for;
      * it is convex and closed-form, so its numbers carry no optimiser variance and are
        reproducible to floating point;
      * it is LINEAR, so a paired prediction difference is exactly
        `w . (f_high - f_low)` - the predicted change is a linear functional of the
        embedding displacement, which is precisely the quantity stage 18 measured.

    `alpha` is chosen on an inner split of the TRAINING molecules. Validation data is
    never used for fitting or selection, only for reporting.
    """

    def __init__(self, weights: np.ndarray, intercept: float, feature_mean: np.ndarray,
                 feature_std: np.ndarray, offsets: dict[int, float], global_offset: float,
                 alpha: float, selection: list[dict[str, Any]]) -> None:
        self.weights = weights
        self.intercept = float(intercept)
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.offsets = {int(k): float(v) for k, v in offsets.items()}
        self.global_offset = float(global_offset)
        self.alpha = float(alpha)
        self.selection = selection

    name = "within_n_ridge"

    # -- fitting ------------------------------------------------------------
    @staticmethod
    def _offsets(qed: np.ndarray, n: np.ndarray) -> tuple[dict[int, float], float]:
        offsets = {}
        for value in np.unique(n):
            offsets[int(value)] = float(qed[n == value].mean())
        return offsets, float(qed.mean())

    @classmethod
    def fit(cls, features: np.ndarray, qed: np.ndarray, n: np.ndarray, *,
            alphas: Sequence[float] = RIDGE_ALPHAS, seed: int = RIDGE_SEED,
            inner_fraction: float = 0.2) -> "WithinNRidgeProbe":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(qed, dtype=np.float64)
        counts = np.asarray(n, dtype=np.int64)
        if x.ndim != 2 or x.shape[0] != y.size or y.size != counts.size:
            raise ValueError("features, qed and n must describe the same molecules")
        if x.shape[0] < 2:
            raise ValueError("the ridge read-out needs at least two training molecules")

        selection: list[dict[str, Any]] = []
        alpha = 1.0
        rng = np.random.default_rng(int(seed))
        order = rng.permutation(x.shape[0])
        held = int(round(float(inner_fraction) * x.shape[0]))
        if held >= 1 and x.shape[0] - held >= 2:
            inner_val, inner_train = order[:held], order[held:]
            inner_offsets, inner_global = cls._offsets(y[inner_train], counts[inner_train])
            mean = x[inner_train].mean(axis=0)
            std = np.maximum(x[inner_train].std(axis=0), 1e-8)
            scaled_train = (x[inner_train] - mean) / std
            scaled_val = (x[inner_val] - mean) / std
            target_train = y[inner_train] - np.asarray(
                [inner_offsets.get(int(c), inner_global) for c in counts[inner_train]])
            target_val = y[inner_val] - np.asarray(
                [inner_offsets.get(int(c), inner_global) for c in counts[inner_val]])
            best = None
            for candidate in alphas:
                weights, intercept = _solve_ridge(scaled_train, target_train, candidate)
                error = float(np.abs(scaled_val @ weights + intercept - target_val).mean())
                selection.append({"alpha": float(candidate), "inner_validation_mae": error})
                if best is None or error < best[0]:
                    best = (error, float(candidate))
            if best is not None:
                alpha = best[1]

        offsets, global_offset = cls._offsets(y, counts)
        mean = x.mean(axis=0)
        std = np.maximum(x.std(axis=0), 1e-8)
        target = y - np.asarray([offsets.get(int(c), global_offset) for c in counts])
        weights, intercept = _solve_ridge((x - mean) / std, target, alpha)
        return cls(weights, intercept, mean, std, offsets, global_offset, alpha, selection)

    # -- the read-out interface --------------------------------------------
    def contrast(self, features: np.ndarray) -> np.ndarray:
        """The within-N QED deviation. Paired differences at fixed N use only this."""
        scaled = (np.asarray(features, dtype=np.float64) - self.feature_mean) / self.feature_std
        return scaled @ self.weights + self.intercept

    def absolute(self, features: np.ndarray, n: Sequence[int] | None = None) -> np.ndarray:
        """Within-N deviation plus the TRAIN mean QED at that N."""
        values = self.contrast(features)
        if n is None:
            return values + self.global_offset
        offsets = np.asarray([self.offsets.get(int(c), self.global_offset) for c in n])
        return values + offsets

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "alpha": self.alpha, "features": int(self.weights.size),
                "weight_norm": float(np.linalg.norm(self.weights)),
                "alpha_selection": self.selection,
                "note": ("linear and closed form: a paired prediction difference is exactly "
                         "w . (f_high - f_low), and alpha was chosen on an inner split of "
                         "TRAIN only")}


class MlpProbeReadout:
    """The stage-19 `EmbeddingQedProbe`, called on ALREADY POOLED features.

    `EmbeddingQedProbe.forward` pools to `cat([mean, std])` and applies `.net`, so
    applying `.net` to `pooled_features(...)` is the SAME function, not an approximation.
    The stage-19 instrument is therefore carried into stage 20 unmodified, while both
    read-outs get scored on one identical feature matrix.
    """

    name = "stage19_mlp"

    def __init__(self, probe) -> None:
        self.probe = probe

    def _device(self) -> torch.device:
        try:
            return next(self.probe.parameters()).device
        except StopIteration:                                     # pragma: no cover
            return torch.device("cpu")

    def contrast(self, features: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(np.asarray(features, dtype=np.float32),
                                 device=self._device())
        with torch.no_grad():
            return self.probe.net(tensor).squeeze(-1).cpu().numpy().astype(np.float64)

    def absolute(self, features: np.ndarray, n: Sequence[int] | None = None) -> np.ndarray:
        return self.contrast(features)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name,
                "note": ("the stage-19 read-out, unmodified; evaluated through .net on "
                         "pooled features, which is exactly its forward pass")}


# ------------------------------------------------------------------- scoring ---
def probe_scores(readout, features: np.ndarray, qed: np.ndarray,
                 n: np.ndarray) -> dict[str, Any]:
    predicted = np.asarray(readout.absolute(features, n), dtype=np.float64)
    truth = np.asarray(qed, dtype=np.float64)
    correlation = (float(np.corrcoef(predicted, truth)[0, 1])
                   if predicted.size > 1 and predicted.std() > 0 and truth.std() > 0
                   else float("nan"))
    return {"molecules": int(truth.size),
            "mae": float(np.abs(predicted - truth).mean()),
            "rmse": float(np.sqrt(((predicted - truth) ** 2).mean())),
            "correlation": correlation,
            "predicted_std": float(predicted.std()),
            "true_std": float(truth.std())}


def stratified_scores(readout, features: np.ndarray, qed: np.ndarray, n: np.ndarray, *,
                      n_values: Sequence[int],
                      qed_edges: Sequence[float] = (0.0, 0.5, 0.7, 1.0)
                      ) -> dict[str, Any]:
    """Accuracy by N and by QED band: an aggregate MAE can hide a blind spot."""
    features = np.asarray(features, dtype=np.float64)
    qed = np.asarray(qed, dtype=np.float64)
    n = np.asarray(n, dtype=np.int64)
    by_n: dict[str, Any] = {}
    for value in n_values:
        mask = n == int(value)
        by_n[str(int(value))] = (probe_scores(readout, features[mask], qed[mask], n[mask])
                                 if mask.sum() >= 2 else {"molecules": int(mask.sum())})
    by_qed: dict[str, Any] = {}
    edges = list(qed_edges)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (qed >= low) & (qed < high)
        label = f"{low:g}-{high:g}"
        by_qed[label] = (probe_scores(readout, features[mask], qed[mask], n[mask])
                         if mask.sum() >= 2 else {"molecules": int(mask.sum())})
    return {"by_n": by_n, "by_qed_band": by_qed,
            "note": "every cell states its own molecule count as the denominator"}


def calibration(readout, features: np.ndarray, qed: np.ndarray, n: np.ndarray, *,
                bins: int = 10) -> dict[str, Any]:
    """Is the read-out calibrated, or systematically compressed?

    A read-out that predicts the right ORDER but a compressed RANGE has a calibration
    slope below 1. That matters here: such a read-out would under-report a paired change
    by exactly that factor, which is why the gain measurement below is necessary.
    """
    predicted = np.asarray(readout.absolute(features, n), dtype=np.float64)
    truth = np.asarray(qed, dtype=np.float64)
    result: dict[str, Any] = {"molecules": int(truth.size)}
    if truth.size < 2 or predicted.std() <= 0:
        result["note"] = "too few molecules, or a constant prediction, to calibrate"
        return result
    slope, intercept = np.polyfit(predicted, truth, 1)
    result.update({
        "slope_of_truth_on_prediction": float(slope),
        "intercept": float(intercept),
        "dispersion_ratio": float(predicted.std() / truth.std()) if truth.std() > 0
        else float("nan"),
        "mean_bias": float((predicted - truth).mean()),
    })
    order = np.argsort(predicted)
    chunks = np.array_split(order, min(int(bins), max(1, truth.size // 2)))
    table, errors, weights = [], [], []
    for chunk in chunks:
        if chunk.size == 0:
            continue
        table.append({"molecules": int(chunk.size),
                      "mean_prediction": float(predicted[chunk].mean()),
                      "mean_truth": float(truth[chunk].mean())})
        errors.append(abs(float(predicted[chunk].mean() - truth[chunk].mean())))
        weights.append(chunk.size)
    result["bins"] = table
    result["expected_calibration_error"] = (
        float(np.average(errors, weights=weights)) if errors else float("nan"))
    result["note"] = ("a slope below 1 means the read-out compresses the QED range, so a "
                      "paired change it reports is an UNDER-estimate by that factor")
    return result


# -------------------------------------------- the contrasts that set the scale ---
def _band(qed: np.ndarray, n: np.ndarray, target_n: int, centre: float,
          window: float) -> np.ndarray:
    return np.flatnonzero((n == int(target_n)) & (np.abs(qed - float(centre)) <= window))


def _pair_up(first: np.ndarray, second: np.ndarray, *, max_pairs: int,
             seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    left = first.copy()
    right = second.copy()
    rng.shuffle(left)
    rng.shuffle(right)
    size = min(left.size, right.size, int(max_pairs))
    return left[:size], right[:size]


def within_n_reference_contrast(readout, features: np.ndarray, qed: np.ndarray,
                                n: np.ndarray, *, target_n: int, qed_low: float,
                                qed_high: float, window: float = 0.05,
                                max_pairs: int = 256, seed: int = BOOTSTRAP_SEED,
                                resamples: int = BOOTSTRAP_RESAMPLES,
                                level: float = DEFAULT_LEVEL) -> dict[str, Any]:
    """THE GAIN. On REAL molecules at fixed N, what does a true QED gap look like?

    This is the measurement stage 19 lacked. Pair a real low-QED molecule with a real
    high-QED molecule at the SAME N and read the predicted difference. The ratio of
    predicted to true difference is the instrument's gain on exactly the contrast the
    paired intervention produces, so a change observed on generated embeddings can be
    converted into an implied real QED difference instead of being compared against an
    unrelated single-prediction error.
    """
    features = np.asarray(features, dtype=np.float64)
    qed = np.asarray(qed, dtype=np.float64)
    n = np.asarray(n, dtype=np.int64)
    low_pool = _band(qed, n, target_n, qed_low, window)
    high_pool = _band(qed, n, target_n, qed_high, window)
    result: dict[str, Any] = {
        "n": int(target_n), "qed_low": float(qed_low), "qed_high": float(qed_high),
        "window": float(window),
        "support": {"low_molecules": int(low_pool.size), "high_molecules": int(high_pool.size),
                    "molecules_at_this_n": int((n == int(target_n)).sum())},
    }
    if low_pool.size == 0 or high_pool.size == 0:
        result["note"] = "no real molecules in one of the two QED bands at this N"
        return result
    left, right = _pair_up(low_pool, high_pool, max_pairs=max_pairs, seed=seed)
    if left.size == 0:
        result["note"] = "no pairs could be formed"
        return result
    true_difference = qed[right] - qed[left]
    predicted = np.asarray(readout.contrast(features[right]), dtype=np.float64) - \
        np.asarray(readout.contrast(features[left]), dtype=np.float64)
    result["pairs"] = int(left.size)
    result["true_difference"] = paired_bootstrap_ci(true_difference, resamples=resamples,
                                                    level=level, seed=seed)
    result["predicted_difference"] = paired_bootstrap_ci(predicted, resamples=resamples,
                                                         level=level, seed=seed)
    denominator = float(true_difference.mean())
    result["gain"] = (float(predicted.mean() / denominator) if denominator else float("nan"))
    if left.size > 1 and denominator:
        rng = np.random.default_rng(int(seed) + 1)
        draws = rng.integers(0, left.size, size=(int(resamples), left.size))
        ratios = predicted[draws].mean(axis=1) / true_difference[draws].mean(axis=1)
        finite = ratios[np.isfinite(ratios)]
        if finite.size:
            low, high = _percentile_interval(finite, level)
            result["gain_ci_low"], result["gain_ci_high"] = low, high
            result["gain_excludes_zero"] = bool(low > 0.0 or high < 0.0)
    if left.size > 1 and predicted.std() > 0 and true_difference.std() > 0:
        result["per_pair_correlation"] = float(np.corrcoef(predicted, true_difference)[0, 1])
    result["note"] = ("gain is predicted / true difference on REAL molecules at this N. A "
                      "gain well below 1 means the read-out UNDER-reports any paired "
                      "change, generated ones included.")
    return result


def null_contrast(readout, features: np.ndarray, qed: np.ndarray, n: np.ndarray, *,
                  target_n: int, qed_centre: float, window: float = 0.05,
                  max_pairs: int = 256, seed: int = BOOTSTRAP_SEED,
                  resamples: int = BOOTSTRAP_RESAMPLES,
                  level: float = DEFAULT_LEVEL) -> dict[str, Any]:
    """THE NOISE FLOOR. Pairs of REAL molecules at the same N and the SAME QED band.

    The true difference here is ~0, so whatever the read-out reports is its own noise on a
    paired contrast of this size. The standard deviation of that is the detection
    threshold for a paired mean - NOT the single-prediction held-out MAE, which is larger
    by roughly sqrt(pairs) and includes per-molecule biases that cancel in a pair.
    """
    features = np.asarray(features, dtype=np.float64)
    qed = np.asarray(qed, dtype=np.float64)
    n = np.asarray(n, dtype=np.int64)
    pool = _band(qed, n, target_n, qed_centre, window)
    result: dict[str, Any] = {"n": int(target_n), "qed_centre": float(qed_centre),
                              "window": float(window),
                              "support": {"molecules": int(pool.size)}}
    if pool.size < 4:
        result["note"] = "too few real molecules in this band to build a null"
        return result
    rng = np.random.default_rng(int(seed))
    shuffled = pool.copy()
    rng.shuffle(shuffled)
    half = shuffled.size // 2
    left, right = shuffled[:half], shuffled[half:2 * half]
    if int(max_pairs) < left.size:
        left, right = left[:int(max_pairs)], right[:int(max_pairs)]
    predicted = np.asarray(readout.contrast(features[right]), dtype=np.float64) - \
        np.asarray(readout.contrast(features[left]), dtype=np.float64)
    true_difference = qed[right] - qed[left]
    result["pairs"] = int(left.size)
    result["true_difference_mean"] = float(true_difference.mean())
    result["predicted_difference"] = paired_bootstrap_ci(predicted, resamples=resamples,
                                                         level=level, seed=seed)
    result["null_sd"] = (float(predicted.std(ddof=1)) if predicted.size > 1
                         else float("nan"))
    if predicted.size > 1:
        low, high = _percentile_interval(predicted, level)
        result["individual_pair_interval"] = [low, high]
    result["note"] = ("the spread of this quantity, not the held-out MAE, is what a paired "
                      "mean has to clear to count as detected")
    return result


def interpret_generated_change(values: Sequence[float], *, null: dict[str, Any],
                               reference: dict[str, Any], requested: float,
                               resamples: int = BOOTSTRAP_RESAMPLES,
                               level: float = DEFAULT_LEVEL,
                               seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Is the observed paired change resolvable, and what real QED gap does it imply?

    Resolution is decided by comparing the observed paired-mean CI with the NULL paired
    mean CI measured on real same-band molecules. The implied real difference divides by
    the measured gain, so a compressed read-out is corrected rather than believed.
    """
    observed = paired_bootstrap_ci(values, resamples=resamples, level=level, seed=seed)
    result: dict[str, Any] = {"observed": observed, "requested_change": float(requested)}
    null_block = (null or {}).get("predicted_difference") or {}
    null_low, null_high = null_block.get("ci_low"), null_block.get("ci_high")
    separated = None
    if all(v is not None and np.isfinite(v) for v in
           (observed.get("ci_low"), observed.get("ci_high"), null_low, null_high)):
        separated = bool(observed["ci_low"] > null_high or observed["ci_high"] < null_low)
    result["null_mean_ci"] = [null_low, null_high]
    result["separated_from_null"] = separated
    null_sd = (null or {}).get("null_sd")
    if null_sd is not None and np.isfinite(null_sd) and null_sd > 0 \
            and np.isfinite(observed.get("mean", float("nan"))):
        result["effect_size_vs_null_sd"] = float(observed["mean"] / null_sd)
    gain = (reference or {}).get("gain")
    if gain is not None and np.isfinite(gain) and abs(gain) > 1e-9:
        result["gain"] = float(gain)
        result["implied_real_qed_difference"] = float(observed["mean"] / gain)
        result["implied_fraction_of_request"] = (
            float(observed["mean"] / gain / requested) if requested else float("nan"))
        bounds = []
        for endpoint in (reference.get("gain_ci_low"), reference.get("gain_ci_high")):
            if endpoint is not None and np.isfinite(endpoint) and abs(endpoint) > 1e-9:
                bounds.append(float(observed["mean"] / endpoint))
        if len(bounds) == 2:
            result["implied_real_qed_difference_range"] = [min(bounds), max(bounds)]
    if separated is None:
        result["status"] = "no_null_available"
    elif separated:
        result["status"] = "resolved"
    else:
        result["status"] = "not_resolved"
    result["note"] = ("status 'not_resolved' means THIS INSTRUMENT cannot see the change at "
                      "this sample size. It is a statement about the instrument, not "
                      "evidence that the embeddings carry no QED information.")
    return result


# ------------------------------------------------------- the distribution shift ---
def distribution_discriminability(real: np.ndarray, generated: np.ndarray, *,
                                  seed: int = BOOTSTRAP_SEED,
                                  holdout_fraction: float = 0.3,
                                  alpha: float = 1.0) -> dict[str, Any]:
    """Can a cheap classifier tell real pooled embeddings from generated ones?

    A Mahalanobis mean-distance ratio near 1 - stage 19's number - does NOT establish
    that generated embeddings are on-manifold: two clouds can share a mean distance and
    still be perfectly separable. A held-out AUC near 0.5 means indistinguishable under
    this read-out's own features; an AUC near 1 means every read-out and decoder fitted on
    real embeddings is extrapolating, whatever the Mahalanobis ratio says.

    Closed-form ridge on +-1 labels, fitted on a training part and scored on a held-out
    part, so the AUC is not an in-sample artefact.
    """
    from scipy.stats import rankdata

    real = np.asarray(real, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    result: dict[str, Any] = {"real_molecules": int(real.shape[0]),
                              "generated_molecules": int(generated.shape[0])}
    if real.ndim != 2 or generated.ndim != 2 or real.shape[1] != generated.shape[1]:
        raise ValueError("real and generated features must have the same width")
    if real.shape[0] < 8 or generated.shape[0] < 8:
        result["note"] = "too few molecules on one side for a held-out discriminability"
        return result
    rng = np.random.default_rng(int(seed))

    def split(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = rng.permutation(values.shape[0])
        cut = max(1, int(round(float(holdout_fraction) * values.shape[0])))
        return values[order[cut:]], values[order[:cut]]

    real_train, real_test = split(real)
    fake_train, fake_test = split(generated)
    x = np.vstack([real_train, fake_train])
    y = np.concatenate([np.full(real_train.shape[0], -1.0),
                        np.full(fake_train.shape[0], 1.0)])
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1e-8)
    weights, intercept = _solve_ridge((x - mean) / std, y, alpha)
    scores = np.concatenate([((real_test - mean) / std) @ weights + intercept,
                             ((fake_test - mean) / std) @ weights + intercept])
    labels = np.concatenate([np.zeros(real_test.shape[0]), np.ones(fake_test.shape[0])])
    ranks = rankdata(scores)
    positives = float(labels.sum())
    negatives = float(labels.size - positives)
    auc = ((ranks[labels == 1].sum() - positives * (positives + 1) / 2.0)
           / (positives * negatives)) if positives and negatives else float("nan")
    result.update({
        "held_out_auc": float(auc),
        "held_out_real": int(real_test.shape[0]),
        "held_out_generated": int(fake_test.shape[0]),
        "note": ("0.5 means indistinguishable under these features; well above 0.5 means a "
                 "read-out or decoder fitted on real embeddings is extrapolating on the "
                 "generated ones, however close their Mahalanobis means are"),
    })
    return result


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "DEFAULT_LEVEL",
    "MlpProbeReadout",
    "RIDGE_ALPHAS",
    "RIDGE_SEED",
    "WithinNRidgeProbe",
    "calibration",
    "distribution_discriminability",
    "encode_pooled",
    "group_ratio_bootstrap_ci",
    "interpret_generated_change",
    "null_contrast",
    "paired_bootstrap_ci",
    "paired_difference_ci",
    "pooled_features",
    "probe_scores",
    "stratified_scores",
    "within_n_reference_contrast",
]
