"""Latent covariance geometry and DIRECTION-RESOLVED decoder robustness (diagnostic S2).

Why this exists. `reconstruction_analysis.embedding_robustness` perturbs embeddings
isotropically in units of each dimension's standard deviation. Because the embedding
covariance is strongly anisotropic (participation ratio about 5 of 64 on earlier
measurements), such a perturbation lands almost entirely in the few high-variance
directions, so a tolerance measured that way says nothing about an error along a
low-variance direction - and a flow's sampling error is not aligned with the data's
variance. This module measures the two separately.

Protocol, fixed on purpose:

  * the eigenbasis and the variance spectrum are estimated on TRAINING embeddings only;
  * every robustness number is then measured on VALIDATION embeddings, which never
    entered the basis estimate (no leakage between fitting the geometry and testing
    sensitivity);
  * the test split is never read by anything in this module.

Perturbation magnitudes are defined in two clearly separated ways, because "the same
size" means different things here:

  absolute  coefficient std = magnitude * (mean per-dimension std of the embeddings).
            The SAME expected displacement length for every direction group, so a
            difference between groups is a statement about direction alone. This is the
            unit `noise_sigma` already uses, so `isotropic` rows at magnitude 0.05 are
            directly comparable to the 92.2%-exact figure measured earlier.

  relative  coefficient std = magnitude * sqrt(eigenvalue) of each direction used.
            Displacement scaled to the data's own spread in that direction, i.e. "how far
            off relative to how far the data itself moves there". Matched in relative
            terms, deliberately NOT matched in euclidean length.

Direction groups: `top` (the k largest eigenvalues), `bottom` (the k smallest),
`isotropic` (all 64 dimensions, the previous behaviour). Every group is reported
alongside an unperturbed baseline (magnitude 0) computed by the same code path, so the
comparison is like for like.

Nothing here trains, writes or modifies a model: the autoencoder is used in eval mode
under `torch.no_grad()`.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import QED
import torch

from zinc_gnf.chemistry import canonical_connectivity_smiles
from zinc_gnf.chemistry_charge_aware import graph_to_mol_charge_aware
from zinc_gnf.dataset import MolecularGraphDataset, iter_eval_batches
from zinc_gnf.decoding_charge_aware import (
    degree_constrained_bond_decode,
    valence_aware_bond_decode_typed,
)
from zinc_gnf.reconstruction_analysis import _real_node_embeddings

DIRECTION_GROUPS = ("isotropic", "top", "bottom")
MAGNITUDE_MODES = ("absolute", "relative")
DEFAULT_MAGNITUDES = (0.0, 0.05, 0.1, 0.2, 0.4)


# ---------------------------------------------------------------------------
# Spectrum.
# ---------------------------------------------------------------------------
@torch.no_grad()
def node_embeddings(model, dataset: MolecularGraphDataset, *, indices: Sequence[int] | None = None,
                    device: str | torch.device = "cpu", batch_size: int = 128) -> torch.Tensor:
    """`[total real nodes, D]` embeddings of `indices` (padding rows dropped)."""
    was_training = model.training
    model.eval()
    order = list(range(len(dataset))) if indices is None else [int(i) for i in indices]
    embeddings = _real_node_embeddings(model, dataset, order, device, seed=0, batch_size=batch_size)
    if was_training:
        model.train()
    return embeddings


def covariance_spectrum(embeddings: torch.Tensor, *, correlation_pairs: int = 10
                        ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    """Eigen-decomposition of the embedding covariance.

    Returns `(summary, eigenvalues, eigenvectors)` with eigenvalues DESCENDING and
    `eigenvectors[:, i]` the direction of `eigenvalues[i]`. The summary is JSON-safe;
    the tensors are what the perturbation uses.

    Effective rank is reported as the number of leading directions needed to reach a
    share of the total variance, which is the quantity that decides whether a diagonal
    whitening can condition this representation: diagonal whitening equalises the 64
    marginals but cannot rotate, so a spectrum that collapses after a handful of
    directions stays collapsed after whitening.
    """
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError(f"need [nodes, dim] with at least two nodes, got {tuple(embeddings.shape)}")
    data = embeddings.to(torch.float64)
    nodes, dimension = data.shape
    mean = data.mean(0)
    centred = data - mean
    covariance = (centred.T @ centred) / (nodes - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)      # ascending
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues, eigenvectors = eigenvalues[order].clamp(min=0.0), eigenvectors[:, order]

    total = float(eigenvalues.sum())
    shares = (eigenvalues / max(total, 1e-300)).tolist()
    cumulative = np.cumsum(shares)
    participation = float(eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum().clamp(min=1e-300))
    per_dimension_std = data.std(0)

    # Correlations between raw dimensions: what a DIAGONAL whitening leaves untouched.
    standard = per_dimension_std.clamp(min=1e-12)
    correlation = covariance / (standard.unsqueeze(0) * standard.unsqueeze(1))
    off_diagonal = correlation - torch.diag(torch.diag(correlation))
    flat = off_diagonal.abs().flatten()
    count = min(int(correlation_pairs) * 2, flat.numel())
    strongest = torch.topk(flat, count).indices if count else torch.empty(0, dtype=torch.long)
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for index in strongest.tolist():
        i, j = divmod(index, dimension)
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({"dimensions": [key[0], key[1]], "correlation": float(correlation[i, j])})
        if len(pairs) >= correlation_pairs:
            break

    def rank_for(share: float) -> int:
        return int(np.searchsorted(cumulative, share) + 1)

    summary = {
        "nodes": int(nodes),
        "dimension": int(dimension),
        "total_variance": total,
        "participation_ratio": participation,
        "effective_rank_90": rank_for(0.90),
        "effective_rank_95": rank_for(0.95),
        "effective_rank_99": rank_for(0.99),
        "effective_rank_999": rank_for(0.999),
        "variance_share_top1": shares[0] if shares else float("nan"),
        "variance_share_top5": float(cumulative[4]) if dimension >= 5 else float("nan"),
        "variance_share_top10": float(cumulative[9]) if dimension >= 10 else float("nan"),
        "variance_share_bottom32": float(sum(shares[dimension - 32:])) if dimension >= 32 else float("nan"),
        "eigenvalues": [float(v) for v in eigenvalues],
        "eigenvalue_shares": [float(v) for v in shares],
        "eigenvalue_max": float(eigenvalues[0]),
        "eigenvalue_min": float(eigenvalues[-1]),
        "eigenvalue_median": float(eigenvalues.median()),
        "condition_number_max_over_min": float(eigenvalues[0] / eigenvalues[-1].clamp(min=1e-300)),
        "condition_number_max_over_median": float(eigenvalues[0] / eigenvalues.median().clamp(min=1e-300)),
        "per_dimension_std_mean": float(per_dimension_std.mean()),
        "per_dimension_std_min": float(per_dimension_std.min()),
        "per_dimension_std_max": float(per_dimension_std.max()),
        "dimensions_below_whitening_floor_0p01": int((per_dimension_std < 0.01).sum()),
        "mean_norm": float(data.norm(dim=1).mean()),
        "strongest_correlations": pairs,
        "max_absolute_correlation": float(off_diagonal.abs().max()) if dimension > 1 else 0.0,
        "mean_absolute_correlation": float(off_diagonal.abs().sum() / (dimension * (dimension - 1)))
        if dimension > 1 else 0.0,
    }
    return summary, eigenvalues, eigenvectors


# ---------------------------------------------------------------------------
# Direction-resolved perturbation.
# ---------------------------------------------------------------------------
def direction_indices(group: str, dimension: int, k: int) -> list[int]:
    """Which eigen-directions a group uses (eigenvalues assumed descending)."""
    if group not in DIRECTION_GROUPS:
        raise ValueError(f"unknown direction group {group!r}; known: {list(DIRECTION_GROUPS)}")
    if not 1 <= k <= dimension:
        raise ValueError(f"k must lie in 1..{dimension}, got {k}")
    if group == "isotropic":
        return list(range(dimension))
    if group == "top":
        return list(range(k))
    return list(range(dimension - k, dimension))


def perturbation(embeddings: torch.Tensor, eigenvalues: torch.Tensor, eigenvectors: torch.Tensor, *,
                 group: str, magnitude: float, mode: str, k: int,
                 generator: torch.Generator | None = None,
                 scale_unit: float | None = None) -> torch.Tensor:
    """Gaussian displacement confined to a group of eigen-directions.

    `absolute`: every used direction gets coefficient std `magnitude * scale_unit`
    (default `scale_unit` = mean per-dimension std), so `top` and `bottom` with the same
    `k` have the same expected displacement length.
    `relative`: direction i gets `magnitude * sqrt(eigenvalue_i)`.
    """
    if mode not in MAGNITUDE_MODES:
        raise ValueError(f"unknown magnitude mode {mode!r}; known: {list(MAGNITUDE_MODES)}")
    if magnitude == 0.0:
        return torch.zeros_like(embeddings)
    used = direction_indices(group, embeddings.shape[1], k)
    basis = eigenvectors[:, used].to(embeddings.dtype)                   # [D, m]
    coefficients = torch.randn((embeddings.shape[0], len(used)), dtype=embeddings.dtype,
                               device=embeddings.device, generator=generator)
    if mode == "absolute":
        unit = scale_unit if scale_unit is not None else 1.0
        coefficients = coefficients * (float(magnitude) * float(unit))
    else:
        spread = eigenvalues[used].clamp(min=0).sqrt().to(embeddings.dtype)
        coefficients = coefficients * (float(magnitude) * spread).unsqueeze(0)
    return coefficients @ basis.T


# ---------------------------------------------------------------------------
# Decoding a perturbed embedding back to a molecule.
# ---------------------------------------------------------------------------
def _decode_and_score(model, dataset: MolecularGraphDataset, indices: Sequence[int],
                      displacement: torch.Tensor, device, batch_size: int) -> dict[str, Any]:
    """Decode perturbed embeddings and score them against the true molecules."""
    has_degree = bool(model.config.degree_head)
    totals = {name: {"exact": 0, "valid": 0, "sanitizable": 0, "qed": []}
              for name in (("constrained", "degree") if has_degree else ("constrained",))}
    degree_correct = degree_total = fallbacks = 0
    reasons: Counter = Counter()
    molecules = 0
    consumed = 0
    offset = 0
    order = [int(i) for i in indices]
    for batch in iter_eval_batches(dataset, batch_size, order):
        members = order[consumed:consumed + batch.batch_size]
        consumed += batch.batch_size
        batch = batch.to(device)
        with torch.no_grad():
            nodes = model.encode(batch)
            real = batch.node_mask
            count = int(real.sum())
            shift = displacement[offset:offset + count].to(device)
            offset += count
            perturbed = nodes.clone()
            perturbed[real] = perturbed[real] + shift
            outputs = model.decode(perturbed, batch, teacher_forcing=False)
        atoms_all = outputs["atom_logits"].argmax(-1).cpu().numpy()
        probabilities_all = torch.softmax(outputs["bond_logits"], dim=-1).cpu().numpy()
        degrees_all = (outputs["degree_logits"].argmax(-1).cpu().numpy() if has_degree else None)
        for position, index in enumerate(members):
            n = int(batch.n_node[position])
            atoms = atoms_all[position, :n]
            probabilities = probabilities_all[position, :n, :n]
            true_atoms, true_bonds = dataset.true_graph(index)
            reference = dataset.smiles[index]
            molecules += 1
            candidates: list[tuple[str, np.ndarray, dict[str, Any]]] = []
            bonds, info = valence_aware_bond_decode_typed(atoms, probabilities)
            candidates.append(("constrained", bonds, info))
            if has_degree:
                predicted = degrees_all[position, :n]
                true_degree = (true_bonds > 0).sum(1).clip(max=6)
                degree_correct += int((predicted == true_degree).sum())
                degree_total += n
                bonds, info = degree_constrained_bond_decode(atoms, probabilities, predicted)
                if info.get("fallback"):
                    fallbacks += 1
                    reasons[info.get("fallback_reason")] += 1
                candidates.append(("degree", bonds, info))
            for name, assembled, _ in candidates:
                entry = totals[name]
                try:
                    with rdBase.BlockLogs():
                        molecule = graph_to_mol_charge_aware(atoms, assembled)
                except Exception:  # noqa: BLE001
                    continue
                entry["sanitizable"] += 1
                try:
                    connected = len(Chem.GetMolFrags(molecule)) == 1
                except Exception:  # noqa: BLE001
                    continue
                if not connected:
                    continue
                entry["valid"] += 1
                try:
                    smiles = canonical_connectivity_smiles(molecule)
                    entry["qed"].append(float(QED.qed(molecule)))
                except Exception:  # noqa: BLE001
                    continue
                entry["exact"] += int(smiles == reference)

    def rate(numerator, denominator):
        return float(numerator / denominator) if denominator else float("nan")

    result: dict[str, Any] = {"molecules": molecules}
    for name, entry in totals.items():
        result[name] = {
            "exact_molecule_match": rate(entry["exact"], molecules),
            "strict_validity": rate(entry["valid"], molecules),
            "sanitizable": rate(entry["sanitizable"], molecules),
            "qed_mean": float(np.mean(entry["qed"])) if entry["qed"] else float("nan"),
            "qed_median": float(np.median(entry["qed"])) if entry["qed"] else float("nan"),
            "qed_molecules": len(entry["qed"]),
        }
    if has_degree:
        result["degree_accuracy"] = rate(degree_correct, degree_total)
        result["degree_fallback_rate"] = rate(fallbacks, molecules)
        result["degree_fallback_reasons"] = dict(reasons.most_common())
    return result


@torch.no_grad()
def anisotropic_robustness(model, dataset: MolecularGraphDataset, *,
                           eigenvalues: torch.Tensor, eigenvectors: torch.Tensor,
                           indices: Sequence[int] | None = None,
                           device: str | torch.device = "cpu",
                           magnitudes: Sequence[float] = DEFAULT_MAGNITUDES,
                           groups: Sequence[str] = DIRECTION_GROUPS,
                           modes: Sequence[str] = MAGNITUDE_MODES,
                           k: int = 5, seed: int = 0, batch_size: int = 128) -> list[dict[str, Any]]:
    """Reconstruction quality under displacements confined to chosen eigen-directions.

    `eigenvalues` / `eigenvectors` must come from the TRAINING split; `indices` selects
    the VALIDATION molecules the measurement runs on. Magnitude 0 is always included as
    the shared unperturbed baseline, measured through the identical code path.
    """
    order = list(range(len(dataset))) if indices is None else [int(i) for i in indices]
    reference = node_embeddings(model, dataset, indices=order, device=device, batch_size=batch_size)
    scale_unit = float(reference.to(torch.float64).std(0).mean())
    rows: list[dict[str, Any]] = []

    baseline = _decode_and_score(model, dataset, order, torch.zeros_like(reference), device, batch_size)
    rows.append({"group": "none", "mode": "baseline", "magnitude": 0.0, "k": 0,
                 "directions_used": 0, "mean_displacement_norm": 0.0,
                 "scale_unit_per_dimension_std": scale_unit, **baseline})

    for mode in modes:
        for group in groups:
            for magnitude in magnitudes:
                if magnitude == 0.0:
                    continue                       # the baseline row above covers it once
                generator = torch.Generator().manual_seed(
                    int(seed) + 1000 * MAGNITUDE_MODES.index(mode)
                    + 100 * DIRECTION_GROUPS.index(group) + int(magnitude * 100))
                displacement = perturbation(reference, eigenvalues, eigenvectors, group=group,
                                            magnitude=magnitude, mode=mode, k=k,
                                            generator=generator, scale_unit=scale_unit)
                scored = _decode_and_score(model, dataset, order, displacement, device, batch_size)
                rows.append({
                    "group": group, "mode": mode, "magnitude": float(magnitude), "k": int(k),
                    "directions_used": len(direction_indices(group, reference.shape[1], k)),
                    "mean_displacement_norm": float(displacement.norm(dim=1).mean()),
                    "scale_unit_per_dimension_std": scale_unit, **scored})
    return rows


def format_spectrum(summary: dict[str, Any]) -> str:
    lines = [f"Embedding covariance on {summary['nodes']:,} training nodes, "
             f"dimension {summary['dimension']}",
             f"  total variance            {summary['total_variance']:.4f}",
             f"  participation ratio       {summary['participation_ratio']:.2f} "
             f"of {summary['dimension']}",
             f"  effective rank  90/95/99/99.9%   {summary['effective_rank_90']} / "
             f"{summary['effective_rank_95']} / {summary['effective_rank_99']} / "
             f"{summary['effective_rank_999']}",
             f"  variance share top 1/5/10        {summary['variance_share_top1']:.4f} / "
             f"{summary['variance_share_top5']:.4f} / {summary['variance_share_top10']:.4f}",
             f"  eigenvalue max/median/min        {summary['eigenvalue_max']:.3e} / "
             f"{summary['eigenvalue_median']:.3e} / {summary['eigenvalue_min']:.3e}",
             f"  condition number max/min         {summary['condition_number_max_over_min']:.3e}",
             f"  per-dimension std mean/min/max   {summary['per_dimension_std_mean']:.4f} / "
             f"{summary['per_dimension_std_min']:.4f} / {summary['per_dimension_std_max']:.4f}",
             f"  dimensions below 0.01 std        {summary['dimensions_below_whitening_floor_0p01']}",
             f"  max |correlation| between dims   {summary['max_absolute_correlation']:.4f} "
             f"(mean {summary['mean_absolute_correlation']:.4f})"]
    if summary["strongest_correlations"]:
        lines.append("  strongest correlated dimension pairs:")
        for pair in summary["strongest_correlations"][:5]:
            lines.append(f"    {pair['dimensions'][0]:>3} - {pair['dimensions'][1]:<3} "
                         f"r = {pair['correlation']:+.4f}")
    return "\n".join(lines)


def format_robustness(rows: Sequence[dict[str, Any]]) -> str:
    decoder = "degree" if rows and "degree" in rows[0] else "constrained"
    header = (f"{'group':<10}{'mode':<10}{'mag':>6}{'dirs':>6}{'|dx|':>8}"
              f"{'exact':>8}{'valid':>8}{'degacc':>8}{'fallbk':>8}{'QEDmean':>9}")
    lines = [f"Direction-resolved robustness (decoder: {decoder}; basis from TRAIN, "
             f"measured on VALIDATION)", header, "-" * len(header)]
    for row in rows:
        entry = row.get(decoder, {})
        lines.append(
            f"{row['group']:<10}{row['mode']:<10}{row['magnitude']:>6.2f}"
            f"{row['directions_used']:>6}{row['mean_displacement_norm']:>8.3f}"
            f"{entry.get('exact_molecule_match', float('nan')):>8.4f}"
            f"{entry.get('strict_validity', float('nan')):>8.4f}"
            f"{row.get('degree_accuracy', float('nan')):>8.4f}"
            f"{row.get('degree_fallback_rate', float('nan')):>8.4f}"
            f"{entry.get('qed_mean', float('nan')):>9.4f}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_MAGNITUDES",
    "DIRECTION_GROUPS",
    "MAGNITUDE_MODES",
    "anisotropic_robustness",
    "covariance_spectrum",
    "direction_indices",
    "format_robustness",
    "format_spectrum",
    "node_embeddings",
    "perturbation",
]
