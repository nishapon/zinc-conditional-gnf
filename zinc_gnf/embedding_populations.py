"""Comparing embedding POPULATIONS: real, node-shuffled and flow-generated.

The question this module exists to answer: the frozen autoencoder reproduces molecular
chemistry almost perfectly from real embeddings (QED 0.730 -> 0.731, aromatic rings
1.858 -> 1.830), yet 300-step flow samples decode to QED 0.240 with 0.025 aromatic
rings. Per-node predictions survive far better than pairwise ones, which suggests the
flow's samples are individually plausible but mutually unrelated. The shuffled-node
control tests exactly that, and needs no flow at all.

Three populations, all with the SAME molecule sizes so nothing differs except the
embeddings themselves:

  real        the autoencoder's embeddings of validation molecules
  shuffled    those same embedding vectors, globally permuted and then re-partitioned
              into molecules of the original sizes. The pooled set of node vectors is
              therefore IDENTICAL to `real` - every per-node marginal, the covariance
              and the principal components are preserved exactly - and the only thing
              destroyed is which nodes belong together
  flow        embeddings sampled from a trained flow checkpoint for the same sizes

What a shuffled-node result can and cannot establish:

  CAN: if `shuffled` reproduces the flow's symptoms (QED near 0.24, aromatic rings near
       0, excess formal charge, high degree-decoder fallback), then losing the inter-node
       relationship is SUFFICIENT to produce them, and the flow's samples are consistent
       with having learned per-node marginals but not the joint structure.
  CANNOT: it cannot prove that this is what the flow actually does - only a direct
       comparison of the flow's own embedding statistics against `real` can support
       that, which is why those statistics are measured here too. Nor can it separate
       "needs more training" from "cannot be learned": a sufficient cause is not a
       necessary one.

Nothing here trains or modifies anything; the autoencoder and the flow are used in eval
mode under `torch.no_grad()`, and the test split is never read.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from zinc_gnf.chemistry_charge_aware import formal_charges
from zinc_gnf.constants_charge_aware import ATOM_VOCAB, CHARGE_OF_TYPE, NUM_BOND_TYPES
from zinc_gnf.dataset import MolecularGraphDataset
from zinc_gnf.decoder import MAX_DEGREE_CLASS
from zinc_gnf.generation import decode_embeddings, generation_batch
from zinc_gnf.latent_geometry import covariance_spectrum, node_embeddings

NEAR_IDENTICAL_TOLERANCE = 1e-3     # relative to the mean per-dimension std


@dataclass
class EmbeddingPopulation:
    """Node embeddings of several molecules, pooled in molecule order."""

    name: str
    counts: list[int]                    # heavy-atom count per molecule
    pooled: torch.Tensor                 # [sum(counts), D]

    def __post_init__(self) -> None:
        if int(sum(self.counts)) != int(self.pooled.shape[0]):
            raise ValueError(f"{self.name}: {sum(self.counts)} nodes expected, "
                             f"{self.pooled.shape[0]} given")

    @property
    def molecules(self) -> int:
        return len(self.counts)

    @property
    def dimension(self) -> int:
        return int(self.pooled.shape[1])

    def per_molecule(self) -> Iterator[torch.Tensor]:
        offset = 0
        for count in self.counts:
            yield self.pooled[offset:offset + count]
            offset += count

    def batches(self, batch_size: int) -> Iterator[tuple[list[int], torch.Tensor]]:
        """Padded `[B, N, D]` chunks plus their counts, for the decoder."""
        offset = 0
        for start in range(0, self.molecules, batch_size):
            counts = self.counts[start:start + batch_size]
            width = max(counts)
            padded = self.pooled.new_zeros((len(counts), width, self.dimension))
            for position, count in enumerate(counts):
                padded[position, :count] = self.pooled[offset:offset + count]
                offset += count
            yield counts, padded


# ---------------------------------------------------------------------------
# Building the populations.
# ---------------------------------------------------------------------------
def real_population(model, dataset: MolecularGraphDataset, indices: Sequence[int], *,
                    device: str | torch.device = "cpu", batch_size: int = 128,
                    name: str = "real") -> EmbeddingPopulation:
    order = [int(i) for i in indices]
    pooled = node_embeddings(model, dataset, indices=order, device=device, batch_size=batch_size)
    counts = [int(dataset.n_node[i]) for i in order]
    return EmbeddingPopulation(name=name, counts=counts, pooled=pooled.cpu())


def shuffle_population(population: EmbeddingPopulation, *, seed: int = 0,
                       name: str = "shuffled") -> tuple[EmbeddingPopulation, dict[str, Any]]:
    """Globally permute the pooled node vectors, keeping every molecule's size.

    The result uses exactly the same multiset of node embeddings, so all per-node
    statistics (marginals, covariance, principal components) are preserved by
    construction and ONLY the grouping of nodes into molecules changes.
    """
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(population.pooled.shape[0], generator=generator)
    shuffled = EmbeddingPopulation(name=name, counts=list(population.counts),
                                   pooled=population.pooled[permutation].clone())
    # how many nodes happened to land back in their original molecule
    molecule_of_node = np.repeat(np.arange(population.molecules), population.counts)
    stayed = int((molecule_of_node[permutation.numpy()] == molecule_of_node).sum())
    return shuffled, {"permuted_nodes": int(permutation.numel()),
                      "nodes_back_in_their_own_molecule": stayed,
                      "fraction_back_in_their_own_molecule": stayed / permutation.numel(),
                      "seed": int(seed)}


def smoothed_population(population: EmbeddingPopulation, *, per_dimension_std: torch.Tensor,
                        sigma: float, floor: float = 0.01, seed: int = 0,
                        name: str | None = None) -> EmbeddingPopulation:
    """Real embeddings plus the SAME Gaussian smoothing the flow is trained against.

    This is the honest reference for every "how close must two embeddings be" question:
    the flow's target is not the deterministic embedding set but its convolution with
    `sigma * per-dimension std` (floored exactly as the whitening floors it). Two
    originally identical vectors of symmetry-equivalent atoms end up about
    `sigma * sqrt(2) * sqrt(D)` apart in whitened units, not 0, so comparing a flow
    against unsmoothed real embeddings overstates what it has to achieve.
    """
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    scale = per_dimension_std.to(population.pooled.dtype).clamp(min=float(floor))
    if scale.shape[0] != population.dimension:
        raise ValueError(f"per_dimension_std has {scale.shape[0]} entries, "
                         f"expected {population.dimension}")
    generator = torch.Generator().manual_seed(int(seed))
    noise = torch.randn(population.pooled.shape, generator=generator)
    return EmbeddingPopulation(name=name or f"real_smoothed_sigma{sigma:g}",
                               counts=list(population.counts),
                               pooled=population.pooled + float(sigma) * scale * noise)


@torch.no_grad()
def flow_population(flow, counts: Sequence[int], *, temperature: float = 1.0, seed: int = 0,
                    device: str | torch.device = "cpu", batch_size: int = 128,
                    name: str | None = None) -> EmbeddingPopulation:
    """Sample embeddings from a trained flow for the GIVEN molecule sizes."""
    from zinc_gnf.flow import node_mask_from_counts

    if flow.config.conditional:
        raise ValueError("this flow is conditional; the unconditional path cannot sample it")
    flow.eval()
    generator = torch.Generator(device=device).manual_seed(int(seed))
    chunks: list[torch.Tensor] = []
    sizes = [int(c) for c in counts]
    for start in range(0, len(sizes), batch_size):
        chunk = sizes[start:start + batch_size]
        mask = node_mask_from_counts(chunk, device=device)
        embeddings = flow.sample(mask, temperature=temperature, generator=generator)
        chunks.append(embeddings[mask].cpu())
    return EmbeddingPopulation(name=name or f"flow_T{temperature:g}", counts=sizes,
                               pooled=torch.cat(chunks))


# ---------------------------------------------------------------------------
# Embedding statistics.
# ---------------------------------------------------------------------------
def _pairwise_statistics(vectors: torch.Tensor) -> tuple[float, float, float, float]:
    """(mean distance, median distance, mean cosine, min distance) within one molecule."""
    if vectors.shape[0] < 2:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    distances = torch.cdist(vectors.unsqueeze(0), vectors.unsqueeze(0)).squeeze(0)
    normalised = vectors / vectors.norm(dim=1, keepdim=True).clamp(min=1e-12)
    cosine = normalised @ normalised.T
    upper = torch.triu(torch.ones_like(distances), diagonal=1).bool()
    return (float(distances[upper].mean()), float(distances[upper].median()),
            float(cosine[upper].mean()), float(distances[upper].min()))


def embedding_population_statistics(population: EmbeddingPopulation, *,
                                    reference_eigenvectors: torch.Tensor | None = None,
                                    reference_scale: float | None = None,
                                    between_molecule_pairs: int = 20000,
                                    seed: int = 0) -> dict[str, Any]:
    """Marginals, covariance, principal components and INTER-NODE structure.

    `reference_eigenvectors` (from the TRAINING embeddings) lets every population be
    projected onto the same basis, so "how much variance sits in the directions the real
    data occupies" is comparable across populations.
    """
    pooled = population.pooled.to(torch.float64)
    scale = float(reference_scale if reference_scale is not None else pooled.std(0).mean())
    summary, eigenvalues, _ = covariance_spectrum(population.pooled, correlation_pairs=5)

    # within-molecule pairwise structure
    means, medians, cosines, minima = [], [], [], []
    for vectors in population.per_molecule():
        mean, median, cosine, minimum = _pairwise_statistics(vectors.to(torch.float64))
        if not np.isnan(mean):
            means.append(mean), medians.append(median), cosines.append(cosine), minima.append(minimum)
    tolerance = NEAR_IDENTICAL_TOLERANCE * scale
    near_identical = [m for m in minima if m <= tolerance]

    # between-molecule baseline on random node pairs from different molecules
    generator = torch.Generator().manual_seed(int(seed))
    molecule_of_node = torch.from_numpy(np.repeat(np.arange(population.molecules),
                                                  population.counts))
    total = pooled.shape[0]
    left = torch.randint(0, total, (between_molecule_pairs,), generator=generator)
    right = torch.randint(0, total, (between_molecule_pairs,), generator=generator)
    keep = molecule_of_node[left] != molecule_of_node[right]
    left, right = left[keep], right[keep]
    if left.numel():
        difference = pooled[left] - pooled[right]
        between_distance = float(difference.norm(dim=1).mean())
        normalised = pooled / pooled.norm(dim=1, keepdim=True).clamp(min=1e-12)
        between_cosine = float((normalised[left] * normalised[right]).sum(1).mean())
    else:
        between_distance = between_cosine = float("nan")

    statistics: dict[str, Any] = {
        "name": population.name,
        "molecules": population.molecules,
        "nodes": int(pooled.shape[0]),
        "dimension": population.dimension,
        "mean_norm": float(pooled.norm(dim=1).mean()),
        "std_norm": float(pooled.norm(dim=1).std()),
        "per_dimension_std_mean": float(pooled.std(0).mean()),
        "per_dimension_mean_absolute": float(pooled.mean(0).abs().mean()),
        "total_variance": summary["total_variance"],
        "participation_ratio": summary["participation_ratio"],
        "effective_rank_99": summary["effective_rank_99"],
        "condition_number_max_over_median": summary["condition_number_max_over_median"],
        "max_absolute_correlation": summary["max_absolute_correlation"],
        "within_molecule_mean_distance": float(np.mean(means)) if means else float("nan"),
        "within_molecule_median_distance": float(np.mean(medians)) if medians else float("nan"),
        "within_molecule_mean_cosine": float(np.mean(cosines)) if cosines else float("nan"),
        "within_molecule_min_distance_mean": float(np.mean(minima)) if minima else float("nan"),
        "between_molecule_mean_distance": between_distance,
        "between_molecule_mean_cosine": between_cosine,
        "within_over_between_distance": (float(np.mean(means)) / between_distance)
        if means and between_distance else float("nan"),
        "molecules_with_near_identical_pair": (len(near_identical) / len(minima))
        if minima else float("nan"),
        "near_identical_tolerance": tolerance,
    }
    if reference_eigenvectors is not None:
        basis = reference_eigenvectors.to(torch.float64)
        centred = pooled - pooled.mean(0, keepdim=True)
        projected = centred @ basis                            # [nodes, D] in the real basis
        variance = projected.var(0, unbiased=True)
        total_variance = float(variance.sum())
        for top in (5, 10, 26):
            if basis.shape[1] >= top:
                statistics[f"variance_share_in_real_top{top}"] = float(
                    variance[:top].sum() / max(total_variance, 1e-300))
        statistics["variance_share_outside_real_top26"] = float(
            variance[26:].sum() / max(total_variance, 1e-300)) if basis.shape[1] > 26 else float("nan")
        statistics["projected_variance_per_real_component"] = [float(v) for v in variance[:10]]
    return statistics


# ---------------------------------------------------------------------------
# What the decoder predicts BEFORE constrained assembly.
# ---------------------------------------------------------------------------
@torch.no_grad()
def decoder_prediction_statistics(model, population: EmbeddingPopulation, *,
                                  device: str | torch.device = "cpu",
                                  batch_size: int = 128) -> dict[str, Any]:
    """Atom-type, charge, degree and raw bond-logit statistics, before any assembly.

    These are the model's opinions as such: no valence rule, no connectivity pass and no
    integer program has touched them yet, so a difference here localises the problem to
    the embeddings rather than to the assembler.
    """
    was_training = model.training
    model.eval()
    has_degree = bool(model.config.degree_head)
    atom_counts: Counter = Counter()
    degree_counts: Counter = Counter()
    bond_argmax_counts: Counter = Counter()
    atom_confidence: list[float] = []
    bond_confidence: list[float] = []
    no_bond_probability: list[float] = []
    expected_bonds_per_atom: list[float] = []
    charged_atoms = charged_molecules = molecules = atoms_total = 0
    even_degree_sum = feasible_degrees = 0
    degree_sums: list[int] = []

    for counts, padded in population.batches(batch_size):
        batch = generation_batch(counts, device=device)
        outputs = model.decode(padded.to(device), batch, teacher_forcing=False)
        atom_probabilities = torch.softmax(outputs["atom_logits"], dim=-1)
        bond_probabilities = torch.softmax(outputs["bond_logits"], dim=-1)
        atom_prediction = atom_probabilities.argmax(-1).cpu().numpy()
        atom_best = atom_probabilities.max(-1).values.cpu().numpy()
        bond_prediction = bond_probabilities.argmax(-1).cpu().numpy()
        bond_best = bond_probabilities.max(-1).values.cpu().numpy()
        bond_none = bond_probabilities[..., 0].cpu().numpy()
        degrees = (outputs["degree_logits"].argmax(-1).cpu().numpy() if has_degree else None)

        for position, count in enumerate(counts):
            molecules += 1
            atoms_total += count
            types = atom_prediction[position, :count]
            atom_counts.update(int(t) for t in types)
            atom_confidence.extend(float(v) for v in atom_best[position, :count])
            charges = formal_charges(types)
            charged = int((charges != 0).sum())
            charged_atoms += charged
            charged_molecules += int(charged > 0)
            upper = np.triu_indices(count, k=1)
            if upper[0].size:
                classes = bond_prediction[position, :count, :count][upper]
                bond_argmax_counts.update(int(c) for c in classes)
                bond_confidence.extend(float(v) for v in bond_best[position, :count, :count][upper])
                none = bond_none[position, :count, :count][upper]
                no_bond_probability.extend(float(v) for v in none)
                expected_bonds_per_atom.append(float(2.0 * (1.0 - none).sum() / count))
            if has_degree:
                predicted = degrees[position, :count]
                degree_counts.update(int(d) for d in predicted)
                total = int(predicted.sum())
                degree_sums.append(total)
                even_degree_sum += int(total % 2 == 0)
                feasible_degrees += int(total % 2 == 0 and bool((predicted <= count - 1).all())
                                        and bool((predicted >= 0).all()))
    if was_training:
        model.train()

    def histogram(counter: Counter, size: int) -> list[float]:
        total = sum(counter.values())
        return [counter[i] / total if total else float("nan") for i in range(size)]

    statistics: dict[str, Any] = {
        "name": population.name,
        "molecules": molecules,
        "atoms": atoms_total,
        "atom_type_histogram": histogram(atom_counts, len(ATOM_VOCAB)),
        "atom_type_labels": [f"{ATOM_VOCAB[i]}{CHARGE_OF_TYPE[i]:+d}" if CHARGE_OF_TYPE[i]
                             else ATOM_VOCAB[i] for i in range(len(ATOM_VOCAB))],
        "charged_atom_fraction": charged_atoms / atoms_total if atoms_total else float("nan"),
        "molecules_with_charged_atom": charged_molecules / molecules if molecules else float("nan"),
        "atom_confidence_mean": float(np.mean(atom_confidence)) if atom_confidence else float("nan"),
        "bond_argmax_histogram": histogram(bond_argmax_counts, NUM_BOND_TYPES),
        "bond_confidence_mean": float(np.mean(bond_confidence)) if bond_confidence else float("nan"),
        "no_bond_probability_mean": float(np.mean(no_bond_probability)) if no_bond_probability
        else float("nan"),
        "expected_bonds_per_atom_mean": float(np.mean(expected_bonds_per_atom))
        if expected_bonds_per_atom else float("nan"),
    }
    if has_degree:
        statistics.update({
            "degree_histogram": histogram(degree_counts, MAX_DEGREE_CLASS + 1),
            "predicted_degree_mean": float(np.mean([d for d in degree_counts.elements()]))
            if degree_counts else float("nan"),
            "molecules_with_even_degree_sum": even_degree_sum / molecules if molecules else float("nan"),
            "molecules_with_feasible_degrees": feasible_degrees / molecules if molecules
            else float("nan"),
            "predicted_degree_sum_mean": float(np.mean(degree_sums)) if degree_sums else float("nan"),
        })
    return statistics


@torch.no_grad()
def decode_population(model, population: EmbeddingPopulation, *,
                      device: str | torch.device = "cpu", batch_size: int = 128,
                      decoders: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Assemble molecules from a population, through the normal generation decode path."""
    if decoders is None:
        decoders = ("constrained", "degree") if model.config.degree_head else ("constrained",)
    records: list[dict[str, Any]] = []
    for counts, padded in population.batches(batch_size):
        records.extend(decode_embeddings(model, padded.to(device), counts, decoders=decoders))
    return records


def format_population_statistics(rows: Sequence[dict[str, Any]]) -> str:
    """Side-by-side embedding statistics, reference population first."""
    width = max(16, *(len(r["name"]) + 2 for r in rows))
    fields = [
        ("molecules", "molecules", "{:,}"),
        ("nodes", "nodes", "{:,}"),
        ("mean_norm", "mean ||x||", "{:.4f}"),
        ("per_dimension_std_mean", "per-dim std mean", "{:.4f}"),
        ("participation_ratio", "participation ratio", "{:.2f}"),
        ("effective_rank_99", "effective rank 99%", "{}"),
        ("variance_share_in_real_top5", "var in real top-5", "{:.4f}"),
        ("variance_share_in_real_top26", "var in real top-26", "{:.4f}"),
        ("within_molecule_mean_distance", "within-mol distance", "{:.4f}"),
        ("between_molecule_mean_distance", "between-mol distance", "{:.4f}"),
        ("within_over_between_distance", "within / between", "{:.4f}"),
        ("within_molecule_mean_cosine", "within-mol cosine", "{:.4f}"),
        ("within_molecule_min_distance_mean", "within-mol min dist", "{:.4f}"),
        ("molecules_with_near_identical_pair", "near-identical pair", "{:.4f}"),
    ]
    header = f"{'statistic':<24}" + "".join(f"{r['name']:>{width}}" for r in rows)
    lines = ["Embedding population statistics (real basis from the TRAINING split)",
             header, "-" * len(header)]
    for key, label, template in fields:
        values = []
        for row in rows:
            value = row.get(key)
            values.append("-" if value is None else
                          ("nan" if isinstance(value, float) and np.isnan(value)
                           else template.format(value)))
        lines.append(f"{label:<24}" + "".join(f"{v:>{width}}" for v in values))
    return "\n".join(lines)


def format_prediction_statistics(rows: Sequence[dict[str, Any]]) -> str:
    width = max(16, *(len(r["name"]) + 2 for r in rows))
    fields = [
        ("charged_atom_fraction", "charged atom frac", "{:.4f}"),
        ("molecules_with_charged_atom", "molecules charged", "{:.4f}"),
        ("atom_confidence_mean", "atom confidence", "{:.4f}"),
        ("bond_confidence_mean", "bond confidence", "{:.4f}"),
        ("no_bond_probability_mean", "P(no bond) mean", "{:.4f}"),
        ("expected_bonds_per_atom_mean", "exp. bonds/atom", "{:.4f}"),
        ("predicted_degree_sum_mean", "degree sum mean", "{:.2f}"),
        ("molecules_with_even_degree_sum", "even degree sum", "{:.4f}"),
        ("molecules_with_feasible_degrees", "feasible degrees", "{:.4f}"),
    ]
    header = f"{'prediction':<24}" + "".join(f"{r['name']:>{width}}" for r in rows)
    lines = ["Decoder predictions BEFORE constrained assembly", header, "-" * len(header)]
    for key, label, template in fields:
        values = []
        for row in rows:
            value = row.get(key)
            values.append("-" if value is None else
                          ("nan" if isinstance(value, float) and np.isnan(value)
                           else template.format(value)))
        lines.append(f"{label:<24}" + "".join(f"{v:>{width}}" for v in values))
    lines.append("")
    lines.append(f"{'bond argmax class':<24}" + "".join(f"{r['name']:>{width}}" for r in rows))
    for index, label in enumerate(("none", "single", "double", "triple")):
        values = [f"{r['bond_argmax_histogram'][index]:.4f}" for r in rows]
        lines.append(f"  {label:<22}" + "".join(f"{v:>{width}}" for v in values))
    return "\n".join(lines)


__all__ = [
    "NEAR_IDENTICAL_TOLERANCE",
    "EmbeddingPopulation",
    "decode_population",
    "decoder_prediction_statistics",
    "embedding_population_statistics",
    "flow_population",
    "format_population_statistics",
    "format_prediction_statistics",
    "real_population",
    "shuffle_population",
    "smoothed_population",
]
