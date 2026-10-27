"""Generation metrics with explicit, written-down definitions.

The literature uses "validity", "uniqueness" and "novelty" for several
different quantities, so every number this module produces states what it
divides by. For one decoder:

    sanitizable      = RDKit-sanitizable samples / all samples
    strict_validity  = sanitizable AND one connected fragment / all samples
                       <- the headline validity; no correction step is applied
    uniqueness       = distinct canonical SMILES / strictly valid samples
    novelty          = unique strictly-valid SMILES not in the TRAINING set
                       / unique strictly-valid SMILES
    nuv              = strict_validity * uniqueness * novelty
    fallback_rate    = samples whose degree-constrained assembly fell back to
                       the greedy decoder / all samples

Novelty is measured against the explicit training SMILES only (never the full
dataset, which would include the held-out test molecules). Novelty against
train+val is reported separately as a secondary number.

Diversity: internal diversity `1 - mean pairwise Tanimoto` over Morgan
fingerprints (radius 2, 2048 bits) on a seeded subsample, plus the count and
ratio of distinct Bemis-Murcko scaffolds.

All rates carry a Wilson 95% interval, because at 10,000 samples the third
decimal place is noise.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Iterable, Sequence

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from zinc_gnf.generation import DECODERS
from zinc_gnf.reconstruction_analysis import wilson_interval

FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 2048
DIVERSITY_SAMPLE = 1000        # pairwise Tanimoto is O(k^2); 1000 molecules = 499,500 pairs
DIVERSITY_SEED = 20260101


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    if denominator == 0:
        return {"value": float("nan"), "count": int(numerator), "total": 0,
                "ci95": [float("nan"), float("nan")]}
    low, high = wilson_interval(int(numerator), int(denominator))
    return {"value": numerator / denominator, "count": int(numerator),
            "total": int(denominator), "ci95": [low, high]}


def morgan_fingerprint(smiles: str):
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(molecule, FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS)


def internal_diversity(smiles: Sequence[str], *, sample: int = DIVERSITY_SAMPLE,
                       seed: int = DIVERSITY_SEED) -> dict[str, Any]:
    """`1 - mean pairwise Tanimoto` on a seeded subsample of distinct molecules."""
    unique = sorted(set(smiles))
    if len(unique) < 2:
        return {"internal_diversity": float("nan"), "molecules": len(unique),
                "pairs": 0, "mean_tanimoto": float("nan")}
    if len(unique) > sample:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(unique), size=sample, replace=False)
        unique = [unique[int(i)] for i in sorted(chosen)]
    fingerprints = [f for f in (morgan_fingerprint(s) for s in unique) if f is not None]
    if len(fingerprints) < 2:
        return {"internal_diversity": float("nan"), "molecules": len(fingerprints),
                "pairs": 0, "mean_tanimoto": float("nan")}
    total = 0.0
    pairs = 0
    for position in range(1, len(fingerprints)):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprints[position],
                                                          fingerprints[:position])
        total += float(sum(similarities))
        pairs += len(similarities)
    mean_similarity = total / pairs
    return {"internal_diversity": 1.0 - mean_similarity, "molecules": len(fingerprints),
            "pairs": pairs, "mean_tanimoto": mean_similarity}


def scaffold_statistics(smiles: Sequence[str]) -> dict[str, Any]:
    """Distinct Bemis-Murcko scaffolds among the unique valid molecules."""
    scaffolds = []
    for item in set(smiles):
        try:
            with rdBase.BlockLogs():
                scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(smiles=item))
        except Exception:  # noqa: BLE001
            continue
    distinct = len(set(scaffolds))
    return {"molecules": len(scaffolds), "distinct_scaffolds": distinct,
            "scaffolds_per_molecule": (distinct / len(scaffolds)) if scaffolds else float("nan")}


def decoder_metrics(records: Sequence[dict[str, Any]], decoder: str, *,
                    train_smiles: set[str], reference_smiles: dict[str, set[str]] | None = None,
                    diversity: bool = True) -> dict[str, Any]:
    """All metrics for one decoder over one sample of generated molecules."""
    total = len(records)
    entries = [r[decoder] for r in records if decoder in r]
    if len(entries) != total:
        raise ValueError(f"{total - len(entries)} records have no '{decoder}' result")
    sanitizable = sum(1 for e in entries if e.get("sanitizable"))
    connected = sum(1 for e in entries if e.get("connected"))
    valid_smiles = [e["smiles"] for e in entries if e.get("valid") and e.get("smiles")]
    fallbacks = sum(1 for e in entries if e.get("fallback"))
    unique = sorted(set(valid_smiles))
    novel = [s for s in unique if s not in train_smiles]

    strict_validity = _rate(len(valid_smiles), total)
    uniqueness = _rate(len(unique), len(valid_smiles))
    novelty = _rate(len(novel), len(unique))
    nuv = (strict_validity["value"] * uniqueness["value"] * novelty["value"]
           if len(unique) else float("nan"))

    metrics: dict[str, Any] = {
        "samples": total,
        "sanitizable": _rate(sanitizable, total),
        "connected_given_sanitizable": _rate(connected, sanitizable),
        "strict_validity": strict_validity,
        "uniqueness": uniqueness,
        "novelty_vs_train": novelty,
        "nuv": nuv,
        "nuv_definition": "strict_validity * uniqueness * novelty_vs_train",
        "fallback_rate": _rate(fallbacks, total),
        "fallback_reasons": dict(Counter(e.get("fallback_reason") for e in entries
                                         if e.get("fallback")).most_common()),
        "valid_molecules": len(valid_smiles),
        "unique_molecules": len(unique),
        "novel_molecules": len(novel),
    }
    for name, reference in (reference_smiles or {}).items():
        metrics[f"novelty_vs_{name}"] = _rate(sum(1 for s in unique if s not in reference), len(unique))
    if diversity and unique:
        metrics["diversity"] = internal_diversity(unique)
        metrics["scaffolds"] = scaffold_statistics(unique)
    if valid_smiles:
        qed_values = [e["qed"] for e in entries if e.get("valid") and e.get("qed") is not None]
        rings = [e["rings"] for e in entries if e.get("valid") and e.get("rings") is not None]
        sizes = [r["n_node"] for r in records if r.get(decoder, {}).get("valid")]
        metrics["qed"] = {"mean": float(np.mean(qed_values)), "std": float(np.std(qed_values)),
                          "molecules": len(qed_values)} if qed_values else None
        metrics["rings_mean"] = float(np.mean(rings)) if rings else float("nan")
        metrics["valid_size_mean"] = float(np.mean(sizes)) if sizes else float("nan")
    return metrics


def generation_metrics(records: Sequence[dict[str, Any]], *, train_smiles: Iterable[str],
                       validation_smiles: Iterable[str] | None = None,
                       decoders: Sequence[str] = DECODERS,
                       headline: str = "degree",
                       diversity: bool = True) -> dict[str, Any]:
    """Metrics for every decoder, plus the headline decoder's numbers at the top.

    `train_smiles` must be the canonical SMILES of the TRAINING split only.
    """
    train_set = set(train_smiles)
    if not train_set:
        raise ValueError("train_smiles is empty; novelty would be meaningless")
    references: dict[str, set[str]] = {}
    if validation_smiles is not None:
        references["train_and_val"] = train_set | set(validation_smiles)
    present = [d for d in decoders if records and d in records[0]]
    result: dict[str, Any] = {
        "samples": len(records),
        "train_reference_molecules": len(train_set),
        "decoders": {name: decoder_metrics(records, name, train_smiles=train_set,
                                           reference_smiles=references, diversity=diversity)
                     for name in present},
    }
    if headline in result["decoders"]:
        head = result["decoders"][headline]
        result["headline_decoder"] = headline
        result["headline"] = {
            "strict_validity": head["strict_validity"]["value"],
            "sanitizable": head["sanitizable"]["value"],
            "uniqueness": head["uniqueness"]["value"],
            "novelty_vs_train": head["novelty_vs_train"]["value"],
            "nuv": head["nuv"],
            "internal_diversity": head.get("diversity", {}).get("internal_diversity"),
            "fallback_rate": head["fallback_rate"]["value"],
        }
    return result


def format_metrics(metrics: dict[str, Any]) -> str:
    """Compact human-readable report (also written beside the JSON)."""
    lines = [f"Generated {metrics['samples']:,} molecules; "
             f"novelty reference = {metrics['train_reference_molecules']:,} training molecules"]
    for name, entry in metrics["decoders"].items():
        marker = "  <- headline" if name == metrics.get("headline_decoder") else ""
        lines.append(f"\n[{name}]{marker}")
        lines.append(f"  sanitizable          {entry['sanitizable']['value']:.4f}")
        lines.append(f"  strict validity      {entry['strict_validity']['value']:.4f} "
                     f"(95% CI {entry['strict_validity']['ci95'][0]:.4f}-{entry['strict_validity']['ci95'][1]:.4f})")
        lines.append(f"  uniqueness           {entry['uniqueness']['value']:.4f} "
                     f"of {entry['valid_molecules']:,} valid")
        lines.append(f"  novelty vs train     {entry['novelty_vs_train']['value']:.4f} "
                     f"of {entry['unique_molecules']:,} unique")
        lines.append(f"  N.U.V.               {entry['nuv']:.4f}   ({entry['nuv_definition']})")
        if entry.get("diversity"):
            lines.append(f"  internal diversity   {entry['diversity']['internal_diversity']:.4f} "
                         f"on {entry['diversity']['molecules']:,} molecules")
        if entry.get("scaffolds"):
            lines.append(f"  distinct scaffolds   {entry['scaffolds']['distinct_scaffolds']:,} "
                         f"({entry['scaffolds']['scaffolds_per_molecule']:.3f} per molecule)")
        if entry["fallback_rate"]["total"]:
            lines.append(f"  decoder fallback     {entry['fallback_rate']['value']:.4f} "
                         f"{entry['fallback_reasons'] or ''}")
        if entry.get("qed"):
            lines.append(f"  QED of valid         {entry['qed']['mean']:.4f} +- {entry['qed']['std']:.4f}")
    return "\n".join(lines)


__all__ = [
    "DIVERSITY_SAMPLE",
    "decoder_metrics",
    "format_metrics",
    "generation_metrics",
    "internal_diversity",
    "morgan_fingerprint",
    "scaffold_statistics",
]
