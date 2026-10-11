"""Diagnostics for autoencoder reconstruction failures (read-only analysis).

Answers, with measurements instead of assumptions:

1. Failure strata: exact-reconstruction rate by molecule size, ring count,
   aromatic fraction, charge, and symmetry, and a failure TAXONOMY that splits
   every miss into
     * `symmetry_unreachable`  - the molecule has indistinguishable atoms, so
       no deterministic one-hot equivariant encoder + pairwise decoder can
       reproduce it (the ceiling), versus
     * `decodable_*`           - a miss that IS reachable in principle (wrong
       atom type, wrong bond, or an invalid assembly): the part training,
       capacity or decoder design can still improve.
2. Bond-metric inflation: pair accuracy against the trivial all-"no bond"
   baseline, per-class precision / recall / F1, balanced accuracy, and how many
   errors per molecule an exact match can tolerate (none).
3. The ceiling's assumptions: its sampling uncertainty, how often 1-WL
   classes coincide with RDKit's topological symmetry classes, and whether
   judging "reconstructed" by molecule IDENTITY (rather than labelled graph
   equality) could rescue any molecule the ceiling calls unreachable.

Nothing here modifies a model, a checkpoint or a dataset.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import itertools
import math
from typing import Any, Iterable, Sequence

import numpy as np
from rdkit import Chem, rdBase
import torch

from zinc_gnf.chemistry import canonical_connectivity_smiles
from zinc_gnf.chemistry_charge_aware import graph_to_mol_charge_aware
from zinc_gnf.constants_charge_aware import ATOM_VOCAB, CHARGE_OF_TYPE
from zinc_gnf.dataset import MolecularGraphDataset, iter_eval_batches
from zinc_gnf.decoding_charge_aware import valence_aware_bond_decode_typed
from zinc_gnf.symmetry import is_decodable, refine_colors

SIZE_BUCKETS = ((1, 14), (15, 19), (20, 24), (25, 29), (30, 34), (35, 38))


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def mixed_class_pairs(atom_types, bond_matrix) -> tuple[list[int], list[list[tuple[int, int]]]]:
    """1-WL colours and, for every unordered colour-class pair whose atom pairs carry
    DIFFERENT bond orders, the list of atom pairs. Non-empty <=> not decodable."""
    atoms = np.asarray(atom_types)
    bonds = np.asarray(bond_matrix)
    colors = refine_colors(atoms, bonds)
    n = len(atoms)
    groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            groups[(min(colors[i], colors[j]), max(colors[i], colors[j]))].append((i, j))
    mixed = [pairs for pairs in groups.values() if len({int(bonds[i, j]) for i, j in pairs}) > 1]
    return colors, mixed


def symmetry_source(smiles: str, atom_types, bond_matrix) -> str:
    """Why is this molecule not decodable?  none | ring_positions | acyclic_chains.

    `ring_positions`: some ambiguous atom pair involves a ring atom (symmetric ring
    positions, e.g. the ortho/meta atoms of a mono- or para-substituted phenyl).
    `acyclic_chains`: only side-chain atoms are ambiguous (e.g. two identical ethyl
    groups on one nitrogen; equivalent leaves on a shared centre are NOT ambiguous).
    """
    _, mixed = mixed_class_pairs(atom_types, bond_matrix)
    if not mixed:
        return "none"
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() != len(atom_types):
        return "unknown"
    in_ring = {a.GetIdx() for a in molecule.GetAtoms() if a.IsInRing()}
    involved = {atom for pairs in mixed for pair in pairs for atom in pair}
    return "ring_positions" if involved & in_ring else "acyclic_chains"


# ---------------------------------------------------------------------------
# Per-molecule descriptors (independent of any model).
# ---------------------------------------------------------------------------
def describe_molecule(smiles: str, atom_types: np.ndarray, bond_matrix: np.ndarray) -> dict[str, Any]:
    n = int(len(atom_types))
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    n_rings = largest_ring = aromatic_atoms = 0
    fused = False
    if molecule is not None:
        info = molecule.GetRingInfo()
        rings = info.AtomRings()
        n_rings = len(rings)
        largest_ring = max((len(r) for r in rings), default=0)
        aromatic_atoms = sum(a.GetIsAromatic() for a in molecule.GetAtoms())
        counts = Counter(i for ring in rings for i in ring)
        fused = any(c > 1 for c in counts.values())
    colors = refine_colors(atom_types, bond_matrix)
    class_sizes = Counter(colors)
    symmetric_atoms = sum(size for size in class_sizes.values() if size > 1)
    charges = [CHARGE_OF_TYPE[int(a)] for a in atom_types]
    return {
        "n_node": n,
        "n_rings": n_rings,
        "largest_ring": largest_ring,
        "fused_rings": bool(fused),
        "aromatic_fraction": aromatic_atoms / n if n else 0.0,
        "n_charged_atoms": int(sum(c != 0 for c in charges)),
        "net_charge": int(sum(charges)),
        "n_classes": len(class_sizes),
        "largest_class": max(class_sizes.values()) if class_sizes else 0,
        "symmetric_atoms": int(symmetric_atoms),
        "symmetric_fraction": symmetric_atoms / n if n else 0.0,
        "decodable": bool(is_decodable(atom_types, bond_matrix)),
        "symmetry_source": symmetry_source(smiles, atom_types, bond_matrix),
    }


def size_bucket(n: int) -> str:
    for low, high in SIZE_BUCKETS:
        if low <= n <= high:
            return f"{low}-{high}"
    return f">{SIZE_BUCKETS[-1][1]}"


def ring_bucket(n_rings: int) -> str:
    return "0" if n_rings == 0 else str(n_rings) if n_rings < 4 else "4+"


def aromatic_bucket(fraction: float) -> str:
    if fraction == 0:
        return "none"
    return "partial (<=50%)" if fraction <= 0.5 else "mostly (>50%)"


def charge_bucket(row: dict[str, Any]) -> str:
    if row["n_charged_atoms"] == 0:
        return "neutral"
    return "charged, net neutral" if row["net_charge"] == 0 else "charged, net ion"


def symmetry_bucket(row: dict[str, Any]) -> str:
    if row["decodable"]:
        return "decodable" if row["symmetric_atoms"] == 0 else "symmetric but decodable"
    return "symmetry-unreachable"


# ---------------------------------------------------------------------------
# Model-based analysis.
# ---------------------------------------------------------------------------
@torch.no_grad()
def analyze_reconstructions(
    model,
    dataset: MolecularGraphDataset,
    *,
    indices: Sequence[int] | None = None,
    device: str | torch.device = "cpu",
    batch_size: int = 128,
    eval_seed: int | None = 0,
) -> dict[str, Any]:
    """Per-molecule outcomes (free-running, predicted atom types) and bond statistics."""
    was_training = model.training
    model.eval()
    order = list(range(len(dataset))) if indices is None else [int(i) for i in indices]
    generator = None
    if eval_seed is not None and model.config.symmetry_noise_dim > 0:
        generator = torch.Generator(device=device).manual_seed(int(eval_seed))

    rows: list[dict[str, Any]] = []
    # confusion over real undirected pairs, rows = true class, columns = predicted class
    confusion = {"raw": np.zeros((4, 4), dtype=np.int64), "constrained": np.zeros((4, 4), dtype=np.int64)}
    pair_error_rates: list[float] = []
    consumed = 0
    for batch in iter_eval_batches(dataset, batch_size, order):
        members = order[consumed:consumed + batch.batch_size]
        consumed += batch.batch_size
        batch = batch.to(device)
        nodes = model.encode(batch, generator=generator)
        out = model.decode(nodes, batch, teacher_forcing=False)
        atom_pred = out["atom_logits"].argmax(-1).cpu().numpy()
        probs = torch.softmax(out["bond_logits"], dim=-1).cpu().numpy()
        raw_all = out["bond_logits"].argmax(-1).cpu().numpy()
        for b, index in enumerate(members):
            n = int(batch.n_node[b])
            true_atoms, true_bonds = dataset.true_graph(index)
            descriptor = describe_molecule(dataset.smiles[index], true_atoms, true_bonds)
            atoms = atom_pred[b, :n]
            raw = raw_all[b, :n, :n].copy()
            np.fill_diagonal(raw, 0)
            constrained, info = valence_aware_bond_decode_typed(atoms, probs[b, :n, :n])

            row = dict(descriptor, row_id=int(dataset.row_id[index]), smiles=dataset.smiles[index])
            atoms_ok = bool(np.array_equal(atoms, true_atoms))
            row["atoms_exact"] = atoms_ok
            row["atom_errors"] = int((atoms != true_atoms).sum())
            iu = np.triu_indices(n, k=1)
            true_pairs = true_bonds[iu]
            for name, bonds in (("raw", raw), ("constrained", constrained)):
                pred_pairs = bonds[iu]
                np.add.at(confusion[name], (true_pairs, pred_pairs), 1)
                try:
                    with rdBase.BlockLogs():
                        mol = graph_to_mol_charge_aware(atoms, bonds)
                    sanitizable, single = True, len(Chem.GetMolFrags(mol)) == 1
                    same = canonical_connectivity_smiles(mol) == dataset.smiles[index]
                except Exception:  # noqa: BLE001
                    sanitizable, single, same = False, False, False
                row[f"{name}_valid"] = bool(sanitizable and single)
                row[f"{name}_graph_exact"] = bool(atoms_ok and np.array_equal(bonds, true_bonds))
                row[f"{name}_exact"] = bool(sanitizable and single and same)
                row[f"{name}_bond_errors"] = int((pred_pairs != true_pairs).sum())
                row[f"{name}_false_bonds"] = int(((pred_pairs > 0) & (true_pairs == 0)).sum())
                row[f"{name}_missed_bonds"] = int(((pred_pairs == 0) & (true_pairs > 0)).sum())
                row[f"{name}_wrong_order"] = int(((pred_pairs > 0) & (true_pairs > 0) & (pred_pairs != true_pairs)).sum())
            row["constrained_connected"] = bool(info["connected"])
            row["n_pairs"] = int(len(true_pairs))
            row["failure"] = classify_failure(row)
            pair_error_rates.append(row["constrained_bond_errors"] / max(1, row["n_pairs"]))
            rows.append(row)
    if was_training:
        model.train()
    return {"rows": rows, "confusion": {k: v.tolist() for k, v in confusion.items()},
            "pair_error_rates": pair_error_rates}


def classify_failure(row: dict[str, Any]) -> str:
    """Mutually exclusive outcome of the constrained reconstruction."""
    if row["constrained_exact"]:
        return "exact"
    if not row["decodable"]:
        return "symmetry_unreachable"
    if not row["atoms_exact"]:
        return "decodable_atom_error"
    if not row["constrained_valid"]:
        return "decodable_invalid_assembly"
    if row["constrained_wrong_order"] and not (row["constrained_false_bonds"] or row["constrained_missed_bonds"]):
        return "decodable_wrong_bond_order_only"
    return "decodable_wrong_connectivity"


def _rate(flags: Iterable[bool]) -> float:
    flags = list(flags)
    return float(np.mean(flags)) if flags else float("nan")


def stratified_table(rows: list[dict[str, Any]], key, order: Sequence[str] | None = None) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    names = list(order) if order else sorted(groups)
    table = []
    for name in names:
        members = groups.get(name, [])
        if not members:
            continue
        exact = sum(r["constrained_exact"] for r in members)
        low, high = wilson_interval(exact, len(members))
        table.append({
            "group": name, "molecules": len(members),
            "constrained_exact": exact / len(members), "constrained_exact_ci95": [low, high],
            "raw_exact": _rate(r["raw_exact"] for r in members),
            "constrained_valid": _rate(r["constrained_valid"] for r in members),
            "atoms_exact": _rate(r["atoms_exact"] for r in members),
            "decodable_fraction": _rate(r["decodable"] for r in members),
            "mean_bond_errors": float(np.mean([r["constrained_bond_errors"] for r in members])),
        })
    return table


def summarize_failures(analysis: dict[str, Any]) -> dict[str, Any]:
    rows = analysis["rows"]
    total = len(rows)
    taxonomy = Counter(r["failure"] for r in rows)
    reachable = sum(r["decodable"] for r in rows)
    exact = taxonomy.get("exact", 0)
    return {
        "molecules": total,
        "constrained_exact": exact / total if total else float("nan"),
        "constrained_exact_ci95": list(wilson_interval(exact, total)),
        "raw_exact": _rate(r["raw_exact"] for r in rows),
        "constrained_validity": _rate(r["constrained_valid"] for r in rows),
        "raw_validity": _rate(r["raw_valid"] for r in rows),
        "ceiling_decodable_fraction": reachable / total if total else float("nan"),
        "ceiling_ci95": list(wilson_interval(reachable, total)),
        "exact_among_decodable": (exact / reachable) if reachable else float("nan"),
        "exact_among_symmetry_unreachable": _rate(r["constrained_exact"] for r in rows if not r["decodable"]),
        "failure_taxonomy": {k: v / total for k, v in sorted(taxonomy.items(), key=lambda kv: -kv[1])},
        "failure_taxonomy_counts": dict(sorted(taxonomy.items(), key=lambda kv: -kv[1])),
        "by_size": stratified_table(rows, lambda r: size_bucket(r["n_node"]),
                                    [f"{a}-{b}" for a, b in SIZE_BUCKETS]),
        "by_rings": stratified_table(rows, lambda r: ring_bucket(r["n_rings"]), ["0", "1", "2", "3", "4+"]),
        "by_fused": stratified_table(rows, lambda r: "fused rings" if r["fused_rings"] else
                                     ("isolated rings" if r["n_rings"] else "acyclic")),
        "by_aromaticity": stratified_table(rows, lambda r: aromatic_bucket(r["aromatic_fraction"]),
                                           ["none", "partial (<=50%)", "mostly (>50%)"]),
        "by_charge": stratified_table(rows, charge_bucket,
                                      ["neutral", "charged, net neutral", "charged, net ion"]),
        "by_symmetry": stratified_table(rows, symmetry_bucket,
                                        ["decodable", "symmetric but decodable", "symmetry-unreachable"]),
        "unreachable_by_source": {
            source: sum(1 for r in rows if not r["decodable"] and r["symmetry_source"] == source)
            for source in ("ring_positions", "acyclic_chains", "unknown")
        },
        "by_symmetric_atoms": stratified_table(
            rows, lambda r: "0" if r["symmetric_atoms"] == 0 else
            ("1-4" if r["symmetric_atoms"] <= 4 else "5-8" if r["symmetric_atoms"] <= 8 else "9+"),
            ["0", "1-4", "5-8", "9+"]),
    }


def bond_metric_diagnostics(analysis: dict[str, Any]) -> dict[str, Any]:
    """Is high bond accuracy inflated by the no-bond class?  Per-class and per-molecule view."""
    rows = analysis["rows"]
    out: dict[str, Any] = {}
    names = ("no_bond", "single", "double", "triple")
    for decoding, matrix in analysis["confusion"].items():
        confusion = np.asarray(matrix, dtype=np.float64)
        total = confusion.sum()
        accuracy = float(np.trace(confusion) / total)
        majority = float(confusion[0].sum() / total)
        per_class = {}
        recalls = []
        for k, name in enumerate(names):
            tp = confusion[k, k]
            support = confusion[k].sum()
            predicted = confusion[:, k].sum()
            recall = tp / support if support else float("nan")
            precision = tp / predicted if predicted else float("nan")
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
            per_class[name] = {"support": int(support), "precision": precision, "recall": recall, "f1": f1}
            if support:
                recalls.append(recall)
        real = confusion[1:].sum()
        errors_among_real = real - np.trace(confusion[1:, 1:])
        out[decoding] = {
            "undirected_pairs": int(total),
            "pair_accuracy": accuracy,
            "majority_class_baseline_accuracy": majority,
            "accuracy_gain_over_baseline": accuracy - majority,
            "balanced_accuracy": float(np.mean(recalls)),
            "real_bond_pairs": int(real),
            "real_bond_error_rate": float(errors_among_real / real) if real else float("nan"),
            "error_rate_all_pairs": 1.0 - accuracy,
            "per_class": per_class,
            "confusion_true_by_predicted": confusion.astype(int).tolist(),
        }
    pairs = np.asarray([r["n_pairs"] for r in rows], dtype=np.float64)
    errors = np.asarray([r["constrained_bond_errors"] for r in rows], dtype=np.float64)
    atoms_ok = np.asarray([r["atoms_exact"] for r in rows], dtype=bool)
    per_pair_error = float(errors.sum() / pairs.sum())
    # If errors were independent per pair, the exact-match rate would be
    # mean((1-p)^pairs): high pair accuracy does not imply high exact match.
    independence = float(np.mean((1 - per_pair_error) ** pairs))
    out["per_molecule"] = {
        "mean_undirected_pairs": float(pairs.mean()),
        "mean_bond_errors_constrained": float(errors.mean()),
        "molecules_with_zero_bond_errors": float(np.mean(errors == 0)),
        "molecules_with_1_to_2_bond_errors": float(np.mean((errors >= 1) & (errors <= 2))),
        "molecules_with_3_plus_bond_errors": float(np.mean(errors >= 3)),
        "atoms_exact_fraction": float(atoms_ok.mean()),
        "pair_error_rate": per_pair_error,
        "exact_graph_rate_if_pair_errors_were_independent": independence,
        "observed_constrained_graph_exact": _rate(r["constrained_graph_exact"] for r in rows),
        "note": ("An exact molecule needs ALL ~%d pairs right; at the observed pair error rate "
                 "independence would predict %.1f%% exact. Errors are clustered in symmetric "
                 "molecules, so the observed rate differs." % (int(round(pairs.mean())), 100 * independence)),
    }
    return out


# ---------------------------------------------------------------------------
# Ceiling assumption checks (no model needed).
# ---------------------------------------------------------------------------
def rdkit_symmetry_classes(smiles: str, n_atoms: int) -> list[int] | None:
    """RDKit topological symmetry classes in the SAME atom order as our graph."""
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() != n_atoms:
        return None
    return [int(c) for c in Chem.CanonicalRankAtoms(molecule, breakTies=False)]


def _same_partition(a: Sequence[int], b: Sequence[int]) -> bool:
    mapping_ab: dict[int, int] = {}
    mapping_ba: dict[int, int] = {}
    for x, y in zip(a, b):
        if mapping_ab.setdefault(x, y) != y or mapping_ba.setdefault(y, x) != x:
            return False
    return True


def uniform_graph_rescue(atom_types: np.ndarray, bond_matrix: np.ndarray, smiles: str,
                         max_mixed_classes: int = 14) -> str:
    """Could ANY colour-pair-uniform graph be the same MOLECULE as the target?

    A pairwise decoder on equivariant embeddings emits one bond order per
    unordered pair of colour classes, so every class pair is "all bonded at
    order k" or "none". If the target is not uniform, the decoder cannot
    reproduce it as a labelled graph; but exact match is judged on molecule
    identity, which is invariant to relabelling. This enumerates every uniform
    choice for the mixed class pairs and tests whether any of them yields the
    target's canonical SMILES.

    Returns "rescued", "not_rescued", "undetermined" (search too large), or
    "not_applicable" when the molecule is already decodable (no ambiguity to rescue).
    """
    atoms = np.asarray(atom_types)
    if is_decodable(atoms, bond_matrix):
        return "not_applicable"
    colors = refine_colors(atoms, bond_matrix)
    n = len(atoms)
    members: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            members[(min(colors[i], colors[j]), max(colors[i], colors[j]))].append((i, j))
    fixed: list[tuple[int, int, int]] = []
    mixed: list[tuple[list[tuple[int, int]], set[int]]] = []
    for pairs in members.values():
        orders = {int(bond_matrix[i, j]) for i, j in pairs}
        if len(orders) == 1:
            fixed.extend((i, j, orders.copy().pop()) for i, j in pairs)
        else:
            mixed.append((pairs, orders))
    if len(mixed) > max_mixed_classes:
        return "undetermined"
    choices = [sorted(orders | {0}) for _, orders in mixed]       # each mixed class: all-k for k in its orders or none
    if math.prod(len(c) for c in choices) > 20000:
        return "undetermined"
    for combination in itertools.product(*choices):
        trial = np.zeros((n, n), dtype=np.int64)
        for i, j, order in fixed:
            trial[i, j] = trial[j, i] = order
        for (pairs, _), order in zip(mixed, combination):
            for i, j in pairs:
                trial[i, j] = trial[j, i] = order
        try:
            with rdBase.BlockLogs():
                mol = graph_to_mol_charge_aware(atoms, trial)
            if len(Chem.GetMolFrags(mol)) == 1 and canonical_connectivity_smiles(mol) == smiles:
                return "rescued"
        except Exception:  # noqa: BLE001
            continue
    return "not_rescued"


def ceiling_assumption_report(dataset: MolecularGraphDataset, *, indices: Sequence[int] | None = None,
                              rescue_sample: int = 300) -> dict[str, Any]:
    order = list(range(len(dataset))) if indices is None else [int(i) for i in indices]
    decodable_flags = []
    agreement = []
    class_gap: list[int] = []
    non_decodable: list[int] = []
    for index in order:
        atoms, bonds = dataset.true_graph(index)
        flag = is_decodable(atoms, bonds)
        decodable_flags.append(flag)
        if not flag:
            non_decodable.append(index)
        colors = refine_colors(atoms, bonds)
        reference = rdkit_symmetry_classes(dataset.smiles[index], len(atoms))
        if reference is not None:
            # RDKit classes ignore bond order/charge differences that 1-WL (with
            # orders and charged atom types) keeps; compare class COUNTS and partitions.
            agreement.append(_same_partition(colors, reference))
            class_gap.append(len(set(colors)) - len(set(reference)))
    k = int(sum(decodable_flags))
    low, high = wilson_interval(k, len(decodable_flags))
    rescue = Counter()
    for index in non_decodable[:rescue_sample]:
        atoms, bonds = dataset.true_graph(index)
        rescue[uniform_graph_rescue(atoms, bonds, dataset.smiles[index])] += 1
    sampled = sum(rescue.values())
    census = symmetry_source_census(dataset, indices=order)
    return {
        "molecules": len(order),
        "unreachable_by_source": census["unreachable_by_source"],
        "unreachable_share_of_unreachable": census["share_of_unreachable"],
        "ceiling": k / len(order) if order else float("nan"),
        "ceiling_ci95": [low, high],
        "wl_partition_equals_rdkit_symmetry_classes": float(np.mean(agreement)) if agreement else float("nan"),
        "wl_extra_classes_vs_rdkit_mean": float(np.mean(class_gap)) if class_gap else float("nan"),
        "wl_extra_classes_vs_rdkit_max": int(max(class_gap)) if class_gap else 0,
        "non_decodable_molecules": len(non_decodable),
        "identity_level_rescue_tested": sampled,
        "identity_level_rescue": dict(rescue),
        "identity_level_rescue_fraction": (rescue.get("rescued", 0) / sampled) if sampled else float("nan"),
        "assumptions": [
            "encoder is permutation-equivariant and message-passing based, hence at most as expressive as 1-WL",
            "node features are deterministic one-hot atom identity (no noise, no positions)",
            "the decoder maps ONE PAIR of node embeddings to a bond class",
            "exact match is judged on the canonical SMILES (checked: identity-level rescue)",
            "1-WL classes coincide with true symmetry classes (checked against RDKit)",
        ],
    }


def symmetry_source_census(dataset: MolecularGraphDataset, *, indices: Sequence[int] | None = None) -> dict[str, Any]:
    """Model-free: how are the symmetry-unreachable molecules distributed by source?"""
    order = list(range(len(dataset))) if indices is None else [int(i) for i in indices]
    counts: Counter = Counter()
    ring_aromatic = Counter()
    for index in order:
        atoms, bonds = dataset.true_graph(index)
        source = symmetry_source(dataset.smiles[index], atoms, bonds)
        counts[source] += 1
    total = len(order)
    unreachable = total - counts.get("none", 0)
    return {
        "molecules": total,
        "decodable_fraction": counts.get("none", 0) / total if total else float("nan"),
        "unreachable_fraction": unreachable / total if total else float("nan"),
        "unreachable_by_source": {k: v for k, v in counts.items() if k != "none"},
        "share_of_unreachable": {k: v / unreachable for k, v in counts.items() if k != "none"} if unreachable else {},
    }


# ---------------------------------------------------------------------------
# Motifs behind the symmetry ceiling.
# ---------------------------------------------------------------------------
MOTIFS = {
    "morpholine": "C1COCCN1",
    "piperazine": "N1CCNCC1",
    "piperidine": "N1CCCCC1",
    "pyrrolidine": "N1CCCC1",
    "cyclohexane": "C1CCCCC1",
    "para-disubstituted benzene": "[!#1;!c]-c1ccc(-[!#1;!c])cc1",
    "ortho-disubstituted benzene": "[!#1;!c]-c1c(-[!#1;!c])cccc1",
}


def motif_census(dataset: MolecularGraphDataset, *, indices: Sequence[int] | None = None) -> dict[str, Any]:
    """Fraction of decodable / symmetry-unreachable molecules containing each motif."""
    order = list(range(len(dataset))) if indices is None else [int(i) for i in indices]
    patterns = {name: Chem.MolFromSmarts(smarts) for name, smarts in MOTIFS.items()}
    hits = {name: [0, 0] for name in patterns}
    any_hit = [0, 0]
    totals = [0, 0]
    for index in order:
        atoms, bonds = dataset.true_graph(index)
        group = 0 if is_decodable(atoms, bonds) else 1
        totals[group] += 1
        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(dataset.smiles[index])
        if molecule is None:
            continue
        found = False
        for name, pattern in patterns.items():
            if molecule.HasSubstructMatch(pattern):
                hits[name][group] += 1
                found = True
        any_hit[group] += int(found)
    def frac(count: int, total: int) -> float:
        return count / total if total else float("nan")
    return {
        "decodable_molecules": totals[0], "unreachable_molecules": totals[1],
        "motifs": {name: {"decodable": frac(h[0], totals[0]), "unreachable": frac(h[1], totals[1])}
                   for name, h in hits.items()},
        "any_motif": {"decodable": frac(any_hit[0], totals[0]), "unreachable": frac(any_hit[1], totals[1])},
    }


# ---------------------------------------------------------------------------
# Architecture-specific ceiling check: does THIS encoder behave like 1-WL?
# ---------------------------------------------------------------------------
@torch.no_grad()
def embedding_class_check(model, dataset: MolecularGraphDataset, *, indices: Sequence[int] | None = None,
                          device: str | torch.device = "cpu", eval_seed: int | None = 0,
                          relative_tolerance: float = 1e-3) -> dict[str, Any]:
    """Within a 1-WL class, are embeddings identical (necessary for the ceiling)? Between
    classes, are they separated (does the architecture reach 1-WL, i.e. is the ceiling tight)?

    Distances are relative to the mean embedding norm of the molecule.
    """
    was_training = model.training
    model.eval()
    order = list(range(len(dataset))) if indices is None else [int(i) for i in indices]
    generator = None
    if eval_seed is not None and model.config.symmetry_noise_dim > 0:
        generator = torch.Generator(device=device).manual_seed(int(eval_seed))
    within_identical = []
    separated = []
    within_max, between_min = [], []
    consumed = 0
    for batch in iter_eval_batches(dataset, 64, order):
        members = order[consumed:consumed + batch.batch_size]
        consumed += batch.batch_size
        nodes = model.encode(batch.to(device), generator=generator).cpu()
        for b, index in enumerate(members):
            atoms, bonds = dataset.true_graph(index)
            n = len(atoms)
            colors = np.asarray(refine_colors(atoms, bonds))
            e = nodes[b, :n]
            scale = float(e.norm(dim=-1).mean()) + 1e-12
            distance = torch.cdist(e, e).numpy() / scale
            same = colors[:, None] == colors[None, :]
            off_diagonal = ~np.eye(n, dtype=bool)
            within = distance[same & off_diagonal]
            between = distance[~same]
            if within.size:
                within_max.append(float(within.max()))
                within_identical.append(bool(within.max() <= relative_tolerance))
            if between.size:
                between_min.append(float(between.min()))
                separated.append(bool(between.min() > relative_tolerance))
    if was_training:
        model.train()

    def rate(flags: list[bool]) -> float:
        return float(np.mean(flags)) if flags else float("nan")

    return {
        "molecules": len(order),
        "relative_tolerance": relative_tolerance,
        "molecules_with_equal_class_atoms": len(within_max),
        "within_class_embeddings_identical": rate(within_identical),
        "within_class_distance_max_relative": float(np.max(within_max)) if within_max else float("nan"),
        "within_class_distance_median_relative": float(np.median(within_max)) if within_max else float("nan"),
        "all_classes_separated": rate(separated),
        "between_class_min_distance_median_relative": float(np.median(between_min)) if between_min else float("nan"),
        "between_class_min_distance_p01_relative": float(np.quantile(between_min, 0.01)) if between_min else float("nan"),
        "reading": ("Deterministic equivariant model: within_class_embeddings_identical ~ 1 confirms the ceiling's "
                    "necessary direction; all_classes_separated ~ 1 means this architecture reaches 1-WL, so the "
                    "ceiling is tight for it. A noise model should show within_class_embeddings_identical ~ 0."),
    }


# ---------------------------------------------------------------------------
# Flow-relevant properties of the embeddings (the flow models these).
# ---------------------------------------------------------------------------
@torch.no_grad()
def _real_node_embeddings(model, dataset, order, device, seed, batch_size=128):
    generator = None
    if seed is not None and model.config.symmetry_noise_dim > 0:
        generator = torch.Generator(device=device).manual_seed(int(seed))
    chunks = []
    for batch in iter_eval_batches(dataset, batch_size, order):
        batch = batch.to(device)
        nodes = model.encode(batch, generator=generator)
        chunks.append(nodes[batch.node_mask].cpu())
    return torch.cat(chunks)


def embedding_statistics(model, dataset: MolecularGraphDataset, *, indices: Sequence[int] | None = None,
                         device: str | torch.device = "cpu") -> dict[str, Any]:
    """Scale, effective dimensionality, and how much of the embedding is random draw.

    `noise_variance_fraction` = share of total embedding variance that changes when only
    the encoder's noise seed changes (0 for a deterministic encoder). It is the part of the
    density the flow must model that carries no molecular information.
    """
    was_training = model.training
    model.eval()
    order = list(range(len(dataset))) if indices is None else [int(i) for i in indices]
    first = _real_node_embeddings(model, dataset, order, device, seed=0)
    second = _real_node_embeddings(model, dataset, order, device, seed=1)
    if was_training:
        model.train()
    centred = first - first.mean(0, keepdim=True)
    covariance = (centred.T @ centred) / max(1, len(first) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp(min=0)
    total_variance = float(eigenvalues.sum())
    participation = float(eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum().clamp(min=1e-30))
    norms = first.norm(dim=-1)
    return {
        "nodes": int(len(first)),
        "dimension": int(first.shape[1]),
        "per_dimension_std_mean": float(first.std(0).mean()),
        "per_dimension_std_max": float(first.std(0).max()),
        "norm_mean": float(norms.mean()), "norm_std": float(norms.std()),
        "total_variance": total_variance,
        "participation_ratio": participation,
        "condition_number_top_over_median": float(eigenvalues.max() / eigenvalues.median().clamp(min=1e-30)),
        "noise_variance_fraction": float(((first - second) ** 2).sum(1).mean() / (2 * max(total_variance, 1e-30))),
    }


@torch.no_grad()
def embedding_robustness(model, dataset: MolecularGraphDataset, *, indices: Sequence[int] | None = None,
                         device: str | torch.device = "cpu",
                         sigmas: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.4),
                         seed: int = 0, batch_size: int = 128,
                         use_degree_decoder: bool | None = None) -> list[dict[str, Any]]:
    """How much embedding error can the decoder absorb?

    A flow never reproduces embeddings exactly; its samples are perturbed relative to real
    ones. This adds Gaussian noise of `sigma` times each dimension's std to real-node
    embeddings and reports free-running constrained decoding validity and exact match.
    """
    was_training = model.training
    model.eval()
    if use_degree_decoder is None:
        use_degree_decoder = bool(getattr(model.config, "degree_head", False))
    from zinc_gnf.decoding_charge_aware import degree_constrained_bond_decode
    order = list(range(len(dataset))) if indices is None else [int(i) for i in indices]
    reference = _real_node_embeddings(model, dataset, order, device, seed=0)
    per_dim_std = reference.std(0).to(device)
    results = []
    for sigma in sigmas:
        noise_generator = torch.Generator(device=device).manual_seed(seed + 17)
        encode_generator = (torch.Generator(device=device).manual_seed(0)
                            if model.config.symmetry_noise_dim > 0 else None)
        valid = exact = total = 0
        consumed = 0
        for batch in iter_eval_batches(dataset, batch_size, order):
            members = order[consumed:consumed + batch.batch_size]
            consumed += batch.batch_size
            batch = batch.to(device)
            nodes = model.encode(batch, generator=encode_generator)
            if sigma > 0:
                perturbation = torch.randn(nodes.shape, device=device, generator=noise_generator)
                nodes = nodes + sigma * per_dim_std * perturbation * batch.node_mask.unsqueeze(-1)
            out = model.decode(nodes, batch, teacher_forcing=False)
            atoms_all = out["atom_logits"].argmax(-1).cpu().numpy()
            probs_all = torch.softmax(out["bond_logits"], dim=-1).cpu().numpy()
            degrees_all = out["degree_logits"].argmax(-1).cpu().numpy() if use_degree_decoder else None
            for b, index in enumerate(members):
                n = int(batch.n_node[b])
                atoms = atoms_all[b, :n]
                total += 1
                try:
                    if use_degree_decoder:
                        bonds, _ = degree_constrained_bond_decode(atoms, probs_all[b, :n, :n], degrees_all[b, :n])
                    else:
                        bonds, _ = valence_aware_bond_decode_typed(atoms, probs_all[b, :n, :n])
                    with rdBase.BlockLogs():
                        molecule = graph_to_mol_charge_aware(atoms, bonds)
                    single = len(Chem.GetMolFrags(molecule)) == 1
                    valid += int(single)
                    exact += int(single and canonical_connectivity_smiles(molecule) == dataset.smiles[index])
                except Exception:  # noqa: BLE001
                    pass
        results.append({"sigma": float(sigma), "validity": valid / total, "exact": exact / total,
                        "decoder": "degree" if use_degree_decoder else "constrained"})
    if was_training:
        model.train()
    return results


def noise_draw_spread(model, dataset: MolecularGraphDataset, *, indices: Sequence[int] | None = None,
                      device: str | torch.device = "cpu", draws: int = 5) -> dict[str, Any]:
    """Constrained exact-match across several encoder-noise seeds (0 spread if deterministic)."""
    from zinc_gnf.autoencoder import evaluate_reconstruction

    values = [evaluate_reconstruction(model, dataset, indices=indices, device=device,
                                      compute_ceiling=False, eval_seed=seed)["constrained"]["exact_molecule_match"]
              for seed in range(draws)]
    return {"draws": draws, "exact_values": values, "mean": float(np.mean(values)),
            "std": float(np.std(values)), "min": float(np.min(values)), "max": float(np.max(values))}


__all__ = [
    "MOTIFS",
    "embedding_class_check",
    "embedding_robustness",
    "embedding_statistics",
    "motif_census",
    "noise_draw_spread",
    "analyze_reconstructions",
    "mixed_class_pairs",
    "symmetry_source",
    "symmetry_source_census",
    "aromatic_bucket",
    "bond_metric_diagnostics",
    "ceiling_assumption_report",
    "charge_bucket",
    "classify_failure",
    "describe_molecule",
    "rdkit_symmetry_classes",
    "ring_bucket",
    "size_bucket",
    "stratified_table",
    "summarize_failures",
    "symmetry_bucket",
    "uniform_graph_rescue",
    "wilson_interval",
]
