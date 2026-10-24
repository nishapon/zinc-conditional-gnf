"""Stage 20: does a QED difference survive the decoder, and what does the degree
integer program do to it?

TWO QUESTIONS, DELIBERATELY SEPARATED
-------------------------------------
1. HELD-OUT TRANSFER. Stage 19's transfer control used TRAINING molecules, which the
   autoencoder was fitted on. A decoder that preserves 94-99% of a QED gap on its own
   training embeddings has not been shown to do so on molecules it never saw.
   `transfer_on_split` repeats the control on any split - validation here - with
   per-molecule values, so the preservation ratio gets a bootstrap confidence interval
   instead of being a bare ratio of two group means, and with aromaticity and per-decoder
   validity beside it.

2. WHAT THE DEGREE DECODER DOES. An exact identity makes this attributable rather than
   speculative. `degree_constrained_bond_decode` returns, on EVERY fallback path
   (`bad_degrees`, `no_solver`, `infeasible_degrees`, `valence_unfixable`,
   `not_connected`), precisely `valence_aware_bond_decode_typed(atoms, probs, caps)` -
   which is the `constrained` decoder. So on any molecule where the degree decoder fell
   back in BOTH conditions, its QED response is identical to the constrained decoder's by
   construction, and cannot contribute to a difference between them. That splits the
   common valid subset into three strata whose contributions ADD UP exactly:

     fallback_both   degree == constrained by construction; contributes exactly 0
     solved_both     the integer program ran in both conditions: the PURE IP effect
     mixed           it ran in one condition only: a decoder-SWITCHING effect

   and because
       gap = mean(d_constrained) - mean(d_degree)
           = sum over strata of (stratum share) x (stratum gap),
   the attenuation decomposes by the law of total expectation with no modelling
   assumption. A measured `fallback_identity_max_deviation` of 0 is the self-check that
   the identity really holds in the code as run.

   One further mechanism is measured directly. The integer program constrains the bond
   SET to match the predicted degree sequence. If the predicted degrees do not change
   when the QED request changes, the constraint is literally identical in both conditions,
   the skeleton is frozen, and the only channel left for QED to move through is the bond
   ORDER chosen within that fixed set - a far narrower channel than the constrained
   decoder, which lets the bond set itself respond. `degree_prediction_change_rate`,
   `bond_set_change_rate` and `bond_order_change_rate` separate exactly that.

Raw-argmax validity is carried through unchanged and reported separately, so native raw
validity is never conflated with validity after constrained or degree assembly.

Everything is read-only: no checkpoint is written, no model is trained, and the test split
is never touched.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np
from rdkit import Chem, rdBase
import torch

from zinc_gnf.decoding_charge_aware import (
    degree_constrained_bond_decode,
    valence_aware_bond_decode_typed,
)
from zinc_gnf.generation import DECODERS, generation_batch
# `_molecule_record` is reused on purpose: "valid" must keep meaning exactly what it means
# everywhere else in this project (sanitizable AND a single connected fragment).
# Re-defining validity here would make stage 20 incomparable to stages 17-19.
from zinc_gnf.generation import _molecule_record

EXAMPLE_MOVE_THRESHOLD = 0.05
EXAMPLE_FLAT_THRESHOLD = 0.01


def _ring_statistics(smiles: str | None) -> dict[str, Any]:
    """Rings and aromatic rings, counted exactly as `molecular_profile` counts them."""
    if not smiles:
        return {"rings": None, "aromatic_rings": None}
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return {"rings": None, "aromatic_rings": None}
    info = molecule.GetRingInfo()
    aromatic = sum(1 for ring in info.AtomRings()
                   if all(molecule.GetAtomWithIdx(i).GetIsAromatic() for i in ring))
    return {"rings": int(info.NumRings()), "aromatic_rings": int(aromatic)}


@torch.no_grad()
def decode_with_graphs(autoencoder, embeddings: torch.Tensor, counts: Sequence[int], *,
                       decoders: Sequence[str] = DECODERS) -> list[dict[str, Any]]:
    """Decode a population and KEEP the bond matrices.

    `zinc_gnf.generation.decode_embeddings` returns molecules but discards the graphs, and
    stage 20 has to compare WHICH bonds two decoders chose - not only the QED that came
    out of them. The decoder call, the symmetrisation of the raw argmax and the validity
    definition are the same as in `generation.decode_embeddings`.
    """
    decoders = tuple(decoders)
    unknown = set(decoders) - set(DECODERS)
    if unknown:
        raise ValueError(f"unknown decoders {sorted(unknown)}; known: {list(DECODERS)}")
    has_degree = bool(autoencoder.config.degree_head)
    if "degree" in decoders and not has_degree:
        raise ValueError("the autoencoder has no degree head, so the degree decoder "
                         "is unavailable")
    device = embeddings.device
    batch = generation_batch([int(c) for c in counts], device=device)
    if embeddings.shape[:2] != batch.node_mask.shape:
        raise ValueError(f"embeddings {tuple(embeddings.shape)} do not match the node counts")
    finite_nodes = torch.isfinite(embeddings).all(dim=-1)
    safe = torch.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    was_training = autoencoder.training
    autoencoder.eval()
    outputs = autoencoder.decode(safe, batch, teacher_forcing=False)
    if was_training:
        autoencoder.train()

    atom_predictions = outputs["atom_logits"].argmax(-1).cpu().numpy()
    probabilities = torch.softmax(outputs["bond_logits"], dim=-1).cpu().numpy()
    bond_predictions = outputs["bond_logits"].argmax(-1).cpu().numpy()
    degree_predictions = (outputs["degree_logits"].argmax(-1).cpu().numpy()
                          if has_degree and "degree_logits" in outputs else None)

    records: list[dict[str, Any]] = []
    for index, count in enumerate((int(c) for c in counts)):
        entry: dict[str, Any] = {"n_node": count,
                                 "nonfinite_embedding": not bool(
                                     finite_nodes[index, :count].all())}
        if entry["nonfinite_embedding"]:
            for name in decoders:
                entry[name] = {"valid": False, "smiles": None, "qed": None, "rings": None,
                               "aromatic_rings": None, "bond_matrix": None,
                               "fallback": False, "fallback_reason": "nonfinite_embedding",
                               "connected": False, "fragments": None, "bonds": 0}
            records.append(entry)
            continue
        atoms = atom_predictions[index, :count]
        probs = probabilities[index, :count, :count]
        entry["atom_types"] = atoms.tolist()
        degrees = None
        if degree_predictions is not None:
            degrees = degree_predictions[index, :count]
            entry["predicted_degrees"] = degrees.tolist()
        for name in decoders:
            if name == "raw":
                bonds = bond_predictions[index, :count, :count].copy()
                np.fill_diagonal(bonds, 0)
                bonds = np.minimum(bonds, bonds.T)
                info: dict[str, Any] = {}
            elif name == "constrained":
                bonds, info = valence_aware_bond_decode_typed(atoms, probs)
            else:
                bonds, info = degree_constrained_bond_decode(atoms, probs, degrees)
            molecule = _molecule_record(atoms, bonds)
            molecule.update(_ring_statistics(molecule.get("smiles")))
            molecule["fallback"] = bool(info.get("fallback", False))
            molecule["fallback_reason"] = info.get("fallback_reason")
            molecule["bonds"] = int((np.asarray(bonds) > 0).sum() // 2)
            molecule["bond_matrix"] = np.asarray(bonds, dtype=np.int64)
            entry[name] = molecule
        records.append(entry)
    return records


# ------------------------------------------------- 1. transfer on a chosen split ---
@torch.no_grad()
def transfer_on_split(autoencoder, dataset, *, n: int, qed_low: float, qed_high: float,
                      window: float = 0.05, max_molecules: int = 256,
                      device: torch.device | str = "cpu",
                      decoders: Sequence[str] = DECODERS, split_name: str = "validation",
                      seed: int = 20261024) -> dict[str, Any]:
    """Encode REAL molecules at two QED targets and decode them: is the gap preserved?

    Per-molecule decoded QED is kept, so the preservation ratio comes with a bootstrap
    interval and the true QED of the SURVIVING molecules is reported next to the true QED
    of all selected ones - a decoder that only manages to build the high-QED molecules
    would otherwise look like one that preserves the gap.
    """
    from zinc_gnf.dataset import iter_eval_batches
    from zinc_gnf.qed_probes import group_ratio_bootstrap_ci

    device = torch.device(device)
    counts = np.asarray(dataset.n_node)
    qed = np.asarray(dataset.qed, dtype=np.float64)
    result: dict[str, Any] = {
        "split": str(split_name), "n": int(n), "qed_low": float(qed_low),
        "qed_high": float(qed_high), "window": float(window), "groups": {},
        "note": ("molecules this autoencoder was NOT fitted on, when split is validation; "
                 "the test split is never loaded"),
    }
    per_molecule: dict[str, dict[str, list[float]]] = {
        name: {"low_decoded": [], "low_true": [], "high_decoded": [], "high_true": []}
        for name in decoders}
    for label, target in (("low", qed_low), ("high", qed_high)):
        selected = np.flatnonzero((counts == int(n)) & (np.abs(qed - target) <= window))
        group: dict[str, Any] = {"requested_qed": float(target),
                                 "molecules_available": int(selected.size)}
        if selected.size == 0:
            group["note"] = "no molecule in this split at this (N, QED)"
            result["groups"][label] = group
            continue
        chosen = [int(i) for i in selected[:int(max_molecules)]]
        group["molecules_used"] = len(chosen)
        group["true_qed_mean"] = float(qed[chosen].mean())
        group["true_qed_sd"] = float(qed[chosen].std())
        embeddings, masks, sizes, truth = [], [], [], []
        for batch in iter_eval_batches(dataset, 64, chosen):
            batch = batch.to(device)
            embeddings.append(autoencoder.encode(batch))
            masks.append(batch.node_mask)
            sizes.extend(int(c) for c in batch.n_node.tolist())
            truth.extend(float(v) for v in batch.qed.tolist())
        width = max(e.shape[1] for e in embeddings)
        padded = torch.cat([torch.nn.functional.pad(e, (0, 0, 0, width - e.shape[1]))
                            for e in embeddings])
        records = decode_with_graphs(autoencoder, padded, sizes, decoders=decoders)
        group["decoders"] = {}
        for name in decoders:
            entries = [r[name] for r in records]
            valid = [(entry, true) for entry, true in zip(entries, truth)
                     if entry.get("valid") and entry.get("qed") is not None]
            decoded = [float(entry["qed"]) for entry, _ in valid]
            aromatic = [entry["aromatic_rings"] for entry, _ in valid
                        if entry["aromatic_rings"] is not None]
            group["decoders"][name] = {
                "validity": len(valid) / max(len(entries), 1),
                "valid_molecules": len(valid),
                "fallback_rate": float(np.mean([bool(e.get("fallback")) for e in entries]))
                if entries else float("nan"),
                "decoded_qed_mean": float(np.mean(decoded)) if decoded else float("nan"),
                "decoded_qed_sd": float(np.std(decoded)) if decoded else float("nan"),
                "aromatic_rings_mean": float(np.mean(aromatic)) if aromatic else float("nan"),
                "rings_mean": float(np.mean([e["rings"] for e, _ in valid
                                             if e["rings"] is not None]))
                if valid else float("nan"),
                "true_qed_mean_of_survivors": float(np.mean([t for _, t in valid]))
                if valid else float("nan"),
            }
            per_molecule[name][f"{label}_decoded"] = decoded
            per_molecule[name][f"{label}_true"] = [t for _, t in valid]
        result["groups"][label] = group

    low, high = result["groups"].get("low", {}), result["groups"].get("high", {})
    if low.get("decoders") and high.get("decoders"):
        result["preservation"] = {}
        for name in decoders:
            block = per_molecule[name]
            result["preservation"][name] = group_ratio_bootstrap_ci(
                block["low_decoded"], block["low_true"],
                block["high_decoded"], block["high_true"], seed=seed)
        result["true_qed_difference_all_selected"] = (
            high.get("true_qed_mean", float("nan")) - low.get("true_qed_mean", float("nan")))
    else:
        result["note_insufficient"] = "one QED group is empty in this split at this N"
    return result


# ------------------------------------------- 2. paired generation, shared noise ---
@torch.no_grad()
def paired_generated_embeddings(flow, *, scaler, n: int, qed_low: float, qed_high: float,
                                count: int, temperature: float,
                                device: torch.device | str = "cpu",
                                seed: int = 17) -> dict[str, Any]:
    """Two populations differing ONLY in the QED request, from the same noise draws.

    Identical in construction to `decoder_information.paired_intervention`, so stage 20's
    numbers sit on the same sampling procedure as stage 19's and the two are comparable.
    """
    from zinc_gnf.conditioning import condition_embedding_for, standardise_condition
    from zinc_gnf.flow import node_mask_from_counts

    device = torch.device(device)
    counts = [int(n)] * int(count)
    mask = node_mask_from_counts(counts, device=device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    noise = torch.randn((int(count), mask.shape[1], flow.config.node_embedding_dim),
                        device=device, generator=generator)
    features = flow.config.condition_features
    produced: dict[str, Any] = {}
    for label, qed in (("low", qed_low), ("high", qed_high)):
        values = torch.from_numpy(standardise_condition(
            features, scaler=scaler, n=counts, qed=[float(qed)] * int(count))).to(device)
        condition = condition_embedding_for(flow, values, None, int(count), device)
        mu, sigma = flow.prior.parameters_for(int(count), condition, device)
        z = ((mu.unsqueeze(1) + float(temperature) * sigma.unsqueeze(1) * noise)
             * mask.unsqueeze(-1).to(noise.dtype))
        x = flow.inverse(z, mask, condition)
        finite = (torch.isfinite(x).all(dim=-1) | ~mask).all(dim=-1)
        produced[label] = {"z": z, "x": x, "finite": finite}
    keep = produced["low"]["finite"] & produced["high"]["finite"]
    index = torch.nonzero(keep).flatten()
    return {
        "n": int(n), "count": int(count), "temperature": float(temperature),
        "seed": int(seed), "qed_low": float(qed_low), "qed_high": float(qed_high),
        "nonfinite_rate": float((~keep).to(torch.float64).mean()),
        "usable_molecules": int(index.numel()),
        "low": produced["low"]["x"][index], "high": produced["high"]["x"][index],
        "mask": mask[index], "counts": [int(n)] * int(index.numel()),
    }


# ------------------------------- 3. constrained vs degree on identical embeddings ---
def _upper(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def _graph_change(low: dict[str, Any], high: dict[str, Any]) -> dict[str, Any]:
    """Did the bond SET change, and did the bond ORDERS change within it?"""
    a, b = low.get("bond_matrix"), high.get("bond_matrix")
    if a is None or b is None or a.shape != b.shape:
        return {"comparable": False}
    left, right = _upper(a), _upper(b)
    present_left, present_right = left > 0, right > 0
    both = present_left & present_right
    return {
        "comparable": True,
        "pairs": int(left.size),
        "bond_set_changed_pairs": int((present_left != present_right).sum()),
        "bond_set_changed": bool((present_left != present_right).any()),
        "shared_bonds": int(both.sum()),
        "bond_order_changed_pairs": int((both & (left != right)).sum()),
        "bond_order_changed": bool((both & (left != right)).any()),
    }


def _stratum(values_constrained: list[float], values_degree: list[float], *,
             label: str, total: int, seed: int) -> dict[str, Any]:
    from zinc_gnf.qed_probes import paired_difference_ci

    block = paired_difference_ci(values_constrained, values_degree, seed=seed)
    block["stratum"] = label
    block["share_of_common_subset"] = (block["pairs"] / total) if total else float("nan")
    block["contribution_to_gap"] = (block["difference"] * block["share_of_common_subset"]
                                    if np.isfinite(block.get("difference", float("nan")))
                                    else 0.0)
    return block


@torch.no_grad()
def decoder_contrast(flow, autoencoder, *, scaler, n: int, qed_low: float, qed_high: float,
                     count: int, temperature: float, device: torch.device | str = "cpu",
                     seed: int = 17, max_examples: int = 5,
                     resamples: int | None = None,
                     paired: dict[str, Any] | None = None) -> dict[str, Any]:
    """Constrained versus degree-constrained decoding of the SAME generated embeddings.

    The comparison is made on a single common subset - molecules valid under both
    conditions AND both decoders - so a difference between the decoders can never be an
    artefact of them having succeeded on different molecules. The per-decoder own-subset
    numbers are reported beside it, and their difference IS the validity-selection effect.

    `paired` may be supplied by the caller so that the read-out measurement and this
    decoder comparison are made on literally the same embeddings rather than on two
    identically seeded regenerations of them.
    """
    from zinc_gnf.qed_probes import paired_bootstrap_ci, paired_difference_ci

    kwargs = {"seed": seed} if resamples is None else {"seed": seed, "resamples": resamples}
    if paired is None:
        paired = paired_generated_embeddings(flow, scaler=scaler, n=n, qed_low=qed_low,
                                             qed_high=qed_high, count=count,
                                             temperature=temperature, device=device,
                                             seed=seed)
    result: dict[str, Any] = {k: paired[k] for k in
                              ("n", "count", "temperature", "seed", "qed_low", "qed_high",
                               "nonfinite_rate", "usable_molecules")}
    result["requested_change"] = float(qed_high) - float(qed_low)
    if not paired["usable_molecules"]:
        result["note"] = "every sample was non-finite; nothing can be measured"
        return result

    decoders = (("raw", "constrained", "degree") if autoencoder.config.degree_head
                else ("raw", "constrained"))
    low = decode_with_graphs(autoencoder, paired["low"], paired["counts"], decoders=decoders)
    high = decode_with_graphs(autoencoder, paired["high"], paired["counts"], decoders=decoders)

    # --- native validity per decoder per condition, raw kept separate -----
    result["validity"] = {}
    for name in decoders:
        result["validity"][name] = {
            "low": float(np.mean([bool(r[name].get("valid")) for r in low])),
            "high": float(np.mean([bool(r[name].get("valid")) for r in high])),
            "valid_in_both": float(np.mean([bool(a[name].get("valid"))
                                            and bool(b[name].get("valid"))
                                            for a, b in zip(low, high)])),
        }
    result["raw_validity_note"] = ("raw is the decoder's argmax with no chemistry; it is "
                                   "reported separately and never merged with validity "
                                   "after constrained or degree assembly")
    if "degree" not in decoders:
        result["note"] = ("this autoencoder has no degree head, so there is no "
                          "degree-constrained decoder to contrast")
        return result

    # --- own-subset response per decoder (selection visible) ---------------
    result["own_subset"] = {}
    for name in decoders:
        changes = [float(b[name]["qed"]) - float(a[name]["qed"])
                   for a, b in zip(low, high)
                   if a[name].get("valid") and b[name].get("valid")
                   and a[name].get("qed") is not None and b[name].get("qed") is not None]
        result["own_subset"][name] = paired_bootstrap_ci(changes, **kwargs)

    # --- the common valid subset: the only fair comparison -----------------
    common: list[int] = [
        i for i, (a, b) in enumerate(zip(low, high))
        if all(a[name].get("valid") and b[name].get("valid")
               and a[name].get("qed") is not None and b[name].get("qed") is not None
               for name in ("constrained", "degree"))]
    result["common_subset"] = {"molecules": len(common),
                               "fraction_of_usable": len(common) /
                               max(paired["usable_molecules"], 1)}
    if not common:
        result["note"] = ("no molecule was valid under both conditions for BOTH decoders; "
                          "the decoders cannot be compared without confounding by validity")
        return result

    d_constrained = [float(high[i]["constrained"]["qed"]) - float(low[i]["constrained"]["qed"])
                     for i in common]
    d_degree = [float(high[i]["degree"]["qed"]) - float(low[i]["degree"]["qed"])
                for i in common]
    result["common_subset"]["constrained"] = paired_bootstrap_ci(d_constrained, **kwargs)
    result["common_subset"]["degree"] = paired_bootstrap_ci(d_degree, **kwargs)
    result["common_subset"]["gap_constrained_minus_degree"] = paired_difference_ci(
        d_constrained, d_degree, **kwargs)
    result["common_subset"]["selection_effect"] = {
        name: (result["common_subset"][name]["mean"] - result["own_subset"][name]["mean"])
        for name in ("constrained", "degree")}
    result["common_subset"]["selection_note"] = (
        "common-subset mean minus own-subset mean: the part of each decoder's apparent "
        "response that came from WHICH molecules survived rather than from the condition")

    # --- mechanism attribution -------------------------------------------
    fallback_low = {i: bool(low[i]["degree"].get("fallback")) for i in common}
    fallback_high = {i: bool(high[i]["degree"].get("fallback")) for i in common}
    strata = {"fallback_both": [], "solved_both": [], "mixed": []}
    for position, i in enumerate(common):
        if fallback_low[i] and fallback_high[i]:
            strata["fallback_both"].append(position)
        elif not fallback_low[i] and not fallback_high[i]:
            strata["solved_both"].append(position)
        else:
            strata["mixed"].append(position)
    decomposition = {}
    for label, positions in strata.items():
        decomposition[label] = _stratum([d_constrained[p] for p in positions],
                                        [d_degree[p] for p in positions],
                                        label=label, total=len(common), seed=seed)
    # The self-check on the identity the decomposition rests on: on `fallback_both` the
    # degree decoder IS the constrained decoder, so the two responses must agree exactly.
    identity = [abs(d_constrained[p] - d_degree[p]) for p in strata["fallback_both"]]
    attribution: dict[str, Any] = {
        "strata": decomposition,
        "fallback_identity_max_deviation": float(max(identity)) if identity else 0.0,
        "fallback_identity_holds": bool(not identity or max(identity) < 1e-9),
        "fallback_identity_note": (
            "on a fallback the degree decoder returns exactly the constrained decoder's "
            "bonds, so a non-zero deviation here would mean this attribution is invalid"),
        "sum_of_contributions": float(sum(block["contribution_to_gap"]
                                          for block in decomposition.values())),
        "fallback_rate": {
            "low": float(np.mean([bool(r["degree"].get("fallback")) for r in low])),
            "high": float(np.mean([bool(r["degree"].get("fallback")) for r in high])),
        },
        "fallback_reasons": {
            "low": dict(Counter(r["degree"].get("fallback_reason") for r in low
                                if r["degree"].get("fallback"))),
            "high": dict(Counter(r["degree"].get("fallback_reason") for r in high
                                 if r["degree"].get("fallback"))),
        },
    }

    # does the integer program's CONSTRAINT itself respond to the request?
    degree_changes = []
    for i in common:
        a, b = low[i].get("predicted_degrees"), high[i].get("predicted_degrees")
        if a is None or b is None or len(a) != len(b):
            continue
        degree_changes.append(int(sum(1 for x, y in zip(a, b) if x != y)))
    attribution["predicted_degrees"] = {
        "molecules_compared": len(degree_changes),
        "change_rate": float(np.mean([c > 0 for c in degree_changes]))
        if degree_changes else float("nan"),
        "mean_atoms_changed": float(np.mean(degree_changes)) if degree_changes
        else float("nan"),
        "note": ("if the predicted degree sequence does not change, the integer "
                 "program's constraint is IDENTICAL in both conditions: the bond set is "
                 "frozen and only bond ORDERS inside it can carry QED"),
    }
    graph_changes: dict[str, Any] = {}
    for name in ("constrained", "degree"):
        blocks = [_graph_change(low[i][name], high[i][name]) for i in common]
        usable = [b for b in blocks if b.get("comparable")]
        graph_changes[name] = {
            "molecules_compared": len(usable),
            "bond_set_change_rate": float(np.mean([b["bond_set_changed"] for b in usable]))
            if usable else float("nan"),
            "mean_bond_set_changed_pairs": float(np.mean([b["bond_set_changed_pairs"]
                                                          for b in usable]))
            if usable else float("nan"),
            "bond_order_change_rate": float(np.mean([b["bond_order_changed"] for b in usable]))
            if usable else float("nan"),
            "mean_bond_order_changed_pairs": float(np.mean([b["bond_order_changed_pairs"]
                                                            for b in usable]))
            if usable else float("nan"),
        }
    attribution["graph_response"] = graph_changes
    attribution["graph_response_note"] = (
        "bond-set change rate is how often the chosen ADJACENCY responded to the request; "
        "bond-order change rate is how often only the orders inside a shared adjacency did")

    # structural side effects on the common subset
    structure: dict[str, Any] = {}
    for name in ("constrained", "degree"):
        for key in ("aromatic_rings", "rings", "bonds"):
            values = [float(high[i][name][key]) - float(low[i][name][key]) for i in common
                      if high[i][name].get(key) is not None
                      and low[i][name].get(key) is not None]
            structure.setdefault(name, {})[f"{key}_change"] = paired_bootstrap_ci(
                values, **kwargs)
        structure[name]["connectivity"] = {
            "low": float(np.mean([bool(low[i][name].get("connected")) for i in common])),
            "high": float(np.mean([bool(high[i][name].get("connected")) for i in common])),
        }
    result["structure"] = structure
    result["attribution"] = attribution

    # --- examples: constrained moved QED, degree did not -------------------
    candidates = []
    for position, i in enumerate(common):
        if abs(d_constrained[position]) >= EXAMPLE_MOVE_THRESHOLD \
                and abs(d_degree[position]) <= EXAMPLE_FLAT_THRESHOLD:
            degrees_low = low[i].get("predicted_degrees") or []
            degrees_high = high[i].get("predicted_degrees") or []
            candidates.append({
                "sample_index": int(i),
                "constrained_qed_change": float(d_constrained[position]),
                "degree_qed_change": float(d_degree[position]),
                "degree_fallback": {"low": fallback_low[i], "high": fallback_high[i]},
                "degree_fallback_reason": {
                    "low": low[i]["degree"].get("fallback_reason"),
                    "high": high[i]["degree"].get("fallback_reason")},
                "predicted_degrees_changed_atoms": int(sum(
                    1 for x, y in zip(degrees_low, degrees_high) if x != y)),
                "constrained_graph_change": _graph_change(low[i]["constrained"],
                                                          high[i]["constrained"]),
                "degree_graph_change": _graph_change(low[i]["degree"], high[i]["degree"]),
                "smiles": {
                    "constrained_low": low[i]["constrained"].get("smiles"),
                    "constrained_high": high[i]["constrained"].get("smiles"),
                    "degree_low": low[i]["degree"].get("smiles"),
                    "degree_high": high[i]["degree"].get("smiles")},
            })
    candidates.sort(key=lambda c: abs(c["constrained_qed_change"]), reverse=True)
    result["examples"] = {
        "molecules_matching": len(candidates),
        "fraction_of_common_subset": len(candidates) / max(len(common), 1),
        "criterion": (f"|constrained QED change| >= {EXAMPLE_MOVE_THRESHOLD} and "
                      f"|degree QED change| <= {EXAMPLE_FLAT_THRESHOLD}"),
        "shown": candidates[:int(max_examples)],
    }
    return result


def summarise_attenuation(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Name the channel the measurements implicate, per temperature, without averaging.

    Reported per temperature on purpose: the stage-19 run behaved differently at T=1.0
    than at T=0.2, so a single pooled number would hide the one finding that distinguishes
    a universal constraint effect from a regime-dependent one.
    """
    by_temperature: dict[str, Any] = {}
    for row in rows:
        common = row.get("common_subset") or {}
        gap = common.get("gap_constrained_minus_degree") or {}
        if not np.isfinite(gap.get("difference", float("nan"))):
            continue
        key = f"{row['temperature']:g}"
        entry = by_temperature.setdefault(key, {"cells": [], "attenuating_cells": 0})
        attribution = row.get("attribution") or {}
        strata = attribution.get("strata") or {}
        solved = strata.get("solved_both") or {}
        mixed = strata.get("mixed") or {}
        degrees = attribution.get("predicted_degrees") or {}
        graph = (attribution.get("graph_response") or {})
        attenuating = bool(gap.get("excludes_zero")) and gap.get("difference", 0.0) > 0
        channels = []
        if attenuating:
            if solved.get("excludes_zero") and solved.get("difference", 0.0) > 0:
                channels.append("integer_programming_constraints")
            if mixed.get("excludes_zero"):
                channels.append("fallback_switching")
            set_rate = (graph.get("degree") or {}).get("bond_set_change_rate")
            reference_rate = (graph.get("constrained") or {}).get("bond_set_change_rate")
            if all(v is not None and np.isfinite(v) for v in (set_rate, reference_rate)) \
                    and reference_rate > 0 and set_rate < 0.5 * reference_rate:
                channels.append("bond_set_frozen_by_degree_sequence")
            selection = (common.get("selection_effect") or {}).get("degree")
            if selection is not None and np.isfinite(selection) \
                    and abs(selection) > 0.5 * abs(gap["difference"]):
                channels.append("validity_selection")
        entry["attenuating_cells"] += int(attenuating)
        entry["cells"].append({
            "n": row["n"],
            "common_molecules": common.get("molecules"),
            "constrained_change": (common.get("constrained") or {}).get("mean"),
            "degree_change": (common.get("degree") or {}).get("mean"),
            "gap": gap.get("difference"),
            "gap_ci": [gap.get("ci_low"), gap.get("ci_high")],
            "gap_excludes_zero": bool(gap.get("excludes_zero")),
            "attenuating": attenuating,
            "channels": channels,
            "degree_prediction_change_rate": degrees.get("change_rate"),
            "degree_fallback_rate_high": (attribution.get("fallback_rate") or {}).get("high"),
            "fallback_identity_holds": attribution.get("fallback_identity_holds"),
        })
    for key, entry in by_temperature.items():
        entry["cells_measured"] = len(entry["cells"])
        channels: Counter = Counter()
        for cell in entry["cells"]:
            channels.update(cell["channels"])
        entry["channels_implicated"] = dict(channels)
    return {
        "by_temperature": by_temperature,
        "note": ("kept per temperature because attenuation was not universal across "
                 "temperatures; 'attenuating' means the constrained-minus-degree gap is "
                 "positive and its bootstrap interval excludes zero"),
    }


__all__ = [
    "EXAMPLE_FLAT_THRESHOLD",
    "EXAMPLE_MOVE_THRESHOLD",
    "decode_with_graphs",
    "decoder_contrast",
    "paired_generated_embeddings",
    "summarise_attenuation",
    "transfer_on_split",
]
