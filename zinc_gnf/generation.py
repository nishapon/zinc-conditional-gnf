"""Sampling molecules from a trained flow through the frozen decoder.

    node counts  ~  empirical TRAIN histogram (never val/test)
    z            ~  prior, scaled by `temperature`
    embeddings   =  flow.inverse(z)                     (un-whitened)
    atoms, bond logits, degrees  =  frozen decoder
    bonds        =  raw argmax | greedy valence-aware | degree-constrained
    molecule     =  graph_to_mol_charge_aware + RDKit sanitization

Every sample is decoded by ALL THREE decoders so raw validity and validity
after degree-constrained assembly are never conflated: `raw` is the decoder's
argmax with no chemistry, `constrained` is the greedy valence-aware assembly
(the teammates' decoder, ported), `degree` is the degree-constrained integer
program with a greedy fallback. `degree` is the headline decoder for a model
whose autoencoder has a degree head.

Nothing here touches the test split, and the autoencoder is used in `eval()`
mode under `torch.no_grad()` only.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import QED
import torch

from zinc_gnf.chemistry import canonical_connectivity_smiles
from zinc_gnf.chemistry_charge_aware import graph_to_mol_charge_aware
from zinc_gnf.constants_charge_aware import MAX_NODES
from zinc_gnf.dataset import MolecularBatch
from zinc_gnf.decoding_charge_aware import (
    degree_constrained_bond_decode,
    valence_aware_bond_decode_typed,
)
from zinc_gnf.flow import NodeFlow, node_mask_from_counts

DECODERS = ("raw", "constrained", "degree")


class NodeCountSampler:
    """Empirical distribution of heavy-atom counts, taken from the TRAIN split.

    p(N) is not learned by the flow: a sample's size is drawn here and the flow
    then fills that many node vectors. Keeping p(N) explicit and train-only is
    what makes the unconditional model comparable to the later N-conditioned
    one, where N comes from the caller instead.
    """

    def __init__(self, n_node: Sequence[int] | np.ndarray) -> None:
        counts = np.asarray([int(n) for n in n_node], dtype=np.int64)
        if counts.size == 0:
            raise ValueError("need at least one node count")
        if counts.min() < 1 or counts.max() > MAX_NODES:
            raise ValueError(f"node counts must lie in 1..{MAX_NODES}")
        self.counts = counts
        values, frequencies = np.unique(counts, return_counts=True)
        self.values = values
        self.probabilities = frequencies / frequencies.sum()

    @property
    def mean(self) -> float:
        return float(self.counts.mean())

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(self.values, size=int(size), p=self.probabilities)

    def histogram(self) -> dict[int, float]:
        return {int(v): float(p) for v, p in zip(self.values, self.probabilities)}


def generation_batch(n_node: Sequence[int], device: torch.device | str = "cpu") -> MolecularBatch:
    """A batch carrying only shapes and masks: no atom types, no bonds.

    Generation knows nothing but how many atoms each molecule has, so the
    atom/bond tensors are zeros and must only ever be read through the masks.
    The decoder is called with `atom_types=None` (no teacher forcing).
    """
    mask = node_mask_from_counts(n_node, device=device)
    batch, width = mask.shape
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
    pair_mask = pair_mask & ~torch.eye(width, dtype=torch.bool, device=mask.device).unsqueeze(0)
    counts = torch.as_tensor([int(n) for n in n_node], dtype=torch.long, device=mask.device)
    return MolecularBatch(
        atom_types=torch.zeros((batch, width), dtype=torch.long, device=mask.device),
        bond_types=torch.zeros((batch, width, width), dtype=torch.long, device=mask.device),
        node_mask=mask, pair_mask=pair_mask, n_node=counts,
        qed=torch.full((batch,), float("nan"), device=mask.device),
        row_id=torch.full((batch,), -1, dtype=torch.long, device=mask.device),
        condition=None,
    )


def _invalid_record(reason: str) -> dict[str, Any]:
    """An entry for a sample that cannot be decoded at all (e.g. a non-finite embedding)."""
    return {"sanitizable": False, "connected": False, "valid": False, "smiles": None,
            "fragments": None, "fallback": False, "fallback_reason": reason, "bonds": 0}


def _molecule_record(atoms: np.ndarray, bonds: np.ndarray) -> dict[str, Any]:
    """Sanitization, connectivity and canonical SMILES for one decoded graph."""
    record: dict[str, Any] = {"sanitizable": False, "connected": False, "valid": False,
                              "smiles": None, "fragments": None}
    try:
        with rdBase.BlockLogs():
            molecule = graph_to_mol_charge_aware(atoms, bonds)
    except Exception:  # noqa: BLE001 - any RDKit/valence failure means invalid
        return record
    record["sanitizable"] = True
    try:
        fragments = len(Chem.GetMolFrags(molecule))
    except Exception:  # noqa: BLE001
        return record
    record["fragments"] = fragments
    record["connected"] = fragments == 1
    # STRICT validity: sanitizable AND a single connected molecule. A sanitizable
    # set of disconnected fragments is not a molecule we generated on purpose.
    record["valid"] = bool(record["sanitizable"] and record["connected"])
    if record["valid"]:
        try:
            record["smiles"] = canonical_connectivity_smiles(molecule)
            record["qed"] = float(QED.qed(molecule))
            record["rings"] = int(molecule.GetRingInfo().NumRings())
        except Exception:  # noqa: BLE001
            record["valid"] = False
            record["smiles"] = None
    return record


@torch.no_grad()
def decode_embeddings(autoencoder, embeddings: torch.Tensor, n_node: Sequence[int], *,
                      decoders: Iterable[str] = DECODERS) -> list[dict[str, Any]]:
    """Turn generated node embeddings into molecules with each requested decoder."""
    decoders = tuple(decoders)
    unknown = set(decoders) - set(DECODERS)
    if unknown:
        raise ValueError(f"unknown decoders {sorted(unknown)}; known: {list(DECODERS)}")
    has_degree = bool(autoencoder.config.degree_head)
    if "degree" in decoders and not has_degree:
        raise ValueError("the autoencoder has no degree head, so the degree decoder is unavailable")
    device = embeddings.device
    batch = generation_batch(n_node, device=device)
    if embeddings.shape[:2] != batch.node_mask.shape:
        raise ValueError(f"embeddings {tuple(embeddings.shape)} do not match the node counts")
    # A flow pushed into a bad regime can emit non-finite embeddings. Those samples are
    # counted as invalid rather than fed to RDKit as nonsense, and the decoder is given a
    # sanitized tensor so one bad sample cannot poison the rest of the batch.
    finite_nodes = torch.isfinite(embeddings).all(dim=-1)
    safe_embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    was_training = autoencoder.training
    autoencoder.eval()
    outputs = autoencoder.decode(safe_embeddings, batch, teacher_forcing=False)
    if was_training:
        autoencoder.train()

    atom_predictions = outputs["atom_logits"].argmax(-1).cpu().numpy()
    bond_probabilities = torch.softmax(outputs["bond_logits"], dim=-1).cpu().numpy()
    bond_predictions = outputs["bond_logits"].argmax(-1).cpu().numpy()
    degree_predictions = (outputs["degree_logits"].argmax(-1).cpu().numpy()
                          if has_degree and "degree_logits" in outputs else None)

    records = []
    for index, count in enumerate(int(n) for n in n_node):
        if not bool(finite_nodes[index, :count].all()):
            records.append({"n_node": count, "atom_types": [], "nonfinite_embedding": True,
                            **{name: _invalid_record("nonfinite_embedding") for name in decoders}})
            continue
        atoms = atom_predictions[index, :count]
        probabilities = bond_probabilities[index, :count, :count]
        record: dict[str, Any] = {"n_node": count, "atom_types": atoms.tolist(),
                                  "nonfinite_embedding": False}
        for name in decoders:
            if name == "raw":
                bonds = bond_predictions[index, :count, :count].copy()
                np.fill_diagonal(bonds, 0)
                bonds = np.minimum(bonds, bonds.T)      # a bond matrix must be symmetric
                info: dict[str, Any] = {}
            elif name == "constrained":
                bonds, info = valence_aware_bond_decode_typed(atoms, probabilities)
            else:
                degrees = degree_predictions[index, :count]
                bonds, info = degree_constrained_bond_decode(atoms, probabilities, degrees)
                record["predicted_degrees"] = degrees.tolist()
            entry = _molecule_record(atoms, bonds)
            entry["fallback"] = bool(info.get("fallback", False))
            entry["fallback_reason"] = info.get("fallback_reason")
            entry["bonds"] = int((np.asarray(bonds) > 0).sum() // 2)
            record[name] = entry
        records.append(record)
    return records


@torch.no_grad()
def generate_molecules(flow: NodeFlow, autoencoder, *, count: int, node_counts: NodeCountSampler,
                       temperature: float = 1.0, seed: int = 0, batch_size: int = 128,
                       decoders: Iterable[str] = DECODERS,
                       condition_values: torch.Generator | None = None,
                       target_n: int | None = None,
                       device: torch.device | str = "cpu") -> dict[str, Any]:
    """Sample `count` molecules. Returns `{"records": [...], "settings": {...}}`.

    `target_n` fixes every sample's size (used for size-controlled checks);
    otherwise sizes are drawn from the training histogram. `condition_values`
    is accepted for the later conditional models and must be None here.
    """
    if condition_values is not None:
        raise NotImplementedError("conditional sampling is not part of the unconditional stage")
    if flow.config.conditional:
        raise ValueError("this flow is conditional; use the conditional generation path")
    if count < 1:
        raise ValueError("count must be >= 1")
    flow.eval()
    rng = np.random.default_rng(int(seed))
    generator = torch.Generator(device=device).manual_seed(int(seed))
    sizes = (np.full(count, int(target_n), dtype=np.int64) if target_n
             else node_counts.sample(count, rng))
    records: list[dict[str, Any]] = []
    round_trip_error: float | None = None
    for start in range(0, count, batch_size):
        chunk = [int(n) for n in sizes[start:start + batch_size]]
        mask = node_mask_from_counts(chunk, device=device)
        embeddings, latents = flow.sample(mask, temperature=temperature, generator=generator,
                                          return_latent=True)
        if round_trip_error is None:
            # Self-check on the first batch only: re-encoding the sampled embeddings must
            # return the latents they came from. A large error means the inverse used for
            # sampling is not the inverse of the density being reported.
            recovered, _ = flow(embeddings, mask)
            scale = max(1e-9, float(latents.abs().max()))
            round_trip_error = float((recovered - latents).abs().max()) / scale
        records.extend(decode_embeddings(autoencoder, embeddings, chunk, decoders=decoders))
    for position, record in enumerate(records):
        record["index"] = position
    nonfinite = sum(1 for r in records if r.get("nonfinite_embedding"))
    return {
        "records": records,
        "settings": {"count": count, "temperature": float(temperature), "seed": int(seed),
                     "nonfinite_embeddings": nonfinite,
                     "nonfinite_embedding_rate": nonfinite / len(records),
                     "latent_round_trip_relative_error": round_trip_error,
                     "target_n": target_n, "decoders": list(decoders),
                     "node_count_source": "train_histogram" if not target_n else "fixed",
                     "mean_sampled_n": float(np.mean(sizes)),
                     "node_count_histogram_mean": node_counts.mean},
    }


def node_count_agreement(records: Sequence[dict[str, Any]], decoder: str = "degree") -> dict[str, Any]:
    """Does a decoded molecule really have the atom count we asked the flow for?

    The answer is yes by construction (the decoder emits one atom per node
    vector), which is exactly why node-count "accuracy" must never be reported
    as evidence of learned conditioning. Measured anyway, as a guard against
    a decoding bug.
    """
    total = matched = 0
    for record in records:
        entry = record.get(decoder, {})
        if not entry.get("valid"):
            continue
        molecule = Chem.MolFromSmiles(entry["smiles"])
        if molecule is None:
            continue
        total += 1
        matched += int(molecule.GetNumHeavyAtoms() == record["n_node"])
    return {"molecules": total, "agreement": (matched / total) if total else float("nan"),
            "note": "structural, not learned: the decoder emits one atom per sampled node vector"}


def size_distribution(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counter = Counter(int(r["n_node"]) for r in records)
    return {"histogram": {int(k): int(v) for k, v in sorted(counter.items())},
            "mean": float(np.mean([r["n_node"] for r in records])) if records else float("nan")}


__all__ = [
    "DECODERS",
    "NodeCountSampler",
    "decode_embeddings",
    "generate_molecules",
    "generation_batch",
    "node_count_agreement",
    "size_distribution",
]
