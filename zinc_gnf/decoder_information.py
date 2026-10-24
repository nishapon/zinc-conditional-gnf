"""Where does the requested QED disappear? A stage-by-stage information audit.

THE QUESTION STAGE 18 SHARPENED
-------------------------------
The conditional signal is NOT small: generated embeddings move 0.665x (median; 1.217x at
N=23, T=0.2) of the displacement the real populations show, with a direction cosine of
+0.926, and every sample was finite. Yet realised QED sits at ~0.56 for every request.

But - and this is the point of this module - a matching MEAN displacement does not imply
the information survives. Three different things could be true, and they have different
fixes:

  H1  the information is in the generated embeddings, and DECODING loses it
  H2  the embeddings move a lot, but not along a direction a QED-aware readout can use
      (the mean-difference direction between two QED populations is not the same thing as
      a QED-PREDICTIVE direction: it also contains every covariate that happens to
      correlate with QED)
  H3  the information reaches the decoder, and graph constraints plus validity selection
      erase it afterwards

WHAT THIS MODULE MEASURES
-------------------------
`EmbeddingQedProbe` is a small read-out trained on REAL TRAINING embeddings and validated
on REAL VALIDATION embeddings. Applied to generated embeddings it answers "is a QED
difference visible before any graph is built?" - which separates H1 from H2. Its own
held-out error is always reported, and `ood_statistics` reports how far generated
embeddings sit from the real ones, because a prediction on out-of-distribution input is
an extrapolation and not ground truth. The probe is diagnostic only: nothing here guides
generation.

`paired_intervention` changes ONLY the QED request with the noise held fixed and follows
the same molecules through every stage: embeddings -> bond logits -> predicted bond types
-> assembled graph -> valid molecule -> QED. Every stage reports its own displacement, so
the stage at which the difference collapses is visible rather than inferred.

`decoder_qed_transfer` is the control that makes all of this interpretable: it decodes
REAL embeddings of molecules at QED 0.4 and at QED 0.8 (same N) and measures whether the
decoder preserves THAT difference. If it does not, the decoder destroys QED differences
even on perfect inputs and no flow-side work can help; if it does, the loss is specific to
generated embeddings.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from zinc_gnf.generation import decode_embeddings, generation_batch
from zinc_gnf.molecular_profile import profile_molecules
from zinc_gnf.structure_objective import structure_statistics

PROBE_SEED = 20261022
DECODERS = ("raw", "constrained", "degree")


class EmbeddingQedProbe(nn.Module):
    """Mean/spread-pooled node embeddings -> QED. Deliberately small: it is a read-out."""

    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(x.dtype)
        counts = weights.sum(dim=1).clamp(min=1.0)
        mean = (x * weights).sum(dim=1) / counts
        spread = (((x - mean.unsqueeze(1)) ** 2) * weights).sum(dim=1) / counts
        return self.net(torch.cat([mean, spread.sqrt()], dim=-1)).squeeze(-1)


def _pooled(autoencoder, dataset, indices: Sequence[int], *, device, batch_size: int = 128):
    from zinc_gnf.dataset import iter_eval_batches

    collected = []
    with torch.no_grad():
        for batch in iter_eval_batches(dataset, batch_size, [int(i) for i in indices]):
            batch = batch.to(device)
            collected.append((autoencoder.encode(batch).detach(), batch.node_mask.detach(),
                              batch.qed.detach()))
    return collected


def fit_embedding_probe(autoencoder, train_set, val_set, *, train_indices: Sequence[int],
                        val_indices: Sequence[int], device: torch.device | str = "cpu",
                        batch_size: int = 128, epochs: int = 10,
                        learning_rate: float = 1e-3,
                        seed: int = PROBE_SEED) -> tuple[EmbeddingQedProbe, dict[str, Any]]:
    """Fit on REAL TRAIN embeddings, validate on REAL VALIDATION embeddings.

    The validation numbers bound what the probe can tell us: a probe with a held-out MAE
    of 0.08 cannot certify a 0.02 QED shift.
    """
    device = torch.device(device)
    train_batches = _pooled(autoencoder, train_set, train_indices, device=device,
                            batch_size=batch_size)
    val_batches = _pooled(autoencoder, val_set, val_indices, device=device,
                          batch_size=batch_size)
    if not train_batches or not val_batches:
        raise ValueError("the probe needs both training and validation molecules")
    dim = int(train_batches[0][0].shape[-1])
    torch.manual_seed(int(seed))
    probe = EmbeddingQedProbe(dim).to(device)
    optimiser = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    history = []
    for epoch in range(int(epochs)):
        probe.train()
        losses = []
        for x, mask, qed in train_batches:
            prediction = probe(x, mask)
            loss = nn.functional.mse_loss(prediction, qed.to(prediction.dtype))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch + 1, "train_mse": float(np.mean(losses))})
    probe.eval()

    def score(batches) -> dict[str, Any]:
        predictions, truths = [], []
        with torch.no_grad():
            for x, mask, qed in batches:
                predictions.append(probe(x, mask).cpu().numpy())
                truths.append(qed.cpu().numpy())
        p = np.concatenate(predictions).astype(np.float64)
        t = np.concatenate(truths).astype(np.float64)
        correlation = (float(np.corrcoef(p, t)[0, 1])
                       if p.size > 1 and p.std() > 0 and t.std() > 0 else float("nan"))
        return {"molecules": int(p.size), "mae": float(np.abs(p - t).mean()),
                "rmse": float(np.sqrt(((p - t) ** 2).mean())), "correlation": correlation,
                "predicted_std": float(p.std()), "true_std": float(t.std())}

    report = {"train": score(train_batches), "validation": score(val_batches),
              "epochs": int(epochs), "history": history, "seed": int(seed),
              "note": ("diagnostic read-out trained on REAL embeddings. Its validation MAE "
                       "bounds what it can certify, and a prediction on generated "
                       "embeddings is an EXTRAPOLATION, not ground truth.")}
    return probe, report


@torch.no_grad()
def ood_statistics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    """How far are generated node embeddings from the real ones?

    Mahalanobis distance in the real embeddings' own (diagonally regularised) covariance,
    so "the probe is extrapolating" is a number rather than an opinion.
    """
    reference = reference.to(torch.float64)
    candidate = candidate.to(torch.float64)
    mean = reference.mean(0)
    centred = reference - mean
    covariance = centred.T @ centred / max(reference.shape[0] - 1, 1)
    covariance = covariance + 1e-6 * torch.eye(covariance.shape[0], dtype=covariance.dtype)
    inverse = torch.linalg.inv(covariance)

    def distances(values: torch.Tensor) -> torch.Tensor:
        delta = values - mean
        return ((delta @ inverse) * delta).sum(-1).clamp(min=0.0).sqrt()

    real = distances(reference)
    generated = distances(candidate)
    quantile = float((real.unsqueeze(0) < generated.unsqueeze(1)).to(torch.float64).mean())
    return {
        "real_mahalanobis_mean": float(real.mean()),
        "real_mahalanobis_q95": float(torch.quantile(real, 0.95)),
        "generated_mahalanobis_mean": float(generated.mean()),
        "generated_mahalanobis_q95": float(torch.quantile(generated, 0.95)),
        "generated_over_real_mean": float(generated.mean() / real.mean().clamp(min=1e-12)),
        "fraction_of_real_below_a_generated_point": quantile,
        "note": ("a generated/real mean ratio well above 1 means the decoder and the probe "
                 "are being asked about inputs they were never fitted on"),
    }


@torch.no_grad()
def decode_stage_statistics(autoencoder, embeddings: torch.Tensor, counts: Sequence[int],
                            mask: torch.Tensor, *, decoders: Sequence[str] = DECODERS
                            ) -> dict[str, Any]:
    """Bond logits, predicted bond types and assembled graphs for one population."""
    batch = generation_batch(list(counts), device=embeddings.device)
    outputs = autoencoder.decode(embeddings, batch, teacher_forcing=False)
    statistics = structure_statistics(outputs["bond_logits"], outputs["atom_logits"], mask)
    records = decode_embeddings(autoencoder, embeddings, list(counts), decoders=decoders)
    per_decoder = {}
    for name in decoders:
        entries = [r[name] for r in records if name in r]
        valid = [e for e in entries if e.get("valid") and e.get("smiles")]
        profile = profile_molecules([e["smiles"] for e in valid], name=name, scaffolds=False)
        fallback = [e for e in entries if "fallback" in e]
        per_decoder[name] = {
            "validity": len(valid) / max(len(entries), 1),
            "fallback_rate": (sum(1 for e in fallback if e.get("fallback")) / len(fallback))
            if fallback else float("nan"),
            "qed_mean": profile["qed"]["mean"],
            "rings_mean": profile["rings"]["mean"],
            "aromatic_rings_mean": profile["aromatic_rings"]["mean"],
            "fragments_mean": float(np.mean([e.get("fragments") or np.nan for e in entries]))
            if entries else float("nan"),
        }
    return {
        "bond_logits": outputs["bond_logits"].detach(),
        "bond_argmax": outputs["bond_logits"].detach().argmax(-1),
        "structure": {k: float(v.mean()) for k, v in statistics.items()},
        "decoders": per_decoder,
        "records": records,
    }


def _paired_qed_change(low_records, high_records, decoder: str) -> dict[str, Any]:
    """QED change for the molecules that are valid under BOTH conditions.

    An unpaired mean difference mixes the conditional effect with which samples happened
    to survive validity; the paired figure isolates the effect, and the survival rates
    are reported beside it so selection is visible.
    """
    pairs = []
    both = low = high = 0
    for a, b in zip(low_records, high_records):
        left, right = a.get(decoder) or {}, b.get(decoder) or {}
        low += int(bool(left.get("valid")))
        high += int(bool(right.get("valid")))
        if left.get("valid") and right.get("valid") \
                and left.get("qed") is not None and right.get("qed") is not None:
            both += 1
            pairs.append(float(right["qed"]) - float(left["qed"]))
    array = np.asarray(pairs, dtype=np.float64)
    total = max(len(low_records), 1)
    return {
        "paired_molecules": both,
        "valid_low": low / total, "valid_high": high / total,
        "paired_fraction": both / total,
        "qed_change_mean": float(array.mean()) if array.size else float("nan"),
        "qed_change_std": float(array.std()) if array.size else float("nan"),
        "qed_change_sem": float(array.std() / np.sqrt(array.size)) if array.size > 1
        else float("nan"),
        "fraction_moving_up": float((array > 0).mean()) if array.size else float("nan"),
    }


@torch.no_grad()
def paired_intervention(flow, autoencoder, *, scaler, n: int, qed_low: float, qed_high: float,
                        count: int, temperature: float, device: torch.device | str = "cpu",
                        probe: EmbeddingQedProbe | None = None, seed: int = 17,
                        decoders: Sequence[str] = DECODERS) -> dict[str, Any]:
    """Change ONLY the QED request, keep the noise, and follow every stage."""
    from zinc_gnf.conditioning import condition_embedding_for, standardise_condition
    from zinc_gnf.flow import node_mask_from_counts

    features = flow.config.condition_features
    device = torch.device(device)
    counts = [int(n)] * int(count)
    mask = node_mask_from_counts(counts, device=device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    noise = torch.randn((count, mask.shape[1], flow.config.node_embedding_dim),
                        device=device, generator=generator)

    stages: dict[str, Any] = {"n": int(n), "qed_low": float(qed_low),
                              "qed_high": float(qed_high), "count": int(count),
                              "temperature": float(temperature), "seed": int(seed)}
    produced: dict[str, Any] = {}
    for label, qed in (("low", qed_low), ("high", qed_high)):
        values = torch.from_numpy(standardise_condition(
            features, scaler=scaler, n=counts, qed=[qed] * count)).to(device)
        condition = condition_embedding_for(flow, values, None, count, device)
        mu, sigma = flow.prior.parameters_for(count, condition, device)
        z = ((mu.unsqueeze(1) + float(temperature) * sigma.unsqueeze(1) * noise)
             * mask.unsqueeze(-1).to(noise.dtype))
        x = flow.inverse(z, mask, condition)
        finite = (torch.isfinite(x).all(dim=-1) | ~mask).all(dim=-1)
        produced[label] = {"z": z, "x": x, "finite": finite}
    keep = produced["low"]["finite"] & produced["high"]["finite"]
    stages["nonfinite_rate"] = float((~keep).to(torch.float64).mean())
    if int(keep.sum()) == 0:
        stages["note"] = "every sample was non-finite; nothing can be measured"
        return stages

    index = torch.nonzero(keep).flatten()
    usable = int(index.numel())
    stages["usable_molecules"] = usable
    kept_counts = [int(n)] * usable
    kept_mask = mask[index]
    low_x, high_x = produced["low"]["x"][index], produced["high"]["x"][index]
    real = kept_mask.unsqueeze(-1).to(torch.float64)
    nodes = real.sum().clamp(min=1.0)

    # stage 1: latent and embedding displacement
    latent_delta = ((produced["high"]["z"][index] - produced["low"]["z"][index]).to(torch.float64)
                    * real)
    embedding_delta = (high_x - low_x).to(torch.float64) * real
    stages["latent"] = {"per_node_norm": float(latent_delta.norm(dim=-1).sum() / nodes)}
    stages["embedding"] = {
        "per_node_norm": float(embedding_delta.norm(dim=-1).sum() / nodes),
        "relative_to_magnitude": float(embedding_delta.norm(dim=-1).sum()
                                       / (low_x.to(torch.float64) * real).norm(dim=-1).sum()
                                       .clamp(min=1e-12)),
    }

    # stage 2: the property read-out, before any graph exists
    if probe is not None:
        low_prediction = probe(low_x, kept_mask)
        high_prediction = probe(high_x, kept_mask)
        difference = (high_prediction - low_prediction).to(torch.float64).cpu().numpy()
        stages["probe"] = {
            "predicted_low_mean": float(low_prediction.mean()),
            "predicted_high_mean": float(high_prediction.mean()),
            "predicted_change_mean": float(difference.mean()),
            "predicted_change_sem": float(difference.std() / np.sqrt(max(difference.size, 1))),
            "fraction_moving_up": float((difference > 0).mean()),
            "requested_change": float(qed_high) - float(qed_low),
        }

    # stage 3: bond logits and predicted bond types
    low_stage = decode_stage_statistics(autoencoder, low_x, kept_counts, kept_mask,
                                        decoders=decoders)
    high_stage = decode_stage_statistics(autoencoder, high_x, kept_counts, kept_mask,
                                         decoders=decoders)
    pair_mask = (kept_mask.unsqueeze(1) & kept_mask.unsqueeze(2))
    pair_mask = pair_mask & ~torch.eye(kept_mask.shape[1], dtype=torch.bool,
                                       device=kept_mask.device).unsqueeze(0)
    logit_delta = (high_stage["bond_logits"] - low_stage["bond_logits"]).to(torch.float64)
    pairs = pair_mask.to(torch.float64).sum().clamp(min=1.0)
    changed = ((high_stage["bond_argmax"] != low_stage["bond_argmax"]) & pair_mask)
    stages["bond_logits"] = {
        "mean_abs_change": float((logit_delta.abs() * pair_mask.unsqueeze(-1)).sum()
                                 / (pairs * logit_delta.shape[-1])),
        "per_pair_l2": float((logit_delta * pair_mask.unsqueeze(-1)).norm(dim=-1).sum() / pairs),
        "argmax_change_rate": float(changed.to(torch.float64).sum() / pairs),
    }
    stages["structure"] = {
        key: {"low": low_stage["structure"][key], "high": high_stage["structure"][key],
              "change": high_stage["structure"][key] - low_stage["structure"][key]}
        for key in low_stage["structure"]
    }

    # stages 4 and 5: assembled graphs and the QED of valid molecules, per decoder
    stages["decoders"] = {}
    for name in decoders:
        stages["decoders"][name] = {
            "low": low_stage["decoders"][name],
            "high": high_stage["decoders"][name],
            "aromatic_rings_change": (high_stage["decoders"][name]["aromatic_rings_mean"]
                                      - low_stage["decoders"][name]["aromatic_rings_mean"]),
            "rings_change": (high_stage["decoders"][name]["rings_mean"]
                             - low_stage["decoders"][name]["rings_mean"]),
            "unpaired_qed_change": (high_stage["decoders"][name]["qed_mean"]
                                    - low_stage["decoders"][name]["qed_mean"]),
            "paired": _paired_qed_change(low_stage["records"], high_stage["records"], name),
        }
    return stages


@torch.no_grad()
def decoder_qed_transfer(autoencoder, dataset, *, n: int, qed_low: float, qed_high: float,
                         window: float = 0.05, max_molecules: int = 256,
                         device: torch.device | str = "cpu",
                         decoders: Sequence[str] = DECODERS) -> dict[str, Any]:
    """THE control: does the decoder preserve a QED difference it is GIVEN?

    Real training molecules at QED_low and QED_high (same N) are encoded and decoded
    through the identical path generation uses. If the decoded QED difference is far
    smaller than the true one, the decoder destroys QED differences on perfect inputs and
    no flow-side work can recover them.
    """
    from zinc_gnf.dataset import iter_eval_batches

    device = torch.device(device)
    counts = np.asarray(dataset.n_node)
    qed = np.asarray(dataset.qed, dtype=np.float64)
    result: dict[str, Any] = {"n": int(n), "qed_low": float(qed_low),
                              "qed_high": float(qed_high), "window": float(window),
                              "groups": {}}
    for label, target in (("low", qed_low), ("high", qed_high)):
        selected = np.flatnonzero((counts == int(n)) & (np.abs(qed - target) <= window))
        group: dict[str, Any] = {"molecules_available": int(selected.size),
                                 "requested_qed": float(target)}
        if selected.size == 0:
            group["note"] = "no training molecule at this (N, QED)"
            result["groups"][label] = group
            continue
        chosen = [int(i) for i in selected[:max_molecules]]
        group["molecules_used"] = len(chosen)
        group["true_qed_mean"] = float(qed[chosen].mean())
        embeddings, masks, sizes = [], [], []
        for batch in iter_eval_batches(dataset, 64, chosen):
            batch = batch.to(device)
            embeddings.append(autoencoder.encode(batch))
            masks.append(batch.node_mask)
            sizes.extend(int(c) for c in batch.n_node.tolist())
        width = max(e.shape[1] for e in embeddings)
        padded = torch.cat([
            torch.nn.functional.pad(e, (0, 0, 0, width - e.shape[1])) for e in embeddings])
        pad_mask = torch.cat([
            torch.nn.functional.pad(m, (0, width - m.shape[1])) for m in masks])
        stage = decode_stage_statistics(autoencoder, padded, sizes, pad_mask,
                                        decoders=decoders)
        group["decoders"] = stage["decoders"]
        group["structure"] = stage["structure"]
        result["groups"][label] = group
    low, high = result["groups"].get("low", {}), result["groups"].get("high", {})
    if low.get("decoders") and high.get("decoders"):
        result["true_qed_difference"] = high["true_qed_mean"] - low["true_qed_mean"]
        result["decoded_qed_difference"] = {
            name: high["decoders"][name]["qed_mean"] - low["decoders"][name]["qed_mean"]
            for name in decoders}
        result["preservation"] = {
            name: (result["decoded_qed_difference"][name]
                   / result["true_qed_difference"]) if result["true_qed_difference"] else
            float("nan") for name in decoders}
        result["verdict"] = (
            "the decoder preserves most of a QED difference it is given, so a flat "
            "generated QED is NOT simply the decoder destroying the signal"
            if max((v for v in result["preservation"].values()
                    if np.isfinite(v)), default=0.0) > 0.5 else
            "the decoder loses most of a QED difference even on REAL embeddings: QED "
            "control cannot be achieved through this decoder, whatever the flow does")
    else:
        result["verdict"] = "insufficient real molecules at this (N, QED) to run the control"
    return result


def support_counts(dataset, *, n_values: Sequence[int], qed_values: Sequence[float],
                   window: float = 0.05) -> list[dict[str, Any]]:
    """Explicit, unambiguous support: every count and its denominator stated."""
    counts = np.asarray(dataset.n_node)
    qed = np.asarray(dataset.qed, dtype=np.float64)
    rows = []
    for n in n_values:
        at_n = int((counts == int(n)).sum())
        row: dict[str, Any] = {"n": int(n), "molecules_at_this_n": at_n,
                               "window": float(window), "by_qed": {}}
        for target in qed_values:
            inside = int(((counts == int(n)) & (np.abs(qed - target) <= window)).sum())
            row["by_qed"][f"{target:g}"] = {
                "molecules_within_window": inside,
                "fraction_of_this_n": (inside / at_n) if at_n else float("nan"),
            }
        pair = [row["by_qed"][f"{t:g}"]["molecules_within_window"] for t in qed_values]
        row["minimum_over_qed_values"] = int(min(pair)) if pair else 0
        rows.append(row)
    return rows


__all__ = [
    "DECODERS",
    "PROBE_SEED",
    "EmbeddingQedProbe",
    "decode_stage_statistics",
    "decoder_qed_transfer",
    "fit_embedding_probe",
    "ood_statistics",
    "paired_intervention",
    "support_counts",
]
