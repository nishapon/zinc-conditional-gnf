#!/usr/bin/env python3
"""Stage 1 pilot: decoder-aware guidance. INFERENCE-TIME GUIDANCE, NOT intrinsic conditioning.

READ-ONLY with respect to every model: the flow and the autoencoder are opened in eval()
with gradients disabled, `flow.inverse` runs under `no_grad`, nothing is trained, no
checkpoint is written and no existing output directory is touched. The test split is
never loaded.

EVERY NUMBER THIS PRODUCES IS POST-HOC GUIDANCE. It does not demonstrate improved
intrinsic conditional generation, and the report says so in its header. Stage 2
(distilling guidance into the flow) is NOT implemented here.

PHASE A -- surrogate validation, a HARD GATE before any guidance runs
--------------------------------------------------------------------
Fit on TRAIN real embeddings only, against the dataset's RDKit QED labels, then evaluate:

  * held-out REAL validation embeddings, overall and PER N (MAE, bias, correlation);
  * unmodified FLOW-GENERATED embeddings, predicted QED against their ACTUAL decoded
    RDKit QED -- the only test that says whether the surrogate transfers to the
    distribution guidance will actually be applied to;
  * both feature sets, `moments` (first moments only) and `structural` (adds the derived
    edge/ring/consistency terms), so "do pooled logit features retain enough information"
    is answered by measurement;
  * a second readout for AROMATIC RING COUNT, which tests the aromaticity claim directly
    rather than by assertion.

If the surrogate fails its precommitted thresholds on GENERATED embeddings, the QED arms
are SKIPPED and the report says so. Optimising against an inaccurate surrogate is the
failure mode this gate exists to prevent.

PHASE B -- bounded pilot, five arms, identical starting points
--------------------------------------------------------------
Per cell (N, QED target) one set of latents is drawn with COMMON noise across targets
(the protocol the existing paired evaluation uses), inverted ONCE under `no_grad`, and
all five arms start from that identical `x_0`:

  native            no guidance (the incumbent)
  qed_only          lambda_manifold = 0
  manifold_only     lambda_qed = 0          (no learned critic at all)
  combined          both
  random            matched-norm isotropic step (the control)

Reported per cell: ACTUAL decoded RDKit QED, QED MAE against the request, raw validity
and corrected validity SEPARATELY, fallback rate, aromatic rings, uniqueness, internal
diversity, heavy-atom count, surrogate-predicted QED beside the actual value
(exploitation detector), displacement, trust-region saturation and non-finite step count.
Paired QED responsiveness across targets is computed on common noise.

Runtime is MEASURED and reported per arm; nothing is assumed about it.

Exit codes: 0 ok; 2 missing input; 3 provenance mismatch; 4 bad arguments; 1 unexpected.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from zinc_gnf.conditioning import (  # noqa: E402
    condition_embedding_for,
    load_condition_scaler,
    standardise_condition,
)
from zinc_gnf.dataset import (  # noqa: E402
    MolecularGraphDataset,
    iter_eval_batches,
    load_training_splits,
)
from zinc_gnf.decoder_guidance import (  # noqa: E402
    FEATURE_SETS,
    RidgeReadout,
    decoder_statistics,
    embedding_features,
    feature_names,
    guide,
    readout_quality,
)
from zinc_gnf.trainer_autoencoder import ResumeMismatchError, load_autoencoder  # noqa: E402
from zinc_gnf.trainer_flow import file_sha256, load_flow  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402

ARMS = ("native", "qed_only", "manifold_only", "combined", "random")
# Precommitted surrogate-acceptance thresholds, fixed BEFORE any data is seen.
SURROGATE_MAX_MAE_REAL = 0.08          # held-out real embeddings, raw QED units
SURROGATE_MIN_CORR_REAL = 0.45
SURROGATE_MAX_MAE_GENERATED = 0.12     # the distribution guidance is applied to
SURROGATE_MIN_CORR_GENERATED = 0.30
BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261113


def bootstrap_ci(values, *, seed: int = BOOTSTRAP_SEED):
    array = np.asarray([v for v in np.ravel(values)
                        if v is not None and np.isfinite(v)], dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": None, "ci95": [None, None], "excludes_zero": False}
    if array.size == 1:
        return {"n": 1, "mean": float(array[0]), "ci95": [None, None],
                "excludes_zero": False}
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, array.size, size=(BOOTSTRAP, array.size))].mean(axis=1)
    low, high = float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
    return {"n": int(array.size), "mean": float(array.mean()), "ci95": [low, high],
            "excludes_zero": bool(low > 0.0 or high < 0.0)}


def pair_masks_for(node_mask: torch.Tensor) -> torch.Tensor:
    width = node_mask.shape[1]
    pair = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
    eye = torch.eye(width, dtype=torch.bool, device=node_mask.device).unsqueeze(0)
    return pair & ~eye


def reference_condition(flow, scaler, n, qed, count, device):
    raw = standardise_condition(flow.config.condition_features, scaler=scaler,
                                n=[n] * count, qed=[qed] * count)
    values = torch.from_numpy(raw).to(device)
    return condition_embedding_for(flow, values, torch.ones_like(values), count, device)


# ------------------------------------------------------------------- phase A ---
@torch.no_grad()
def collect_real(autoencoder, dataset, indices, *, device, batch_size, feature_set):
    """Features, RDKit-QED labels, aromatic-ring labels and N for real molecules."""
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdMolDescriptors

    features, qeds, aromatics, counts = [], [], [], []
    for batch in iter_eval_batches(dataset, batch_size, list(indices)):
        batch = batch.to(device)
        x = autoencoder.encode(batch)
        f = embedding_features(autoencoder, x, batch.node_mask, batch.pair_mask,
                               feature_set=feature_set)
        features.append(f.detach().cpu().numpy().astype(np.float64))
        qeds.extend(float(v) for v in batch.qed.detach().cpu())
        counts.extend(int(v) for v in batch.n_node.detach().cpu())
    # aromatic-ring labels come from the stored SMILES, not from decoding
    for index in list(indices):
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(dataset.smiles[int(index)])
        aromatics.append(float(rdMolDescriptors.CalcNumAromaticRings(mol))
                         if mol is not None else float("nan"))
    return (np.concatenate(features), np.asarray(qeds), np.asarray(aromatics[:len(qeds)]),
            np.asarray(counts))


@torch.no_grad()
def collect_generated(flow, autoencoder, *, scaler, targets_n, qed_values, samples,
                      temperature, device, feature_set, eps_seed, decoders):
    """Unmodified generated embeddings: surrogate features and ACTUAL decoded RDKit QED."""
    from zinc_gnf.generation import decode_embeddings
    from rdkit import Chem, rdBase
    from rdkit.Chem import QED

    features, actual, counts = [], [], []
    for n in targets_n:
        for q in qed_values:
            mask = torch.ones((samples, n), dtype=torch.bool, device=device)
            pair = pair_masks_for(mask)
            generator = torch.Generator(device=device).manual_seed(int(eps_seed) + n)
            eps = torch.randn((samples, n, flow.config.node_embedding_dim),
                              device=device, generator=generator)
            cond = reference_condition(flow, scaler, n, q, samples, device)
            with torch.no_grad():          # f^-1 is NEVER in a gradient path (v1's failure)
                mu, sigma = flow.prior.parameters_for(samples, cond, device)
                z = (mu.unsqueeze(1) + float(temperature) * sigma.unsqueeze(1) * eps) \
                    * mask.unsqueeze(-1)
                x = flow.inverse(z, mask, cond).detach()
            f = embedding_features(autoencoder, x, mask, pair, feature_set=feature_set)
            records = decode_embeddings(autoencoder, x, [n] * samples,
                                        decoders=(decoders[0],))
            for row, record in zip(f.detach().cpu().numpy().astype(np.float64), records):
                entry = record.get(decoders[0]) or {}
                if not (entry.get("valid") and entry.get("smiles")):
                    continue
                with rdBase.BlockLogs():
                    mol = Chem.MolFromSmiles(entry["smiles"])
                if mol is None:
                    continue
                features.append(row)
                actual.append(float(QED.qed(mol)))
                counts.append(n)
    if not features:
        return None, None, None
    return np.stack(features), np.asarray(actual), np.asarray(counts)


def validate_surrogates(flow, autoencoder, train_set, val_set, *, scaler, device,
                        batch_size, train_molecules, val_molecules, targets_n,
                        qed_values, samples, temperature, eps_seed, decoders, seed=7):
    """Phase A. Fits and validates both feature sets; returns the chosen readout or None."""
    rng = np.random.default_rng(seed)
    train_idx = rng.permutation(len(train_set))[:train_molecules].tolist()
    val_idx = rng.permutation(len(val_set))[:val_molecules].tolist()
    report: dict[str, Any] = {
        "thresholds": {
            "real_max_mae": SURROGATE_MAX_MAE_REAL,
            "real_min_correlation": SURROGATE_MIN_CORR_REAL,
            "generated_max_mae": SURROGATE_MAX_MAE_GENERATED,
            "generated_min_correlation": SURROGATE_MIN_CORR_GENERATED,
            "note": ("precommitted before any data was seen. The GENERATED thresholds "
                     "are the binding ones: that is the distribution guidance acts on."),
        },
        "feature_sets": {},
    }
    chosen, chosen_name = None, None
    for feature_set in FEATURE_SETS:
        block: dict[str, Any] = {}
        xt, yt, at, nt = collect_real(autoencoder, train_set, train_idx, device=device,
                                      batch_size=batch_size, feature_set=feature_set)
        xv, yv, av, nv = collect_real(autoencoder, val_set, val_idx, device=device,
                                      batch_size=batch_size, feature_set=feature_set)
        block["feature_names"] = feature_names(
            autoencoder.config.num_atom_types, autoencoder.config.num_bond_types,
            7 if autoencoder.config.degree_head else 0, feature_set=feature_set)
        block["dimension"] = int(xt.shape[1])
        qed_readout, fit_info = RidgeReadout.fit(xt, yt, target="qed")
        block["qed_fit"] = fit_info
        block["qed_real_heldout"] = readout_quality(qed_readout, xv, yv, groups=nv)
        # does the feature set retain AROMATICITY information? direct test.
        finite = np.isfinite(at)
        if finite.sum() >= 50:
            arom_readout, arom_info = RidgeReadout.fit(xt[finite], at[finite],
                                                       target="aromatic_rings")
            keep = np.isfinite(av)
            block["aromatic_fit"] = arom_info
            block["aromatic_real_heldout"] = readout_quality(
                arom_readout, xv[keep], av[keep], groups=nv[keep])
        xg, yg, ng = collect_generated(
            flow, autoencoder, scaler=scaler, targets_n=targets_n, qed_values=qed_values,
            samples=samples, temperature=temperature, device=device,
            feature_set=feature_set, eps_seed=eps_seed, decoders=decoders)
        if xg is None:
            block["qed_generated"] = {"note": "no valid generated molecules to score"}
            block["accepted"] = False
        else:
            block["qed_generated"] = readout_quality(qed_readout, xg, yg, groups=ng)
            overall_real = block["qed_real_heldout"]["overall"]
            overall_gen = block["qed_generated"]["overall"]
            block["accepted"] = bool(
                overall_real["mae"] is not None
                and overall_real["mae"] <= SURROGATE_MAX_MAE_REAL
                and (overall_real["correlation"] or 0) >= SURROGATE_MIN_CORR_REAL
                and overall_gen["mae"] is not None
                and overall_gen["mae"] <= SURROGATE_MAX_MAE_GENERATED
                and (overall_gen["correlation"] or 0) >= SURROGATE_MIN_CORR_GENERATED)
        block["readout"] = qed_readout.to_dict()
        report["feature_sets"][feature_set] = block
        if block.get("accepted") and chosen is None:
            chosen, chosen_name = qed_readout, feature_set
    # prefer `structural` when both pass: it is the superset
    if report["feature_sets"].get("structural", {}).get("accepted"):
        chosen_name = "structural"
        chosen = RidgeReadout(
            np.asarray(report["feature_sets"]["structural"]["readout"]["weights"]),
            report["feature_sets"]["structural"]["readout"]["bias"],
            np.asarray(report["feature_sets"]["structural"]["readout"]["feature_mean"]),
            np.asarray(report["feature_sets"]["structural"]["readout"]["feature_scale"]),
            report["feature_sets"]["structural"]["readout"]["alpha"], "qed")
    report["selected_feature_set"] = chosen_name
    report["qed_arms_enabled"] = chosen is not None
    if chosen is None:
        report["verdict"] = (
            "NO feature set met the precommitted surrogate thresholds on GENERATED "
            "embeddings. The QED arms are SKIPPED: optimising against an inaccurate "
            "surrogate would produce a meaningless result. The manifold arm still runs, "
            "because it uses no learned critic. If `structural` beats `moments` on the "
            "aromaticity readout but still fails on generated embeddings, the limitation "
            "is the generated distribution, not the feature set.")
    return report, chosen, chosen_name


# ------------------------------------------------------------------- phase B ---
@torch.no_grad()
def manifold_targets(autoencoder, dataset, *, n, device, batch_size, max_molecules,
                     feature_set):
    """Decoder statistics MEASURED on real molecules at this N. Per-N by construction.

    The targets are per-N and NOT per-QED: the manifold term's job is to keep the sample
    chemically plausible for its size, and the QED term is the only thing that is allowed
    to express the requested property. Making the manifold target QED-dependent would
    confound the two arms.
    """
    eligible = np.flatnonzero(dataset.n_node == n)
    if eligible.size < 20:
        return None, {"molecules": int(eligible.size), "note": "insufficient support"}
    chosen = eligible[:max_molecules].tolist()
    accumulated: dict[str, list] = {}
    for batch in iter_eval_batches(dataset, batch_size, chosen):
        batch = batch.to(device)
        x = autoencoder.encode(batch)
        outputs = autoencoder.decoder(x, batch.node_mask, batch.pair_mask,
                                      atom_types=None)
        statistics = decoder_statistics(outputs, batch.node_mask, batch.pair_mask)
        for key in ("bond_histogram", "degree_histogram", "edges_per_node"):
            if key in statistics:
                accumulated.setdefault(key, []).append(statistics[key].detach().cpu())
        rings = statistics["rings_from_bonds"] / statistics["n_node"]
        accumulated.setdefault("rings_per_node", []).append(rings.detach().cpu())
    targets, summary = {}, {"molecules": len(chosen)}
    for key, chunks in accumulated.items():
        stacked = torch.cat(chunks, dim=0)
        value = stacked.mean(dim=0)
        targets[key] = value.to(device)
        summary[key] = (value.tolist() if value.dim() else float(value))
    return targets, summary


@torch.no_grad()
def real_dispersion(autoencoder, dataset, *, n, device, batch_size, max_molecules):
    """Trust-region budget from the real within-N embedding spread at this size.

    The statistic is the STANDARD DEVIATION OF THE FROBENIUS NORM of real embeddings at
    this N. That choice is deliberate and has a limitation worth stating:

      * it is PERMUTATION-INVARIANT. Node embeddings carry no canonical ordering, so a
        distance to a within-N centroid would measure node-ordering mismatch as much as
        chemical difference; the norm does not have that confound.
      * it is therefore a ONE-DIMENSIONAL proxy. It bounds displacement by how much real
        molecules of this size differ in magnitude, not by the shape of the real set. It
        is a conservative budget, not a claim that the guided point stays on the data
        manifold -- that is what the manifold arm and the measured validity are for.

    `norm_mean` is reported alongside so the radius can be read as a FRACTION of the real
    embedding scale, which is the only way the displacement numbers are interpretable.
    """
    eligible = np.flatnonzero(dataset.n_node == n)
    if eligible.size < 10:
        return None
    pooled = []
    for batch in iter_eval_batches(dataset, batch_size, eligible[:max_molecules].tolist()):
        batch = batch.to(device)
        x = autoencoder.encode(batch)
        pooled.append((x * batch.node_mask.unsqueeze(-1)).flatten(1).norm(dim=1).cpu())
    norms = torch.cat(pooled)
    return {"molecules": int(norms.numel()), "norm_std": float(norms.std()),
            "norm_mean": float(norms.mean()),
            "statistic": "std of the Frobenius norm of real embeddings at this N"}


def summarise(rows, *, qed_target, baseline=None) -> dict[str, Any]:
    keys = ("qed", "aromatic_rings", "heavy_atoms")
    total = len(rows)
    valid = [r for r in rows if r.get("valid") and all(k in r for k in keys)]
    block: dict[str, Any] = {
        "attempts": total, "valid": len(valid),
        "validity": len(valid) / max(1, total),
        "fallback_rate": sum(bool(r.get("fallback")) for r in rows) / max(1, total),
        "unique_smiles": len({r["smiles"] for r in valid}) if valid else 0,
        "uniqueness_over_valid": (len({r["smiles"] for r in valid}) / len(valid)
                                  if valid else None),
    }
    for key in keys:
        series = [r[key] for r in valid]
        block[f"{key}_mean"] = float(np.mean(series)) if series else None
    if valid:
        block["qed_mae_vs_target"] = float(
            np.mean([abs(r["qed"] - float(qed_target)) for r in valid]))
        surrogate = [r["surrogate_qed"] for r in valid if r.get("surrogate_qed") is not None]
        if surrogate:
            block["surrogate_qed_mean"] = float(np.mean(surrogate))
            block["surrogate_minus_actual"] = float(
                np.mean(surrogate) - np.mean([r["qed"] for r in valid]))
    else:
        block["qed_mae_vs_target"] = None
    if baseline is not None:
        pairs = [(b, r) for b, r in zip(baseline, rows)
                 if b.get("valid") and "qed" in b and r.get("valid") and "qed" in r]
        block["paired_jointly_valid"] = len(pairs)
        block["paired_qed_delta_vs_native"] = bootstrap_ci(
            [r["qed"] - b["qed"] for b, r in pairs])
        block["paired_heavy_atom_delta"] = bootstrap_ci(
            [r["heavy_atoms"] - b["heavy_atoms"] for b, r in pairs])
    return block


@torch.no_grad()
def decode_rows(autoencoder, embeddings, n, decoders, *, readout, feature_set,
                node_mask, pair_mask):
    from rdkit import Chem, rdBase
    from rdkit.Chem import QED, rdMolDescriptors
    from zinc_gnf.generation import decode_embeddings

    surrogate = None
    if readout is not None:
        f = embedding_features(autoencoder, embeddings, node_mask, pair_mask,
                               feature_set=feature_set)
        surrogate = readout.predict_numpy(f.detach().cpu().numpy().astype(np.float64))
    records = decode_embeddings(autoencoder, embeddings,
                                [n] * embeddings.shape[0], decoders=tuple(decoders))
    out: dict[str, list] = {d: [] for d in decoders}
    for d in decoders:
        for index, record in enumerate(records):
            entry = record.get(d) or {}
            row: dict[str, Any] = {"valid": bool(entry.get("valid")),
                                   "fallback": bool(entry.get("fallback", False)),
                                   "smiles": entry.get("smiles")}
            if surrogate is not None:
                row["surrogate_qed"] = float(surrogate[index])
            if entry.get("valid") and entry.get("smiles"):
                with rdBase.BlockLogs():
                    mol = Chem.MolFromSmiles(entry["smiles"])
                if mol is not None:
                    row["qed"] = float(QED.qed(mol))
                    row["aromatic_rings"] = float(
                        rdMolDescriptors.CalcNumAromaticRings(mol))
                    row["heavy_atoms"] = float(mol.GetNumHeavyAtoms())
            out[d].append(row)
    return out


def internal_diversity(smiles_list) -> float | None:
    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import AllChem
    prints = []
    for s in smiles_list:
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(s)
        if mol is not None:
            prints.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
    if len(prints) < 2:
        return None
    total, count = 0.0, 0
    for i in range(len(prints)):
        sims = DataStructs.BulkTanimotoSimilarity(prints[i], prints[i + 1:])
        total += float(sum(sims)); count += len(sims)
    return 1.0 - total / max(1, count)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flow-checkpoint", required=True)
    p.add_argument("--ae-checkpoint", default=None)
    p.add_argument("--data-dir", default="data/processed/zinc250k_charge_aware_moflow_fixed")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target-n", default="15,23,30")
    p.add_argument("--qed-values", default="0.4,0.6,0.8")
    p.add_argument("--samples-per-cell", type=int, default=8)
    p.add_argument("--guidance-steps", type=int, default=15)
    p.add_argument("--step-size", type=float, default=0.15)
    p.add_argument("--trust-radius-scale", type=float, default=1.0)
    p.add_argument("--lambda-qed", type=float, default=1.0)
    p.add_argument("--lambda-manifold", type=float, default=1.0)
    p.add_argument("--surrogate-train-molecules", type=int, default=3000)
    p.add_argument("--surrogate-val-molecules", type=int, default=800)
    p.add_argument("--surrogate-generated-samples", type=int, default=12)
    p.add_argument("--real-reference-molecules", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--decoders", default="degree,constrained")
    p.add_argument("--eps-seed", type=int, default=20261114)
    p.add_argument("--allow-ae-mismatch", action="store_true")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        targets_n = [int(v) for v in args.target_n.split(",") if v.strip()]
        qed_values = [float(v) for v in args.qed_values.split(",") if v.strip()]
        decoders = [d.strip() for d in args.decoders.split(",") if d.strip()]
        if not targets_n or len(qed_values) < 2 or not decoders:
            raise ValueError("need at least one N, two QED targets and one decoder")
        if args.samples_per_cell < 2 or args.guidance_steps < 1:
            raise ValueError("samples-per-cell >= 2 and guidance-steps >= 1")
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4

    started = time.perf_counter()
    try:
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        device = torch.device(args.device)
        scaler = load_condition_scaler(args.data_dir)
        train_records, val_records, fingerprint = load_training_splits(args.data_dir)
        train_set = MolecularGraphDataset(train_records)
        val_set = MolecularGraphDataset(val_records)
        del train_records, val_records

        flow, probe = load_flow(args.flow_checkpoint, device=device)
        flow.eval().requires_grad_(False)
        recorded = probe.get("autoencoder") or {}
        ae_path = Path(args.ae_checkpoint or recorded.get("checkpoint", ""))
        if not ae_path.is_file():
            raise FileNotFoundError(f"autoencoder not found: {ae_path}")
        digest = file_sha256(ae_path)
        match = bool(recorded.get("sha256") == digest)
        if not match and not args.allow_ae_mismatch:
            raise ResumeMismatchError(
                f"autoencoder SHA256 mismatch: recorded {str(recorded.get('sha256'))[:16]}, "
                f"actual {digest[:16]}")
        autoencoder, _ = load_autoencoder(ae_path, device=device)
        autoencoder.eval().requires_grad_(False)          # FROZEN during guidance
        print(f"flow step={probe.get('step')}  AE provenance match={match}", flush=True)

        payload: dict[str, Any] = {
            "framing": ("INFERENCE-TIME GUIDANCE, NOT intrinsic conditional generation. "
                        "Every number here is post-hoc guidance applied to a frozen flow "
                        "and a frozen decoder. It is a mechanism test, not a model "
                        "improvement claim."),
            "flow": {"checkpoint": str(args.flow_checkpoint), "step": probe.get("step"),
                     "sha256": file_sha256(args.flow_checkpoint)},
            "autoencoder": {"checkpoint": str(ae_path), "sha256_actual": digest,
                            "sha256_recorded_in_flow": recorded.get("sha256"),
                            "sha256_match": match,
                            "mismatch_override_used": bool(not match)},
            "data_fingerprint": fingerprint,
            "settings": vars(args) | {"arms": list(ARMS)},
        }

        print("\n--- PHASE A: surrogate validation (hard gate) ---", flush=True)
        phase_a_start = time.perf_counter()
        surrogate_report, readout, feature_set = validate_surrogates(
            flow, autoencoder, train_set, val_set, scaler=scaler, device=device,
            batch_size=args.batch_size, train_molecules=args.surrogate_train_molecules,
            val_molecules=args.surrogate_val_molecules, targets_n=targets_n,
            qed_values=qed_values, samples=args.surrogate_generated_samples,
            temperature=args.temperature, eps_seed=args.eps_seed, decoders=decoders)
        surrogate_report["seconds"] = time.perf_counter() - phase_a_start
        payload["phase_a_surrogate"] = surrogate_report
        for name, block in surrogate_report["feature_sets"].items():
            real = block.get("qed_real_heldout", {}).get("overall", {})
            gen = block.get("qed_generated", {}).get("overall", {})
            arom = block.get("aromatic_real_heldout", {}).get("overall", {})
            print(f"  [{name}] real QED MAE={real.get('mae')} corr={real.get('correlation')} "
                  f"| generated QED MAE={gen.get('mae')} corr={gen.get('correlation')} "
                  f"| aromatic corr={arom.get('correlation')} -> accepted="
                  f"{block.get('accepted')}", flush=True)
        qed_enabled = readout is not None
        if not qed_enabled:
            print(f"  {surrogate_report['verdict']}", flush=True)
        feature_set = feature_set or "structural"

        print("\n--- PHASE B: bounded pilot ---", flush=True)
        payload["by_n"] = {}
        for n in targets_n:
            radius_base = real_dispersion(autoencoder, val_set, n=n, device=device,
                                          batch_size=args.batch_size,
                                          max_molecules=args.real_reference_molecules)
            targets, target_summary = manifold_targets(
                autoencoder, train_set, n=n, device=device, batch_size=args.batch_size,
                max_molecules=args.real_reference_molecules, feature_set=feature_set)
            block: dict[str, Any] = {"manifold_targets": target_summary,
                                     "real_dispersion": radius_base}
            if radius_base is None or targets is None:
                block["note"] = "insufficient real support at this N"
                payload["by_n"][str(n)] = block
                continue
            radius = float(args.trust_radius_scale) * radius_base["norm_std"]
            block["trust_radius"] = radius
            block["trust_radius_as_fraction_of_real_norm"] = (
                radius / max(1e-12, radius_base["norm_mean"]))
            mask = torch.ones((args.samples_per_cell, n), dtype=torch.bool, device=device)
            pair = pair_masks_for(mask)
            # COMMON noise across QED targets, the existing paired protocol
            generator = torch.Generator(device=device).manual_seed(int(args.eps_seed) + n)
            eps = torch.randn((args.samples_per_cell, n,
                               flow.config.node_embedding_dim), device=device,
                              generator=generator)
            block["cells"] = {}
            for q in qed_values:
                cond = reference_condition(flow, scaler, n, q, args.samples_per_cell, device)
                with torch.no_grad():      # f^-1 is NEVER in a gradient path (v1's failure)
                    mu, sigma = flow.prior.parameters_for(args.samples_per_cell, cond,
                                                          device)
                    z = (mu.unsqueeze(1)
                         + float(args.temperature) * sigma.unsqueeze(1) * eps) \
                        * mask.unsqueeze(-1)
                    x0 = flow.inverse(z, mask, cond).detach()      # ONCE per cell
                cell: dict[str, Any] = {"arms": {}}
                native_rows = None
                for arm in ARMS:
                    if arm in ("qed_only", "combined") and not qed_enabled:
                        cell["arms"][arm] = {"skipped": "surrogate failed Phase A"}
                        continue
                    lam_q = {"native": 0.0, "qed_only": args.lambda_qed,
                             "manifold_only": 0.0, "combined": args.lambda_qed,
                             "random": 0.0}[arm]
                    lam_m = {"native": 0.0, "qed_only": 0.0,
                             "manifold_only": args.lambda_manifold,
                             "combined": args.lambda_manifold, "random": 0.0}[arm]
                    arm_start = time.perf_counter()
                    if arm == "native":
                        guided, report = x0, {"steps_requested": 0, "nonfinite_steps": 0,
                                              "displacement_mean": 0.0}
                    else:
                        rng = torch.Generator(device=device).manual_seed(
                            int(args.eps_seed) + 977 * n)
                        guided, report = guide(
                            autoencoder, x0, mask, pair, readout=readout,
                            targets=targets, qed_target=q, lambda_qed=lam_q,
                            lambda_manifold=lam_m, steps=args.guidance_steps,
                            step_size=args.step_size, trust_radius=radius,
                            feature_set=feature_set,
                            direction="random" if arm == "random" else "gradient",
                            generator=rng)
                    rows = decode_rows(autoencoder, guided, n, decoders,
                                       readout=readout, feature_set=feature_set,
                                       node_mask=mask, pair_mask=pair)
                    if arm == "native":
                        native_rows = rows
                    entry: dict[str, Any] = {"guidance": report,
                                             "seconds": time.perf_counter() - arm_start,
                                             "decoders": {}}
                    for d in decoders:
                        entry["decoders"][d] = summarise(
                            rows[d], qed_target=q,
                            baseline=None if arm == "native" else (native_rows or {}).get(d))
                        valid_smiles = [r["smiles"] for r in rows[d]
                                        if r.get("valid") and r.get("smiles")]
                        entry["decoders"][d]["internal_diversity"] = internal_diversity(
                            valid_smiles)
                    cell["arms"][arm] = entry
                    head = entry["decoders"][decoders[0]]
                    print(f"  N={n} q={q} [{arm:<14}] actual QED="
                          f"{head.get('qed_mean')} MAE={head.get('qed_mae_vs_target')} "
                          f"raw/corr validity={head.get('validity'):.3f} "
                          f"arom={head.get('aromatic_rings_mean')} "
                          f"uniq={head.get('uniqueness_over_valid')} "
                          f"surr={head.get('surrogate_qed_mean')} "
                          f"{entry['seconds']:.1f}s", flush=True)
                block["cells"][f"{q:.2f}"] = cell
            # paired responsiveness across targets on common noise, per arm
            block["paired_response_low_to_high"] = {}
            lo, hi = f"{min(qed_values):.2f}", f"{max(qed_values):.2f}"
            for arm in ARMS:
                try:
                    a = block["cells"][lo]["arms"][arm]["decoders"][decoders[0]]
                    b = block["cells"][hi]["arms"][arm]["decoders"][decoders[0]]
                except (KeyError, TypeError):
                    continue
                if a.get("qed_mean") is not None and b.get("qed_mean") is not None:
                    block["paired_response_low_to_high"][arm] = {
                        "delta_mean_qed": b["qed_mean"] - a["qed_mean"],
                        "note": ("difference of cell means at common noise; not a "
                                 "per-molecule paired statistic because the two cells "
                                 "use different conditions"),
                    }
            payload["by_n"][str(n)] = block

        payload["elapsed_seconds"] = time.perf_counter() - started
        payload["runtime_note"] = (
            f"MEASURED, not assumed: {payload['elapsed_seconds']:.1f}s total for "
            f"{args.samples_per_cell} samples/cell. Scaling to 100/cell multiplies "
            f"Phase B by about {100 / max(1, args.samples_per_cell):.1f}x.")
        atomic_write_json(payload, out_dir / "guidance_pilot.json")
        (out_dir / "guidance_pilot.txt").write_text(format_pilot(payload, decoders))
        print(f"\n{payload['runtime_note']}")
        print(f"wrote {out_dir / 'guidance_pilot.json'}")
        print(f"wrote {out_dir / 'guidance_pilot.txt'}")
        return 0
    except FileNotFoundError as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 2
    except ResumeMismatchError as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 3
    except Exception:  # noqa: BLE001
        print("FATAL: unexpected\n" + traceback.format_exc(), file=sys.stderr, flush=True)
        return 1


def format_pilot(payload: dict[str, Any], decoders: list[str]) -> str:
    head = decoders[0]
    lines = ["Decoder-aware guidance pilot (Stage 1)", "=" * 78, "",
             payload["framing"], "",
             f"flow: {payload['flow']['checkpoint']} (step {payload['flow']['step']})",
             f"autoencoder provenance match: {payload['autoencoder']['sha256_match']}",
             f"headline decoder: {head}", "",
             "PHASE A -- surrogate validation", "-" * 78]
    a = payload.get("phase_a_surrogate", {})
    for name, block in (a.get("feature_sets") or {}).items():
        real = block.get("qed_real_heldout", {}).get("overall", {})
        gen = block.get("qed_generated", {}).get("overall", {})
        arom = block.get("aromatic_real_heldout", {}).get("overall", {})
        lines += [f"  [{name}] dim={block.get('dimension')}",
                  f"    QED, held-out REAL      MAE={real.get('mae')} "
                  f"corr={real.get('correlation')}",
                  f"    QED, GENERATED (binding) MAE={gen.get('mae')} "
                  f"corr={gen.get('correlation')}",
                  f"    AROMATIC RINGS, REAL     corr={arom.get('correlation')} "
                  f"MAE={arom.get('mae')}",
                  f"    accepted: {block.get('accepted')}"]
    lines += [f"  selected feature set: {a.get('selected_feature_set')}",
              f"  QED arms enabled: {a.get('qed_arms_enabled')}"]
    if a.get("verdict"):
        lines.append(f"  {a['verdict']}")
    lines += ["", "PHASE B -- bounded pilot", "-" * 78]
    for n, block in (payload.get("by_n") or {}).items():
        dispersion = block.get("real_dispersion") or {}
        lines.append(
            f"--- N={n}  trust_radius={block.get('trust_radius')}  "
            f"(real norm: mean={dispersion.get('norm_mean')} "
            f"std={dispersion.get('norm_std')}; radius is "
            f"{block.get('trust_radius_as_fraction_of_real_norm')} of the mean) ---")
        if block.get("note"):
            lines += [f"  {block['note']}", ""]
            continue
        for q, cell in (block.get("cells") or {}).items():
            lines.append(f"  QED target {q}")
            for arm, entry in (cell.get("arms") or {}).items():
                if entry.get("skipped"):
                    lines.append(f"    {arm:<14} SKIPPED ({entry['skipped']})")
                    continue
                d = entry["decoders"][head]
                g = entry.get("guidance", {})
                lines.append(
                    f"    {arm:<14} QED={d.get('qed_mean')} "
                    f"MAE={d.get('qed_mae_vs_target')} "
                    f"validity={d.get('validity'):.3f} "
                    f"fallback={d.get('fallback_rate'):.3f} "
                    f"arom={d.get('aromatic_rings_mean')} "
                    f"uniq={d.get('uniqueness_over_valid')} "
                    f"intdiv={d.get('internal_diversity')} "
                    f"atoms={d.get('heavy_atoms_mean')}")
                if d.get("surrogate_qed_mean") is not None:
                    lines.append(
                        f"       surrogate={d['surrogate_qed_mean']:.4f} "
                        f"surrogate-actual={d.get('surrogate_minus_actual')}  "
                        f"<- divergence here is exploitation")
                if d.get("paired_qed_delta_vs_native"):
                    p = d["paired_qed_delta_vs_native"]
                    lines.append(f"       paired dQED vs native={p.get('mean')} "
                                 f"CI={p.get('ci95')} n={p.get('n')}")
                lines.append(f"       displacement={g.get('displacement_mean')} "
                             f"nonfinite_steps={g.get('nonfinite_steps')} "
                             f"saturated={g.get('trust_region_saturated_fraction')} "
                             f"{entry.get('seconds', 0):.1f}s")
        if block.get("paired_response_low_to_high"):
            lines.append("  response (mean QED at highest target minus lowest):")
            for arm, r in block["paired_response_low_to_high"].items():
                lines.append(f"    {arm:<14} {r['delta_mean_qed']:+.4f}")
        lines.append("")
    lines += [payload.get("runtime_note", ""), "",
              "READ THIS AS GUIDANCE, NOT CONDITIONING. Scale to 100 samples/cell only",
              "if the pilot shows actual RDKit QED moving toward the target with raw",
              "validity not below native and uniqueness holding."]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
