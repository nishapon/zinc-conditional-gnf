#!/usr/bin/env python3
"""Step 3: frozen autoencoder reconstruction chemistry audit (validation only).

A molecule's true graph is encoded and decoded WITHOUT ground-truth atom labels
at decode time. Compare the real molecule with raw, valence-constrained, and
(optional) heavy-degree-constrained reconstructions. Measure validity,
connectivity, exact identity, aromatic rings and QED, including matched-N
strata and selection coverage. QED here is reconstruction fidelity, NOT
conditional-generation accuracy.

No training, no sampling from the GNF, no checkpoint mutation, no test split.
Uses the EXACT autoencoder recorded in the supplied Stage 17 GNF checkpoint
and rejects SHA-256 mismatches rather than silently swapping autoencoders.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import QED, rdMolDescriptors

from zinc_gnf.chemistry import canonical_connectivity_smiles
from zinc_gnf.chemistry_charge_aware import graph_to_mol_charge_aware
from zinc_gnf.dataset import MolecularGraphDataset, iter_eval_batches, load_training_splits, seeded_subset_indices
from zinc_gnf.decoding_charge_aware import degree_constrained_bond_decode, valence_aware_bond_decode_typed
from zinc_gnf.trainer_autoencoder import load_autoencoder
from zinc_gnf.trainer_flow import file_sha256
from zinc_gnf.utils.runtime import atomic_write_json


def safe_molecule(atoms: np.ndarray, bonds: np.ndarray):
    try:
        with rdBase.BlockLogs():
            mol = graph_to_mol_charge_aware(atoms, bonds)
        return mol, (len(Chem.GetMolFrags(mol)) == 1)
    except Exception:
        return None, False


def features(mol) -> dict[str, float | int]:
    return {
        "qed": float(QED.qed(mol)),
        "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "total_rings": int(rdMolDescriptors.CalcNumRings(mol)),
        "n": mol.GetNumHeavyAtoms(),
    }


def finite_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or np.std(xs) < 1e-10 or np.std(ys) < 1e-10:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def aggregate(rows: list[dict], mode: str) -> dict:
    total = len(rows)
    valid = [r for r in rows if r[mode]["strict_valid"]]
    vr = [r[mode] for r in valid]
    n_valid = len(valid)
    vq = [float(x["qed"]) for x in vr]
    rq = [float(r["true_qed"]) for r in valid]
    va = [float(x["aromatic_rings"]) for x in vr]
    ra = [float(r["true_aromatic_rings"]) for r in valid]
    arom_diff = [a - b for a, b in zip(va, ra)]
    qdiff = [a - b for a, b in zip(vq, rq)]
    return {
        "molecules": total,
        "strict_valid_count": n_valid,
        "strict_valid_rate_all": n_valid / total if total else None,
        "sanitizable_rate_all": sum(r[mode]["sanitizable"] for r in rows) / total if total else None,
        "exact_smiles_rate_all": sum(r[mode]["exact_smiles"] for r in rows) / total if total else None,
        "atom_count_preserved_rate_all": sum(r[mode]["exact_n"] for r in rows) / total if total else None,
        "real_qed_mean_all": finite_mean([float(r["true_qed"]) for r in rows]),
        "real_aromatic_rings_mean_all": finite_mean([float(r["true_aromatic_rings"]) for r in rows]),
        "real_qed_mean_valid_selection": finite_mean(rq),
        "decoded_qed_mean_valid": finite_mean(vq),
        "qed_mae_valid_paired": finite_mean([abs(v) for v in qdiff]),
        "qed_bias_valid_paired": finite_mean(qdiff),
        "qed_correlation_valid_paired": corr(rq, vq),
        "real_aromatic_rings_mean_valid_selection": finite_mean(ra),
        "decoded_aromatic_rings_mean_valid": finite_mean(va),
        "aromatic_rings_mean_delta_valid_paired": finite_mean(arom_diff),
        "aromatic_ring_abs_error_valid_paired": finite_mean([abs(v) for v in arom_diff]),
        "fraction_with_at_least_one_aromatic_ring_real_valid_selection": (
            sum(v > 0 for v in ra) / n_valid if n_valid else None),
        "fraction_with_at_least_one_aromatic_ring_decoded_valid": (
            sum(v > 0 for v in va) / n_valid if n_valid else None),
        "mean_valence_rescue_edges": finite_mean([float(r[mode].get("rescue_edges", 0)) for r in rows]),
        "degree_fallback_rate": (sum(bool(r[mode].get("fallback", False)) for r in rows) / total
                                  if total and mode == "degree" else None),
    }


def audit(args):
    if args.max_molecules < 1 or args.batch_size < 1:
        raise ValueError("--max-molecules and --batch-size must be positive")
    if not Path(args.flow_checkpoint).is_file():
        raise FileNotFoundError(f"flow checkpoint not found: {args.flow_checkpoint}")
    # Read checkpoint metadata only; don't instantiate a second GNF on CPU.
    meta = torch.load(args.flow_checkpoint, map_location="cpu", weights_only=False)
    recorded = meta.get("autoencoder") or {}
    ae_path = Path(args.ae_checkpoint or recorded.get("checkpoint", ""))
    if not ae_path.is_file():
        raise FileNotFoundError(f"autoencoder checkpoint not found: {ae_path}; "
                                "specify --ae-checkpoint if the file has been moved")
    digest = file_sha256(ae_path)
    if recorded.get("sha256") and digest != recorded["sha256"]:
        raise ValueError("autoencoder SHA-256 does NOT match Stage 17 flow checkpoint; "
                         "refusing to evaluate a different model")
    model, ae_meta = load_autoencoder(ae_path, device=args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _, val_records, fingerprint = load_training_splits(args.data_dir)
    ds = MolecularGraphDataset(val_records)
    del val_records
    order = seeded_subset_indices(len(ds), args.max_molecules, args.seed)
    modes = ["raw", "constrained"]
    if model.config.degree_head and not args.skip_degree:
        from scipy.optimize import milp  # noqa: F401 - explicit availability test
        modes.append("degree")
    print(f"AE chemistry audit: {len(order)} validation molecules; "
          f"modes={modes}; flow checkpoint {args.flow_checkpoint}", flush=True)
    print(f"autoencoder={ae_path} sha256={digest[:16]} step={ae_meta.get('step')}", flush=True)
    print("IMPORTANT: no GNF sampling; reconstruction of REAL molecule embeddings only", flush=True)

    rows: list[dict] = []
    counts = defaultdict(int)
    torch.manual_seed(args.seed)
    generator = (torch.Generator(device=args.device).manual_seed(args.seed)
                 if model.config.symmetry_noise_dim else None)
    with torch.no_grad():
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start + args.batch_size]
            batch = next(iter_eval_batches(ds, len(ids), ids)).to(args.device)
            nodes = model.encode(batch, generator=generator)
            outputs = model.decode(nodes, batch, teacher_forcing=False)
            atom_preds = outputs["atom_logits"].argmax(-1).cpu().numpy()
            bond_preds = outputs["bond_logits"].argmax(-1).cpu().numpy()
            bond_probs = outputs["bond_logits"].softmax(dim=-1).cpu().numpy()
            degree_preds = (outputs["degree_logits"].argmax(-1).cpu().numpy()
                            if "degree" in modes else None)
            for bi, idx in enumerate(ids):
                n = int(batch.n_node[bi])
                atoms, bonds = ds.true_graph(idx)
                rmol, connected = safe_molecule(atoms, bonds)
                if rmol is None or not connected:
                    raise RuntimeError(f"stored validation molecule {idx} could not round-trip through RDKit")
                rf = features(rmol)
                predicted_atoms = atom_preds[bi, :n].astype(np.int64)
                probabilities = bond_probs[bi, :n, :n]
                raw_bonds = bond_preds[bi, :n, :n].astype(np.int64).copy()
                np.fill_diagonal(raw_bonds, 0)
                candidates = {"raw": (raw_bonds, {})}
                cb, cinfo = valence_aware_bond_decode_typed(predicted_atoms, probabilities)
                candidates["constrained"] = (cb, cinfo)
                if "degree" in modes:
                    db, dinfo = degree_constrained_bond_decode(
                        predicted_atoms, probabilities, degree_preds[bi, :n])
                    candidates["degree"] = (db, dinfo)
                row = {
                    "dataset_index": int(idx), "row_id": int(ds.row_id[idx]),
                    "n": n, "true_qed": rf["qed"],
                    "true_aromatic_rings": rf["aromatic_rings"],
                    "true_total_rings": rf["total_rings"],
                    "atoms_correct": int(np.sum(predicted_atoms == atoms)),
                    "atoms_total": n,
                    "atom_sequence_exact": bool(np.array_equal(predicted_atoms, atoms)),
                    "bond_correct_upper": int(np.sum(
                        (np.triu(raw_bonds == bonds, k=1)))),
                    "bond_total_upper": n * (n - 1) // 2,
                    "true_bonds_upper": int(np.sum(np.triu(bonds > 0, k=1))),
                    "real_bonds_correct_upper": int(np.sum(np.triu((raw_bonds == bonds) & (bonds > 0), k=1))),
                }
                ref_smiles = canonical_connectivity_smiles(rmol)
                for mode, (proposed_bonds, info) in candidates.items():
                    dm, single = safe_molecule(predicted_atoms, proposed_bonds)
                    m = {"sanitizable": bool(dm is not None),
                         "strict_valid": bool(dm is not None and single),
                         "exact_n": bool(dm is not None and dm.GetNumHeavyAtoms() == n),
                         "exact_smiles": False,
                         "qed": None, "aromatic_rings": None,
                         "rescue_edges": int(info.get("rescue_edges", 0)),
                         "fallback": bool(info.get("fallback", False))}
                    if m["strict_valid"]:
                        mf = features(dm)
                        m.update(qed=mf["qed"], aromatic_rings=mf["aromatic_rings"],
                                 exact_smiles=(canonical_connectivity_smiles(dm) == ref_smiles))
                    row[mode] = m
                    counts[f"{mode}_valid"] += int(m["strict_valid"])
                rows.append(row)
            print(f"processed {len(rows)}/{len(order)}", flush=True)

    summary = {
        "experiment": "Stage 17 frozen autoencoder real-molecule chemistry reconstruction",
        "scope": "validation split only; no sampling; decoder receives no truth atom labels",
        "flow_checkpoint": args.flow_checkpoint,
        "autoencoder_checkpoint": str(ae_path),
        "autoencoder_sha256": digest,
        "autoencoder_config": model.config.to_dict(),
        "flow_step": meta.get("step"), "ae_step": ae_meta.get("step"),
        "subset_seed": args.seed, "n_evaluated": len(rows),
        "dataset_fingerprint": fingerprint,
        "atom_accuracy": sum(r["atoms_correct"] for r in rows) / sum(r["atoms_total"] for r in rows),
        "atom_sequence_exact": sum(r["atom_sequence_exact"] for r in rows) / len(rows),
        "bond_pair_accuracy_raw": sum(r["bond_correct_upper"] for r in rows) / sum(r["bond_total_upper"] for r in rows),
        "real_bond_accuracy_raw": (sum(r["real_bonds_correct_upper"] for r in rows) /
                                   max(1, sum(r["true_bonds_upper"] for r in rows))),
        "modes": {mode: aggregate(rows, mode) for mode in modes},
        "by_n": {},
    }
    for n in sorted(set(r["n"] for r in rows)):
        sub = [r for r in rows if r["n"] == n]
        summary["by_n"][str(n)] = {mode: aggregate(sub, mode) for mode in modes}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(summary, out / "ae_chemistry_audit.json")
    if args.save_per_molecule:
        with (out / "ae_chemistry_per_molecule.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, allow_nan=False) + "\n")
    text = [f"Frozen AE chemistry audit | validation n={len(rows)} | seed={args.seed}",
            f"Flow: {args.flow_checkpoint}", f"AE: {ae_path}",
            "THIS IS RECONSTRUCTION OF REAL MOLECULES, NOT GENERATION / QED TARGET ACCURACY.",
            f"Atom accuracy {summary['atom_accuracy']:.4f} | entire atom sequence exact {summary['atom_sequence_exact']:.4f}",
            f"Raw pair-bond accuracy {summary['bond_pair_accuracy_raw']:.4f} | real-bond accuracy {summary['real_bond_accuracy_raw']:.4f}"]
    for mode in modes:
        m = summary["modes"][mode]
        def fmt(x):
            return "NA" if x is None else f"{x:.4f}"
        text += [f"\n{mode}: strict valid {fmt(m['strict_valid_rate_all'])} (n={m['strict_valid_count']}/{len(rows)}); "
                 f"exact SMILES {fmt(m['exact_smiles_rate_all'])}",
                 f"  QED reconstruction: real(valid subset) {fmt(m['real_qed_mean_valid_selection'])} "
                 f"decoded(valid) {fmt(m['decoded_qed_mean_valid'])} "
                 f"MAE {fmt(m['qed_mae_valid_paired'])} bias {fmt(m['qed_bias_valid_paired'])} "
                 f"correlation {fmt(m['qed_correlation_valid_paired'])}",
                 f"  Aromatic rings: real(all) {fmt(m['real_aromatic_rings_mean_all'])} "
                 f"real(valid subset) {fmt(m['real_aromatic_rings_mean_valid_selection'])} "
                 f"decoded(valid) {fmt(m['decoded_aromatic_rings_mean_valid'])} "
                 f"paired delta {fmt(m['aromatic_rings_mean_delta_valid_paired'])}"]
    for target in (15, 23, 30):
        if str(target) not in summary["by_n"]:
            continue
        text.append(f"\nN={target} (n={sum(r['n']==target for r in rows)})")
        for mode in modes:
            m = summary["by_n"][str(target)][mode]
            text.append(f"  {mode}: validity {fmt(m['strict_valid_rate_all'])} "
                        f"QED MAE {fmt(m['qed_mae_valid_paired'])} "
                        f"arom real(valid) {fmt(m['real_aromatic_rings_mean_valid_selection'])} "
                        f"decoded {fmt(m['decoded_aromatic_rings_mean_valid'])}")
    (out / "ae_chemistry_audit.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    print("\n".join(text), flush=True)
    print(f"Saved reports to {out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow-checkpoint", default="outputs/checkpoints/warm_start_stageA/last.pt")
    p.add_argument("--ae-checkpoint", default=None)
    p.add_argument("--data-dir", default="data/processed/zinc250k_charge_aware_moflow_fixed")
    p.add_argument("--max-molecules", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=20261028)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--skip-degree", action="store_true", help="skip degree-constrained decoding even if available")
    p.add_argument("--save-per-molecule", action="store_true")
    p.add_argument("--output-dir", default="outputs/metrics/ae_chemistry_audit")
    args = p.parse_args()
    try:
        audit(args)
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
