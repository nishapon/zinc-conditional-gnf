#!/usr/bin/env python3
"""Extract slide assets from EXISTING artefacts: molecule images + architecture facts.

Generates no molecules and loads no GPU. Reads the saved samples.jsonl trees, renders
RDKit molecule grids, and prints a compact factual summary that can be pasted into a
conversation.

Four blocks:
  A  architecture facts for both checkpoints and the shared decoder, plus a search for
     the QED contrastive-loss settings in the training code and configs
  B  showcase molecule grids, selected BY RULE (valid, prefer no assembler fallback,
     then nearest achieved-to-requested QED) so nothing is cherry-picked by eye
  C  chemistry by heavy-atom count against a SIZE-MATCHED real-ZINC reference
  D  achieved-QED spread per cell, for distribution plots

Writes PNGs and CSVs under <output-dir>; prints a short summary to stdout.

Exit codes: 0 ran (individual blocks may report failure); 4 bad arguments.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GRID_N = (15, 23, 30)
GRID_QED = (0.4, 0.6, 0.8)
MODELS = ("baseline", "conditional")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-t02", default="outputs/metrics/unconditional_vs_nqed_contrast")
    parser.add_argument("--results-t085", default="outputs/metrics/unconditional_vs_nqed_contrast_t085")
    parser.add_argument("--baseline-checkpoint", default="outputs/checkpoints/flow_n_only/seed_3/best.pt")
    parser.add_argument("--conditional-checkpoint",
                        default="outputs/checkpoints/flow_nqed_ae128_qedcontrast/seed_3/best.pt")
    parser.add_argument("--ae-checkpoint", default="outputs/ablation/ae_degree_128d/seed_3/last.pt")
    parser.add_argument("--data-dir", default="data/processed/zinc250k_charge_aware_moflow_fixed")
    parser.add_argument("--extra-root", default=None,
                        help="second directory to search for checkpoints, data and the "
                             "original training code (the dirty project worktree)")
    parser.add_argument("--output-dir", default="deck_assets")
    parser.add_argument("--per-grid", type=int, default=6)
    parser.add_argument("--reference-per-size", type=int, default=3000)
    return parser.parse_args(argv)


def roots(args):
    out = [ROOT]
    if args.extra_root:
        out.append(Path(args.extra_root))
    return out


def find(args, relative):
    for root in roots(args):
        candidate = root / relative
        if candidate.exists():
            return candidate
    return None


def cell_tag(n, qed):
    return ("n%d_qed%.3f" % (int(n), float(qed))).replace(".", "p")


def load_cell(results, model, n, qed):
    pattern = str(Path(results) / "conditional" / "grid" / model / cell_tag(n, qed)
                  / "rep*" / "samples.jsonl")
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def summarise(values):
    values = sorted(float(v) for v in values if v is not None)
    if not values:
        return None
    mean = sum(values) / len(values)
    sd = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    def q(f):
        return values[min(len(values) - 1, int(f * (len(values) - 1)))]
    return {"n": len(values), "mean": mean, "sd": sd, "q10": q(0.10),
            "median": q(0.50), "q90": q(0.90), "q95": q(0.95)}


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    names = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------------ block A ---
def block_a(args, out):
    print("=" * 72)
    print("BLOCK A :: ARCHITECTURE FACTS")
    print("=" * 72)
    rows = []
    try:
        import torch

        from zinc_gnf.flow import FlowConfig, NodeFlow

        for label, relative in (("BASELINE (N-only)", args.baseline_checkpoint),
                                ("CONDITIONAL (N+QED)", args.conditional_checkpoint)):
            path = find(args, relative)
            if path is None:
                print("\n%s: NOT FOUND -> %s" % (label, relative))
                continue
            blob = torch.load(path, map_location="cpu", weights_only=False)
            flow_config = dict(blob["flow_config"])
            train_config = dict(blob.get("train_config") or {})
            print("\n--- %s   step=%s   %s" % (label, blob.get("step"), path))
            for key in ("condition_features", "node_embedding_dim", "num_blocks",
                        "attention_dim", "num_heads", "hidden_dim",
                        "condition_embedding_dim", "condition_hidden_dim",
                        "node_autoregressive", "actnorm", "invertible_linear",
                        "max_log_scale", "noise_sigma", "prior_min_sigma",
                        "prior_init_sigma", "prior_hidden_dim", "whiten"):
                print("  %-26s %s" % (key, flow_config.get(key)))
            for key in ("seed", "batch_size", "max_steps", "learning_rate",
                        "warmup_steps", "grad_clip_norm"):
                print("  %-26s %s" % ("train." + key, train_config.get(key)))
            interesting = {}
            for source in (flow_config, train_config):
                for key, value in source.items():
                    if any(token in key.lower() for token in
                           ("qed", "contrast", "weight", "margin", "temp", "lambda")):
                        interesting[key] = value
            print("  QED / CONTRASTIVE FIELDS IN CHECKPOINT: %s" % (interesting or "{}"))

            model = NodeFlow(FlowConfig.from_dict(flow_config), init_seed=0)
            film = [n for n, _ in model.named_modules() if n.endswith("film")]
            affine = [n for n, _ in model.named_modules()
                      if re.fullmatch(r"blocks\.\d+\.couplings\.\d+", n)]
            autoreg = [n for n, _ in model.named_modules()
                       if n.endswith("node_coupling")]
            width = int(flow_config["node_embedding_dim"])
            half = width // 2
            print("  %-26s %s" % ("TOTAL PARAMETERS", format(model.num_parameters(), ",")))
            print("  %-26s %d  (channel split %d / %d of %d)"
                  % ("affine couplings", len(affine), half, half, width))
            print("  %-26s %d  (transforms ALL %d channels)"
                  % ("node-autoregressive", len(autoreg), width))
            print("  %-26s %d" % ("FiLM sites", len(film)))
            rows.append({"model": label, "checkpoint": str(path),
                         "parameters": model.num_parameters(),
                         "affine_couplings": len(affine),
                         "node_autoregressive": len(autoreg),
                         "film_sites": len(film), "channel_split": "%d/%d" % (half, half),
                         **{k: flow_config.get(k) for k in
                            ("condition_features", "node_embedding_dim", "num_blocks",
                             "attention_dim", "num_heads", "hidden_dim",
                             "condition_embedding_dim", "max_log_scale", "noise_sigma")},
                         "train_seed": train_config.get("seed"),
                         "train_steps": train_config.get("max_steps"),
                         "train_lr": train_config.get("learning_rate"),
                         "checkpoint_qed_fields": json.dumps(interesting, default=str)})
    except Exception as error:  # noqa: BLE001
        print("BLOCK A (flows) FAILED: %s: %s" % (type(error).__name__, error))

    try:
        from zinc_gnf.trainer_autoencoder import load_autoencoder

        path = find(args, args.ae_checkpoint)
        if path is None:
            print("\nDECODER: NOT FOUND -> %s" % args.ae_checkpoint)
        else:
            autoencoder, _ = load_autoencoder(path, device="cpu")
            config = autoencoder.config
            print("\n--- DECODER (frozen, shared by both arms)   %s" % path)
            for key in ("node_embedding_dim", "num_processing_steps", "num_heads",
                        "hidden_dim", "degree_head", "symmetry_noise_dim",
                        "num_atom_types", "num_bond_types", "weight_sharing"):
                print("  %-26s %s" % (key, getattr(config, key, "n/a")))
            total = sum(p.numel() for p in autoencoder.parameters())
            print("  %-26s %s" % ("AE PARAMETERS", format(total, ",")))
            rows.append({"model": "DECODER (frozen)", "checkpoint": str(path),
                         "parameters": total,
                         "node_embedding_dim": config.node_embedding_dim,
                         "degree_head": bool(config.degree_head)})
    except Exception as error:  # noqa: BLE001
        print("BLOCK A (decoder) FAILED: %s: %s" % (type(error).__name__, error))

    print("\n--- SEARCH FOR THE QED CONTRASTIVE OBJECTIVE ---")
    pattern = re.compile(r"contrast|qed_weight|qed_loss|margin|lambda_qed", re.IGNORECASE)
    hits = 0
    for root in roots(args):
        for folder, suffix in (("zinc_gnf", "*.py"), ("scripts", "*.py"),
                               ("configs", "*.yaml"), ("configs", "*.json")):
            for path in sorted((root / folder).glob(suffix)) if (root / folder).is_dir() else []:
                try:
                    text = path.read_text(errors="ignore")
                except OSError:
                    continue
                for number, line in enumerate(text.splitlines(), 1):
                    if pattern.search(line) and len(line.strip()) < 160:
                        print("  %s:%d: %s" % (path.name, number, line.strip()[:150]))
                        hits += 1
                        if hits >= 30:
                            print("  ... (truncated at 30 matches)")
                            return rows
    if hits == 0:
        print("  no match for contrast/qed_weight/margin in either worktree's "
              "zinc_gnf, scripts or configs")
    return rows


# ------------------------------------------------------------------ block B ---
def block_b(args, out):
    print("\n" + "=" * 72)
    print("BLOCK B :: SHOWCASE MOLECULES (rule-selected) + PNG GRIDS")
    print("=" * 72)
    chosen = []
    try:
        from rdkit import Chem, RDLogger, rdBase
        from rdkit.Chem import Draw

        RDLogger.DisableLog("rdApp.*")
        figures = out / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        made = 0
        for label, results in (("T0.20", args.results_t02), ("T0.85", args.results_t085)):
            base = find(args, results)
            if base is None:
                print("  results tree not found: %s" % results)
                continue
            for model in MODELS:
                for n in GRID_N:
                    for qed in GRID_QED:
                        rows = load_cell(base, model, n, qed)
                        if not rows:
                            continue
                        usable = [r for r in rows
                                  if (r.get("degree") or {}).get("valid")
                                  and (r.get("degree") or {}).get("qed") is not None
                                  and (r.get("degree") or {}).get("smiles")]
                        clean = [r for r in usable if not r["degree"].get("fallback")]
                        pool = clean if len(clean) >= args.per_grid else usable
                        pool = sorted(pool, key=lambda r: abs(r["degree"]["qed"] - qed))
                        pool = pool[:args.per_grid]
                        if not pool:
                            continue
                        with rdBase.BlockLogs():
                            pairs = []
                            for record in pool:
                                entry = record["degree"]
                                molecule = Chem.MolFromSmiles(entry["smiles"])
                                if molecule is None:
                                    continue
                                legend = ("QED %.2f | N %s | arom %s | rings %s"
                                          % (entry["qed"], entry.get("heavy_atoms"),
                                             entry.get("aromatic_rings"),
                                             entry.get("rings")))
                                pairs.append((molecule, legend))
                            if not pairs:
                                continue
                            image = Draw.MolsToGridImage(
                                [m for m, _ in pairs], molsPerRow=3,
                                subImgSize=(300, 300),
                                legends=[g for _, g in pairs])
                            name = ("grid_%s_%s_N%d_QED%.1f.png"
                                    % (label, model, n, qed))
                            target = figures / name
                            data = getattr(image, "data", None)
                            if data:
                                target.write_bytes(data)
                            else:
                                image.save(str(target))
                            made += 1
                        for record in pool:
                            entry = record["degree"]
                            chosen.append({
                                "temperature": label, "model": model, "N": n,
                                "target_qed": qed,
                                "achieved_qed": round(float(entry["qed"]), 4),
                                "abs_error": round(abs(entry["qed"] - qed), 4),
                                "heavy_atoms": entry.get("heavy_atoms"),
                                "aromatic_rings": entry.get("aromatic_rings"),
                                "rings": entry.get("rings"),
                                "molecular_weight": round(entry.get("molecular_weight") or 0, 1),
                                "logp": round(entry.get("logp") or 0, 2),
                                "fallback": bool(entry.get("fallback")),
                                "smiles": entry["smiles"]})
        write_csv(out / "showcase.csv", chosen)
        json.dump(chosen, (out / "showcase.json").open("w"), indent=1)
        print("  rendered %d PNG grids -> %s" % (made, figures))
        print("  SELECTION RULE: valid; prefer assembler fallback=False; then nearest")
        print("  |achieved - requested| QED; top %d per cell." % args.per_grid)
        print("\n  TOP 3 PER CELL:")
        seen = {}
        for row in chosen:
            key = (row["temperature"], row["model"], row["N"], row["target_qed"])
            seen[key] = seen.get(key, 0)
            if seen[key] < 3:
                seen[key] += 1
                print("   %s %-11s N%-2d tgt%.1f | QED %.3f err %.3f arom %s rings %-2s | %s"
                      % (row["temperature"], row["model"], row["N"], row["target_qed"],
                         row["achieved_qed"], row["abs_error"], row["aromatic_rings"],
                         row["rings"], row["smiles"]))
    except Exception as error:  # noqa: BLE001
        print("BLOCK B FAILED: %s: %s" % (type(error).__name__, error))
    return chosen


# ------------------------------------------------------------------ block C ---
def block_c(args, out):
    print("\n" + "=" * 72)
    print("BLOCK C :: CHEMISTRY BY SIZE vs SIZE-MATCHED REAL ZINC")
    print("=" * 72)
    rows = []
    properties = ("aromatic_rings", "rings", "molecular_weight", "logp", "qed")
    try:
        base = find(args, args.results_t02)
        print("  population,N,property,n,mean,sd,median,q90")
        for model in MODELS:
            for n in GRID_N:
                collected = {}
                for qed in GRID_QED:
                    for record in load_cell(base, model, n, qed) if base else []:
                        entry = record.get("degree") or {}
                        if not entry.get("valid"):
                            continue
                        for name in properties:
                            collected.setdefault(name, []).append(entry.get(name))
                for name in properties:
                    stats = summarise(collected.get(name, []))
                    if stats:
                        print("  %s,%d,%s,%d,%.3f,%.3f,%.3f,%.3f"
                              % (model, n, name, stats["n"], stats["mean"], stats["sd"],
                                 stats["median"], stats["q90"]))
                        rows.append({"population": model, "N": n, "property": name,
                                     **{k: round(v, 4) if isinstance(v, float) else v
                                        for k, v in stats.items()}})
    except Exception as error:  # noqa: BLE001
        print("BLOCK C (generated) FAILED: %s: %s" % (type(error).__name__, error))

    try:
        from rdkit import Chem, RDLogger, rdBase
        from rdkit.Chem import Crippen, Descriptors, QED

        from zinc_gnf.dataset import load_training_splits

        RDLogger.DisableLog("rdApp.*")
        data_dir = find(args, args.data_dir)
        if data_dir is None:
            print("  data dir not found: %s" % args.data_dir)
            return rows
        records, _, _ = load_training_splits(data_dir)
        reference = {}
        with rdBase.BlockLogs():
            for record in records:
                molecule = Chem.MolFromSmiles(str(record.get("smiles", "")))
                if molecule is None:
                    continue
                size = int(molecule.GetNumHeavyAtoms())
                if size not in GRID_N:
                    continue
                bucket = reference.setdefault(size, {})
                if len(bucket.get("qed", [])) >= args.reference_per_size:
                    continue
                info = molecule.GetRingInfo()
                aromatic = sum(1 for ring in info.AtomRings()
                               if all(molecule.GetAtomWithIdx(i).GetIsAromatic()
                                      for i in ring))
                bucket.setdefault("aromatic_rings", []).append(float(aromatic))
                bucket.setdefault("rings", []).append(float(info.NumRings()))
                bucket.setdefault("molecular_weight", []).append(
                    float(Descriptors.MolWt(molecule)))
                bucket.setdefault("logp", []).append(float(Crippen.MolLogP(molecule)))
                bucket.setdefault("qed", []).append(float(QED.qed(molecule)))
        for size in sorted(reference):
            for name in properties:
                stats = summarise(reference[size].get(name, []))
                if stats:
                    print("  REAL_ZINC,%d,%s,%d,%.3f,%.3f,%.3f,%.3f"
                          % (size, name, stats["n"], stats["mean"], stats["sd"],
                             stats["median"], stats["q90"]))
                    rows.append({"population": "REAL_ZINC", "N": size, "property": name,
                                 **{k: round(v, 4) if isinstance(v, float) else v
                                    for k, v in stats.items()}})
    except Exception as error:  # noqa: BLE001
        print("BLOCK C (reference) FAILED: %s: %s" % (type(error).__name__, error))
    write_csv(out / "chemistry_by_size.csv", rows)
    return rows


# ------------------------------------------------------------------ block D ---
def block_d(args, out):
    print("\n" + "=" * 72)
    print("BLOCK D :: ACHIEVED-QED SPREAD PER CELL")
    print("=" * 72)
    rows = []
    try:
        print("  T,model,N,target,attempts,valid,mean,sd,q10,median,q90,hit0.1_of_attempts")
        for label, results in (("0.20", args.results_t02), ("0.85", args.results_t085)):
            base = find(args, results)
            if base is None:
                continue
            for model in MODELS:
                for n in GRID_N:
                    for qed in GRID_QED:
                        records = load_cell(base, model, n, qed)
                        if not records:
                            continue
                        values = [float((r.get("degree") or {})["qed"]) for r in records
                                  if (r.get("degree") or {}).get("valid")
                                  and (r.get("degree") or {}).get("qed") is not None]
                        stats = summarise(values)
                        if not stats:
                            continue
                        hits = sum(1 for v in values if abs(v - qed) <= 0.1)
                        print("  %s,%s,%d,%.1f,%d,%d,%.4f,%.4f,%.3f,%.3f,%.3f,%.4f"
                              % (label, model, n, qed, len(records), stats["n"],
                                 stats["mean"], stats["sd"], stats["q10"],
                                 stats["median"], stats["q90"], hits / len(records)))
                        rows.append({"temperature": label, "model": model, "N": n,
                                     "target_qed": qed, "attempts": len(records),
                                     "valid": stats["n"],
                                     "achieved_mean": round(stats["mean"], 4),
                                     "achieved_sd": round(stats["sd"], 4),
                                     "q10": round(stats["q10"], 4),
                                     "median": round(stats["median"], 4),
                                     "q90": round(stats["q90"], 4),
                                     "hit_pm0.1_over_attempts": round(hits / len(records), 4)})
    except Exception as error:  # noqa: BLE001
        print("BLOCK D FAILED: %s: %s" % (type(error).__name__, error))
    write_csv(out / "qed_spread.csv", rows)
    return rows


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.output_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print("FATAL: %s" % error, file=sys.stderr)
        return 4
    architecture = block_a(args, out)
    write_csv(out / "architecture.csv", architecture)
    block_b(args, out)
    block_c(args, out)
    block_d(args, out)
    print("\nWROTE: %s/{architecture,showcase,chemistry_by_size,qed_spread}.csv"
          % out)
    print("WROTE: %s/figures/grid_*.png" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
