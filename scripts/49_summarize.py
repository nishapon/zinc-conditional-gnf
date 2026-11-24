#!/usr/bin/env python3
"""Summarise saved evaluation artefacts into CSV tables and plots. NO regeneration.

Reads only what scripts 46 and 48 already wrote. Safe to run repeatedly on a login node;
it never loads a flow, never touches a GPU, and never samples.

Produces, under `<results_dir>/summary/`:

  conditional_per_target.csv     one row per (model, target, repetition)
  conditional_across_reps.csv    one row per (model, target), mean +/- sd over repetitions
  conditional_macro_pooled.csv   macro and pooled summaries, kept separate
  response_fit.csv               achieved-vs-requested slope, intercept, standard errors
  chemistry.csv                  generated vs real ZINC, SIZE-MATCHED where possible
  diagnostic_references.csv      hypothetical constant / Gaussian references, labelled
  generation_quality.csv         scripts/46 output if present, run-level mean +/- sd
  figures/*.png                  distributions, achieved-vs-requested with identity line

Degrades gracefully: CSVs need only the standard library; plots are skipped without
matplotlib; the real-ZINC chemistry column is skipped without rdkit or the dataset.

Exit codes: 0 ok; 2 nothing to summarise; 4 bad arguments; 1 unexpected.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zinc_gnf.eval_support import (  # noqa: E402
    DEFAULT_TOLERANCES,
    constant_reference,
    read_jsonl,
    response_fit,
    run_level_summary,
)

CHEM_FIELDS = ("aromatic_rings", "rings", "molecular_weight", "logp")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/cluster.json")
    parser.add_argument("--target-set", default="grid")
    parser.add_argument("--decoder", default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-reference-chemistry", action="store_true")
    parser.add_argument("--reference-molecules", type=int, default=20000)
    return parser.parse_args(argv)


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or sorted({k for row in rows for k in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def summarise_statistics(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return {}
    values.sort()
    mean = sum(values) / len(values)
    sd = ((sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
          if len(values) > 1 else 0.0)

    def quantile(q):
        if len(values) == 1:
            return values[0]
        position = q * (len(values) - 1)
        low = int(position)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (position - low)

    return {"n": len(values), "mean": mean, "sd": sd, "q05": quantile(0.05),
            "median": quantile(0.5), "q95": quantile(0.95)}


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        config = json.loads(Path(args.config).read_text())
        paths, run = config["paths"], config.get("run", {})
        decoder = args.decoder or run.get("headline_decoder", "degree")
        tolerances = tuple(run.get("tolerances", DEFAULT_TOLERANCES))
        results = Path(paths["results_dir"])
        out = results / "summary"
        figures = out / "figures"
        out.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError, KeyError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 4

    try:
        root = results / "conditional" / args.target_set
        written = []

        # ---------------- conditional per-target and across-repetition ----------------
        per_rep, per_cell = [], []
        for metrics_file in sorted(root.glob("*/*/rep*/metrics.json")):
            per_rep.append(json.loads(metrics_file.read_text()))
        summary_file = root / "summary.json"
        if summary_file.is_file():
            per_cell = json.loads(summary_file.read_text()).get("per_cell", [])

        if per_rep:
            keys = ["model", "cell", "requested_n", "requested_qed", "repetition",
                    "sampling_seed", "temperature", "support", "extrapolation",
                    "qed_supplied_to_model", "attempted", "valid", "scored",
                    "validity_over_attempts", "achieved_qed_mean", "achieved_qed_sd",
                    "qed_mae", "qed_bias", "cell_mean_abs_error",
                    "achieved_atoms_mean", "achieved_atoms_sd",
                    "size_match_over_all_attempts", "uniqueness_over_valid",
                    "novelty_over_unique_valid", "nuv_over_attempts"]
            for tol in tolerances:
                tag = f"{tol:g}".replace(".", "p")
                keys += [f"hit_rate_within_{tag}_among_scored",
                         f"valid_hit_yield_within_{tag}_over_all_attempts"]
            written.append(write_csv(out / "conditional_per_target.csv", per_rep, keys))

        if per_cell:
            rows = []
            for entry in per_cell:
                row = {"model": entry["model"], "cell": entry["cell"], "n": entry["n"],
                       "qed": entry["qed"], "repetitions": entry["repetitions"],
                       "support": entry.get("support"),
                       "extrapolation": entry.get("extrapolation")}
                for field, block in entry.get("across_repetitions", {}).items():
                    row[f"{field}_mean"] = block["mean"]
                    row[f"{field}_sd"] = block["sd"]
                rows.append(row)
            written.append(write_csv(out / "conditional_across_reps.csv", rows))

            macro_rows = []
            data = json.loads(summary_file.read_text())
            for model, block in data.get("by_model", {}).items():
                for name in ("macro_qed_mae_across_repetitions",
                             "pooled_qed_mae_across_repetitions"):
                    stats = block.get(name) or {}
                    macro_rows.append({
                        "model": model,
                        "weighting": "macro (equal per cell)" if "macro" in name
                        else "pooled (weighted by scored molecules)",
                        "qed_mae_mean": stats.get("mean"),
                        "qed_mae_sd_across_sampling_repetitions": stats.get("sd"),
                        "repetitions": stats.get("runs"),
                        "variability_kind": "SAMPLING repetitions of one checkpoint; "
                                            "NOT independently trained seeds"})
            written.append(write_csv(out / "conditional_macro_pooled.csv", macro_rows))

        # ---------------- response fit: achieved vs requested ----------------
        fits = []
        by_model_points: dict[str, list[tuple[float, float]]] = {}
        achieved_by_cell: dict[tuple[str, int, float], list[float]] = {}
        for samples_file in sorted(root.glob("*/*/rep*/samples.jsonl")):
            for record in read_jsonl(samples_file):
                entry = record.get(decoder) or {}
                if not entry.get("valid") or entry.get("qed") is None:
                    continue
                model = record["model"]
                requested = float(record["requested_qed"])
                by_model_points.setdefault(model, []).append(
                    (requested, float(entry["qed"])))
                achieved_by_cell.setdefault(
                    (model, int(record["requested_n"]), requested), []
                ).append(float(entry["qed"]))
        for model, points in by_model_points.items():
            per_sample = response_fit([p[0] for p in points], [p[1] for p in points])
            fits.append({"model": model, "level": "per valid sample", **per_sample})
            means = {}
            for (m, n, requested), values in achieved_by_cell.items():
                if m == model:
                    means.setdefault(requested, []).extend(values)
            requested_values = sorted(means)
            fits.append({"model": model, "level": "per-target mean (hides within-target "
                                                  "spread; reported for comparability)",
                         **response_fit(requested_values,
                                        [sum(means[r]) / len(means[r])
                                         for r in requested_values])})
            # within-N slopes: the ZINC N-QED association cannot explain these
            for n_value in sorted({k[1] for k in achieved_by_cell if k[0] == model}):
                subset = [(k[2], v) for k, values in achieved_by_cell.items()
                          if k[0] == model and k[1] == n_value for v in values]
                if subset:
                    fits.append({"model": model, "level": f"within N={n_value}",
                                 **response_fit([s[0] for s in subset],
                                                [s[1] for s in subset])})
        if fits:
            written.append(write_csv(out / "response_fit.csv", fits))

        # ---------------- diagnostic references ----------------
        requested_set = sorted({k[2] for k in achieved_by_cell}) or [0.4, 0.6, 0.8]
        observed_sd = summarise_statistics(
            [v for values in achieved_by_cell.values() for v in values]).get("sd", 0.0)
        reference_rows = []
        for sigma in sorted({0.0, 0.05, 0.1, 0.15, 0.2, round(observed_sd, 3)}):
            block = constant_reference(requested_set, sigma=sigma)
            reference_rows.append({
                "kind": block["kind"], "sigma": sigma,
                "optimal_constant": block["optimal_constant"],
                "mae_at_optimal_constant": block["mae_at_optimal_constant"],
                "weighted_median_of_targets": block["weighted_median_of_targets"],
                "mae_at_weighted_median": block["mae_at_weighted_median"],
                "weighted_mean_of_targets": block["weighted_mean_of_targets"],
                "mae_at_weighted_mean": block["mae_at_weighted_mean"],
                "note": block["note"]})
        written.append(write_csv(out / "diagnostic_references.csv", reference_rows))

        # ---------------- chemistry, size-matched ----------------
        generated: dict[tuple[str, int], dict[str, list[float]]] = {}
        for samples_file in sorted(root.glob("*/*/rep*/samples.jsonl")):
            for record in read_jsonl(samples_file):
                entry = record.get(decoder) or {}
                if not entry.get("valid"):
                    continue
                size = entry.get("heavy_atoms")
                if size is None:
                    continue
                bucket = generated.setdefault((record["model"], int(size)), {})
                for field in CHEM_FIELDS:
                    if entry.get(field) is not None:
                        bucket.setdefault(field, []).append(float(entry[field]))

        reference: dict[int, dict[str, list[float]]] = {}
        reference_note = "not computed"
        if not args.no_reference_chemistry:
            try:
                from rdkit import Chem, RDLogger, rdBase
                from rdkit.Chem import Crippen, Descriptors

                from zinc_gnf.dataset import load_training_splits

                RDLogger.DisableLog("rdApp.*")
                train_records, _, _ = load_training_splits(paths["data_dir"])
                wanted = {size for _, size in generated}
                with rdBase.BlockLogs():
                    seen = 0
                    for record in train_records:
                        if seen >= args.reference_molecules:
                            break
                        molecule = Chem.MolFromSmiles(str(record.get("smiles", "")))
                        if molecule is None:
                            continue
                        size = int(molecule.GetNumHeavyAtoms())
                        if wanted and size not in wanted:
                            continue
                        seen += 1
                        rings = molecule.GetRingInfo()
                        bucket = reference.setdefault(size, {})
                        bucket.setdefault("aromatic_rings", []).append(float(sum(
                            1 for ring in rings.AtomRings()
                            if all(molecule.GetAtomWithIdx(i).GetIsAromatic()
                                   for i in ring))))
                        bucket.setdefault("rings", []).append(float(rings.NumRings()))
                        bucket.setdefault("molecular_weight", []).append(
                            float(Descriptors.MolWt(molecule)))
                        bucket.setdefault("logp", []).append(
                            float(Crippen.MolLogP(molecule)))
                reference_note = (f"real ZINC training molecules, SIZE-MATCHED on heavy "
                                  f"atom count ({seen} scanned)")
            except Exception as error:  # noqa: BLE001
                reference_note = f"skipped: {type(error).__name__}: {error}"

        chemistry_rows = []
        for (model, size), fields in sorted(generated.items()):
            for field, values in sorted(fields.items()):
                row = {"model": model, "heavy_atoms": size, "property": field,
                       "reference_note": reference_note}
                row.update({f"generated_{k}": v
                            for k, v in summarise_statistics(values).items()})
                real = (reference.get(size) or {}).get(field)
                if real:
                    row.update({f"real_zinc_{k}": v
                                for k, v in summarise_statistics(real).items()})
                    row["difference_of_means"] = (
                        row["generated_mean"] - row["real_zinc_mean"])
                chemistry_rows.append(row)
        if chemistry_rows:
            written.append(write_csv(out / "chemistry.csv", chemistry_rows))

        # ---------------- generation-quality comparison from script 46 ----------------
        quality = results / "uncond_vs_cond" / "uncond_vs_cond.json"
        if quality.is_file():
            payload = json.loads(quality.read_text())
            rows = []
            for arm, block in (payload.get("arms") or {}).items():
                for metric, stats in (block.get("summary") or {}).items():
                    if not isinstance(stats, dict) or "mean" not in stats:
                        continue
                    rows.append({"arm": arm, "metric": metric, "mean": stats["mean"],
                                 "sd_across_sampling_repetitions": stats["sd"],
                                 "repetitions": stats.get("runs"),
                                 "temperature": (payload.get("shared_settings") or {})
                                 .get("temperature"),
                                 "attempts_per_run": (payload.get("shared_settings") or {})
                                 .get("count_per_run"),
                                 "variability_kind": "SAMPLING repetitions, not seeds"})
            if rows:
                written.append(write_csv(out / "generation_quality.csv", rows))

        # ---------------- figures ----------------
        if not args.no_plots and achieved_by_cell:
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                figures.mkdir(parents=True, exist_ok=True)
                models = sorted({k[0] for k in achieved_by_cell})
                # 1. achieved QED distributions per requested target
                for model in models:
                    targets = sorted({k[2] for k in achieved_by_cell if k[0] == model})
                    if not targets:
                        continue
                    figure, axis = plt.subplots(figsize=(8, 4.5))
                    data = [[v for k, values in achieved_by_cell.items()
                             if k[0] == model and k[2] == t for v in values]
                            for t in targets]
                    axis.violinplot([d for d in data if d], showmeans=True)
                    axis.set_xticks(range(1, len([d for d in data if d]) + 1))
                    axis.set_xticklabels([f"{t:g}" for t, d in zip(targets, data) if d])
                    axis.set_xlabel("requested QED (scalar target)")
                    axis.set_ylabel("achieved QED of valid molecules")
                    axis.set_title(f"{model}: achieved QED by request "
                                   f"(duplicates preserved)")
                    figure.tight_layout()
                    figure.savefig(figures / f"achieved_qed_distribution_{model}.png",
                                   dpi=150)
                    plt.close(figure)
                # 2. achieved mean vs requested, with identity line
                figure, axis = plt.subplots(figsize=(6, 6))
                low = min(k[2] for k in achieved_by_cell)
                high = max(k[2] for k in achieved_by_cell)
                axis.plot([low, high], [low, high], "k--", linewidth=1,
                          label="identity (perfect control)")
                for model in models:
                    targets = sorted({k[2] for k in achieved_by_cell if k[0] == model})
                    means, sds, used = [], [], []
                    for t in targets:
                        values = [v for k, vs in achieved_by_cell.items()
                                  if k[0] == model and k[2] == t for v in vs]
                        if not values:
                            continue
                        stats = summarise_statistics(values)
                        used.append(t)
                        means.append(stats["mean"])
                        sds.append(stats["sd"])
                    if used:
                        axis.errorbar(used, means, yerr=sds, marker="o", capsize=3,
                                      label=model)
                axis.set_xlabel("requested QED")
                axis.set_ylabel("achieved QED (mean, error bars = sd of samples)")
                axis.set_title("Achieved vs requested QED")
                axis.legend()
                figure.tight_layout()
                figure.savefig(figures / "achieved_vs_requested_qed.png", dpi=150)
                plt.close(figure)
                written.append(figures)
            except Exception as error:  # noqa: BLE001
                print(f"plots skipped: {type(error).__name__}: {error}", file=sys.stderr)

        written = [w for w in written if w]
        if not written:
            print(f"FATAL: nothing to summarise under {root}. Run scripts/48 first.",
                  file=sys.stderr)
            return 2
        print("wrote:")
        for item in written:
            print(f"  {item}")
        print("\nReminders encoded in the CSVs: every sd over repetitions is SAMPLING "
              "variability from one checkpoint; diagnostic_references.csv rows are "
              "hypothetical and are neither measured generators nor MAE lower bounds.")
        return 0
    except Exception:  # noqa: BLE001
        print("FATAL: unexpected\n" + traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
