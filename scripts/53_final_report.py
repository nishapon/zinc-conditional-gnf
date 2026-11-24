#!/usr/bin/env python3
"""Final report from SAVED artefacts only. No generation, no flow loading, no GPU.

Reads whatever scripts 46, 48, 51 and 52 have written and emits:

  report.md                 compact tables separated by N, plus supported claims and
                            limitations
  comparison_by_n.csv       one row per (N, QED, model): achieved QED, MAE, yields
  uncertainty.csv           the three different spreads, kept apart on purpose
  figures/by_n_*.png        achieved vs requested QED, one panel per N

THREE KINDS OF SPREAD, never merged:
  sample spread        sd of achieved QED across molecules within one cell/repetition.
                       Describes the model's output distribution.
  sampling-run SD      sd of a cell statistic across sampling repetitions of ONE
                       checkpoint. Describes Monte-Carlo noise in the measurement.
  training-seed spread UNAVAILABLE. One checkpoint per arm was trained, so no statement
                       about robustness to retraining is supportable.

Exit codes: 0 ok; 2 nothing to report; 4 bad arguments; 1 unexpected.
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
    ARM_LABELS,
    STRUCTURAL_SIZE_NOTE,
    read_jsonl,
    response_fit,
    run_level_summary,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/cluster.json")
    parser.add_argument("--target-sets", nargs="*", default=["grid", "percentile"])
    parser.add_argument("--decoder", default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def write_csv(path, rows):
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    names = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def spread(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    sd = ((sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
          if len(values) > 1 else 0.0)
    return mean, sd


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(out)


def fmt(value, digits=4):
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        config = json.loads(Path(args.config).read_text())
        paths, run = config["paths"], config.get("run", {})
        decoder = args.decoder or run.get("headline_decoder", "degree")
        results = Path(paths["results_dir"])
        out = results / "report"
        out.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError, KeyError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 4

    try:
        comparison_rows, uncertainty_rows = [], []
        per_cell_samples: dict[tuple, list] = {}
        found_any = False

        for target_set in args.target_sets:
            root = results / "conditional" / target_set
            if not root.is_dir():
                continue
            found_any = True
            for samples_file in sorted(root.glob("*/*/rep*/samples.jsonl")):
                model = samples_file.parents[2].name
                records = read_jsonl(samples_file)
                if not records:
                    continue
                key = (target_set, model, int(records[0]["requested_n"]),
                       float(records[0]["requested_qed"]))
                per_cell_samples.setdefault(key, []).append(
                    {"repetition": int(records[0].get("repetition", -1)),
                     "achieved": [float((r.get(decoder) or {}).get("qed"))
                                  for r in records
                                  if (r.get(decoder) or {}).get("valid")
                                  and (r.get(decoder) or {}).get("qed") is not None],
                     "attempted": len(records)})

        for (target_set, model, n_value, qed_value), reps in sorted(
                per_cell_samples.items()):
            pooled = [v for rep in reps for v in rep["achieved"]]
            attempted = sum(rep["attempted"] for rep in reps)
            achieved_mean, sample_sd = spread(pooled)
            per_rep_mae, per_rep_mean = [], []
            for rep in reps:
                if rep["achieved"]:
                    per_rep_mae.append(sum(abs(v - qed_value) for v in rep["achieved"])
                                       / len(rep["achieved"]))
                    per_rep_mean.append(sum(rep["achieved"]) / len(rep["achieved"]))
            mae_mean, mae_run_sd = spread(per_rep_mae)
            _, mean_run_sd = spread(per_rep_mean)
            hits10 = sum(1 for v in pooled if abs(v - qed_value) <= 0.1)
            hits01 = sum(1 for v in pooled if abs(v - qed_value) <= 0.01)
            comparison_rows.append({
                "target_set": target_set, "model": model,
                "arm_label": ARM_LABELS.get(model, model),
                "N": n_value, "requested_qed": qed_value,
                "attempts_total": attempted, "valid_scored_total": len(pooled),
                "achieved_qed_mean": achieved_mean,
                "achieved_qed_sample_sd": sample_sd,
                "qed_mae": mae_mean,
                "qed_mae_sampling_run_sd": mae_run_sd,
                "achieved_mean_sampling_run_sd": mean_run_sd,
                "qed_bias": (achieved_mean - qed_value) if achieved_mean else None,
                "hit_rate_pm0.1_among_scored": hits10 / len(pooled) if pooled else None,
                "valid_hit_yield_pm0.1_over_attempts": hits10 / attempted if attempted else None,
                "valid_hit_yield_pm0.01_over_attempts": hits01 / attempted if attempted else None,
                "repetitions": len(reps)})
            uncertainty_rows.append({
                "target_set": target_set, "model": model, "N": n_value,
                "requested_qed": qed_value,
                "sample_spread_sd_of_achieved_qed": sample_sd,
                "sampling_run_sd_of_cell_mae": mae_run_sd,
                "sampling_run_sd_of_cell_mean": mean_run_sd,
                "training_seed_spread": None,
                "training_seed_note": ("UNAVAILABLE: one trained checkpoint per arm, so "
                                       "no retraining robustness can be claimed")})

        if not found_any:
            print(f"FATAL: no conditional artefacts under {results / 'conditional'}",
                  file=sys.stderr)
            return 2

        write_csv(out / "comparison_by_n.csv", comparison_rows)
        write_csv(out / "uncertainty.csv", uncertainty_rows)

        # ---- within-N slopes per model, with sampling-run spread ----
        slope_rows = []
        for target_set in args.target_sets:
            for model in sorted({k[1] for k in per_cell_samples if k[0] == target_set}):
                for n_value in sorted({k[2] for k in per_cell_samples
                                       if k[0] == target_set and k[1] == model}):
                    per_rep_slopes = []
                    reps_count = max(
                        (len(v) for k, v in per_cell_samples.items()
                         if k[:3] == (target_set, model, n_value)), default=0)
                    for index in range(reps_count):
                        pairs = []
                        for key, reps in per_cell_samples.items():
                            if key[:3] != (target_set, model, n_value):
                                continue
                            if index < len(reps):
                                pairs += [(key[3], v) for v in reps[index]["achieved"]]
                        if pairs:
                            fit = response_fit([p[0] for p in pairs],
                                               [p[1] for p in pairs])
                            if fit.get("slope") is not None:
                                per_rep_slopes.append(fit["slope"])
                    if per_rep_slopes:
                        summary = run_level_summary(per_rep_slopes)
                        slope_rows.append({
                            "target_set": target_set, "model": model, "N": n_value,
                            "slope_mean_over_repetitions": summary["mean"],
                            "slope_sampling_run_sd": summary["sd"],
                            "repetitions": summary["runs"]})
        write_csv(out / "within_n_slopes.csv", slope_rows)

        # ---- optional extras ----
        numerical = results / "numerical" / "numerical_check.json"
        audit = results / "audit" / "grid" / "audit.json"
        quality = results / "uncond_vs_cond" / "uncond_vs_cond.json"

        lines = ["# QED conditioning: final report from saved artefacts", "",
                 f"Comparison: **{ARM_LABELS['comparison']}**", "",
                 f"Headline decoder: `{decoder}`. "
                 f"Every number below is recomputed from saved per-attempt records.", "",
                 "## Achieved QED by requested size and target", ""]
        headers = ["set", "model", "N", "req QED", "attempts", "valid",
                   "achieved mean", "sample sd", "MAE", "MAE run-sd",
                   "yield +/-0.1", "yield +/-0.01"]
        rows = [[r["target_set"], r["model"], r["N"], fmt(r["requested_qed"], 2),
                 r["attempts_total"], r["valid_scored_total"],
                 fmt(r["achieved_qed_mean"]), fmt(r["achieved_qed_sample_sd"]),
                 fmt(r["qed_mae"]), fmt(r["qed_mae_sampling_run_sd"]),
                 fmt(r["valid_hit_yield_pm0.1_over_attempts"]),
                 fmt(r["valid_hit_yield_pm0.01_over_attempts"])]
                for r in sorted(comparison_rows,
                                key=lambda r: (r["target_set"], r["N"],
                                               r["requested_qed"], r["model"]))]
        lines += [table(headers, rows), "",
                  "## Within-size QED response", "",
                  table(["set", "model", "N", "slope (mean over reps)", "run sd", "reps"],
                        [[r["target_set"], r["model"], r["N"],
                          fmt(r["slope_mean_over_repetitions"]),
                          fmt(r["slope_sampling_run_sd"]), r["repetitions"]]
                         for r in slope_rows]), "",
                  "A slope near 1 would mean the achieved QED tracks the request. A slope "
                  "near 0 means the request is being ignored. The run sd says whether a "
                  "slope is stable across sampling repetitions or an artefact of one "
                  "sample.", "",
                  "## Three kinds of spread", "",
                  table(["kind", "what it describes", "available here"],
                        [["sample spread", "sd of achieved QED across molecules in a cell",
                          "yes"],
                         ["sampling-run SD",
                          "sd of a cell statistic across repetitions of ONE checkpoint",
                          "yes"],
                         ["training-seed spread",
                          "sd across independently trained models",
                          "**NO** - one checkpoint per arm"]]), "",
                  "## Supported claims", "",
                  "1. Both arms receive the atom count structurally. " +
                  STRUCTURAL_SIZE_NOTE, "",
                  "2. Any difference between the arms is QED conditioning **plus** the "
                  "QED contrastive objective the conditional checkpoint was trained with. "
                  "It is not a FiLM-only contrast.", "",
                  "3. Per-attempt yields include failed attempts in the denominator, so "
                  "they are comparable across arms even where validity differs.", "",
                  "## Limitations", "",
                  "- One trained checkpoint per arm: **no** statement about robustness to "
                  "retraining is supportable from these artefacts.",
                  "- Repetitions are sampling repetitions, not training seeds.",
                  "- Uniqueness and novelty are the only deduplicated quantities; all "
                  "accuracy numbers keep duplicates.",
                  "- A shuffled-(N,QED) likelihood gap permutes both conditions and so is "
                  "not evidence about QED alone.", ""]

        if numerical.is_file():
            payload = json.loads(numerical.read_text())
            lines += ["## Numerical check", "",
                      table(["model", "real round-trip abs max",
                             "latent round-trip abs max", "logdet/node",
                             "decoder identical"],
                            [[name,
                              fmt(((block.get("real_molecules_round_trip") or {})
                                   .get("absolute") or {}).get("max"), 3),
                              fmt(((block.get("generated_latents_round_trip") or {})
                                   .get("absolute") or {}).get("max"), 3),
                              fmt((block.get("log_det_per_node") or {}).get("mean"), 2),
                              fmt((block.get("decoder_determinism") or {})
                                  .get("identical_smiles_fraction"), 3)]
                             for name, block in (payload.get("models") or {}).items()
                             if isinstance(block, dict)]), "",
                      "A large log-determinant is a warning sign, not proof of inversion "
                      "failure; the round-trip errors are the evidence.", ""]
        else:
            lines += ["## Numerical check", "", "PENDING: run scripts/52.", ""]

        if audit.is_file():
            payload = json.loads(audit.read_text())
            lines += ["## Artefact audit", "",
                      f"- incomplete cells: {payload.get('incomplete_cells')}",
                      f"- denominator mismatches: "
                      f"{payload.get('denominator_mismatches')}",
                      f"- one decoder across arms: "
                      f"{(payload.get('provenance_checks') or {}).get('one_decoder_across_arms')}",
                      f"- one architecture across arms: "
                      f"{(payload.get('provenance_checks') or {}).get('one_architecture_across_arms')}",
                      f"- old slope~0.7: "
                      f"{(payload.get('old_slope_0p7') or {}).get('verdict')}", "",
                      "Ring definitions used: see `audit.json -> ring_note`.", ""]
        else:
            lines += ["## Artefact audit", "", "PENDING: run scripts/51.", ""]

        if quality.is_file():
            payload = json.loads(quality.read_text())
            settings = payload.get("shared_settings") or {}
            lines += ["## MoFlow-style generation comparison", "",
                      f"{settings.get('count_per_run')} attempts x "
                      f"{settings.get('runs')} sampling repetitions at temperature "
                      f"{settings.get('temperature')}.", "",
                      table(["arm", "metric", "mean", "run sd"],
                            [[arm, metric, fmt(stats.get("mean")), fmt(stats.get("sd"))]
                             for arm, block in (payload.get("arms") or {}).items()
                             for metric, stats in (block.get("summary") or {}).items()
                             if isinstance(stats, dict) and "mean" in stats]), "",
                      "MoFlow-STYLE sampling and reporting, not an exact reproduction. "
                      "Remaining differences are listed in the artefact's "
                      "`moflow_protocol_differences` field.", ""]
        else:
            lines += ["## MoFlow-style generation comparison", "",
                      "PENDING: run scripts/46.", ""]

        (out / "report.md").write_text("\n".join(lines))

        if not args.no_plots and per_cell_samples:
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                figures = out / "figures"
                figures.mkdir(parents=True, exist_ok=True)
                for target_set in args.target_sets:
                    sizes = sorted({k[2] for k in per_cell_samples if k[0] == target_set})
                    if not sizes:
                        continue
                    figure, axes = plt.subplots(1, len(sizes),
                                                figsize=(4.2 * len(sizes), 4.2),
                                                squeeze=False)
                    for column, n_value in enumerate(sizes):
                        axis = axes[0][column]
                        targets = sorted({k[3] for k in per_cell_samples
                                          if k[0] == target_set and k[2] == n_value})
                        if targets:
                            axis.plot([min(targets), max(targets)],
                                      [min(targets), max(targets)], "k--",
                                      linewidth=1, label="identity")
                        for model in sorted({k[1] for k in per_cell_samples
                                             if k[0] == target_set}):
                            xs, ys, es = [], [], []
                            for target in targets:
                                reps = per_cell_samples.get(
                                    (target_set, model, n_value, target))
                                if not reps:
                                    continue
                                pooled = [v for rep in reps for v in rep["achieved"]]
                                if not pooled:
                                    continue
                                mean, sd = spread(pooled)
                                xs.append(target)
                                ys.append(mean)
                                es.append(sd)
                            if xs:
                                axis.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                                              label=model)
                        axis.set_title(f"N = {n_value}")
                        axis.set_xlabel("requested QED")
                        if column == 0:
                            axis.set_ylabel("achieved QED (error bars = sample sd)")
                        axis.legend(fontsize=8)
                    figure.suptitle(f"Achieved vs requested QED by size ({target_set})")
                    figure.tight_layout()
                    figure.savefig(figures / f"by_n_{target_set}.png", dpi=150)
                    plt.close(figure)
            except Exception as error:  # noqa: BLE001
                print(f"plots skipped: {type(error).__name__}: {error}", file=sys.stderr)

        print(f"wrote {out / 'report.md'}, comparison_by_n.csv, uncertainty.csv, "
              f"within_n_slopes.csv")
        return 0
    except Exception:  # noqa: BLE001
        print("FATAL: unexpected\n" + traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
