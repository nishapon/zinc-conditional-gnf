#!/usr/bin/env python3
"""Stage 11 — the FINAL unconditional report: one checkpoint at two training budgets.

Compares two already-saved generation runs of the SAME architecture (Arm B,
node-autoregressive) at the SAME sampling temperature, so the only difference is how
long the flow trained: 10,000 steps versus 50,000 steps. This is the figure that closes
the unconditional stage.

Everything is READ from files `scripts/05_generate.py` already wrote. Nothing is
trained, sampled or re-decoded. The per-decoder chemistry (QED, QED >= 0.7, rings,
aromatic rings, aromatic/all rings, validity, uniqueness, fallback, internal diversity)
is recomputed from each run's `samples.jsonl` with RDKit, so the numbers in the figure
do not depend on trusting the saved `metrics.json`; where the two disagree by more than
a rounding margin, the report says so instead of silently preferring one. Novelty is
taken from `metrics.json`, because it needs the training SMILES, which the sample file
does not contain.

Three validities are kept strictly apart and never merged:

  raw          argmax bond matrix, no chemistry applied at all - the only one that is
               purely the model's own bond prediction
  constrained  greedy valence-aware assembly with a connectivity pass
  degree       degree-constrained integer program with a greedy fallback; reported
               together with its fallback rate, because validity reached through
               fallback is a property of the assembler, not of the flow

Writes under --output-dir:
    arm_b_10k_vs_50k.png / .pdf      four panels
    arm_b_50k_molecules.png / .svg   12 valid molecules from the 50k model, with QED
    final_summary.md                 the numbers in prose, including what got WORSE
    figure_data.json                 every plotted value plus provenance

Exit codes: 0 ok; 2 missing input; 4 bad arguments; 1 unexpected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib
matplotlib.use("Agg")                                        # headless cluster node
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from zinc_gnf.generation_report import (  # noqa: E402
    DECODER_LABELS,
    DECODER_MEANING,
    DECODERS,
    REAL_ZINC_VALIDATION,
    choose_molecules,
    load_references,
    load_run,
    molecule_grid,
    rate,
    recompute_decoder,
    show,
    undefined,
)
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402

COLOURS = {"10k": "#8C8C8C", "50k": "#DD8452"}
GRID_MOLECULES = 12
HEADLINE = "degree"
# Metrics where a LOWER value is the better one.
LOWER_IS_BETTER = {"fallback_rate"}
AGREEMENT_TOLERANCE = 0.005


def analyse(run: dict[str, Any], *, diversity: bool = True) -> dict[str, Any]:
    """Per-decoder numbers recomputed from samples.jsonl, with metrics.json alongside."""
    metrics = run["metrics"]
    saved = metrics.get("decoders", {})
    headline = metrics.get("headline_decoder", HEADLINE)
    result: dict[str, Any] = {
        "label": run["label"],
        "headline_decoder": headline,
        "samples": metrics.get("samples"),
        "flow_checkpoint": metrics.get("flow_checkpoint"),
        "flow_step": metrics.get("flow_step"),
        "temperature": (metrics.get("settings") or {}).get("temperature"),
        "autoencoder_sha256": str((metrics.get("autoencoder") or {}).get("sha256", ""))[:16],
        "recomputed_from_samples": bool(run["records"]),
        "decoders": {},
        "disagreements": [],
    }
    for name in DECODERS:
        block = saved.get(name)
        entry: dict[str, Any] = {}
        if run["records"]:
            entry.update(recompute_decoder(run["records"], name, diversity=diversity))
        if block is not None:
            entry["novelty_vs_train"] = rate(block.get("novelty_vs_train"))
            entry["nuv"] = block.get("nuv")
            entry["saved_strict_validity"] = rate(block.get("strict_validity"))
            entry["saved_uniqueness"] = rate(block.get("uniqueness"))
            entry["saved_qed_mean"] = (block.get("qed") or {}).get("mean")
            if not entry:
                continue
            # The recomputation is the figure's source of truth; a disagreement with the
            # saved metrics is a fact about the data and is reported, not smoothed over.
            for key, saved_key in (("strict_validity", "saved_strict_validity"),
                                   ("uniqueness", "saved_uniqueness")):
                mine, theirs = entry.get(key), entry.get(saved_key)
                if undefined(mine) or undefined(theirs):
                    continue
                if abs(float(mine) - float(theirs)) > AGREEMENT_TOLERANCE:
                    result["disagreements"].append(
                        f"{run['label']} / {name}: recomputed {key} {float(mine):.4f} vs "
                        f"{float(theirs):.4f} in metrics.json")
        if entry:
            result["decoders"][name] = entry
        elif block is None and not run["records"]:
            continue
    if headline not in result["decoders"]:
        raise ValueError(f"{run['label']}: neither metrics.json nor samples.jsonl provide the "
                         f"'{headline}' decoder")
    return result


def head(analysis: dict[str, Any]) -> dict[str, Any]:
    return analysis["decoders"][analysis["headline_decoder"]]


def reference_block(references: dict[str, dict[str, float]]) -> tuple[str, dict[str, float]]:
    """The real-ZINC reference to draw against, and its label."""
    for label, block in sorted(references.items()):
        if "real" in label.lower():
            return label, block
    # No --reference given: fall back to the values we measured ourselves, labelled.
    return "real ZINC250K (validation)", dict(REAL_ZINC_VALIDATION)


# ---------------------------------------------------------------------------
# The figure.
# ---------------------------------------------------------------------------
def comparison_figure(analyses: list[dict[str, Any]], *, reference_label: str,
                      reference: dict[str, float], subtitle: str, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.5))
    figure.suptitle("Unconditional ZINC250K flow, Arm B (node-autoregressive): "
                    "10,000 vs 50,000 training steps\n" + subtitle, fontsize=12, y=0.995)
    labels = [a["label"] for a in analyses]
    colours = [COLOURS.get(a["label"], "#4C72B0") for a in analyses]
    width = 0.38

    def annotate(axis, bars, values, extra=None):
        for index, (bar, value) in enumerate(zip(bars, values)):
            text = show(value, "{:.3f}")
            if extra is not None:
                text += f"\n{extra[index]}"
            axis.annotate(text,
                          (bar.get_x() + bar.get_width() / 2, 0.0 if undefined(value) else value),
                          textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)

    # --- A: validity by decoder, never merged -------------------------------
    axis = axes[0][0]
    present = [d for d in DECODERS if all(d in a["decoders"] for a in analyses)]
    positions = np.arange(len(present))
    for index, analysis in enumerate(analyses):
        values = [analysis["decoders"][d].get("strict_validity", float("nan")) for d in present]
        bars = axis.bar(positions + (index - 0.5) * width, values, width,
                        label=f"{analysis['label']} steps", color=colours[index])
        annotate(axis, bars, values)
    axis.set_xticks(positions)
    axis.set_xticklabels([DECODER_LABELS[d] for d in present], fontsize=8)
    axis.set_ylabel("strict validity (sanitizable AND connected)")
    axis.set_title("A. Validity by decoder — raw is the model, the others add chemistry",
                   fontsize=10)
    axis.set_ylim(0, 1.12)
    axis.legend(fontsize=8, loc="upper left")
    axis.grid(axis="y", alpha=0.3)

    # --- B: QED -------------------------------------------------------------
    axis = axes[0][1]
    values = [head(a).get("qed_mean", float("nan")) for a in analyses]
    extra = [f"QED>=0.7: {show(head(a).get('qed_at_least_0p7'), '{:.1%}')}" for a in analyses]
    bars = axis.bar(labels, values, color=colours, width=0.5)
    annotate(axis, bars, values, extra)
    if not undefined(reference.get("qed_mean")):
        axis.axhline(reference["qed_mean"], linestyle="--", color="black", linewidth=1,
                     label=f"{reference_label}: {reference['qed_mean']:.3f} "
                           f"(QED>=0.7 {show(reference.get('qed_at_least_0p7'), '{:.1%}')})")
        axis.legend(fontsize=7, loc="upper left")
    axis.set_ylabel("mean QED of valid molecules (degree decoder)")
    axis.set_title("B. Drug-likeness", fontsize=10)
    axis.set_ylim(0, 1.0)
    axis.grid(axis="y", alpha=0.3)

    # --- C: aromaticity -----------------------------------------------------
    axis = axes[1][0]
    keys = [("aromatic_rings_mean", "aromatic rings\nper molecule"),
            ("aromatic_ring_fraction", "aromatic /\nall rings"),
            ("molecules_with_aromatic_ring", "molecules with\nan aromatic ring")]
    positions = np.arange(len(keys))
    for index, analysis in enumerate(analyses):
        values = [head(analysis).get(key, float("nan")) for key, _ in keys]
        bars = axis.bar(positions + (index - 0.5) * width, values, width,
                        label=f"{analysis['label']} steps", color=colours[index])
        annotate(axis, bars, values)
    for offset, (key, _) in enumerate(keys):
        value = reference.get(key)
        if not undefined(value):
            axis.hlines(float(value), offset - 0.5, offset + 0.5, linestyle="--",
                        color="black", linewidth=1.2)
            axis.annotate(f"real {float(value):.3f}", (offset, float(value)),
                          textcoords="offset points", xytext=(0, 4), ha="center",
                          fontsize=7.5, style="italic")
    axis.set_xticks(positions)
    axis.set_xticklabels([label for _, label in keys], fontsize=8)
    axis.set_ylabel("value (dashed = real ZINC250K)")
    axis.set_title("C. Aromaticity — the hardest target", fontsize=10)
    axis.grid(axis="y", alpha=0.3)
    axis.legend(fontsize=8, loc="upper right")

    # --- D: fallback, uniqueness, novelty, diversity -------------------------
    axis = axes[1][1]
    keys = [("fallback_rate", "degree fallback\n(lower better)"),
            ("uniqueness", "uniqueness"), ("novelty_vs_train", "novelty\nvs train"),
            ("internal_diversity", "internal\ndiversity")]
    positions = np.arange(len(keys))
    for index, analysis in enumerate(analyses):
        values = [head(analysis).get(key, float("nan")) for key, _ in keys]
        bars = axis.bar(positions + (index - 0.5) * width, values, width,
                        label=f"{analysis['label']} steps", color=colours[index])
        annotate(axis, bars, values)
    if not undefined(reference.get("degree_fallback_rate")):
        axis.hlines(float(reference["degree_fallback_rate"]), -0.5, 0.5, linestyle="--",
                    color="black", linewidth=1.2)
        axis.annotate(f"real {float(reference['degree_fallback_rate']):.3f}",
                      (0, float(reference["degree_fallback_rate"])),
                      textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7.5,
                      style="italic")
    axis.set_xticks(positions)
    axis.set_xticklabels([label for _, label in keys], fontsize=8)
    axis.set_ylabel("fraction")
    axis.set_title("D. Decoder fallback, uniqueness, novelty, diversity", fontsize=10)
    axis.set_ylim(0, 1.34)
    # Upper centre, two columns: the lower-left corner carries the real-fallback marker
    # and its label, which a legend there would cover.
    axis.legend(fontsize=8, loc="upper center", ncol=2, framealpha=0.9)
    axis.grid(axis="y", alpha=0.3)

    figure.text(0.5, 0.012,
                "All chemistry recomputed from the saved samples with RDKit. Uniqueness "
                "divides by strictly valid samples, novelty by unique valid SMILES against "
                "the TRAINING split only.\nNo validity-correction step is applied anywhere; "
                "'degree' validity includes molecules assembled by the greedy fallback, which "
                "is why panel D's fallback rate must be read with panel A.",
                ha="center", va="bottom", fontsize=7.5, style="italic")
    figure.tight_layout(rect=(0, 0.055, 1, 0.95))
    figure.savefig(output.with_suffix(".png"), dpi=200)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


# ---------------------------------------------------------------------------
# Summary.
# ---------------------------------------------------------------------------
def final_summary(analyses: list[dict[str, Any]], *, reference_label: str,
                  reference: dict[str, float], subtitle: str, chosen: list[tuple[str, float]],
                  unique_total: int) -> str:
    early, late = analyses
    head_early, head_late = head(early), head(late)

    def row(name: str, key: str, template: str = "{:.4f}", reference_key: str | None = None) -> str:
        a, b = head_early.get(key), head_late.get(key)
        delta = (float(b) - float(a)) if not (undefined(a) or undefined(b)) else None
        arrow = "—"
        if delta is not None:
            better = (delta < 0) if key in LOWER_IS_BETTER else (delta > 0)
            arrow = f"{delta:+.4f} ({'better' if better else 'worse'})" if abs(delta) > 1e-9 \
                else "±0.0000 (unchanged)"
        value = reference.get(reference_key or key)
        return (f"| {name} | {show(a, template)} | {show(b, template)} | {arrow} | "
                f"{show(value, template)} |")

    lines = [
        "# Final unconditional result — Arm B at 10,000 vs 50,000 steps",
        "",
        subtitle,
        "",
        "Same architecture, same frozen autoencoder, same dataset split, same decoders and "
        "the same sampling temperature in both columns: the only difference is the training "
        "budget. All chemistry below is recomputed from each run's `samples.jsonl` with "
        "RDKit; novelty comes from `metrics.json` because it needs the training SMILES.",
        "",
        "## Headline metrics (degree-constrained decoder)",
        "",
        f"| Metric | 10,000 steps | 50,000 steps | change | {reference_label} |",
        "|---|---:|---:|---:|---:|",
        row("mean QED", "qed_mean"),
        row("median QED", "qed_median"),
        row("fraction QED >= 0.7", "qed_at_least_0p7"),
        row("aromatic rings / molecule", "aromatic_rings_mean"),
        row("rings / molecule", "rings_mean"),
        row("aromatic / all rings", "aromatic_ring_fraction"),
        row("molecules with an aromatic ring", "molecules_with_aromatic_ring"),
        row("degree-decoder fallback", "fallback_rate", reference_key="degree_fallback_rate"),
        row("samples the integer program solved", "samples_solved_by_program"),
        row("validity among solved samples", "validity_among_solved"),
        row("uniqueness (of valid)", "uniqueness"),
        row("novelty vs training set", "novelty_vs_train"),
        row("internal diversity", "internal_diversity"),
        row("heavy atoms / molecule", "heavy_atoms_mean"),
        "",
        "## Validity: three different quantities, never merged",
        "",
        "| Decoder | 10,000 steps | 50,000 steps | change | What it measures |",
        "|---|---:|---:|---:|---|",
    ]
    for name in DECODERS:
        if not all(name in a["decoders"] for a in analyses):
            continue
        a = early["decoders"][name].get("strict_validity")
        b = late["decoders"][name].get("strict_validity")
        delta = f"{float(b) - float(a):+.4f}" if not (undefined(a) or undefined(b)) else "—"
        lines.append(f"| {name} | {show(a)} | {show(b)} | {delta} | {DECODER_MEANING[name]} |")
    lines += [
        "",
        "**Raw validity is the only number that reflects the model's own bond predictions.** "
        "The degree-constrained validity is reached with a valence-feasibility integer program "
        "and a greedy fallback, so it must be read together with the fallback rate: a high "
        "validity obtained mostly through fallback is a property of the chemistry rules, not "
        "of the flow. 'Validity among solved samples' is the honest middle ground — those are "
        "the molecules whose predicted degrees were feasible as predicted.",
        "",
    ]

    # What moved, in both directions. Nothing is dropped.
    tracked = [("mean QED", "qed_mean"), ("fraction QED >= 0.7", "qed_at_least_0p7"),
               ("aromatic rings / molecule", "aromatic_rings_mean"),
               ("aromatic / all rings", "aromatic_ring_fraction"),
               ("degree-decoder fallback", "fallback_rate"),
               ("uniqueness", "uniqueness"), ("novelty vs train", "novelty_vs_train"),
               ("internal diversity", "internal_diversity")]
    tracked += [(f"{name} validity", f"validity::{name}") for name in DECODERS
                if all(name in a["decoders"] for a in analyses)]
    improved: list[str] = []
    worsened: list[str] = []
    unchanged: list[str] = []
    for name, key in tracked:
        if key.startswith("validity::"):
            decoder = key.split("::", 1)[1]
            a = early["decoders"][decoder].get("strict_validity")
            b = late["decoders"][decoder].get("strict_validity")
        else:
            a, b = head_early.get(key), head_late.get(key)
        if undefined(a) or undefined(b):
            unchanged.append(f"- **{name}**: N/A (undefined for at least one run).")
            continue
        delta = float(b) - float(a)
        if abs(delta) <= 1e-9:
            unchanged.append(f"- **{name}**: unchanged ({float(a):.4f}).")
            continue
        better = (delta < 0) if key in LOWER_IS_BETTER else (delta > 0)
        text = f"- **{name}**: {float(a):.4f} -> {float(b):.4f} ({delta:+.4f})."
        (improved if better else worsened).append(text)
    lines += ["## What improved with 5x the training", "", *(improved or ["- Nothing."]), "",
              "## What got WORSE or did not move", "",
              *(worsened or ["- Nothing got worse."]), *unchanged, ""]

    # Gaps that remain, stated against real data.
    lines += ["## Remaining gap to real ZINC250K", ""]
    for name, key in (("mean QED", "qed_mean"), ("fraction QED >= 0.7", "qed_at_least_0p7"),
                      ("aromatic rings / molecule", "aromatic_rings_mean"),
                      ("aromatic / all rings", "aromatic_ring_fraction")):
        value, target = head_late.get(key), reference.get(key)
        if undefined(value) or undefined(target):
            continue
        lines.append(f"- **{name}**: {float(value):.4f} generated vs {float(target):.4f} real "
                     f"({float(value) - float(target):+.4f}).")
    lines += [
        "",
        "## Limitations",
        "",
        "- The 50,000-step run is the full planned budget, not a converged model proven by a "
        "flat validation curve; no early-stopping claim is made here.",
        "- Arm B is **node-order dependent** (it models ordered node tuples in the dataset's "
        "atom order) and carries **67.6% more parameters** than Arm A (2,573,696 vs "
        "1,535,360), so its advantage over Arm A mixes mechanism with capacity. This figure "
        "does not re-open that comparison; it only varies training time.",
        "- Uniqueness, novelty and diversity are computed on this sample budget only, and "
        "novelty references the TRAINING split. The held-out test split is never used.",
        "- Aromaticity remains the clearest failure, and it is NOT fixed by any decoder: the "
        "degree-constrained program enforces valences, not ring aromaticity.",
    ]
    disagreements = [d for a in analyses for d in a["disagreements"]]
    if disagreements:
        lines += ["", "## Recomputation disagreements with the saved metrics.json", ""]
        lines += [f"- {d}" for d in disagreements]
        lines.append("")
        lines.append("The figure uses the RDKit recomputation from `samples.jsonl`.")
    if chosen:
        lines += ["",
                  f"The molecule grid shows {len(chosen)} of {unique_total} unique valid "
                  f"50,000-step molecules, evenly spaced across the QED range rather than "
                  f"selected for appearance (QED {chosen[0][1]:.3f} to {chosen[-1][1]:.3f})."]
    lines += ["", "## Provenance", ""]
    for analysis in analyses:
        lines.append(f"- **{analysis['label']} steps**: `{analysis['metrics_path']}`, "
                     f"{analysis.get('samples')} samples, temperature "
                     f"{analysis.get('temperature')}, flow "
                     f"`{analysis.get('flow_checkpoint')}` step {analysis.get('flow_step')}, "
                     f"autoencoder sha256 {analysis.get('autoencoder_sha256')}, chemistry "
                     f"{'recomputed from samples.jsonl' if analysis['recomputed_from_samples'] else 'from metrics.json only'}.")
    lines.append(f"- **{reference_label}**: {reference.get('source', 'see diagnostics output')}.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--early", required=True,
                        help="metrics.json or its directory for the 10,000-step run")
    parser.add_argument("--late", required=True,
                        help="metrics.json or its directory for the 50,000-step run")
    parser.add_argument("--early-label", default="10k")
    parser.add_argument("--late-label", default="50k")
    parser.add_argument("--output-dir", default="outputs/figures/arm_b_final")
    parser.add_argument("--reference", action="append", default=[],
                        help="an earlier diagnostics/populations JSON with real reference "
                             "values (repeatable); without it the measured validation "
                             "constants are used and labelled as such")
    parser.add_argument("--select", choices=("spread", "best", "random"), default="spread")
    parser.add_argument("--grid-molecules", type=int, default=GRID_MOLECULES)
    parser.add_argument("--no-diversity", action="store_true",
                        help="skip the O(k^2) Tanimoto diversity recomputation")
    parser.add_argument("--note", default="", help="extra text for the figure subtitle")
    arguments = parser.parse_args(argv)

    try:
        output_directory = Path(arguments.output_dir)
        if arguments.grid_molecules < 1:
            raise ValueError("--grid-molecules must be >= 1")
        output_directory.mkdir(parents=True, exist_ok=True)
        runs = [load_run(Path(arguments.early), arguments.early_label),
                load_run(Path(arguments.late), arguments.late_label)]
        analyses = [analyse(run, diversity=not arguments.no_diversity) for run in runs]
        for run, analysis in zip(runs, analyses):
            analysis["metrics_path"] = run["metrics_path"]
            analysis["samples_path"] = run["samples_path"]
        references = load_references([Path(p) for p in arguments.reference])
        reference_label, reference = reference_block(references)

        temperatures = {a.get("temperature") for a in analyses}
        if len(temperatures) > 1:
            raise ValueError(f"the two runs were sampled at different temperatures "
                             f"{sorted(str(t) for t in temperatures)}; this report compares "
                             f"training budgets at ONE temperature")
        temperature = next(iter(temperatures))
        counts = " / ".join(str(a.get("samples")) for a in analyses)
        subtitle = (f"{counts} samples, temperature {temperature}, identical frozen "
                    f"autoencoder and decoders; chemistry recomputed from the saved samples")
        if arguments.note:
            subtitle += f" — {arguments.note}"

        comparison_figure(analyses, reference_label=reference_label, reference=reference,
                          subtitle=subtitle, output=output_directory / "arm_b_10k_vs_50k")

        chosen: list[tuple[str, float]] = []
        unique_total = 0
        decoder = analyses[1]["headline_decoder"]
        if runs[1]["records"]:
            chosen = choose_molecules(runs[1]["records"], decoder,
                                      count=arguments.grid_molecules, mode=arguments.select)
            unique_total = len({r[decoder]["smiles"] for r in runs[1]["records"]
                                if (r.get(decoder) or {}).get("valid")
                                and r[decoder].get("smiles")})
            if chosen:
                molecule_grid(chosen, output=output_directory / "arm_b_50k_molecules",
                              subject=f"Arm B, {arguments.late_label} steps",
                              mode=arguments.select, total_unique=unique_total)
            else:
                print("WARNING: the 50k run produced no valid molecules, so no grid was drawn",
                      file=sys.stderr, flush=True)
        else:
            print(f"WARNING: no samples.jsonl beside {runs[1]['metrics_path']}; the molecule "
                  f"grid and the recomputed chemistry are unavailable",
                  file=sys.stderr, flush=True)

        summary = final_summary(analyses, reference_label=reference_label, reference=reference,
                                subtitle=subtitle, chosen=chosen, unique_total=unique_total)
        (output_directory / "final_summary.md").write_text(summary, encoding="utf-8")
        atomic_write_json({"subtitle": subtitle, "temperature": temperature,
                           "runs": {a["label"]: a for a in analyses},
                           "reference": {"label": reference_label, **reference},
                           "selection": arguments.select,
                           "grid": [{"smiles": s, "qed": q} for s, q in chosen],
                           "grid_unique_pool": unique_total},
                          output_directory / "figure_data.json")
        print(summary, flush=True)
        print(f"wrote {output_directory}/arm_b_10k_vs_50k.png/.pdf, "
              f"arm_b_50k_molecules.png/.svg, final_summary.md, figure_data.json", flush=True)
    except FileNotFoundError as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 2
    except (ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"FATAL: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 4
    except Exception:  # noqa: BLE001
        print("FATAL: unexpected error\n" + traceback.format_exc(), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
