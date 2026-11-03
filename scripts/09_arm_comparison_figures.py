#!/usr/bin/env python3
"""Stage 9 — presentation figures and a results summary from SAVED generation outputs.

Reads what `scripts/05_generate.py` already wrote for Arm A and Arm B (a `metrics.json`
and, if present, a `samples.jsonl`) and produces:

  arm_comparison.png / .pdf   four panels: validity by decoder, mean QED, degree-decoder
                              fallback, diversity (internal diversity, uniqueness, novelty)
  arm_b_molecules.png / .svg  a 4 x 3 grid of valid Arm B molecules with their QED values
  results_summary.md          the numbers written out in prose, generated from the data
  figure_data.json            every plotted value plus provenance, so a figure can be
                              traced back to the run that produced it

Nothing is trained, sampled or recomputed: this job only reads saved files. It writes
only inside --output-dir.

Scientific accuracy is the point of the validity panel. Three validities are kept
separate and never merged:

  raw          argmax bond matrix, no chemistry applied at all
  constrained  greedy valence-aware assembly with a connectivity pass
  degree       degree-constrained integer program, with a greedy fallback when the
               predicted degrees are infeasible

Because the degree decoder falls back to the greedy one, its validity is reported
together with the fallback rate and, when `samples.jsonl` is available, with validity
restricted to the samples the integer program actually solved. A high validity obtained
mostly through fallback is a property of the chemistry rules, not of the model.

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
    choose_molecules,
    decoder_numbers,
    load_references,
    load_run,
    molecule_grid,
    number,
    rate,
    resolve,
    show,
    undefined,
)
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402

ARM_COLOURS = {"A": "#4C72B0", "B": "#DD8452"}
GRID_MOLECULES = 12


def load_arm(path: Path, label: str) -> dict[str, Any]:
    """Backwards-compatible name for `load_run` (one arm of the comparison)."""
    return load_run(path, label)


# ---------------------------------------------------------------------------
# Figure 1: the comparison.
# ---------------------------------------------------------------------------
def comparison_figure(arms: list[dict[str, Any]], numbers: list[dict[str, Any]],
                      references: dict[str, dict[str, float]], *, subtitle: str,
                      output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    figure.suptitle("Unconditional ZINC250K flow: Arm A (baseline) vs Arm B "
                    "(node-autoregressive)\n" + subtitle, fontsize=12, y=0.995)

    # --- A: validity, one group per decoder, never merged -------------------
    axis = axes[0][0]
    present = [d for d in DECODERS if all(d in n for n in numbers)]
    positions = np.arange(len(present))
    width = 0.38
    for index, (arm, number) in enumerate(zip(arms, numbers)):
        values = [number[d]["strict_validity"] for d in present]
        offset = (index - 0.5) * width
        bars = axis.bar(positions + offset, values, width, label=f"Arm {arm['label']}",
                        color=ARM_COLOURS[arm["label"]])
        for bar, value in zip(bars, values):
            axis.annotate(show(value, "{:.3f}"),
                          (bar.get_x() + bar.get_width() / 2, 0.0 if undefined(value) else value),
                          textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    axis.set_xticks(positions)
    axis.set_xticklabels([DECODER_LABELS[d] for d in present], fontsize=8)
    axis.set_ylabel("strict validity (sanitizable AND connected)")
    axis.set_title("A. Validity by decoder — these are different quantities", fontsize=10)
    axis.set_ylim(0, 1.05)
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.3)

    # --- B: QED -------------------------------------------------------------
    axis = axes[0][1]
    labels = [f"Arm {a['label']}" for a in arms]
    values = [n[n["headline_decoder"]]["qed_mean"] for n in numbers]
    bars = axis.bar(labels, values, color=[ARM_COLOURS[a["label"]] for a in arms], width=0.5)
    for bar, value, entry in zip(bars, values, numbers):
        fraction = entry[entry["headline_decoder"]]["qed_at_least_0p7"]
        axis.annotate(f"{show(value, '{:.3f}')}\nQED>=0.7: {show(fraction, '{:.1%}')}",
                      (bar.get_x() + bar.get_width() / 2, 0.0 if undefined(value) else value),
                      textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    drawn_reference = False
    for style, (label, reference) in zip(("--", ":"), sorted(references.items())):
        if undefined(reference.get("qed_mean")):
            continue
        axis.axhline(reference["qed_mean"], linestyle=style, color="black", linewidth=1,
                     label=f"{label}: {reference['qed_mean']:.3f} "
                           f"(QED>=0.7 {show(reference.get('qed_at_least_0p7'), '{:.1%}')})")
        drawn_reference = True
    axis.set_ylabel("mean QED of valid molecules")
    axis.set_title("B. Drug-likeness (headline decoder)", fontsize=10)
    axis.set_ylim(0, 1.0)
    if drawn_reference:
        axis.legend(fontsize=7, loc="upper left")
    axis.grid(axis="y", alpha=0.3)

    # --- C: fallback --------------------------------------------------------
    axis = axes[1][0]
    values = [n[n["headline_decoder"]].get("fallback_rate", float("nan")) for n in numbers]
    bars = axis.bar(labels, values, color=[ARM_COLOURS[a["label"]] for a in arms], width=0.5)
    for bar, value, entry in zip(bars, values, numbers):
        head = entry[entry["headline_decoder"]]
        extra = ""
        if not undefined(head.get("validity_among_solved")):
            extra = f"\nvalidity when solved: {head['validity_among_solved']:.3f}"
        axis.annotate(f"{show(value, '{:.3f}')}{extra}",
                      (bar.get_x() + bar.get_width() / 2, 0.0 if undefined(value) else value),
                      textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    for style, (label, reference) in zip(("--", ":"), sorted(references.items())):
        if not undefined(reference.get("degree_fallback_rate")):
            axis.axhline(reference["degree_fallback_rate"], linestyle=style, color="black",
                         linewidth=1, label=f"{label}: {reference['degree_fallback_rate']:.3f}")
            axis.legend(fontsize=7, loc="upper right")
    axis.set_ylabel("fraction of samples falling back to greedy")
    axis.set_title("C. Degree-constrained program infeasible (lower is better)", fontsize=10)
    axis.set_ylim(0, 1.05)
    axis.grid(axis="y", alpha=0.3)

    # --- D: diversity -------------------------------------------------------
    axis = axes[1][1]
    keys = [("internal_diversity", "internal\ndiversity"), ("uniqueness", "uniqueness"),
            ("novelty_vs_train", "novelty\nvs train")]
    positions = np.arange(len(keys))
    for index, (arm, number) in enumerate(zip(arms, numbers)):
        head = number[number["headline_decoder"]]
        values = [head.get(key, float("nan")) for key, _ in keys]
        offset = (index - 0.5) * width
        bars = axis.bar(positions + offset, values, width, label=f"Arm {arm['label']}",
                        color=ARM_COLOURS[arm["label"]])
        for bar, value in zip(bars, values):
            axis.annotate(show(value, "{:.3f}"),
                          (bar.get_x() + bar.get_width() / 2, 0.0 if undefined(value) else value),
                          textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    axis.set_xticks(positions)
    axis.set_xticklabels([label for _, label in keys], fontsize=8)
    axis.set_ylabel("fraction")
    axis.set_title("D. Diversity, uniqueness and novelty (of valid molecules)", fontsize=10)
    axis.set_ylim(0, 1.18)                                   # headroom for the value labels
    axis.legend(fontsize=8, loc="lower left", framealpha=0.9)
    axis.grid(axis="y", alpha=0.3)

    figure.text(0.5, 0.005,
                "Uniqueness divides by strictly valid samples; novelty divides by unique "
                "strictly-valid SMILES and references the TRAINING split only. "
                "No validity correction step is applied anywhere.",
                ha="center", fontsize=7.5, style="italic")
    figure.tight_layout(rect=(0, 0.025, 1, 0.955))
    figure.savefig(output.with_suffix(".png"), dpi=200)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


# Figure 2 (the molecule grid) and the file readers live in
# `zinc_gnf/generation_report.py`, shared with stage 11.


# ---------------------------------------------------------------------------
# Summary.
# ---------------------------------------------------------------------------
def results_summary(arms, numbers, references, *, subtitle: str, chosen_count: int,
                    unique_count: int) -> str:
    head_a, head_b = (n[n["headline_decoder"]] for n in numbers)
    labels = [a["label"] for a in arms]

    def row(name: str, key: str, template: str = "{:.4f}") -> str:
        values = [show(entry[entry["headline_decoder"]].get(key), template) for entry in numbers]
        reference = ""
        for label, block in sorted(references.items()):
            if key in block and not undefined(block[key]):
                reference = f"{block[key]:.4f} ({label})"
                break
        return f"| {name} | {values[0]} | {values[1]} | {reference or '—'} |"

    def delta(key: str) -> float:
        return head_b.get(key, float("nan")) - head_a.get(key, float("nan"))

    lines = [
        "# Arm A vs Arm B — unconditional ZINC250K flow",
        "",
        subtitle,
        "",
        "Arm A is the committed baseline (channel-split couplings). Arm B adds one "
        "node-autoregressive coupling per block, which transforms every channel of node *i* "
        "given the full vectors of the preceding nodes. Everything else — the frozen "
        "autoencoder, decoder, dataset split, smoothing sigma, whitening, prior, optimiser, "
        "learning rate and schedule — is identical, and Arm B is the identity map at "
        "initialisation, so the two arms start from the same density.",
        "",
        "## Validity: three different quantities",
        "",
        "Uniqueness is shown beside each validity because it divides by the number of valid "
        "molecules: a decoder that produced none has an **undefined** uniqueness (N/A), not a "
        "uniqueness of zero.",
        "",
        "| Decoder | Validity A | Validity B | Uniqueness A | Uniqueness B | What it measures |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name in DECODERS:
        if not all(name in n for n in numbers):
            continue
        values = [show(n[name]["strict_validity"]) for n in numbers]
        unique = [show(n[name].get("uniqueness")) for n in numbers]
        meaning = {"raw": "argmax bond matrix, no chemistry applied",
                   "constrained": "greedy valence-aware assembly with a connectivity pass",
                   "degree": "degree-constrained integer program, greedy fallback when infeasible"}
        lines.append(f"| {name} | {values[0]} | {values[1]} | {unique[0]} | {unique[1]} | "
                     f"{meaning[name]} |")
    lines += [
        "",
        "**Raw validity is the only one that reflects the model's own bond predictions.** The "
        "other two include chemistry rules that the decoder applies regardless of what the "
        "flow produced, so a high constrained or degree validity with a high fallback rate is "
        "largely a property of the assembler.",
        "",
        "## Headline metrics (degree-constrained decoder)",
        "",
        "| Metric | Arm A | Arm B | Reference |",
        "|---|---:|---:|---|",
        row("mean QED", "qed_mean"),
        row("fraction QED >= 0.7", "qed_at_least_0p7"),
        row("aromatic rings / molecule", "aromatic_rings_mean"),
        row("rings / molecule", "rings_mean"),
        row("aromatic / all rings", "aromatic_ring_fraction"),
        row("degree-decoder fallback", "fallback_rate"),
        row("samples the program solved", "samples_solved_by_program"),
        row("validity among solved samples", "validity_among_solved"),
        row("uniqueness (of valid)", "uniqueness"),
        row("novelty vs training set", "novelty_vs_train"),
        row("internal diversity", "internal_diversity"),
        row("N.U.V.", "nuv"),
        "",
        "## What improved",
        "",
    ]
    improvements = [
        ("raw validity", numbers[1].get("raw", {}).get("strict_validity", float("nan"))
         - numbers[0].get("raw", {}).get("strict_validity", float("nan"))),
        ("mean QED", delta("qed_mean")),
        ("fraction QED >= 0.7", delta("qed_at_least_0p7")),
        ("aromatic rings / molecule", delta("aromatic_rings_mean")),
        ("degree-decoder fallback (negative is better)", delta("fallback_rate")),
        ("internal diversity", delta("internal_diversity")),
    ]
    for name, value in improvements:
        if undefined(value):
            # Undefined is reported, not silently dropped: a metric that divides by the
            # number of valid molecules has no value when an arm produced none.
            lines.append(f"- **{name}**: N/A (undefined for at least one arm, e.g. no "
                         f"valid molecules to divide by).")
            continue
        direction = "better" if (value > 0) != name.endswith("(negative is better)") else "worse"
        lines.append(f"- **{name}**: {value:+.4f} for Arm B ({direction}).")
    lines += [
        "",
        "## Limitations that remain",
        "",
        f"- These are **10,000-step** checkpoints out of a 50,000-step budget, read at the "
        f"planned pause. Neither arm is a converged model.",
        "- Arm B carries **67.6% more parameters** than Arm A (2,573,696 vs 1,535,360), so a "
        "difference between the arms is a difference of *mechanism plus capacity*. A "
        "capacity-matched baseline would be needed to separate the two.",
        "- Arm B is **node-order dependent**: it models ordered node tuples in the dataset's "
        "atom order and is no longer exchangeable. Decoding is order-agnostic, so this does "
        "not affect the molecules, but the density is not a set density.",
    ]
    undefined_metrics = sorted(
        key for key in ("uniqueness", "novelty_vs_train", "internal_diversity", "qed_mean",
                        "qed_at_least_0p7", "nuv")
        if undefined(head_a.get(key)) or undefined(head_b.get(key)))
    if undefined_metrics:
        lines.append("- Shown as **N/A** because they are undefined for at least one arm "
                     "(their denominator is the number of valid molecules, which was zero): "
                     + ", ".join(undefined_metrics) + ".")
    for name in DECODERS:
        for entry, label in zip(numbers, labels):
            if name in entry and undefined(entry[name].get("uniqueness")) \
                    and not undefined(entry[name].get("strict_validity")) \
                    and entry[name]["strict_validity"] == 0.0:
                lines.append(f"- {label}'s **{name}** decoder produced no valid molecule at all, "
                             f"so its uniqueness, novelty and diversity are undefined rather "
                             f"than zero.")
    fallback = head_b.get("fallback_rate", float("nan"))
    if not undefined(fallback) and fallback > 0.25:
        lines.append(f"- Arm B's degree-constrained program is still infeasible for "
                     f"{fallback:.1%} of samples, so a large share of its reported validity "
                     f"comes from the greedy fallback rather than from the predicted degrees.")
    qed_fraction = head_b.get("qed_at_least_0p7", float("nan"))
    target = next((block["qed_at_least_0p7"] for block in references.values()), float("nan"))
    if not undefined(qed_fraction) and not undefined(target):
        lines.append(f"- The fraction above QED 0.7 is {qed_fraction:.1%} against {target:.1%} in "
                     f"real validation data, so drug-likeness is not yet matched.")
    lines += [
        "- Uniqueness and novelty are computed on this sample budget only; novelty references "
        "the training split, never the held-out test split.",
        f"- The molecule grid shows {chosen_count} of {unique_count} unique valid Arm B "
        f"molecules, evenly spaced across the QED range rather than selected for appearance.",
        "",
        "## Provenance",
        "",
    ]
    for arm, number in zip(arms, numbers):
        metrics = arm["metrics"]
        lines.append(f"- **Arm {arm['label']}**: `{arm['metrics_path']}`, "
                     f"{number.get('samples')} samples, "
                     f"flow `{metrics.get('flow_checkpoint', 'unknown')}` "
                     f"step {metrics.get('flow_step', 'unknown')}, "
                     f"autoencoder sha256 "
                     f"{str((metrics.get('autoencoder') or {}).get('sha256', ''))[:16]}, "
                     f"temperature {(metrics.get('settings') or {}).get('temperature', 'unknown')}.")
    for label, block in sorted(references.items()):
        lines.append(f"- **{label}**: `{block['source']}`.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm-a", required=True, help="metrics.json or its directory (baseline)")
    parser.add_argument("--arm-b", required=True, help="metrics.json or its directory (node-AR)")
    parser.add_argument("--output-dir", default="outputs/figures/arm_comparison")
    parser.add_argument("--reference", action="append", default=[],
                        help="an earlier diagnostics/populations JSON for real and "
                             "reconstruction reference values (repeatable)")
    parser.add_argument("--select", choices=("spread", "best", "random"), default="spread")
    parser.add_argument("--grid-molecules", type=int, default=GRID_MOLECULES)
    parser.add_argument("--note", default="", help="extra text for the figure subtitle")
    arguments = parser.parse_args(argv)

    try:
        output_directory = Path(arguments.output_dir)
        output_directory.mkdir(parents=True, exist_ok=True)
        if arguments.grid_molecules < 1:
            raise ValueError("--grid-molecules must be >= 1")
        arms = [load_arm(Path(arguments.arm_a), "A"), load_arm(Path(arguments.arm_b), "B")]
        numbers = [decoder_numbers(arm) for arm in arms]
        for arm, number in zip(arms, numbers):
            if number["headline_decoder"] not in number:
                raise ValueError(f"arm {arm['label']}: metrics.json has no "
                                 f"'{number['headline_decoder']}' decoder block")
        references = load_references([Path(p) for p in arguments.reference])

        settings = (arms[0]["metrics"].get("settings") or {})
        temperature = settings.get("temperature", "unknown")
        subtitle = (f"{numbers[0].get('samples')} samples per arm, temperature {temperature}, "
                    f"flow step {arms[0]['metrics'].get('flow_step', 'unknown')} "
                    f"(paused), frozen autoencoder unchanged")
        if arguments.note:
            subtitle += f" — {arguments.note}"

        comparison_figure(arms, numbers, references, subtitle=subtitle,
                          output=output_directory / "arm_comparison")

        chosen: list[tuple[str, float]] = []
        unique_total = 0
        if arms[1]["records"]:
            decoder = numbers[1]["headline_decoder"]
            chosen = choose_molecules(arms[1]["records"], decoder,
                                      count=arguments.grid_molecules, mode=arguments.select)
            unique_total = len({r[decoder]["smiles"] for r in arms[1]["records"]
                                if (r.get(decoder) or {}).get("valid")
                                and r[decoder].get("smiles")})
            if chosen:
                molecule_grid(chosen, output=output_directory / "arm_b_molecules",
                              subject="Arm B", mode=arguments.select,
                              total_unique=unique_total)
            else:
                print("WARNING: Arm B produced no valid molecules, so no grid was drawn",
                      file=sys.stderr, flush=True)
        else:
            print(f"WARNING: no samples.jsonl beside {arms[1]['metrics_path']}; the molecule "
                  f"grid and the solved-sample metrics are unavailable",
                  file=sys.stderr, flush=True)

        summary = results_summary(arms, numbers, references, subtitle=subtitle,
                                  chosen_count=len(chosen), unique_count=unique_total)
        (output_directory / "results_summary.md").write_text(summary, encoding="utf-8")
        atomic_write_json({"subtitle": subtitle, "arms": {a["label"]: n for a, n in zip(arms, numbers)},
                           "references": references, "selection": arguments.select,
                           "grid": [{"smiles": s, "qed": q} for s, q in chosen],
                           "inputs": {a["label"]: {"metrics": a["metrics_path"],
                                                   "samples": a["samples_path"]} for a in arms}},
                          output_directory / "figure_data.json")
        print(summary, flush=True)
        print(f"wrote {output_directory}/arm_comparison.png/.pdf, arm_b_molecules.png/.svg, "
              f"results_summary.md, figure_data.json", flush=True)
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
    sys.exit(main())
