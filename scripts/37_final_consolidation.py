#!/usr/bin/env python3
"""Final verification and consolidation. READ-ONLY: no training, no generation, no model.

Reads the JSON artefacts earlier stages already wrote and produces the definitive
comparison table, a corrected statistical read of the paired QED response, a checkpoint
identity audit, and a figure/table manifest for the report. It never samples, never
trains, and never rewrites an input.

WHY THIS SCRIPT EXISTS: TWO DEFECTS IN THE ORIGINAL EVALUATION
--------------------------------------------------------------
1. THE MODEL-TO-MODEL TEST WAS WRONG. `scripts/36_evaluate_relabel.py` set

       response_improved = (candidate_mean > reference_mean) and candidate_CI_excludes_0

   The second clause asks whether the FINE-TUNED model's own response differs from zero.
   It says nothing about whether it differs from the ORIGINAL model's response. Two
   models can both have responses comfortably excluding zero and be statistically
   indistinguishable from each other. That flag could therefore report an improvement
   that the data do not support, and every verdict derived from it is void. This script
   recomputes the comparison on the DIFFERENCE between models and reports the original
   flag as INVALID wherever it appears.

2. THE PER-SAMPLE DELTAS WERE NOT PERSISTED. `paired_response` computed the per-molecule
   deltas, bootstrapped them, and discarded them; only the summary survived into
   `relabel_evaluation.json`. The exact paired test on the model-to-model difference
   needs those deltas, so it CANNOT be computed from the saved artefacts.

   What can be computed is a CONSERVATIVE bound. The evaluation used common prior noise
   across models, so the two models' per-molecule deltas are positively correlated and

       SE(difference) = sqrt(SE_a^2 + SE_b^2 - 2*rho*SE_a*SE_b)  <=  sqrt(SE_a^2 + SE_b^2)

   Treating the models as independent therefore OVERSTATES the standard error. A
   difference that is significant under that bound is genuinely significant; one that is
   not is INCONCLUSIVE, not refuted. This script labels the two outcomes differently and
   never collapses them into a single claim.

   Regenerating the deltas would require re-running generation, which is out of scope
   here. The script reads them if a future run persists them under `paired_deltas`, and
   says so plainly when it has to fall back.

CHECKPOINT IDENTITY
-------------------
`scripts/35_finetune_relabel.py` initialises its best-so-far to the step-0 BASELINE
validation loss and writes `best.pt` only when a later evaluation beats it. So an absent
`best.pt` is not a bug: it means the fine-tune never improved real validation NLL, and
the conclusions necessarily describe `last.pt`, the end of the fixed budget. This script
states which file the conclusions attach to instead of leaving it implicit.

Exit codes: 0 ok; 2 missing input; 4 bad arguments; 1 unexpected.
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

Z95 = 1.959963984540054
# Headline numbers pulled out of each known artefact, so the findings table is traceable
# to a file rather than to memory.
ARTIFACT_KEYS = {
    "qed_gap_decomposition": ("delta_z", "delta_prior", "delta_logdet", "total"),
    "acp_gate": ("verdict",),
    "guidance_pilot": ("phase_a_surrogate",),
    "qed_support_rerank": ("verdict",),
    "finetune_report": ("real_validation_nll_change", "nonfinite_steps"),
}


# ----------------------------------------------------------------- statistics ---
def standard_error_from_ci(ci, *, z: float = Z95) -> float | None:
    """Recover the bootstrap standard error from a symmetric 95% interval."""
    if not ci or ci[0] is None or ci[1] is None:
        return None
    width = float(ci[1]) - float(ci[0])
    if not math.isfinite(width) or width < 0:
        return None
    return width / (2.0 * z)


def two_sided_p(z: float) -> float:
    """Normal two-sided p-value. `math.erfc` only, no scipy dependency."""
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni across the comparisons actually made.

    The original verdict applied no multiplicity control at all while testing one
    comparison per N, so the adjusted values are reported alongside the raw ones.
    """
    usable = {k: v for k, v in pvalues.items() if v is not None and math.isfinite(v)}
    order = sorted(usable, key=lambda k: usable[k])
    total = len(order)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(order):
        value = min(1.0, (total - rank) * usable[key])
        running = max(running, value)          # Holm is monotone non-decreasing
        adjusted[key] = running
    for key in pvalues:
        adjusted.setdefault(key, None)
    return adjusted


def paired_difference_exact(candidate_deltas, reference_deltas, *, seed: int = 20261115,
                            draws: int = 20000) -> dict[str, Any]:
    """Exact paired bootstrap on the per-molecule model-to-model difference.

    Only reachable when a run persisted `paired_deltas`. Pairs are matched by position,
    which is valid precisely because the evaluation shared one prior draw across models.
    """
    import numpy as np

    pairs = [(float(a), float(b)) for a, b in zip(candidate_deltas, reference_deltas)
             if a is not None and b is not None
             and math.isfinite(float(a)) and math.isfinite(float(b))]
    if len(pairs) < 2:
        return {"method": "exact_paired", "pairs": len(pairs), "mean": None,
                "ci95": [None, None], "significant": False,
                "note": "fewer than two jointly scorable pairs"}
    differences = np.asarray([a - b for a, b in pairs], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = differences[rng.integers(0, differences.size,
                                     size=(draws, differences.size))].mean(axis=1)
    low, high = float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
    standard_error = float(differences.std(ddof=1) / math.sqrt(differences.size))
    z = float(differences.mean() / standard_error) if standard_error > 0 else 0.0
    return {"method": "exact_paired", "pairs": int(differences.size),
            "mean": float(differences.mean()), "ci95": [low, high],
            "standard_error": standard_error, "z": z, "p_value": two_sided_p(z),
            "significant": bool(low > 0.0 or high < 0.0),
            "note": ("exact: per-molecule differences on a shared prior draw, so the "
                     "correlation between models is handled by the pairing itself")}


def paired_difference_conservative(candidate, reference) -> dict[str, Any]:
    """Independence-bounded test on the difference, from the saved marginal CIs only.

    Conservative by construction: common noise makes the two models' deltas positively
    correlated, so sqrt(SE_a^2 + SE_b^2) is an UPPER bound on the true standard error of
    the difference. Significant here means significant; not significant means
    INCONCLUSIVE, and the two are never merged.
    """
    mean_a, mean_b = candidate.get("mean"), reference.get("mean")
    se_a = standard_error_from_ci(candidate.get("ci95"))
    se_b = standard_error_from_ci(reference.get("ci95"))
    if mean_a is None or mean_b is None or se_a is None or se_b is None:
        return {"method": "conservative_independent_bound", "mean": None,
                "ci95": [None, None], "significant": False, "inconclusive": True,
                "note": "a marginal mean or interval was missing from the artefact"}
    difference = float(mean_a) - float(mean_b)
    bound = math.sqrt(se_a ** 2 + se_b ** 2)
    low, high = difference - Z95 * bound, difference + Z95 * bound
    significant = bool(low > 0.0 or high < 0.0)
    z = difference / bound if bound > 0 else 0.0
    return {
        "method": "conservative_independent_bound",
        "candidate_mean": float(mean_a), "reference_mean": float(mean_b),
        "candidate_se": se_a, "reference_se": se_b,
        "mean": difference, "standard_error_upper_bound": bound,
        "ci95": [low, high], "z": z, "p_value": two_sided_p(z),
        "significant": significant,
        "inconclusive": not significant,
        "note": ("UPPER bound on the standard error: common prior noise makes the two "
                 "models' per-molecule deltas positively correlated, so the true "
                 "interval is narrower. Significant under this bound is a real finding; "
                 "not significant is INCONCLUSIVE and needs the per-sample deltas, "
                 "which the saved artefact does not contain."),
    }


# ------------------------------------------------------------------ artefacts ---
def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"artefact not found: {path}")
    return json.loads(path.read_text())


def extract_responses(evaluation: dict[str, Any], *, decoder: str) -> dict[str, Any]:
    """Per model, per N: the saved paired response and any persisted per-sample deltas."""
    out: dict[str, Any] = {}
    for label, block in (evaluation.get("models") or {}).items():
        entry: dict[str, Any] = {"step": block.get("step"),
                                 "checkpoint": block.get("checkpoint"),
                                 "real_validation": block.get("real_validation"),
                                 "relabel_finetune": block.get("relabel_finetune"),
                                 "by_n": {}}
        for n, per_n in (block.get("by_n") or {}).items():
            response = ((per_n.get("paired_response") or {}).get(decoder) or {})
            entry["by_n"][n] = {
                "mean": response.get("mean"),
                "ci95": response.get("ci95"),
                "jointly_valid_pairs": response.get("jointly_valid_pairs"),
                "candidate_ci_excludes_zero": response.get("excludes_zero"),
                # only present if a future run persisted them
                "paired_deltas": response.get("paired_deltas"),
                "noise_seed": per_n.get("noise_seed"),
                "cells": {target: {d: (cell.get(d) or {})
                                   for d in (cell or {})}
                          for target, cell in (per_n.get("cells") or {}).items()},
            }
        out[label] = entry
    return out


def compare_models(responses, *, candidate: str, reference: str) -> dict[str, Any]:
    """The corrected model-to-model comparison, per N, with multiplicity control."""
    if candidate not in responses or reference not in responses:
        return {"evaluated": False,
                "note": f"the evaluation does not contain both {candidate!r} and "
                        f"{reference!r}"}
    result: dict[str, Any] = {"evaluated": True, "candidate": candidate,
                              "reference": reference, "by_n": {}}
    pvalues: dict[str, float] = {}
    for n, candidate_block in responses[candidate]["by_n"].items():
        reference_block = responses[reference]["by_n"].get(n) or {}
        if candidate_block.get("paired_deltas") and reference_block.get("paired_deltas"):
            test = paired_difference_exact(candidate_block["paired_deltas"],
                                           reference_block["paired_deltas"])
        else:
            test = paired_difference_conservative(candidate_block, reference_block)
        test["original_flag_response_improved"] = (
            "INVALID: compared the candidate's own interval against zero, not against "
            "the reference model")
        result["by_n"][n] = test
        pvalues[n] = test.get("p_value")
    result["holm_adjusted_p"] = holm_adjust(pvalues)
    result["comparisons"] = len([v for v in pvalues.values() if v is not None])
    significant = [n for n, t in result["by_n"].items() if t.get("significant")]
    inconclusive = [n for n, t in result["by_n"].items() if t.get("inconclusive")]
    result["significant_n"] = significant
    result["inconclusive_n"] = inconclusive
    result["conclusion"] = (
        "IMPROVEMENT NOT ESTABLISHED" if not significant else
        f"improvement established at N={significant} under the stated test")
    result["method_note"] = (
        "The test is on the DIFFERENCE between models, which is what the claim requires. "
        "Where the exact per-molecule deltas were not persisted the bound is "
        "conservative, so a non-significant result means the data cannot settle the "
        "question rather than that the models are equal.")
    return result


def checkpoint_audit(paths: dict[str, Path]) -> dict[str, Any]:
    """Do best.pt and last.pt differ, and which do the conclusions describe?

    Torch is imported lazily so the table and statistics work in an environment without
    it. Checkpoints are opened read-only and nothing is written back.
    """
    audit: dict[str, Any] = {"checkpoints": {}, "present": [], "absent": []}
    for label, path in paths.items():
        if path.is_file():
            audit["present"].append(label)
        else:
            audit["absent"].append(label)
            audit["checkpoints"][label] = {"path": str(path), "exists": False}
    if "best" in audit["absent"]:
        audit["best_absent_meaning"] = (
            "best.pt was never written. scripts/35 seeds its best-so-far with the "
            "step-0 BASELINE real validation NLL and only writes best.pt when a later "
            "evaluation beats it, so an absent file means the fine-tune never improved "
            "real validation likelihood. The conclusions therefore describe last.pt, "
            "the end of the fixed budget.")
    if not audit["present"]:
        audit["note"] = "no checkpoint was readable; nothing to compare"
        return audit

    import torch

    loaded: dict[str, Any] = {}
    for label in audit["present"]:
        blob = torch.load(paths[label], map_location="cpu", weights_only=False)
        state = blob.get("model_state_dict") or {}
        finetune = (blob.get("relabel_finetune") or {})
        loaded[label] = state
        audit["checkpoints"][label] = {
            "path": str(paths[label]), "exists": True,
            "step": blob.get("step"),
            "best_metric_recorded": blob.get("best_metric"),
            "parameters": int(sum(v.numel() for v in state.values())),
            "tensors": len(state),
            "parent_checkpoint": finetune.get("parent_checkpoint"),
            "parent_step": finetune.get("parent_step"),
            "label_decoder": finetune.get("label_decoder"),
            "relabel_weight": finetune.get("relabel_weight"),
            "real_validation_at_save": finetune.get("real_validation"),
        }
    if len(loaded) >= 2:
        labels = sorted(loaded)
        for i, left in enumerate(labels):
            for right in labels[i + 1:]:
                a, b = loaded[left], loaded[right]
                shared = sorted(set(a) & set(b))
                worst = 0.0
                worst_tensor = None
                for key in shared:
                    if a[key].shape != b[key].shape:
                        worst, worst_tensor = float("inf"), key
                        break
                    gap = float((a[key].double() - b[key].double()).abs().max())
                    if gap > worst:
                        worst, worst_tensor = gap, key
                audit[f"{left}_vs_{right}"] = {
                    "shared_tensors": len(shared),
                    "only_in_first": sorted(set(a) - set(b)),
                    "only_in_second": sorted(set(b) - set(a)),
                    "max_absolute_weight_difference": worst,
                    "largest_difference_in": worst_tensor,
                    "identical": bool(worst == 0.0),
                }
    return audit


def definitive_table(responses, comparison, *, settings, decoder: str) -> dict[str, Any]:
    """One row per (checkpoint, N), with the protocol columns kept visible."""
    rows = []
    for label, block in responses.items():
        for n, per_n in block["by_n"].items():
            test = (comparison.get("by_n") or {}).get(n) or {}
            applies = label == comparison.get("candidate")
            rows.append({
                "checkpoint": label,
                "step": block.get("step"),
                "n": int(n),
                "paired_response_mean": per_n.get("mean"),
                "paired_response_ci95": per_n.get("ci95"),
                "jointly_valid_pairs": per_n.get("jointly_valid_pairs"),
                "vs_reference_difference": test.get("mean") if applies else None,
                "vs_reference_ci95": test.get("ci95") if applies else None,
                "vs_reference_method": test.get("method") if applies else None,
                "vs_reference_significant": test.get("significant") if applies else None,
                "real_validation_nll": (block.get("real_validation") or {}).get("val_loss"),
                "condition_gap": (block.get("real_validation") or {}).get(
                    "val_condition_gap"),
                "eval_noise_seed": per_n.get("noise_seed"),
                "samples_per_cell": settings.get("samples_per_cell"),
                "temperature": settings.get("temperature"),
                "decoder": decoder,
                "pairing": "per-molecule, shared prior draw, jointly valid pairs only",
            })
    rows.sort(key=lambda r: (r["checkpoint"], r["n"]))
    return {"rows": rows,
            "protocol_note": (
                "All rows share one evaluation protocol: the same prior draw per N "
                "reused across checkpoints and requests, the same temperature and "
                "samples per cell, and the degree-constrained decoder as the headline. "
                "The response is the per-molecule QED difference between the highest and "
                "lowest request at fixed N, over jointly valid pairs only."),
            "seed_note": (
                "eval_noise_seed is the evaluation stream and is DISTINCT from the seed "
                "that collected the relabeling pool; scripts/36 refuses to run if they "
                "coincide. Earlier stages used their own seeds, so numbers from "
                "different stages are not sample-matched to these and must not be "
                "pooled.")}


def markdown_table(table) -> str:
    header = ("| checkpoint | step | N | paired dQED | 95% CI | pairs | vs reference | "
              "CI of difference | significant | real val NLL | condition gap |")
    divider = "|" + "---|" * 11
    lines = [header, divider]

    def cell(value, spec=".4f"):
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, (int, float)):
            return format(value, spec)
        if isinstance(value, list) and len(value) == 2:
            if value[0] is None:
                return "n/a"
            return f"[{value[0]:+.4f}, {value[1]:+.4f}]"
        return "n/a" if value is None else str(value)

    for row in table["rows"]:
        lines.append(
            f"| {row['checkpoint']} | {cell(row['step'], 'd') if isinstance(row['step'], int) else 'n/a'} "
            f"| {row['n']} | {cell(row['paired_response_mean'])} "
            f"| {cell(row['paired_response_ci95'])} "
            f"| {cell(row['jointly_valid_pairs'], 'd') if isinstance(row['jointly_valid_pairs'], int) else 'n/a'} "
            f"| {cell(row['vs_reference_difference'])} "
            f"| {cell(row['vs_reference_ci95'])} "
            f"| {cell(row['vs_reference_significant'])} "
            f"| {cell(row['real_validation_nll'])} "
            f"| {cell(row['condition_gap'])} |")
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluation", required=True,
                        help="relabel_evaluation.json from scripts/36")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", action="append", default=[],
                        help="label=path, repeatable; e.g. best=.../best.pt")
    parser.add_argument("--artifact", action="append", default=[],
                        help="label=path to another stage's JSON, repeatable")
    parser.add_argument("--candidate", default="finetuned")
    parser.add_argument("--reference", default="final")
    parser.add_argument("--decoder", default="degree")
    parser.add_argument("--skip-checkpoint-audit", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        def pairs(items, flag):
            out: dict[str, Path] = {}
            for item in items:
                if "=" not in item:
                    raise ValueError(f"{flag} must be label=path, got {item!r}")
                label, path = item.split("=", 1)
                if not label.strip() or not path.strip():
                    raise ValueError(f"{flag} must be label=path, got {item!r}")
                out[label.strip()] = Path(path.strip())
            return out

        checkpoints = pairs(args.checkpoint, "--checkpoint")
        artifacts = pairs(args.artifact, "--artifact")
        if args.candidate == args.reference:
            raise ValueError("--candidate and --reference must differ")
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4

    started = time.perf_counter()
    try:
        evaluation = read_json(Path(args.evaluation))
        responses = extract_responses(evaluation, decoder=args.decoder)
        if not responses:
            raise FileNotFoundError(
                f"{args.evaluation} contains no models; nothing to consolidate")
        comparison = compare_models(responses, candidate=args.candidate,
                                    reference=args.reference)
        table = definitive_table(responses, comparison,
                                 settings=evaluation.get("settings") or {},
                                 decoder=args.decoder)

        payload: dict[str, Any] = {
            "framing": ("Final verification and consolidation. READ-ONLY over saved "
                        "artefacts: nothing was trained, sampled or recomputed from a "
                        "model. The model-to-model statistics replace the original "
                        "verdict, which tested the candidate's interval against zero "
                        "rather than against the reference model."),
            "inputs": {"evaluation": str(args.evaluation),
                        "checkpoints": {k: str(v) for k, v in checkpoints.items()},
                        "artifacts": {k: str(v) for k, v in artifacts.items()}},
            "decoder": args.decoder,
            "models_found": sorted(responses),
            "per_model": responses,
            "model_comparison": comparison,
            "definitive_table": table,
        }

        if checkpoints and not args.skip_checkpoint_audit:
            payload["checkpoint_audit"] = checkpoint_audit(checkpoints)

        collected: dict[str, Any] = {}
        for label, path in artifacts.items():
            if not path.is_file():
                collected[label] = {"path": str(path), "present": False}
                continue
            blob = json.loads(path.read_text())
            collected[label] = {
                "path": str(path), "present": True,
                "keys": sorted(blob)[:40],
                "verdict": blob.get("verdict") if not isinstance(
                    blob.get("verdict"), dict) else blob["verdict"].get("verdict"),
                "selected_headline": {k: blob.get(k)
                                      for k in ARTIFACT_KEYS.get(label, ())
                                      if k in blob},
            }
        payload["artifacts"] = collected
        payload["elapsed_seconds"] = time.perf_counter() - started

        from zinc_gnf.utils.runtime import atomic_write_json
        atomic_write_json(payload, out_dir / "final_consolidation.json")
        (out_dir / "definitive_table.md").write_text(markdown_table(table) + "\n")
        (out_dir / "final_consolidation.txt").write_text(format_report(payload))
        print(format_report(payload))
        print(f"\nwrote {out_dir / 'final_consolidation.json'}")
        print(f"wrote {out_dir / 'definitive_table.md'}")
        print(f"wrote {out_dir / 'final_consolidation.txt'}")
        return 0
    except FileNotFoundError as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 2
    except Exception:  # noqa: BLE001
        print("FATAL: unexpected\n" + traceback.format_exc(), file=sys.stderr, flush=True)
        return 1


def _fmt(value, spec=".4f"):
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return format(value, spec)
    if isinstance(value, list) and len(value) == 2 and value[0] is not None:
        return f"[{value[0]:+.4f}, {value[1]:+.4f}]"
    return "n/a" if value is None else str(value)


def format_report(payload: dict[str, Any]) -> str:
    lines = ["Final verification and consolidation", "=" * 78, "",
             payload["framing"], "",
             f"headline decoder: {payload['decoder']}",
             f"models found: {', '.join(payload['models_found'])}", "",
             "-- CORRECTED MODEL-TO-MODEL COMPARISON " + "-" * 37]
    comparison = payload["model_comparison"]
    if not comparison.get("evaluated"):
        lines.append(f"  {comparison.get('note')}")
    else:
        lines.append(f"  candidate={comparison['candidate']}  "
                     f"reference={comparison['reference']}  "
                     f"comparisons={comparison['comparisons']}")
        for n, test in comparison["by_n"].items():
            adjusted = (comparison.get("holm_adjusted_p") or {}).get(n)
            lines += [
                f"  N={n}: difference={_fmt(test.get('mean'))} "
                f"CI={_fmt(test.get('ci95'))} "
                f"p={_fmt(test.get('p_value'), '.4g')} "
                f"holm_p={_fmt(adjusted, '.4g')}",
                f"       method={test.get('method')}  "
                f"significant={_fmt(test.get('significant'))}  "
                f"inconclusive={_fmt(test.get('inconclusive'))}"]
        lines += [f"  significant at N: {comparison['significant_n'] or 'none'}",
                  f"  inconclusive at N: {comparison['inconclusive_n'] or 'none'}",
                  f"  CONCLUSION: {comparison['conclusion']}", "",
                  f"  {comparison['method_note']}"]
    if payload.get("checkpoint_audit"):
        audit = payload["checkpoint_audit"]
        lines += ["", "-- CHECKPOINT IDENTITY " + "-" * 53]
        for label, block in audit["checkpoints"].items():
            if not block.get("exists"):
                lines.append(f"  {label}: ABSENT ({block['path']})")
                continue
            lines.append(f"  {label}: step {block.get('step')}  "
                         f"{block.get('parameters')} parameters in "
                         f"{block.get('tensors')} tensors")
            if block.get("parent_step") is not None:
                lines.append(f"       fine-tuned from step {block['parent_step']}, "
                             f"label decoder {block.get('label_decoder')}, "
                             f"weight {block.get('relabel_weight')}")
        if audit.get("best_absent_meaning"):
            lines += ["", f"  {audit['best_absent_meaning']}"]
        for key, block in audit.items():
            if not key.endswith("_vs_last") and "_vs_" not in key:
                continue
            if not isinstance(block, dict):
                continue
            lines += [f"  {key}: identical={_fmt(block.get('identical'))}  "
                      f"max |dw|={_fmt(block.get('max_absolute_weight_difference'), '.3e')} "
                      f"(largest in {block.get('largest_difference_in')})"]
    lines += ["", "-- DEFINITIVE TABLE " + "-" * 56, "",
              markdown_table(payload["definitive_table"]), "",
              f"  {payload['definitive_table']['protocol_note']}", "",
              f"  {payload['definitive_table']['seed_note']}"]
    if payload.get("artifacts"):
        lines += ["", "-- SUPPORTING ARTEFACTS " + "-" * 52]
        for label, block in payload["artifacts"].items():
            status = "present" if block.get("present") else "MISSING"
            lines.append(f"  {label:<26} {status}  verdict={block.get('verdict')}")
    lines += ["", "-- WHAT THIS DOES AND DOES NOT ESTABLISH " + "-" * 35,
              "  A difference that is significant under the conservative bound is a real",
              "  finding. A difference that is not significant is INCONCLUSIVE: the saved",
              "  artefacts do not contain the per-molecule deltas needed for the exact",
              "  paired test, and absence of evidence here is not evidence of equality.",
              "  The original `response_improved` flag is INVALID wherever it appears,",
              "  because it compared the candidate's own interval against zero."]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
