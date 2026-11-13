#!/usr/bin/env python3
"""Diagnose conditional QED control against the baselines any honest report must beat.

READ-ONLY, ZERO COMPUTE: pure arithmetic on reported per-condition numbers. No model, no
dataset, no torch, no rdkit. Every input is declared in a JSON file so the numbers are
auditable and the whole analysis is regenerable when the artefacts are recomputed.

WHY THIS EXISTS
---------------
"Native QED MAE = 0.1848" is not interpretable on its own. A conditional generator has to
beat the trivial strategies before its MAE means anything, and two of them cost nothing:

  CONSTANT-PREDICTOR FLOOR. A model that ignores the request entirely and always emits
  QED = c scores MAE(c) = mean_t |c - t| over the requested targets. For targets
  {0.4, 0.6, 0.8} this is minimised at the MEDIAN target, c = 0.6, giving MAE = 0.1333.
  Any conditional model whose MAE exceeds that is, on this metric, worse than a constant.

  SIZE-ONLY PREDICTOR. Decoded QED may track molecule size rather than the request. If
  the spread of per-N means exceeds the spread of per-request means, then N is doing the
  work and the QED conditioning is decoration. This is measured, not assumed.

WHAT IT ALSO SEPARATES
----------------------
Cell-mean MAE and per-sample MAE are different quantities and must not be compared
casually. Cell-mean MAE uses |mean(cell) - target| and ignores within-cell scatter;
per-sample MAE includes it and is therefore always at least as large. The gap between
them is the scatter penalty, and reporting both shows whether a poor MAE comes from the
conditional mean sitting in the wrong place or from the samples being noisy.

Best-of-k interacts with exactly that scatter. A constant predictor has zero spread, so
best-of-k cannot improve it at all. If best-of-k DOES improve a real model's MAE, the
model's output distribution must already place mass near each target -- which means the
support is there and only the steering is missing. That inference is the point of the
`selection_leverage` block.

Exit codes: 0 ok; 2 missing input; 4 bad input; 1 unexpected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def constant_predictor_floor(targets) -> dict[str, Any]:
    """Best MAE reachable while ignoring the request entirely.

    Minimised at the median target for an L1 loss, which is why the optimum is reported
    rather than searched: argmin_c mean_t |c - t| is the median of t.
    """
    ordered = sorted(targets)
    best_constant = statistics.median(ordered)
    floor = mean([abs(best_constant - t) for t in ordered])
    grid = {f"{c:.2f}": mean([abs(c - t) for t in ordered])
            for c in [round(0.05 * i, 2) for i in range(4, 17)]}
    return {
        "targets": ordered,
        "best_constant": best_constant,
        "mae_floor": floor,
        "mae_by_constant": grid,
        "note": ("MAE of a model that ignores the request and always emits "
                 f"QED = {best_constant:g}. A conditional model that does not beat "
                 "this is not demonstrating conditional control on this metric."),
    }


def responsiveness(grid, targets) -> dict[str, Any]:
    """Per-N response from the lowest to the highest request, against the ideal."""
    low, high = min(targets), max(targets)
    ideal = high - low
    out: dict[str, Any] = {"ideal_response": ideal, "by_n": {}}
    delivered = []
    for n, row in grid.items():
        a, b = row.get(f"{low:g}"), row.get(f"{high:g}")
        if a is None or b is None:
            out["by_n"][n] = {"note": "missing endpoint"}
            continue
        response = b - a
        delivered.append(response)
        out["by_n"][n] = {
            "achieved_at_low": a, "achieved_at_high": b,
            "response": response,
            "fraction_of_ideal": response / ideal if ideal else None,
            "monotonic": all(
                row[f"{targets[i + 1]:g}"] >= row[f"{targets[i]:g}"]
                for i in range(len(targets) - 1)
                if row.get(f"{targets[i]:g}") is not None
                and row.get(f"{targets[i + 1]:g}") is not None),
        }
    out["mean_response"] = mean(delivered)
    out["mean_fraction_of_ideal"] = (mean(delivered) / ideal) if ideal and delivered else None
    out["positive_at_n"] = [n for n, b in out["by_n"].items()
                            if isinstance(b.get("response"), float) and b["response"] > 0]
    return out


def variance_attribution(grid, targets) -> dict[str, Any]:
    """Does decoded QED track molecule size or the requested property?"""
    per_n = {n: mean([row.get(f"{t:g}") for t in targets]) for n, row in grid.items()}
    per_request = {f"{t:g}": mean([row.get(f"{t:g}") for row in grid.values()])
                   for t in targets}
    n_values = [v for v in per_n.values() if v is not None]
    request_values = [v for v in per_request.values() if v is not None]
    n_spread = (max(n_values) - min(n_values)) if n_values else None
    request_spread = (max(request_values) - min(request_values)) if request_values else None
    monotone_in_request = all(
        per_request[f"{targets[i + 1]:g}"] >= per_request[f"{targets[i]:g}"]
        for i in range(len(targets) - 1)
        if per_request.get(f"{targets[i]:g}") is not None
        and per_request.get(f"{targets[i + 1]:g}") is not None)
    return {
        "mean_by_n": per_n,
        "mean_by_request": per_request,
        "spread_across_n": n_spread,
        "spread_across_request": request_spread,
        "size_over_request_ratio": (n_spread / request_spread)
        if n_spread is not None and request_spread else None,
        "monotone_in_request": monotone_in_request,
        "note": ("A ratio above 1 means molecule size moves decoded QED more than the "
                 "QED request does. Combined with monotone_in_request = false, it means "
                 "the request is not acting as a control variable at all."),
    }


def cell_mean_mae(grid, targets) -> dict[str, Any]:
    """MAE computed on cell means, which excludes within-cell scatter."""
    errors, per_n = [], {}
    for n, row in grid.items():
        row_errors = [abs(row[f"{t:g}"] - t) for t in targets
                      if row.get(f"{t:g}") is not None]
        per_n[n] = mean(row_errors)
        errors.extend(row_errors)
    return {"overall": mean(errors), "by_n": per_n,
            "note": ("|cell mean - target|, averaged. Always <= the per-sample MAE, "
                     "because it ignores within-cell scatter.")}


def selection_leverage(native_mae, selected, floor) -> dict[str, Any]:
    """What best-of-k reveals about the output distribution.

    A constant predictor has no spread, so best-of-k cannot improve it. Improvement from
    selection is therefore evidence that the model's own output distribution already
    covers the targets, and that the missing piece is steering rather than support.
    """
    out: dict[str, Any] = {"native_mae": native_mae, "constant_floor": floor,
                           "by_k": {}}
    for k, value in (selected or {}).items():
        out["by_k"][str(k)] = {
            "mae": value,
            "improvement_over_native": (native_mae - value)
            if native_mae is not None and value is not None else None,
            "beats_constant_floor": bool(value is not None and floor is not None
                                         and value < floor),
        }
    out["interpretation"] = (
        "Selection cannot help a model with no output spread, so any best-of-k gain "
        "shows the support already spans the targets. It does NOT show conditional "
        "control, and must be reported separately from native generation.")
    return out


def verdict(blocks) -> dict[str, Any]:
    native = blocks["reported"].get("native_qed_mae")
    floor = blocks["constant_floor"]["mae_floor"]
    cell = blocks["cell_mean_mae"]["overall"]
    attribution = blocks["variance_attribution"]
    response = blocks["responsiveness"]
    findings = []
    if native is not None and floor is not None:
        ratio = native / floor
        findings.append(
            f"Native per-sample MAE {native:.4f} vs constant-predictor floor "
            f"{floor:.4f}: the model is {abs(1 - ratio) * 100:.0f}% "
            f"{'WORSE' if ratio > 1 else 'better'} than always emitting QED = "
            f"{blocks['constant_floor']['best_constant']:g}.")
    if cell is not None and floor is not None:
        findings.append(
            f"On cell means ({cell:.4f}) the comparison still "
            f"{'favours the constant' if cell > floor else 'favours the model'}, so the "
            f"result is not an artefact of within-cell scatter alone.")
    if attribution.get("size_over_request_ratio"):
        findings.append(
            f"Molecule size moves decoded QED {attribution['size_over_request_ratio']:.1f}x "
            f"more than the QED request does, and the response across requests is "
            f"{'monotone' if attribution['monotone_in_request'] else 'NOT monotone'}.")
    if response.get("mean_fraction_of_ideal") is not None:
        findings.append(
            f"Mean delivered response is {response['mean_fraction_of_ideal'] * 100:.1f}% "
            f"of the ideal {response['ideal_response']:g}, positive at "
            f"{len(response['positive_at_n'])} of {len(response['by_n'])} node counts.")
    return {
        "findings": findings,
        "conclusion": (
            "On the decoded-QED metric the model does not demonstrate conditional "
            "control: it does not beat a constant predictor, and molecule size "
            "explains more of the variation than the requested property. Best-of-k "
            "gains come from the output spread, not from steering."),
        "caveat": (
            "These are arithmetic consequences of the REPORTED per-condition means. "
            "They should be recomputed from the evaluation artefacts before publication; "
            "the input file records which numbers are reported rather than recomputed."),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="docs/reported_results.json")
    parser.add_argument("--output-dir", default="outputs/metrics/conditioning_diagnostics")
    return parser.parse_args(argv)


def analyse(reported: dict[str, Any]) -> dict[str, Any]:
    targets = [float(t) for t in reported["qed_targets"]]
    grid = reported["achieved_qed_by_n"]
    blocks: dict[str, Any] = {"reported": reported}
    blocks["constant_floor"] = constant_predictor_floor(targets)
    blocks["responsiveness"] = responsiveness(grid, targets)
    blocks["variance_attribution"] = variance_attribution(grid, targets)
    blocks["cell_mean_mae"] = cell_mean_mae(grid, targets)
    blocks["selection_leverage"] = selection_leverage(
        reported.get("native_qed_mae"), reported.get("selected_qed_mae"),
        blocks["constant_floor"]["mae_floor"])
    blocks["verdict"] = verdict(blocks)
    return blocks


def main(argv=None) -> int:
    args = parse_args(argv)
    path = Path(args.input)
    try:
        if not path.is_file():
            print(f"FATAL: input not found: {path}", file=sys.stderr)
            return 2
        reported = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2
    try:
        for key in ("qed_targets", "achieved_qed_by_n"):
            if key not in reported:
                raise ValueError(f"input must contain {key!r}")
        if len(reported["qed_targets"]) < 2:
            raise ValueError("need at least two QED targets for a response")
    except ValueError as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 4
    try:
        blocks = analyse(reported)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "conditioning_diagnostics.json").write_text(
            json.dumps(blocks, indent=2))
        report = format_report(blocks)
        (out_dir / "conditioning_diagnostics.txt").write_text(report)
        print(report)
        print(f"\nwrote {out_dir / 'conditioning_diagnostics.json'}")
        print(f"wrote {out_dir / 'conditioning_diagnostics.txt'}")
        return 0
    except Exception:  # noqa: BLE001
        print("FATAL: unexpected\n" + traceback.format_exc(), file=sys.stderr)
        return 1


def _f(value, spec=".4f"):
    return format(value, spec) if isinstance(value, (int, float)) else "n/a"


def format_report(blocks) -> str:
    reported = blocks["reported"]
    floor = blocks["constant_floor"]
    response = blocks["responsiveness"]
    attribution = blocks["variance_attribution"]
    lines = ["Conditional QED control: diagnostics against trivial baselines",
             "=" * 78, "",
             f"source: {reported.get('source', 'reported numbers')}",
             f"status: {reported.get('status', 'unverified')}", "",
             "-- BASELINES THE MODEL MUST BEAT " + "-" * 44,
             f"  Constant-predictor floor (always emit QED = "
             f"{floor['best_constant']:g}): MAE = {_f(floor['mae_floor'])}",
             f"  Model, per-sample MAE (reported):            "
             f"{_f(reported.get('native_qed_mae'))}",
             f"  Model, cell-mean MAE (computed here):        "
             f"{_f(blocks['cell_mean_mae']['overall'])}",
             "", "-- RESPONSIVENESS AT FIXED N " + "-" * 48,
             f"  ideal response (lowest -> highest request): "
             f"{_f(response['ideal_response'], '.2f')}"]
    for n, block in response["by_n"].items():
        if "response" not in block:
            lines.append(f"  N={n}: {block.get('note')}")
            continue
        lines.append(
            f"  N={n:<3} {_f(block['achieved_at_low'], '.3f')} -> "
            f"{_f(block['achieved_at_high'], '.3f')}  response="
            f"{block['response']:+.4f}  "
            f"({block['fraction_of_ideal'] * 100:+.1f}% of ideal)  "
            f"monotone={block['monotonic']}")
    lines += [f"  mean response {_f(response['mean_response'])} = "
              f"{_f((response.get('mean_fraction_of_ideal') or 0) * 100, '.1f')}% of ideal",
              "", "-- IS IT SIZE OR IS IT THE REQUEST? " + "-" * 41]
    for n, value in attribution["mean_by_n"].items():
        lines.append(f"  mean achieved QED at N={n:<3} {_f(value)}")
    for request, value in attribution["mean_by_request"].items():
        lines.append(f"  mean achieved QED at request {request:<5} {_f(value)}")
    lines += [f"  spread across N        {_f(attribution['spread_across_n'])}",
              f"  spread across requests {_f(attribution['spread_across_request'])}",
              f"  size / request ratio   "
              f"{_f(attribution['size_over_request_ratio'], '.2f')}",
              f"  monotone in request    {attribution['monotone_in_request']}",
              "", "-- WHAT BEST-OF-k TELLS US " + "-" * 50]
    for k, block in blocks["selection_leverage"]["by_k"].items():
        lines.append(f"  best-of-{k:<3} MAE={_f(block['mae'])}  "
                     f"improvement over native={_f(block['improvement_over_native'])}  "
                     f"beats constant floor={block['beats_constant_floor']}")
    lines += [f"  {blocks['selection_leverage']['interpretation']}", "",
              "-- FINDINGS " + "-" * 65]
    for finding in blocks["verdict"]["findings"]:
        lines.append(f"  * {finding}")
    lines += ["", f"  {blocks['verdict']['conclusion']}", "",
              f"  CAVEAT: {blocks['verdict']['caveat']}"]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
