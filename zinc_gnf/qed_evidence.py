"""Stage 20: four independent findings, each with its own uncertainty - no single verdict.

WHY THE STAGE-19 DECISION LOGIC IS REPLACED
-------------------------------------------
Stage 19 chose exactly one hypothesis from a priority ladder: H1-decoder, then
H2-direction, then H3-constraints, then H1-decoding. Because `H2-direction` fired first,
the H3 test was never reached - so the degree-decoder attenuation that the same run's
tables actually showed could not appear in the verdict. The mechanisms are not mutually
exclusive: a flow can move samples along a weakly QED-predictive direction AND a
constrained assembler can erase part of what does arrive. A ladder that reports one
cannot express that.

This module therefore grades four axes INDEPENDENTLY:

  E1  the generated embeddings move weakly, or not along a QED-predictive direction
  E2  the decoder loses QED information on HELD-OUT real embeddings
  E3  degree-constrained assembly attenuates the QED response that does arrive
  E4  generated embeddings are shifted relative to the real ones the decoder was fitted on

Each axis returns one of four statuses, and the distinction between the last two is the
point of the exercise:

  supported           the measurement shows the effect, and its interval excludes the
                      null
  not_supported       the measurement is precise enough to rule the effect out
  inconclusive        measured, but the interval spans both answers - a sample-size or
                      support problem
  needs_further_test  the INSTRUMENT is not adequate to answer it. This is reported as a
                      fact about the instrument, never as evidence about the model.

The last status exists specifically so that "our probe could not see it" can never again
be recorded as "the embeddings carry no QED information".
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

# A read-out must show a real, non-zero within-N gain on REAL molecules before anything it
# says about generated embeddings counts as evidence.
INSTRUMENT_GAIN_FLOOR = 0.10
# Implied real QED difference, as a fraction of what was requested.
WEAK_RESPONSE_FRACTION = 0.5
STRONG_RESPONSE_FRACTION = 0.8
# Decoder preservation of a QED gap it is GIVEN, on held-out real embeddings.
TRANSFER_LOSS_CEILING = 0.5
TRANSFER_OK_FLOOR = 0.8
# Held-out real-vs-generated discriminability.
SHIFT_AUC_SUPPORTED = 0.75
SHIFT_AUC_CLEAN = 0.60

AXES = ("E1_generated_direction", "E2_decoder_transfer", "E3_degree_attenuation",
        "E4_distribution_shift")


def _finite(value: Any) -> bool:
    return value is not None and isinstance(value, (int, float)) and np.isfinite(value)


def _usable_instrument(readout: dict[str, Any]) -> dict[str, Any]:
    """Does this read-out have a demonstrated within-N gain on REAL molecules?"""
    gains = [block for block in (readout.get("reference_contrasts") or [])
             if _finite(block.get("gain"))]
    qualifying = [block for block in gains
                  if block.get("gain_excludes_zero")
                  and abs(block["gain"]) >= INSTRUMENT_GAIN_FLOOR]
    best = max(qualifying, key=lambda b: abs(b["gain"])) if qualifying else None
    return {
        "name": readout.get("name"),
        "cells_with_a_gain": len(gains),
        "cells_qualifying": len(qualifying),
        "median_gain": float(np.median([b["gain"] for b in gains])) if gains else float("nan"),
        "best_gain": best.get("gain") if best else float("nan"),
        "best_gain_n": best.get("n") if best else None,
        "usable": bool(qualifying),
        "held_out_mae": (readout.get("validation") or {}).get("mae"),
        "held_out_correlation": (readout.get("validation") or {}).get("correlation"),
        "calibration_slope": (readout.get("calibration") or {}).get(
            "slope_of_truth_on_prediction"),
    }


def assess_evidence(*, readouts: Sequence[dict[str, Any]], transfer: Sequence[dict[str, Any]],
                    attenuation: dict[str, Any], shift: dict[str, Any],
                    generated_readings: Sequence[dict[str, Any]] = (),
                    headline_decoder: str = "degree") -> dict[str, Any]:
    """Grade the four axes independently. No axis can suppress another."""
    findings: dict[str, Any] = {}
    instruments = [_usable_instrument(readout) for readout in readouts]
    usable = [block for block in instruments if block["usable"]]

    # ---------------------------------------------------------------- E1 ----
    e1: dict[str, Any] = {
        "axis": "E1_generated_direction",
        "claim": ("the generated embeddings do not move along a QED-predictive direction "
                  "strongly enough to deliver the requested QED"),
        "evidence": {"instruments": instruments},
    }
    if not usable:
        e1["status"] = "needs_further_test"
        e1["reasoning"] = (
            "no read-out demonstrated a within-N gain on REAL molecules whose interval "
            f"excludes zero and whose magnitude reaches {INSTRUMENT_GAIN_FLOOR}. Nothing "
            "either read-out reports about generated embeddings can be read as evidence "
            "about the embeddings: the instruments have not been shown to detect the "
            "contrast even where it certainly exists.")
        e1["uncertainty"] = "unbounded: the instrument's sensitivity is unmeasured"
        e1["what_would_settle_it"] = (
            "a read-out with a demonstrated within-N gain - a richer feature set than "
            "mean/spread pooling (node-level or graph-level features), or a read-out "
            "trained on the within-N contrast directly as a paired objective")
    else:
        best = max(usable, key=lambda b: abs(b["best_gain"]))
        cells = [cell for cell in (generated_readings or [])
                 if cell.get("instrument") == best["name"]]
        resolved = [c for c in cells if c.get("status") == "resolved"]
        fractions = [c["implied_fraction_of_request"] for c in resolved
                     if _finite(c.get("implied_fraction_of_request"))]
        e1["evidence"]["instrument_used"] = best
        e1["evidence"]["cells_measured"] = len(cells)
        e1["evidence"]["cells_resolved"] = len(resolved)
        e1["evidence"]["implied_fraction_of_request"] = {
            "median": float(np.median(fractions)) if fractions else float("nan"),
            "values": fractions}
        if not cells:
            e1["status"] = "needs_further_test"
            e1["reasoning"] = "the usable read-out was never applied to generated embeddings"
            e1["uncertainty"] = "not measured"
            e1["what_would_settle_it"] = "apply the usable read-out to the paired generations"
        elif not resolved:
            e1["status"] = "inconclusive"
            e1["reasoning"] = (
                f"the read-out {best['name']} has a demonstrated gain "
                f"({best['best_gain']:+.3f} at N={best['best_gain_n']}), but on generated "
                "embeddings its paired change did not separate from the null measured on "
                "real same-band pairs. The change is below this instrument's resolution at "
                "this sample size; that is not the same as the change being zero.")
            e1["uncertainty"] = "the paired-mean interval overlaps the real-pair null interval"
            e1["what_would_settle_it"] = ("more pairs per cell, or a read-out with a larger "
                                          "within-N gain, which narrows the null")
        else:
            median = float(np.median(fractions)) if fractions else float("nan")
            if _finite(median) and median < WEAK_RESPONSE_FRACTION:
                e1["status"] = "supported"
                e1["reasoning"] = (
                    f"on the cells where {best['name']} resolves a change, the implied real "
                    f"QED difference is a median {median * 100:.0f}% of what was requested, "
                    "after dividing by the gain measured on real molecules at the same N. "
                    "The movement is real but too small to deliver the request.")
            elif _finite(median) and median >= STRONG_RESPONSE_FRACTION:
                e1["status"] = "not_supported"
                e1["reasoning"] = (
                    f"the implied real QED difference is a median {median * 100:.0f}% of the "
                    "request, so the embeddings do carry most of the requested change and "
                    "the loss is downstream of them.")
            else:
                e1["status"] = "inconclusive"
                e1["reasoning"] = (
                    f"the implied real QED difference is a median {median * 100:.0f}% of the "
                    "request - between the weak and strong bands, so the embeddings are "
                    "partly but not wholly responsible.")
            e1["uncertainty"] = ("the implied difference inherits the gain's interval; both "
                                 "are reported per cell")
            e1["what_would_settle_it"] = (
                "the same measurement at more N values with adequate low-QED support, so "
                "the implied fraction is not driven by two or three cells")
    findings["E1_generated_direction"] = e1

    # ---------------------------------------------------------------- E2 ----
    validation_rows = [row for row in transfer if row.get("split") == "validation"
                       and row.get("preservation")]
    blocks = [row["preservation"].get(headline_decoder) or {} for row in validation_rows]
    ratios = [b["ratio"] for b in blocks if _finite(b.get("ratio"))]
    uppers = [b["ci_high"] for b in blocks if _finite(b.get("ci_high"))]
    lowers = [b["ci_low"] for b in blocks if _finite(b.get("ci_low"))]
    e2: dict[str, Any] = {
        "axis": "E2_decoder_transfer",
        "claim": ("the decoder loses QED information even when it is GIVEN a real "
                  "held-out embedding carrying that difference"),
        "evidence": {"split": "validation", "decoder": headline_decoder,
                     "cells": len(validation_rows),
                     "median_preservation": float(np.median(ratios)) if ratios
                     else float("nan"),
                     "per_cell": [{"n": row["n"],
                                   "ratio": (row["preservation"].get(headline_decoder)
                                             or {}).get("ratio"),
                                   "ci": [(row["preservation"].get(headline_decoder)
                                           or {}).get("ci_low"),
                                          (row["preservation"].get(headline_decoder)
                                           or {}).get("ci_high")],
                                   "low_molecules": (row["preservation"].get(headline_decoder)
                                                     or {}).get("low_molecules"),
                                   "high_molecules": (row["preservation"].get(headline_decoder)
                                                      or {}).get("high_molecules")}
                                  for row in validation_rows]},
    }
    train_rows = [row for row in transfer if row.get("split") == "train"
                  and row.get("preservation")]
    if train_rows:
        train_ratios = [(row["preservation"].get(headline_decoder) or {}).get("ratio")
                        for row in train_rows]
        train_ratios = [r for r in train_ratios if _finite(r)]
        e2["evidence"]["median_preservation_train"] = (
            float(np.median(train_ratios)) if train_ratios else float("nan"))
        if ratios and train_ratios:
            e2["evidence"]["train_minus_validation"] = (
                float(np.median(train_ratios) - np.median(ratios)))
    if not ratios:
        e2["status"] = "needs_further_test"
        e2["reasoning"] = ("no validation cell had real molecules in both QED bands at the "
                           "same N, so the held-out control could not run")
        e2["uncertainty"] = "not measured"
        e2["what_would_settle_it"] = ("a wider QED window, or N values whose low-QED band is "
                                      "populated in the validation split")
    elif uppers and max(uppers) < TRANSFER_LOSS_CEILING:
        e2["status"] = "supported"
        e2["reasoning"] = (f"held-out preservation is below {TRANSFER_LOSS_CEILING} with the "
                           "whole interval below it in every cell: the decoder destroys most "
                           "of a difference it is handed, so no flow-side work can deliver "
                           "QED through it")
        e2["uncertainty"] = "bootstrap intervals per cell, all below the ceiling"
        e2["what_would_settle_it"] = "already decisive"
    elif lowers and min(lowers) > TRANSFER_OK_FLOOR:
        e2["status"] = "not_supported"
        e2["reasoning"] = (f"held-out preservation exceeds {TRANSFER_OK_FLOOR} with the whole "
                           "interval above it in every cell: on real embeddings the decoder "
                           "transmits the QED difference, so it is not the binding "
                           "constraint for inputs of that quality")
        e2["uncertainty"] = "bootstrap intervals per cell, all above the floor"
        e2["what_would_settle_it"] = "already decisive for real inputs"
    else:
        e2["status"] = "inconclusive"
        e2["reasoning"] = ("held-out preservation intervals straddle the decision bands, so "
                           "the decoder is neither cleared nor convicted on held-out inputs")
        e2["uncertainty"] = "intervals span the ceiling or the floor"
        e2["what_would_settle_it"] = "more held-out molecules per (N, QED) cell"
    findings["E2_decoder_transfer"] = e2

    # ---------------------------------------------------------------- E3 ----
    by_temperature = (attenuation.get("by_temperature") or {})
    attenuating = {key: entry for key, entry in by_temperature.items()
                   if entry.get("attenuating_cells")}
    channels: dict[str, int] = {}
    for entry in attenuating.values():
        for channel, hits in (entry.get("channels_implicated") or {}).items():
            channels[channel] = channels.get(channel, 0) + int(hits)
    identity_ok = all(cell.get("fallback_identity_holds") is not False
                      for entry in by_temperature.values()
                      for cell in entry.get("cells", []))
    e3: dict[str, Any] = {
        "axis": "E3_degree_attenuation",
        "claim": ("degree-constrained assembly suppresses part of the QED response that "
                  "does reach the decoder"),
        "evidence": {
            "temperatures_measured": sorted(by_temperature),
            "temperatures_with_attenuation": sorted(attenuating),
            "cells_per_temperature": {key: entry.get("cells_measured")
                                      for key, entry in by_temperature.items()},
            "attenuating_cells_per_temperature": {key: entry.get("attenuating_cells")
                                                  for key, entry in by_temperature.items()},
            "channels_implicated": channels,
            "fallback_identity_holds_everywhere": bool(identity_ok),
        },
    }
    if not by_temperature:
        e3["status"] = "needs_further_test"
        e3["reasoning"] = "no cell produced a common valid subset for both decoders"
        e3["uncertainty"] = "not measured"
        e3["what_would_settle_it"] = ("more samples per cell, or a temperature where both "
                                      "decoders reach usable validity")
    elif not identity_ok:
        e3["status"] = "needs_further_test"
        e3["reasoning"] = ("the fallback identity failed: on a fallback the degree decoder "
                           "must return exactly the constrained decoder's bonds, so the "
                           "stratified attribution cannot be trusted in this run")
        e3["uncertainty"] = "the attribution's own precondition is violated"
        e3["what_would_settle_it"] = "find the divergence before reading the decomposition"
    elif attenuating:
        e3["status"] = "supported"
        e3["reasoning"] = (
            "in " + ", ".join(f"T={key} ({entry['attenuating_cells']}/"
                              f"{entry['cells_measured']} cells)"
                              for key, entry in sorted(attenuating.items()))
            + " the constrained-minus-degree gap is positive on the COMMON valid subset "
              "with its bootstrap interval excluding zero, so this is not a validity "
              "artefact. Implicated channels: "
            + (", ".join(f"{k} x{v}" for k, v in sorted(channels.items())) or "none isolated")
            + ". Attenuation is reported per temperature because it was not universal "
              "across them.")
        e3["uncertainty"] = ("per-cell paired bootstrap intervals on the gap; cells outside "
                             "the listed temperatures showed no measurable gap")
        e3["what_would_settle_it"] = ("an assembly ablation: the same embeddings decoded with "
                                      "the degree constraint relaxed to an inequality, which "
                                      "would show how much response the hard equality costs")
    else:
        e3["status"] = "not_supported"
        e3["reasoning"] = ("no temperature produced a positive constrained-minus-degree gap "
                           "whose interval excludes zero on the common valid subset")
        e3["uncertainty"] = "per-cell paired bootstrap intervals, all spanning zero"
        e3["what_would_settle_it"] = "more pairs per cell if a small gap still matters"
    findings["E3_degree_attenuation"] = e3

    # ---------------------------------------------------------------- E4 ----
    auc = (shift or {}).get("held_out_auc")
    mahalanobis = (shift or {}).get("generated_over_real_mean")
    e4: dict[str, Any] = {
        "axis": "E4_distribution_shift",
        "claim": ("generated embeddings are shifted relative to the real ones the decoder "
                  "and the read-outs were fitted on, so both are extrapolating"),
        "evidence": {"held_out_auc": auc, "mahalanobis_generated_over_real": mahalanobis,
                     "note": ("AUC is a held-out real-vs-generated discriminability on "
                              "pooled embeddings; a Mahalanobis mean ratio near 1 does not "
                              "by itself establish that generated embeddings are "
                              "on-manifold")},
    }
    if not _finite(auc):
        e4["status"] = "needs_further_test"
        e4["reasoning"] = "the discriminability could not be computed"
        e4["uncertainty"] = "not measured"
        e4["what_would_settle_it"] = "more real and generated molecules on both sides"
    elif auc >= SHIFT_AUC_SUPPORTED:
        e4["status"] = "supported"
        e4["reasoning"] = (
            f"a cheap held-out classifier separates real from generated pooled embeddings at "
            f"AUC {auc:.3f}"
            + (f" even though the Mahalanobis mean ratio is {mahalanobis:.2f}"
               if _finite(mahalanobis) else "")
            + ": the two clouds are distinguishable, so every read-out and the decoder are "
              "extrapolating on generated inputs and their numbers there are bounded by that")
        e4["uncertainty"] = "a single held-out split; the AUC is a point estimate"
        e4["what_would_settle_it"] = ("repeated splits for an interval, and a decoder-side "
                                      "check: decode real embeddings perturbed towards the "
                                      "generated cloud and watch where QED transfer breaks")
    elif auc <= SHIFT_AUC_CLEAN:
        e4["status"] = "not_supported"
        e4["reasoning"] = (f"AUC {auc:.3f} is close to chance, so under these features the "
                           "generated embeddings are not distinguishable from the real ones")
        e4["uncertainty"] = "a single held-out split"
        e4["what_would_settle_it"] = ("a richer feature set, which could still separate them; "
                                      "this clears them only under pooled features")
    else:
        e4["status"] = "inconclusive"
        e4["reasoning"] = (f"AUC {auc:.3f} sits between chance and clear separation: some "
                           "shift, not enough to call either way")
        e4["uncertainty"] = "a single held-out split in the ambiguous band"
        e4["what_would_settle_it"] = "repeated splits, and the perturbation check above"
    findings["E4_distribution_shift"] = e4

    summary = {
        "supported": [key for key, value in findings.items() if value["status"] == "supported"],
        "not_supported": [key for key, value in findings.items()
                          if value["status"] == "not_supported"],
        "inconclusive": [key for key, value in findings.items()
                         if value["status"] == "inconclusive"],
        "needs_further_test": [key for key, value in findings.items()
                               if value["status"] == "needs_further_test"],
    }
    return {
        "findings": findings,
        "summary": summary,
        "thresholds": {
            "instrument_gain_floor": INSTRUMENT_GAIN_FLOOR,
            "weak_response_fraction": WEAK_RESPONSE_FRACTION,
            "strong_response_fraction": STRONG_RESPONSE_FRACTION,
            "transfer_loss_ceiling": TRANSFER_LOSS_CEILING,
            "transfer_ok_floor": TRANSFER_OK_FLOOR,
            "shift_auc_supported": SHIFT_AUC_SUPPORTED,
            "shift_auc_clean": SHIFT_AUC_CLEAN,
        },
        "note": ("the four axes are graded independently and can all be true at once. "
                 "'needs_further_test' is a statement about the instrument, never evidence "
                 "about the model."),
    }


def recommend_intervention(evidence: dict[str, Any]) -> dict[str, Any]:
    """Compare A, B and C on benefit, risk and cost, and name one - with its preconditions.

    This is a recommendation, not a change. Nothing in stage 20 edits the architecture.
    """
    findings = evidence.get("findings") or {}

    def status(axis: str) -> str:
        return (findings.get(axis) or {}).get("status", "needs_further_test")

    e1, e2, e3, e4 = (status(axis) for axis in AXES)
    options = {
        "A_flow": {
            "name": "improve the conditional flow's QED-predictive transformation",
            "addresses": ["E1_generated_direction"],
            "expected_benefit": ("directly targets the displacement's DIRECTION rather than "
                                 "its size, which stage 18 already showed is large. Options "
                                 "range from a QED-predictive auxiliary objective on the "
                                 "flow's output embeddings to the gated stage B already "
                                 "implemented in scripts/17_structure_finetune.py"),
            "risks": ("stages 16 and 17 both showed conditioning trading away structure: "
                      "aromatic rings fell to ~0.35 and raw validity to ~9% when the "
                      "conditional objective competed with the generator. Any flow-side "
                      "change needs the structure guardrail and a matched control"),
            "compute_cost": ("moderate: one GPU run of a few thousand steps per arm plus a "
                             "matched control and evaluation, in the range of the stage-17 "
                             "and stage-18 jobs (8 h GPU budget each)"),
            "reversible": True,
        },
        "B_decoder": {
            "name": "improve bond decoding and constraint-aware graph assembly",
            "addresses": ["E3_degree_attenuation", "E2_decoder_transfer"],
            "expected_benefit": ("acts on a loss that is measured per molecule on identical "
                                 "embeddings, so the effect size is known before any model "
                                 "is retrained. The natural first step costs no training at "
                                 "all: relax the degree equality to an inequality, or rank "
                                 "candidate assemblies instead of committing to one integer "
                                 "program solution"),
            "risks": ("the degree constraint is what currently buys validity and acts as the "
                      "symmetry tie-breaker; relaxing it may cost raw validity and "
                      "connectivity, so it must be evaluated against the stage-17 structure "
                      "numbers, not in isolation"),
            "compute_cost": ("low: the assembly is CPU work on embeddings that already "
                             "exist. A decoder-head change would add one autoencoder "
                             "fine-tune, but the first ablation needs none"),
            "reversible": True,
        },
        "C_representation": {
            "name": ("replace the embedding-based representation with a direct conditional "
                     "atom/bond flow (MoFlow-style)"),
            "addresses": ["E2_decoder_transfer", "E4_distribution_shift"],
            "expected_benefit": ("removes the learned-embedding bottleneck entirely: the "
                                 "property would condition a DISCRETE graph variable rather "
                                 "than a continuous geometry that a separately trained "
                                 "decoder has to invert"),
            "risks": ("the largest change in the project and the least reversible: a new "
                      "model, new training, and the loss of the unconditional Arm B "
                      "generator's structure quality as a starting point. It should not be "
                      "started on an inconclusive measurement"),
            "compute_cost": ("high: a reimplementation plus full training runs, far beyond "
                             "the stage 17-20 budget"),
            "reversible": False,
        },
    }

    if e2 == "supported" and e4 == "supported":
        primary, confidence = "C_representation", "moderate"
        rationale = ("the decoder loses a QED difference it is GIVEN on held-out real "
                     "embeddings AND the generated embeddings are measurably shifted from "
                     "the ones it was fitted on. Both point past the flow: the embedding "
                     "representation is the binding constraint, which is the condition under "
                     "which a direct atom/bond flow is the right answer rather than an "
                     "expensive detour.")
    elif e2 == "supported":
        primary, confidence = "B_decoder", "moderate"
        rationale = ("the decoder fails to transmit a QED difference on held-out real "
                     "embeddings, so flow-side work cannot deliver QED through it. Decoder "
                     "and assembly work is the targeted response, and it is where the "
                     "evidence points before the much larger step C.")
    elif e3 == "supported" and e1 == "supported":
        primary, confidence = "B_decoder", "high"
        rationale = ("BOTH mechanisms are supported: the embeddings carry less QED-predictive "
                     "movement than requested, and degree-constrained assembly then "
                     "suppresses part of what does arrive. They are not exclusive, and the "
                     "assembly loss is the cheaper of the two to act on and is measured on "
                     "identical embeddings - so B first, then A, measuring the flow against "
                     "a decoder that no longer erases the response. Acting on A first risks "
                     "attributing a decoder loss to the flow.")
    elif e3 == "supported":
        primary, confidence = "B_decoder", "high"
        rationale = ("degree-constrained assembly attenuates a response that does reach the "
                     "decoder, while the embeddings were not shown to be the weak link. The "
                     "loss is downstream of the flow, so the flow is the wrong thing to "
                     "change.")
    elif e1 == "supported":
        primary, confidence = "A_flow", "moderate"
        rationale = ("the generated embeddings move, but the implied real QED difference is "
                     "well below the request, while the decoder transmits differences it is "
                     "given and assembly shows no measurable attenuation. The deficit is in "
                     "the flow's conditional transformation.")
    elif e1 == "needs_further_test":
        primary, confidence = "B_decoder", "low"
        rationale = ("no read-out has a demonstrated within-N gain, so the embedding axis is "
                     "UNMEASURED rather than cleared - A cannot be justified on this "
                     "evidence and C certainly cannot. The work that is justified is the "
                     "decoder-side ablation, which needs no probe at all because it compares "
                     "two decoders on identical embeddings. Fixing the instrument is the "
                     "precondition for ever recommending A or C.")
    else:
        primary, confidence = "B_decoder", "low"
        rationale = ("no axis is supported at the required precision. The decoder-side "
                     "ablation is the cheapest way to convert an inconclusive axis into a "
                     "measured one, and it requires no training.")

    return {
        "primary": primary,
        "confidence": confidence,
        "rationale": rationale,
        "axis_status": {axis: status(axis) for axis in AXES},
        "options": options,
        "preconditions": (
            ["restore a read-out with a demonstrated within-N gain before any flow-side or "
             "representation-level decision: an unmeasured axis is not a cleared axis"]
            if e1 == "needs_further_test" else
            ["re-run the decoder ablation with the degree equality relaxed, to size the "
             "response the hard constraint costs, before committing engineering effort"]),
        "what_would_change_this": [
            "E2 turning supported on held-out data moves the recommendation towards C",
            "E3 turning not_supported at every temperature, with E1 supported, moves it to A",
            "E1 becoming not_supported while E3 stays supported strengthens B further",
            "E4 supported together with E2 supported is the only combination that justifies C",
        ],
        "not_implemented_note": ("a recommendation only: stage 20 changes no architecture, "
                                 "trains no flow and overwrites no checkpoint"),
    }


def format_report(payload: dict[str, Any]) -> str:
    """The human-readable report. Tables first, findings second, recommendation last."""
    lines: list[str] = ["Stage 20: QED pathway audit - conditional signal and decoder "
                        "constraints", ""]
    flow = payload.get("flow") or {}
    lines += [f"flow checkpoint : {flow.get('checkpoint')} (step {flow.get('step')})",
              f"sha256          : {flow.get('sha256')}",
              f"grid            : N={payload.get('target_n')} QED {payload.get('qed_pair')} "
              f"T={payload.get('temperatures')} x {payload.get('count')} samples", ""]

    lines += ["=" * 78,
              "A. decoder QED transfer on REAL molecules - does the decoder preserve a "
              "difference it is GIVEN?",
              "=" * 78,
              f"{'split':<12}{'N':>4}{'decoder':>13}{'preserved':>11}"
              f"{'95% CI':>20}{'low n':>7}{'high n':>7}{'arom.':>8}{'valid':>8}"]
    for row in payload.get("transfer", []):
        for name, block in (row.get("preservation") or {}).items():
            group = ((row.get("groups") or {}).get("high") or {}).get("decoders", {})
            cell = group.get(name) or {}
            interval = (f"[{block.get('ci_low'):+.3f}, {block.get('ci_high'):+.3f}]"
                        if block.get("ci_low") is not None
                        and np.isfinite(block.get("ci_low", float('nan'))) else "n/a")
            lines.append(
                f"{row.get('split', '?'):<12}{row.get('n', 0):>4}{name:>13}"
                f"{block.get('ratio', float('nan')):>11.3f}{interval:>20}"
                f"{block.get('low_molecules', 0):>7}{block.get('high_molecules', 0):>7}"
                f"{cell.get('aromatic_rings_mean', float('nan')):>8.2f}"
                f"{cell.get('validity', float('nan')):>8.3f}")
    lines += ["", "preservation = (decoded high - decoded low) / (true high - true low), "
                  "bootstrapped over molecules.",
              "The validation rows are the ones that matter: stage 19 measured this on "
              "TRAINING molecules only.", ""]

    lines += ["=" * 78,
              "B. the read-outs - can either instrument SEE a within-N QED change?",
              "=" * 78,
              f"{'read-out':<18}{'val MAE':>9}{'val corr':>10}{'calib slope':>13}"
              f"{'N':>5}{'gain':>9}{'gain 95% CI':>22}{'null sd':>10}{'pairs':>7}"]
    for readout in payload.get("readouts", []):
        validation = readout.get("validation") or {}
        calibration_block = readout.get("calibration") or {}
        for block in readout.get("reference_contrasts") or []:
            null = next((b for b in readout.get("null_contrasts") or []
                         if b.get("n") == block.get("n")), {})
            interval = (f"[{block.get('gain_ci_low'):+.3f}, {block.get('gain_ci_high'):+.3f}]"
                        if _finite(block.get("gain_ci_low")) else "n/a")
            lines.append(
                f"{readout.get('name', '?'):<18}{validation.get('mae', float('nan')):>9.4f}"
                f"{validation.get('correlation', float('nan')):>10.4f}"
                f"{calibration_block.get('slope_of_truth_on_prediction', float('nan')):>13.3f}"
                f"{block.get('n', 0):>5}{block.get('gain', float('nan')):>9.3f}"
                f"{interval:>22}{null.get('null_sd', float('nan')):>10.4f}"
                f"{block.get('pairs', 0):>7}")
    lines += ["", "gain = predicted / true difference on REAL molecules at that N. It is the "
                  "instrument's",
              "sensitivity to exactly the contrast the intervention produces. null sd is the "
              "spread of the",
              "predicted difference between real molecules in the SAME QED band - the "
              "detection threshold for",
              "a paired mean. A single-prediction held-out MAE is NOT that threshold.", ""]

    lines += ["=" * 78,
              "C. generated embeddings - resolved change and implied real QED difference",
              "=" * 78,
              f"{'read-out':<18}{'T':>5}{'N':>5}{'paired change':>15}{'95% CI':>22}"
              f"{'vs null':>20}{'implied':>9}{'% of ask':>10}"]
    for cell in payload.get("generated_readings") or []:
        observed = cell.get("observed") or {}
        interval = (f"[{observed.get('ci_low'):+.4f}, {observed.get('ci_high'):+.4f}]"
                    if _finite(observed.get("ci_low")) else "n/a")
        fraction = cell.get("implied_fraction_of_request")
        lines.append(
            f"{cell.get('instrument', '?'):<18}{cell.get('temperature', 0):>5.2g}"
            f"{cell.get('n', 0):>5}{observed.get('mean', float('nan')):>15.4f}"
            f"{interval:>22}{str(cell.get('status')):>20}"
            f"{cell.get('implied_real_qed_difference', float('nan')):>9.3f}"
            + (f"{fraction * 100:>9.0f}%" if _finite(fraction) else f"{'n/a':>10}"))
    lines += ["", "'not_resolved' means this instrument cannot see the change at this sample "
                  "size. It is not",
              "evidence that the embeddings carry no QED information.", ""]

    lines += ["=" * 78,
              "D. constrained vs degree-constrained decoding of IDENTICAL embeddings",
              "=" * 78,
              f"{'T':>5}{'N':>5}{'common':>8}{'constrained':>13}{'degree':>10}"
              f"{'gap':>9}{'gap 95% CI':>22}{'deg pred move':>15}{'fallback':>10}"]
    for key, entry in sorted(((payload.get("attenuation") or {}).get("by_temperature")
                              or {}).items()):
        for cell in entry.get("cells", []):
            interval = (f"[{cell['gap_ci'][0]:+.4f}, {cell['gap_ci'][1]:+.4f}]"
                        if cell.get("gap_ci") and _finite(cell["gap_ci"][0]) else "n/a")
            lines.append(
                f"{float(key):>5.2g}{cell.get('n', 0):>5}"
                f"{cell.get('common_molecules', 0):>8}"
                f"{cell.get('constrained_change', float('nan')):>13.4f}"
                f"{cell.get('degree_change', float('nan')):>10.4f}"
                f"{cell.get('gap', float('nan')):>9.4f}{interval:>22}"
                f"{cell.get('degree_prediction_change_rate', float('nan')):>15.3f}"
                f"{cell.get('degree_fallback_rate_high', float('nan')):>10.3f}")
        channels = entry.get("channels_implicated") or {}
        if channels:
            lines.append(f"  T={key} channels implicated: "
                         + ", ".join(f"{k} x{v}" for k, v in sorted(channels.items())))
    lines += ["", "Both decoders are scored on ONE common subset of molecules valid under "
                  "both conditions and",
              "both decoders, so a gap cannot be an artefact of different survivors. On a "
              "fallback the degree",
              "decoder returns exactly the constrained decoder's bonds, so the fallback "
              "stratum contributes",
              "exactly zero to the gap - which is what makes the attribution below "
              "arithmetic rather than opinion.", ""]

    lines += ["=" * 78, "E. support - every count with its denominator", "=" * 78]
    for row in payload.get("support", []):
        parts = ", ".join(f"QED {key}: {value['molecules_within_window']:,}"
                          for key, value in (row.get("by_qed") or {}).items())
        lines.append(f"  {row.get('split', 'train'):<11} N={row['n']:<3} "
                     f"{row['molecules_at_this_n']:,} molecules at this N; {parts}; "
                     f"minimum {row['minimum_over_qed_values']:,}")
    lines += [""]

    lines += ["=" * 78, "FINDINGS - four independent axes, graded separately", "=" * 78]
    for axis, finding in (payload.get("evidence") or {}).get("findings", {}).items():
        lines += [f"[{finding['status'].upper()}] {axis}",
                  f"  claim    : {finding['claim']}",
                  f"  reasoning: {finding['reasoning']}",
                  f"  stated uncertainty: {finding.get('uncertainty')}",
                  f"  would settle it   : {finding.get('what_would_settle_it')}", ""]
    summary = (payload.get("evidence") or {}).get("summary") or {}
    for key in ("supported", "inconclusive", "needs_further_test", "not_supported"):
        lines.append(f"{key:>20}: {', '.join(summary.get(key) or []) or '-'}")
    lines += ["", "No axis suppresses another: more than one mechanism can be, and here may "
                  "be, true at once.", ""]

    recommendation = payload.get("recommendation") or {}
    lines += ["=" * 78,
              f"RECOMMENDATION: {recommendation.get('primary')} "
              f"(confidence: {recommendation.get('confidence')})",
              "=" * 78,
              recommendation.get("rationale", ""), ""]
    for key, option in (recommendation.get("options") or {}).items():
        marker = ">>" if key == recommendation.get("primary") else "  "
        lines += [f"{marker} {key}: {option['name']}",
                  f"     benefit: {option['expected_benefit']}",
                  f"     risks  : {option['risks']}",
                  f"     cost   : {option['compute_cost']}",
                  f"     reversible: {option['reversible']}", ""]
    lines += ["preconditions:"] + [f"  - {item}" for item in
                                   recommendation.get("preconditions") or []]
    lines += ["", "what would change this:"] + [f"  - {item}" for item in
                                                recommendation.get("what_would_change_this")
                                                or []]
    lines += ["", recommendation.get("not_implemented_note", ""), ""]
    return "\n".join(lines)


__all__ = [
    "AXES",
    "INSTRUMENT_GAIN_FLOOR",
    "SHIFT_AUC_CLEAN",
    "SHIFT_AUC_SUPPORTED",
    "STRONG_RESPONSE_FRACTION",
    "TRANSFER_LOSS_CEILING",
    "TRANSFER_OK_FLOOR",
    "WEAK_RESPONSE_FRACTION",
    "assess_evidence",
    "format_report",
    "recommend_intervention",
]
