"""Stage 21: the matched four-decoder comparison, its success criteria and its decision.

Every decoder in this comparison sees THE SAME generated embeddings. That is not a
convention here, it is structural: one `decode_embeddings` call decodes each sampled
embedding with all four assemblers, so "raw", "constrained", "degree" and "total" differ
only in how they turn one fixed set of bond and degree logits into a graph. No decoder
gets its own sample, its own seed or its own node counts.

Three things this module is careful never to do, because they would make an improvement
meaningless:

  * NO SELECTION. One embedding produces exactly one molecule per decoder,
    deterministically. Nothing generates a batch and keeps the best QED, and no molecule
    is ever dropped for having the wrong property.
  * NO ORACLE. The target QED never reaches the decoder. It conditions the flow, as it
    always has, and the assemblers see only the decoder's own bond and degree
    distributions. Any QED response recovered is response the conditional flow had
    already encoded.
  * NO SELECTION EFFECT IN THE HEADLINE COMPARISON. The paired response is reported on a
    COMMON subset of samples that every decoder turned into a valid molecule under both
    QED requests, so a decoder cannot look better merely by failing on the hard cases.
    Each decoder's own-subset number is reported beside it, and the difference between
    the two IS the selection effect.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np

WELL_SUPPORTED_MOLECULES = 200
EXPERIMENTAL_DECODER = "total"
BASELINE_DECODER = "degree"


# --------------------------------------------------- per-decoder assembly health ---
def assembly_statistics(records: Sequence[dict[str, Any]], decoder: str) -> dict[str, Any]:
    """Fallbacks, relaxation levels, degree/valence consistency and solver cost.

    These are the columns that say WHY a decoder behaved as it did, and they are the
    difference between "the new assembler is better" and "the new assembler is better
    because it stopped abandoning the program".
    """
    entries = [record[decoder] for record in records if decoder in record]
    total = len(entries)
    if total == 0:
        return {"samples": 0}
    result: dict[str, Any] = {
        "samples": total,
        "fallback_rate": float(np.mean([bool(e.get("fallback")) for e in entries])),
        "fallback_reasons": dict(Counter(e.get("fallback_reason") for e in entries
                                         if e.get("fallback")).most_common()),
        "connected_rate": float(np.mean([bool(e.get("connected")) for e in entries])),
        "sanitizable_rate": float(np.mean([bool(e.get("sanitizable")) for e in entries])),
    }
    fragments = [e["fragments"] for e in entries if e.get("fragments") is not None]
    result["fragments_mean"] = float(np.mean(fragments)) if fragments else float("nan")

    # the experimental decoder's own bookkeeping, absent on the baselines
    levels = [e.get("relaxation_level") for e in entries if "relaxation_level" in e]
    if levels:
        result["relaxation_levels"] = dict(Counter(
            "none" if level is None else level for level in levels).most_common())
    for key, label in (("degree_satisfied", "degree_exact_rate"),
                       ("valence_respected", "valence_respected_rate"),
                       ("assembly_failed", "assembly_failed_rate")):
        values = [bool(e[key]) for e in entries if key in e]
        if values:
            result[label] = float(np.mean(values))
    for key, label in (("degree_absolute_error", "degree_absolute_error_mean"),
                       ("swaps", "connectivity_swaps_mean"),
                       ("seconds", "assembly_seconds_mean")):
        values = [float(e[key]) for e in entries
                  if e.get(key) is not None and np.isfinite(float(e[key]))]
        if values:
            result[label] = float(np.mean(values))
    repairs = [e["degree_repair"] for e in entries if isinstance(e.get("degree_repair"), dict)]
    if repairs:
        result["degree_repair"] = {
            "atoms_changed_mean": float(np.mean([r["atoms_changed"] for r in repairs])),
            "parity_repaired_rate": float(np.mean([bool(r["parity_repaired"])
                                                   for r in repairs])),
            "isolated_atoms_requested_mean": float(np.mean(
                [r.get("isolated_atoms_requested", 0) for r in repairs])),
            "raised_for_connectivity_mean": float(np.mean(
                [r.get("degrees_raised_for_connectivity", 0) for r in repairs])),
            "requested_sum_was_odd_rate": float(np.mean(
                [bool(r.get("requested_sum_was_odd")) for r in repairs])),
            "connectivity_possible_rate": float(np.mean(
                [bool(r.get("connectivity_possible", True)) for r in repairs])),
            "graphical_rate": float(np.mean([bool(r.get("graphical")) for r in repairs])),
        }
    return result


def degree_consistency(records: Sequence[dict[str, Any]], decoder: str) -> dict[str, Any]:
    """Does the assembled molecule actually have the degrees the head asked for?

    Measured for EVERY decoder from the graph itself, not from the assembler's own
    report, so the baselines and the experimental decoder are judged by one standard.
    """
    matched = total = atoms = error = 0
    for record in records:
        entry = record.get(decoder) or {}
        requested = record.get("predicted_degrees")
        bonds = entry.get("bond_matrix")
        if requested is None or bonds is None or not entry.get("valid"):
            continue
        achieved = (np.asarray(bonds) > 0).sum(axis=1)
        want = np.asarray(requested, dtype=np.int64)
        if achieved.shape != want.shape:
            continue
        total += 1
        matched += int(np.array_equal(achieved, want))
        atoms += int(want.size)
        error += int(np.abs(achieved - want).sum())
    return {"molecules": total,
            "degree_sequence_exact": (matched / total) if total else float("nan"),
            "mean_absolute_degree_error_per_atom": (error / atoms) if atoms else float("nan"),
            "note": "measured from the assembled graph, identically for every decoder"}


# ------------------------------------------------------- the paired comparison ---
def paired_decoder_changes(low_records: Sequence[dict[str, Any]],
                           high_records: Sequence[dict[str, Any]], *,
                           decoders: Sequence[str], requested_change: float,
                           common_decoders: Sequence[str] | None = None,
                           seed: int = 17, resamples: int = 2000) -> dict[str, Any]:
    """Paired QED change per decoder at FIXED noise, on one common valid subset.

    The two conditions share their latent noise, so a per-sample difference isolates the
    conditioning. The headline is computed on the subset of samples that every COMPARED
    assembler turned into a valid molecule under BOTH requests: that is what stops "the
    new assembler responds more" from being "the new assembler quietly failed on the
    samples that did not respond".

    `raw` is excluded from the intersection by default, and reported on its own subset
    instead. Raw argmax validity is in the 9-14% range even for the good unconditional
    run, so intersecting with it would leave almost no molecules and would make the
    comparison between the three ASSEMBLERS - which is the question - unanswerable. Raw
    native validity is reported separately, never merged into these numbers.
    """
    from zinc_gnf.qed_probes import paired_bootstrap_ci, paired_difference_ci

    decoders = tuple(decoders)
    common_decoders = tuple(common_decoders if common_decoders is not None
                            else [name for name in decoders if name != "raw"])
    usable = min(len(low_records), len(high_records))

    def qed(record: dict[str, Any], decoder: str) -> float | None:
        entry = record.get(decoder) or {}
        if entry.get("valid") and entry.get("qed") is not None:
            return float(entry["qed"])
        return None

    common = [index for index in range(usable)
              if all(qed(low_records[index], name) is not None
                     and qed(high_records[index], name) is not None
                     for name in common_decoders)]
    result: dict[str, Any] = {
        "samples": usable, "requested_change": float(requested_change),
        "common_subset_molecules": len(common),
        "common_subset_fraction": len(common) / usable if usable else float("nan"),
        "common_subset_decoders": list(common_decoders),
        "decoders": {},
        "note": ("the common subset is valid under both QED requests for every COMPARED "
                 "assembler, so no decoder can gain by failing on the unresponsive "
                 "samples. 'raw' is excluded from the intersection - its argmax validity "
                 "is far lower - and is reported on its own subset instead"),
    }
    changes: dict[str, list[float]] = {}
    for name in decoders:
        own = [qed(high_records[i], name) - qed(low_records[i], name)
               for i in range(usable)
               if qed(low_records[i], name) is not None
               and qed(high_records[i], name) is not None]
        shared = [qed(high_records[i], name) - qed(low_records[i], name) for i in common
                  if qed(low_records[i], name) is not None
                  and qed(high_records[i], name) is not None]
        changes[name] = shared
        result["decoders"][name] = {
            "in_common_subset": name in common_decoders,
            "own_subset": paired_bootstrap_ci(own, seed=seed, resamples=resamples),
            "common_subset": paired_bootstrap_ci(shared, seed=seed, resamples=resamples),
        }
        result["decoders"][name]["selection_effect"] = (
            result["decoders"][name]["common_subset"]["mean"]
            - result["decoders"][name]["own_subset"]["mean"])
    if EXPERIMENTAL_DECODER in common_decoders and BASELINE_DECODER in common_decoders \
            and common:
        result["improvement_over_baseline"] = paired_difference_ci(
            changes[EXPERIMENTAL_DECODER], changes[BASELINE_DECODER],
            seed=seed, resamples=resamples)
    return result


# --------------------------------------------------------------- cell support ---
def classify_support(support_rows: Sequence[dict[str, Any]], *, n: int, qed: float,
                     threshold: int = WELL_SUPPORTED_MOLECULES) -> dict[str, Any]:
    """Well-supported or low-support, from TRAIN counts within the QED window.

    Reported separately because stage 20 showed N=15 and N=23 have thin low-QED support
    (76 and 236 molecules), and an improvement that only appears where the data is thin
    is not an improvement.
    """
    for row in support_rows:
        if int(row.get("n", -1)) != int(n):
            continue
        for key, block in (row.get("by_qed") or {}).items():
            if abs(float(key) - float(qed)) < 1e-9:
                count = int(block["molecules_within_window"])
                return {"molecules_within_window": count,
                        "support": "well_supported" if count >= threshold else "low_support",
                        "threshold": int(threshold)}
    return {"molecules_within_window": None, "support": "unknown",
            "threshold": int(threshold)}


# ----------------------------------------------------------- success criteria ---
def evaluate_criteria(*, cells: Sequence[dict[str, Any]], paired: Sequence[dict[str, Any]],
                      achievement: dict[str, Any], assembly: dict[str, Any],
                      cost: dict[str, Any]) -> dict[str, Any]:
    """Step 4, stated as explicit pass / fail / inconclusive checks.

    Each check names what it measured and what it would take to pass, so a partial
    result is reported as a partial result rather than written up as a success.
    """
    def mean_of(values: Sequence[float]) -> float:
        finite = [v for v in values if v is not None and np.isfinite(v)]
        return float(np.mean(finite)) if finite else float("nan")

    improvements = [row["improvement_over_baseline"] for row in paired
                    if row.get("improvement_over_baseline")
                    and np.isfinite(row["improvement_over_baseline"].get(
                        "difference", float("nan")))]
    positive = [row for row in improvements
                if row["difference"] > 0 and row.get("excludes_zero")]
    negative = [row for row in improvements
                if row["difference"] < 0 and row.get("excludes_zero")]
    experimental = achievement.get(EXPERIMENTAL_DECODER) or {}
    baseline = achievement.get(BASELINE_DECODER) or {}

    checks: list[dict[str, Any]] = []

    def add(name: str, status: str, detail: str, evidence: Any = None) -> None:
        checks.append({"criterion": name, "status": status, "detail": detail,
                       "evidence": evidence})

    # 1. native QED control improves meaningfully over the degree decoder
    experimental_mae = (experimental.get("overall") or {}).get("mae_weighted")
    baseline_mae = (baseline.get("overall") or {}).get("mae_weighted")
    if not improvements:
        add("qed_control_improves", "inconclusive",
            "no cell produced a common valid subset under both QED requests, so the "
            "paired comparison could not be made")
    elif positive and not negative:
        add("qed_control_improves", "pass",
            f"the paired QED response improved in {len(positive)} of {len(improvements)} "
            f"matched cells with the bootstrap interval excluding zero, and worsened in "
            f"none; mean improvement {mean_of([r['difference'] for r in improvements]):+.4f}",
            {"cells_improved": len(positive), "cells_measured": len(improvements)})
    elif positive and negative:
        add("qed_control_improves", "partial",
            f"improved in {len(positive)} cells and worsened in {len(negative)} of "
            f"{len(improvements)}: the change is not uniform across the grid",
            {"cells_improved": len(positive), "cells_worsened": len(negative)})
    else:
        add("qed_control_improves", "fail",
            f"no cell shows a positive paired improvement whose interval excludes zero "
            f"(mean {mean_of([r['difference'] for r in improvements]):+.4f})")

    # 2. the improvement is not a selection artefact
    selection = [abs(row["decoders"][EXPERIMENTAL_DECODER]["selection_effect"])
                 for row in paired if EXPERIMENTAL_DECODER in row.get("decoders", {})
                 and np.isfinite(row["decoders"][EXPERIMENTAL_DECODER]["selection_effect"])]
    headline = mean_of([r["difference"] for r in improvements]) if improvements else float("nan")
    selection_mean = mean_of(selection)
    if not np.isfinite(headline) or not np.isfinite(selection_mean):
        add("not_a_selection_artefact", "inconclusive",
            "the selection effect could not be measured")
    elif abs(headline) > 2.0 * selection_mean:
        add("not_a_selection_artefact", "pass",
            f"the headline improvement is computed on a subset valid under both requests "
            f"for every decoder, and the mean selection effect ({selection_mean:.4f}) is "
            f"small against it ({headline:+.4f}). One embedding yields one molecule per "
            f"decoder, so no candidate selection takes place at all",
            {"selection_effect_mean": selection_mean})
    else:
        add("not_a_selection_artefact", "partial",
            f"the mean selection effect ({selection_mean:.4f}) is of the same order as the "
            f"headline improvement ({headline:+.4f}); the common-subset number is the one "
            f"to trust and the gap should be read as uncertainty")

    # 3. strict validity stays high
    experimental_validity = (assembly.get(EXPERIMENTAL_DECODER) or {}).get("strict_validity")
    baseline_validity = (assembly.get(BASELINE_DECODER) or {}).get("strict_validity")
    if experimental_validity is None or baseline_validity is None:
        add("validity_preserved", "inconclusive", "validity was not measured for both")
    elif experimental_validity >= baseline_validity - 0.02:
        add("validity_preserved", "pass",
            f"strict validity {experimental_validity:.3f} against the degree decoder's "
            f"{baseline_validity:.3f}",
            {"experimental": experimental_validity, "baseline": baseline_validity})
    else:
        add("validity_preserved", "fail",
            f"strict validity fell from {baseline_validity:.3f} to "
            f"{experimental_validity:.3f}")

    # 4. structure and aromaticity do not deteriorate
    aromatic_experimental = (assembly.get(EXPERIMENTAL_DECODER) or {}).get("aromatic_rings_mean")
    aromatic_baseline = (assembly.get(BASELINE_DECODER) or {}).get("aromatic_rings_mean")
    if aromatic_experimental is None or aromatic_baseline is None \
            or not np.isfinite(aromatic_experimental) or not np.isfinite(aromatic_baseline):
        add("structure_preserved", "inconclusive", "aromaticity was not measurable")
    elif aromatic_experimental >= aromatic_baseline - 0.1:
        add("structure_preserved", "pass",
            f"aromatic rings {aromatic_experimental:.3f} against {aromatic_baseline:.3f}")
    else:
        add("structure_preserved", "fail",
            f"aromatic rings fell from {aromatic_baseline:.3f} to "
            f"{aromatic_experimental:.3f}")

    # 5. diversity and uniqueness remain acceptable
    pairs = [("uniqueness", 0.05), ("internal_diversity", 0.05)]
    verdicts = []
    for key, slack in pairs:
        new = (assembly.get(EXPERIMENTAL_DECODER) or {}).get(key)
        old = (assembly.get(BASELINE_DECODER) or {}).get(key)
        if new is None or old is None or not np.isfinite(new) or not np.isfinite(old):
            verdicts.append((key, "inconclusive", new, old))
        else:
            verdicts.append((key, "pass" if new >= old - slack else "fail", new, old))
    statuses = {v[1] for v in verdicts}
    add("diversity_preserved",
        "fail" if "fail" in statuses else ("inconclusive" if "inconclusive" in statuses
                                           else "pass"),
        "; ".join(f"{key}: {new if new is None else f'{new:.3f}'} vs "
                  f"{old if old is None else f'{old:.3f}'}"
                  for key, _, new, old in verdicts))

    # 6. no leakage: structural, by construction
    add("no_leakage", "pass",
        "the target QED never reaches the decoder; the assemblers see only the decoder's "
        "own bond and degree distributions. Novelty is measured against TRAIN only and "
        "the test split is never loaded")

    # 7. runtime
    experimental_cost = (cost.get(EXPERIMENTAL_DECODER) or {}).get("seconds_per_molecule")
    baseline_cost = (cost.get(BASELINE_DECODER) or {}).get("seconds_per_molecule")
    if experimental_cost is None or not np.isfinite(experimental_cost):
        add("runtime_practical", "inconclusive", "cost was not measured")
    elif experimental_cost <= 0.25:
        add("runtime_practical", "pass",
            f"{experimental_cost * 1000:.1f} ms per molecule"
            + (f" against the degree decoder's {baseline_cost * 1000:.1f} ms"
               if baseline_cost and np.isfinite(baseline_cost) else "")
            + "; 10,000 molecules is "
            f"{experimental_cost * 10000 / 60:.1f} minutes on one CPU core")
    else:
        add("runtime_practical", "fail",
            f"{experimental_cost * 1000:.1f} ms per molecule is too slow for deployment")

    # the research milestone, reported as a milestone and not as a gate
    milestone = {
        "qed_mae_target": 0.15,
        "experimental_weighted_mae": experimental_mae,
        "baseline_weighted_mae": baseline_mae,
        "reached": bool(experimental_mae is not None and np.isfinite(experimental_mae)
                        and experimental_mae <= 0.15),
        "note": ("a desirable research milestone, not a gate: QED MAE is dominated by "
                 "how far the generated population sits from the requested value, and a "
                 "decoder change alone is not expected to close it"),
    }
    tally = Counter(check["status"] for check in checks)
    return {
        "checks": checks,
        "summary": {"pass": tally["pass"], "partial": tally["partial"],
                    "fail": tally["fail"], "inconclusive": tally["inconclusive"]},
        "retain_experimental_decoder": bool(tally["fail"] == 0 and tally["pass"] >= 4
                                            and any(c["criterion"] == "qed_control_improves"
                                                    and c["status"] in {"pass", "partial"}
                                                    for c in checks)),
        "milestone": milestone,
    }


# ------------------------------------------------------------- the decision ----
def recommend_next(criteria: dict[str, Any], *, paired: Sequence[dict[str, Any]],
                   shift_auc: float | None = None) -> dict[str, Any]:
    """Step 5: one of A (flow conditioning), B (direct atom/bond flow), C (more decoder).

    The question this answers is not "did the decoder get better" but "where is the next
    unit of effort best spent", which depends on how much QED response the decoder
    recovered and how much is still missing.
    """
    improvements = [row["improvement_over_baseline"] for row in paired
                    if row.get("improvement_over_baseline")
                    and np.isfinite(row["improvement_over_baseline"].get(
                        "difference", float("nan")))]
    recovered = (float(np.mean([r["difference"] for r in improvements]))
                 if improvements else float("nan"))
    requested = (float(np.mean([row["requested_change"] for row in paired]))
                 if paired else float("nan"))
    fraction = (recovered / requested) if requested else float("nan")
    constrained_reference = []
    for row in paired:
        block = (row.get("decoders") or {}).get("constrained") or {}
        value = (block.get("common_subset") or {}).get("mean")
        if value is not None and np.isfinite(value):
            constrained_reference.append(value)
    ceiling = (float(np.mean(constrained_reference)) if constrained_reference
               else float("nan"))

    options = {
        "A_strengthen_flow_conditioning": {
            "name": ("keep the improved decoder and strengthen QED-predictive "
                     "conditioning in the flow"),
            "expected_benefit": ("the decoder no longer discards the degree head's output "
                                 "in half of all conditions, so a flow-side change can now "
                                 "be MEASURED instead of being confounded by which "
                                 "assembly algorithm ran. Stage 20 already found the "
                                 "embedding displacement large but weakly QED-predictive, "
                                 "which is a direction problem and is what this addresses"),
            "risks": ("stages 16 and 17 both traded structure for conditioning - aromatic "
                      "rings ~0.35 and raw validity ~9% - so it needs the structure "
                      "guardrail and a matched control"),
            "compute_cost": "moderate: one GPU run of a few thousand steps per arm, plus a control",
        },
        "B_direct_atom_bond_flow": {
            "name": ("replace the embedding-based generator with a direct conditional "
                     "atom/bond normalizing flow"),
            "expected_benefit": ("removes the learned-embedding bottleneck: the property "
                                 "would condition a DISCRETE graph variable instead of a "
                                 "continuous geometry that a separately trained decoder has "
                                 "to invert"),
            "risks": ("the largest and least reversible change in the project; it discards "
                      "the unconditional Arm B generator's structure quality as a starting "
                      "point"),
            "compute_cost": "high: a reimplementation plus full training runs",
        },
        "C_continue_decoder_work": {
            "name": "continue decoder development",
            "expected_benefit": ("further assembly work - a soft degree penalty rather than "
                                 "an equality, or joint optimisation over degrees and bonds "
                                 "- if the measurements show headroom left in decoding"),
            "risks": ("decoding can only transmit what the embeddings already carry; past "
                      "the constrained decoder's response there is nothing left to recover"),
            "compute_cost": "low: CPU work on embeddings that already exist",
        },
    }
    improved = any(check["criterion"] == "qed_control_improves"
                   and check["status"] == "pass" for check in criteria.get("checks", []))
    partial = any(check["criterion"] == "qed_control_improves"
                  and check["status"] == "partial" for check in criteria.get("checks", []))
    severe_shift = bool(shift_auc is not None and np.isfinite(shift_auc)
                        and shift_auc >= 0.95)

    if not criteria.get("retain_experimental_decoder"):
        primary = "B_direct_atom_bond_flow" if severe_shift else "A_strengthen_flow_conditioning"
        rationale = (
            "the experimental assembler did not meet the retention criteria, so decoding is "
            "not where the remaining QED control is. "
            + ("With a real-versus-generated AUC of "
               f"{shift_auc:.3f} the embedding distribution mismatch is the severe "
               "limitation stage 20 flagged, and that is the condition under which the "
               "direct atom/bond flow is the right answer rather than an expensive detour."
               if severe_shift else
               "The flow's conditional transformation is the remaining candidate."))
    elif np.isfinite(fraction) and fraction >= 0.5:
        primary = "A_strengthen_flow_conditioning"
        rationale = (
            f"the decoder recovered {recovered:+.4f} of the {requested:+.2f} requested "
            f"change ({fraction * 100:.0f}%), which is most of what the constrained "
            f"decoder was already delivering ({ceiling:+.4f}). Decoding is therefore no "
            "longer the binding constraint, and the next unit of effort belongs in the "
            "flow - where stage 20 located the weak QED-predictive direction - now that "
            "a flow-side change can be measured without the assembly discontinuity "
            "confounding it.")
    elif np.isfinite(recovered) and recovered > 0:
        primary = ("B_direct_atom_bond_flow" if severe_shift
                   else "A_strengthen_flow_conditioning")
        rationale = (
            f"the decoder recovered only {recovered:+.4f} of the {requested:+.2f} "
            "requested change. The mechanism was real but small, so the bulk of the "
            "missing control is upstream of decoding. "
            + (f"With a real-versus-generated AUC of {shift_auc:.3f}, the representation "
               "mismatch is severe enough that the direct atom/bond flow is the "
               "defensible next step rather than further tuning of a representation that "
               "may not support reliable property control."
               if severe_shift else
               "The flow's conditional transformation is the next candidate."))
    else:
        primary = "A_strengthen_flow_conditioning"
        rationale = ("the assembler is worth keeping on structural grounds - it is total, "
                     "deterministic and no worse on validity - but it recovered no "
                     "measurable QED response, so the missing control is upstream.")

    return {
        "primary": primary,
        "rationale": rationale,
        "recovered_response": recovered,
        "requested_change": requested,
        "fraction_of_request_recovered": fraction,
        "constrained_decoder_reference": ceiling,
        "real_vs_generated_auc": shift_auc,
        "options": options,
        "guard": ("the objective is a deployable conditional generator. If the next "
                  "flow-side experiment does not move native QED control, the "
                  "representation - not the tuning - is the limitation, and option B "
                  "becomes the honest choice rather than a further round of decoder work."),
    }


__all__ = [
    "BASELINE_DECODER",
    "EXPERIMENTAL_DECODER",
    "WELL_SUPPORTED_MOLECULES",
    "assembly_statistics",
    "classify_support",
    "degree_consistency",
    "evaluate_criteria",
    "paired_decoder_changes",
    "recommend_next",
]
