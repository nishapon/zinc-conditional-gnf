"""Stage 25: the matched comparison of the two factorisations.

THE QUESTION
------------
Does conditioning bond generation on atom chemistry - `p(X|c) p(A|X,c)` instead of
`p(A|c) p(X|A,c)` - improve joint chemical compatibility, aromaticity and native
validity, while keeping genuine QED conditioning?

Both orderings are exact factorisations of the same joint. Neither is incapable of
representing valid molecules, so the comparison is empirical and must be run under
identical conditions: the same requested N, QED, temperatures, sample counts and seeds,
the same validation subset for the likelihood numbers, and the shuffled-QED arm of the
NEW architecture as the null for every QED number that architecture reports.

WHAT IS MEASURED, AND WHY EACH ONE IS HERE
------------------------------------------
NATIVE first. Raw validity is the argmax graph with no chemistry applied; it is the
architectural number and is never merged with post-correction validity. Correction keeps
the largest connected component and can DROP ATOMS, so exact N is reported before and
after it, and a corrected-validity figure is read beside the atoms it cost.

VALENCE by element and formal charge, because stage 24 showed the failure is local: bond
counts were right (24.45 against 24.65 real at N=23) while 89.5% of graphs carried an
over-valent atom. If atoms-first works, this is where it shows first.

AROMATICITY as a SEPARATE criterion. Better valence does not imply learned aromatic
chemistry: a Kekule ring is perceived aromatic only when the single/double alternation is
exactly right, so alternating-ring counts are reported beside aromatic-ring counts and
both are read against the real-molecule ceiling rather than against zero.

QED RESTRICTED TO VALID, EXACT-N MOLECULES, with its support count attached. A slope over
corrected fragments is a slope over the wrong molecules, and a hit rate over four
surviving molecules is not a hit rate. The unrestricted figure is reported too, so the
restriction cannot be used to flatter either arm.

No oracle selection anywhere: one latent draw yields one molecule, and the requested QED
is never compared with RDKit to choose a sample. The test split is never loaded.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from zinc_gnf.dataset import MolecularGraphDataset
from zinc_gnf.graph_flow_generation import generate_conditional, real_structure_reference
from zinc_gnf.metrics_generation import decoder_metrics, internal_diversity
from zinc_gnf.qed_evaluation import qed_target_metrics
from zinc_gnf.valence_diagnostics import correction_damage, ring_profile, valence_profile

DECODERS = ("raw", "corrected")
# "corrected" is reported, never the headline. Stage 23 reached 79.5% corrected exact-N
# on 29.73% raw validity; that gap is what correction did, not what the model learned.
HEADLINE_DECODER = "raw"


def _mean(values: Sequence[Any]) -> float:
    finite = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def _merge_fractions(blocks: Sequence[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for block in blocks for key in block})
    return {key: _mean([block.get(key) for block in blocks]) for key in keys}


# ------------------------------------------------------------------ QED metrics ---
def within_n_response_slope(cells: Sequence[dict[str, Any]], *, key: str,
                            resamples: int, seed: int) -> dict[str, Any]:
    """Within-N slope of realised on requested QED, with a bootstrap CI.

    Deliberately the SAME estimator as `scripts/24_evaluate_graph_flow.response_slope` -
    fitted within each N and averaged, bootstrapped by resampling molecules inside each
    cell - so stage 25's slopes are comparable with every slope this project has already
    reported. A test asserts the two agree exactly on the same input, so if either is
    ever changed the test fails instead of the two silently diverging.

    `key` selects which population of realised values to read, which is how the
    restricted ("valid and exact N") slope is computed by the same code as the
    unrestricted one rather than by a second implementation.
    """
    grouped: dict[int, list[tuple[float, list[float]]]] = {}
    for cell in cells:
        values = (cell.get("realised") or {}).get(key) or []
        if len(values) >= 2:
            grouped.setdefault(int(cell["n"]), []).append(
                (float(cell["requested_qed"]), [float(v) for v in values]))
    usable = {n: rows for n, rows in grouped.items() if len(rows) >= 2}
    result: dict[str, Any] = {
        "population": key, "n_values": sorted(usable),
        "cells_used": sum(len(rows) for rows in usable.values()),
        "molecules_used": sum(len(values) for rows in usable.values()
                              for _, values in rows)}
    if not usable:
        result.update({"slope": float("nan"), "ci_low": float("nan"),
                       "ci_high": float("nan"), "excludes_zero": False,
                       "note": "no N had two or more populated QED cells"})
        return result

    def fit(sampler) -> float:
        slopes = []
        for rows in usable.values():
            x = np.asarray([requested for requested, _ in rows], dtype=np.float64)
            y = np.asarray([sampler(values) for _, values in rows], dtype=np.float64)
            if x.std() > 0:
                slopes.append(float(np.polyfit(x, y, 1)[0]))
        return float(np.mean(slopes)) if slopes else float("nan")

    result["slope"] = fit(lambda values: float(np.mean(values)))
    rng = np.random.default_rng(int(seed))

    def resample(values: list[float]) -> float:
        array = np.asarray(values, dtype=np.float64)
        return float(array[rng.integers(0, array.size, array.size)].mean())

    draws = np.asarray([fit(resample) for _ in range(int(resamples))], dtype=np.float64)
    draws = draws[np.isfinite(draws)]
    if draws.size:
        low, high = float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
        result.update({"ci_low": low, "ci_high": high,
                       "excludes_zero": bool(low > 0.0 or high < 0.0)})
    else:
        result.update({"ci_low": float("nan"), "ci_high": float("nan"),
                       "excludes_zero": False})
    result["note"] = ("slope 1.0 would be perfect following, 0.0 none; fitted within each "
                      "N and averaged, so it cannot be earned by N alone")
    return result


def qed_on_valid_exact_n(records: Sequence[dict[str, Any]], *, requested_qed: float,
                         decoder: str) -> dict[str, Any]:
    """QED accuracy over molecules that are BOTH chemically valid and exactly N atoms.

    The support count is part of the result, not a footnote. Exact N is the heavy-atom
    count of the sanitised molecule, so correction that drops atoms disqualifies a sample
    here even though it would still count as "valid" elsewhere.
    """
    kept = [r for r in records
            if (r.get(decoder) or {}).get("valid")
            and (r.get(decoder) or {}).get("exact_n")
            and (r.get(decoder) or {}).get("qed") is not None]
    values = np.asarray([float(r[decoder]["qed"]) for r in kept], dtype=np.float64)
    errors = np.abs(values - float(requested_qed)) if values.size else values
    return {
        "requested_qed": float(requested_qed),
        "support_molecules": int(values.size),
        "support_rate": float(values.size / max(len(records), 1)),
        "realised_qed_mean": float(values.mean()) if values.size else float("nan"),
        "realised_qed_sd": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
        "mae": float(errors.mean()) if values.size else float("nan"),
        "hit_rate_tol_05": float((errors <= 0.05).mean()) if values.size else float("nan"),
        "hit_rate_tol_10": float((errors <= 0.10).mean()) if values.size else float("nan"),
        "note": ("restricted to valid AND exact-N molecules; read the support count "
                 "before reading the rates"),
    }


# -------------------------------------------------------------------- one cell ----
def cell_report(model, *, scaler, n: int, qed: float, temperature: float, count: int,
                seed: int, device: str = "cpu",
                real_reference: dict[str, Any] | None = None) -> dict[str, Any]:
    """One (N, QED, temperature) cell: native generation plus every diagnostic."""
    output = generate_conditional(model, scaler=scaler, n=int(n), qed=float(qed),
                                  count=int(count), temperature=float(temperature),
                                  seed=int(seed), device=device, correct=True)
    records = output["records"]
    graphs = output["graphs"]
    cell: dict[str, Any] = {
        "n": int(n), "requested_qed": float(qed), "temperature": float(temperature),
        "samples": len(records),
        "structure": output["structure"],
        "valence": valence_profile(graphs),
        "rings": ring_profile(graphs),
        "correction_damage": correction_damage(graphs),
        "latent_round_trip": output["settings"]["latent_round_trip_relative_error"],
        "nonfinite_rate": output["settings"]["nonfinite_rate"],
        "realised": {}, "qed": {}, "aromatic": {},
        "real_reference": real_reference,
    }
    for decoder in DECODERS:
        valid = [r for r in records if (r.get(decoder) or {}).get("valid")]
        exact = [r for r in valid if r[decoder].get("exact_n")]
        cell["realised"][decoder] = [float(r[decoder]["qed"]) for r in valid
                                     if r[decoder].get("qed") is not None]
        cell["realised"][f"{decoder}_exact_n"] = [
            float(r[decoder]["qed"]) for r in exact if r[decoder].get("qed") is not None]
        cell["qed"][decoder] = qed_target_metrics(records, requested_qed=float(qed),
                                                  decoder=decoder, tolerance=0.1)
        cell["qed"][f"{decoder}_exact_n"] = qed_on_valid_exact_n(
            records, requested_qed=float(qed), decoder=decoder)
        cell["aromatic"][decoder] = {
            "valid_molecules": len(valid),
            "exact_n_valid_molecules": len(exact),
            "aromatic_rings_per_valid": _mean(
                [r[decoder].get("aromatic_rings") for r in valid]),
            "rings_per_valid": _mean([r[decoder].get("rings") for r in valid]),
            "with_an_aromatic_ring": sum(
                1 for r in valid if (r[decoder].get("aromatic_rings") or 0) > 0),
        }
    cell["records"] = records
    return cell


# --------------------------------------------------------------------- one arm ----
def likelihood_report(model, *, val_records: Sequence[dict[str, Any]], output_dir: Path,
                      val_molecules: int, batch_size: int, seed: int,
                      device: str = "cpu") -> dict[str, Any]:
    """Validation likelihood, the conditional permutation gap and inverse health.

    Computed by the TRAINER's own `evaluate`, with no training and nothing written, so
    the definitions are identical to the ones the training logs reported and the three
    arms are scored on the same seeded validation subset.
    """
    from zinc_gnf.trainer_graph_flow import GraphFlowTrainConfig, GraphFlowTrainer
    from zinc_gnf.utils.logging import null_logger

    val_set = MolecularGraphDataset(list(val_records))
    # the batcher refuses a batch larger than the dataset, which a small validation split
    # would otherwise trip; the batch size only affects chunking here, not the numbers
    batch = max(1, min(int(batch_size), len(val_set)))
    trainer = GraphFlowTrainer(
        model=model,
        config=GraphFlowTrainConfig(val_molecules=int(val_molecules),
                                    batch_size=batch, seed=int(seed)),
        train_set=val_set, val_set=val_set, output_dir=Path(output_dir),
        logger=null_logger(), device=device)
    summary = trainer.evaluate()
    return {
        "molecules": summary.get("molecules"),
        "validation_nll_per_dim": summary.get("nll_per_dim"),
        "validation_discrete_bound_per_dim": summary.get("bound_per_dim"),
        "condition_gap_per_atom": summary.get("condition_gap_per_atom"),
        "condition_gap_sem": summary.get("condition_gap_sem"),
        "condition_win_rate": summary.get("condition_win_rate"),
        "inverse_round_trip_worst": summary.get("inverse_round_trip_worst"),
        "inverse_acceptable": summary.get("inverse_acceptable"),
        "note": ("the permutation gap scores the SAME molecules under their true QED and "
                 "under QED permuted WITHIN N; zero means the density ignores QED. The "
                 "NLL is a dequantized-continuous density and the bound is an upper "
                 "bound on the discrete NLL - comparable BETWEEN these arms, because "
                 "they share a representation, and with nothing else"),
    }


def arm_report(model, *, scaler, label: str, targets_n: Sequence[int],
               targets_qed: Sequence[float], temperatures: Sequence[float], count: int,
               seed: int, train_records: Sequence[dict[str, Any]],
               train_smiles: set[str], validation_smiles: set[str] | None = None,
               resamples: int = 2000, device: str = "cpu", logger=None
               ) -> dict[str, Any]:
    """Everything stage 25 is judged on, for ONE model, over the whole grid."""
    real_reference = {int(n): real_structure_reference(train_records, n=int(n))
                      for n in targets_n}
    cells: list[dict[str, Any]] = []
    for temperature in temperatures:
        for n in targets_n:
            for qed in targets_qed:
                cell = cell_report(model, scaler=scaler, n=n, qed=qed,
                                   temperature=temperature, count=count, seed=seed,
                                   device=device,
                                   real_reference=real_reference.get(int(n)))
                cells.append(cell)
                if logger is not None:
                    structure, valence = cell["structure"], cell["valence"]
                    logger.info(
                        f"[{label}] T={temperature:g} N={n} QED={qed:g}: raw valid "
                        f"{structure['raw_valid_rate'] * 100:.2f}%, bonds "
                        f"{structure['bonds_mean']:.2f} vs real "
                        f"{(cell['real_reference'] or {}).get('bonds_mean', float('nan')):.2f}"
                        f", components {structure['components_mean']:.2f}, over-valent "
                        f"{structure['over_valent_atoms_mean']:.2f}/graph "
                        f"({valence['over_valent_atom_rate'] * 100:.1f}% of atoms), "
                        f"aromatic rings/graph "
                        f"{cell['correction_damage']['aromatic_rings_before_per_graph']:.3f}"
                        f", exact-N valid QED support "
                        f"{cell['qed']['raw_exact_n']['support_molecules']}/{cell['samples']}")

    atoms_seen: Counter = Counter()
    atoms_over: Counter = Counter()
    ring_sizes: Counter = Counter()
    for cell in cells:
        for key, entry in cell["valence"]["by_atom_type"].items():
            atoms_seen[key] += int(entry["atoms"])
            atoms_over[key] += int(entry["over_valent"])
        for size, number in cell["rings"]["ring_size_counts"].items():
            ring_sizes[int(size)] += int(number)

    summary: dict[str, Any] = {
        "label": label,
        "factorisation": model.config.factorisation,
        "atom_context_mode": model.config.atom_context_mode,
        "bond_conditioner": model.config.bond_conditioner,
        "parameters": model.parameter_report(),
        "grid_cells": len(cells),
        "samples_per_cell": int(count),
        # --- native structure ---
        "raw_valid_rate": _mean([c["structure"]["raw_valid_rate"] for c in cells]),
        "bonds_mean": _mean([c["structure"]["bonds_mean"] for c in cells]),
        "real_bonds_mean": _mean([(c["real_reference"] or {}).get("bonds_mean")
                                  for c in cells]),
        "components_mean": _mean([c["structure"]["components_mean"] for c in cells]),
        "over_valent_atoms_mean": _mean(
            [c["structure"]["over_valent_atoms_mean"] for c in cells]),
        "over_valent_atom_rate": _mean(
            [c["valence"]["over_valent_atom_rate"] for c in cells]),
        "infeasible_atom_rate": _mean([c["valence"]["infeasible_rate"] for c in cells]),
        "over_valent_rate_by_atom_type": {
            key: atoms_over[key] / max(atoms_seen[key], 1) for key in sorted(atoms_seen)},
        "atoms_by_atom_type": dict(sorted(atoms_seen.items())),
        "atom_type_fraction": _merge_fractions(
            [c["structure"]["atom_type_fraction"] for c in cells]),
        "bond_type_fraction": _merge_fractions(
            [c["structure"]["bond_type_fraction"] for c in cells]),
        # --- correction, reported and never merged with the above ---
        "corrected_exact_n_rate": _mean(
            [c["structure"]["correction_exact_n_rate"] for c in cells]),
        "correction_atoms_lost_mean": _mean(
            [c["structure"]["correction_atoms_lost_mean"] for c in cells]),
        "correction_orders_removed_mean": _mean(
            [c["structure"]["correction_orders_removed_mean"] for c in cells]),
        # --- aromaticity, a separate criterion ---
        "rings_per_graph": _mean([c["rings"]["rings_per_graph"] for c in cells]),
        "ring_size_counts": dict(sorted(ring_sizes.items())),
        "alternating_ring_share": _mean(
            [(c["rings"]["pattern_distribution_5_and_6"] or {}).get("alternating", 0.0)
             for c in cells]),
        "aromatic_rings_per_graph_raw": _mean(
            [c["correction_damage"]["aromatic_rings_before_per_graph"] for c in cells]),
        "aromatic_rings_per_graph_corrected": _mean(
            [c["correction_damage"]["aromatic_rings_after_per_graph"] for c in cells]),
        "alternating_rings_destroyed_total": int(sum(
            c["correction_damage"]["alternating_rings_destroyed"] for c in cells)),
        # --- numerics ---
        "latent_round_trip_worst": max(
            [max(c["latent_round_trip"].values()) for c in cells], default=float("nan")),
        "nonfinite_rate": _mean([c["nonfinite_rate"] for c in cells]),
    }

    for decoder in DECODERS:
        records = [{"n_node": r["n_node"], decoder: r[decoder]}
                   for cell in cells for r in cell["records"]]
        metrics = decoder_metrics(
            records, decoder, train_smiles=set(train_smiles),
            reference_smiles=({"train_and_val": set(train_smiles) | set(validation_smiles)}
                              if validation_smiles else None),
            diversity=False)
        unique = sorted({r[decoder]["smiles"] for r in records
                         if r[decoder].get("valid") and r[decoder].get("smiles")})
        diversity = internal_diversity(unique) if len(unique) > 1 else {}
        summary[decoder] = {
            "strict_validity": metrics["strict_validity"]["value"],
            "sanitizable": metrics["sanitizable"]["value"],
            "uniqueness": metrics["uniqueness"]["value"],
            "novelty_vs_train": metrics["novelty_vs_train"]["value"],
            "internal_diversity": diversity.get("internal_diversity"),
            "valid_molecules": metrics["valid_molecules"],
            "unique_molecules": metrics["unique_molecules"],
            "qed_mae": _mean([c["qed"][decoder]["mae"] for c in cells]),
            "hit_rate_tol_10": _mean([c["qed"][decoder]["hit_rate"] for c in cells]),
            "aromatic_rings_per_valid": _mean(
                [c["aromatic"][decoder]["aromatic_rings_per_valid"] for c in cells]),
            "response_slope": within_n_response_slope(cells, key=decoder,
                                                      resamples=resamples, seed=seed),
            "exact_n_rate": _mean(
                [c["qed"][f"{decoder}_exact_n"]["support_rate"] for c in cells]),
            "valid_exact_n": {
                "support_molecules": int(sum(
                    c["qed"][f"{decoder}_exact_n"]["support_molecules"] for c in cells)),
                "qed_mae": _mean([c["qed"][f"{decoder}_exact_n"]["mae"] for c in cells]),
                "hit_rate_tol_10": _mean(
                    [c["qed"][f"{decoder}_exact_n"]["hit_rate_tol_10"] for c in cells]),
                "hit_rate_tol_05": _mean(
                    [c["qed"][f"{decoder}_exact_n"]["hit_rate_tol_05"] for c in cells]),
                "response_slope": within_n_response_slope(
                    cells, key=f"{decoder}_exact_n", resamples=resamples, seed=seed),
                "note": ("the QED numbers that count: valid AND exact-N molecules only, "
                         "with the support they rest on"),
            },
        }

    summary["cells"] = [{k: v for k, v in cell.items() if k != "records"}
                        for cell in cells]
    summary["real_structure_reference"] = {str(k): v for k, v in real_reference.items()}
    summary["note"] = ("raw is the architectural number; corrected is reported but is "
                       "never the headline and can drop atoms")
    return summary


# ------------------------------------------------------------------- the verdict ---
def compare(arms: dict[str, dict[str, Any]], *, real_ceiling: dict[str, Any] | None = None,
            baseline: str = "bonds_first", candidate: str = "atoms_first",
            null: str = "atoms_first_null") -> dict[str, Any]:
    """Apply stage 25's stated acceptance criteria. Every one of them, separately.

    The criteria are deliberately conjunctive: raw validity alone is not a result if the
    atom distribution has collapsed or exact N has been traded away, and aromaticity is
    its own criterion rather than something valence is allowed to stand in for.
    """
    new, old = arms.get(candidate), arms.get(baseline)
    if not new or not old:
        return {"verdict": "incomparable",
                "reason": f"need both {candidate!r} and {baseline!r} arms"}
    control = arms.get(null) or {}
    headline = HEADLINE_DECODER

    def delta(key: str) -> float:
        return float(new.get(key, float("nan"))) - float(old.get(key, float("nan")))

    ceiling = float((real_ceiling or {}).get("aromatic_rings_per_molecule", float("nan")))
    new_slope = new[headline]["valid_exact_n"]["response_slope"]
    null_slope = ((control.get(headline) or {}).get("valid_exact_n") or {}).get(
        "response_slope") or {}
    checks = {
        "native_raw_validity_improved": {
            "value": delta("raw_valid_rate"),
            "passed": bool(delta("raw_valid_rate") > 0),
            "detail": (f"raw validity {old['raw_valid_rate']:.4f} -> "
                       f"{new['raw_valid_rate']:.4f}"),
        },
        "valence_failures_reduced": {
            "value": delta("over_valent_atom_rate"),
            "passed": bool(delta("over_valent_atom_rate") < 0),
            "detail": (f"over-valent atom rate {old['over_valent_atom_rate']:.4f} -> "
                       f"{new['over_valent_atom_rate']:.4f}"),
        },
        "atom_distribution_not_degraded": {
            "value": _distribution_distance(new["atom_type_fraction"],
                                           old["atom_type_fraction"]),
            "passed": None,
            "detail": ("total-variation distance between the two arms' atom "
                       "distributions; judge BOTH against the real distribution, which "
                       "is in real_structure_reference - a validity gain bought by "
                       "turning the molecule into carbon is not a gain"),
        },
        "exact_n_preserved": {
            "value": float(new[headline]["exact_n_rate"]
                           - old[headline]["exact_n_rate"]),
            "passed": bool(new[headline]["exact_n_rate"] >= old[headline]["exact_n_rate"]),
            "detail": (f"native exact-N (valid molecules) "
                       f"{old[headline]['exact_n_rate']:.4f} -> "
                       f"{new[headline]['exact_n_rate']:.4f}"),
        },
        "aromaticity_learned": {
            "value": delta("aromatic_rings_per_graph_raw"),
            "passed": bool(new["aromatic_rings_per_graph_raw"]
                           > old["aromatic_rings_per_graph_raw"]),
            "detail": (f"aromatic rings per RAW graph "
                       f"{old['aromatic_rings_per_graph_raw']:.4f} -> "
                       f"{new['aromatic_rings_per_graph_raw']:.4f}; real ceiling "
                       f"{ceiling:.3f}. A SEPARATE criterion: better valence does not "
                       f"imply learned aromatic chemistry"),
        },
        "alternation_learned": {
            "value": delta("alternating_ring_share"),
            "passed": bool(new["alternating_ring_share"] > old["alternating_ring_share"]),
            "detail": (f"share of 5-/6-rings with Kekule alternation "
                       f"{old['alternating_ring_share']:.4f} -> "
                       f"{new['alternating_ring_share']:.4f}"),
        },
        "qed_response_beats_its_null": {
            "value": (float(new_slope.get("slope", float("nan")))
                      - float(null_slope.get("slope", float("nan")))),
            "passed": (bool(new_slope.get("slope", float("-inf"))
                            > null_slope.get("slope", float("inf")))
                       if null_slope else None),
            "detail": (f"exact-N valid response slope "
                       f"{new_slope.get('slope', float('nan')):+.4f} "
                       f"[{new_slope.get('ci_low', float('nan')):+.4f}, "
                       f"{new_slope.get('ci_high', float('nan')):+.4f}] on "
                       f"{new_slope.get('molecules_used', 0)} molecules against the "
                       f"shuffled-QED arm's "
                       f"{null_slope.get('slope', float('nan')):+.4f}"
                       + ("" if null_slope else " (NO null arm supplied: the QED "
                                                 "criterion cannot be judged)")),
        },
        "inverse_still_sound": {
            "value": new["latent_round_trip_worst"],
            "passed": bool(new["latent_round_trip_worst"] <= 1e-3),
            "detail": ("sampling IS the inverse; above 1e-3 the molecules are not drawn "
                       "from the trained density and nothing else here is valid"),
        },
        "capacity_not_the_explanation": {
            "value": (new["parameters"]["total"] / max(old["parameters"]["total"], 1)),
            "passed": bool(new["parameters"]["total"] <= old["parameters"]["total"]),
            "detail": (f"{new['parameters']['total']:,} against "
                       f"{old['parameters']['total']:,} parameters. If the new arm WINS "
                       f"with fewer, capacity cannot explain it; if it LOSES with fewer, "
                       f"the question stays open and a capacity-matched rerun is owed"),
        },
    }
    decisive = [name for name, check in checks.items()
                if check["passed"] is False
                and name in ("native_raw_validity_improved", "inverse_still_sound")]
    aromatic_passed = bool(checks["aromaticity_learned"]["passed"])
    if decisive:
        verdict = "no"
    elif checks["native_raw_validity_improved"]["passed"] and aromatic_passed:
        verdict = "yes_on_both_criteria"
    elif checks["native_raw_validity_improved"]["passed"]:
        verdict = "validity_only"
    else:
        verdict = "no"
    return {
        "verdict": verdict,
        "checks": checks,
        "reading": {
            "yes_on_both_criteria": ("native validity AND aromaticity both improved; "
                                     "this licenses a longer run, not a deployment claim"),
            "validity_only": ("valence improved and aromatic chemistry did not. Stage 24 "
                              "predicted these could come apart, which is why they are "
                              "separate criteria"),
            "no": ("the factorisation did not deliver on its stated criteria. Check "
                   "capacity_not_the_explanation and the condition-gap trend before "
                   "concluding it cannot"),
        }[verdict],
        "warning": ("corrected validity is NOT success. It is reported beside the atoms "
                    "it dropped, and the headline is the raw, native number"),
    }


def _distribution_distance(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * float(sum(abs(float(left.get(k, 0.0)) - float(right.get(k, 0.0)))
                           for k in keys))


__all__ = [
    "DECODERS",
    "HEADLINE_DECODER",
    "arm_report",
    "cell_report",
    "compare",
    "likelihood_report",
    "qed_on_valid_exact_n",
    "within_n_response_slope",
]
