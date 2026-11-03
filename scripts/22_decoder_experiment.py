#!/usr/bin/env python3
"""Stage 21 — matched four-decoder comparison: does total assembly preserve more signal?

Stage 20 decomposed the degree decoder's QED attenuation at N=23, T=0.2 as

    integer-programming contribution        +0.0212
    solver / fallback switching             +0.0386   <- the larger half
    total attenuation                       +0.0598   CI [+0.0348, +0.0844]

so the dominant cause is the DISCONTINUITY around the integer program, not the program.
`degree_constrained_bond_decode` abandons its program and runs a different algorithm on
five exits, two of which fire for non-chemical reasons: an odd predicted degree sum (the
sum of independent per-atom argmaxes, whose parity flips whenever ONE atom's degree moves
by one) and a predicted degree of zero (which guarantees disconnection). The degree head's
output is therefore discarded in one QED condition and used in the other, and the measured
response is dominated by which algorithm ran.

`zinc_gnf.assembly_total` closes every exit by repairing inside the formulation: the
degree sequence is moved to the nearest one a connected molecule can realise, the bond set
comes from one program over that sequence, connectivity is restored by degree-preserving
2-swaps, and bond orders come from a second small program under valence - so no order is
ever demoted. There is no other algorithm to fall back to.

WHAT THIS JOB MEASURES
All four assemblers decode THE SAME generated embeddings: one `decode_with_graphs` call
per cell decodes each sampled embedding with raw, constrained, degree and total, so they
differ only in how they turn one fixed set of logits into a graph. Nothing here selects
molecules by property, consults an oracle, or sees the target QED below the flow.

  grid        N x requested QED x temperature, every combination, one shared seed
  per cell    raw native validity (separately), strict validity, exact heavy-atom count,
              QED MAE / hit rate / bias, aromatic and total rings, connectivity and
              fragments, fallback frequency, relaxation level, degree and valence
              consistency measured FROM THE GRAPH for every decoder alike
  paired      at fixed noise, QED low -> high, on a COMMON subset valid under both
              requests for EVERY decoder, with bootstrap intervals and the experimental
              decoder's improvement over the degree decoder
  population  uniqueness, novelty against TRAIN only, internal diversity
  cost        seconds per molecule per decoder, timed on identical embeddings
  support     each cell marked well-supported or low-support from TRAIN counts

Then the Step 4 criteria as explicit pass / partial / fail / inconclusive checks, and one
Step 5 recommendation. No flow is trained, no checkpoint written, the test split is never
loaded, and the three existing decoders are untouched.

Read-only. Exit codes: 0 ok; 2 missing input; 4 bad arguments; 1 unexpected.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import traceback
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from zinc_gnf.conditioning import load_condition_scaler  # noqa: E402
from zinc_gnf.dataset import (  # noqa: E402
    MolecularGraphDataset,
    load_training_splits,
)
from zinc_gnf.decoder_contrast import (  # noqa: E402
    decode_with_graphs,
    paired_generated_embeddings,
)
from zinc_gnf.decoder_experiment import (  # noqa: E402
    BASELINE_DECODER,
    EXPERIMENTAL_DECODER,
    assembly_statistics,
    classify_support,
    degree_consistency,
    evaluate_criteria,
    paired_decoder_changes,
    recommend_next,
)
from zinc_gnf.decoder_information import support_counts  # noqa: E402
from zinc_gnf.metrics_generation import decoder_metrics, internal_diversity  # noqa: E402
from zinc_gnf.qed_evaluation import qed_achievement_table, qed_target_metrics  # noqa: E402
from zinc_gnf.trainer_autoencoder import load_autoencoder, resolve_device  # noqa: E402
from zinc_gnf.trainer_flow import file_sha256, load_flow  # noqa: E402
from zinc_gnf.utils.logging import JobLogger  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402
from zinc_gnf.utils.seed import set_seed  # noqa: E402

ALL_DECODERS = ("raw", "constrained", "degree", "total")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flow-checkpoint", required=True, help="READ-ONLY")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="data/processed/zinc250k_charge_aware_moflow_fixed")
    parser.add_argument("--ae-checkpoint", default=None)
    parser.add_argument("--target-n", default="15,23,30")
    parser.add_argument("--target-qed", default="0.4,0.6,0.8")
    parser.add_argument("--temperature", default="0.2,0.6,1.0")
    parser.add_argument("--count", type=int, default=256, help="samples per grid cell")
    parser.add_argument("--paired-count", type=int, default=256,
                        help="samples per paired (N, T) cell")
    parser.add_argument("--paired-qed", default="0.4,0.8")
    parser.add_argument("--qed-tolerance", type=float, default=0.1)
    parser.add_argument("--qed-window", type=float, default=0.05)
    parser.add_argument("--timing-samples", type=int, default=32)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--shift-auc", type=float, default=None,
                        help="stage 20's real-vs-generated AUC, for the step 5 decision")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


def _mean(values: list[Any]) -> float:
    """Mean over the finite entries, NaN when there are none (never a warning)."""
    finite = [float(v) for v in values
              if v is not None and np.isfinite(np.asarray(v, dtype=np.float64))]
    return float(np.mean(finite)) if finite else float("nan")


def _strip(records: list[dict[str, Any]]) -> None:
    """Drop the bond matrices once the statistics are computed: they are not JSON."""
    for record in records:
        for name in ALL_DECODERS:
            entry = record.get(name)
            if isinstance(entry, dict):
                entry.pop("bond_matrix", None)


@torch.no_grad()
def _generate_cell(flow, autoencoder, *, scaler, n: int, qed: float, count: int,
                   temperature: float, device, seed: int,
                   decoders: tuple[str, ...]) -> list[dict[str, Any]]:
    """One cell: `count` samples at a fixed (N, QED, T), decoded by EVERY assembler.

    Decoding all four from one sample is what makes the comparison matched - they see
    byte-identical embeddings, so nothing is attributable to sampling.
    """
    from zinc_gnf.conditioning import condition_embedding_for, standardise_condition
    from zinc_gnf.flow import node_mask_from_counts

    counts = [int(n)] * int(count)
    mask = node_mask_from_counts(counts, device=device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    values = torch.from_numpy(standardise_condition(
        flow.config.condition_features, scaler=scaler, n=counts,
        qed=[float(qed)] * int(count))).to(device)
    condition = condition_embedding_for(flow, values, None, int(count), device)
    embeddings = flow.sample(mask, condition, temperature=float(temperature),
                             generator=generator)
    if isinstance(embeddings, tuple):
        embeddings = embeddings[0]
    records = decode_with_graphs(autoencoder, embeddings, counts, decoders=decoders)
    for index, record in enumerate(records):
        record["index"] = index
    return records


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    arguments = parse_args(argv)
    output_root = Path(arguments.output_dir)
    try:
        targets_n = [int(v) for v in str(arguments.target_n).split(",") if v.strip()]
        targets_qed = [float(v) for v in str(arguments.target_qed).split(",") if v.strip()]
        temperatures = [float(v) for v in str(arguments.temperature).split(",") if v.strip()]
        paired_qed = [float(v) for v in str(arguments.paired_qed).split(",") if v.strip()]
        if len(paired_qed) != 2 or paired_qed[0] >= paired_qed[1]:
            raise ValueError("--paired-qed must be 'low,high' with low < high")
        if not targets_n or not targets_qed or not temperatures:
            raise ValueError("--target-n, --target-qed and --temperature must be non-empty")
        output_root.mkdir(parents=True, exist_ok=True)
        logger = JobLogger("decoder_experiment", output_root, filename="experiment.log")
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4

    started = time.perf_counter()
    resamples = int(arguments.bootstrap_resamples)
    try:
        set_seed(arguments.seed, deterministic=False)
        device = resolve_device(arguments.device)
        flow, flow_meta = load_flow(Path(arguments.flow_checkpoint), device=device)
        if "qed" not in flow.config.condition_features:
            raise ValueError(f"this flow does not condition on QED "
                             f"(features {list(flow.config.condition_features)})")
        recorded = flow_meta.get("autoencoder") or {}
        ae_path = Path(arguments.ae_checkpoint or recorded.get("checkpoint", ""))
        if not ae_path.exists():
            raise FileNotFoundError(f"autoencoder checkpoint not found: {ae_path}")
        autoencoder, _ = load_autoencoder(ae_path, device=device)
        autoencoder.eval()
        for parameter in autoencoder.parameters():
            parameter.requires_grad_(False)
        if not autoencoder.config.degree_head:
            raise ValueError("this autoencoder has no degree head, so neither the degree "
                             "decoder nor the experimental assembler can run")
        decoders = ALL_DECODERS
        scaler = load_condition_scaler(arguments.data_dir)
        train_records, val_records, fingerprint = load_training_splits(arguments.data_dir)
        train_set = MolecularGraphDataset(train_records)
        train_smiles = {record["smiles"] for record in train_records
                        if record.get("smiles")}
        validation_smiles = {record["smiles"] for record in val_records
                             if record.get("smiles")}
        del train_records, val_records

        logger.header("Stage 21: matched four-decoder comparison")
        logger.info(f"flow={arguments.flow_checkpoint} step={flow_meta.get('step')} "
                    f"sha256={file_sha256(arguments.flow_checkpoint)[:16]} (READ-ONLY)")
        logger.info(f"grid N={targets_n} x QED={targets_qed} x T={temperatures}, "
                    f"{arguments.count} samples per cell, decoders={list(decoders)}")
        logger.info("all four decoders decode the SAME embeddings; the target QED never "
                    "reaches the decoder and no molecule is selected by property")

        support = support_counts(train_set, n_values=targets_n, qed_values=targets_qed,
                                 window=float(arguments.qed_window))
        for row in support:
            parts = ", ".join(f"QED {key}: {value['molecules_within_window']:,}"
                              for key, value in row["by_qed"].items())
            logger.info(f"support N={row['n']}: {row['molecules_at_this_n']:,} training "
                        f"molecules; {parts}")

        # ---- the grid ----------------------------------------------------
        logger.header("the grid")
        cells: list[dict[str, Any]] = []
        population: dict[str, list[dict[str, Any]]] = {name: [] for name in decoders}
        for temperature in temperatures:
            for n in targets_n:
                for qed in targets_qed:
                    seed = int(arguments.seed)
                    records = _generate_cell(
                        flow, autoencoder, scaler=scaler, n=n, qed=qed,
                        count=int(arguments.count), temperature=temperature,
                        device=device, seed=seed, decoders=decoders)
                    cell: dict[str, Any] = {
                        "n": int(n), "requested_qed": float(qed),
                        "temperature": float(temperature), "samples": len(records),
                        "seed": seed,
                        "support": classify_support(support, n=n, qed=qed),
                        "decoders": {},
                    }
                    for name in decoders:
                        valid = [r[name] for r in records if r[name].get("valid")]
                        aromatic = [e["aromatic_rings"] for e in valid
                                    if e.get("aromatic_rings") is not None]
                        rings = [e["rings"] for e in valid if e.get("rings") is not None]
                        cell["decoders"][name] = {
                            "qed": qed_target_metrics(
                                records, requested_qed=qed, decoder=name,
                                tolerance=float(arguments.qed_tolerance)),
                            "assembly": assembly_statistics(records, name),
                            "degree_consistency": degree_consistency(records, name),
                            "aromatic_rings_mean": (float(np.mean(aromatic)) if aromatic
                                                    else float("nan")),
                            "rings_mean": float(np.mean(rings)) if rings else float("nan"),
                            "heavy_atom_count_exact": float(np.mean(
                                [1.0 if e.get("valid") and e.get("smiles")
                                 and _heavy_atoms(e["smiles"]) == int(n) else 0.0
                                 for e in (r[name] for r in records)])),
                        }
                        population[name].extend(
                            {"n_node": r["n_node"], name: r[name]} for r in records)
                    cells.append(cell)
                    head = cell["decoders"]
                    logger.info(
                        f"T={temperature:g} N={n} QED={qed:g} [{cell['support']['support']}]: "
                        + " | ".join(
                            f"{name} valid "
                            f"{head[name]['qed']['valid_molecules']}/{len(records)} "
                            f"QED {head[name]['qed']['realised_qed_mean']:.3f} "
                            f"MAE {head[name]['qed']['mae']:.3f}"
                            for name in decoders))
                    logger.info(
                        "    fallback/relaxation: "
                        + " | ".join(
                            f"{name} fb {head[name]['assembly'].get('fallback_rate', 0):.2f}"
                            + (f" lv {head[name]['assembly'].get('relaxation_levels')}"
                               if "relaxation_levels" in head[name]["assembly"] else "")
                            for name in ("degree", "total")))
                    _strip(records)

        # ---- the paired comparison ---------------------------------------
        logger.header("the paired comparison at fixed noise")
        paired: list[dict[str, Any]] = []
        for temperature in temperatures:
            for n in targets_n:
                sample = paired_generated_embeddings(
                    flow, scaler=scaler, n=n, qed_low=paired_qed[0],
                    qed_high=paired_qed[1], count=int(arguments.paired_count),
                    temperature=temperature, device=device, seed=int(arguments.seed))
                if not sample["usable_molecules"]:
                    logger.warn(f"T={temperature:g} N={n}: every sample non-finite")
                    continue
                low = decode_with_graphs(autoencoder, sample["low"], sample["counts"],
                                         decoders=decoders)
                high = decode_with_graphs(autoencoder, sample["high"], sample["counts"],
                                          decoders=decoders)
                row = paired_decoder_changes(
                    low, high, decoders=decoders,
                    requested_change=paired_qed[1] - paired_qed[0],
                    seed=int(arguments.seed), resamples=resamples)
                row.update({"n": int(n), "temperature": float(temperature),
                            "nonfinite_rate": sample["nonfinite_rate"],
                            "support_low": classify_support(
                                support_counts(train_set, n_values=[n],
                                               qed_values=paired_qed,
                                               window=float(arguments.qed_window)),
                                n=n, qed=paired_qed[0])})
                paired.append(row)
                logger.info(
                    f"T={temperature:g} N={n}: common subset "
                    f"{row['common_subset_molecules']}/{row['samples']} | "
                    + " | ".join(
                        f"{name} {row['decoders'][name]['common_subset']['mean']:+.4f}"
                        for name in decoders))
                improvement = row.get("improvement_over_baseline")
                if improvement and np.isfinite(improvement.get("difference", float("nan"))):
                    logger.info(
                        f"    total - degree = {improvement['difference']:+.4f} "
                        f"[{improvement.get('ci_low', float('nan')):+.4f}, "
                        f"{improvement.get('ci_high', float('nan')):+.4f}]"
                        + ("  <- excludes zero" if improvement.get("excludes_zero") else ""))
                del low, high, sample

        # ---- population metrics ------------------------------------------
        logger.header("population metrics (uniqueness, novelty vs TRAIN, diversity)")
        aggregate: dict[str, Any] = {}
        for name in decoders:
            records = population[name]
            metrics = decoder_metrics(records, name, train_smiles=train_smiles,
                                      reference_smiles={"train_and_val":
                                                        train_smiles | validation_smiles},
                                      diversity=False)
            smiles = sorted({r[name]["smiles"] for r in records
                             if r[name].get("valid") and r[name].get("smiles")})
            metrics["diversity"] = internal_diversity(smiles) if len(smiles) > 1 else {}
            aggregate[name] = {
                "strict_validity": metrics["strict_validity"]["value"],
                "sanitizable": metrics["sanitizable"]["value"],
                "uniqueness": metrics["uniqueness"]["value"],
                "novelty_vs_train": metrics["novelty_vs_train"]["value"],
                "novelty_vs_train_and_val": metrics.get(
                    "novelty_vs_train_and_val", {}).get("value"),
                "nuv": metrics["nuv"],
                "internal_diversity": metrics["diversity"].get("internal_diversity"),
                "fallback_rate": metrics["fallback_rate"]["value"],
                "fallback_reasons": metrics["fallback_reasons"],
                "valid_molecules": metrics["valid_molecules"],
                "unique_molecules": metrics["unique_molecules"],
                "rings_mean": metrics.get("rings_mean"),
                "aromatic_rings_mean": _mean(
                    [c["decoders"][name]["aromatic_rings_mean"] for c in cells]),
                "full_metrics": metrics,
            }
            logger.info(f"[{name}] strict validity {aggregate[name]['strict_validity']:.4f} | "
                        f"uniqueness {aggregate[name]['uniqueness']:.4f} | novelty "
                        f"{aggregate[name]['novelty_vs_train']:.4f} | diversity "
                        f"{aggregate[name]['internal_diversity']} | fallback "
                        f"{aggregate[name]['fallback_rate']:.4f}")
        logger.info("raw native validity is the 'raw' row and is never merged with "
                    "validity after constrained, degree or total assembly")

        # ---- QED achievement per decoder ---------------------------------
        achievement: dict[str, Any] = {}
        for name in decoders:
            achievement[name] = qed_achievement_table(
                [{"n": c["n"], "requested_qed": c["requested_qed"],
                  "temperature": c["temperature"],
                  "metrics": c["decoders"][name]["qed"]} for c in cells],
                tolerance=float(arguments.qed_tolerance))
            for label in ("well_supported", "low_support"):
                subset = [c for c in cells if c["support"]["support"] == label]
                if subset:
                    achievement[name][label] = qed_achievement_table(
                        [{"n": c["n"], "requested_qed": c["requested_qed"],
                          "temperature": c["temperature"],
                          "metrics": c["decoders"][name]["qed"]} for c in subset],
                        tolerance=float(arguments.qed_tolerance))
            overall = achievement[name].get("overall") or {}
            logger.info(f"[{name}] weighted QED MAE "
                        f"{overall.get('mae_weighted', float('nan')):.4f} | hit rate "
                        f"{overall.get('hit_rate_weighted', float('nan')):.4f} | pooled "
                        f"slope {overall.get('response_slope_pooled', float('nan')):+.4f} "
                        f"| within-N correlation "
                        f"{overall.get('qed_independent_of_n', float('nan')):+.4f}")

        # ---- cost per molecule -------------------------------------------
        logger.header("computational cost per molecule, on identical embeddings")
        cost: dict[str, Any] = {}
        timing_n = targets_n[len(targets_n) // 2]
        timing = paired_generated_embeddings(
            flow, scaler=scaler, n=timing_n, qed_low=paired_qed[0],
            qed_high=paired_qed[1], count=int(arguments.timing_samples),
            temperature=temperatures[0], device=device, seed=int(arguments.seed))
        if timing["usable_molecules"]:
            for name in decoders:
                begin = time.perf_counter()
                decode_with_graphs(autoencoder, timing["low"], timing["counts"],
                                   decoders=(name,))
                elapsed = time.perf_counter() - begin
                cost[name] = {
                    "molecules": timing["usable_molecules"],
                    "seconds_total": elapsed,
                    "seconds_per_molecule": elapsed / timing["usable_molecules"],
                    "minutes_per_10k": elapsed / timing["usable_molecules"] * 10000 / 60,
                }
                logger.info(f"[{name}] {cost[name]['seconds_per_molecule'] * 1000:7.2f} ms "
                            f"per molecule -> {cost[name]['minutes_per_10k']:.1f} min "
                            f"per 10,000 on one core (N={timing_n}, includes RDKit)")
        del timing

        # ---- the criteria and the decision -------------------------------
        logger.header("step 4: success criteria")
        criteria = evaluate_criteria(cells=cells, paired=paired, achievement=achievement,
                                     assembly=aggregate, cost=cost)
        for check in criteria["checks"]:
            logger.info(f"[{check['status'].upper()}] {check['criterion']}: "
                        f"{check['detail']}")
        logger.info(f"retain the experimental decoder: "
                    f"{criteria['retain_experimental_decoder']}")
        milestone = criteria["milestone"]
        logger.info(f"QED MAE milestone (<= {milestone['qed_mae_target']}): experimental "
                    f"{milestone['experimental_weighted_mae']}, baseline "
                    f"{milestone['baseline_weighted_mae']}, reached "
                    f"{milestone['reached']}")

        logger.header("step 5: the next development path")
        recommendation = recommend_next(criteria, paired=paired,
                                       shift_auc=arguments.shift_auc)
        logger.info(f"RECOMMENDATION {recommendation['primary']}: "
                    f"{recommendation['rationale']}")

        payload = {
            "stage": 21,
            "flow": {"checkpoint": str(arguments.flow_checkpoint),
                     "step": flow_meta.get("step"),
                     "sha256": file_sha256(arguments.flow_checkpoint),
                     "condition_features": list(flow.config.condition_features)},
            "autoencoder": {"checkpoint": str(ae_path), "degree_head": True},
            "dataset_fingerprint": fingerprint,
            "decoders": list(decoders),
            "experimental_decoder": EXPERIMENTAL_DECODER,
            "baseline_decoder": BASELINE_DECODER,
            "grid": {"target_n": targets_n, "target_qed": targets_qed,
                     "temperatures": temperatures, "count": int(arguments.count),
                     "paired_qed": paired_qed,
                     "paired_count": int(arguments.paired_count),
                     "qed_tolerance": float(arguments.qed_tolerance),
                     "qed_window": float(arguments.qed_window)},
            "seed": int(arguments.seed),
            "bootstrap_resamples": resamples,
            "support": support,
            "cells": cells,
            "paired": paired,
            "population": aggregate,
            "achievement": achievement,
            "cost": cost,
            "criteria": criteria,
            "recommendation": recommendation,
            "guarantees": {
                "matched": ("every decoder decodes the same embeddings from one sample; "
                            "no decoder has its own seed, node counts or temperature"),
                "no_selection": ("one embedding yields exactly one molecule per decoder, "
                                 "deterministically; nothing generates a batch and keeps "
                                 "the best QED"),
                "no_oracle": ("the target QED conditions the FLOW only; the assemblers see "
                              "the decoder's own bond and degree distributions and never "
                              "a property value"),
                "no_test_data": "the test split is never loaded; novelty uses TRAIN only",
                "baselines_untouched": ("raw, constrained and degree are unchanged; `total` "
                                        "is an additional option outside the default set"),
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_write_json(payload, output_root / "decoder_experiment.json")
        (output_root / "decoder_experiment.txt").write_text(
            _format_report(payload), encoding="utf-8")
        logger.info(f"wrote {output_root / 'decoder_experiment.json'} and "
                    f"{output_root / 'decoder_experiment.txt'}")
        logger.info(f"elapsed {payload['elapsed_seconds']:.1f}s")
        return 0
    except FileNotFoundError as error:
        logger.error(f"missing input: {error}")
        return 2
    except ValueError as error:
        logger.error(f"bad arguments: {error}")
        return 4
    except Exception:  # noqa: BLE001
        logger.error("unexpected failure:\n" + traceback.format_exc())
        return 1


def _heavy_atoms(smiles: str) -> int:
    from rdkit import Chem, rdBase

    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    return int(molecule.GetNumHeavyAtoms()) if molecule is not None else -1


def _format_report(payload: dict[str, Any]) -> str:  # noqa: C901
    def show(value: Any, template: str = "{:.4f}") -> str:
        if value is None:
            return "N/A"
        try:
            if not np.isfinite(float(value)):
                return "N/A"
            return template.format(float(value))
        except (TypeError, ValueError):
            return str(value)

    decoders = payload["decoders"]
    lines = ["Stage 21: matched four-decoder comparison", "",
             f"flow      : {payload['flow']['checkpoint']} (step {payload['flow']['step']})",
             f"sha256    : {payload['flow']['sha256']}",
             f"grid      : N={payload['grid']['target_n']} x "
             f"QED={payload['grid']['target_qed']} x T={payload['grid']['temperatures']} "
             f"x {payload['grid']['count']} samples",
             f"decoders  : {decoders} (experimental: {payload['experimental_decoder']})", ""]

    lines += ["=" * 100,
              "A. population totals - raw native validity is its OWN row and is never "
              "merged with the others",
              "=" * 100,
              f"{'decoder':<13}{'strict valid':>13}{'sanitizable':>12}{'unique':>9}"
              f"{'novel':>8}{'diversity':>11}{'arom.rings':>11}{'rings':>8}"
              f"{'fallback':>10}{'ms/mol':>9}"]
    for name in decoders:
        block = payload["population"].get(name) or {}
        cost = (payload["cost"].get(name) or {}).get("seconds_per_molecule")
        lines.append(
            f"{name:<13}{show(block.get('strict_validity')):>13}"
            f"{show(block.get('sanitizable')):>12}{show(block.get('uniqueness'), '{:.3f}'):>9}"
            f"{show(block.get('novelty_vs_train'), '{:.3f}'):>8}"
            f"{show(block.get('internal_diversity'), '{:.3f}'):>11}"
            f"{show(block.get('aromatic_rings_mean'), '{:.3f}'):>11}"
            f"{show(block.get('rings_mean'), '{:.2f}'):>8}"
            f"{show(block.get('fallback_rate'), '{:.3f}'):>10}"
            f"{show(None if cost is None else cost * 1000, '{:.1f}'):>9}")
    lines += [""]

    lines += ["=" * 100, "B. QED achievement per decoder, overall and by support",
              "=" * 100,
              f"{'decoder':<13}{'subset':<16}{'cells':>7}{'weighted MAE':>14}"
              f"{'hit rate':>10}{'slope':>9}{'within-N corr':>15}"]
    for name in decoders:
        table = payload["achievement"].get(name) or {}
        for label in ("overall", "well_supported", "low_support"):
            block = (table if label == "overall" else table.get(label)) or {}
            overall = block.get("overall")
            if not overall:
                continue
            lines.append(
                f"{name:<13}{label:<16}{overall.get('cells', 0):>7}"
                f"{show(overall.get('mae_weighted')):>14}"
                f"{show(overall.get('hit_rate_weighted'), '{:.3f}'):>10}"
                f"{show(overall.get('response_slope_pooled'), '{:+.3f}'):>9}"
                f"{show(overall.get('qed_independent_of_n'), '{:+.4f}'):>15}")
    lines += ["", "QED MAE is dominated by where the generated population sits, not by "
                  "decoding alone; the",
              "paired table below is the measurement that isolates the decoder.", ""]

    lines += ["=" * 100,
              "C. PAIRED QED change at fixed noise, on ONE common subset valid under both "
              "requests for EVERY decoder",
              "=" * 100,
              f"{'T':>5}{'N':>5}{'common':>8}"
              + "".join(f"{name + ('*' if name == 'raw' else ''):>12}"
                        for name in decoders)
              + f"{'total-degree':>14}{'95% CI':>22}"]
    for row in payload["paired"]:
        improvement = row.get("improvement_over_baseline") or {}
        interval = (f"[{improvement['ci_low']:+.4f}, {improvement['ci_high']:+.4f}]"
                    if improvement.get("ci_low") is not None
                    and np.isfinite(improvement.get("ci_low", float("nan"))) else "n/a")
        lines.append(
            f"{row['temperature']:>5.2g}{row['n']:>5}{row['common_subset_molecules']:>8}"
            + "".join(f"{show((row['decoders'][name]['common_subset'] or {}).get('mean'), '{:+.4f}'):>12}"
                      for name in decoders)
            + f"{show(improvement.get('difference'), '{:+.4f}'):>14}{interval:>22}")
    lines += ["", "A positive total-degree gap whose interval excludes zero is QED "
                  "response the degree decoder",
              "was losing. The subset is shared across constrained / degree / total, so "
              "it cannot come from",
              "one of them failing on hard samples. (*) raw is NOT in the intersection - "
              "its argmax validity is",
              "far lower and would empty the subset - so its column is its own subset "
              "and is not comparable.", ""]

    lines += ["=" * 100, "D. why: assembly health of the two degree-based decoders",
              "=" * 100]
    for name in (payload["baseline_decoder"], payload["experimental_decoder"]):
        rates: dict[str, list[float]] = {}
        levels: dict[str, int] = {}
        for cell in payload["cells"]:
            block = (cell["decoders"].get(name) or {}).get("assembly") or {}
            for key in ("fallback_rate", "degree_exact_rate", "connected_rate",
                        "assembly_failed_rate", "connectivity_swaps_mean",
                        "fragments_mean"):
                if key in block and np.isfinite(float(block[key])):
                    rates.setdefault(key, []).append(float(block[key]))
            for level, hits in (block.get("relaxation_levels") or {}).items():
                levels[level] = levels.get(level, 0) + int(hits)
            repair = block.get("degree_repair") or {}
            for key in ("parity_repaired_rate", "isolated_atoms_requested_mean",
                        "raised_for_connectivity_mean", "requested_sum_was_odd_rate"):
                if key in repair and np.isfinite(float(repair[key])):
                    rates.setdefault(key, []).append(float(repair[key]))
        lines.append(f"  [{name}]")
        for key, values in sorted(rates.items()):
            lines.append(f"      {key:<34}{np.mean(values):.4f}")
        if levels:
            lines.append(f"      relaxation levels used            {levels}")
        consistency = [((cell['decoders'].get(name) or {}).get('degree_consistency') or {})
                       for cell in payload["cells"]]
        exact = [c["degree_sequence_exact"] for c in consistency
                 if c.get("degree_sequence_exact") is not None
                 and np.isfinite(c["degree_sequence_exact"])]
        if exact:
            lines.append(f"      degree sequence exact (from graph) {np.mean(exact):.4f}")
    lines += ["", "`fallback_rate` on the degree decoder is the discontinuity stage 20 "
                  "measured: on a fallback it",
              "is a DIFFERENT algorithm, so two QED conditions can be decoded by two "
              "different decoders.",
              "`requested_sum_was_odd_rate` is how often the predicted degree sum could "
              "not be realised by any",
              "graph - the trigger that the experimental assembler repairs instead of "
              "abandoning.", ""]

    lines += ["=" * 100, "STEP 4: SUCCESS CRITERIA", "=" * 100]
    for check in payload["criteria"]["checks"]:
        lines += [f"[{check['status'].upper()}] {check['criterion']}",
                  f"    {check['detail']}"]
    summary = payload["criteria"]["summary"]
    lines += ["", f"pass {summary['pass']} | partial {summary['partial']} | "
                  f"fail {summary['fail']} | inconclusive {summary['inconclusive']}",
              f"retain the experimental decoder: "
              f"{payload['criteria']['retain_experimental_decoder']}", ""]
    milestone = payload["criteria"]["milestone"]
    lines += [f"QED MAE milestone (<= {milestone['qed_mae_target']}): experimental "
              f"{show(milestone['experimental_weighted_mae'])}, baseline "
              f"{show(milestone['baseline_weighted_mae'])} -> reached: "
              f"{milestone['reached']}",
              f"    {milestone['note']}", ""]

    recommendation = payload["recommendation"]
    lines += ["=" * 100, f"STEP 5 RECOMMENDATION: {recommendation['primary']}", "=" * 100,
              recommendation["rationale"], "",
              f"recovered {show(recommendation['recovered_response'], '{:+.4f}')} of a "
              f"requested {show(recommendation['requested_change'], '{:+.2f}')} "
              f"({show(recommendation['fraction_of_request_recovered'], '{:.1%}')}); the "
              f"constrained decoder's own response was "
              f"{show(recommendation['constrained_decoder_reference'], '{:+.4f}')}", ""]
    for key, option in recommendation["options"].items():
        marker = ">>" if key == recommendation["primary"] else "  "
        lines += [f"{marker} {key}: {option['name']}",
                  f"     benefit: {option['expected_benefit']}",
                  f"     risks  : {option['risks']}",
                  f"     cost   : {option['compute_cost']}", ""]
    lines += [recommendation["guard"], ""]
    lines += ["guarantees:"] + [f"  - {key}: {value}"
                                for key, value in payload["guarantees"].items()]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
