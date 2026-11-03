#!/usr/bin/env python3
"""Stage 22b — matched evaluation: direct graph flow vs the stage-17 embedding GNF.

Every model is asked for the same grid, the same counts and the same seeds. Four columns
may appear, and the second is what makes the first interpretable:

  graph_flow        the prototype, sampled NATIVELY from p(G | N, QED)
  graph_flow_null   the SAME architecture trained on QED permuted within N. If the real
                    arm does not separate from this, it has learned nothing about QED.
  graph_flow_prior  the real arm with the condition switched off INSIDE the couplings,
                    leaving only the conditional prior. A prior-only flow shifts the
                    latent cloud and can move average QED without reshaping p(G|c) - the
                    exact artefact stage 21 produced - so this column separates real
                    control from a shifted mean.
  baseline          the stage-17 warm-start embedding GNF through its degree decoder

Reported separately, never merged:
  * RAW validity (argmax of the generated tensors, no chemistry) - the architectural
    number, comparable with "validity without check" in the literature
  * strict validity AFTER correction, plus the repair rate
  * exact N before and after correction, because correction keeps the largest connected
    component and can drop atoms
  * QED MAE, hit +-0.05 and +-0.1, response slope with a bootstrap CI, within-N correlation
  * aromatic rings, total rings, fragments
  * uniqueness, novelty against TRAIN only, internal diversity
  * seconds per molecule
  * likelihoods, each labelled with its own representation and NEVER compared across them

No oracle reranking and no candidate selection: one latent draw yields one molecule, and
the requested QED is never compared with RDKit to choose a sample. The test split is
never loaded.

Exit codes: 0 ok; 2 missing input; 4 bad arguments; 1 unexpected.
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
from zinc_gnf.dataset import MolecularGraphDataset, load_training_splits  # noqa: E402
from zinc_gnf.decoder_information import support_counts  # noqa: E402
from zinc_gnf.graph_flow import neutralise_coupling_condition  # noqa: E402
from zinc_gnf.graph_flow_generation import (  # noqa: E402
    generate_conditional,
    real_structure_reference,
)
from zinc_gnf.metrics_generation import decoder_metrics, internal_diversity  # noqa: E402
from zinc_gnf.qed_evaluation import qed_achievement_table, qed_target_metrics  # noqa: E402
from zinc_gnf.qed_probes import paired_bootstrap_ci  # noqa: E402
from zinc_gnf.trainer_graph_flow import load_graph_flow  # noqa: E402
from zinc_gnf.utils.logging import JobLogger  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402
from zinc_gnf.utils.seed import set_seed  # noqa: E402

GRAPH_DECODERS = ("raw", "corrected")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph-flow-checkpoint", required=True, help="READ-ONLY")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="data/processed/zinc250k_charge_aware_moflow_fixed")
    parser.add_argument("--null-checkpoint", default=None,
                        help="the --shuffle-qed arm; the null for every number here")
    parser.add_argument("--previous-graph-checkpoint", default=None,
                        help="an earlier graph flow (e.g. the stage-22 pairwise "
                             "conditioner) to compare against under identical requests")
    parser.add_argument("--baseline-flow-checkpoint", default=None,
                        help="the stage-17 warm start (READ-ONLY)")
    parser.add_argument("--baseline-ae-checkpoint", default=None)
    parser.add_argument("--target-n", default="15,23,30")
    parser.add_argument("--target-qed", default="0.4,0.6,0.8")
    parser.add_argument("--temperature", default="0.6,0.85,1.0")
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--qed-window", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


def _finite(values: Any) -> list[float]:
    return [float(v) for v in values if v is not None and np.isfinite(float(v))]


def _mean(values: Any) -> float:
    finite = _finite(values)
    return float(np.mean(finite)) if finite else float("nan")


def response_slope(cells: list[dict[str, Any]], decoder: str, *, resamples: int,
                   seed: int) -> dict[str, Any]:
    """Within-N slope of realised on requested QED, with a bootstrap CI.

    Molecules are resampled WITHIN each cell and the slope refitted, so the interval
    reflects the sampling noise in the realised values rather than an asymptotic formula.
    Slopes are fitted within each N and then averaged, because N and QED are correlated in
    ZINC250K and a pooled slope could be earned by N alone.
    """
    grouped: dict[int, list[tuple[float, list[float]]]] = {}
    for cell in cells:
        values = cell["realised"].get(decoder) or []
        if len(values) >= 2:
            grouped.setdefault(int(cell["n"]), []).append(
                (float(cell["requested_qed"]), [float(v) for v in values]))
    usable = {n: rows for n, rows in grouped.items() if len(rows) >= 2}
    result: dict[str, Any] = {"decoder": decoder, "n_values": sorted(usable),
                              "cells_used": sum(len(r) for r in usable.values())}
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


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    arguments = parse_args(argv)
    output_root = Path(arguments.output_dir)
    try:
        targets_n = [int(v) for v in str(arguments.target_n).split(",") if v.strip()]
        targets_qed = [float(v) for v in str(arguments.target_qed).split(",") if v.strip()]
        temperatures = [float(v) for v in str(arguments.temperature).split(",") if v.strip()]
        if not targets_n or not targets_qed or not temperatures:
            raise ValueError("--target-n, --target-qed and --temperature must be non-empty")
        output_root.mkdir(parents=True, exist_ok=True)
        logger = JobLogger("graph_flow_eval", output_root, filename="evaluate.log")
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4

    started = time.perf_counter()
    resamples = int(arguments.bootstrap_resamples)
    try:
        set_seed(arguments.seed, deterministic=False)
        device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu")
            if arguments.device == "auto" else arguments.device)
        scaler = load_condition_scaler(arguments.data_dir)
        train_records, val_records, fingerprint = load_training_splits(arguments.data_dir)
        train_set = MolecularGraphDataset(train_records)
        train_smiles = {r["smiles"] for r in train_records if r.get("smiles")}
        validation_smiles = {r["smiles"] for r in val_records if r.get("smiles")}
        del val_records

        model, meta = load_graph_flow(Path(arguments.graph_flow_checkpoint), device=device)
        logger.header("Stage 22b: matched evaluation of the direct graph flow")
        logger.info(f"graph flow {arguments.graph_flow_checkpoint} step={meta.get('step')} "
                    f"(READ-ONLY); arm="
                    f"{(meta.get('provenance') or {}).get('arm')}")
        logger.info(f"grid N={targets_n} x QED={targets_qed} x T={temperatures} x "
                    f"{arguments.count} samples; seed {arguments.seed}")
        logger.info("native sampling only: no guidance, no reranking, no candidate "
                    "selection; the test split is never loaded")

        arms: dict[str, Any] = {"graph_flow": {"model": model, "kind": "graph"}}
        if arguments.null_checkpoint:
            null_model, null_meta = load_graph_flow(Path(arguments.null_checkpoint),
                                                    device=device)
            arms["graph_flow_null"] = {"model": null_model, "kind": "graph"}
            logger.info(f"null arm {arguments.null_checkpoint} step={null_meta.get('step')} "
                        f"arm={(null_meta.get('provenance') or {}).get('arm')}")
        if arguments.previous_graph_checkpoint:
            previous, previous_meta = load_graph_flow(
                Path(arguments.previous_graph_checkpoint), device=device)
            arms["graph_flow_previous"] = {"model": previous, "kind": "graph"}
            logger.info(f"previous graph flow {arguments.previous_graph_checkpoint} "
                        f"step={previous_meta.get('step')} conditioner="
                        f"{previous.config.bond_conditioner} "
                        f"max_log_scale={previous.config.max_log_scale} (READ-ONLY)")
        arms["graph_flow_prior"] = {"model": model, "kind": "graph", "prior_only": True}
        baseline = None
        if arguments.baseline_flow_checkpoint:
            from zinc_gnf.trainer_autoencoder import load_autoencoder
            from zinc_gnf.trainer_flow import load_flow

            base_flow, base_meta = load_flow(Path(arguments.baseline_flow_checkpoint),
                                             device=device)
            recorded = (base_meta.get("autoencoder") or {}).get("checkpoint", "")
            ae_path = Path(arguments.baseline_ae_checkpoint or recorded)
            if not ae_path.exists():
                raise FileNotFoundError(f"baseline autoencoder not found: {ae_path}")
            autoencoder, _ = load_autoencoder(ae_path, device=device)
            autoencoder.eval()
            baseline = {"flow": base_flow, "autoencoder": autoencoder,
                        "meta": base_meta, "kind": "embedding"}
            arms["baseline"] = baseline
            logger.info(f"baseline {arguments.baseline_flow_checkpoint} "
                        f"step={base_meta.get('step')} (READ-ONLY, stage-17 warm start)")

        # Real structural reference per N: without it the generated numbers have no scale.
        real_reference = {int(n): real_structure_reference(train_records, n=int(n))
                          for n in targets_n}
        for n, reference in real_reference.items():
            if reference is None:
                logger.warn(f"no training molecule has N={n}; structural comparison at "
                            f"this N is unavailable")
            else:
                logger.info(f"REAL N={n}: {reference['bonds_mean']:.1f} bonds, "
                            f"{reference['components_mean']:.2f} components, carbon "
                            f"{reference['atom_type_fraction'].get('C', 0.0):.3f}")
        support = support_counts(train_set, n_values=targets_n, qed_values=targets_qed,
                                 window=float(arguments.qed_window))
        for row in support:
            parts = ", ".join(f"QED {k}: {v['molecules_within_window']:,}"
                              for k, v in row["by_qed"].items())
            logger.info(f"support N={row['n']}: {row['molecules_at_this_n']:,} training "
                        f"molecules; {parts}")

        # ---- the grid ----------------------------------------------------
        logger.header("the grid")
        cells: dict[str, list[dict[str, Any]]] = {name: [] for name in arms}
        population: dict[str, list[dict[str, Any]]] = {name: [] for name in arms}
        cost: dict[str, Any] = {}
        for name, arm in arms.items():
            decoders = (("raw", "degree") if arm["kind"] == "embedding"
                        else GRAPH_DECODERS)
            elapsed = 0.0
            produced = 0
            round_trip: list[dict[str, float]] = []
            for temperature in temperatures:
                for n in targets_n:
                    for qed in targets_qed:
                        begin = time.perf_counter()
                        if arm["kind"] == "graph":
                            if arm.get("prior_only"):
                                with neutralise_coupling_condition(arm["model"]):
                                    output = generate_conditional(
                                        arm["model"], scaler=scaler, n=n, qed=qed,
                                        count=int(arguments.count),
                                        temperature=temperature, seed=arguments.seed,
                                        device=device)
                            else:
                                output = generate_conditional(
                                    arm["model"], scaler=scaler, n=n, qed=qed,
                                    count=int(arguments.count), temperature=temperature,
                                    seed=arguments.seed, device=device)
                            records = output["records"]
                            round_trip.append(
                                output["settings"]["latent_round_trip_relative_error"])
                            structure = output.get("structure")
                        else:
                            structure = None
                            records = _baseline_sample(
                                arm, scaler=scaler, n=n, qed=qed,
                                count=int(arguments.count), temperature=temperature,
                                seed=arguments.seed, device=device)
                        elapsed += time.perf_counter() - begin
                        produced += len(records)
                        cell: dict[str, Any] = {
                            "n": int(n), "requested_qed": float(qed),
                            "temperature": float(temperature), "samples": len(records),
                            "decoders": {}, "realised": {},
                            "structure": structure,
                            "real_reference": real_reference.get(int(n)),
                        }
                        if structure:
                            reference = real_reference.get(int(n)) or {}
                            logger.info(
                                f"    structure: {structure['bonds_mean']:.1f} bonds vs "
                                f"real {reference.get('bonds_mean', float('nan')):.1f}, "
                                f"components {structure['components_mean']:.2f}, largest "
                                f"{structure['largest_component_mean']:.1f}/{n}, "
                                f"over-valent {structure['over_valent_atoms_mean']:.2f}, "
                                f"raw valid {structure['raw_valid_rate'] * 100:.2f}%")
                        for decoder in decoders:
                            values = [float(r[decoder]["qed"]) for r in records
                                      if r[decoder].get("valid")
                                      and r[decoder].get("qed") is not None]
                            cell["realised"][decoder] = values
                            cell["decoders"][decoder] = {
                                "qed_tol_10": qed_target_metrics(
                                    records, requested_qed=qed, decoder=decoder,
                                    tolerance=0.1),
                                "qed_tol_05": qed_target_metrics(
                                    records, requested_qed=qed, decoder=decoder,
                                    tolerance=0.05),
                                "realised_qed": paired_bootstrap_ci(
                                    values, seed=arguments.seed, resamples=resamples),
                                "exact_n_rate": _mean(
                                    [1.0 if r[decoder].get("exact_n") else 0.0
                                     for r in records]
                                    if "exact_n" in (records[0].get(decoder) or {})
                                    else [np.nan]),
                                "aromatic_rings_mean": _mean(
                                    [r[decoder].get("aromatic_rings") for r in records
                                     if r[decoder].get("valid")]),
                                "rings_mean": _mean(
                                    [r[decoder].get("rings") for r in records
                                     if r[decoder].get("valid")]),
                                "fragments_mean": _mean(
                                    [r[decoder].get("fragments") for r in records]),
                                "repair_rate": _mean(
                                    [1.0 if r[decoder].get("fallback") else 0.0
                                     for r in records]),
                            }
                            population[name].extend(
                                {"n_node": r["n_node"], decoder: r[decoder]}
                                for r in records)
                        cells[name].append(cell)
                        head = cell["decoders"][decoders[-1]]["qed_tol_10"]
                        logger.info(
                            f"[{name}] T={temperature:g} N={n} QED={qed:g}: "
                            f"valid {head['valid_molecules']}/{len(records)} "
                            f"realised {head['realised_qed_mean']:.3f} "
                            f"MAE {head['mae']:.3f}")
            cost[name] = {"molecules": produced, "seconds_total": elapsed,
                          "seconds_per_molecule": elapsed / max(produced, 1)}
            if round_trip:
                worst = max(max(entry.values()) for entry in round_trip)
                cost[name]["latent_round_trip_worst"] = float(worst)
                cost[name]["round_trip_note"] = (
                    "sampling is the INVERSE of the flow; above ~1e-3 the molecules are "
                    "not drawn from the trained density and nothing else here is valid")
                logger.info(f"[{name}] worst latent round-trip relative error "
                            f"{worst:.2e}"
                            + ("  <- TOO LARGE: samples are not from the trained density"
                               if worst > 1e-3 else ""))
            logger.info(f"[{name}] {cost[name]['seconds_per_molecule'] * 1000:.2f} ms per "
                        f"molecule over {produced} samples (includes RDKit)")

        # ---- aggregate ----------------------------------------------------
        logger.header("aggregate")
        summary: dict[str, Any] = {}
        for name, arm in arms.items():
            decoders = (("raw", "degree") if arm["kind"] == "embedding"
                        else GRAPH_DECODERS)
            summary[name] = {"kind": arm["kind"], "decoders": {}}
            for decoder in decoders:
                rows = [{"n": c["n"], "requested_qed": c["requested_qed"],
                         "temperature": c["temperature"],
                         "metrics": c["decoders"][decoder]["qed_tol_10"]}
                        for c in cells[name]]
                achievement = qed_achievement_table(rows, tolerance=0.1)
                records = [r for r in population[name] if decoder in r]
                metrics = decoder_metrics(records, decoder, train_smiles=train_smiles,
                                          reference_smiles={
                                              "train_and_val":
                                              train_smiles | validation_smiles},
                                          diversity=False)
                smiles = sorted({r[decoder]["smiles"] for r in records
                                 if r[decoder].get("valid") and r[decoder].get("smiles")})
                diversity = internal_diversity(smiles) if len(smiles) > 1 else {}
                overall = achievement.get("overall") or {}
                summary[name]["decoders"][decoder] = {
                    "strict_validity": metrics["strict_validity"]["value"],
                    "sanitizable": metrics["sanitizable"]["value"],
                    "uniqueness": metrics["uniqueness"]["value"],
                    "novelty_vs_train": metrics["novelty_vs_train"]["value"],
                    "internal_diversity": diversity.get("internal_diversity"),
                    "repair_rate": metrics["fallback_rate"]["value"],
                    "qed_mae_weighted": overall.get("mae_weighted"),
                    "hit_rate_tol_10": overall.get("hit_rate_weighted"),
                    "hit_rate_tol_05": _mean(
                        [c["decoders"][decoder]["qed_tol_05"]["hit_rate"]
                         for c in cells[name]]),
                    "within_n_correlation": overall.get("qed_independent_of_n"),
                    "response_slope": response_slope(cells[name], decoder,
                                                     resamples=resamples,
                                                     seed=arguments.seed),
                    "aromatic_rings_mean": _mean(
                        [c["decoders"][decoder]["aromatic_rings_mean"]
                         for c in cells[name]]),
                    "rings_mean": _mean([c["decoders"][decoder]["rings_mean"]
                                         for c in cells[name]]),
                    "exact_n_rate": _mean([c["decoders"][decoder]["exact_n_rate"]
                                           for c in cells[name]]),
                    "valid_molecules": metrics["valid_molecules"],
                    "achievement": achievement,
                }
                block = summary[name]["decoders"][decoder]
                slope = block["response_slope"]
                logger.info(
                    f"[{name}/{decoder}] raw-or-strict validity "
                    f"{block['strict_validity']:.4f} | QED MAE "
                    f"{block['qed_mae_weighted']} | hit10 {block['hit_rate_tol_10']} | "
                    f"slope {slope['slope']:+.4f} "
                    f"[{slope['ci_low']:+.4f}, {slope['ci_high']:+.4f}]"
                    + ("  <- excludes zero" if slope["excludes_zero"] else "")
                    + f" | within-N corr {block['within_n_correlation']}")

        # ---- likelihoods, labelled and not compared -----------------------
        likelihood = {
            "graph_flow": {
                "representation": "dequantized discrete graph (atoms + upper-triangle bonds)",
                "validation_nll_per_dim": (meta.get("history") or [{}])[-1].get(
                    "nll_per_dim") if meta.get("history") else None,
                "validation_discrete_bound_per_dim": (meta.get("history") or [{}])[-1].get(
                    "bound_per_dim") if meta.get("history") else None,
                "condition_gap_per_atom": (meta.get("history") or [{}])[-1].get(
                    "condition_gap_per_atom") if meta.get("history") else None,
                "condition_win_rate": (meta.get("history") or [{}])[-1].get(
                    "condition_win_rate") if meta.get("history") else None,
            },
            "warning": ("these are NOT comparable with the embedding flow's density: a "
                        "dequantized discrete-graph bound and a continuous density over "
                        "learned node embeddings are different base measures over "
                        "different variables. They are reported side by side and compared "
                        "to nothing."),
        }
        if baseline is not None:
            likelihood["baseline"] = {
                "representation": "continuous density over learned node embeddings",
                "step": baseline["meta"].get("step"),
                "note": "read from the stage-17 checkpoint; see the warning above",
            }

        payload = {
            "stage": 22,
            "graph_flow": {"checkpoint": str(arguments.graph_flow_checkpoint),
                           "step": meta.get("step"),
                           "provenance": meta.get("provenance"),
                           "model_config": (meta.get("train_config") or {})},
            "null_checkpoint": arguments.null_checkpoint,
            "baseline_checkpoint": arguments.baseline_flow_checkpoint,
            "dataset_fingerprint": fingerprint,
            "grid": {"target_n": targets_n, "target_qed": targets_qed,
                     "temperatures": temperatures, "count": int(arguments.count),
                     "seed": int(arguments.seed)},
            "support": support,
            "arms": {name: {"kind": arm["kind"],
                            "prior_only": bool(arm.get("prior_only"))}
                     for name, arm in arms.items()},
            "cells": {name: [{k: v for k, v in cell.items() if k != "realised"}
                             for cell in rows] for name, rows in cells.items()},
            "real_structure_reference": real_reference,
            "summary": summary,
            "cost": cost,
            "likelihood": likelihood,
            "guarantees": {
                "matched": "same grid, counts, seeds and temperatures for every arm",
                "no_selection": ("one latent draw yields one molecule; nothing generates "
                                 "a batch and keeps the best QED"),
                "no_oracle": ("the requested QED is never compared with RDKit to choose a "
                              "sample; RDKit only sanitises and measures"),
                "raw_separate": ("raw validity is the argmax graph with no chemistry and "
                                 "is never merged with post-correction validity"),
                "no_test_data": "the test split is never loaded; novelty uses TRAIN only",
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_write_json(payload, output_root / "graph_flow_evaluation.json")
        (output_root / "graph_flow_evaluation.txt").write_text(_report(payload),
                                                               encoding="utf-8")
        logger.info(f"wrote {output_root / 'graph_flow_evaluation.json'} and "
                    f"{output_root / 'graph_flow_evaluation.txt'}")
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


@torch.no_grad()
def _baseline_sample(arm: dict[str, Any], *, scaler, n: int, qed: float, count: int,
                     temperature: float, seed: int, device) -> list[dict[str, Any]]:
    """The stage-17 embedding GNF, sampled under identical requests."""
    from zinc_gnf.conditioning import standardise_condition
    from zinc_gnf.generation import NodeCountSampler, generate_molecules

    flow = arm["flow"]
    counts = [int(n)] * int(count)
    values = standardise_condition(flow.config.condition_features, scaler=scaler,
                                   n=counts, qed=[float(qed)] * int(count))
    sampler = NodeCountSampler([int(n)])
    output = generate_molecules(
        flow, arm["autoencoder"], count=int(count), node_counts=sampler,
        temperature=float(temperature), seed=int(seed), decoders=("raw", "degree"),
        condition_values=values, target_n=int(n), device=device)
    records = output["records"]
    for record in records:
        for decoder in ("raw", "degree"):
            entry = record.get(decoder) or {}
            entry.setdefault("aromatic_rings", None)
            entry["exact_n"] = bool(entry.get("valid")) and _heavy(entry.get("smiles")) == int(n)
        record.setdefault("requested_n", int(n))
    _add_aromatic(records, ("raw", "degree"))
    return records


def _heavy(smiles: str | None) -> int:
    from rdkit import Chem, rdBase

    if not smiles:
        return -1
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    return int(molecule.GetNumHeavyAtoms()) if molecule is not None else -1


def _add_aromatic(records: list[dict[str, Any]], decoders) -> None:
    from rdkit import Chem, rdBase

    for record in records:
        for decoder in decoders:
            entry = record.get(decoder) or {}
            smiles = entry.get("smiles")
            if not smiles:
                continue
            with rdBase.BlockLogs():
                molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                continue
            info = molecule.GetRingInfo()
            entry["aromatic_rings"] = sum(
                1 for ring in info.AtomRings()
                if all(molecule.GetAtomWithIdx(i).GetIsAromatic() for i in ring))
            entry.setdefault("rings", int(info.NumRings()))


def _report(payload: dict[str, Any]) -> str:
    def show(value: Any, template: str = "{:.4f}") -> str:
        if value is None:
            return "N/A"
        try:
            if not np.isfinite(float(value)):
                return "N/A"
            return template.format(float(value))
        except (TypeError, ValueError):
            return str(value)

    lines = ["Stage 22: direct conditional graph flow vs the embedding GNF", "",
             f"graph flow : {payload['graph_flow']['checkpoint']} "
             f"(step {payload['graph_flow']['step']})",
             f"null arm   : {payload.get('null_checkpoint')}",
             f"baseline   : {payload.get('baseline_checkpoint')}",
             f"grid       : N={payload['grid']['target_n']} x "
             f"QED={payload['grid']['target_qed']} x T={payload['grid']['temperatures']} "
             f"x {payload['grid']['count']} samples, seed {payload['grid']['seed']}", ""]

    lines += ["=" * 108,
              "A. THE HEADLINE - does realised QED follow the request? "
              "(slope 1.0 = perfect, 0.0 = none)",
              "=" * 108,
              f"{'arm':<19}{'decoder':<11}{'slope':>9}{'95% CI':>22}"
              f"{'within-N corr':>15}{'QED MAE':>10}{'hit.10':>8}{'hit.05':>8}"]
    for arm, block in payload["summary"].items():
        for decoder, entry in block["decoders"].items():
            slope = entry["response_slope"]
            interval = (f"[{slope['ci_low']:+.4f}, {slope['ci_high']:+.4f}]"
                        if np.isfinite(slope.get("ci_low", float("nan"))) else "n/a")
            lines.append(
                f"{arm:<19}{decoder:<11}{show(slope['slope'], '{:+.4f}'):>9}{interval:>22}"
                f"{show(entry['within_n_correlation'], '{:+.4f}'):>15}"
                f"{show(entry['qed_mae_weighted']):>10}"
                f"{show(entry['hit_rate_tol_10'], '{:.3f}'):>8}"
                f"{show(entry['hit_rate_tol_05'], '{:.3f}'):>8}")
    lines += ["", "READ THIS TABLE FIRST, AND READ IT AGAINST THE NULL ARM.",
              "`graph_flow_null` is the same architecture trained on QED permuted within "
              "N. If the real arm's",
              "slope does not separate from it, the model learned nothing about QED and "
              "the rest is irrelevant.",
              "`graph_flow_prior` has the condition switched off INSIDE the couplings: if "
              "it matches the full",
              "arm, the model is only shifting its prior - the artefact stage 21 "
              "produced - not controlling QED.", ""]

    lines += ["=" * 108, "B. validity and structure - raw is the ARCHITECTURAL number",
              "=" * 108,
              f"{'arm':<19}{'decoder':<11}{'validity':>10}{'sanitizable':>12}"
              f"{'exact N':>9}{'arom.rings':>11}{'rings':>8}{'repair':>8}"
              f"{'unique':>8}{'novel':>8}{'diversity':>10}{'ms/mol':>8}"
              f"{'round-trip':>12}"]
    for arm, block in payload["summary"].items():
        per_molecule = (payload["cost"].get(arm) or {}).get("seconds_per_molecule")
        for decoder, entry in block["decoders"].items():
            lines.append(
                f"{arm:<19}{decoder:<11}{show(entry['strict_validity']):>10}"
                f"{show(entry['sanitizable'], '{:.3f}'):>12}"
                f"{show(entry['exact_n_rate'], '{:.3f}'):>9}"
                f"{show(entry['aromatic_rings_mean'], '{:.3f}'):>11}"
                f"{show(entry['rings_mean'], '{:.2f}'):>8}"
                f"{show(entry['repair_rate'], '{:.3f}'):>8}"
                f"{show(entry['uniqueness'], '{:.3f}'):>8}"
                f"{show(entry['novelty_vs_train'], '{:.3f}'):>8}"
                f"{show(entry['internal_diversity'], '{:.3f}'):>10}"
                f"{show(None if per_molecule is None else per_molecule * 1000, '{:.1f}'):>8}"
                f"{show((payload['cost'].get(arm) or {}).get('latent_round_trip_worst'), '{:.1e}'):>12}")
    lines += ["", "graph flow: `raw` is the argmax of the generated tensors with NO "
                  "chemistry - directly",
              "comparable with 'validity without check' in the literature (MoFlow "
              "reported 82% and GraphAF 68%",
              "on ZINC250K). `corrected` applies MoFlow's post-hoc valency correction, "
              "which keeps the largest",
              "connected component and can therefore DROP ATOMS - which is why exact N is "
              "reported for both.",
              "baseline: `raw` is the decoder's argmax, `degree` its headline "
              "degree-constrained assembly.",
              "round-trip is the worst latent round-trip relative error over the grid. "
              "SAMPLING IS THE INVERSE of",
              "the flow, so above ~1e-3 the molecules are not drawn from the trained "
              "density and no other number",
              "in this report means anything. Check it before reading table A.", ""]

    lines += ["=" * 108,
              "C. STRUCTURE of the RAW graphs against real molecules at the same N",
              "=" * 108,
              f"{'arm':<19}{'T':>5}{'N':>4}{'bonds':>8}{'real':>7}{'comps':>7}"
              f"{'largest':>9}{'over-val':>9}{'raw valid':>10}{'lost':>7}{'carbon':>8}"
              f"{'real C':>8}"]
    for arm, rows in (payload.get("cells") or {}).items():
        for cell in rows:
            structure = cell.get("structure")
            if not structure:
                continue
            reference = cell.get("real_reference") or {}
            lines.append(
                f"{arm:<19}{cell['temperature']:>5.2g}{cell['n']:>4}"
                f"{show(structure['bonds_mean'], '{:.1f}'):>8}"
                f"{show(reference.get('bonds_mean'), '{:.1f}'):>7}"
                f"{show(structure['components_mean'], '{:.2f}'):>7}"
                f"{show(structure['largest_component_mean'], '{:.1f}'):>9}"
                f"{show(structure['over_valent_atoms_mean'], '{:.2f}'):>9}"
                f"{show(structure['raw_valid_rate'], '{:.4f}'):>10}"
                f"{show(structure['correction_atoms_lost_mean'], '{:.1f}'):>7}"
                f"{show(structure['atom_type_fraction'].get('C'), '{:.3f}'):>8}"
                f"{show((reference.get('atom_type_fraction') or {}).get('C'), '{:.3f}'):>8}")
    lines += ["", "This is the table stage 23 exists to move. Stage 22 produced 17.4 bonds "
                  "against 24.7 real at",
              "N=23 with 7.00 components, and 74.2% of graphs were over-valent AND "
              "disconnected at once - so",
              "'bonds' must rise TOWARDS 'real' while 'over-val' does NOT: buying validity "
              "by adding bonds",
              "indiscriminately would show up here as both columns climbing together.",
              "'lost' is the atoms the corrector drops, which is what caps exact N before "
              "anything else.", ""]

    lines += ["=" * 108, "D. support - every count with its denominator", "=" * 108]
    for row in payload["support"]:
        parts = ", ".join(f"QED {k}: {v['molecules_within_window']:,}"
                          for k, v in row["by_qed"].items())
        lines.append(f"  N={row['n']:<3} {row['molecules_at_this_n']:,} training "
                     f"molecules; {parts}; minimum {row['minimum_over_qed_values']:,}")
    lines += [""]

    lines += ["=" * 108, "E. likelihoods - LABELLED, AND COMPARED TO NOTHING", "=" * 108]
    for key, block in payload["likelihood"].items():
        if key == "warning":
            continue
        lines.append(f"  [{key}] {block.get('representation')}")
        for field, value in block.items():
            if field == "representation":
                continue
            lines.append(f"      {field:<38}{show(value)}")
    lines += ["", f"  {payload['likelihood']['warning']}", ""]

    lines += ["=" * 108, "guarantees", "=" * 108]
    lines += [f"  - {key}: {value}" for key, value in payload["guarantees"].items()]
    lines += ["", "This is a PROOF OF CONCEPT at a bounded budget. If the condition-gap "
                  "trend in the training",
              "summary says 'rising_undertrained', these numbers describe an undertrained "
              "model and say nothing",
              "about the architecture. Nothing here establishes deployment readiness.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
