#!/usr/bin/env python3
"""Stage 25 — atoms-first against bonds-first, under identical requests.

THE QUESTION. Stage 24 established that the bonds-first flow ran its bond stack with no
atom information at all, while valence is a constraint between an atom's type and the sum
of its incident bond orders. Atoms-first reverses the factorisation so the bond
conditioner receives endpoint identity, formal charge and allowed valence. Both orderings
are exact factorisations of the same joint - the hypothesis is about learnability and
sampling quality, not about expressive power - so the answer has to be measured.

THE ARMS, all sampled natively from p(G | N, QED) with no guidance and no selection:

  atoms_first        the stage-25 arm, p(X|c) p(A|X,c)
  atoms_first_null   the SAME architecture, budget and seed with QED PERMUTED WITHIN N.
                     The null for every QED number the new architecture reports.
  bonds_first        the stage-23 18,000-step checkpoint, p(A|c) p(X|A,c). READ-ONLY, and
                     the thing stage 25 has to beat.

WHAT IS REPORTED, per arm and per cell: native raw validity and exact N; valence failure
rates by element and formal charge; connected components; bond counts and bond-order
distributions; atom-type distributions; aromatic rings and Kekule alternation; ring sizes;
corrected validity with the bonds it modified and the atoms it LOST; uniqueness, novelty
and internal diversity; QED MAE, hit rate and response slope - both unrestricted and
restricted to valid exact-N molecules with their support counts; the conditional-likelihood
permutation gap; the inverse round-trip error; and the validation likelihood.

The verdict block applies stage 25's acceptance criteria conjunctively: a raw-validity
gain is not a result if the atom distribution collapsed or exact N was traded away, and
aromaticity is a SEPARATE criterion rather than something valence may stand in for. No
QED-oracle selection or reranking anywhere. The test split is never loaded.

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
from zinc_gnf.factorisation_comparison import (  # noqa: E402
    HEADLINE_DECODER,
    arm_report,
    compare,
    likelihood_report,
)
from zinc_gnf.trainer_graph_flow import load_graph_flow  # noqa: E402
from zinc_gnf.utils.logging import JobLogger  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402
from zinc_gnf.utils.seed import set_seed  # noqa: E402
from zinc_gnf.valence_diagnostics import real_reference_profile  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--atoms-first-checkpoint", required=True, help="READ-ONLY")
    parser.add_argument("--bonds-first-checkpoint", required=True,
                        help="the stage-23 baseline (READ-ONLY)")
    parser.add_argument("--null-checkpoint", default=None,
                        help="the atoms-first --shuffle-qed arm; without it the QED "
                             "criterion cannot be judged and the report says so")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="data/processed/zinc250k_charge_aware_moflow_fixed")
    parser.add_argument("--target-n", default="15,23,30")
    parser.add_argument("--target-qed", default="0.4,0.6,0.8")
    parser.add_argument("--temperature", default="1.0,0.85")
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--qed-window", type=float, default=0.05)
    parser.add_argument("--val-molecules", type=int, default=1024)
    parser.add_argument("--real-molecules", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


def _table(payload: dict[str, Any]) -> str:
    """The side-by-side the stage is read from. Raw and corrected never merged."""
    arms = payload["arms"]
    order = [name for name in ("bonds_first", "atoms_first", "atoms_first_null")
             if name in arms]
    lines = ["=" * 104,
             "STAGE 25: atoms-first against bonds-first, identical requests",
             "=" * 104, ""]

    def row(title: str, getter, template: str = "{:.4f}") -> str:
        cells = []
        for name in order:
            try:
                value = getter(arms[name])
            except (KeyError, TypeError):
                value = None
            cells.append("n/a" if value is None or (isinstance(value, float)
                                                    and not np.isfinite(value))
                         else template.format(value))
        return f"  {title:<42}" + "".join(f"{cell:>20}" for cell in cells)

    lines += ["A. NATIVE generation - argmax graphs, no chemistry applied",
              "  " + " " * 42 + "".join(f"{name:>20}" for name in order),
              row("raw validity", lambda a: a["raw_valid_rate"]),
              row("exact N | valid (native)",
                  lambda a: a[HEADLINE_DECODER]["exact_n_rate"]),
              row("over-valent atoms / graph", lambda a: a["over_valent_atoms_mean"],
                  "{:.3f}"),
              row("over-valent atom rate", lambda a: a["over_valent_atom_rate"]),
              row("infeasible for any atom type", lambda a: a["infeasible_atom_rate"]),
              row("bonds / graph", lambda a: a["bonds_mean"], "{:.2f}"),
              row("real bonds / graph", lambda a: a["real_bonds_mean"], "{:.2f}"),
              row("connected components", lambda a: a["components_mean"], "{:.3f}"),
              row("uniqueness (raw)", lambda a: a[HEADLINE_DECODER]["uniqueness"]),
              row("novelty vs train (raw)",
                  lambda a: a[HEADLINE_DECODER]["novelty_vs_train"]),
              row("internal diversity (raw)",
                  lambda a: a[HEADLINE_DECODER]["internal_diversity"]),
              "",
              "B. AROMATICITY - a separate criterion, not implied by valence",
              row("aromatic rings / raw graph",
                  lambda a: a["aromatic_rings_per_graph_raw"]),
              row("aromatic rings / valid molecule",
                  lambda a: a[HEADLINE_DECODER]["aromatic_rings_per_valid"], "{:.3f}"),
              row("rings / graph", lambda a: a["rings_per_graph"], "{:.3f}"),
              row("alternating share of 5-/6-rings",
                  lambda a: a["alternating_ring_share"]),
              "",
              "C. AFTER CORRECTION - reported, never the headline",
              row("corrected validity",
                  lambda a: a["corrected"]["strict_validity"]),
              row("corrected exact N", lambda a: a["corrected_exact_n_rate"]),
              row("atoms LOST to correction",
                  lambda a: a["correction_atoms_lost_mean"], "{:.3f}"),
              row("bond orders removed",
                  lambda a: a["correction_orders_removed_mean"], "{:.3f}"),
              row("aromatic rings after correction",
                  lambda a: a["aromatic_rings_per_graph_corrected"]),
              "",
              "D. QED - restricted to VALID, EXACT-N molecules, with their support",
              row("exact-N valid molecules",
                  lambda a: a[HEADLINE_DECODER]["valid_exact_n"]["support_molecules"],
                  "{:.0f}"),
              row("QED MAE (valid exact-N)",
                  lambda a: a[HEADLINE_DECODER]["valid_exact_n"]["qed_mae"]),
              row("hit +-0.1 (valid exact-N)",
                  lambda a: a[HEADLINE_DECODER]["valid_exact_n"]["hit_rate_tol_10"]),
              row("response slope (valid exact-N)",
                  lambda a: a[HEADLINE_DECODER]["valid_exact_n"]["response_slope"]["slope"],
                  "{:+.4f}"),
              row("slope CI low",
                  lambda a: a[HEADLINE_DECODER]["valid_exact_n"]["response_slope"]["ci_low"],
                  "{:+.4f}"),
              row("slope CI high",
                  lambda a: a[HEADLINE_DECODER]["valid_exact_n"]["response_slope"]["ci_high"],
                  "{:+.4f}"),
              row("QED MAE (all valid, unrestricted)",
                  lambda a: a[HEADLINE_DECODER]["qed_mae"]),
              "",
              "E. LIKELIHOOD and NUMERICS - same validation subset for every arm",
              row("validation nll / dim",
                  lambda a: a["likelihood"]["validation_nll_per_dim"]),
              row("discrete bound / dim",
                  lambda a: a["likelihood"]["validation_discrete_bound_per_dim"]),
              row("condition gap / atom",
                  lambda a: a["likelihood"]["condition_gap_per_atom"], "{:+.5f}"),
              row("condition gap sem",
                  lambda a: a["likelihood"]["condition_gap_sem"], "{:.5f}"),
              row("condition win rate",
                  lambda a: a["likelihood"]["condition_win_rate"], "{:.3f}"),
              row("inverse round trip (worst)",
                  lambda a: a["latent_round_trip_worst"], "{:.2e}"),
              row("parameters", lambda a: a["parameters"]["total"], "{:.0f}"),
              ""]

    ceiling = payload.get("real_ceiling") or {}
    patterns = (ceiling.get("rings") or {}).get("pattern_distribution_5_and_6") or {}
    lines += ["F. the REAL ceiling, through our own reconstruction path",
              f"  aromatic rings / molecule       "
              f"{ceiling.get('aromatic_rings_per_molecule', float('nan')):.3f}",
              f"  molecules reconstructable       "
              f"{ceiling.get('reconstructable_rate', float('nan')):.3f}",
              f"  alternating share of 5-/6-rings "
              f"{patterns.get('alternating', float('nan')):.3f}",
              ""]

    verdict = payload.get("verdict") or {}
    lines += ["=" * 104, f"VERDICT: {verdict.get('verdict', 'n/a')}", "=" * 104,
              f"  {verdict.get('reading', '')}", ""]
    for name, check in (verdict.get("checks") or {}).items():
        mark = {True: "PASS", False: "FAIL", None: "----"}[check.get("passed")]
        lines.append(f"  [{mark}] {name}: {check.get('detail', '')}")
    lines += ["", f"  {verdict.get('warning', '')}", "=" * 104]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    arguments = parse_args(argv)
    output_root = Path(arguments.output_dir)
    try:
        targets_n = [int(v) for v in str(arguments.target_n).split(",") if v.strip()]
        targets_qed = [float(v) for v in str(arguments.target_qed).split(",") if v.strip()]
        temperatures = [float(v) for v in str(arguments.temperature).split(",") if v.strip()]
        if not targets_n or not targets_qed or not temperatures:
            raise ValueError("--target-n, --target-qed and --temperature must be non-empty")
        if arguments.count < 1:
            raise ValueError("--count must be positive")
        output_root.mkdir(parents=True, exist_ok=True)
        logger = JobLogger("stage25", output_root, filename="comparison.log")
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4

    started = time.perf_counter()
    try:
        set_seed(arguments.seed, deterministic=False)
        device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu")
            if arguments.device == "auto" else arguments.device)
        scaler = load_condition_scaler(arguments.data_dir)
        train_records, val_records, fingerprint = load_training_splits(arguments.data_dir)
        train_smiles = {r["smiles"] for r in train_records if r.get("smiles")}
        validation_smiles = {r["smiles"] for r in val_records if r.get("smiles")}

        requested = {"atoms_first": arguments.atoms_first_checkpoint,
                     "bonds_first": arguments.bonds_first_checkpoint}
        if arguments.null_checkpoint:
            requested["atoms_first_null"] = arguments.null_checkpoint

        logger.header("Stage 25: atoms-first against bonds-first")
        loaded: dict[str, Any] = {}
        for name, path in requested.items():
            model, meta = load_graph_flow(Path(path), device=device)
            loaded[name] = {"model": model, "meta": meta, "path": str(path)}
            logger.info(f"{name}: {path} step={meta.get('step')} "
                        f"factorisation={model.config.factorisation} "
                        f"conditioner={model.config.bond_conditioner} "
                        f"parameters={model.parameter_report()['total']:,} (READ-ONLY)")
        if loaded["atoms_first"]["model"].config.factorisation != "atoms_first":
            raise ValueError(
                f"--atoms-first-checkpoint is "
                f"{loaded['atoms_first']['model'].config.factorisation!r}, not "
                f"'atoms_first'; the comparison would be between two identical orderings")
        if loaded["bonds_first"]["model"].config.factorisation != "bonds_first":
            raise ValueError("--bonds-first-checkpoint is not a bonds-first model")
        if "atoms_first_null" not in loaded:
            logger.warn("no null arm supplied: the QED criterion cannot be judged, and "
                        "the verdict block will say so rather than pass it by default")

        support = support_counts(MolecularGraphDataset(train_records),
                                 n_values=targets_n, qed_values=targets_qed,
                                 window=float(arguments.qed_window))
        for row in support:
            logger.info(f"support N={row['n']}: {row['molecules_at_this_n']:,} training "
                        f"molecules, fewest within +-{arguments.qed_window} of a target: "
                        f"{row['minimum_over_qed_values']:,}")
        real_ceiling = real_reference_profile(train_records,
                                             limit=int(arguments.real_molecules))
        logger.info(f"REAL ceiling: {real_ceiling['aromatic_rings_per_molecule']:.3f} "
                    f"aromatic rings per molecule, "
                    f"{real_ceiling['reconstructable_rate'] * 100:.1f}% reconstructable "
                    f"through our own path")

        logger.header("the grid, identical for every arm")
        arms: dict[str, Any] = {}
        for name, entry in loaded.items():
            report = arm_report(
                entry["model"], scaler=scaler, label=name, targets_n=targets_n,
                targets_qed=targets_qed, temperatures=temperatures,
                count=int(arguments.count), seed=int(arguments.seed),
                train_records=train_records, train_smiles=train_smiles,
                validation_smiles=validation_smiles,
                resamples=int(arguments.bootstrap_resamples),
                device=str(device), logger=logger)
            report["checkpoint"] = entry["path"]
            report["step"] = entry["meta"].get("step")
            report["provenance"] = entry["meta"].get("provenance")
            report["likelihood"] = likelihood_report(
                entry["model"], val_records=val_records,
                output_dir=output_root / "likelihood" / name,
                val_molecules=int(arguments.val_molecules), batch_size=32,
                seed=int(arguments.seed), device=str(device))
            arms[name] = report
            headline = report[HEADLINE_DECODER]
            logger.info(
                f"[{name}] RAW validity {report['raw_valid_rate']:.4f} | exact-N | valid "
                f"{headline['exact_n_rate']:.4f} | aromatic rings/graph "
                f"{report['aromatic_rings_per_graph_raw']:.4f} | alternating share "
                f"{report['alternating_ring_share']:.4f} | exact-N valid QED support "
                f"{headline['valid_exact_n']['support_molecules']} | slope "
                f"{headline['valid_exact_n']['response_slope']['slope']:+.4f} | "
                f"nll/dim {report['likelihood']['validation_nll_per_dim']:.4f} | "
                f"cond gap {report['likelihood']['condition_gap_per_atom']:+.5f} "
                f"+-{report['likelihood']['condition_gap_sem']:.5f} | round trip "
                f"{report['latent_round_trip_worst']:.2e}")

        verdict = compare(arms, real_ceiling=real_ceiling)
        payload = {
            "stage": 25,
            "question": ("does conditioning bond generation on atom chemistry improve "
                         "joint chemical compatibility, aromaticity and native validity "
                         "while keeping genuine QED conditioning?"),
            "dataset_fingerprint": fingerprint,
            "grid": {"target_n": targets_n, "target_qed": targets_qed,
                     "temperatures": temperatures, "count": int(arguments.count),
                     "seed": int(arguments.seed)},
            "support": support,
            "real_ceiling": real_ceiling,
            "arms": arms,
            "verdict": verdict,
            "guarantees": {
                "matched": ("same grid, counts, seeds, temperatures and validation "
                            "subset for every arm"),
                "no_selection": ("one latent draw yields one molecule; nothing generates "
                                 "a batch and keeps the best QED"),
                "no_oracle": ("the requested QED is never compared with RDKit to choose "
                              "a sample; RDKit only sanitises and measures"),
                "raw_separate": ("raw validity is the argmax graph with no chemistry and "
                                 "is never merged with post-correction validity"),
                "no_test_data": "the test split is never loaded; novelty uses TRAIN only",
                "read_only": "every checkpoint is opened read-only and none is written",
                "exact_factorisations": ("both orderings are exact factorisations of the "
                                         "same joint; this measures learnability, not "
                                         "expressive power"),
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_write_json(payload, output_root / "factorisation_comparison.json")
        text = _table(payload)
        (output_root / "factorisation_comparison.txt").write_text(text, encoding="utf-8")
        logger.info(f"wrote {output_root / 'factorisation_comparison.json'}")
        logger.info(f"wrote {output_root / 'factorisation_comparison.txt'}")
        for line in text.splitlines():
            logger.info(line)
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


if __name__ == "__main__":
    raise SystemExit(main())
