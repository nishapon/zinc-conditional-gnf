#!/usr/bin/env python3
"""Stage 24 — why are the bond counts right and the molecules still invalid?

Stage 23's topology conditioner worked: aggregate raw validity 29.73%, corrected exact-N
79.5%, and at N=23 the generated bond count is 24.45 against 24.65 real with 1.20
components. What remains is local: 89.5% of graphs at N=23 contain an over-valent atom
(96.9% at N=30), and aromatic rings sit at 0.008 per corrected molecule.

This job measures the causes. Read-only, CPU-only, no training, no checkpoint written.

  A. VALENCE. Over-valence by element, formal charge and allowed valence; bond-order sum
     histograms per atom type; excess magnitude; the joint P(atom type, bond-order sum)
     against real ZINC250K; whether argmax shifts the atom marginal; and how all of it
     scales with N, requested QED and temperature.

  B. AROMATICITY. The ceiling first - aromatic rings in REAL training molecules through
     our own serialise/Kekulise/reconstruct path, because if real molecules lose
     aromaticity there, no model could keep it. Then ring-size distributions, the
     single/double alternation pattern in 5- and 6-rings, ring atom types, and RDKit's
     perception after sanitisation.

  C. ATTRIBUTION, by counterfactual on the existing checkpoint:
       best-case relabelling  keep the generated bonds, relabel atoms as favourably as
                              possible. If graphs become valence-feasible, the bond
                              pattern is realisable and the ATOM CHOICE is the failure -
                              which is what the bonds-first factorisation predicts, since
                              the bond block never sees atom types.
       oracle atoms           real atom types at the same N with generated bonds.
       oracle bonds           generated atoms with a real bond matrix: does the atom block
                              pick types that fit bonds it is given?
       correction damage      aromaticity before and after `valency_correction`, on the
                              same graphs. The rule deletes the LARGEST-order bond of an
                              over-valent atom, and in a Kekule ring those are the DOUBLE
                              bonds - so this separates "never made an aromatic ring" from
                              "the repair destroyed it".

The oracle arms use real TRAINING graphs as a structural reference. They are diagnostics,
never a generation aid, and never a property oracle: no molecule is scored or selected by
QED anywhere in this script.

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

from zinc_gnf.conditioning import load_condition_scaler, standardise_condition  # noqa: E402
from zinc_gnf.constants_charge_aware import ATOM_VOCAB  # noqa: E402
from zinc_gnf.dataset import load_training_splits  # noqa: E402
from zinc_gnf.graph_flow_generation import population_structure  # noqa: E402
from zinc_gnf.trainer_graph_flow import load_graph_flow  # noqa: E402
from zinc_gnf.utils.logging import JobLogger  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402
from zinc_gnf.utils.seed import set_seed  # noqa: E402
from zinc_gnf.valence_diagnostics import (  # noqa: E402
    argmax_shift,
    best_case_atom_assignment,
    correction_damage,
    joint_atom_bond_distribution,
    real_reference_profile,
    ring_profile,
    valence_profile,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="READ-ONLY")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="data/processed/zinc250k_charge_aware_moflow_fixed")
    parser.add_argument("--target-n", default="15,23,30")
    parser.add_argument("--target-qed", default="0.4,0.8")
    parser.add_argument("--temperature", default="1.0,0.85")
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--real-molecules", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args(argv)


@torch.no_grad()
def _sample(model, *, scaler, n: int, qed: float, count: int, temperature: float,
            seed: int) -> dict[str, Any]:
    counts = [int(n)] * int(count)
    values = torch.from_numpy(standardise_condition(
        model.config.condition_features, scaler=scaler, n=counts,
        qed=[float(qed)] * len(counts)))
    context = model.conditioning(values)
    sampled = model.sample(counts, context,
                           temperature=float(temperature),
                           generator=torch.Generator().manual_seed(int(seed)))
    return {"graphs": model.discrete_graphs(sampled), "sampled": sampled}


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
        logger = JobLogger("valence_audit", output_root, filename="audit.log")
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4

    started = time.perf_counter()
    try:
        set_seed(arguments.seed, deterministic=False)
        scaler = load_condition_scaler(arguments.data_dir)
        train_records, _, fingerprint = load_training_splits(arguments.data_dir)
        model, meta = load_graph_flow(Path(arguments.checkpoint), device="cpu")

        logger.header("Stage 24: valence and aromaticity audit")
        logger.info(f"checkpoint {arguments.checkpoint} step={meta.get('step')} (READ-ONLY)")
        logger.info(f"conditioner={model.config.bond_conditioner} "
                    f"max_log_scale={model.config.max_log_scale} "
                    f"noise_scale={model.config.noise_scale}")
        logger.info("no training, no checkpoint written, no molecule selected by property")

        # ---- B first: the ceiling ---------------------------------------
        logger.header("B. the aromaticity ceiling on REAL training molecules")
        real: dict[str, Any] = {}
        for n in [None, *targets_n]:
            profile = real_reference_profile(train_records, n=n,
                                             limit=arguments.real_molecules)
            real["all" if n is None else str(n)] = profile
            if profile.get("molecules"):
                logger.info(
                    f"REAL {'all N' if n is None else f'N={n}'}: "
                    f"{profile['molecules']} molecules, reconstructable "
                    f"{profile['reconstructable_rate'] * 100:.1f}%, aromatic rings "
                    f"{profile['aromatic_rings_per_molecule']:.3f}/molecule, "
                    f"{profile['molecules_with_an_aromatic_ring']} have one")
        ceiling = real.get("all", {})
        if ceiling.get("reconstructable_rate", 0) < 0.99:
            logger.warn("real molecules do NOT all survive our own reconstruction path; "
                        "that bounds every generated number and must be fixed first")
        else:
            logger.info("real molecules survive serialisation, Kekulisation and "
                        "reconstruction, so the representation can carry aromaticity and "
                        "the gap is the model or the repair")
        rings = (ceiling.get("rings") or {})
        logger.info(f"real ring patterns (5- and 6-rings): "
                    f"{rings.get('pattern_distribution_5_and_6')}")

        # ---- A and C over the grid ---------------------------------------
        logger.header("A and C. valence failure and its attribution, per cell")
        cells: list[dict[str, Any]] = []
        for temperature in temperatures:
            for n in targets_n:
                for qed in targets_qed:
                    produced = _sample(model, scaler=scaler, n=n, qed=qed,
                                       count=int(arguments.count),
                                       temperature=temperature, seed=arguments.seed)
                    graphs = produced["graphs"]
                    sampled = produced["sampled"]
                    reference = real.get(str(n)) or {}
                    valence = valence_profile(graphs)
                    best_case = best_case_atom_assignment(graphs)

                    # counterfactual: real atom types at this N, generated bonds
                    donors = [r for r in train_records if int(r["n_node"]) == int(n)]
                    oracle_atoms = None
                    if donors:
                        picker = np.random.default_rng(arguments.seed)
                        swapped = []
                        for graph in graphs:
                            donor = donors[int(picker.integers(0, len(donors)))]
                            swapped.append({
                                "atom_types": np.asarray(donor["atom_types"],
                                                         dtype=np.int64),
                                "bond_matrix": graph["bond_matrix"]})
                        oracle_atoms = {
                            "valence": valence_profile(swapped),
                            "structure": population_structure(swapped, requested_n=n),
                            "note": ("REAL atom types with GENERATED bonds; a diagnostic "
                                     "reference, not a generation method"),
                        }

                    # counterfactual: generated atoms, a real bond matrix
                    oracle_bonds = None
                    if donors:
                        picker = np.random.default_rng(arguments.seed + 1)
                        swapped = []
                        for graph in graphs:
                            donor = donors[int(picker.integers(0, len(donors)))]
                            swapped.append({
                                "atom_types": graph["atom_types"],
                                "bond_matrix": np.asarray(donor["bond_matrix"],
                                                          dtype=np.int64)})
                        oracle_bonds = {
                            "valence": valence_profile(swapped),
                            "structure": population_structure(swapped, requested_n=n),
                            "note": ("GENERATED atom types with a REAL bond matrix; shows "
                                     "whether the atom block picks types that fit bonds"),
                        }

                    cell = {
                        "n": int(n), "requested_qed": float(qed),
                        "temperature": float(temperature), "samples": len(graphs),
                        "structure": population_structure(graphs, requested_n=n),
                        "valence": valence,
                        "joint_atom_bond": joint_atom_bond_distribution(graphs),
                        "rings": ring_profile(graphs),
                        "best_case_relabelling": {
                            k: v for k, v in best_case.items() if k != "per_graph"},
                        "correction_damage": correction_damage(graphs),
                        "argmax_shift_atoms": argmax_shift(
                            sampled["atoms_continuous"].cpu().numpy(),
                            sampled["node_mask"].cpu().numpy(),
                            labels=list(ATOM_VOCAB)),
                        "argmax_shift_bonds": argmax_shift(
                            sampled["bonds_continuous"].cpu().numpy(),
                            sampled["pair_mask"].cpu().numpy(),
                            labels=["none", "single", "double", "triple"]),
                        "oracle_atoms_generated_bonds": oracle_atoms,
                        "generated_atoms_oracle_bonds": oracle_bonds,
                        "real_reference": {k: v for k, v in reference.items()
                                           if k in ("aromatic_rings_per_molecule",
                                                    "atom_distribution", "molecules")},
                    }
                    cells.append(cell)

                    structure = cell["structure"]
                    logger.info(
                        f"T={temperature:g} N={n} QED={qed:g}: raw valid "
                        f"{structure['raw_valid_rate'] * 100:.2f}%, bonds "
                        f"{structure['bonds_mean']:.2f}, over-valent atoms "
                        f"{structure['over_valent_atoms_mean']:.2f}, components "
                        f"{structure['components_mean']:.2f}")
                    logger.info(
                        f"    over-valent atom rate {valence['over_valent_atom_rate'] * 100:.1f}%, "
                        f"infeasible for ANY atom type "
                        f"{valence['infeasible_rate'] * 100:.2f}% "
                        f"(bond-order sum > {valence['max_valence_in_vocabulary']})")
                    worst = sorted(valence["by_element"].items(),
                                   key=lambda kv: -kv[1]["over_valent_rate"])[:4]
                    logger.info("    worst elements: " + ", ".join(
                        f"{k} {v['over_valent_rate'] * 100:.1f}% of {v['atoms']}"
                        for k, v in worst))
                    logger.info(
                        f"    COUNTERFACTUAL best-case relabelling: "
                        f"{best_case['valence_feasible_rate'] * 100:.1f}% of graphs become "
                        f"valence-feasible, relabelling "
                        f"{best_case['atoms_relabelled_rate'] * 100:.1f}% of atoms")
                    if oracle_atoms:
                        logger.info(
                            f"    COUNTERFACTUAL real atoms + generated bonds: over-valent "
                            f"atom rate "
                            f"{oracle_atoms['valence']['over_valent_atom_rate'] * 100:.1f}%, "
                            f"raw valid "
                            f"{oracle_atoms['structure']['raw_valid_rate'] * 100:.2f}%")
                    if oracle_bonds:
                        logger.info(
                            f"    COUNTERFACTUAL generated atoms + real bonds: over-valent "
                            f"atom rate "
                            f"{oracle_bonds['valence']['over_valent_atom_rate'] * 100:.1f}%, "
                            f"raw valid "
                            f"{oracle_bonds['structure']['raw_valid_rate'] * 100:.2f}%")
                    damage = cell["correction_damage"]
                    logger.info(
                        f"    aromatic rings {damage['aromatic_rings_before_per_graph']:.4f} "
                        f"-> {damage['aromatic_rings_after_per_graph']:.4f} per graph; "
                        f"lost to correction {damage['aromatic_lost_to_correction']}, "
                        f"alternating rings destroyed {damage['alternating_rings_destroyed']}")
                    patterns = cell["rings"]["pattern_distribution_5_and_6"]
                    logger.info(f"    ring patterns (5/6): {patterns}; rings per graph "
                                f"{cell['rings']['rings_per_graph']:.2f} vs real "
                                f"{(reference.get('rings') or {}).get('rings_per_graph')}")
                    shift = cell["argmax_shift_atoms"]
                    logger.info(
                        f"    argmax vs soft atom marginal: max difference "
                        f"{shift.get('max_absolute_difference', float('nan')):.4f} "
                        f"(carbon argmax "
                        f"{shift.get('argmax_distribution', {}).get('C', float('nan')):.3f} "
                        f"vs soft "
                        f"{shift.get('soft_distribution', {}).get('C', float('nan')):.3f})")

        payload = {
            "stage": 24,
            "checkpoint": str(arguments.checkpoint),
            "step": meta.get("step"),
            "model_config": model.config.to_dict(),
            "dataset_fingerprint": fingerprint,
            "grid": {"target_n": targets_n, "target_qed": targets_qed,
                     "temperatures": temperatures, "count": int(arguments.count),
                     "seed": int(arguments.seed)},
            "real_reference": real,
            "cells": cells,
            "guarantees": {
                "read_only": "no checkpoint is modified and nothing is trained",
                "no_property_oracle": ("no molecule is scored or selected by QED; the "
                                       "oracle arms substitute real STRUCTURES as a "
                                       "diagnostic reference only"),
                "no_test_data": "the test split is never loaded",
            },
            "reading": {
                "valence": ("infeasible_rate is the share of atoms whose bond-order sum "
                            "exceeds the largest cap in the vocabulary; those cannot be "
                            "repaired by relabelling and mean the BONDS are wrong"),
                "attribution": ("if best_case_relabelling.valence_feasible_rate is high "
                                "while raw validity is low, the bond pattern is realisable "
                                "and the ATOM CHOICE is the failure - which is what a "
                                "bonds-first factorisation predicts, because the bond "
                                "block never sees atom types"),
                "aromaticity": ("compare aromatic_rings_before_per_graph with the real "
                                "ceiling AND with after-correction: a large "
                                "alternating_rings_destroyed means the repair is the "
                                "proximate cause, a near-zero before-count means the model "
                                "never produced the alternation"),
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_write_json(payload, output_root / "valence_aromaticity_audit.json")
        logger.info(f"wrote {output_root / 'valence_aromaticity_audit.json'}")
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
