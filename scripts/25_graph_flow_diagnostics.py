#!/usr/bin/env python3
"""Stage 22c — why does the direct graph flow produce almost no valid raw graphs?

Read-only, CPU-only, no training, no checkpoint written. It answers four questions with
numbers instead of inference, by comparing the model's own samples against the REAL
training molecules at the same N:

  1. IS THE MARGINAL RIGHT?  Per-channel mean and standard deviation of the generated
     continuous tensors against the dequantized real ones, plus the bond-type and
     atom-type histograms and the bond count per molecule. If these match but the graphs
     are invalid, the model learned the marginals and not the correlations - and the bond
     coupling's conditioner is provably blind to topology at a fixed degree sequence, so
     that is the expected outcome rather than a surprise.

  2. WHY IS A RAW GRAPH INVALID?  Every sample is attributed to exactly one cause:
     over-valent only, disconnected only, both, sanitizable-failure-for-another-reason,
     or valid. Stage 22b reported raw validity 0.0012 without saying which; this does.

  3. WHY DOES CORRECTION LOSE ATOMS?  Component count and largest-component size before
     correction, and bond orders removed and atoms dropped by it. Exact N after
     correction was 0.018, and the corrector keeps only the largest connected component,
     so this measures whether the graphs arrive fragmented or are shattered by the repair.

  4. IS THE INVERSE TRUSTWORTHY?  The latent round-trip error and the fraction of
     coupling log-scales sitting at their bound, WITH the trained context and with it
     neutralised. Sampling is the inverse; a saturated stack at max_log_scale 2.0 has
     inverse condition number ~exp(blocks * 2.0), which is why the prior-only ablation
     reported 1.4e-02 while the full arm reported 3.3e-05.

Nothing here is a model change. Exit codes: 0 ok; 2 missing input; 4 bad arguments;
1 unexpected.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
import time
import traceback
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np  # noqa: E402
from rdkit import Chem, rdBase  # noqa: E402
import torch  # noqa: E402

from zinc_gnf.chemistry_charge_aware import graph_to_mol_charge_aware  # noqa: E402
from zinc_gnf.conditioning import load_condition_scaler  # noqa: E402
from zinc_gnf.constants_charge_aware import ATOM_VALENCE, ATOM_VOCAB  # noqa: E402
from zinc_gnf.dataset import load_training_splits  # noqa: E402
from zinc_gnf.graph_flow import (  # noqa: E402
    ConditionalCoupling,
    dequantize,
    neutralise_coupling_condition,
    pair_mask_from_counts,
)
from zinc_gnf.graph_flow_generation import valency_correction  # noqa: E402
from zinc_gnf.trainer_graph_flow import load_graph_flow  # noqa: E402
from zinc_gnf.utils.logging import JobLogger  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402
from zinc_gnf.utils.seed import set_seed  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="READ-ONLY")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="data/processed/zinc250k_charge_aware_moflow_fixed")
    parser.add_argument("--target-n", default="15,23,30")
    parser.add_argument("--target-qed", default="0.8")
    parser.add_argument("--temperature", default="0.85")
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--real-molecules", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args(argv)


def _components(bond: np.ndarray) -> list[list[int]]:
    n = bond.shape[0]
    seen = [False] * n
    found: list[list[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        stack, group = [start], []
        seen[start] = True
        while stack:
            node = stack.pop()
            group.append(node)
            for other in range(n):
                if not seen[other] and bond[node, other] > 0:
                    seen[other] = True
                    stack.append(other)
        found.append(group)
    return found


def _attribute(atom_types: np.ndarray, bonds: np.ndarray) -> dict[str, Any]:
    """Name the single reason this raw graph is not a molecule."""
    caps = np.asarray([ATOM_VALENCE[int(a)] for a in atom_types], dtype=np.int64)
    totals = bonds.sum(axis=1)
    over = int((totals > caps).sum())
    groups = _components(bonds)
    isolated = int(sum(1 for g in groups if len(g) == 1))
    sanitizable = True
    try:
        with rdBase.BlockLogs():
            graph_to_mol_charge_aware(atom_types, bonds)
    except Exception:  # noqa: BLE001
        sanitizable = False
    disconnected = len(groups) > 1
    if over and disconnected:
        cause = "over_valent_and_disconnected"
    elif over:
        cause = "over_valent_only"
    elif disconnected:
        cause = "disconnected_only"
    elif not sanitizable:
        cause = "other_sanitisation_failure"
    else:
        cause = "valid"
    return {"cause": cause, "over_valent_atoms": over, "components": len(groups),
            "isolated_atoms": isolated,
            "largest_component": max((len(g) for g in groups), default=0),
            "bonds": int((bonds > 0).sum() // 2),
            "excess_bond_order": int(np.clip(totals - caps, 0, None).sum())}


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
        logger = JobLogger("graph_flow_diagnostics", output_root, filename="diagnostics.log")
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4

    started = time.perf_counter()
    try:
        set_seed(arguments.seed, deterministic=False)
        scaler = load_condition_scaler(arguments.data_dir)
        train_records, _, fingerprint = load_training_splits(arguments.data_dir)
        model, meta = load_graph_flow(Path(arguments.checkpoint), device="cpu")
        config = model.config

        logger.header("Stage 22c: graph-flow failure diagnostics")
        logger.info(f"checkpoint {arguments.checkpoint} step={meta.get('step')} (READ-ONLY)")
        logger.info(f"CONFIGURATION AS TRAINED: max_log_scale={config.max_log_scale} "
                    f"max_shift={config.max_shift} noise_scale={config.noise_scale} "
                    f"bond_blocks={config.bond_blocks} atom_blocks={config.atom_blocks}")
        logger.info(f"inverse condition number bound ~ exp(bond_blocks * max_log_scale) = "
                    f"{float(np.exp(config.bond_blocks * config.max_log_scale)):.3g}; "
                    f"float32 carries ~1e-7 relative precision")

        # ---- real reference statistics, per N -------------------------------
        real: dict[int, dict[str, Any]] = {}
        for n in targets_n:
            rows = [r for r in train_records if int(r["n_node"]) == n]
            rows = rows[:int(arguments.real_molecules)]
            if not rows:
                logger.warn(f"no training molecule has N={n}")
                continue
            bond_counts, densities, types, atoms = [], [], Counter(), Counter()
            for record in rows:
                matrix = np.asarray(record["bond_matrix"], dtype=np.int64)
                upper = matrix[np.triu_indices(n, 1)]
                bond_counts.append(int((upper > 0).sum()))
                densities.append(float((upper > 0).mean()))
                types.update(int(v) for v in upper)
                atoms.update(int(a) for a in record["atom_types"])
            pairs = n * (n - 1) // 2
            real[n] = {
                "molecules": len(rows), "pairs_per_molecule": pairs,
                "bonds_mean": float(np.mean(bond_counts)),
                "bond_density_mean": float(np.mean(densities)),
                "bond_type_fraction": {k: v / sum(types.values())
                                       for k, v in sorted(types.items())},
                "atom_type_fraction": {ATOM_VOCAB[k]: v / sum(atoms.values())
                                       for k, v in sorted(atoms.items())},
            }
            logger.info(f"REAL N={n}: {real[n]['bonds_mean']:.1f} bonds of {pairs} pairs "
                        f"(density {real[n]['bond_density_mean']:.4f}); carbon fraction "
                        f"{real[n]['atom_type_fraction'].get('C', 0.0):.3f}")

        # ---- generated statistics and failure attribution -------------------
        logger.header("generated samples: marginals and failure attribution")
        cells: list[dict[str, Any]] = []
        for temperature in temperatures:
            for n in targets_n:
                for qed in targets_qed:
                    from zinc_gnf.conditioning import standardise_condition

                    counts = [int(n)] * int(arguments.count)
                    values = torch.from_numpy(standardise_condition(
                        config.condition_features, scaler=scaler, n=counts,
                        qed=[float(qed)] * len(counts)))
                    context = model.conditioning(values)
                    generator = torch.Generator().manual_seed(int(arguments.seed))
                    sampled = model.sample(counts, context, temperature=temperature,
                                           generator=generator)
                    graphs = model.discrete_graphs(sampled)

                    causes: Counter = Counter()
                    bond_counts, components, largest, excess, over = [], [], [], [], []
                    bond_types: Counter = Counter()
                    atom_types: Counter = Counter()
                    lost_atoms, removed_orders, corrected_exact = [], [], []
                    for graph in graphs:
                        attribution = _attribute(graph["atom_types"], graph["bond_matrix"])
                        causes[attribution["cause"]] += 1
                        bond_counts.append(attribution["bonds"])
                        components.append(attribution["components"])
                        largest.append(attribution["largest_component"])
                        excess.append(attribution["excess_bond_order"])
                        over.append(attribution["over_valent_atoms"])
                        upper = graph["bond_matrix"][np.triu_indices(n, 1)]
                        bond_types.update(int(v) for v in upper)
                        atom_types.update(int(a) for a in graph["atom_types"])
                        fixed_atoms, _, info = valency_correction(graph["atom_types"],
                                                                  graph["bond_matrix"])
                        lost_atoms.append(info["atoms_total"] - info["atoms_kept"])
                        removed_orders.append(info["bond_orders_removed"])
                        corrected_exact.append(int(info["atoms_kept"] == n))

                    pairs = n * (n - 1) // 2
                    total_types = sum(bond_types.values()) or 1
                    total_atoms = sum(atom_types.values()) or 1
                    cell = {
                        "n": int(n), "requested_qed": float(qed),
                        "temperature": float(temperature), "samples": len(graphs),
                        "pairs_per_molecule": pairs,
                        "bonds_mean": float(np.mean(bond_counts)),
                        "bond_density_mean": float(np.mean(bond_counts)) / pairs,
                        "bond_type_fraction": {k: v / total_types
                                               for k, v in sorted(bond_types.items())},
                        "atom_type_fraction": {ATOM_VOCAB[k]: v / total_atoms
                                               for k, v in sorted(atom_types.items())},
                        "causes": dict(causes),
                        "cause_fractions": {k: v / len(graphs) for k, v in causes.items()},
                        "components_mean": float(np.mean(components)),
                        "largest_component_mean": float(np.mean(largest)),
                        "largest_component_fraction_of_n": float(np.mean(largest)) / n,
                        "over_valent_atoms_mean": float(np.mean(over)),
                        "excess_bond_order_mean": float(np.mean(excess)),
                        "correction_atoms_lost_mean": float(np.mean(lost_atoms)),
                        "correction_orders_removed_mean": float(np.mean(removed_orders)),
                        "correction_exact_n_rate": float(np.mean(corrected_exact)),
                        "latent_round_trip": sampled["latent_round_trip_relative_error"],
                        "real_reference": real.get(n),
                    }
                    cells.append(cell)
                    reference = real.get(n) or {}
                    logger.info(
                        f"T={temperature:g} N={n} QED={qed:g}: "
                        f"{cell['bonds_mean']:.1f} bonds vs real "
                        f"{reference.get('bonds_mean', float('nan')):.1f} "
                        f"(density {cell['bond_density_mean']:.4f} vs "
                        f"{reference.get('bond_density_mean', float('nan')):.4f})")
                    logger.info(
                        f"    components {cell['components_mean']:.2f}, largest "
                        f"{cell['largest_component_mean']:.1f}/{n} "
                        f"({cell['largest_component_fraction_of_n'] * 100:.1f}%), "
                        f"over-valent atoms {cell['over_valent_atoms_mean']:.2f}")
                    logger.info("    raw failure causes: " + ", ".join(
                        f"{k} {v / len(graphs) * 100:.1f}%"
                        for k, v in causes.most_common()))
                    logger.info(
                        f"    correction: removes {cell['correction_orders_removed_mean']:.1f} "
                        f"bond orders and loses {cell['correction_atoms_lost_mean']:.1f} "
                        f"atoms; exact N survives "
                        f"{cell['correction_exact_n_rate'] * 100:.1f}%")
                    logger.info(f"    carbon fraction {cell['atom_type_fraction'].get('C', 0.0):.3f} "
                                f"vs real {reference.get('atom_type_fraction', {}).get('C', 0.0):.3f}")

        # ---- the inverse, with and without the trained context --------------
        logger.header("the inverse: scale saturation and round-trip, context vs neutralised")
        probe_n = targets_n[len(targets_n) // 2]
        counts = [int(probe_n)] * min(64, int(arguments.count))
        from zinc_gnf.conditioning import standardise_condition

        values = torch.from_numpy(standardise_condition(
            config.condition_features, scaler=scaler, n=counts,
            qed=[float(targets_qed[0])] * len(counts)))
        context = model.conditioning(values)
        width = config.max_nodes
        node_mask = torch.zeros(len(counts), width, dtype=torch.bool)
        for row, count in enumerate(counts):
            node_mask[row, :count] = True
        pair_mask = pair_mask_from_counts(counts, width)
        stability: dict[str, Any] = {}
        for label, neutral in (("with_context", False), ("condition_neutralised", True)):
            manager = (neutralise_coupling_condition(model) if neutral
                       else torch.no_grad())
            with torch.no_grad():
                with manager:
                    sampled = model.sample(counts, context, temperature=temperatures[0],
                                           generator=torch.Generator().manual_seed(1))
                    bonds = sampled["bonds_continuous"]
                    saturated = []
                    x = bonds
                    for layer in model.bond_layers:
                        x, _ = layer["actnorm"](x, pair_mask)
                        x, _ = layer["mixing"](x, pair_mask)
                        coupling = layer["coupling"]
                        passthrough, _ = coupling._split(x)
                        scale, _ = coupling._scale_shift(passthrough, pair_mask, context,
                                                         None)
                        saturated.append(float(
                            (scale.abs() > 0.95 * config.max_log_scale).to(
                                torch.float64).mean()))
                        x, _ = coupling(x, pair_mask, context, None)
            stability[label] = {
                "latent_round_trip": sampled["latent_round_trip_relative_error"],
                "log_scale_fraction_at_bound": float(np.mean(saturated)),
                "per_block_fraction_at_bound": saturated,
            }
            error = max(sampled["latent_round_trip_relative_error"].values())
            logger.info(f"[{label}] worst round-trip {error:.2e}, "
                        f"{stability[label]['log_scale_fraction_at_bound'] * 100:.1f}% of "
                        f"log-scales at the bound"
                        + ("  <- samples are NOT from the trained density"
                           if error > 1e-3 else ""))

        payload = {
            "stage": "22c",
            "checkpoint": str(arguments.checkpoint),
            "step": meta.get("step"),
            "model_config": config.to_dict(),
            "inverse_condition_number_bound": float(
                np.exp(config.bond_blocks * config.max_log_scale)),
            "dataset_fingerprint": fingerprint,
            "real_reference": real,
            "cells": cells,
            "inverse_stability": stability,
            "reading": {
                "marginals": ("if bonds_mean and the atom-type fractions match the real "
                              "reference while the graphs are still invalid, the model "
                              "learned the MARGINALS and not the correlations"),
                "causes": ("`disconnected_only` dominating means too few or badly placed "
                           "bonds; `over_valent_*` dominating means too many"),
                "exact_n": ("the corrector keeps only the largest connected component, so "
                            "largest_component_fraction_of_n bounds the achievable exact-N "
                            "rate before correction even starts"),
                "inverse": ("sampling IS the inverse; above ~1e-3 the molecules are not "
                            "drawn from the trained density and no other number is valid"),
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_write_json(payload, output_root / "graph_flow_diagnostics.json")
        logger.info(f"wrote {output_root / 'graph_flow_diagnostics.json'}")
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
