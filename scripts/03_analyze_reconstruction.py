#!/usr/bin/env python3
"""Stage 3 — analyse WHY a trained autoencoder fails to reconstruct molecules.

Read-only: loads a checkpoint and the VALIDATION split (the test split needs
`--allow-test`), writes a JSON report and a short text summary, and changes
nothing else. It reports

  * failure strata by size, ring count, fusion, aromaticity, charge, symmetry;
  * a failure taxonomy (symmetry-unreachable vs reachable-but-wrong);
  * bond metrics against the trivial "no bond" baseline;
  * the exact-reconstruction ceiling with its 95% interval and the checks of
    its assumptions (1-WL vs RDKit symmetry classes; identity-level rescue).

CPU is enough (a GPU only speeds up the forward pass). Exit codes: 0 ok;
2 file missing; 3 checkpoint/vocabulary mismatch; 4 bad arguments; 1 other.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from zinc_gnf.dataset import (  # noqa: E402
    DatasetFormatError,
    MolecularGraphDataset,
    load_test_split,
    load_training_splits,
    seeded_subset_indices,
)
from zinc_gnf.reconstruction_analysis import (  # noqa: E402
    analyze_reconstructions,
    bond_metric_diagnostics,
    ceiling_assumption_report,
    summarize_failures,
)
from zinc_gnf.trainer_autoencoder import ResumeMismatchError, load_autoencoder, resolve_device  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402


def format_table(title: str, table: list[dict]) -> list[str]:
    lines = [f"  {title}:"]
    for row in table:
        lo, hi = row["constrained_exact_ci95"]
        lines.append(f"    {row['group']:<24} n={row['molecules']:>5}  exact={row['constrained_exact']:.3f} "
                     f"[{lo:.3f},{hi:.3f}]  valid={row['constrained_valid']:.3f}  "
                     f"decodable={row['decodable_fraction']:.2f}  bond_err/mol={row['mean_bond_errors']:.2f}")
    return lines


def text_summary(summary: dict, bonds: dict, ceiling: dict, header: str) -> str:
    c = bonds["constrained"]
    lines = [header,
             f"  molecules {summary['molecules']}  constrained exact {summary['constrained_exact']:.3f} "
             f"[{summary['constrained_exact_ci95'][0]:.3f},{summary['constrained_exact_ci95'][1]:.3f}]  "
             f"raw exact {summary['raw_exact']:.3f}  validity constrained {summary['constrained_validity']:.3f} "
             f"raw {summary['raw_validity']:.3f}",
             f"  ceiling {summary['ceiling_decodable_fraction']:.3f} "
             f"[{summary['ceiling_ci95'][0]:.3f},{summary['ceiling_ci95'][1]:.3f}]  "
             f"exact among decodable {summary['exact_among_decodable']:.3f}  "
             f"exact among symmetry-unreachable {summary['exact_among_symmetry_unreachable']:.3f}",
             "  failure taxonomy (fraction of all molecules):"]
    for name, value in summary["failure_taxonomy"].items():
        lines.append(f"    {name:<34} {value:.3f}")
    for title, key in (("by size (heavy atoms)", "by_size"), ("by ring count", "by_rings"),
                       ("by ring fusion", "by_fused"), ("by aromaticity", "by_aromaticity"),
                       ("by charge", "by_charge"), ("by symmetry", "by_symmetry")):
        lines += format_table(title, summary[key])
    lines += ["  bond metrics (constrained decoding):",
              f"    pair accuracy {c['pair_accuracy']:.4f} vs all-no-bond baseline "
              f"{c['majority_class_baseline_accuracy']:.4f}  (gain {c['accuracy_gain_over_baseline']:.4f}); "
              f"balanced accuracy {c['balanced_accuracy']:.4f}; real-bond error rate {c['real_bond_error_rate']:.4f}"]
    for name, v in c["per_class"].items():
        lines.append(f"    {name:<8} support={v['support']:>7} precision={v['precision']:.4f} "
                     f"recall={v['recall']:.4f} f1={v['f1']:.4f}")
    lines.append("    " + bonds["per_molecule"]["note"])
    lines += ["  ceiling assumptions:",
              f"    ceiling {ceiling['ceiling']:.3f} [{ceiling['ceiling_ci95'][0]:.3f},{ceiling['ceiling_ci95'][1]:.3f}] "
              f"on {ceiling['molecules']} molecules",
              f"    1-WL partition == RDKit symmetry classes for {ceiling['wl_partition_equals_rdkit_symmetry_classes']:.3f} "
              f"of molecules (mean extra 1-WL classes {ceiling['wl_extra_classes_vs_rdkit_mean']:.3f})",
              f"    identity-level rescue: {ceiling['identity_level_rescue']} of "
              f"{ceiling['identity_level_rescue_tested']} tested non-decodable molecules"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/processed/zinc250k_charge_aware_moflow_fixed")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--max-molecules", type=int, default=3000)
    parser.add_argument("--ceiling-molecules", type=int, default=None,
                        help="molecules for the ceiling/assumption checks (default: the whole split)")
    parser.add_argument("--rescue-sample", type=int, default=300)
    parser.add_argument("--subset-seed", type=int, default=12345)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-per-molecule-csv", action="store_true")
    arguments = parser.parse_args(argv)

    if arguments.split == "test" and not arguments.allow_test:
        print("FATAL: the test split is held out for final evaluation; pass --allow-test",
              file=sys.stderr, flush=True)
        return 4
    try:
        device = resolve_device(arguments.device)
        model, checkpoint = load_autoencoder(arguments.checkpoint, device=device)
        records = (load_training_splits(arguments.data_dir)[1] if arguments.split == "val"
                   else load_test_split(arguments.data_dir)[0])
        dataset = MolecularGraphDataset(records)
        indices = seeded_subset_indices(len(dataset), arguments.max_molecules, arguments.subset_seed)
        print(f"analysing {len(indices):,} of {len(dataset):,} {arguments.split} molecules "
              f"(checkpoint step {checkpoint.get('step')}, device {device})", flush=True)
        analysis = analyze_reconstructions(model, dataset, indices=indices, device=device,
                                           batch_size=arguments.batch_size)
        summary = summarize_failures(analysis)
        bonds = bond_metric_diagnostics(analysis)
        ceiling_indices = (indices if arguments.ceiling_molecules is None and False else
                           seeded_subset_indices(len(dataset), arguments.ceiling_molecules, arguments.subset_seed))
        print(f"checking ceiling assumptions on {len(ceiling_indices):,} molecules ...", flush=True)
        ceiling = ceiling_assumption_report(dataset, indices=ceiling_indices, rescue_sample=arguments.rescue_sample)
    except FileNotFoundError as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 2
    except (ResumeMismatchError, DatasetFormatError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 3
    except (ValueError, RuntimeError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1

    out = Path(arguments.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {"checkpoint": str(arguments.checkpoint), "checkpoint_step": checkpoint.get("step"),
              "split": arguments.split, "molecules_analysed": len(indices),
              "model_config": checkpoint.get("model_config"), "summary": summary,
              "bond_metrics": bonds, "ceiling": ceiling}
    atomic_write_json(report, out / "reconstruction_analysis.json")
    header = (f"Reconstruction analysis: {arguments.checkpoint} (step {checkpoint.get('step')}), "
              f"{arguments.split} split")
    text = text_summary(summary, bonds, ceiling, header)
    (out / "reconstruction_analysis.txt").write_text(text + "\n", encoding="utf-8")
    if not arguments.no_per_molecule_csv:
        import csv
        columns = ["row_id", "smiles", "n_node", "n_rings", "fused_rings", "aromatic_fraction", "n_charged_atoms",
                   "net_charge", "n_classes", "largest_class", "symmetric_atoms", "decodable", "atoms_exact",
                   "raw_exact", "constrained_exact", "constrained_valid", "constrained_bond_errors", "failure"]
        with (out / "per_molecule.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(analysis["rows"])
    print(text, flush=True)
    print(f"report: {out / 'reconstruction_analysis.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
