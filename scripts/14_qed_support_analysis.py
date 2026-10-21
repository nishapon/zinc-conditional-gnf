#!/usr/bin/env python3
"""Stage 14 — is weak QED control the data, the targets, the labels, or the model?

Part A of the stage-14 investigation, on the EXACT processed dataset and split the 50k
joint run used. Read-only: no training, no sampling, no checkpoint is modified.

What it produces under --output-dir:

  qed_distribution.png/.pdf    overall QED frequency, train vs validation, plus the
                               per-N QED frequency for every requested N
  n_qed_heatmap.png/.pdf       counts on the (N, QED) grid with the conditional
                               quantiles (median, 5th, 95th) drawn over it
  generated_overlay.png/.pdf   the generated QED distribution at each requested target
                               (from existing samples.jsonl files) over the real
                               distribution at the SAME N - distributions, not means
  qed_support.json             every number, machine-readable
  qed_support.txt              the per-N support table, the feasibility verdicts and
                               the label check, in readable form

Four candidate explanations are tested separately and reported separately:
  * data imbalance        per-N counts and high-QED mass
  * infeasible targets    each requested (N, QED) classified against the empirical
                          conditional distribution at that N
  * label / scaler error  QED recomputed by RDKit from the STORED GRAPH (not the SMILES)
                          and the standardised condition re-derived from scaler.json
  * the model             generated vs real distributions at matched N

With --flow-checkpoint it also runs a NUMERICAL STABILITY check on that checkpoint
(forward/inverse round trip on real embeddings and on prior samples, plus the
gauge-corrected log-determinant), because a log|det J|/node near 180 is only alarming if
the inverse actually degrades - that is measurable, not a matter of opinion.

Exit codes: 0 ok; 2 missing data; 4 bad arguments; 1 unexpected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from zinc_gnf.dataset import DatasetFormatError, load_training_splits  # noqa: E402
from zinc_gnf.qed_support import (  # noqa: E402
    HIGH_QED_THRESHOLDS,
    QED_BINS,
    by_n,
    compare_distributions,
    feasibility,
    format_support_report,
    n_qed_heatmap,
    qed_histogram,
    read_generated_qed,
    support_table,
    verify_labels,
)
from zinc_gnf.utils.logging import JobLogger  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402

DEFAULT_N = "15,23,30"
DEFAULT_QED = "0.4,0.6,0.8"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/processed/zinc250k_charge_aware_moflow_fixed")
    parser.add_argument("--output-dir", default="outputs/figures/qed_support")
    parser.add_argument("--target-n", default=DEFAULT_N)
    parser.add_argument("--target-qed", default=DEFAULT_QED)
    parser.add_argument("--generated-root", default=None,
                        help="directory holding the 50k joint run's cells, e.g. "
                             "outputs/samples/flow_nqed/seed_1 (its T*_N*_QED* "
                             "subdirectories are found automatically)")
    parser.add_argument("--decoder", default="degree")
    parser.add_argument("--label-sample", type=int, default=2000,
                        help="molecules rebuilt from their stored graph for the label check")
    parser.add_argument("--flow-checkpoint", default=None,
                        help="also run the numerical forward/inverse stability check")
    parser.add_argument("--ae-checkpoint", default=None)
    parser.add_argument("--stability-molecules", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


def stability_check(flow_path: Path, ae_path: Path | None, data_dir: str, *, molecules: int,
                    device: str, logger: JobLogger) -> dict[str, Any]:
    """Forward/inverse round trip and the gauge-corrected log-determinant.

    Two round trips, because they fail for different reasons:
      real      f(x) then f^-1: does the map invert on the data it was trained on?
      prior     f^-1(z) then f: does it invert where SAMPLING evaluates it? This is the
                one that matters for generation, and the one a large log|det J| threatens.
    The log-determinant is reported both raw and gauge-corrected by subtracting the
    whitening term sum(log std), which is a constant of the data, not a sign of blow-up.
    """
    import torch

    from zinc_gnf.conditioning import batch_condition, condition_embedding_for
    from zinc_gnf.dataset import MolecularGraphDataset, iter_eval_batches, seeded_subset_indices
    from zinc_gnf.trainer_autoencoder import load_autoencoder, resolve_device
    from zinc_gnf.trainer_flow import (
        VALIDATION_SUBSET_SEED,
        encode_batch,
        load_flow,
    )

    resolved = resolve_device(device)
    flow, probe = load_flow(flow_path, device=resolved)
    recorded = probe.get("autoencoder") or {}
    path = ae_path or Path(recorded.get("checkpoint", ""))
    if not path.exists():
        raise FileNotFoundError(f"autoencoder checkpoint not found: {path}")
    autoencoder, _ = load_autoencoder(path, device=resolved)
    autoencoder.eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)

    _, val_records, _ = load_training_splits(data_dir)
    val_set = MolecularGraphDataset(val_records)
    del val_records
    indices = seeded_subset_indices(len(val_set), molecules, VALIDATION_SUBSET_SEED)

    forward_errors: list[float] = []
    log_dets: list[float] = []
    nodes = 0
    with torch.no_grad():
        for batch in iter_eval_batches(val_set, 64, indices):
            batch = batch.to(resolved)
            embeddings = encode_batch(autoencoder, batch)
            condition = None
            if flow.config.conditional:
                values, present = batch_condition(batch, flow.config.condition_features)
                condition = condition_embedding_for(flow, values, present,
                                                    batch.batch_size, resolved)
            z, log_det = flow(embeddings, batch.node_mask, condition)
            back = flow.inverse(z, batch.node_mask, condition)
            mask = batch.node_mask.unsqueeze(-1)
            scale = (embeddings * mask).abs().max().clamp(min=1e-9)
            forward_errors.append(float(((back - embeddings) * mask).abs().max() / scale))
            log_dets.append(float(log_det.sum()))
            nodes += int(batch.node_mask.sum())

        # Sampling direction: start in the latent space, as generation does.
        sample_batch = next(iter(iter_eval_batches(val_set, 64, indices)))
        sample_batch = sample_batch.to(resolved)
        condition = None
        if flow.config.conditional:
            values, present = batch_condition(sample_batch, flow.config.condition_features)
            condition = condition_embedding_for(flow, values, present,
                                                sample_batch.batch_size, resolved)
        generator = torch.Generator(device=resolved).manual_seed(4242)
        embeddings, latents = flow.sample(sample_batch.node_mask, condition, temperature=1.0,
                                          generator=generator, return_latent=True)
        recovered, _ = flow(embeddings, sample_batch.node_mask, condition)
        latent_scale = latents.abs().max().clamp(min=1e-9)
        prior_error = float((recovered - latents).abs().max() / latent_scale)

    whitening = float(torch.log(flow.effective_std).sum())
    per_node = float(np.sum(log_dets) / max(nodes, 1))
    result = {
        "flow_checkpoint": str(flow_path), "step": probe.get("step"),
        "molecules": len(indices), "nodes": nodes,
        "real_round_trip_relative_error_max": float(np.max(forward_errors)),
        "real_round_trip_relative_error_mean": float(np.mean(forward_errors)),
        "prior_round_trip_relative_error_max": prior_error,
        "log_det_per_node": per_node,
        "whitening_log_det_per_node": -whitening,
        # The gauge-free part: what the couplings themselves contribute.
        "log_det_per_node_excluding_whitening": per_node + whitening,
        "verdict": ("stable: both round trips are at float32 noise level"
                    if max(float(np.max(forward_errors)), prior_error) < 1e-3
                    else "UNSTABLE: the inverse used for sampling does not reproduce its input"),
    }
    logger.info(f"stability: real round trip {result['real_round_trip_relative_error_max']:.2e}, "
                f"prior round trip {prior_error:.2e}, log|det J|/node {per_node:.2f} "
                f"({result['log_det_per_node_excluding_whitening']:.2f} excluding the "
                f"whitening gauge term) -> {result['verdict']}")
    return result


def distribution_figure(train_qed: list[float], val_qed: list[float],
                        grouped: dict[int, list[float]], targets_n: list[int], *,
                        output: Path) -> None:
    panels = 1 + len(targets_n)
    columns = min(panels, 2)
    rows = int(np.ceil(panels / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(6.5 * columns, 4.2 * rows),
                                squeeze=False)
    figure.suptitle("ZINC250K QED distribution: overall and conditioned on N", fontsize=13)
    axis = axes[0][0]
    axis.hist(train_qed, bins=QED_BINS, density=True, alpha=0.65, label=f"train (n={len(train_qed):,})",
              color="#4C72B0")
    axis.hist(val_qed, bins=QED_BINS, density=True, histtype="step", linewidth=1.8,
              label=f"validation (n={len(val_qed):,})", color="#C44E52")
    for threshold in HIGH_QED_THRESHOLDS:
        axis.axvline(threshold, linestyle=":", color="black", linewidth=1)
        axis.annotate(f"QED {threshold:g}", (threshold, axis.get_ylim()[1] * 0.92),
                      rotation=90, fontsize=7, ha="right", va="top")
    axis.set_xlabel("QED")
    axis.set_ylabel("density")
    axis.set_title("All molecules", fontsize=10)
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)

    for position, n in enumerate(targets_n, start=1):
        axis = axes[position // columns][position % columns]
        values = grouped.get(n, [])
        if values:
            axis.hist(values, bins=QED_BINS, density=True, alpha=0.75, color="#55A868",
                      label=f"N={n} (n={len(values):,})")
            for quantile, style in ((0.05, ":"), (0.5, "-"), (0.95, ":")):
                axis.axvline(float(np.quantile(values, quantile)), linestyle=style,
                             color="black", linewidth=1.2)
            axis.set_title(f"N = {n}: median {np.median(values):.3f}, "
                           f">=0.7 {(np.asarray(values) >= 0.7).mean() * 100:.1f}%",
                           fontsize=10)
        else:
            # A requested N with no training molecules at all is itself a finding; say so
            # on the panel instead of drawing an empty legend.
            axis.set_title(f"N = {n}: NO training molecules", fontsize=10)
            axis.text(0.5, 0.5, "no training molecules at this N", ha="center",
                      va="center", transform=axis.transAxes, fontsize=10, style="italic")
        axis.set_xlabel("QED")
        axis.set_ylabel("density")
        if axis.get_legend_handles_labels()[1]:
            axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
    for extra in range(panels, rows * columns):
        axes[extra // columns][extra % columns].axis("off")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output.with_suffix(".png"), dpi=200)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def heatmap_figure(heatmap: dict[str, Any], targets: list[tuple[int, float]], *,
                   output: Path) -> None:
    counts = np.asarray(heatmap["counts"], dtype=np.float64)
    density = np.asarray(heatmap["conditional_density"], dtype=np.float64)
    n_values = heatmap["n_values"]
    edges = heatmap["qed_bin_edges"]
    figure, axes = plt.subplots(1, 2, figsize=(15.0, 6.2))
    for axis, matrix, title, label in (
            (axes[0], counts, "Counts on the (N, QED) grid", "molecules"),
            (axes[1], density, "Density WITHIN each N (what a conditional model must match)",
             "fraction of molecules at that N")):
        image = axis.imshow(matrix, aspect="auto", origin="lower", cmap="viridis",
                            extent=(edges[0], edges[-1], n_values[0] - 0.5, n_values[-1] + 0.5))
        figure.colorbar(image, ax=axis, label=label)
        for quantile, style, colour in ((0.05, ":", "white"), (0.5, "-", "white"),
                                        (0.95, ":", "white")):
            axis.plot(heatmap["conditional_quantiles"][str(quantile)], n_values, style,
                      color=colour, linewidth=1.5,
                      label=f"q{int(quantile * 100)} of QED | N" if quantile == 0.5 else None)
        for n, qed in targets:
            axis.plot([qed], [n], marker="*", markersize=13, color="#FF3B30",
                      markeredgecolor="black", markeredgewidth=0.6)
        axis.set_xlabel("QED")
        axis.set_ylabel("heavy atoms N")
        axis.set_title(title, fontsize=10)
        axis.legend(fontsize=8, loc="lower left")
    figure.suptitle("ZINC250K training split: QED given N, with the requested targets (stars) "
                    "and the conditional 5th / 50th / 95th percentiles", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output.with_suffix(".png"), dpi=200)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def overlay_figure(comparisons: list[dict[str, Any]], *, output: Path) -> None:
    if not comparisons:
        return
    columns = min(3, len(comparisons))
    rows = int(np.ceil(len(comparisons) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5.4 * columns, 4.0 * rows),
                                squeeze=False)
    figure.suptitle("Generated QED at each requested target vs real molecules of the same N",
                    fontsize=12)
    for position, entry in enumerate(comparisons):
        axis = axes[position // columns][position % columns]
        statistics = entry["comparison"]
        comparable = "generated" in statistics            # both populations non-empty
        if entry["generated_values"]:
            axis.hist(entry["generated_values"], bins=QED_BINS, density=True, alpha=0.7,
                      color="#DD8452", label="generated")
        if entry["real_values"]:
            axis.hist(entry["real_values"], bins=QED_BINS, density=True, alpha=0.55,
                      color="#8C8C8C", label=f"real N={entry['n']}")
        axis.axvline(entry["requested_qed"], color="#FF3B30", linewidth=2,
                     label=f"requested {entry['requested_qed']:g}")
        if comparable:
            axis.set_title(f"N={entry['n']}, QED {entry['requested_qed']:g}: generated mean "
                           f"{statistics['generated']['mean']:.3f}, overlap "
                           f"{statistics['histogram_overlap']:.2f}", fontsize=9)
        else:
            missing = ("no valid generated samples" if not entry["generated_values"]
                       else f"no real molecules at N={entry['n']}")
            axis.set_title(f"N={entry['n']}, QED {entry['requested_qed']:g}: {missing}",
                           fontsize=9)
        axis.set_xlabel("QED")
        axis.set_ylabel("density")
        if axis.get_legend_handles_labels()[1]:
            axis.legend(fontsize=7)
        axis.grid(alpha=0.3)
    for extra in range(len(comparisons), rows * columns):
        axes[extra // columns][extra % columns].axis("off")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output.with_suffix(".png"), dpi=200)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def find_generated_cells(root: Path) -> list[tuple[int, float, float, Path]]:
    """Discover `T<t>_N<n>_QED<q>/samples.jsonl` (stage 13) or `N<n>/` (stage 12)."""
    found: list[tuple[int, float, float, Path]] = []
    if not root.exists():
        return found
    for directory in sorted(root.iterdir()):
        samples = directory / "samples.jsonl"
        if not samples.exists():
            continue
        name = directory.name
        temperature, n, qed = float("nan"), None, None
        for part in name.split("_"):
            if part.startswith("T"):
                try:
                    temperature = float(part[1:])
                except ValueError:
                    pass
            elif part.startswith("QED"):
                try:
                    qed = float(part[3:])
                except ValueError:
                    pass
            elif part.startswith("N"):
                try:
                    n = int(part[1:])
                except ValueError:
                    pass
        if n is None:
            continue
        metrics = directory / "metrics.json"
        if qed is None and metrics.exists():
            payload = json.loads(metrics.read_text())
            qed = (payload.get("requested") or {}).get("qed")
        found.append((n, float(qed) if qed is not None else float("nan"), temperature, samples))
    return found


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    output_root = Path(arguments.output_dir)
    try:
        targets_n = [int(t) for t in str(arguments.target_n).split(",") if t.strip()]
        targets_qed = [float(t) for t in str(arguments.target_qed).split(",") if t.strip()]
        if not targets_n or not targets_qed:
            raise ValueError("--target-n and --target-qed must be non-empty")
        output_root.mkdir(parents=True, exist_ok=True)
        logger = JobLogger("qed_support", output_root, filename="qed_support.log")
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4

    started = time.perf_counter()
    try:
        train_records, val_records, fingerprint = load_training_splits(arguments.data_dir)
        logger.header("QED support analysis (Part A)")
        logger.info(f"data {arguments.data_dir}: train={len(train_records):,} "
                    f"val={len(val_records):,} (test NOT loaded) "
                    f"protocol={fingerprint.get('split_protocol')}")

        train_qed = [float(r["qed"]) for r in train_records]
        val_qed = [float(r["qed"]) for r in val_records]
        overall = {"train": qed_histogram(train_qed), "validation": qed_histogram(val_qed)}
        grouped_train = by_n(train_records)
        grouped_val = by_n(val_records)
        rows = support_table(grouped_train)
        heatmap = n_qed_heatmap(grouped_train)
        requests = [(n, q) for n in targets_n for q in targets_qed]
        feasible = feasibility(grouped_train, requests=requests)

        scaler = None
        scaler_path = Path(arguments.data_dir) / "scaler.json"
        if scaler_path.exists():
            from zinc_gnf.conditioning import load_condition_scaler
            scaler = load_condition_scaler(arguments.data_dir)
            logger.info(f"scaler.json: mean={scaler.mean} std={scaler.std} (train-only)")
        else:
            logger.warn(f"{scaler_path} not found; the condition check is skipped")
        labels = verify_labels(train_records, scaler=scaler, sample=arguments.label_sample)
        logger.info(f"label check on {labels['checked']:,} molecules rebuilt from their "
                    f"stored graphs: max |RDKit - stored| = "
                    f"{labels['qed_label_max_abs_difference']:.3e}, "
                    f"{labels['unbuildable_graphs']} unbuildable -> {labels['verdict']}")

        for row in rows:
            if row["n"] in targets_n:
                logger.info(f"N={row['n']}: {row['molecules']:,} molecules, QED mean "
                            f"{row['qed_mean']:.4f}, q05-q95 {row['q5']:.3f}-{row['q95']:.3f}, "
                            f">=0.7 {row['fraction_ge_0p7'] * 100:.1f}% "
                            f"({row['count_ge_0p7']:,}), >=0.8 "
                            f"{row['fraction_ge_0p8'] * 100:.1f}% ({row['count_ge_0p8']:,})")
        for row in feasible:
            logger.info(f"request N={row['n']} QED={row['requested_qed']:.2f}: "
                        f"{row['verdict']} ({row.get('molecules_in_window', 0):,} molecules "
                        f"within +-0.05, z={row.get('target_z_at_this_n', float('nan')):+.2f})")

        # ---- generated overlay ------------------------------------------------
        comparisons: list[dict[str, Any]] = []
        if arguments.generated_root:
            cells = find_generated_cells(Path(arguments.generated_root))
            if not cells:
                logger.warn(f"no samples.jsonl found under {arguments.generated_root}; "
                            f"the overlay is skipped")
            for n, qed, temperature, samples in cells:
                generated = read_generated_qed(samples, decoder=arguments.decoder)
                real = grouped_val.get(n, []) or grouped_train.get(n, [])
                comparison = compare_distributions(generated, real, requested_qed=qed)
                comparisons.append({"n": n, "requested_qed": qed, "temperature": temperature,
                                    "samples_path": str(samples),
                                    "generated_values": generated, "real_values": real,
                                    "comparison": comparison})
                logger.info(
                    f"generated N={n} QED={qed:g} (T={temperature:g}): "
                    f"{comparison['generated_molecules']:,} valid, mean "
                    f"{comparison.get('generated', {}).get('mean', float('nan')):.4f} vs real "
                    f"{comparison.get('real', {}).get('mean', float('nan')):.4f}, overlap "
                    f"{comparison.get('histogram_overlap', float('nan')):.3f}, "
                    f"std ratio {comparison.get('generated_std_over_real_std', float('nan')):.3f}")

        # ---- figures ----------------------------------------------------------
        distribution_figure(train_qed, val_qed, grouped_train, targets_n,
                            output=output_root / "qed_distribution")
        heatmap_figure(heatmap, requests, output=output_root / "n_qed_heatmap")
        overlay_figure(comparisons, output=output_root / "generated_overlay")

        stability = None
        if arguments.flow_checkpoint:
            stability = stability_check(
                Path(arguments.flow_checkpoint),
                Path(arguments.ae_checkpoint) if arguments.ae_checkpoint else None,
                arguments.data_dir, molecules=arguments.stability_molecules,
                device=arguments.device, logger=logger)

        report = format_support_report(overall, rows, feasible, labels)
        extra_lines = []
        if comparisons:
            extra_lines += ["", "Generated vs real QED at matched N (distributions)", "",
                            "|   N | requested | T | generated mean | real mean | generated std |"
                            " real std | histogram overlap | Wasserstein |",
                            "|----:|----------:|--:|---------------:|----------:|--------------:|"
                            "---------:|------------------:|------------:|"]
            for entry in comparisons:
                statistics = entry["comparison"]
                if "generated" not in statistics:
                    reason = ("no valid generated samples" if not entry["generated_values"]
                              else f"no real molecules at N={entry['n']}")
                    extra_lines.append(f"| {entry['n']} | {entry['requested_qed']:.2f} | "
                                       f"{entry['temperature']:g} | {reason} | | | | | |")
                    continue
                extra_lines.append(
                    f"| {entry['n']} | {entry['requested_qed']:.2f} | {entry['temperature']:g} | "
                    f"{statistics['generated']['mean']:.4f} | {statistics['real']['mean']:.4f} | "
                    f"{statistics['generated']['std']:.4f} | {statistics['real']['std']:.4f} | "
                    f"{statistics['histogram_overlap']:.3f} | "
                    f"{statistics['wasserstein_distance']:.4f} |")
        if stability:
            extra_lines += ["", f"Numerical stability: {stability['verdict']} "
                            f"(real round trip {stability['real_round_trip_relative_error_max']:.2e}, "
                            f"sampling round trip "
                            f"{stability['prior_round_trip_relative_error_max']:.2e}, "
                            f"log|det J|/node {stability['log_det_per_node']:.2f}, "
                            f"{stability['log_det_per_node_excluding_whitening']:.2f} without "
                            f"the whitening gauge term)"]
        (output_root / "qed_support.txt").write_text(
            report + "\n" + "\n".join(extra_lines) + "\n", encoding="utf-8")

        atomic_write_json({
            "data_fingerprint": fingerprint,
            "overall": overall,
            "support_by_n": rows,
            "heatmap": heatmap,
            "requested": {"n": targets_n, "qed": targets_qed},
            "feasibility": feasible,
            "labels": labels,
            "scaler": scaler.to_dict() if scaler is not None else None,
            "generated": [{k: v for k, v in entry.items()
                           if k not in ("generated_values", "real_values")}
                          for entry in comparisons],
            "stability": stability,
            "seconds": time.perf_counter() - started,
        }, output_root / "qed_support.json")
        print(report, flush=True)
        print("\n".join(extra_lines), flush=True)
        logger.info(f"wrote {output_root}/qed_distribution.png, n_qed_heatmap.png, "
                    f"generated_overlay.png, qed_support.{{json,txt}}")
        return 0

    except FileNotFoundError as error:
        print(f"FATAL (missing input): {error}", file=sys.stderr, flush=True)
        return 2
    except DatasetFormatError as error:
        print(f"FATAL (dataset): {error}", file=sys.stderr, flush=True)
        return 2
    except ValueError as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
