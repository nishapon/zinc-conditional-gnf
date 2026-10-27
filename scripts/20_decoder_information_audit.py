#!/usr/bin/env python3
"""Stage 19 — the decoder-information audit: which of three hypotheses is true?

Stage 18 showed the conditional signal is NOT small (median delivered/required
displacement 0.665, cosine +0.926, 1.217 at N=23/T=0.2, all samples finite), while
realised QED stays near 0.56 for every request. A matching MEAN displacement does not
prove the decoder is at fault, so this job measures each stage separately and reports
which hypothesis the data support:

  H1  the information is in the generated embeddings and DECODING loses it
  H2  the embeddings move, but not along a QED-PREDICTIVE direction (the mean difference
      between two QED populations contains every covariate correlated with QED, so a high
      cosine against it is not the same as carrying QED information)
  H3  the information reaches the decoder, and graph constraints plus validity selection
      erase it afterwards

Experiments, all on existing checkpoints, nothing retrained except a small diagnostic
read-out:

  A  decoder QED-transfer CONTROL: encode REAL training molecules at QED 0.4 and at 0.8
     (same N) and decode them through the identical path generation uses. If the decoder
     loses that difference on perfect inputs, QED control is impossible through it.
  B  paired QED intervention: identical latent noise, only the QED request changes;
     displacement is measured at the latent, embedding, bond-logit, bond-argmax, graph and
     final-QED stages, with paired statistics and standard errors, per decoder
     (raw / constrained / degree) so graph constraints are visible separately.
  C  a QED read-out trained on REAL TRAIN embeddings and validated on REAL VALIDATION
     embeddings, applied diagnostically to generated embeddings. Its held-out error is
     reported, and Mahalanobis statistics say how far the generated embeddings sit outside
     the distribution it was fitted on - a prediction there is an extrapolation, never
     ground truth. It is never used to guide generation or to report headline QED.
  E  dataset support with every count and denominator stated explicitly.

Read-only. Exit codes: 0 ok; 2 missing input; 4 bad arguments; 1 unexpected.
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

from zinc_gnf.conditioning import load_condition_scaler  # noqa: E402
from zinc_gnf.dataset import (  # noqa: E402
    DatasetFormatError,
    MolecularGraphDataset,
    iter_eval_batches,
    load_training_splits,
    seeded_subset_indices,
)
from zinc_gnf.decoder_information import (  # noqa: E402
    DECODERS,
    decoder_qed_transfer,
    fit_embedding_probe,
    ood_statistics,
    paired_intervention,
    support_counts,
)
from zinc_gnf.trainer_autoencoder import load_autoencoder, resolve_device  # noqa: E402
from zinc_gnf.trainer_flow import (  # noqa: E402
    STATISTICS_SEED,
    VALIDATION_SUBSET_SEED,
    file_sha256,
    load_flow,
)
from zinc_gnf.utils.logging import JobLogger  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402
from zinc_gnf.utils.seed import set_seed  # noqa: E402

# A probe can only certify a shift it can resolve; below this the probe says nothing.
PROBE_RESOLUTION_FACTOR = 1.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flow-checkpoint", required=True, help="READ-ONLY")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="data/processed/zinc250k_charge_aware_moflow_fixed")
    parser.add_argument("--ae-checkpoint", default=None)
    parser.add_argument("--target-n", default="15,23,30")
    parser.add_argument("--qed-pair", default="0.4,0.8")
    parser.add_argument("--count", type=int, default=256, help="samples per condition")
    parser.add_argument("--temperature", default="0.2,1.0")
    parser.add_argument("--probe-train-molecules", type=int, default=20000)
    parser.add_argument("--probe-val-molecules", type=int, default=4000)
    parser.add_argument("--probe-epochs", type=int, default=10)
    parser.add_argument("--transfer-molecules", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    arguments = parse_args(argv)
    output_root = Path(arguments.output_dir)
    try:
        targets_n = [int(t) for t in str(arguments.target_n).split(",") if t.strip()]
        pair = [float(t) for t in str(arguments.qed_pair).split(",") if t.strip()]
        temperatures = [float(t) for t in str(arguments.temperature).split(",") if t.strip()]
        if len(pair) != 2 or pair[0] >= pair[1]:
            raise ValueError("--qed-pair must be 'low,high' with low < high")
        if not targets_n or not temperatures:
            raise ValueError("--target-n and --temperature must be non-empty")
        output_root.mkdir(parents=True, exist_ok=True)
        logger = JobLogger("decoder_information", output_root, filename="audit.log")
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4

    started = time.perf_counter()
    try:
        set_seed(arguments.seed, deterministic=False)
        device = resolve_device(arguments.device)
        flow, probe_meta = load_flow(Path(arguments.flow_checkpoint), device=device)
        if "qed" not in flow.config.condition_features:
            raise ValueError(f"this flow does not condition on QED "
                             f"(features {list(flow.config.condition_features)})")
        recorded = probe_meta.get("autoencoder") or {}
        ae_path = Path(arguments.ae_checkpoint or recorded.get("checkpoint", ""))
        if not ae_path.exists():
            raise FileNotFoundError(f"autoencoder checkpoint not found: {ae_path}")
        autoencoder, _ = load_autoencoder(ae_path, device=device)
        autoencoder.eval()
        for parameter in autoencoder.parameters():
            parameter.requires_grad_(False)
        decoders = tuple(DECODERS) if autoencoder.config.degree_head else ("raw", "constrained")
        scaler = load_condition_scaler(arguments.data_dir)
        train_records, val_records, fingerprint = load_training_splits(arguments.data_dir)
        train_set = MolecularGraphDataset(train_records)
        val_set = MolecularGraphDataset(val_records)
        del train_records, val_records

        logger.header("Stage 19: decoder-information audit")
        logger.info(f"flow={arguments.flow_checkpoint} step={probe_meta.get('step')} "
                    f"sha256={file_sha256(arguments.flow_checkpoint)[:16]} (READ-ONLY)")
        logger.info(f"N={targets_n} QED {pair[0]} -> {pair[1]} at T={temperatures}, "
                    f"{arguments.count} samples per condition, decoders={list(decoders)}")

        # ---- E. support, with the denominator stated -------------------------
        support = support_counts(train_set, n_values=targets_n, qed_values=pair)
        for row in support:
            parts = ", ".join(
                f"QED {key}: {value['molecules_within_window']:,} "
                f"({value['fraction_of_this_n'] * 100:.1f}% of this N)"
                for key, value in row["by_qed"].items())
            logger.info(f"support N={row['n']}: {row['molecules_at_this_n']:,} training "
                        f"molecules at this N; {parts}; minimum over the two QED values = "
                        f"{row['minimum_over_qed_values']:,}")
        logger.info("the stage-18 report's 'support' column was the MINIMUM of the two QED "
                    "groups at that N (not a matched subset); both counts are given above")

        # ---- C. the probe, on real embeddings only ---------------------------
        probe_train = seeded_subset_indices(len(train_set), arguments.probe_train_molecules,
                                            STATISTICS_SEED)
        probe_val = seeded_subset_indices(len(val_set), arguments.probe_val_molecules,
                                          VALIDATION_SUBSET_SEED)
        probe, probe_report = fit_embedding_probe(
            autoencoder, train_set, val_set, train_indices=probe_train,
            val_indices=probe_val, device=device, epochs=arguments.probe_epochs)
        validation = probe_report["validation"]
        logger.info(f"QED read-out: trained on {probe_report['train']['molecules']:,} REAL "
                    f"training molecules; held-out MAE {validation['mae']:.4f}, correlation "
                    f"{validation['correlation']:.4f} on {validation['molecules']:,} "
                    f"validation molecules")
        logger.info(f"it can therefore only certify a QED shift larger than about "
                    f"{validation['mae'] * PROBE_RESOLUTION_FACTOR:.4f}; a prediction on "
                    f"generated embeddings is an EXTRAPOLATION and is never used as ground "
                    f"truth or to guide generation")

        # real embeddings for the out-of-distribution reference
        reference_nodes = []
        with torch.no_grad():
            for batch in iter_eval_batches(val_set, 128, probe_val[:1024]):
                batch = batch.to(device)
                nodes = autoencoder.encode(batch)
                reference_nodes.append(nodes[batch.node_mask].detach())
        reference_nodes = torch.cat(reference_nodes)

        # ---- A. the decoder QED-transfer control -----------------------------
        transfer = []
        for n in targets_n:
            row = decoder_qed_transfer(autoencoder, train_set, n=n, qed_low=pair[0],
                                       qed_high=pair[1],
                                       max_molecules=arguments.transfer_molecules,
                                       device=device, decoders=decoders)
            transfer.append(row)
            if "preservation" in row:
                preserved = ", ".join(f"{name} {value:+.3f}"
                                      for name, value in row["preservation"].items())
                logger.info(f"decoder transfer N={n}: true QED difference "
                            f"{row['true_qed_difference']:+.4f}; decoded difference "
                            + ", ".join(f"{name} {value:+.4f}" for name, value
                                        in row["decoded_qed_difference"].items())
                            + f"; preserved fraction {preserved}")
            else:
                logger.warn(f"decoder transfer N={n}: {row['verdict']}")

        # ---- B. the paired intervention, stage by stage ----------------------
        interventions = []
        for temperature in temperatures:
            for n in targets_n:
                row = paired_intervention(flow, autoencoder, scaler=scaler, n=n,
                                          qed_low=pair[0], qed_high=pair[1],
                                          count=arguments.count, temperature=temperature,
                                          device=device, probe=probe, seed=arguments.seed,
                                          decoders=decoders)
                if row.get("usable_molecules"):
                    with torch.no_grad():
                        from zinc_gnf.conditioning import (
                            condition_embedding_for,
                            standardise_condition,
                        )
                        from zinc_gnf.flow import node_mask_from_counts

                        counts = [int(n)] * int(arguments.count)
                        mask = node_mask_from_counts(counts, device=device)
                        values = torch.from_numpy(standardise_condition(
                            flow.config.condition_features, scaler=scaler, n=counts,
                            qed=[pair[1]] * len(counts))).to(device)
                        condition = condition_embedding_for(flow, values, None, len(counts),
                                                            device)
                        generator = torch.Generator(device=device).manual_seed(arguments.seed)
                        z = flow.prior.sample(mask, condition, temperature=temperature,
                                              generator=generator)
                        generated = flow.inverse(z, mask, condition)
                        finite = torch.isfinite(generated).all(-1) & mask
                        row["out_of_distribution"] = ood_statistics(
                            reference_nodes, generated[finite].detach())
                interventions.append(row)
                probe_block = row.get("probe", {})
                logger.info(
                    f"T={temperature:g} N={n}: embedding move "
                    f"{row.get('embedding', {}).get('per_node_norm', float('nan')):.4f}/node, "
                    f"bond-logit move "
                    f"{row.get('bond_logits', {}).get('per_pair_l2', float('nan')):.4f}/pair, "
                    f"bond argmax changed "
                    f"{row.get('bond_logits', {}).get('argmax_change_rate', float('nan')) * 100:.2f}% "
                    f"of pairs, probe QED change "
                    f"{probe_block.get('predicted_change_mean', float('nan')):+.4f} "
                    f"(requested {probe_block.get('requested_change', float('nan')):+.2f})")
                for name in decoders:
                    block = row.get("decoders", {}).get(name, {})
                    paired = block.get("paired", {})
                    logger.info(
                        f"    [{name}] validity {block.get('low', {}).get('validity', float('nan')):.3f}"
                        f" -> {block.get('high', {}).get('validity', float('nan')):.3f}, "
                        f"aromatic rings change {block.get('aromatic_rings_change', float('nan')):+.3f}, "
                        f"paired QED change {paired.get('qed_change_mean', float('nan')):+.4f} "
                        f"+- {paired.get('qed_change_sem', float('nan')):.4f} on "
                        f"{paired.get('paired_molecules', 0)} paired molecules")
                if "out_of_distribution" in row:
                    ood = row["out_of_distribution"]
                    logger.info(f"    generated embeddings sit at "
                                f"{ood['generated_over_real_mean']:.2f}x the real mean "
                                f"Mahalanobis distance")

        # ---- the verdict ------------------------------------------------------
        usable = [r for r in interventions if r.get("usable_molecules")]
        headline = arguments.__dict__.get("headline", "degree")
        decoder_name = "degree" if "degree" in decoders else "constrained"

        def median(values) -> float:
            values = [v for v in values if v is not None and np.isfinite(v)]
            return float(np.median(values)) if values else float("nan")

        probe_change = median([r.get("probe", {}).get("predicted_change_mean") for r in usable])
        requested = pair[1] - pair[0]
        decoded_change = median([r["decoders"][decoder_name]["paired"]["qed_change_mean"]
                                 for r in usable if decoder_name in r.get("decoders", {})])
        raw_change = median([r["decoders"]["raw"]["paired"]["qed_change_mean"]
                             for r in usable if "raw" in r.get("decoders", {})])
        argmax_rate = median([r.get("bond_logits", {}).get("argmax_change_rate")
                              for r in usable])
        preservation = median([max((v for v in row.get("preservation", {}).values()
                                    if np.isfinite(v)), default=np.nan)
                               for row in transfer])
        probe_resolves = bool(np.isfinite(probe_change)
                              and abs(probe_change) > validation["mae"] * PROBE_RESOLUTION_FACTOR)

        if np.isfinite(preservation) and preservation < 0.5:
            hypothesis = "H1-decoder"
            reading = ("the decoder loses most of a QED difference even on REAL embeddings "
                       f"(preserved fraction {preservation:.2f}), so QED control cannot be "
                       "achieved through this decoder however good the flow is: the decoder "
                       "or the representation is the binding constraint")
        elif not probe_resolves:
            hypothesis = "H2-direction"
            reading = ("a QED read-out trained on real embeddings sees a change of only "
                       f"{probe_change:+.4f} in the generated embeddings (its own held-out "
                       f"MAE is {validation['mae']:.4f}), so the large displacement is not "
                       "along a QED-predictive direction. The flow moves the samples "
                       "somewhere, but not where QED lives.")
        elif np.isfinite(raw_change) and abs(raw_change) > abs(decoded_change) + 0.01:
            hypothesis = "H3-constraints"
            reading = ("the QED change survives into the raw decoded graphs "
                       f"({raw_change:+.4f}) but is smaller after constrained/degree "
                       f"assembly ({decoded_change:+.4f}), so graph constraints and "
                       "validity selection are erasing it")
        else:
            hypothesis = "H1-decoding"
            reading = ("the read-out sees the QED change in the embeddings and the decoder "
                       "preserves QED differences on real inputs, yet the decoded QED does "
                       f"not move ({decoded_change:+.4f} against a requested {requested:+.2f}): "
                       "the loss happens in decoding GENERATED embeddings specifically - "
                       "check the out-of-distribution column")
        verdict = {
            "hypothesis": hypothesis, "reading": reading,
            "requested_qed_change": requested,
            "median_probe_change": probe_change,
            "probe_held_out_mae": validation["mae"],
            "probe_can_resolve_the_change": probe_resolves,
            "median_raw_paired_qed_change": raw_change,
            f"median_{decoder_name}_paired_qed_change": decoded_change,
            "median_bond_argmax_change_rate": argmax_rate,
            "median_decoder_qed_preservation_on_real": preservation,
        }
        logger.info(f"VERDICT [{hypothesis}]: {reading}")

        payload = {
            "flow": {"checkpoint": str(arguments.flow_checkpoint),
                     "step": probe_meta.get("step"),
                     "sha256": file_sha256(arguments.flow_checkpoint),
                     "condition_features": list(flow.config.condition_features)},
            "autoencoder": {"checkpoint": str(ae_path), "sha256": file_sha256(ae_path)},
            "settings": {"targets_n": targets_n, "qed_pair": pair,
                         "temperatures": temperatures, "count": arguments.count,
                         "decoders": list(decoders), "seed": arguments.seed},
            "support": support,
            "support_definition": ("counts are training molecules with EXACTLY this N and "
                                   "|QED - target| <= 0.05; 'minimum_over_qed_values' is what "
                                   "the stage-18 report printed as 'support'"),
            "probe": probe_report,
            "decoder_qed_transfer": transfer,
            "interventions": [{k: v for k, v in row.items() if k != "records"}
                              for row in interventions],
            "verdict": verdict,
            "data_fingerprint": fingerprint,
            "seconds": time.perf_counter() - started,
        }
        atomic_write_json(payload, output_root / "decoder_information.json")

        lines = ["Stage 19: decoder-information audit", "", f"VERDICT [{hypothesis}]",
                 reading, "",
                 "A. does the decoder preserve a QED difference it is GIVEN? (real "
                 "embeddings, same N)", "",
                 "|   N | true difference | " + " | ".join(f"{name} decoded" for name in decoders)
                 + " | preserved (best) |",
                 "|----:|----------------:|" + "|".join(["---:"] * len(decoders)) + "|---:|"]
        for row in transfer:
            if "preservation" not in row:
                lines.append(f"| {row['n']} | insufficient real molecules | "
                             + " | ".join(["-"] * len(decoders)) + " | - |")
                continue
            best = max((v for v in row["preservation"].values() if np.isfinite(v)), default=np.nan)
            lines.append(f"| {row['n']} | {row['true_qed_difference']:+.4f} | "
                         + " | ".join(f"{row['decoded_qed_difference'][name]:+.4f}"
                                      for name in decoders)
                         + f" | {best:+.3f} |")
        lines += ["", "B/C. paired intervention: only the QED request changes, noise fixed", "",
                  "|   T |   N | embedding/node | bond-logit/pair | argmax changed | probe QED | "
                  f"raw paired QED | {decoder_name} paired QED | OOD x |",
                  "|----:|----:|---------------:|----------------:|---------------:|----------:|"
                  "---------------:|------------------:|------:|"]
        for row in interventions:
            if not row.get("usable_molecules"):
                lines.append(f"| {row['temperature']:g} | {row['n']} | non-finite samples "
                             f"({row.get('nonfinite_rate', float('nan')) * 100:.0f}%) | | | | | | |")
                continue
            raw_paired = row["decoders"].get("raw", {}).get("paired", {}).get("qed_change_mean",
                                                                              float("nan"))
            head_paired = row["decoders"][decoder_name]["paired"]["qed_change_mean"]
            lines.append(
                f"| {row['temperature']:g} | {row['n']} | "
                f"{row['embedding']['per_node_norm']:.4f} | "
                f"{row['bond_logits']['per_pair_l2']:.4f} | "
                f"{row['bond_logits']['argmax_change_rate'] * 100:.2f}% | "
                f"{row.get('probe', {}).get('predicted_change_mean', float('nan')):+.4f} | "
                f"{raw_paired:+.4f} | {head_paired:+.4f} | "
                f"{row.get('out_of_distribution', {}).get('generated_over_real_mean', float('nan')):.2f} |")
        lines += ["",
                  f"The read-out's own held-out MAE is {validation['mae']:.4f} on real "
                  f"validation embeddings, so it cannot certify a shift smaller than that. "
                  f"Its predictions on generated embeddings are extrapolation (see the OOD "
                  f"column) and are reported as diagnostics only - never as achieved QED.",
                  "",
                  "E. support (training molecules with exactly this N and |QED - target| <= 0.05)",
                  "",
                  "|   N | at this N | " + " | ".join(f"QED {t:g}" for t in pair) + " | minimum |",
                  "|----:|----------:|" + "|".join(["---:"] * len(pair)) + "|---:|"]
        for row in support:
            lines.append(f"| {row['n']} | {row['molecules_at_this_n']:,} | "
                         + " | ".join(f"{row['by_qed'][f'{t:g}']['molecules_within_window']:,}"
                                      for t in pair)
                         + f" | {row['minimum_over_qed_values']:,} |")
        lines.append("")
        lines.append("'minimum' is what the stage-18 report printed as its support column.")
        (output_root / "decoder_information.txt").write_text("\n".join(lines) + "\n",
                                                             encoding="utf-8")
        print("\n".join(lines), flush=True)
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
