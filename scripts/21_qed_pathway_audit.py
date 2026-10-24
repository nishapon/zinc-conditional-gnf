#!/usr/bin/env python3
"""Stage 20 — resolve the conditional-signal and decoder-constraint contributions.

Stage 19 printed one verdict, `H2-direction`, from a priority ladder that tests H2 before
H3 - so the degree-decoder attenuation visible in its own tables could not reach the
verdict. Two further problems made that verdict weaker than it looked:

  * the probe's single-prediction held-out MAE (0.0885) was used as the detection
    threshold for the MEAN of a PAIRED difference over hundreds of pairs. Those are
    different quantities, and the comparison is biased towards "no signal";
  * the transfer control ran on TRAINING molecules, which the autoencoder was fitted on,
    so a preserved QED gap there does not establish that the decoder generalises;
  * the N=23 raw estimate rested on 6 valid pairs.

This job fixes the measurements and reports FOUR axes independently, so no axis can
suppress another:

  A  decoder QED transfer on HELD-OUT real molecules (validation; the test split is never
     loaded), with per-molecule values so the preservation ratio gets a bootstrap
     interval, next to aromaticity and per-decoder validity. The train split is measured
     too, purely so the generalisation gap is visible.
  B  the read-outs. The stage-19 MLP probe is carried over unmodified and a second,
     closed-form within-N ridge read-out is fitted on TRAIN only (alpha chosen on an inner
     split of train). For each, on REAL validation molecules: MAE and correlation overall
     and stratified by N and QED band, calibration, the within-N GAIN (what a true QED gap
     looks like through this instrument at fixed N), and the NULL (what it reports for
     real pairs in the SAME band - the actual detection threshold for a paired mean).
  C  the generated embeddings read through those instruments, with paired bootstrap
     intervals, a resolution decision against the real-pair null, and the implied real QED
     difference after dividing by the measured gain.
  D  constrained versus degree-constrained decoding of IDENTICAL embeddings on one common
     valid-pair subset, with the attenuation decomposed over fallback / solved / mixed
     strata - a decomposition that is exact because the degree decoder returns precisely
     the constrained decoder's bonds whenever it falls back.
  E  support, every count with its denominator, on both splits.

Then four independent findings and ONE recommendation among A (flow), B (decoder) or
C (representation), with benefits, risks, costs and preconditions. Nothing is retrained
except the two diagnostic read-outs; no checkpoint is written; no architecture changes.

The read-outs are diagnostic only: they never guide generation, never rerank a sample and
are never reported as achieved QED.

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
    seeded_subset_indices,
)
from zinc_gnf.decoder_contrast import (  # noqa: E402
    decoder_contrast,
    paired_generated_embeddings,
    summarise_attenuation,
    transfer_on_split,
)
from zinc_gnf.decoder_information import (  # noqa: E402
    fit_embedding_probe,
    ood_statistics,
    support_counts,
)
from zinc_gnf.qed_evidence import (  # noqa: E402
    assess_evidence,
    format_report,
    recommend_intervention,
)
from zinc_gnf.qed_probes import (  # noqa: E402
    MlpProbeReadout,
    WithinNRidgeProbe,
    calibration,
    distribution_discriminability,
    encode_pooled,
    interpret_generated_change,
    null_contrast,
    pooled_features,
    probe_scores,
    stratified_scores,
    within_n_reference_contrast,
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flow-checkpoint", required=True, help="READ-ONLY")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="data/processed/zinc250k_charge_aware_moflow_fixed")
    parser.add_argument("--ae-checkpoint", default=None)
    parser.add_argument("--target-n", default="15,23,30")
    parser.add_argument("--qed-pair", default="0.4,0.8")
    parser.add_argument("--count", type=int, default=384, help="samples per condition")
    parser.add_argument("--temperature", default="0.2,0.6,1.0")
    parser.add_argument("--qed-window", type=float, default=0.05)
    parser.add_argument("--probe-train-molecules", type=int, default=20000)
    parser.add_argument("--probe-val-molecules", type=int, default=4000)
    parser.add_argument("--probe-epochs", type=int, default=10)
    parser.add_argument("--transfer-molecules", type=int, default=256)
    parser.add_argument("--reference-pairs", type=int, default=256)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--max-examples", type=int, default=5)
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
        logger = JobLogger("qed_pathway", output_root, filename="audit.log")
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4

    started = time.perf_counter()
    resamples = int(arguments.bootstrap_resamples)
    window = float(arguments.qed_window)
    requested = pair[1] - pair[0]
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
        decoders = (("raw", "constrained", "degree") if autoencoder.config.degree_head
                    else ("raw", "constrained"))
        headline = "degree" if "degree" in decoders else "constrained"
        scaler = load_condition_scaler(arguments.data_dir)
        train_records, val_records, fingerprint = load_training_splits(arguments.data_dir)
        train_set = MolecularGraphDataset(train_records)
        val_set = MolecularGraphDataset(val_records)
        del train_records, val_records

        logger.header("Stage 20: QED pathway audit")
        logger.info(f"flow={arguments.flow_checkpoint} step={flow_meta.get('step')} "
                    f"sha256={file_sha256(arguments.flow_checkpoint)[:16]} (READ-ONLY)")
        logger.info(f"N={targets_n} QED {pair[0]} -> {pair[1]} at T={temperatures}, "
                    f"{arguments.count} samples per condition, decoders={list(decoders)}")
        logger.info("the test split is never loaded; no checkpoint is written")

        # ---- E. support on BOTH splits ---------------------------------------
        support: list[dict[str, Any]] = []
        for name, dataset in (("train", train_set), ("validation", val_set)):
            for row in support_counts(dataset, n_values=targets_n, qed_values=pair,
                                      window=window):
                row["split"] = name
                support.append(row)
                parts = ", ".join(
                    f"QED {key}: {value['molecules_within_window']:,}"
                    for key, value in row["by_qed"].items())
                logger.info(f"support [{name}] N={row['n']}: {row['molecules_at_this_n']:,} "
                            f"molecules at this N; {parts}; minimum "
                            f"{row['minimum_over_qed_values']:,}")
        logger.info("low-QED support is the limiting denominator; a cell with a tiny "
                    "minimum is reported but never read as a model failure")

        # ---- A. transfer on held-out molecules, and on train for the gap -----
        logger.header("A. decoder QED transfer on real molecules")
        transfer: list[dict[str, Any]] = []
        for split_name, dataset in (("validation", val_set), ("train", train_set)):
            for n in targets_n:
                row = transfer_on_split(autoencoder, dataset, n=n, qed_low=pair[0],
                                        qed_high=pair[1], window=window,
                                        max_molecules=arguments.transfer_molecules,
                                        device=device, decoders=decoders,
                                        split_name=split_name, seed=arguments.seed)
                transfer.append(row)
                block = (row.get("preservation") or {}).get(headline)
                if block and np.isfinite(block.get("ratio", float("nan"))):
                    logger.info(
                        f"[{split_name}] N={n}: true gap "
                        f"{block['true_difference']:+.4f}, decoded "
                        f"{block['measured_difference']:+.4f}, preserved "
                        f"{block['ratio']:.3f} "
                        f"[{block.get('ci_low', float('nan')):+.3f}, "
                        f"{block.get('ci_high', float('nan')):+.3f}] "
                        f"on {block['low_molecules']}/{block['high_molecules']} molecules "
                        f"({headline} decoder)")
                else:
                    logger.warn(f"[{split_name}] N={n}: "
                                f"{row.get('note_insufficient', 'no preservation measurable')}")

        # ---- B. the two read-outs -------------------------------------------
        logger.header("B. the read-outs: can either instrument see a within-N change?")
        probe_train = seeded_subset_indices(len(train_set), arguments.probe_train_molecules,
                                            STATISTICS_SEED)
        probe_val = seeded_subset_indices(len(val_set), arguments.probe_val_molecules,
                                          VALIDATION_SUBSET_SEED)
        mlp_probe, mlp_report = fit_embedding_probe(
            autoencoder, train_set, val_set, train_indices=probe_train,
            val_indices=probe_val, device=device, epochs=arguments.probe_epochs)
        train_pool = encode_pooled(autoencoder, train_set, probe_train, device=device)
        val_pool = encode_pooled(autoencoder, val_set, probe_val, device=device)
        ridge = WithinNRidgeProbe.fit(train_pool["features"], train_pool["qed"],
                                      train_pool["n"], seed=arguments.seed)
        logger.info(f"ridge read-out: alpha {ridge.alpha:g} chosen on an inner split of "
                    f"TRAIN ({train_pool['qed'].size:,} molecules); validation never used "
                    f"for fitting")

        readouts: list[dict[str, Any]] = []
        instruments = [MlpProbeReadout(mlp_probe), ridge]
        for readout in instruments:
            entry: dict[str, Any] = {"name": readout.name, "definition": readout.to_dict()}
            entry["validation"] = probe_scores(readout, val_pool["features"],
                                               val_pool["qed"], val_pool["n"])
            entry["train"] = probe_scores(readout, train_pool["features"],
                                          train_pool["qed"], train_pool["n"])
            entry["stratified"] = stratified_scores(readout, val_pool["features"],
                                                    val_pool["qed"], val_pool["n"],
                                                    n_values=targets_n)
            entry["calibration"] = calibration(readout, val_pool["features"],
                                               val_pool["qed"], val_pool["n"])
            entry["reference_contrasts"] = []
            entry["null_contrasts"] = []
            for n in targets_n:
                entry["reference_contrasts"].append(within_n_reference_contrast(
                    readout, val_pool["features"], val_pool["qed"], val_pool["n"],
                    target_n=n, qed_low=pair[0], qed_high=pair[1], window=window,
                    max_pairs=arguments.reference_pairs, seed=arguments.seed,
                    resamples=resamples))
                entry["null_contrasts"].append(null_contrast(
                    readout, val_pool["features"], val_pool["qed"], val_pool["n"],
                    target_n=n, qed_centre=pair[1], window=window,
                    max_pairs=arguments.reference_pairs, seed=arguments.seed,
                    resamples=resamples))
            readouts.append(entry)
            logger.info(f"[{readout.name}] validation MAE "
                        f"{entry['validation']['mae']:.4f}, correlation "
                        f"{entry['validation']['correlation']:.4f}, calibration slope "
                        f"{entry['calibration'].get('slope_of_truth_on_prediction', float('nan')):.3f}")
            for reference, null in zip(entry["reference_contrasts"], entry["null_contrasts"]):
                logger.info(
                    f"    N={reference['n']}: within-N gain "
                    f"{reference.get('gain', float('nan')):+.3f} "
                    f"[{reference.get('gain_ci_low', float('nan')):+.3f}, "
                    f"{reference.get('gain_ci_high', float('nan')):+.3f}] on "
                    f"{reference.get('pairs', 0)} real pairs; null sd "
                    f"{null.get('null_sd', float('nan')):.4f} on {null.get('pairs', 0)} "
                    f"same-band pairs")
        logger.info("the GAIN, not the held-out MAE, sets the scale: a single-prediction MAE "
                    "is not the detection threshold for the mean of a paired difference")

        # ---- C and D: one generation per cell, read and decoded --------------
        logger.header("C and D. generated embeddings: read-out response and decoder contrast")
        generated_readings: list[dict[str, Any]] = []
        contrasts: list[dict[str, Any]] = []
        generated_pooled: list[np.ndarray] = []
        generated_nodes: list[torch.Tensor] = []
        for temperature in temperatures:
            for n in targets_n:
                paired = paired_generated_embeddings(
                    flow, scaler=scaler, n=n, qed_low=pair[0], qed_high=pair[1],
                    count=arguments.count, temperature=temperature, device=device,
                    seed=arguments.seed)
                if not paired["usable_molecules"]:
                    logger.warn(f"T={temperature:g} N={n}: every sample non-finite "
                                f"({paired['nonfinite_rate'] * 100:.1f}%); nothing measured")
                    contrasts.append({"n": n, "temperature": temperature,
                                      "nonfinite_rate": paired["nonfinite_rate"],
                                      "note": "every sample was non-finite"})
                    continue
                low_features = pooled_features(paired["low"], paired["mask"]) \
                    .to(torch.float64).cpu().numpy()
                high_features = pooled_features(paired["high"], paired["mask"]) \
                    .to(torch.float64).cpu().numpy()
                generated_pooled.append(high_features)
                generated_nodes.append(paired["high"][paired["mask"]].detach())
                for readout, entry in zip(instruments, readouts):
                    reference = next((b for b in entry["reference_contrasts"]
                                      if b.get("n") == n), {})
                    null = next((b for b in entry["null_contrasts"] if b.get("n") == n), {})
                    values = (np.asarray(readout.contrast(high_features), dtype=np.float64)
                              - np.asarray(readout.contrast(low_features), dtype=np.float64))
                    reading = interpret_generated_change(
                        values, null=null, reference=reference, requested=requested,
                        resamples=resamples, seed=arguments.seed)
                    reading.update({"instrument": readout.name, "n": int(n),
                                    "temperature": float(temperature),
                                    "pairs": int(values.size)})
                    generated_readings.append(reading)
                    observed = reading["observed"]
                    logger.info(
                        f"T={temperature:g} N={n} [{readout.name}]: paired change "
                        f"{observed['mean']:+.4f} "
                        f"[{observed.get('ci_low', float('nan')):+.4f}, "
                        f"{observed.get('ci_high', float('nan')):+.4f}] on "
                        f"{observed['pairs']} pairs -> {reading['status']}; implied real QED "
                        f"difference {reading.get('implied_real_qed_difference', float('nan')):+.3f} "
                        f"against a requested {requested:+.2f}")

                row = decoder_contrast(flow, autoencoder, scaler=scaler, n=n,
                                       qed_low=pair[0], qed_high=pair[1],
                                       count=arguments.count, temperature=temperature,
                                       device=device, seed=arguments.seed,
                                       max_examples=arguments.max_examples,
                                       resamples=resamples, paired=paired)
                contrasts.append(row)
                common = row.get("common_subset") or {}
                if common.get("molecules"):
                    gap = common.get("gap_constrained_minus_degree") or {}
                    logger.info(
                        f"    common valid subset {common['molecules']}: constrained "
                        f"{common['constrained']['mean']:+.4f}, degree "
                        f"{common['degree']['mean']:+.4f}, gap "
                        f"{gap.get('difference', float('nan')):+.4f} "
                        f"[{gap.get('ci_low', float('nan')):+.4f}, "
                        f"{gap.get('ci_high', float('nan')):+.4f}]"
                        + ("  <- excludes zero" if gap.get("excludes_zero") else ""))
                    attribution = row.get("attribution") or {}
                    degrees = attribution.get("predicted_degrees") or {}
                    logger.info(
                        f"    predicted degrees changed in "
                        f"{degrees.get('change_rate', float('nan')) * 100:.1f}% of molecules; "
                        f"degree fallback rate "
                        f"{(attribution.get('fallback_rate') or {}).get('high', float('nan')) * 100:.1f}%; "
                        f"fallback identity holds: "
                        f"{attribution.get('fallback_identity_holds')}")
                    for label, block in (attribution.get("strata") or {}).items():
                        logger.info(f"      stratum {label:<14} {block['pairs']:>4} pairs "
                                    f"({block['share_of_common_subset'] * 100:5.1f}%), gap "
                                    f"{block.get('difference', float('nan')):+.4f}, "
                                    f"contribution "
                                    f"{block.get('contribution_to_gap', float('nan')):+.4f}")
                    logger.info(f"    raw validity (native, separate): "
                                f"{row['validity']['raw']['low']:.3f} -> "
                                f"{row['validity']['raw']['high']:.3f}")
                else:
                    logger.warn(f"    {row.get('note', 'no common valid subset')}")

        # ---- the distribution shift -----------------------------------------
        logger.header("the distribution shift: are generated embeddings on-manifold?")
        shift: dict[str, Any] = {}
        if generated_pooled:
            shift = distribution_discriminability(val_pool["features"],
                                                  np.vstack(generated_pooled),
                                                  seed=arguments.seed)
            logger.info(f"held-out real-vs-generated AUC on pooled embeddings: "
                        f"{shift.get('held_out_auc', float('nan')):.3f} "
                        f"(0.5 = indistinguishable)")
        if generated_nodes:
            from zinc_gnf.dataset import iter_eval_batches

            reference_nodes = []
            with torch.no_grad():
                for batch in iter_eval_batches(val_set, 128, probe_val[:1024]):
                    batch = batch.to(device)
                    nodes = autoencoder.encode(batch)
                    reference_nodes.append(nodes[batch.node_mask].detach())
            node_statistics = ood_statistics(torch.cat(reference_nodes),
                                             torch.cat(generated_nodes))
            shift.update({k: v for k, v in node_statistics.items() if k != "note"})
            shift["mahalanobis_note"] = node_statistics["note"]
            logger.info(f"node-level Mahalanobis generated/real mean ratio: "
                        f"{shift.get('generated_over_real_mean', float('nan')):.3f} "
                        f"(carried over from stage 19 for continuity; the AUC is the "
                        f"sharper statistic)")

        # ---- the four findings and the recommendation ------------------------
        attenuation = summarise_attenuation(contrasts)
        evidence = assess_evidence(readouts=readouts, transfer=transfer,
                                   attenuation=attenuation, shift=shift,
                                   generated_readings=generated_readings,
                                   headline_decoder=headline)
        recommendation = recommend_intervention(evidence)
        logger.header("findings")
        for axis, finding in evidence["findings"].items():
            logger.info(f"[{finding['status'].upper()}] {axis}: {finding['reasoning']}")
        logger.info(f"RECOMMENDATION {recommendation['primary']} "
                    f"(confidence {recommendation['confidence']}): "
                    f"{recommendation['rationale']}")

        payload = {
            "stage": 20,
            "flow": {"checkpoint": str(arguments.flow_checkpoint),
                     "step": flow_meta.get("step"),
                     "sha256": file_sha256(arguments.flow_checkpoint),
                     "condition_features": list(flow.config.condition_features)},
            "autoencoder": {"checkpoint": str(ae_path),
                            "degree_head": bool(autoencoder.config.degree_head)},
            "dataset_fingerprint": fingerprint,
            "target_n": targets_n, "qed_pair": pair, "temperatures": temperatures,
            "count": int(arguments.count), "qed_window": window,
            "bootstrap_resamples": resamples,
            "decoders": list(decoders), "headline_decoder": headline,
            "seed": int(arguments.seed),
            "support": support,
            "support_definition": ("every count states its denominator; "
                                   "minimum_over_qed_values is the MINIMUM of the two QED "
                                   "groups at that N, not a matched subset"),
            "transfer": transfer,
            "readouts": readouts,
            "generated_readings": generated_readings,
            "contrasts": contrasts,
            "attenuation": attenuation,
            "distribution_shift": shift,
            "evidence": evidence,
            "recommendation": recommendation,
            "probe_provenance": {
                "stage19_mlp": mlp_report,
                "within_n_ridge": ridge.to_dict(),
                "note": ("both read-outs are fitted on TRAIN only and are diagnostic: "
                         "neither guides generation, reranks samples, nor is reported as "
                         "achieved QED"),
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_write_json(payload, output_root / "qed_pathway.json")
        (output_root / "qed_pathway.txt").write_text(format_report(payload),
                                                     encoding="utf-8")
        logger.info(f"wrote {output_root / 'qed_pathway.json'} and "
                    f"{output_root / 'qed_pathway.txt'}")
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
