#!/usr/bin/env python3
"""Stage 8 — real vs node-shuffled vs flow-generated embeddings (the shuffled-node control).

The frozen autoencoder reproduces molecular chemistry almost exactly from real
embeddings, while 300-step flow samples decode to QED 0.240 with 0.025 aromatic rings
(but 1.075 total rings). Per-node predictions survive much better than pairwise ones.
This job tests whether destroying ONLY the inter-node relationship is enough to produce
that pattern.

Three populations with IDENTICAL molecule sizes:

  real          the autoencoder's embeddings of validation molecules
  shuffled      the same node vectors, globally permuted and re-partitioned into molecules
                of the original sizes, so every per-node marginal, the covariance and the
                principal components are preserved EXACTLY and only the grouping changes
  real_smoothed real embeddings plus the SAME Gaussian smoothing the flow trains against
                (--noise-sigma, default 0.05) - the flow's actual target, and the honest
                reference for "how close must symmetry-equivalent embeddings be"
  flow_T*       embeddings sampled from a trained flow checkpoint for the same sizes

For each population it reports
  * embedding geometry: norms, marginals, covariance spectrum, participation ratio,
    variance projected onto the real (TRAINING) eigenbasis, within-molecule pairwise
    distances / cosines / minimum distances, the between-molecule baseline, and how
    often a molecule contains a near-identical embedding pair (real molecules do,
    because symmetry-equivalent atoms get identical embeddings);
  * decoder predictions BEFORE any assembly: atom-type and charge-aware-type histograms,
    confidences, degree histogram, degree-sum parity and feasibility, bond-logit argmax
    histogram and P(no bond);
  * the assembled molecules: QED, QED >= 0.7, total rings, aromatic rings, aromatic/all
    rings, ring bond orders, bond composition, strict validity and degree fallback.

What this can and cannot establish is printed in the report and repeated here: if the
shuffled control reproduces the flow's symptoms, losing the inter-node relationship is a
SUFFICIENT explanation, and the flow's samples are consistent with having learned
per-node marginals but not the joint structure. It cannot prove that is what the flow
does - the flow's own embedding statistics, measured here, are the evidence for that -
and it cannot distinguish "not trained enough" from "cannot be learned".

Read-only: nothing is trained; the autoencoder, the flow checkpoint, the dataset and all
earlier outputs are untouched, and the test split is never loaded.

Exit codes: 0 ok; 2 missing checkpoint or data; 3 autoencoder/flow mismatch;
4 bad arguments; 1 unexpected.
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

import torch  # noqa: E402

from zinc_gnf.dataset import (  # noqa: E402
    DatasetFormatError,
    MolecularGraphDataset,
    load_training_splits,
    seeded_subset_indices,
)
from zinc_gnf.embedding_populations import (  # noqa: E402
    decode_population,
    smoothed_population,
    decoder_prediction_statistics,
    embedding_population_statistics,
    flow_population,
    format_population_statistics,
    format_prediction_statistics,
    real_population,
    shuffle_population,
)
from zinc_gnf.latent_geometry import covariance_spectrum, node_embeddings  # noqa: E402
from zinc_gnf.molecular_profile import (  # noqa: E402
    compare_profiles,
    format_comparison,
    profile_molecules,
)
from zinc_gnf.trainer_autoencoder import (  # noqa: E402
    ResumeMismatchError,
    load_autoencoder,
    resolve_device,
)
from zinc_gnf.trainer_flow import file_sha256, load_flow  # noqa: E402
from zinc_gnf.utils.logging import JobLogger  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402

REFERENCE_SUBSET_SEED = 4242       # TRAINING subset for the reference basis (as the trainer uses)
EVAL_SUBSET_SEED = 12345           # VALIDATION subset, same as every other diagnostic

CAVEAT = (
    "WHAT THIS SHOWS AND WHAT IT DOES NOT.\n"
    "  The shuffled population keeps exactly the same multiset of node embeddings as the\n"
    "  real one, so its marginals, covariance and principal components are identical by\n"
    "  construction; only which nodes belong to the same molecule has changed.\n"
    "  * If shuffled reproduces the flow's symptoms, then losing the inter-node relation\n"
    "    is SUFFICIENT to cause them, and the flow's output is consistent with having\n"
    "    learned per-node marginals but not the joint structure.\n"
    "  * It does NOT prove that is what the flow does. The flow's own embedding\n"
    "    statistics in this report (participation ratio, variance inside the real\n"
    "    eigenbasis, within/between distance ratio, near-identical pairs) are the\n"
    "    evidence for that.\n"
    "  * A sufficient cause is not a necessary one: this cannot separate 'not trained\n"
    "    enough' from 'cannot be learned by this architecture'. Only a longer run with\n"
    "    the same measurements can.\n"
    "  * Total rings versus aromatic rings separates a missing ring skeleton\n"
    "    (connectivity) from a ring with the wrong bond orders (kekule pattern)."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ae-checkpoint", default="outputs/ablation/ae_degree/seed_3/last.pt")
    parser.add_argument("--data-dir", default="data/processed/zinc250k_charge_aware_moflow_fixed")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--flow-checkpoint", default=None,
                        help="a trained flow; omit to run the shuffled control alone")
    parser.add_argument("--temperature", default="0.85,1.0")
    parser.add_argument("--noise-sigma", type=float, default=0.05,
                        help="add a 'real + Gaussian smoothing' reference population using the "
                             "flow's own sigma (0 disables it)")
    parser.add_argument("--molecules", type=int, default=2000,
                        help="VALIDATION molecules; every population uses these same sizes")
    parser.add_argument("--reference-molecules", type=int, default=20000,
                        help="TRAINING molecules for the reference eigenbasis")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--allow-ae-mismatch", action="store_true",
                        help="use an autoencoder other than the one the flow was trained with")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    output = Path(arguments.output_dir)
    try:
        temperatures = [float(t) for t in str(arguments.temperature).split(",") if t.strip()]
        if arguments.molecules < 2:
            raise ValueError("--molecules must be >= 2")
        if arguments.flow_checkpoint and not temperatures:
            raise ValueError("no temperature given for the flow populations")
        output.mkdir(parents=True, exist_ok=True)
        logger = JobLogger("populations", output, filename="populations.log")
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4

    try:
        started = time.perf_counter()
        device = resolve_device(arguments.device)
        checkpoint = Path(arguments.ae_checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"autoencoder checkpoint not found: {checkpoint}")
        autoencoder, probe = load_autoencoder(checkpoint, device=device)
        for parameter in autoencoder.parameters():
            parameter.requires_grad_(False)
        autoencoder.eval()
        autoencoder_digest = file_sha256(checkpoint)

        train_records, val_records, fingerprint = load_training_splits(arguments.data_dir)
        train_set = MolecularGraphDataset(train_records)
        val_set = MolecularGraphDataset(val_records)
        del train_records, val_records

        logger.header("Embedding populations: real vs shuffled vs flow")
        logger.info(f"autoencoder={checkpoint} sha256={autoencoder_digest[:16]} "
                    f"step={probe.get('step')} degree_head={autoencoder.config.degree_head}")
        logger.info(f"data={arguments.data_dir} train={len(train_set):,} val={len(val_set):,} "
                    f"(test split NOT loaded)")

        # Reference eigenbasis from the TRAINING split only.
        reference_indices = seeded_subset_indices(len(train_set), arguments.reference_molecules,
                                                  REFERENCE_SUBSET_SEED)
        logger.info(f"[ref] encoding {len(reference_indices):,} TRAINING molecules for the basis")
        reference_embeddings = node_embeddings(autoencoder, train_set, indices=reference_indices,
                                               device=device, batch_size=arguments.batch_size)
        reference_summary, _, reference_eigenvectors = covariance_spectrum(reference_embeddings)
        reference_scale = reference_summary["per_dimension_std_mean"]
        # per-dimension std of the TRAINING embeddings: the scale the flow's smoothing uses
        reference_std = reference_embeddings.to(torch.float64).std(0).to(torch.float32)
        del reference_embeddings

        # The validation molecules every population shares.
        evaluation_indices = seeded_subset_indices(len(val_set), arguments.molecules,
                                                   EVAL_SUBSET_SEED)
        logger.info(f"[pop] {len(evaluation_indices):,} VALIDATION molecules; all populations "
                    f"use these same sizes")
        populations = [real_population(autoencoder, val_set, evaluation_indices, device=device,
                                       batch_size=arguments.batch_size, name="real")]
        shuffled, shuffle_report = shuffle_population(populations[0], seed=arguments.seed)
        populations.append(shuffled)
        if arguments.noise_sigma > 0:
            # The flow's actual target: real embeddings convolved with the training noise.
            populations.append(smoothed_population(
                populations[0], per_dimension_std=reference_std, sigma=arguments.noise_sigma,
                seed=arguments.seed))
            logger.info(f"[pop] real_smoothed: sigma={arguments.noise_sigma} of the per-dimension "
                        f"training std (the distribution the flow is trained against)")
        logger.info(f"[pop] shuffled: {shuffle_report['permuted_nodes']:,} node vectors permuted; "
                    f"{shuffle_report['fraction_back_in_their_own_molecule']:.4f} landed back in "
                    f"their own molecule")

        flow_metadata: dict[str, Any] = {}
        if arguments.flow_checkpoint:
            flow_path = Path(arguments.flow_checkpoint)
            if not flow_path.exists():
                raise FileNotFoundError(f"flow checkpoint not found: {flow_path}")
            flow, flow_probe = load_flow(flow_path, device=device)
            recorded = (flow_probe.get("autoencoder") or {}).get("sha256")
            if recorded and recorded != autoencoder_digest:
                message = (f"the flow was trained with autoencoder sha256 {recorded[:16]} but "
                           f"{checkpoint} has {autoencoder_digest[:16]}")
                if not arguments.allow_ae_mismatch:
                    raise ResumeMismatchError(message)
                logger.warn(message + " (continuing because --allow-ae-mismatch was given)")
            flow_metadata = {"checkpoint": str(flow_path), "step": flow_probe.get("step"),
                             "flow_config": flow_probe.get("flow_config")}
            logger.info(f"[pop] flow={flow_path} step={flow_probe.get('step')} "
                        f"temperatures={temperatures}")
            for temperature in temperatures:
                populations.append(flow_population(
                    flow, populations[0].counts, temperature=temperature, seed=arguments.seed,
                    device=device, batch_size=arguments.batch_size))
        else:
            logger.info("[pop] no flow checkpoint given: running the shuffled control alone")

        result: dict[str, Any] = {
            "autoencoder": {"checkpoint": str(checkpoint), "sha256": autoencoder_digest,
                            "step": probe.get("step")},
            "flow": flow_metadata,
            "data": {k: fingerprint.get(k) for k in
                     ("representation", "split_protocol", "counts", "counts_digest")},
            "settings": {"molecules": len(evaluation_indices),
                         "reference_molecules": len(reference_indices),
                         "temperatures": temperatures, "seed": arguments.seed,
                         "device": str(device),
                         "reference_basis_split": "train", "population_split": "validation",
                         "noise_sigma": arguments.noise_sigma},
            "shuffle": shuffle_report,
            "reference_spectrum": {k: reference_summary[k] for k in
                                   ("participation_ratio", "effective_rank_99",
                                    "per_dimension_std_mean", "max_absolute_correlation",
                                    "condition_number_max_over_median")},
            "embedding_statistics": [], "decoder_predictions": [],
        }
        profiles = []
        for population in populations:
            logger.info(f"[{population.name}] embedding statistics")
            result["embedding_statistics"].append(embedding_population_statistics(
                population, reference_eigenvectors=reference_eigenvectors,
                reference_scale=reference_scale, seed=arguments.seed))
            logger.info(f"[{population.name}] decoder predictions before assembly")
            result["decoder_predictions"].append(decoder_prediction_statistics(
                autoencoder, population, device=device, batch_size=arguments.batch_size))
            logger.info(f"[{population.name}] assembling molecules")
            records = decode_population(autoencoder, population, device=device,
                                        batch_size=arguments.batch_size)
            headline = "degree" if autoencoder.config.degree_head else "constrained"
            smiles = [r[headline]["smiles"] if r[headline]["valid"] else None for r in records]
            profile = profile_molecules(smiles, name=population.name)
            profile["degree_fallback_rate"] = (
                sum(1 for r in records if r[headline].get("fallback")) / len(records))
            profile["decoder"] = headline
            profiles.append(profile)
            logger.info(f"[{population.name}] QED {profile['qed']['mean']:.4f}, "
                        f"QED>=0.7 {profile['qed_at_least_0p7']:.4f}, "
                        f"rings {profile['rings']['mean']:.3f}, "
                        f"aromatic rings {profile['aromatic_rings']['mean']:.3f}, "
                        f"aromatic/all rings {profile['aromatic_ring_fraction']:.4f}, "
                        f"validity {profile['structural_validity']:.4f}, "
                        f"fallback {profile['degree_fallback_rate']:.4f}")
        result["profiles"] = compare_profiles(profiles)
        result["seconds"] = time.perf_counter() - started
        atomic_write_json(result, output / "embedding_populations.json")

        report = "\n\n".join([
            format_population_statistics(result["embedding_statistics"]),
            format_prediction_statistics(result["decoder_predictions"]),
            format_comparison(result["profiles"]),
            "degree-decoder fallback      " + "".join(
                f"{p['degree_fallback_rate']:>20.4f}" for p in profiles),
            CAVEAT,
        ]) + "\n"
        (output / "embedding_populations.txt").write_text(report, encoding="utf-8")
        print(report, flush=True)
        logger.info(f"wrote {output / 'embedding_populations.json'} and .txt "
                    f"in {result['seconds']:.1f}s")
    except FileNotFoundError as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 2
    except ResumeMismatchError as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 3
    except (DatasetFormatError, ValueError) as error:
        print(f"FATAL: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 4
    except Exception:  # noqa: BLE001
        print("FATAL: unexpected error\n" + traceback.format_exc(), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
