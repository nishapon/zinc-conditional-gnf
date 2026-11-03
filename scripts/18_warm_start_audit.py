#!/usr/bin/env python3
"""Stage 17a — can a conditional flow be warm-started from the unconditional Arm B run?

Three questions, answered before any training is proposed:

A. COMPATIBILITY. Every tensor of both state dictionaries is categorised: transferable
   (same name, same shape), conditional-only (the FiLM layers, the condition embedding and
   the prior heads), unconditional-only (the prior's plain mu / pre_sigma), shape
   mismatches and anything unexpected. The whitening buffers `data_mean` / `data_std` must
   match as well, since two different whitenings mean two different densities and no
   transfer would be meaningful. Nothing is assumed interchangeable; it is checked.

B. FUNCTIONAL EQUIVALENCE AT INITIALISATION. The warm-started conditional model is
   compared with the unconditional one on the same real embeddings, for SEVERAL condition
   vectors, on five quantities: latents, log-determinant, log-likelihood, the inverse, and
   samples drawn from the same RNG stream. FiLM is applied as `h * (1 + gamma) + beta`, so
   zero FiLM weights make it exactly the identity; the prior heads' weights are zeroed and
   their biases set to the unconditional mu / pre_sigma, so mu(c), sigma(c) reproduce the
   unconditional prior for every c. The expected result is therefore BIT-EXACT, and the
   script exits non-zero if it is not - an experiment built on a broken transfer would be
   worthless.

C. WHICH CONDITIONAL COMPONENT COSTS STRUCTURE. On the EXISTING conditional checkpoint,
   with no training at all, four variants are decoded and measured: as trained, with FiLM
   neutralised, with the prior's condition-dependence removed, and with both. If structure
   recovers when FiLM is switched off, FiLM is the culprit; if it recovers when the prior
   is frozen, the prior is; if neither, the damage is in the backbone weights themselves
   and only a warm start (or a different architecture) can help.

The warm-started checkpoint is written to a NEW path. No existing checkpoint is read
anything but read-only, and none is overwritten.

Exit codes: 0 ok; 2 missing input; 3 incompatible or not equivalent; 4 bad arguments;
1 unexpected.
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

from zinc_gnf.conditioning import (  # noqa: E402
    batch_condition,
    condition_embedding_for,
    load_condition_scaler,
    standardise_condition,
)
from zinc_gnf.dataset import (  # noqa: E402
    DatasetFormatError,
    MolecularGraphDataset,
    iter_eval_batches,
    load_training_splits,
    seeded_subset_indices,
)
from zinc_gnf.flow import FlowConfig, NodeFlow, node_mask_from_counts  # noqa: E402
from zinc_gnf.generation import decode_embeddings  # noqa: E402
from zinc_gnf.molecular_profile import profile_molecules  # noqa: E402
from zinc_gnf.structure_objective import structure_statistics  # noqa: E402
from zinc_gnf.trainer_autoencoder import load_autoencoder, resolve_device  # noqa: E402
from zinc_gnf.trainer_flow import (  # noqa: E402
    VALIDATION_SUBSET_SEED,
    encode_batch,
    file_sha256,
    load_flow,
)
from zinc_gnf.utils.checkpointing import save_checkpoint  # noqa: E402
from zinc_gnf.utils.logging import JobLogger  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402
from zinc_gnf.utils.seed import set_seed  # noqa: E402
from zinc_gnf.warm_start import (  # noqa: E402
    equivalence_report,
    freeze_backbone,
    neutralised,
    transfer_report,
    warm_start,
)

RESERVED = {"epoch", "best_metric", "configuration", "model_state_dict",
            "optimizer_state_dict", "torch_rng_state", "cuda_rng_state"}
EQUIVALENCE_TOLERANCE = 1e-5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unconditional-checkpoint", required=True, help="READ-ONLY source")
    parser.add_argument("--conditional-checkpoint", default=None,
                        help="the existing joint run, for the component ablation (READ-ONLY)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warm-start-checkpoint", default=None,
                        help="where to write the warm-started model; default: "
                             "<output-dir>/warm_start.pt (never an existing path)")
    parser.add_argument("--data-dir", default="data/processed/zinc250k_charge_aware_moflow_fixed")
    parser.add_argument("--ae-checkpoint", default=None)
    parser.add_argument("--molecules", type=int, default=64,
                        help="validation molecules for the equivalence check")
    parser.add_argument("--ablation-molecules", type=int, default=256,
                        help="samples per ablation variant (0 = skip the ablation)")
    parser.add_argument("--ablation-n", type=int, default=23)
    parser.add_argument("--ablation-qed", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1,
                        help="seeds the conditional adapters' initialisation; the FiLM and "
                             "prior heads are zeroed anyway, but the condition embedding "
                             "and prior.hidden are drawn from the global RNG, so the "
                             "written checkpoint is reproducible only with this fixed")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    arguments = parse_args(argv)
    output_root = Path(arguments.output_dir)
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        logger = JobLogger("warm_start", output_root, filename="warm_start.log")
    except OSError as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4

    started = time.perf_counter()
    try:
        set_seed(arguments.seed, deterministic=False)
        device = resolve_device(arguments.device)
        source_path = Path(arguments.unconditional_checkpoint)
        unconditional, unconditional_probe = load_flow(source_path, device=device)
        if unconditional.config.conditional:
            raise ValueError(f"{source_path} is a CONDITIONAL checkpoint; this audit expects "
                             f"the unconditional Arm B run as its source")
        logger.header("Warm-start feasibility audit")
        logger.info(f"source (unconditional) {source_path} step={unconditional_probe.get('step')} "
                    f"sha256={file_sha256(source_path)[:16]} "
                    f"node_autoregressive={unconditional.config.node_autoregressive}")

        # The conditional twin: same architecture, plus the two conditional features.
        target_config = FlowConfig.from_dict({**unconditional.config.to_dict(),
                                              "condition_features": ["n", "qed"]})
        target = NodeFlow(target_config, init_seed=arguments.seed).to(device)
        source_state = {k: v.to(device) for k, v in unconditional.state_dict().items()}

        # ---- A. compatibility -------------------------------------------------
        report = transfer_report(source_state, target)
        logger.info(f"compatibility: {report['transferable_count']} tensors transfer exactly; "
                    f"{len(report['conditional_only'])} conditional-only; "
                    f"{len(report['unconditional_only'])} unconditional-only; "
                    f"{len(report['shape_mismatch'])} shape mismatches; "
                    f"{len(report['unexpected'])} unexpected")
        logger.info(report["verdict"])
        for entry in report["conditional_only"][:8]:
            logger.info(f"  conditional-only: {entry['name']} {entry['shape']}")
        for entry in report["unconditional_only"]:
            logger.info(f"  unconditional-only: {entry['name']} {entry['shape']} "
                        f"(consumed by the prior transfer)")
        if not report["compatible"]:
            atomic_write_json({"transfer": report}, output_root / "warm_start.json")
            logger.error("incompatible checkpoints; nothing was written")
            return 3
        for name in ("data_mean", "data_std"):
            if not torch.equal(source_state[name], target.state_dict()[name]):
                pass            # expected: the target starts with defaults, hence the transfer
        transfer = warm_start(target, source_state, neutral=True)
        for action in transfer["actions"]:
            logger.info(f"  {action}")
        for name in ("data_mean", "data_std"):
            assert torch.equal(target.state_dict()[name], source_state[name])
        logger.info("whitening buffers data_mean / data_std transferred identically, so both "
                    "models describe the same density")

        # ---- B. functional equivalence ---------------------------------------
        ae_recorded = (unconditional_probe.get("autoencoder") or {})
        ae_path = Path(arguments.ae_checkpoint or ae_recorded.get("checkpoint", ""))
        if not ae_path.exists():
            raise FileNotFoundError(f"autoencoder checkpoint not found: {ae_path}")
        autoencoder, _ = load_autoencoder(ae_path, device=device)
        autoencoder.eval()
        for parameter in autoencoder.parameters():
            parameter.requires_grad_(False)
        scaler = load_condition_scaler(arguments.data_dir)
        _, val_records, fingerprint = load_training_splits(arguments.data_dir)
        val_set = MolecularGraphDataset(val_records)
        del val_records
        indices = seeded_subset_indices(len(val_set), arguments.molecules,
                                        VALIDATION_SUBSET_SEED)
        batch = next(iter(iter_eval_batches(val_set, arguments.molecules, indices))).to(device)
        with torch.no_grad():
            embeddings = encode_batch(autoencoder, batch)
        probe_conditions = torch.from_numpy(standardise_condition(
            ("n", "qed"), scaler=scaler, n=[10, 23, 38, 23, 23], qed=[0.5, 0.2, 0.9, 0.4, 0.8])
        ).to(device)
        equivalence = equivalence_report(unconditional, target, embeddings,
                                         batch.node_mask, probe_conditions)
        for row in equivalence["rows"]:
            logger.info(f"  condition {np.round(row['condition'], 4).tolist()}: "
                        f"dz {row['max_abs_latent_difference']:.3e}, dlogdet "
                        f"{row['max_abs_log_det_difference']:.3e}, dlogp "
                        f"{row['max_abs_log_prob_difference']:.3e}, dinverse "
                        f"{row['max_abs_inverse_difference']:.3e}, dsample "
                        f"{row['max_abs_sample_difference']:.3e}")
        logger.info(f"equivalence: {equivalence['verdict']}")
        if not equivalence["equivalent"]:
            atomic_write_json({"transfer": transfer, "equivalence": equivalence},
                              output_root / "warm_start.json")
            logger.error("the warm-started model is NOT the unconditional model at "
                         "initialisation; refusing to write a checkpoint")
            return 3

        freeze = freeze_backbone(target)
        logger.info(f"stage A parameter split: {freeze['trainable_parameters']:,} trainable "
                    f"adapter parameters ({freeze['trainable_fraction'] * 100:.2f}% of the "
                    f"model), {freeze['frozen_parameters']:,} frozen backbone parameters")

        destination = Path(arguments.warm_start_checkpoint or (output_root / "warm_start.pt"))
        if destination.exists():
            raise ValueError(f"{destination} already exists; refusing to overwrite it")
        extra = {k: v for k, v in unconditional_probe.items() if k not in RESERVED}
        extra.update({
            "step": 0,
            "flow_config": target_config.to_dict(),
            "warm_start": {"source": str(source_path),
                           "source_sha256": file_sha256(source_path),
                           "source_step": unconditional_probe.get("step"),
                           "transfer": {k: v for k, v in transfer.items()
                                        if k not in ("transferable",)},
                           "equivalence": equivalence,
                           "parameter_split": freeze},
        })
        for parameter in target.parameters():
            parameter.requires_grad_(True)              # the fine-tune decides what to freeze
        save_checkpoint(destination, target, None, epoch=0, best_metric=float("inf"),
                        configuration={"flow": target_config.to_dict()}, extra=extra)
        logger.info(f"warm-started checkpoint written to {destination}; the source is unchanged")

        # ---- C. which conditional component costs structure? -----------------
        ablation: list[dict[str, Any]] = []
        if arguments.conditional_checkpoint and arguments.ablation_molecules:
            conditional, _ = load_flow(Path(arguments.conditional_checkpoint), device=device)
            if set(conditional.config.condition_features) != {"n", "qed"}:
                raise ValueError("--conditional-checkpoint must be the joint (n, qed) run")
            count = arguments.ablation_molecules
            counts = [arguments.ablation_n] * count
            mask = node_mask_from_counts(counts, device=device)
            values = torch.from_numpy(standardise_condition(
                ("n", "qed"), scaler=scaler, n=counts,
                qed=[arguments.ablation_qed] * count)).to(device)
            decoders = ("raw", "degree") if autoencoder.config.degree_head else ("raw",)

            def measure(label: str) -> dict[str, Any]:
                with torch.no_grad():
                    embedding = condition_embedding_for(conditional, values, None, count, device)
                    generator = torch.Generator(device=device).manual_seed(7)
                    z = conditional.prior.sample(mask, embedding,
                                                 temperature=arguments.temperature,
                                                 generator=generator)
                    sampled = conditional.inverse(z, mask, embedding)
                    # NON-FINITE FIRST. The node-autoregressive inverse is a sequential
                    # recursion: node i is recovered as (y_i - t_i) * exp(-s_i) from the
                    # already-recovered nodes, and while the log-scale is tanh-clamped the
                    # TRANSLATION head is not, so the recursion can be expansive and
                    # overflow float32 across up to N nodes x 12 blocks. In the trained
                    # conditional model FiLM is part of what keeps that recursion
                    # contractive, so switching FiLM off can produce inf -> NaN. A NaN is a
                    # numerical FAILURE, never evidence of structural recovery, so the
                    # statistics below are suppressed when any sample is non-finite.
                    finite_nodes = torch.isfinite(sampled).all(dim=-1) | ~mask
                    finite = finite_nodes.all(dim=-1)
                    nonfinite_rate = float((~finite).to(torch.float64).mean())
                    if nonfinite_rate > 0.0:
                        logger.warn(
                            f"ablation [{label}]: {nonfinite_rate * 100:.1f}% of samples are "
                            f"NON-FINITE (the sequential node-autoregressive inverse "
                            f"overflowed). This variant is numerically broken; its chemistry "
                            f"is NOT comparable and is reported as unavailable.")
                        return {"variant": label, "nonfinite_rate": nonfinite_rate,
                                "usable": False,
                                "note": ("non-finite embeddings: the unbounded translation "
                                         "head makes the node-autoregressive inverse "
                                         "expansive once FiLM no longer regulates it. "
                                         "NaN is not structural recovery.")}
                    from zinc_gnf.generation import generation_batch
                    outputs = autoencoder.decode(sampled, generation_batch(counts, device=device),
                                                 teacher_forcing=False)
                    statistics = structure_statistics(outputs["bond_logits"],
                                                      outputs["atom_logits"], mask)
                records = decode_embeddings(autoencoder, sampled, counts, decoders=decoders)
                head = "degree" if "degree" in decoders else "raw"
                smiles = [r[head].get("smiles") if (r.get(head) or {}).get("valid") else None
                          for r in records]
                profile = profile_molecules(smiles, name=label, scaffolds=False)
                row = {
                    "variant": label,
                    "usable": True,
                    "nonfinite_rate": nonfinite_rate,
                    "bonds_per_atom": float(statistics["bonds_per_atom"].mean()),
                    "cycle_rank": float(statistics["cycle_rank"].mean()),
                    "double_bond_fraction": float(statistics["double_bond_fraction"].mean()),
                    "isolated_atom_rate": float(statistics["isolated_atom_rate"].mean()),
                    "raw_validity": sum(1 for r in records if (r.get("raw") or {}).get("valid"))
                    / len(records),
                    "corrected_validity": sum(1 for r in records
                                              if (r.get(head) or {}).get("valid")) / len(records),
                    "fallback_rate": sum(1 for r in records
                                         if (r.get(head) or {}).get("fallback")) / len(records),
                    "aromatic_rings_mean": profile["aromatic_rings"]["mean"],
                    "qed_mean": profile["qed"]["mean"],
                }
                logger.info(f"ablation [{label}]: bonds/atom {row['bonds_per_atom']:.4f}, "
                            f"cycle rank {row['cycle_rank']:.3f}, isolated "
                            f"{row['isolated_atom_rate']:.4f}, raw validity "
                            f"{row['raw_validity']:.4f}, aromatic rings "
                            f"{row['aromatic_rings_mean']:.3f}, QED {row['qed_mean']:.4f}")
                return row

            ablation.append(measure("as_trained"))
            with neutralised(conditional, film=True):
                ablation.append(measure("film_neutralised"))
            with neutralised(conditional, prior=True, reference_condition=values[:1]):
                ablation.append(measure("prior_condition_frozen"))
            with neutralised(conditional, film=True, prior=True,
                             reference_condition=values[:1]):
                ablation.append(measure("both_neutralised"))
            baseline = ablation[0]
            usable = [row for row in ablation[1:] if row.get("usable")]
            broken = [row["variant"] for row in ablation[1:] if not row.get("usable")]
            if broken:
                logger.warn(f"variants that produced NON-FINITE samples and therefore say "
                            f"NOTHING about structure: {broken}. A NaN is a numerical "
                            f"failure, not recovery - see the mechanism note in the log.")
            if not baseline.get("usable"):
                logger.error("even the as-trained variant is non-finite at this temperature; "
                             "the ablation cannot be interpreted")
            elif usable:
                best = max(usable, key=lambda r: r["aromatic_rings_mean"])
                improved = best["aromatic_rings_mean"] > baseline["aromatic_rings_mean"]
                logger.info(
                    f"among the FINITE variants the best is '{best['variant']}' "
                    f"(aromatic rings {baseline['aromatic_rings_mean']:.3f} -> "
                    f"{best['aromatic_rings_mean']:.3f}, raw validity "
                    f"{baseline['raw_validity']:.4f} -> {best['raw_validity']:.4f}): "
                    + ("switching that component off recovers structure, so conditioning is "
                       "the proximate cause" if improved else
                       "no finite variant recovers structure, so the loss is in the BACKBONE "
                       "weights - which is what the warm start replaces"))
            else:
                logger.warn("no ablation variant was finite, so this ablation is "
                            "uninformative; re-run it at a higher temperature or with "
                            "float64 if a verdict is needed")

        payload = {
            "source": {"checkpoint": str(source_path),
                       "sha256": file_sha256(source_path),
                       "step": unconditional_probe.get("step")},
            "warm_start_checkpoint": str(destination),
            "transfer": transfer,
            "equivalence": equivalence,
            "parameter_split": freeze,
            "component_ablation": ablation,
            "ablation_settings": {"n": arguments.ablation_n, "qed": arguments.ablation_qed,
                                  "temperature": arguments.temperature,
                                  "samples": arguments.ablation_molecules,
                                  "conditional_checkpoint": arguments.conditional_checkpoint},
            "data_fingerprint": fingerprint,
            "seed": arguments.seed,
            "seconds": time.perf_counter() - started,
        }
        atomic_write_json(payload, output_root / "warm_start.json")
        lines = ["Warm-start feasibility audit", "", report["verdict"], "",
                 f"transferable tensors: {report['transferable_count']}",
                 f"conditional-only (initialised neutral): {len(report['conditional_only'])}",
                 f"unconditional-only (consumed by the prior transfer): "
                 f"{len(report['unconditional_only'])}", "",
                 f"equivalence at initialisation: {equivalence['verdict']}",
                 *(f"  {key}: {value:.3e}" for key, value in equivalence["worst"].items()),
                 "",
                 f"stage A trains {freeze['trainable_parameters']:,} of "
                 f"{freeze['trainable_parameters'] + freeze['frozen_parameters']:,} parameters "
                 f"({freeze['trainable_fraction'] * 100:.2f}%)"]
        if ablation:
            lines += ["", "Component ablation on the EXISTING conditional checkpoint "
                      f"(N={arguments.ablation_n}, QED={arguments.ablation_qed}, "
                      f"T={arguments.temperature}):", "",
                      "| variant | bonds/atom | cycle rank | isolated | raw validity | "
                      "aromatic rings | QED |", "|---|---:|---:|---:|---:|---:|---:|"]
            for row in ablation:
                if not row.get("usable"):
                    lines.append(f"| {row['variant']} | NON-FINITE "
                                 f"({row['nonfinite_rate'] * 100:.0f}% of samples) - not "
                                 f"comparable | | | | | |")
                    continue
                lines.append(f"| {row['variant']} | {row['bonds_per_atom']:.4f} | "
                             f"{row['cycle_rank']:.3f} | {row['isolated_atom_rate']:.4f} | "
                             f"{row['raw_validity']:.4f} | "
                             f"{row['aromatic_rings_mean']:.3f} | {row['qed_mean']:.4f} |")
            lines += ["", "A NON-FINITE row is a numerical failure, not structural recovery: "
                      "the node-autoregressive inverse is a sequential recursion whose "
                      "translation head is unbounded, so removing FiLM can make it expansive "
                      "and overflow float32."]
        (output_root / "warm_start.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
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
