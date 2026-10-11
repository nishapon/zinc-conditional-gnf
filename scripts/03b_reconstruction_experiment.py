#!/usr/bin/env python3
"""Stage 3b — controlled reconstruction experiments (short, bounded, reproducible).

One named VARIANT of the autoencoder is trained for a FIXED, small budget on a
fixed subset of the training pool and scored on a fixed validation subset, so
variants can be compared like for like. Every result is stratified by whether
a molecule is reachable at all for a deterministic one-hot equivariant encoder
(the symmetry ceiling), because that is where variants differ.

Variants (all keep the 17-type one-hot identity input; none changes the
teammates' flow methodology):

  baseline    teammates' v5 design (default)
  noise4      + 4 random symmetry-breaking noise channels beside the identity
  noise8      + 8 noise channels
  deepbond    bond head with 3 layers (more expressive pairwise decoder)
  refine3     3 rounds of valence-aware bond-logit refinement
  wide128     node embedding dimension 128 (only worth keeping if it clearly helps)
  steps12     12 message-passing steps (deeper encoder)
  position    DIAGNOSTIC ONLY: deterministic atom-position features (non-equivariant;
              a flow cannot supply consistent positions, so never a final model)
  degree      per-atom heavy-degree head + degree-constrained assembly (deterministic,
              equivariant: the encoder, embeddings and flow data are unchanged)
  degree_noise4  degree head together with 4 noise channels

Subset modes:
  --subset all            normal training pool
  --subset decodable      only molecules a deterministic encoder CAN reconstruct
  --subset unreachable    only molecules it CANNOT (symmetry-ambiguous)
  --overfit               train AND evaluate on the same N molecules (capacity /
                          bug check: a correct model should memorise them)

This script trains briefly. It writes only under --output-dir, never touches
existing checkpoints or datasets, and never loads the test split. Cluster use
goes through sbatch (slurm/03_reconstruction_variant.sh).

Exit codes: 0 ok; 2 data missing; 4 bad arguments; 1 unexpected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from zinc_gnf.autoencoder import AutoencoderConfig, evaluate_reconstruction  # noqa: E402
from zinc_gnf.dataset import (  # noqa: E402
    DatasetFormatError,
    MolecularGraphDataset,
    load_training_splits,
    seeded_subset_indices,
)
from zinc_gnf.reconstruction_analysis import (  # noqa: E402
    embedding_class_check,
    embedding_robustness,
    embedding_statistics,
    noise_draw_spread,
)
from zinc_gnf.symmetry import is_decodable  # noqa: E402
from zinc_gnf.trainer_autoencoder import AutoencoderTrainer, TrainConfig, load_autoencoder  # noqa: E402
from zinc_gnf.utils.logging import JobLogger  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402

VARIANTS = {
    "baseline": {},
    "noise4": {"symmetry_noise_dim": 4},
    "noise8": {"symmetry_noise_dim": 8},
    "deepbond": {"bond_head_layers": 3},
    "refine3": {"num_bond_refine_steps": 3},
    "wide128": {"node_embedding_dim": 128},
    "steps12": {"num_processing_steps": 12},
    "position": {"position_features": True},
    # Equivariant per-atom degree head + degree-constrained assembly (deterministic).
    "degree": {"degree_head": True},
    "degree_noise4": {"degree_head": True, "symmetry_noise_dim": 4},
}


def select_by_mode(records: list[dict], mode: str) -> list[dict]:
    if mode == "all":
        return records
    flags = [is_decodable(r["atom_types"], r["bond_matrix"]) for r in records]
    keep = [r for r, f in zip(records, flags) if f == (mode == "decodable")]
    if not keep:
        raise ValueError(f"no molecules left for --subset {mode}")
    return keep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="baseline")
    parser.add_argument("--data-dir", default="data/processed/zinc250k_charge_aware_moflow_fixed")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subset", choices=("all", "decodable", "unreachable"), default="all")
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--train-molecules", type=int, default=30000)
    parser.add_argument("--val-molecules", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-flow-metrics", action="store_true",
                        help="skip the embedding statistics / robustness / noise-spread measurements")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override any AutoencoderConfig field, e.g. --set node_embedding_dim=96")
    arguments = parser.parse_args(argv)

    output = Path(arguments.output_dir)
    if (output / "result.json").exists():
        print(f"FATAL: {output / 'result.json'} already exists; experiments never overwrite results",
              file=sys.stderr, flush=True)
        return 4
    try:
        overrides = dict(VARIANTS[arguments.variant])
        for item in arguments.set:
            key, _, value = item.partition("=")
            overrides[key] = json.loads(value)
        model_config = AutoencoderConfig.from_dict(overrides)
        logger = JobLogger("experiment", output, filename="train.log")
        started = time.perf_counter()
        train_records, val_records, fingerprint = load_training_splits(arguments.data_dir)
        picker = np.random.default_rng(arguments.seed)
        train_pool = select_by_mode(train_records, arguments.subset)
        val_pool = select_by_mode(val_records, arguments.subset)
        del train_records, val_records
        order = picker.permutation(len(train_pool))[: arguments.train_molecules]
        train_sel = [train_pool[i] for i in sorted(order)]
        if arguments.overfit:
            val_sel = train_sel
        else:
            vorder = np.random.default_rng(12345).permutation(len(val_pool))[: arguments.val_molecules]
            val_sel = [val_pool[i] for i in sorted(vorder)]
        train_set, val_set = MolecularGraphDataset(train_sel), MolecularGraphDataset(val_sel)
        fingerprint = dict(fingerprint, used={"train": len(train_set), "val": len(val_set)},
                           subset=arguments.subset, overfit=arguments.overfit)
        logger.info(f"variant={arguments.variant} subset={arguments.subset} overfit={arguments.overfit} "
                    f"train={len(train_set):,} val={len(val_set):,} steps={arguments.steps} seed={arguments.seed}")

        train_config = TrainConfig(
            data_dir=arguments.data_dir, output_dir=str(output), seed=arguments.seed, device=arguments.device,
            batch_size=arguments.batch_size, max_steps=arguments.steps, learning_rate=arguments.learning_rate,
            log_every=max(1, arguments.steps // 8), eval_every=arguments.eval_every,
            save_every=arguments.eval_every, val_loss_molecules=len(val_set),
            val_recon_molecules=len(val_set),
        )
        trainer = AutoencoderTrainer(model_config, train_config, logger,
                                     datasets=(train_set, val_set, fingerprint))
        summary = trainer.train(resume="none")

        model, _ = load_autoencoder(output / "last.pt", device=trainer.device)
        final = evaluate_reconstruction(model, val_set, device=trainer.device, batch_size=128)
        by_symmetry = final.get("exact_molecule_by_symmetry", {})
        use_degree = model.config.degree_head and "degree" in final
        head_key = "degree" if use_degree else "constrained"
        head_sym = "degree_exact" if use_degree else "constrained_exact"
        head_decodable = (by_symmetry.get("decodable", {}).get(head_sym))
        result = {
            "variant": arguments.variant, "model_config": model_config.to_dict(),
            "subset": arguments.subset, "overfit": arguments.overfit, "seed": arguments.seed,
            "train_molecules": len(train_set), "val_molecules": len(val_set), "steps": arguments.steps,
            "batch_size": arguments.batch_size, "device": str(trainer.device),
            "parameters": sum(p.numel() for p in model.parameters()),
            "train_seconds": summary["seconds"], "wall_seconds": time.perf_counter() - started,
            "final": {
                "headline_decoder": head_key,
                "headline_exact": final[head_key]["exact_molecule_match"],
                "headline_validity": final[head_key]["validity"],
                "headline_exact_among_decodable": head_decodable,
                "headline_exact_among_unreachable": by_symmetry.get("non_decodable", {}).get(head_sym),
                "degree_exact": final.get("degree", {}).get("exact_molecule_match"),
                "degree_validity": final.get("degree", {}).get("validity"),
                "degree_accuracy": final.get("degree_accuracy"),
                "degree_fallback_rate": final.get("degree_decode_fallback_rate"),
                "degree_exact_among_decodable": by_symmetry.get("decodable", {}).get("degree_exact"),
                "degree_exact_among_unreachable": by_symmetry.get("non_decodable", {}).get("degree_exact"),
                "constrained_exact": final["constrained"]["exact_molecule_match"],
                "raw_exact": final["raw"]["exact_molecule_match"],
                "constrained_validity": final["constrained"]["validity"],
                "raw_validity": final["raw"]["validity"],
                "atom_accuracy": final["atom_accuracy"], "bond_accuracy": final["bond_accuracy"],
                "real_bond_accuracy": final["real_bond_accuracy"], "mean_loss": final["mean_loss"],
                "ceiling": final.get("equivariant_exact_ceiling"),
                "exact_among_decodable": final.get("constrained_exact_molecule_given_decodable"),
                "exact_among_unreachable": by_symmetry.get("non_decodable", {}).get("constrained_exact"),
                "unreachable_molecules": by_symmetry.get("non_decodable", {}).get("molecules"),
                "decodable_molecules": by_symmetry.get("decodable", {}).get("molecules"),
            },
            "curve": [{k: h.get(k) for k in ("step", "val_mean_loss", "val_exact_molecule_constrained",
                                              "val_validity_constrained")}
                      for h in (summary.get("last_validation"), summary.get("best_validation")) if h],
        }
        if not arguments.no_flow_metrics:
            # Properties of the embeddings that a downstream flow would have to model.
            device = trainer.device
            result["flow_relevant"] = {
                "embedding_statistics": embedding_statistics(model, val_set, device=device),
                "decoder_robustness_to_embedding_error": embedding_robustness(model, val_set, device=device),
                "noise_draw_spread": noise_draw_spread(model, val_set, device=device, draws=5),
                "class_separation": embedding_class_check(model, val_set, device=device),
            }
        atomic_write_json(result, output / "result.json")
    except FileNotFoundError as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 2
    except (ValueError, DatasetFormatError, KeyError, json.JSONDecodeError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1

    f = result["final"]
    print(f"RESULT variant={result['variant']} subset={result['subset']} overfit={result['overfit']} "
          f"params={result['parameters']:,}\n"
          f"  constrained exact={f['constrained_exact']:.3f} raw exact={f['raw_exact']:.3f} "
          f"validity c/r={f['constrained_validity']:.3f}/{f['raw_validity']:.3f} "
          f"atom={f['atom_accuracy']:.4f} bond={f['bond_accuracy']:.4f} loss={f['mean_loss']:.3f}\n"
          f"  ceiling={f['ceiling']:.3f} exact|decodable={f['exact_among_decodable']:.3f} "
          f"exact|unreachable={f['exact_among_unreachable']} "
          f"(n decodable={f['decodable_molecules']}, unreachable={f['unreachable_molecules']})", flush=True)
    if f.get("degree_exact") is not None:
        print(f"  DEGREE decoder: exact={f['degree_exact']:.3f} validity={f['degree_validity']:.3f} "
              f"degree_acc={f['degree_accuracy']:.4f} fallback={f['degree_fallback_rate']:.3f} "
              f"exact|decodable={f['degree_exact_among_decodable']:.3f} "
              f"exact|unreachable={f['degree_exact_among_unreachable']:.3f}", flush=True)
    flow = result.get("flow_relevant")
    if flow:
        st, sp = flow["embedding_statistics"], flow["noise_draw_spread"]
        robust = " ".join(f"s={r['sigma']}:{r['exact']:.3f}" for r in flow["decoder_robustness_to_embedding_error"])
        print(f"  flow-relevant: noise_variance_fraction={st['noise_variance_fraction']:.3f} "
              f"participation_ratio={st['participation_ratio']:.1f}/{st['dimension']} "
              f"exact over 5 noise draws={sp['mean']:.3f}+-{sp['std']:.3f}\n"
              f"  exact under embedding error: {robust}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
