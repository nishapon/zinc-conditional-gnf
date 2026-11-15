#!/usr/bin/env python3
"""Train ONE arm of experiment C: the bond-aware pairwise coupling conditioner.

Separate from `scripts/44_train_arm.py` (experiments A and B) and from
`scripts/04_train_flow.py`, which is not touched. Drives
`zinc_gnf.bond_arms.BondArmTrainer`, a subclass of the experiment-arm trainer that
overrides two hooks and nothing else.

  control              the existing conditioner; continued training only
  pairwise             the bond-aware pathway, LIKELIHOOD TRAINING ALONE
  pairwise_supervised  the same pathway plus the bond-prediction loss

All three use the same parent checkpoint, step budget, learning rate, seed, data and
decoder. `pairwise` and `pairwise_supervised` are architecturally identical and differ
only in `--bond-loss-weight`, which is forced to 0 for `pairwise`, so the architecture and
the training signal can be separated.

--inspect-only builds the arm's flow, prints the measured parameter and activation cost
and the bond-target statistics on one real batch, and exits WITHOUT training. Run this
before queueing a GPU job: it is the memory and runtime estimate.

Exit codes: 0 ok; 2 missing input; 3 refused (provenance or architecture); 4 bad
arguments; 1 unexpected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zinc_gnf.bond_arms import (  # noqa: E402
    BOND_ARCHITECTURE_KEYS,
    BOND_ARMS,
    DEFAULT_BOND_LOSS_WEIGHT,
    BondArmConfig,
    BondArmTrainer,
    bond_flow_config,
    estimate_cost,
)
from zinc_gnf.dataset import (  # noqa: E402
    DeterministicBatcher,
    MolecularGraphDataset,
    load_training_splits,
    make_batch,
)
from zinc_gnf.experiment_arms import prepare_warm_start  # noqa: E402
from zinc_gnf.trainer_flow import FlowTrainConfig  # noqa: E402
from zinc_gnf.utils.logging import JobLogger  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=BOND_ARMS)
    parser.add_argument("--start-checkpoint", required=True,
                        help="shared parent; translated into the arm directory, never "
                             "modified in place")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir",
                        default="data/processed/zinc250k_charge_aware_moflow_fixed")
    parser.add_argument("--ae-checkpoint",
                        default="outputs/ablation/ae_degree_128d/seed_3/last.pt")
    parser.add_argument("--screening-steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--val-molecules", type=int, default=5000)
    parser.add_argument("--bond-pair-dim", type=int, default=16,
                        help="pair scorer width; the dominant memory term scales with it")
    parser.add_argument("--bond-loss-weight", type=float, default=None,
                        help=f"auxiliary bond loss weight (default "
                             f"{DEFAULT_BOND_LOSS_WEIGHT} for pairwise_supervised, "
                             f"forced to 0 for the other arms)")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def _bond_target_statistics(train_set, batch_size: int) -> dict:
    """Positive rate and degree on one real batch. The imbalance the loss must handle."""
    import torch

    from zinc_gnf.bond_pairwise import (
        bond_targets,
        pair_mask_from_node_mask,
        upper_triangle_mask,
    )

    indices = DeterministicBatcher(len(train_set), batch_size, 0).next_indices()
    batch = make_batch(train_set, indices)
    pair_mask = pair_mask_from_node_mask(batch.node_mask)
    counted = upper_triangle_mask(pair_mask)
    target = bond_targets(batch.bond_types)
    pairs = float(counted.sum())
    positives = float((target * counted.to(target.dtype)).sum())
    nodes = float(batch.node_mask.sum())
    degree = float((target * pair_mask.to(target.dtype)).sum() / max(nodes, 1.0))
    return {
        "molecules": int(batch.batch_size),
        "width": int(batch.node_mask.shape[1]),
        "unordered_real_pairs": pairs,
        "bonded_pairs": positives,
        "positive_rate": positives / max(pairs, 1.0),
        "all_negative_accuracy": (pairs - positives) / max(pairs, 1.0),
        "pos_weight_that_will_be_used": (pairs - positives) / max(positives, 1.0),
        "mean_true_degree": degree,
        "note": ("all_negative_accuracy is what 'predict no bond everywhere' scores, so "
                 "bond_accuracy must be read against it; bond_f1 is 0 for that "
                 "predictor by construction"),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        weight = args.bond_loss_weight
        if weight is None:
            weight = (DEFAULT_BOND_LOSS_WEIGHT
                      if args.arm == "pairwise_supervised" else 0.0)
        arm = BondArmConfig(arm=args.arm, bond_pair_dim=args.bond_pair_dim,
                            bond_loss_weight=float(weight))
        arm.validate()
        out_dir = Path(args.output_dir)
        if Path(args.start_checkpoint).resolve().parent == out_dir.resolve():
            raise ValueError("--output-dir is the parent checkpoint's own directory; "
                             "refusing to risk overwriting the shared warm start")
        out_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 4

    started = time.perf_counter()
    try:
        import torch

        from zinc_gnf.flow import NodeFlow

        start = Path(args.start_checkpoint)
        if not start.is_file():
            raise FileNotFoundError(f"parent checkpoint not found: {start}")

        train_records, val_records, fingerprint = load_training_splits(args.data_dir)
        train_set = MolecularGraphDataset(train_records)
        val_set = MolecularGraphDataset(val_records)
        del train_records, val_records

        probe = torch.load(start, map_location="cpu", weights_only=False)
        flow_config = bond_flow_config(probe["flow_config"], arm)
        max_nodes = int(train_set.max_nodes)

        # measured, not guessed: build the arm's own flow and read the cost off it
        preview = NodeFlow(flow_config, init_seed=args.seed)
        cost = estimate_cost(preview, batch_size=args.batch_size, max_nodes=max_nodes)
        bonds = _bond_target_statistics(train_set, args.batch_size)
        plan = {"arm": arm.to_dict(), "cost": cost, "bond_targets": bonds,
                "architecture_keys_allowed": list(BOND_ARCHITECTURE_KEYS),
                "parent_flow_config": probe["flow_config"],
                "arm_flow_config": flow_config.to_dict()}
        print(json.dumps(plan, indent=2, default=str))
        atomic_write_json(plan, out_dir / "bond_arm_plan.json")
        del preview

        if args.inspect_only:
            print("\ninspect-only: nothing was trained")
            print(f"new parameters: {cost['new_parameters']:,} "
                  f"(+{100 * cost['parameter_increase_fraction']:.1f}%)")
            print(f"extra activation memory at batch {args.batch_size}, N={max_nodes}: "
                  f"{cost['activation_mb_total']:.0f} MB")
            print(f"bond positive rate: {bonds['positive_rate']:.4f} "
                  f"(all-negative accuracy {bonds['all_negative_accuracy']:.4f})")
            return 0

        resume_target = out_dir / "last.pt"
        interrupted = resume_target.exists()
        train_config = FlowTrainConfig(
            data_dir=args.data_dir, output_dir=str(out_dir),
            ae_checkpoint=args.ae_checkpoint, seed=args.seed, device=args.device,
            batch_size=args.batch_size,
            # step is reset to 0 by the translation, so max_steps IS the number of
            # additional optimizer steps and the warm-up is experiment-local
            max_steps=args.screening_steps,
            learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
            eval_every=args.eval_every, save_every=args.save_every,
            val_molecules=args.val_molecules, early_stopping_patience=None)
        train_config.validate()

        warm_start = None
        parent_provenance = None
        if not interrupted:
            sampler_state = DeterministicBatcher(
                len(train_set), train_config.batch_size,
                train_config.seed).state_dict()
            warm_start = prepare_warm_start(
                start, resume_target, arm_config=arm, train_config=train_config,
                flow_config=flow_config, sampler_state=sampler_state,
                # the ONLY keys this experiment may add; everything else must match
                architecture_exceptions=BOND_ARCHITECTURE_KEYS)
            parent_provenance = warm_start["parent"]
            print("PATH 1 (new experiment from parent):")
            print(json.dumps(warm_start, indent=2, default=str))
        else:
            existing = torch.load(resume_target, map_location="cpu",
                                   weights_only=False)
            parent_provenance = existing.get("parent")
            print(f"PATH 2 (resuming interrupted experiment at step "
                  f"{existing.get('step')}; parent "
                  f"{(parent_provenance or {}).get('checkpoint')})")

        logger = JobLogger(f"bond_arm_{arm.arm}", out_dir, filename="train.log")
        trainer = BondArmTrainer(flow_config, train_config, logger=logger,
                                 datasets=(train_set, val_set, fingerprint),
                                 arm_config=arm,
                                 parent_provenance=parent_provenance)
        if interrupted:
            trainer.verify_arm(torch.load(resume_target, map_location="cpu",
                                           weights_only=False))
        summary = trainer.train(resume="auto")
        atomic_write_json(
            {"arm": arm.to_dict(), "arm_report": trainer.arm_report,
             "start_checkpoint": str(start), "start_step": probe.get("step"),
             "screening_steps": args.screening_steps,
             "max_steps": train_config.max_steps,
             "learning_rate": args.learning_rate,
             "interrupted_resume": interrupted, "warm_start": warm_start,
             "cost": cost, "bond_targets": bonds, "summary": summary,
             "elapsed_seconds": time.perf_counter() - started},
            out_dir / "bond_arm_summary.json")
        print(f"\narm {arm.arm} finished in "
              f"{(time.perf_counter() - started) / 60:.1f} min")
        print(f"wrote {out_dir / 'bond_arm_summary.json'}")
        return 0
    except FileNotFoundError as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2
    except RuntimeError as error:
        print(f"FATAL (refused): {error}", file=sys.stderr)
        return 3
    except Exception:  # noqa: BLE001
        print("FATAL: unexpected\n" + traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
