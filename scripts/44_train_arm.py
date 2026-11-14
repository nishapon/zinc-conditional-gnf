#!/usr/bin/env python3
"""Train ONE experiment arm from a shared starting checkpoint. Isolated by construction.

Does not touch `scripts/04_train_flow.py`: it drives `zinc_gnf.experiment_arms.ArmTrainer`,
which subclasses `FlowTrainer` and overrides only the two hooks. Every arm uses the same
starting checkpoint, step budget, learning rate, seed and objective; the ONLY difference
is the named intervention.

  control         nothing changed; continued training only, so a gain must beat it
  balanced        BalancedSampler at --balanced-alpha; gated on measured imbalance
  prior_frozen    CoupledNodePrior at rho=0.01 with pre_rho frozen
  prior_learned   CoupledNodePrior at rho=0.01 with pre_rho trainable

--inspect-only measures the joint N-QED imbalance and exits WITHOUT training, so the
`balanced` arm can be gated before a GPU budget is spent. The arm itself also refuses at
startup if the distribution is already near-balanced.

Exit codes: 0 ok; 2 missing input; 3 refused (provenance or imbalance gate); 4 bad
arguments; 1 unexpected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from zinc_gnf.dataset import MolecularGraphDataset, load_training_splits  # noqa: E402
from zinc_gnf.experiment_arms import (  # noqa: E402
    ARMS,
    ArmConfig,
    ArmTrainer,
    imbalance_report,
)
from zinc_gnf.flow import FlowConfig  # noqa: E402
from zinc_gnf.trainer_flow import FlowTrainConfig  # noqa: E402
from zinc_gnf.utils.logging import JobLogger  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--start-checkpoint", required=True,
                        help="shared warm start; copied into the arm directory, never "
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
    parser.add_argument("--balanced-alpha", type=float, default=0.5)
    parser.add_argument("--coupled-init-rho", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        arm = ArmConfig(arm=args.arm, balanced_alpha=args.balanced_alpha,
                        coupled_init_rho=args.coupled_init_rho)
        arm.validate()
        out_dir = Path(args.output_dir)
        if Path(args.start_checkpoint).resolve().parent == out_dir.resolve():
            raise ValueError("--output-dir is the starting checkpoint's own directory; "
                             "refusing to risk overwriting the shared warm start")
        out_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 4

    started = time.perf_counter()
    try:
        train_records, val_records, fingerprint = load_training_splits(args.data_dir)
        train_set = MolecularGraphDataset(train_records)
        val_set = MolecularGraphDataset(val_records)
        del train_records, val_records

        report = imbalance_report(np.asarray(train_set.n_node),
                                  np.asarray(train_set.qed),
                                  qed_bin_width=arm.qed_bin_width)
        print(json.dumps(report, indent=2, default=float))
        atomic_write_json(report, out_dir / "imbalance_report.json")
        if args.inspect_only:
            print("\ninspect-only: nothing was trained")
            print(f"VERDICT: {report['verdict']}")
            return 0
        if arm.uses_balanced_sampler and not report["meaningful_imbalance"]:
            print(f"FATAL: {report['verdict']}", file=sys.stderr)
            return 3

        start = Path(args.start_checkpoint)
        if not start.is_file():
            raise FileNotFoundError(f"starting checkpoint not found: {start}")
        # resume reads last.pt from the output directory, so the shared warm start is
        # COPIED in. The original is never opened for writing.
        resume_target = out_dir / "last.pt"
        if not resume_target.exists():
            shutil.copy2(start, resume_target)
            print(f"warm start copied: {start} -> {resume_target}")

        probe = __import__("torch").load(start, map_location="cpu", weights_only=False)
        flow_config = FlowConfig.from_dict(probe["flow_config"])
        train_config = FlowTrainConfig(
            data_dir=args.data_dir, output_dir=str(out_dir),
            ae_checkpoint=args.ae_checkpoint, seed=args.seed, device=args.device,
            batch_size=args.batch_size,
            max_steps=int(probe.get("step", 0)) + args.screening_steps,
            learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
            eval_every=args.eval_every, save_every=args.save_every,
            val_molecules=args.val_molecules, early_stopping_patience=None)
        train_config.validate()

        logger = JobLogger(f"arm_{arm.arm}", out_dir, filename="train.log")
        trainer = ArmTrainer(flow_config, train_config, logger=logger,
                             datasets=(train_set, val_set, fingerprint),
                             arm_config=arm)
        summary = trainer.train(resume="auto")
        atomic_write_json(
            {"arm": arm.to_dict(), "arm_report": trainer.arm_report,
             "start_checkpoint": str(start), "start_step": probe.get("step"),
             "screening_steps": args.screening_steps,
             "max_steps": train_config.max_steps,
             "learning_rate": args.learning_rate,
             "imbalance": report, "summary": summary,
             "elapsed_seconds": time.perf_counter() - started},
            out_dir / "arm_summary.json")
        print(f"\narm {arm.arm} finished in "
              f"{(time.perf_counter() - started) / 60:.1f} min")
        print(f"wrote {out_dir / 'arm_summary.json'}")
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
