#!/usr/bin/env python3
"""Stage 18a — does the conditioning MOVE the samples, and by how much of what is needed?

The question Stage 17 left open. The QED-only likelihood gap is nearly identical in all
three models (+0.0741 / +0.0768 / +0.0867) while the sampled response differs 4.5-fold
(slope 0.0315 / 0.0901 / 0.1409), and at N=23 the warm start returns 0.5633 / 0.5611 /
0.5704 for requests 0.4 / 0.6 / 0.8. A likelihood gap says the density notices the
condition; generation is about where the samples land. This job measures the second, with
no training:

  delivered   || mean generated embedding at QED_high - at QED_low ||, with the SAME
              noise draws for both, so the difference is conditioning and not variance
  required    || mean REAL embedding at (N, QED_high) - at (N, QED_low) ||, from TRAINING
              molecules within +-0.05 of each target
  ratio       delivered / required, plus the cosine between the two displacement
              directions, and delivered / within-condition sampling spread

and it reports the SUPPORT (how many real molecules exist at each (N, QED)), so a
badly supported target is never read as a model failure.

It also counts NON-FINITE samples per cell and refuses to report a displacement computed
from them. That matters: the node-autoregressive inverse is a sequential recursion whose
log-scale is tanh-clamped but whose TRANSLATION head is not, so it can be expansive and
overflow float32 - which is what produced the NaNs when FiLM was neutralised in stage 17.
A NaN is a numerical failure, never evidence about chemistry.

The decision this feeds: ratio << 1 means prior and FiLM do not deliver the displacement
the data demands, so capacity is the binding constraint (stage B is justified, with a
structure guardrail); a good ratio with a poor cosine means the mechanism moves samples
in the wrong direction (a different fix); a good ratio and cosine means control is lost
after the embeddings, i.e. in the decoder's resolution.

Read-only: no checkpoint is modified and the test split is never loaded.
Exit codes: 0 ok; 2 missing input; 4 bad arguments; 1 unexpected.
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

from zinc_gnf.conditional_signal import (  # noqa: E402
    delivered_displacement,
    format_signal_report,
    required_displacement,
    signal_ratio,
)
from zinc_gnf.conditioning import load_condition_scaler  # noqa: E402
from zinc_gnf.dataset import (  # noqa: E402
    DatasetFormatError,
    MolecularGraphDataset,
    load_training_splits,
)
from zinc_gnf.trainer_autoencoder import load_autoencoder, resolve_device  # noqa: E402
from zinc_gnf.trainer_flow import file_sha256, load_flow  # noqa: E402
from zinc_gnf.utils.logging import JobLogger  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402
from zinc_gnf.utils.seed import set_seed  # noqa: E402

# Below this delivered/required ratio the conditioning cannot express what the data
# demands, and extra capacity (stage B) is the justified next step.
CAPACITY_RATIO = 0.2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flow-checkpoint", required=True, help="READ-ONLY")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", default="model", help="name for this model in the report")
    parser.add_argument("--data-dir", default="data/processed/zinc250k_charge_aware_moflow_fixed")
    parser.add_argument("--ae-checkpoint", default=None)
    parser.add_argument("--target-n", default="15,23,30")
    parser.add_argument("--qed-pair", default="0.4,0.8", help="low,high")
    parser.add_argument("--count", type=int, default=256, help="samples per condition")
    parser.add_argument("--temperature", default="0.2,1.0", help="comma-separated")
    parser.add_argument("--real-molecules", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
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
        logger = JobLogger("conditional_signal", output_root, filename="signal.log")
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4

    started = time.perf_counter()
    try:
        set_seed(arguments.seed, deterministic=False)
        device = resolve_device(arguments.device)
        flow, probe = load_flow(Path(arguments.flow_checkpoint), device=device)
        if "qed" not in flow.config.condition_features:
            raise ValueError(f"this flow does not condition on QED "
                             f"(features {list(flow.config.condition_features)})")
        recorded = probe.get("autoencoder") or {}
        ae_path = Path(arguments.ae_checkpoint or recorded.get("checkpoint", ""))
        if not ae_path.exists():
            raise FileNotFoundError(f"autoencoder checkpoint not found: {ae_path}")
        autoencoder, _ = load_autoencoder(ae_path, device=device)
        autoencoder.eval()
        for parameter in autoencoder.parameters():
            parameter.requires_grad_(False)
        scaler = load_condition_scaler(arguments.data_dir)
        train_records, _, fingerprint = load_training_splits(arguments.data_dir)
        train_set = MolecularGraphDataset(train_records)
        del train_records

        logger.header(f"Conditional signal audit [{arguments.label}]")
        logger.info(f"flow={arguments.flow_checkpoint} step={probe.get('step')} "
                    f"sha256={file_sha256(arguments.flow_checkpoint)[:16]} (READ-ONLY)")
        logger.info(f"N={targets_n} QED {pair[0]} -> {pair[1]} at temperatures "
                    f"{temperatures}, {arguments.count} samples per condition")

        results: list[dict[str, Any]] = []
        for temperature in temperatures:
            for n in targets_n:
                delivered = delivered_displacement(
                    flow, scaler=scaler, n=n, qed_low=pair[0], qed_high=pair[1],
                    count=arguments.count, temperature=temperature, device=device,
                    seed=arguments.seed)
                required = required_displacement(
                    autoencoder, train_set, n=n, qed_low=pair[0], qed_high=pair[1],
                    max_molecules=arguments.real_molecules, device=device)
                ratio = signal_ratio(delivered, required)
                results.append({"label": arguments.label, "temperature": temperature,
                                "n": n, "delivered": delivered, "required": required,
                                "ratio": ratio})
                support = min(required.get("low_molecules", 0), required.get("high_molecules", 0))
                logger.info(
                    f"T={temperature:g} N={n}: delivered "
                    f"{delivered.get('embedding_mean_shift_norm', float('nan')):.4f}, required "
                    f"{required.get('embedding_mean_shift_norm', float('nan')):.4f}, ratio "
                    f"{ratio['ratio']:.3f}, cosine {ratio['cosine']:+.3f}, "
                    f"shift/spread {delivered.get('shift_over_spread', float('nan')):.3f}, "
                    f"support {support:,} real molecules, non-finite "
                    f"{max(delivered['per_condition']['low']['finite']['nonfinite_rate'], delivered['per_condition']['high']['finite']['nonfinite_rate']) * 100:.1f}%")
                logger.info(f"    {ratio['verdict']}")

        usable = [r for r in results if np.isfinite(r["ratio"]["ratio"])]
        median_ratio = float(np.median([r["ratio"]["ratio"] for r in usable])) if usable \
            else float("nan")
        median_cosine = float(np.median([r["ratio"]["cosine"] for r in usable])) if usable \
            else float("nan")
        decision = {
            "median_ratio": median_ratio,
            "median_cosine": median_cosine,
            "capacity_ratio_threshold": CAPACITY_RATIO,
            "stage_b_justified": bool(np.isfinite(median_ratio) and median_ratio < CAPACITY_RATIO
                                      and np.isfinite(median_cosine) and median_cosine > 0.3),
            "reading": (
                "capacity-bound: the conditioning delivers a small fraction of the required "
                "displacement in broadly the right direction, so giving the conditional path "
                "more capacity (stage B) is the justified next step"
                if np.isfinite(median_ratio) and median_ratio < CAPACITY_RATIO
                and np.isfinite(median_cosine) and median_cosine > 0.3 else
                "direction-bound: the conditioning moves samples the wrong way, so more "
                "capacity in the same mechanism is not indicated"
                if np.isfinite(median_cosine) and median_cosine <= 0.3 else
                "the displacement is already comparable with the data's: control is lost "
                "after the embeddings, so look at the decoder rather than the flow"
                if np.isfinite(median_ratio) else
                "not measurable - see the non-finite column"),
        }
        logger.info(f"DECISION: median ratio {median_ratio:.3f}, median cosine "
                    f"{median_cosine:+.3f} -> {decision['reading']}")

        report = format_signal_report(results)
        (output_root / "signal.txt").write_text(
            report + "\n\n" + f"DECISION: {decision['reading']}\n", encoding="utf-8")
        atomic_write_json({"flow": {"checkpoint": str(arguments.flow_checkpoint),
                                    "step": probe.get("step"),
                                    "sha256": file_sha256(arguments.flow_checkpoint),
                                    "condition_features": list(flow.config.condition_features)},
                           "label": arguments.label,
                           "settings": {"targets_n": targets_n, "qed_pair": pair,
                                        "temperatures": temperatures,
                                        "count": arguments.count, "seed": arguments.seed},
                           "results": results, "decision": decision,
                           "data_fingerprint": fingerprint,
                           "seconds": time.perf_counter() - started},
                          output_root / "signal.json")
        print(report, flush=True)
        print(f"\nDECISION: {decision['reading']}", flush=True)
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
