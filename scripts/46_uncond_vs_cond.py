#!/usr/bin/env python3
"""The unconditional-vs-conditional generation-quality table. ONE metric implementation.

WHY THIS SCRIPT HAS TO EXIST
----------------------------
`scripts/42_checkpoint_eval.py` is the right design for comparing two checkpoints under
shared noise, but its `verify_compatibility` REFUSES a pair whose `condition_features`
differ -- which is exactly the pair wanted here. So that script cannot produce an
unconditional-vs-conditional row, and hand-merging a `05_generate.py` run with a
`13_generate_nqed.py` run risks two different budgets, temperatures and novelty
references. This driver runs both arms in one process so that cannot happen.

WHAT IS HELD IDENTICAL
----------------------
  * metric implementation   `metrics_generation.generation_metrics`, called once per arm
  * sample budget           `--count` attempts per arm
  * novelty reference       canonical SMILES of the TRAIN split only, same set object
  * validity correction     all three decoders reported; `--headline degree`
  * temperature             one `--temperature` for both
  * node counts             THE SAME sampled N vector is given to both arms
  * decoder                 the same frozen autoencoder checkpoint for both

WHAT NECESSARILY DIFFERS, AND IS REPORTED
-----------------------------------------
The conditional arm additionally receives a QED request. The requests are drawn as whole
(N, QED) PAIRS from the training split, so their empirical relationship is preserved
rather than assumed independent -- sampling N and QED from their separate marginals would
hand the conditional model combinations that ZINC never contains and then score it on
them. The joint is summarised in the report (per-N QED quantiles, Spearman correlation).

Architecture differences between the two checkpoints are read off their own stored
`flow_config` and printed as a diff. They are NOT silently absorbed: if the two
checkpoints differ in anything beyond `condition_features` and
`condition_embedding_dim`/`condition_hidden_dim`, the report says so and the comparison
must be described as architecture-plus-conditioning, not as the effect of conditioning.

ATOM COUNT IS STRUCTURAL IN BOTH ARMS
-------------------------------------
`generate_molecules` takes the node count from `NodeCountSampler` (the training
histogram) or `target_n`, and the decoder emits exactly one atom per node vector. The
unconditional arm therefore also receives the atom count. Any atom-count agreement is
plumbing, not conditioning, in BOTH arms -- which is why this table's only genuine
conditioning claim is about QED.

Exit codes: 0 ok; 2 missing input; 3 refused (incompatible checkpoints); 4 bad arguments;
1 unexpected.
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

import numpy as np  # noqa: E402

from zinc_gnf.conditioning import load_condition_scaler, standardise_condition  # noqa: E402
from zinc_gnf.dataset import load_training_splits  # noqa: E402
from zinc_gnf.eval_support import (  # noqa: E402
    MOFLOW_DIFFERENCES,
    MOFLOW_SETTINGS,
    FixedCounts,
    aggregate_runs,
)
from zinc_gnf.generation import (  # noqa: E402
    DECODERS,
    generate_molecules,
    node_count_agreement,
)
from zinc_gnf.metrics_generation import generation_metrics  # noqa: E402
from zinc_gnf.trainer_autoencoder import load_autoencoder, resolve_device  # noqa: E402
from zinc_gnf.trainer_flow import file_sha256, load_flow  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402

# Flow-config keys that MUST differ between an unconditional and a conditional model, and
# whose difference is the thing under test rather than a confound.
CONDITIONING_KEYS = ("condition_features", "condition_embedding_dim",
                     "condition_hidden_dim")

# MoFlow protocol constants, the shared size stand-in and the run-level aggregation
# all live in zinc_gnf/eval_support.py so scripts 46, 48 and 49 cannot drift apart.

# ---------------------------------------------------------------- the requests ---
def sample_requests(train_records, count: int, seed: int) -> dict:
    """Draw whole (N, QED) pairs from the training split, preserving their relationship.

    Sampling N and QED from their separate marginals would be the independence assumption
    the brief explicitly forbids: in ZINC the two are associated, so independent draws
    produce requests such as (N=15, QED=0.4) far more often than the data contains them,
    and the conditional model would then be scored largely on combinations it never saw.
    Drawing rows keeps the empirical joint exactly.
    """
    rng = np.random.default_rng(int(seed))
    counts = np.asarray([int(r["n_node"]) for r in train_records], dtype=np.int64)
    qed = np.asarray([float(r["qed"]) for r in train_records], dtype=np.float64)
    finite = np.isfinite(qed)
    if not finite.all():
        counts, qed = counts[finite], qed[finite]
    picks = rng.integers(0, counts.shape[0], size=int(count))
    return {"n": counts[picks], "qed": qed[picks],
            "pool_molecules": int(counts.shape[0]),
            "note": ("requests are whole training rows, so the (N, QED) joint of the "
                     "requests equals the training joint up to sampling error")}


def joint_summary(counts: np.ndarray, qed: np.ndarray) -> dict:
    """QED by heavy-atom count, and the association between them. No independence assumed."""
    order_n = np.argsort(counts, kind="stable")
    ranks_n = np.empty_like(order_n, dtype=np.float64)
    ranks_n[order_n] = np.arange(counts.shape[0], dtype=np.float64)
    order_q = np.argsort(qed, kind="stable")
    ranks_q = np.empty_like(order_q, dtype=np.float64)
    ranks_q[order_q] = np.arange(qed.shape[0], dtype=np.float64)
    spearman = float(np.corrcoef(ranks_n, ranks_q)[0, 1])
    pearson = float(np.corrcoef(counts.astype(np.float64), qed)[0, 1])
    rows = []
    for value in sorted(set(int(c) for c in counts)):
        keep = counts == value
        if int(keep.sum()) < 20:
            continue
        sub = qed[keep]
        rows.append({"n": value, "molecules": int(keep.sum()),
                     "qed_mean": float(sub.mean()), "qed_std": float(sub.std()),
                     "qed_q05": float(np.quantile(sub, 0.05)),
                     "qed_q50": float(np.quantile(sub, 0.50)),
                     "qed_q95": float(np.quantile(sub, 0.95))})
    # support for the deliberate 3x3 grid, measured rather than assumed
    grid = []
    for n_target in (15, 23, 30):
        near = np.abs(counts - n_target) <= 1
        for q_target in (0.4, 0.6, 0.8):
            if int(near.sum()) == 0:
                grid.append({"n": n_target, "qed": q_target, "molecules_within": 0,
                             "fraction_of_that_n": 0.0})
                continue
            within = near & (np.abs(qed - q_target) <= 0.05)
            grid.append({"n": n_target, "qed": q_target,
                         "molecules_at_that_n": int(near.sum()),
                         "molecules_within": int(within.sum()),
                         "fraction_of_that_n": float(within.sum() / near.sum())})
    return {"spearman_n_qed": spearman, "pearson_n_qed": pearson,
            "by_n": rows, "grid_support": grid,
            "interpretation": ("a nonzero correlation means QED cannot be requested "
                               "independently of N without leaving the training support; "
                               "grid_support counts training molecules within +/-1 atom "
                               "and +/-0.05 QED of each deliberate target")}


def achieved_sizes(records, decoder: str) -> dict:
    """Requested vs ACHIEVED heavy-atom counts for one decoder.

    The per-sample records carry the REQUEST as `n_node`; the achieved count has to be
    read off the assembled molecule, exactly as `generation.node_count_agreement` does.
    Reported over valid molecules AND over all attempts, because an invalid sample has no
    size and must not be quietly dropped from a size claim.
    """
    from rdkit import Chem, rdBase

    requested = [int(r["n_node"]) for r in records]
    achieved: list[int] = []
    matched = 0
    with rdBase.BlockLogs():
        for record in records:
            entry = record.get(decoder) or {}
            if not entry.get("valid") or not entry.get("smiles"):
                continue
            molecule = Chem.MolFromSmiles(entry["smiles"])
            if molecule is None:
                continue
            count = int(molecule.GetNumHeavyAtoms())
            achieved.append(count)
            matched += int(count == int(record["n_node"]))
    values, counts = np.unique(requested, return_counts=True)
    out = {
        "attempts": len(records),
        "requested_mean": float(np.mean(requested)),
        "requested_histogram": {int(k): int(v) for k, v in zip(values, counts)},
        "valid_molecules_sized": len(achieved),
        "atom_count_match_among_valid": (matched / len(achieved)) if achieved
        else float("nan"),
        "atom_count_match_over_all_attempts": matched / max(len(records), 1),
        "note": ("atom count is STRUCTURAL in both arms: N sets how many node vectors "
                 "are drawn and the decoder emits one atom per vector, so a match here "
                 "is plumbing in both arms, not evidence of conditioning"),
    }
    if achieved:
        achieved_values, achieved_counts = np.unique(achieved, return_counts=True)
        out.update({"achieved_mean": float(np.mean(achieved)),
                    "achieved_std": float(np.std(achieved)),
                    "achieved_min": int(np.min(achieved)),
                    "achieved_max": int(np.max(achieved)),
                    "achieved_histogram": {int(k): int(v) for k, v
                                           in zip(achieved_values, achieved_counts)}})
    return out


def architecture_diff(unconditional: dict, conditional: dict) -> dict:
    """Every flow-config difference, split into the intended one and the confounds."""
    keys = set(unconditional) | set(conditional)
    differences = {k: {"unconditional": unconditional.get(k),
                       "conditional": conditional.get(k)}
                   for k in sorted(keys)
                   if str(unconditional.get(k)) != str(conditional.get(k))}
    confounds = {k: v for k, v in differences.items() if k not in CONDITIONING_KEYS}
    return {
        "all_differences": differences,
        "conditioning_differences": {k: v for k, v in differences.items()
                                     if k in CONDITIONING_KEYS},
        "remaining_architecture_differences": confounds,
        "clean_conditioning_contrast": not confounds,
        "verdict": ("the two checkpoints differ ONLY in conditioning, so the table "
                    "measures conditioning"
                    if not confounds else
                    f"the two checkpoints also differ in {sorted(confounds)}, so this "
                    f"table measures conditioning PLUS those changes and must be "
                    f"described that way"),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.config:
        import json as _json

        try:
            _cfg = _json.loads(Path(args.config).read_text())
        except (OSError, ValueError) as error:
            print(f"FATAL: cannot read --config {args.config}: {error}", file=sys.stderr)
            return 4
        _paths, _run = _cfg.get("paths", {}), _cfg.get("run", {})
        args.data_dir = args.data_dir or _paths.get("data_dir")
        args.ae_checkpoint = args.ae_checkpoint or _paths.get("ae_checkpoint")
        args.conditional_checkpoint = (args.conditional_checkpoint
                                       or _paths.get("conditional_checkpoint"))
        args.unconditional_checkpoint = (args.unconditional_checkpoint
                                         or _paths.get("baseline_checkpoint"))
        args.output_dir = args.output_dir or (
            str(Path(_paths.get("results_dir", "outputs")) / "uncond_vs_cond"))
        if not args.moflow_protocol:
            args.count = _run.get("comparison_attempts", args.count)
            args.runs = _run.get("comparison_runs", args.runs)
            args.temperature = _run.get("comparison_temperature", args.temperature)
        args.batch_size = _run.get("batch_size", args.batch_size)
        args.headline = _run.get("headline_decoder", args.headline)
        args.seed = _run.get("sampling_seed", args.seed)
    missing = [name for name in ("unconditional_checkpoint", "conditional_checkpoint",
                                 "ae_checkpoint", "output_dir", "data_dir")
               if not getattr(args, name)]
    if missing:
        print(f"FATAL: missing {missing}; supply them as flags or via --config",
              file=sys.stderr)
        return 4
    try:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.moflow_protocol:
            args.count = MOFLOW_SETTINGS["count"]
            args.runs = MOFLOW_SETTINGS["runs"]
            args.temperature = MOFLOW_SETTINGS["temperature"]
            print("MoFlow protocol: 10,000 attempts x 5 runs at temperature 0.85 "
                  "(paper section 5.1).")
            print("This is NOT 'exactly MoFlow'. Differences that remain:")
            for item in MOFLOW_DIFFERENCES:
                print(f"  - {item}")
        if args.count < 1:
            raise ValueError("--count must be >= 1")
        done = out_dir / "uncond_vs_cond.json"
        if done.is_file() and done.stat().st_size > 0 and not args.force:
            print(f"{done} already exists; nothing recomputed. Use --force to redo it.")
            return 0
        if args.runs < 1:
            raise ValueError("--runs must be >= 1")
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 4

    started = time.perf_counter()
    try:
        import torch

        device = resolve_device(args.device)
        for path in (args.unconditional_checkpoint, args.conditional_checkpoint,
                     args.ae_checkpoint):
            if not Path(path).is_file():
                raise FileNotFoundError(f"not found: {path}")

        unconditional, uncond_probe = load_flow(args.unconditional_checkpoint,
                                                device=device)
        conditional, cond_probe = load_flow(args.conditional_checkpoint, device=device)
        if unconditional.config.conditional:
            raise RuntimeError(
                f"--unconditional-checkpoint conditions on "
                f"{list(unconditional.config.condition_features)}; it is not an "
                f"unconditional baseline")
        if set(conditional.config.condition_features) != {"n", "qed"}:
            raise RuntimeError(
                f"--conditional-checkpoint conditions on "
                f"{list(conditional.config.condition_features)}, expected n and qed")

        diff = architecture_diff(uncond_probe["flow_config"], cond_probe["flow_config"])
        # one decoder for both arms, and it must match both flows' embedding width
        autoencoder, ae_probe = load_autoencoder(args.ae_checkpoint, device=device)
        for name, flow in (("unconditional", unconditional), ("conditional", conditional)):
            if autoencoder.config.node_embedding_dim != flow.config.node_embedding_dim:
                raise RuntimeError(
                    f"the {name} flow has node_embedding_dim "
                    f"{flow.config.node_embedding_dim} but the decoder has "
                    f"{autoencoder.config.node_embedding_dim}; one decoder must serve "
                    f"both arms or the comparison changes two things at once")
        autoencoder.eval()
        for parameter in autoencoder.parameters():
            parameter.requires_grad_(False)

        train_records, _, fingerprint = load_training_splits(args.data_dir)
        train_smiles = {str(r["smiles"]) for r in train_records}
        requests = sample_requests(train_records, args.count, args.seed)
        joint = joint_summary(requests["n"], requests["qed"])


        header = {
            "unconditional_checkpoint": {
                "path": str(args.unconditional_checkpoint),
                "sha256": file_sha256(args.unconditional_checkpoint),
                "step": uncond_probe.get("step"),
                "train_config": uncond_probe.get("train_config"),
                "flow_config": uncond_probe["flow_config"]},
            "conditional_checkpoint": {
                "path": str(args.conditional_checkpoint),
                "sha256": file_sha256(args.conditional_checkpoint),
                "step": cond_probe.get("step"),
                "train_config": cond_probe.get("train_config"),
                "flow_config": cond_probe["flow_config"]},
            "decoder": {"path": str(args.ae_checkpoint),
                        "sha256": file_sha256(args.ae_checkpoint),
                        "degree_head": bool(autoencoder.config.degree_head),
                        "node_embedding_dim": autoencoder.config.node_embedding_dim},
            "architecture_diff": diff,
            "shared_settings": {"count_per_run": args.count, "runs": args.runs,
                                "temperature": args.temperature,
                                "headline_decoder": args.headline,
                                "base_seed": args.seed,
                                "seed_kind": ("sampling seeds only; one trained "
                                              "checkpoint per arm, so run-level sd is "
                                              "sampling variability, NOT training-seed "
                                              "variability"),
                                "novelty_reference": "train split canonical SMILES",
                                "train_reference_molecules": len(train_smiles)},
            "request_joint": joint,
            "data_fingerprint": fingerprint,
        }
        print(json.dumps({k: header[k] for k in
                          ("architecture_diff", "request_joint", "shared_settings")},
                         indent=2, default=str))
        atomic_write_json(header, out_dir / "comparison_header.json")
        if args.inspect_only:
            print("\ninspect-only: nothing was generated")
            print(f"clean conditioning contrast: {diff['clean_conditioning_contrast']}")
            print(f"VERDICT: {diff['verdict']}")
            return 0

        if not bool(autoencoder.config.degree_head) and args.headline == "degree":
            raise RuntimeError(
                "--headline degree needs a decoder with a degree head; this autoencoder "
                "has none, so the 'corrected validity' column cannot be produced")

        # The SAME node counts for both arms. Sizes are structural, so matching them
        # removes the only way the two arms could differ in size by construction.
        sizes = requests["n"]
        scaler = load_condition_scaler(args.data_dir)
        condition_values = standardise_condition(
            conditional.config.condition_features, scaler=scaler,
            n=list(int(v) for v in sizes), qed=list(float(v) for v in requests["qed"]))

        arms: dict = {"unconditional": {"runs": []}, "conditional": {"runs": []}}
        for run in range(args.runs):
            # Fresh requests per run, SHARED by both arms, so within a run the two arms
            # answer identical (N, QED) requests with identical sizes.
            if run == 0:
                run_sizes, run_values = sizes, condition_values
            else:
                run_requests = sample_requests(train_records, args.count,
                                               args.seed + run)
                run_sizes = run_requests["n"]
                run_values = standardise_condition(
                    conditional.config.condition_features, scaler=scaler,
                    n=list(int(v) for v in run_sizes),
                    qed=list(float(v) for v in run_requests["qed"]))
            for name, flow, values in (("unconditional", unconditional, None),
                                       ("conditional", conditional, run_values)):
                print(f"\nrun {run + 1}/{args.runs}, arm {name}: "
                      f"{args.count:,} attempts ...")
                # `_FixedCounts` is stateful, so each call gets its own instance
                drawn = generate_molecules(
                    flow, autoencoder, count=args.count,
                    node_counts=FixedCounts(run_sizes),
                    temperature=args.temperature, seed=args.seed + run,
                    batch_size=args.batch_size,
                    condition_values=None if values is None
                    else torch.from_numpy(values),
                    device=device)
                metrics = generation_metrics(drawn["records"],
                                             train_smiles=train_smiles,
                                             headline=args.headline)
                head = metrics["decoders"][args.headline]
                arms[name]["runs"].append({
                    "run": run,
                    "sampling_seed": args.seed + run,
                    "settings": drawn["settings"],
                    "metrics": metrics,
                    "sizes": achieved_sizes(drawn["records"], args.headline),
                    "node_count_agreement": node_count_agreement(drawn["records"],
                                                                 args.headline),
                })
                print(f"  raw validity      "
                      f"{metrics['decoders']['raw']['strict_validity']:.4f}")
                print(f"  corrected ({args.headline}) validity "
                      f"{head['strict_validity']:.4f}")
                print(f"  uniqueness        {head['uniqueness']:.4f}")
                print(f"  novelty           {head['novelty_vs_train']:.4f}")
                print(f"  NUV (of attempts) {head['nuv']:.4f}")

        for name in arms:
            arms[name]["summary"] = aggregate_runs(arms[name]["runs"], args.headline)
        if args.runs > 1:
            print(f"\nrun-level summary over {args.runs} runs "
                  f"(mean +/- sd; SAMPLING variability from ONE checkpoint, "
                  f"not training-seed variability):")
            for name in ("unconditional", "conditional"):
                print(f"  {name}:")
                for key, value in arms[name]["summary"].items():
                    print(f"    {key:28s} {value['mean']:.4f} +/- {value['sd']:.4f}")

        payload = {**header, "arms": arms,
                   "moflow_protocol_differences": MOFLOW_DIFFERENCES
                   if args.moflow_protocol else None,
                   "elapsed_seconds": time.perf_counter() - started}
        atomic_write_json(payload, out_dir / "uncond_vs_cond.json")
        print(f"\nwrote {out_dir / 'uncond_vs_cond.json'}")
        print("\nNUV is the fraction of ALL attempts that are valid, unique and novel: "
              "validity is per attempt, uniqueness per valid, novelty per unique, so "
              "their product has the attempt denominator.")
        if not diff["clean_conditioning_contrast"]:
            print(f"\nCAVEAT: {diff['verdict']}")
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
