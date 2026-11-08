#!/usr/bin/env python3
"""Three-way evaluation: fine-tuned vs final vs early, under COMMON noise. READ-ONLY.

Loads no optimizer, writes no checkpoint, trains nothing. Every model is opened in eval()
with gradients disabled and `flow.inverse` runs under `no_grad`.

THE PROTOCOL
------------
For each N one `eps` tensor is drawn with EVAL_NOISE_SEED and reused for EVERY model and
EVERY requested QED. That gives two kinds of pairing at once:

  * ACROSS REQUESTS, within a model: molecule i at QED 0.4 and molecule i at QED 0.8 come
    from the same prior draw, so `delta_i = QED(x_i | 0.8) - QED(x_i | 0.4)` is a genuine
    per-molecule paired response -- the "intrinsic paired QED response at fixed N" that a
    difference of cell means cannot give. Only JOINTLY VALID pairs are used; scoring a
    pair where one side failed to decode produced a 25-percentage-point bias in an
    earlier version of this project.
  * ACROSS MODELS, within a request: the same draw is scored under all three
    checkpoints, so differences are not sampling noise between runs.

LEAKAGE GUARD
-------------
EVAL_NOISE_SEED must differ from the seed that collected the relabeling pool. If the
pool manifest records the same seed, this script EXITS 3 rather than reporting a response
that could be memorisation of its own training draws.

THE LABEL-DECODER QUESTION
--------------------------
The fine-tune's labels are QED under the degree-constrained assembler. So the response is
measured under `raw`, `constrained` AND `degree` separately. If it appears only under
`degree`, the model learned a correction-dependent distribution rather than intrinsic
conditioning, and the report says so in those words. That distinction is the single most
important output of this script.

THE SUCCESS CRITERION, FIXED IN CODE
------------------------------------
A lower QED MAE is NOT sufficient and is deliberately excluded from the pass condition: a
model can lower MAE by collapsing toward common QED values. To pass, the fine-tune must
improve the PAIRED response at fixed N with a bootstrap CI excluding zero, AND hold raw
validity, aromatic rings, uniqueness and internal diversity within precommitted margins.

Exit codes: 0 ok; 2 missing input; 3 provenance or leakage refusal; 4 bad arguments;
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from zinc_gnf.conditioning import (  # noqa: E402
    condition_embedding_for,
    load_condition_scaler,
    standardise_condition,
)
from zinc_gnf.dataset import MolecularGraphDataset, load_training_splits  # noqa: E402
from zinc_gnf.generation import DECODERS, decode_embeddings  # noqa: E402
from zinc_gnf.relabel import EVAL_NOISE_SEED, real_validation_nll  # noqa: E402
from zinc_gnf.trainer_autoencoder import ResumeMismatchError, load_autoencoder  # noqa: E402
from zinc_gnf.trainer_flow import file_sha256, load_flow  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402

BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261114
# Precommitted pass margins. Lower MAE is reported but is NOT in the pass condition.
MAX_RAW_VALIDITY_DROP = 0.03          # absolute, percentage points / 100
MIN_AROMATIC_RETENTION = 0.90
MIN_UNIQUENESS_RETENTION = 0.90
MIN_DIVERSITY_RETENTION = 0.95


def bootstrap_ci(values, *, seed: int = BOOTSTRAP_SEED):
    array = np.asarray([v for v in np.ravel(values)
                        if v is not None and np.isfinite(v)], dtype=np.float64)
    if array.size < 2:
        return {"n": int(array.size),
                "mean": float(array[0]) if array.size else None,
                "ci95": [None, None], "excludes_zero": False}
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, array.size, size=(BOOTSTRAP, array.size))].mean(axis=1)
    low, high = float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
    return {"n": int(array.size), "mean": float(array.mean()), "ci95": [low, high],
            "excludes_zero": bool(low > 0.0 or high < 0.0)}


def internal_diversity_of(smiles) -> float | None:
    unique = sorted({s for s in smiles if s})
    if len(unique) < 2:
        return None
    from zinc_gnf.metrics_generation import internal_diversity
    return float(internal_diversity(unique)["internal_diversity"])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True,
                        help="label=path, repeatable; e.g. final=.../best.pt")
    parser.add_argument("--reference", default="final",
                        help="the label the fine-tuned model is compared against")
    parser.add_argument("--candidate", default="finetuned")
    parser.add_argument("--ae-checkpoint", required=True)
    parser.add_argument("--pool-dir", default=None,
                        help="checked for a noise-seed collision with the evaluation")
    parser.add_argument("--data-dir",
                        default="data/processed/zinc250k_charge_aware_moflow_fixed")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-n", default="15,23,30")
    parser.add_argument("--qed-values", default="0.4,0.6,0.8")
    parser.add_argument("--samples-per-cell", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--headline", default="degree", choices=DECODERS)
    parser.add_argument("--val-molecules", type=int, default=2000)
    parser.add_argument("--eval-seed", type=int, default=EVAL_NOISE_SEED)
    parser.add_argument("--skip-real-nll", action="store_true")
    parser.add_argument("--allow-ae-mismatch", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


@torch.no_grad()
def generate_and_score(flow, autoencoder, *, scaler, n, qed, eps, batch_size,
                       temperature, device, decoders):
    """Decode the SAME prior draws under one model and one request. Returns per-sample rows."""
    from rdkit import Chem, rdBase
    from rdkit.Chem import QED, rdMolDescriptors

    rows: list[dict[str, Any]] = []
    total = eps.shape[0]
    for start in range(0, total, batch_size):
        chunk = eps[start:start + batch_size]
        size = chunk.shape[0]
        mask = torch.ones((size, n), dtype=torch.bool, device=device)
        values = torch.from_numpy(
            standardise_condition(flow.config.condition_features, scaler=scaler,
                                  n=[n] * size, qed=[qed] * size)).to(device)
        condition = condition_embedding_for(flow, values, torch.ones_like(values),
                                            size, device)
        mu, sigma = flow.prior.parameters_for(size, condition, device)
        z = (mu.unsqueeze(1) + float(temperature) * sigma.unsqueeze(1) * chunk) \
            * mask.unsqueeze(-1)
        embeddings = flow.inverse(z, mask, condition)
        records = decode_embeddings(autoencoder, embeddings, [n] * size,
                                    decoders=tuple(decoders))
        for record in records:
            row: dict[str, Any] = {}
            for decoder in decoders:
                entry = record.get(decoder) or {}
                scored: dict[str, Any] = {
                    "valid": bool(entry.get("valid")),
                    "fallback": bool(entry.get("fallback", False)),
                    "smiles": entry.get("smiles"), "qed": None,
                    "aromatic_rings": None, "heavy_atoms": None,
                }
                if entry.get("valid") and entry.get("smiles"):
                    with rdBase.BlockLogs():
                        molecule = Chem.MolFromSmiles(str(entry["smiles"]))
                    if molecule is not None:
                        scored["qed"] = float(QED.qed(molecule))
                        scored["aromatic_rings"] = float(
                            rdMolDescriptors.CalcNumAromaticRings(molecule))
                        scored["heavy_atoms"] = float(molecule.GetNumHeavyAtoms())
                row[decoder] = scored
            rows.append(row)
    return rows


def cell_summary(rows, decoder, *, target) -> dict[str, Any]:
    entries = [r[decoder] for r in rows]
    scored = [e for e in entries if e["qed"] is not None]
    smiles = [e["smiles"] for e in scored if e["smiles"]]
    unique = sorted(set(smiles))
    return {
        "samples": len(entries),
        "validity": sum(1 for e in entries if e["valid"]) / max(1, len(entries)),
        "scored": len(scored),
        "fallback_rate": sum(1 for e in entries if e["fallback"]) / max(1, len(entries)),
        "qed_mean": float(np.mean([e["qed"] for e in scored])) if scored else None,
        "qed_std": float(np.std([e["qed"] for e in scored])) if scored else None,
        "qed_mae_vs_target": (float(np.mean([abs(e["qed"] - target) for e in scored]))
                              if scored else None),
        "aromatic_rings_mean": (float(np.mean([e["aromatic_rings"] for e in scored]))
                                if scored else None),
        "heavy_atoms_mean": (float(np.mean([e["heavy_atoms"] for e in scored]))
                             if scored else None),
        "unique_molecules": len(unique),
        "uniqueness": (len(unique) / len(scored)) if scored else None,
        "internal_diversity": internal_diversity_of(unique),
    }


def paired_response(low_rows, high_rows, decoder) -> dict[str, Any]:
    """Per-molecule QED response on the SAME prior draw. Jointly valid pairs only."""
    deltas = []
    for low, high in zip(low_rows, high_rows):
        a, b = low[decoder], high[decoder]
        if a["qed"] is not None and b["qed"] is not None:
            deltas.append(b["qed"] - a["qed"])
    result = bootstrap_ci(deltas)
    result["jointly_valid_pairs"] = len(deltas)
    result["pairing"] = ("per-molecule on a common prior draw, jointly valid pairs only; "
                         "unpaired scoring biases this statistic")
    return result


def verdict(payload, *, candidate: str, reference: str, decoder: str) -> dict[str, Any]:
    """Mechanical pass/fail. Lower MAE is reported but is NOT part of the condition."""
    models = payload["models"]
    if candidate not in models or reference not in models:
        return {"evaluated": False,
                "note": f"need both {candidate!r} and {reference!r} to compare"}
    checks: dict[str, Any] = {"by_n": {}, "thresholds": {
        "max_raw_validity_drop": MAX_RAW_VALIDITY_DROP,
        "min_aromatic_retention": MIN_AROMATIC_RETENTION,
        "min_uniqueness_retention": MIN_UNIQUENESS_RETENTION,
        "min_diversity_retention": MIN_DIVERSITY_RETENTION,
    }}
    passes = []
    for n in models[reference]["by_n"]:
        new = models[candidate]["by_n"].get(n) or {}
        old = models[reference]["by_n"].get(n) or {}
        new_response = (new.get("paired_response") or {}).get(decoder) or {}
        old_response = (old.get("paired_response") or {}).get(decoder) or {}
        entry: dict[str, Any] = {
            "paired_response_candidate": new_response.get("mean"),
            "paired_response_reference": old_response.get("mean"),
            "candidate_ci_excludes_zero": bool(new_response.get("excludes_zero")),
            "response_improved": bool(
                new_response.get("mean") is not None
                and old_response.get("mean") is not None
                and new_response["mean"] > old_response["mean"]
                and new_response.get("excludes_zero")),
        }
        quality = {}
        for metric, minimum in (("aromatic_rings_mean", MIN_AROMATIC_RETENTION),
                                ("uniqueness", MIN_UNIQUENESS_RETENTION),
                                ("internal_diversity", MIN_DIVERSITY_RETENTION)):
            ratios = []
            for target in new.get("cells", {}):
                after = ((new["cells"][target].get(decoder) or {}).get(metric))
                before = ((old.get("cells", {}).get(target, {}).get(decoder) or {})
                          .get(metric))
                if after and before:
                    ratios.append(after / before)
            quality[metric] = {"retention": float(np.mean(ratios)) if ratios else None,
                               "holds": bool(ratios and min(ratios) >= minimum)}
        drops = []
        for target in new.get("cells", {}):
            after = (new["cells"][target].get("raw") or {}).get("validity")
            before = (old.get("cells", {}).get(target, {}).get("raw") or {}).get("validity")
            if after is not None and before is not None:
                drops.append(before - after)
        entry["raw_validity_worst_drop"] = float(max(drops)) if drops else None
        entry["raw_validity_holds"] = bool(drops and max(drops) <= MAX_RAW_VALIDITY_DROP)
        entry["chemical_quality"] = quality
        entry["passes"] = bool(entry["response_improved"] and entry["raw_validity_holds"]
                               and all(q["holds"] for q in quality.values()))
        passes.append(entry["passes"])
        checks["by_n"][n] = entry

    # label-decoder specificity: does the gain survive outside the label decoder?
    specificity: dict[str, Any] = {}
    for other in DECODERS:
        improved = []
        for n in models[reference]["by_n"]:
            new_response = ((models[candidate]["by_n"][n].get("paired_response") or {})
                            .get(other) or {})
            old_response = ((models[reference]["by_n"][n].get("paired_response") or {})
                            .get(other) or {})
            if new_response.get("mean") is not None and old_response.get("mean") is not None:
                improved.append(new_response["mean"] - old_response["mean"])
        specificity[other] = {"mean_response_change": float(np.mean(improved))
                              if improved else None,
                              "per_n": improved}
    checks["label_decoder_specificity"] = specificity
    label_gain = (specificity.get(decoder) or {}).get("mean_response_change")
    others = [v["mean_response_change"] for k, v in specificity.items()
              if k != decoder and v["mean_response_change"] is not None]
    checks["intrinsic_rather_than_correction_dependent"] = bool(
        label_gain is not None and others and label_gain > 0
        and all(value > 0.5 * label_gain for value in others))
    checks["verdict"] = ("GO" if passes and all(passes)
                         else "PARTIAL" if any(passes) else "NO-GO")
    checks["interpretation"] = (
        "The pass condition is an IMPROVED per-molecule paired QED response at fixed N "
        "with a bootstrap CI excluding zero, together with raw validity, aromaticity, "
        "uniqueness and internal diversity inside the precommitted margins. A lower QED "
        "MAE is reported but is deliberately NOT part of the condition, because MAE can "
        "be lowered by collapsing toward common QED values. "
        "`intrinsic_rather_than_correction_dependent` is the second gate: the labels are "
        "QED under the label decoder, so a response that appears only there is a "
        "correction-dependent distribution, not intrinsic conditioning.")
    return checks


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        targets_n = [int(v) for v in args.target_n.split(",") if v.strip()]
        qed_values = [float(v) for v in args.qed_values.split(",") if v.strip()]
        models: dict[str, Path] = {}
        for item in args.model:
            if "=" not in item:
                raise ValueError(f"--model must be label=path, got {item!r}")
            label, path = item.split("=", 1)
            if not label.strip() or not path.strip():
                raise ValueError(f"--model must be label=path, got {item!r}")
            models[label.strip()] = Path(path.strip())
        if len(qed_values) < 2:
            raise ValueError("need at least two requested QED values for a response")
        if not targets_n or not models:
            raise ValueError("need at least one N and one model")
        if args.samples_per_cell < 2 or args.batch_size < 1:
            raise ValueError("samples-per-cell >= 2 and batch-size >= 1")
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4

    started = time.perf_counter()
    try:
        if args.pool_dir:
            manifest_path = Path(args.pool_dir) / "pool_manifest.json"
            if manifest_path.is_file():
                pool_seed = (json.loads(manifest_path.read_text())
                             .get("metadata", {}).get("pool_config", {})
                             .get("noise_seed"))
                if pool_seed is not None and int(pool_seed) == int(args.eval_seed):
                    raise ResumeMismatchError(
                        f"the evaluation noise seed ({args.eval_seed}) equals the seed "
                        f"that collected the relabeling pool. Evaluating on the model's "
                        f"own training draws would manufacture a response out of "
                        f"memorisation. Use a different --eval-seed.")

        device = torch.device(args.device)
        scaler = load_condition_scaler(args.data_dir)
        ae_path = Path(args.ae_checkpoint)
        if not ae_path.is_file():
            raise FileNotFoundError(f"autoencoder not found: {ae_path}")
        digest = file_sha256(ae_path)
        autoencoder, _ = load_autoencoder(ae_path, device=device)
        autoencoder.eval().requires_grad_(False)
        decoders = [args.headline] + [d for d in DECODERS if d != args.headline]

        loaded: dict[str, Any] = {}
        for label, path in models.items():
            if not path.is_file():
                raise FileNotFoundError(f"{label}: checkpoint not found: {path}")
            flow, probe = load_flow(path, device=device)
            flow.eval().requires_grad_(False)
            recorded = probe.get("autoencoder") or {}
            match = bool(recorded.get("sha256") == digest)
            if not match and not args.allow_ae_mismatch:
                raise ResumeMismatchError(
                    f"{label}: autoencoder SHA256 mismatch (recorded "
                    f"{str(recorded.get('sha256'))[:16]}, actual {digest[:16]})")
            loaded[label] = {"flow": flow, "probe": probe, "path": path,
                             "ae_match": match}
            print(f"loaded {label}: step={probe.get('step')} "
                  f"dim={flow.config.node_embedding_dim} AE match={match}", flush=True)

        dimensions = {m["flow"].config.node_embedding_dim for m in loaded.values()}
        common_noise = len(dimensions) == 1
        if not common_noise:
            print(f"WARNING: embedding dimensions differ {sorted(dimensions)}; noise "
                  f"cannot be shared across models and cross-model pairing is void",
                  flush=True)

        val_set = None
        if not args.skip_real_nll:
            _, val_records, _ = load_training_splits(args.data_dir)
            val_set = MolecularGraphDataset(val_records)
            del val_records

        payload: dict[str, Any] = {
            "framing": ("Three-way comparison under COMMON prior noise. Paired response "
                        "is per-molecule at fixed N on jointly valid pairs. The "
                        "fine-tune's labels are QED under the "
                        f"{args.headline} decoder, so the per-decoder breakdown decides "
                        "whether any gain is intrinsic or correction-dependent."),
            "settings": vars(args) | {"common_noise_across_models": common_noise},
            "autoencoder": {"checkpoint": str(ae_path), "sha256": digest},
            "models": {},
        }

        for label, bundle in loaded.items():
            flow = bundle["flow"]
            block: dict[str, Any] = {
                "checkpoint": str(bundle["path"]),
                "step": bundle["probe"].get("step"),
                "sha256": file_sha256(bundle["path"]),
                "ae_provenance_match": bundle["ae_match"],
                "relabel_finetune": bundle["probe"].get("relabel_finetune"),
                "by_n": {},
            }
            if val_set is not None:
                block["real_validation"] = real_validation_nll(
                    flow, autoencoder, val_set, scaler=scaler, device=device,
                    val_molecules=args.val_molecules)
                print(f"  {label}: real val NLL/node="
                      f"{block['real_validation']['val_loss']:.4f} "
                      f"gap={block['real_validation'].get('val_condition_gap')}",
                      flush=True)
            for n in targets_n:
                # ONE noise draw per (N), shared across every request and every model
                generator = torch.Generator(device=device).manual_seed(
                    int(args.eval_seed) + 7919 * n)
                eps = torch.randn((args.samples_per_cell, n,
                                   flow.config.node_embedding_dim),
                                  device=device, generator=generator)
                cells: dict[str, Any] = {}
                raw_rows: dict[str, Any] = {}
                for qed in qed_values:
                    rows = generate_and_score(
                        flow, autoencoder, scaler=scaler, n=n, qed=qed, eps=eps,
                        batch_size=args.batch_size, temperature=args.temperature,
                        device=device, decoders=decoders)
                    raw_rows[f"{qed:.2f}"] = rows
                    cells[f"{qed:.2f}"] = {d: cell_summary(rows, d, target=qed)
                                           for d in decoders}
                    head = cells[f"{qed:.2f}"][args.headline]
                    print(f"  {label} N={n} q={qed}: QED={head['qed_mean']} "
                          f"MAE={head['qed_mae_vs_target']} "
                          f"validity={head['validity']:.3f} "
                          f"raw={cells[f'{qed:.2f}']['raw']['validity']:.3f} "
                          f"arom={head['aromatic_rings_mean']} "
                          f"uniq={head['uniqueness']}", flush=True)
                low, high = f"{min(qed_values):.2f}", f"{max(qed_values):.2f}"
                block["by_n"][str(n)] = {
                    "cells": cells,
                    "paired_response": {
                        d: paired_response(raw_rows[low], raw_rows[high], d)
                        for d in decoders},
                    "noise_seed": int(args.eval_seed) + 7919 * n,
                }
                response = block["by_n"][str(n)]["paired_response"][args.headline]
                print(f"  {label} N={n}: PAIRED dQED({low}->{high})="
                      f"{response.get('mean')} CI={response.get('ci95')} "
                      f"pairs={response['jointly_valid_pairs']}", flush=True)
            payload["models"][label] = block

        payload["verdict"] = verdict(payload, candidate=args.candidate,
                                     reference=args.reference, decoder=args.headline)
        payload["elapsed_seconds"] = time.perf_counter() - started
        atomic_write_json(payload, out_dir / "relabel_evaluation.json")
        (out_dir / "relabel_evaluation.txt").write_text(
            format_evaluation(payload, decoders=decoders, headline=args.headline))
        print(f"\nverdict: {payload['verdict'].get('verdict')}")
        print(f"wrote {out_dir / 'relabel_evaluation.json'}")
        print(f"wrote {out_dir / 'relabel_evaluation.txt'}")
        return 0
    except FileNotFoundError as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 2
    except ResumeMismatchError as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 3
    except Exception:  # noqa: BLE001
        print("FATAL: unexpected\n" + traceback.format_exc(), file=sys.stderr, flush=True)
        return 1


def _fmt(value, spec=".4f"):
    return format(value, spec) if isinstance(value, (int, float)) else str(value)


def format_evaluation(payload, *, decoders, headline) -> str:
    lines = ["Relabeling evaluation: fine-tuned vs final vs early", "=" * 78, "",
             payload["framing"], "",
             f"common noise across models: "
             f"{payload['settings']['common_noise_across_models']}",
             f"evaluation noise seed: {payload['settings']['eval_seed']}", ""]
    for label, block in payload["models"].items():
        lines += [f"### {label}  step {block['step']}  ({block['checkpoint']})"]
        if block.get("relabel_finetune"):
            finetune = block["relabel_finetune"]
            lines.append(f"    fine-tuned from step {finetune.get('parent_step')} on "
                         f"{finetune.get('pool_samples')} relabeled samples, "
                         f"label decoder {finetune.get('label_decoder')}, "
                         f"weight {finetune.get('relabel_weight')}")
        if block.get("real_validation"):
            validation = block["real_validation"]
            lines.append(f"    real val NLL/node={_fmt(validation['val_loss'])}  "
                         f"condition gap={_fmt(validation.get('val_condition_gap'))}  "
                         f"win rate={_fmt(validation.get('val_condition_win_rate'))}")
        for n, entry in block["by_n"].items():
            lines.append(f"  N={n}")
            for target, cell in entry["cells"].items():
                head = cell[headline]
                lines.append(
                    f"    q={target} [{headline}] QED={_fmt(head['qed_mean'])} "
                    f"MAE={_fmt(head['qed_mae_vs_target'])} "
                    f"validity={_fmt(head['validity'], '.3f')} "
                    f"arom={_fmt(head['aromatic_rings_mean'])} "
                    f"uniq={_fmt(head['uniqueness'])} "
                    f"intdiv={_fmt(head['internal_diversity'])} "
                    f"atoms={_fmt(head['heavy_atoms_mean'], '.1f')}")
                lines.append("    " + " " * 8 + "validity by decoder: " + "  ".join(
                    f"{d}={_fmt(cell[d]['validity'], '.3f')}" for d in decoders))
            lines.append("    PAIRED response (same prior draw, jointly valid pairs):")
            for d in decoders:
                response = entry["paired_response"][d]
                lines.append(
                    f"      {d:<12} dQED={_fmt(response.get('mean'))} "
                    f"CI={response.get('ci95')} "
                    f"pairs={response['jointly_valid_pairs']} "
                    f"excludes_zero={response.get('excludes_zero')}")
        lines.append("")
    checks = payload["verdict"]
    lines += ["-- VERDICT " + "-" * 66]
    if not checks.get("evaluated", True):
        lines.append(f"  {checks.get('note')}")
    else:
        for n, entry in checks["by_n"].items():
            lines += [
                f"  N={n}: paired response "
                f"{_fmt(entry['paired_response_reference'])} -> "
                f"{_fmt(entry['paired_response_candidate'])}  "
                f"improved={entry['response_improved']}  "
                f"CI excludes zero={entry['candidate_ci_excludes_zero']}",
                f"       raw validity worst drop="
                f"{_fmt(entry['raw_validity_worst_drop'])} "
                f"holds={entry['raw_validity_holds']}",
                f"       quality retention " + "  ".join(
                    f"{k}={_fmt(v['retention'])}({v['holds']})"
                    for k, v in entry["chemical_quality"].items()),
                f"       -> passes={entry['passes']}"]
        lines.append("  label-decoder specificity (response change by decoder):")
        for decoder, entry in checks["label_decoder_specificity"].items():
            lines.append(f"    {decoder:<12} {_fmt(entry['mean_response_change'])}")
        lines += [f"  intrinsic rather than correction-dependent: "
                  f"{checks['intrinsic_rather_than_correction_dependent']}",
                  f"  VERDICT: {checks['verdict']}", "", checks["interpretation"]]
    lines += ["", "If the verdict is NO-GO, stop experimenting and consolidate. A lower",
              "MAE on its own does not count, and a response visible only under the",
              "label decoder is a correction-dependent distribution."]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
