#!/usr/bin/env python3
"""Collect a relabeling pool: generated embeddings paired with their ACTUAL decoded QED.

Generates from the FROZEN final flow and the FROZEN autoencoder, decodes with all three
assemblers, measures RDKit QED, and saves the ORIGINAL detached embedding tensors. It
never re-encodes SMILES: re-encoding would produce different embeddings and would be a
different experiment.

NO TRAINING AND NO GRADIENT. Both models are opened in eval() with `requires_grad_(False)`
and every forward/inverse call runs under `no_grad`.

WHAT IS SAVED, AND WHERE
------------------------
  <pool-dir>/pool.pt              the embedding tensors and node counts (LOCAL ONLY)
  <pool-dir>/pool_manifest.json   labels, validity per decoder, splits, provenance
  <pool-dir>/pool_summary.txt     the readable support/selection table

`pool.pt` is model output, not source: `.gitignore` excludes `outputs/relabel_pool/` and
the collector refuses to write anywhere inside the repository's tracked tree unless the
path is under `outputs/`.

LABELS ARE DECODER-INDEXED
--------------------------
The label is QED(RDKit(degree-constrained assembly of argmax(decoder(x)))). Validity
under `raw` and `constrained` is recorded for the same embedding so the evaluation can
test whether any later improvement is intrinsic or specific to the label decoder. The
`retained_fraction` makes the validity-selection bias explicit: only samples that decode
to a scorable molecule under the label decoder can carry a label at all.

SEEDS
-----
The pool uses POOL_NOISE_SEED. The evaluation uses a DIFFERENT seed (EVAL_NOISE_SEED) and
`36_evaluate_relabel.py` refuses to run if the two ever coincide, because evaluating on
the prior draws the fine-tune was trained on would manufacture a response out of
memorisation. The pool is additionally split into train/holdout, stratified within each
(N, request) cell.

Exit codes: 0 ok; 2 missing input; 3 provenance mismatch; 4 bad arguments; 1 unexpected.
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

from zinc_gnf.conditioning import load_condition_scaler, standardise_condition  # noqa: E402
from zinc_gnf.conditioning import condition_embedding_for  # noqa: E402
from zinc_gnf.generation import DECODERS, decode_embeddings  # noqa: E402
from zinc_gnf.relabel import (  # noqa: E402
    HOLDOUT_FRACTION,
    LABEL_DECODER,
    POOL_NOISE_SEED,
    PoolConfig,
    RelabelPool,
    assert_disjoint_seeds,
    assign_splits,
)
from zinc_gnf.trainer_autoencoder import ResumeMismatchError, load_autoencoder  # noqa: E402
from zinc_gnf.trainer_flow import file_sha256, load_flow  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402


def within_allowed_output(path: Path) -> bool:
    """Tensors must land under `outputs/` (git-ignored) or fully outside the repo."""
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(ROOT.resolve())
    except ValueError:
        return True                      # outside the repository entirely: fine
    return relative.parts[:1] == ("outputs",)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--ae-checkpoint", default=None)
    parser.add_argument("--data-dir",
                        default="data/processed/zinc250k_charge_aware_moflow_fixed")
    parser.add_argument("--pool-dir", required=True)
    parser.add_argument("--target-n", default="15,23,30")
    parser.add_argument("--qed-requests", default="0.4,0.6,0.8")
    parser.add_argument("--samples-per-cell", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--label-decoder", default=LABEL_DECODER, choices=DECODERS)
    parser.add_argument("--holdout-fraction", type=float, default=HOLDOUT_FRACTION)
    parser.add_argument("--noise-seed", type=int, default=POOL_NOISE_SEED)
    parser.add_argument("--allow-ae-mismatch", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


@torch.no_grad()
def generate_cell(flow, autoencoder, *, scaler, n, qed, count, batch_size, temperature,
                  device, seed, decoders):
    """One (N, request) cell: embeddings, all three decodings, and the RDKit QED label.

    `decoders[0]` is the LABEL decoder. Validity under the other two is recorded for the
    same embedding so the evaluation can separate intrinsic conditioning from a
    correction-dependent effect.
    """
    from rdkit import Chem, rdBase
    from rdkit.Chem import QED

    rows = []
    produced = 0
    while produced < count:
        size = min(batch_size, count - produced)
        mask = torch.ones((size, n), dtype=torch.bool, device=device)
        generator = torch.Generator(device=device).manual_seed(
            int(seed) + 7919 * n + int(round(1000 * qed)) + produced)
        eps = torch.randn((size, n, flow.config.node_embedding_dim), device=device,
                          generator=generator)
        values = torch.from_numpy(
            standardise_condition(flow.config.condition_features, scaler=scaler,
                                  n=[n] * size, qed=[qed] * size)).to(device)
        condition = condition_embedding_for(flow, values, torch.ones_like(values),
                                            size, device)
        mu, sigma = flow.prior.parameters_for(size, condition, device)
        z = (mu.unsqueeze(1) + float(temperature) * sigma.unsqueeze(1) * eps) \
            * mask.unsqueeze(-1)
        embeddings = flow.inverse(z, mask, condition).detach()
        records = decode_embeddings(autoencoder, embeddings, [n] * size,
                                    decoders=tuple(decoders))
        for index, record in enumerate(records):
            label_entry = record.get(decoders[0]) or {}
            row: dict[str, Any] = {
                "embedding": embeddings[index, :n].detach().cpu().clone(),
                "n_node": n, "requested_qed": float(qed),
                "valid": {d: bool((record.get(d) or {}).get("valid")) for d in decoders},
                "fallback": bool(label_entry.get("fallback", False)),
                "actual_qed": None, "smiles": None,
            }
            # the label is measured with RDKit from the label decoder's own SMILES
            if label_entry.get("valid") and label_entry.get("smiles"):
                with rdBase.BlockLogs():
                    molecule = Chem.MolFromSmiles(str(label_entry["smiles"]))
                if molecule is not None:
                    row["actual_qed"] = float(QED.qed(molecule))
                    row["smiles"] = str(label_entry["smiles"])
            rows.append(row)
        produced += size
    return rows


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        assert_disjoint_seeds()
        targets_n = [int(v) for v in args.target_n.split(",") if v.strip()]
        qed_requests = [float(v) for v in args.qed_requests.split(",") if v.strip()]
        if not targets_n or not qed_requests:
            raise ValueError("need at least one N and one requested QED")
        if args.samples_per_cell < 2 or args.batch_size < 1:
            raise ValueError("samples-per-cell >= 2 and batch-size >= 1")
        if not 0.0 < args.holdout_fraction < 0.5:
            raise ValueError("holdout-fraction must lie in (0, 0.5)")
        pool_dir = Path(args.pool_dir)
        if not within_allowed_output(pool_dir):
            raise ValueError(
                f"{pool_dir} is inside the repository but not under outputs/; tensors "
                f"must never be committed. Use outputs/relabel_pool/... or a path "
                f"outside the repository.")
        pool_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        return 4

    started = time.perf_counter()
    try:
        device = torch.device(args.device)
        scaler = load_condition_scaler(args.data_dir)
        flow, probe = load_flow(args.flow_checkpoint, device=device)
        flow.eval().requires_grad_(False)
        recorded = probe.get("autoencoder") or {}
        ae_path = Path(args.ae_checkpoint or recorded.get("checkpoint", ""))
        if not ae_path.is_file():
            raise FileNotFoundError(f"autoencoder not found: {ae_path}")
        digest = file_sha256(ae_path)
        match = bool(recorded.get("sha256") == digest)
        if not match and not args.allow_ae_mismatch:
            raise ResumeMismatchError(
                f"autoencoder SHA256 mismatch: recorded "
                f"{str(recorded.get('sha256'))[:16]}, actual {digest[:16]}")
        autoencoder, _ = load_autoencoder(ae_path, device=device)
        autoencoder.eval().requires_grad_(False)
        decoders = [args.label_decoder] + [d for d in DECODERS if d != args.label_decoder]
        print(f"flow step={probe.get('step')}  AE provenance match={match}  "
              f"label decoder={args.label_decoder}", flush=True)

        pool = RelabelPool()
        per_cell: dict[str, Any] = {}
        for n in targets_n:
            for qed in qed_requests:
                cell_started = time.perf_counter()
                rows = generate_cell(
                    flow, autoencoder, scaler=scaler, n=n, qed=qed,
                    count=args.samples_per_cell, batch_size=args.batch_size,
                    temperature=args.temperature, device=device,
                    seed=args.noise_seed, decoders=decoders)
                kept = [r for r in rows if r["actual_qed"] is not None]
                for row in kept:
                    pool.embeddings.append(row["embedding"])
                    pool.n_node.append(int(row["n_node"]))
                    pool.requested_qed.append(float(row["requested_qed"]))
                    pool.actual_qed.append(float(row["actual_qed"]))
                    pool.smiles.append(str(row["smiles"]))
                    pool.fallback.append(bool(row["fallback"]))
                    for decoder in decoders:
                        pool.valid.setdefault(decoder, []).append(
                            bool(row["valid"][decoder]))
                labels = np.asarray([r["actual_qed"] for r in kept], dtype=np.float64)
                per_cell[f"N{n}_QED{qed:g}"] = {
                    "generated": len(rows),
                    "labelled": len(kept),
                    "retained_fraction": len(kept) / max(1, len(rows)),
                    "validity": {d: float(np.mean([r["valid"][d] for r in rows]))
                                 for d in decoders},
                    "fallback_rate": float(np.mean([r["fallback"] for r in rows])),
                    "actual_qed_mean": float(labels.mean()) if labels.size else None,
                    "actual_qed_std": float(labels.std()) if labels.size else None,
                    "seconds": time.perf_counter() - cell_started,
                }
                block = per_cell[f"N{n}_QED{qed:g}"]
                print(f"  N={n} q={qed}: {block['labelled']}/{block['generated']} "
                      f"labelled ({block['retained_fraction']:.3f})  "
                      f"actual QED={block['actual_qed_mean']}  "
                      f"{block['seconds']:.1f}s", flush=True)

        if len(pool) < 20:
            raise FileNotFoundError(
                f"only {len(pool)} labelled samples; too few to fine-tune on. Raise "
                f"--samples-per-cell or check decoder validity.")
        pool.split = assign_splits(pool.n_node, pool.requested_qed,
                                   holdout_fraction=args.holdout_fraction)
        pool.metadata = {
            "framing": ("Generated embeddings labelled with their ACTUAL decoded QED "
                        "under the degree-constrained assembler. The label is "
                        "DECODER-INDEXED: it means 'QED under this decoder', not an "
                        "intrinsic property of the embedding."),
            "pool_config": PoolConfig(
                target_n=tuple(targets_n), qed_requests=tuple(qed_requests),
                samples_per_cell=args.samples_per_cell, temperature=args.temperature,
                noise_seed=args.noise_seed, label_decoder=args.label_decoder).to_dict(),
            "flow": {"checkpoint": str(args.flow_checkpoint),
                     "step": probe.get("step"),
                     "sha256": file_sha256(args.flow_checkpoint)},
            "autoencoder": {"checkpoint": str(ae_path), "sha256_actual": digest,
                            "sha256_recorded_in_flow": recorded.get("sha256"),
                            "sha256_match": match,
                            "mismatch_override_used": bool(not match)},
            "decoders_recorded": decoders,
            "holdout_fraction": args.holdout_fraction,
            "per_cell": per_cell,
            "train_samples": sum(1 for s in pool.split if s == "train"),
            "holdout_samples": sum(1 for s in pool.split if s == "holdout"),
            "selection_bias_note": (
                "Only samples that decode to a scorable molecule under the label "
                "decoder carry a label, so the pool is a validity-selected subsample of "
                "the model's output. retained_fraction per cell quantifies it."),
        }
        manifest = pool.save(pool_dir)
        (pool_dir / "pool_summary.txt").write_text(format_pool(manifest))
        print(f"\n{len(pool)} labelled samples "
              f"({pool.metadata['train_samples']} train / "
              f"{pool.metadata['holdout_samples']} holdout)")
        print(f"QED support: {json.dumps(pool.qed_support()['quantiles'])}")
        print(f"elapsed {time.perf_counter() - started:.1f}s")
        print(f"wrote {pool_dir / 'pool.pt'} (LOCAL ONLY, never committed)")
        print(f"wrote {pool_dir / 'pool_manifest.json'}")
        print(f"wrote {pool_dir / 'pool_summary.txt'}")
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


def format_pool(manifest: dict[str, Any]) -> str:
    metadata = manifest["metadata"]
    support = manifest["qed_support"]
    lines = ["Relabeling pool", "=" * 78, "", metadata["framing"], "",
             f"flow: {metadata['flow']['checkpoint']} (step {metadata['flow']['step']})",
             f"AE provenance match: {metadata['autoencoder']['sha256_match']}",
             f"label decoder: {metadata['pool_config']['label_decoder']}",
             f"noise seed: {metadata['pool_config']['noise_seed']}",
             f"samples: {manifest['samples']} "
             f"({metadata['train_samples']} train / {metadata['holdout_samples']} holdout)",
             "", "-- ACTUAL decoded QED support (the relabeling ceiling) " + "-" * 23,
             f"  mean={support.get('mean')}  std={support.get('std')}  "
             f"IQR={support.get('iqr')}",
             f"  p5={support.get('quantiles', {}).get('p5')} "
             f"p25={support.get('quantiles', {}).get('p25')} "
             f"p50={support.get('quantiles', {}).get('p50')} "
             f"p75={support.get('quantiles', {}).get('p75')} "
             f"p95={support.get('quantiles', {}).get('p95')}",
             f"  fraction >= 0.75: {support.get('fraction_at_least_0p75')}",
             "", "-- per cell " + "-" * 65]
    for key, block in metadata["per_cell"].items():
        lines.append(f"  {key:<16} labelled {block['labelled']:>4}/{block['generated']:<4} "
                     f"({block['retained_fraction']:.3f})  "
                     f"QED={block['actual_qed_mean']}  "
                     f"fallback={block['fallback_rate']:.3f}")
        lines.append(f"  {'':<16} validity " + "  ".join(
            f"{d}={v:.3f}" for d, v in block["validity"].items()))
    lines += ["", metadata["selection_bias_note"], "",
              "The pool is the CEILING for relabeling: the fine-tune can only learn to",
              "address QED values that already appear above. It cannot invent new",
              "chemistry, and the label means 'QED under the label decoder' only."]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
