#!/usr/bin/env python3
"""Numerical check for both flows: forward/inverse consistency, finiteness, reproducibility.

Small and cheap. It does NOT retrain, does not sweep, and generates only a few hundred
molecules' worth of latents. Run it before any expensive generation.

Three measurements, each reported as ABSOLUTE and RELATIVE error with quantiles rather
than a pass/fail verdict:

  encode->forward->inverse   real molecules encoded by the frozen decoder, pushed through
                             the flow and back. This is the error that matters for the
                             density being well defined.
  prior->inverse->forward    latents drawn from the prior, decoded to embeddings and
                             re-encoded by the flow. This is the error that matters for
                             generation.
  log-determinant            |log|det J||/node, reported next to the errors because a
                             large value is a WARNING SIGN, not proof of failure; the
                             round-trip errors are the evidence.

Reproducibility: the prior->inverse pass is repeated twice with the same seed in the same
process, and once after reloading the checkpoint, so a difference isolates RNG/model-load
ordering from kernel nondeterminism. Decoder determinism is checked separately by
decoding the same embeddings twice.

Exit codes: 0 ran; 2 missing input; 4 bad arguments; 1 unexpected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/cluster.json")
    parser.add_argument("--molecules", type=int, default=256,
                        help="real molecules for the encode->forward->inverse test")
    parser.add_argument("--latents", type=int, default=256,
                        help="prior draws for the generation-direction test")
    parser.add_argument("--target-n", type=int, default=23)
    parser.add_argument("--target-qed", type=float, default=0.6)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def error_statistics(difference, reference, label):
    """Absolute and relative error quantiles. No threshold is applied here."""
    import torch

    absolute = difference.abs().reshape(-1).double()
    scale = reference.abs().reshape(-1).double().clamp(min=1e-12)
    relative = absolute / scale
    quantiles = torch.tensor([0.5, 0.9, 0.99, 1.0], dtype=torch.float64)

    def summarise(values):
        if values.numel() == 0:
            return {}
        picked = torch.quantile(values, quantiles.to(values.device))
        return {"mean": float(values.mean()), "median": float(picked[0]),
                "q90": float(picked[1]), "q99": float(picked[2]),
                "max": float(picked[3])}

    return {"quantity": label, "elements": int(absolute.numel()),
            "absolute": summarise(absolute), "relative": summarise(relative),
            "nonfinite_elements": int((~torch.isfinite(difference)).sum()),
            "note": ("relative error uses |reference| floored at 1e-12, so near-zero "
                     "coordinates inflate it; read absolute and relative together")}


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        config = json.loads(Path(args.config).read_text())
        paths = config["paths"]
        out = Path(paths["results_dir"]) / "numerical"
        out.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError, KeyError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 4

    try:
        import torch

        from zinc_gnf.conditioning import load_condition_scaler, standardise_condition
        from zinc_gnf.dataset import (
            MolecularGraphDataset,
            iter_eval_batches,
            load_training_splits,
        )
        from zinc_gnf.flow import node_mask_from_counts
        from zinc_gnf.trainer_autoencoder import load_autoencoder, resolve_device
        from zinc_gnf.trainer_flow import encode_batch, file_sha256, load_flow

        device = resolve_device(args.device)
        torch.manual_seed(args.seed)
        scaler = load_condition_scaler(paths["data_dir"])
        train_records, _, _ = load_training_splits(paths["data_dir"])
        dataset = MolecularGraphDataset(train_records)
        autoencoder, _ = load_autoencoder(paths["ae_checkpoint"], device=device)
        autoencoder.eval()
        for parameter in autoencoder.parameters():
            parameter.requires_grad_(False)

        candidates = {"baseline": paths.get("baseline_checkpoint"),
                      "conditional": paths.get("conditional_checkpoint")}
        report = {"settings": {k: v for k, v in vars(args).items()},
                  "decoder_sha256": file_sha256(paths["ae_checkpoint"]),
                  "models": {}}

        for label, path in candidates.items():
            if not path or not Path(path).is_file():
                report["models"][label] = {"status": f"missing checkpoint: {path!r}"}
                continue
            flow, probe = load_flow(path, device=device)
            flow.eval()
            features = tuple(flow.config.condition_features)
            block: dict = {"checkpoint": str(path), "sha256": file_sha256(path),
                           "condition_features": list(features),
                           "training_seed": (probe.get("train_config") or {}).get("seed"),
                           "step": probe.get("step")}

            # ---- 1. real molecules: encode -> forward -> inverse ----
            indices = [i for i in range(len(dataset))][:args.molecules]
            with torch.no_grad():
                batch = next(iter(iter_eval_batches(dataset, args.molecules, indices)))
                batch = batch.to(device)
                embeddings = encode_batch(autoencoder, batch)
                condition = None
                if features:
                    values = standardise_condition(
                        features, scaler=scaler,
                        n=[int(v) for v in batch.n_node.tolist()],
                        qed=([float(v) for v in batch.qed.tolist()]
                             if "qed" in features else None))
                    condition = flow.conditioning(
                        torch.from_numpy(values).to(device),
                        torch.ones_like(torch.from_numpy(values)).to(device),
                        batch.batch_size, device)
                latent, log_det = flow.forward(embeddings, batch.node_mask, condition)
                recovered = flow.inverse(latent, batch.node_mask, condition)
                mask = batch.node_mask.unsqueeze(-1).to(embeddings.dtype)
                block["real_molecules_round_trip"] = error_statistics(
                    (recovered - embeddings) * mask, embeddings * mask,
                    "encode -> forward -> inverse (embedding space)")
                nodes = batch.node_mask.sum().clamp(min=1)
                block["log_det_per_node"] = {
                    "mean": float((log_det.sum() / nodes)),
                    "abs_max_per_graph": float(log_det.abs().max()),
                    "interpretation": ("a large magnitude is a WARNING SIGN about "
                                       "float32 inversion, not proof of failure; the "
                                       "round-trip errors above are the evidence")}
                block["latent_finite"] = bool(torch.isfinite(latent).all())
                block["latent_abs_max"] = float(latent.abs().max())

            # ---- 2. generated latents: prior -> inverse -> forward ----
            def generation_direction(seed):
                with torch.no_grad():
                    mask = node_mask_from_counts([args.target_n] * args.latents,
                                                 device=device)
                    condition_values = None
                    if features:
                        values = standardise_condition(
                            features, scaler=scaler,
                            n=[args.target_n] * args.latents,
                            qed=([args.target_qed] * args.latents
                                 if "qed" in features else None))
                        condition_values = flow.conditioning(
                            torch.from_numpy(values).to(device),
                            torch.ones_like(torch.from_numpy(values)).to(device),
                            args.latents, device)
                    generator = torch.Generator(device=device).manual_seed(int(seed))
                    drawn = flow.prior.sample(mask, condition_values,
                                              temperature=args.temperature,
                                              generator=generator)
                    embedded = flow.inverse(drawn, mask, condition_values)
                    back, _ = flow.forward(embedded, mask, condition_values)
                    weights = mask.unsqueeze(-1).to(drawn.dtype)
                    return drawn, embedded, back, weights

            drawn, embedded, back, weights = generation_direction(args.seed)
            block["generated_latents_round_trip"] = error_statistics(
                (back - drawn) * weights, drawn * weights,
                "prior -> inverse -> forward (latent space)")
            block["generated_embeddings_finite"] = bool(torch.isfinite(embedded).all())
            block["generated_embeddings_abs_max"] = float(embedded.abs().max())

            # ---- 3. reproducibility ----
            drawn_again, embedded_again, _, _ = generation_direction(args.seed)
            block["repeat_same_process"] = {
                "latents_bitwise_identical": bool(torch.equal(drawn, drawn_again)),
                "embeddings_bitwise_identical": bool(torch.equal(embedded,
                                                                 embedded_again)),
                "embedding_max_abs_difference": float(
                    (embedded - embedded_again).abs().max()),
            }
            reloaded, _ = load_flow(path, device=device)
            reloaded.eval()
            with torch.no_grad():
                mask = node_mask_from_counts([args.target_n] * args.latents, device=device)
                condition_values = None
                if features:
                    values = standardise_condition(
                        features, scaler=scaler, n=[args.target_n] * args.latents,
                        qed=([args.target_qed] * args.latents
                             if "qed" in features else None))
                    condition_values = reloaded.conditioning(
                        torch.from_numpy(values).to(device),
                        torch.ones_like(torch.from_numpy(values)).to(device),
                        args.latents, device)
                generator = torch.Generator(device=device).manual_seed(int(args.seed))
                drawn_reloaded = reloaded.prior.sample(
                    mask, condition_values, temperature=args.temperature,
                    generator=generator)
                embedded_reloaded = reloaded.inverse(drawn_reloaded, mask,
                                                     condition_values)
            block["repeat_after_reload"] = {
                "latents_bitwise_identical": bool(torch.equal(drawn, drawn_reloaded)),
                "embeddings_bitwise_identical": bool(
                    torch.equal(embedded, embedded_reloaded)),
                "embedding_max_abs_difference": float(
                    (embedded - embedded_reloaded).abs().max()),
                "interpretation": ("identical here but different across SCRIPTS points "
                                   "at model-load order or global RNG consumption, not "
                                   "at the flow itself; different here points at "
                                   "nondeterministic kernels")}

            # ---- 4. decoder determinism ----
            from zinc_gnf.generation import decode_embeddings

            first = decode_embeddings(autoencoder, embedded,
                                      [args.target_n] * args.latents)
            second = decode_embeddings(autoencoder, embedded,
                                       [args.target_n] * args.latents)
            same = sum(1 for a, b in zip(first, second)
                       if (a.get("degree") or {}).get("smiles")
                       == (b.get("degree") or {}).get("smiles"))
            block["decoder_determinism"] = {
                "identical_smiles_fraction": same / max(len(first), 1),
                "note": ("below 1.0 means the decoder or assembler is not "
                         "deterministic, which would explain run-to-run differences "
                         "even with identical latents")}
            report["models"][label] = block
            print(f"{label}: real round-trip abs max "
                  f"{block['real_molecules_round_trip']['absolute'].get('max'):.3e}, "
                  f"latent round-trip abs max "
                  f"{block['generated_latents_round_trip']['absolute'].get('max'):.3e}, "
                  f"logdet/node {block['log_det_per_node']['mean']:.2f}, "
                  f"decoder identical "
                  f"{block['decoder_determinism']['identical_smiles_fraction']:.3f}")

        report["cross_model_note"] = (
            "Compare the two models' round-trip errors against EACH OTHER. A baseline "
            "whose training log warned about a large log-determinant is only a problem "
            "if its round-trip error is materially worse than the conditional model's.")
        (out / "numerical_check.json").write_text(
            json.dumps(report, indent=2, default=str))
        print(f"\nwrote {out / 'numerical_check.json'}")
        return 0
    except FileNotFoundError as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2
    except Exception:  # noqa: BLE001
        print("FATAL: unexpected\n" + traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
