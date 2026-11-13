#!/usr/bin/env python3
"""Where does the conditional signal die? Latent vs raw decode vs degree-corrected.

READ-ONLY with respect to every model and every existing result. Both models are opened
in eval() with `requires_grad_(False)`, every pass runs under `no_grad`, nothing is
trained, no checkpoint is written, and output goes to a NEW directory.

It does draw new samples, because the question cannot be answered without controlled
generation: measuring a response to a condition change requires holding the prior noise
fixed while only the condition moves.

THE DESIGN
----------
Three arms, all sharing one prior-noise draw per N so every comparison is per-molecule:

  QED ARM        mask N fixed, condition N fixed, requested QED varied.
  N-CONDITION    mask N fixed, requested QED fixed, the CONDITION's N varied away from
  ARM            the mask's N. This is the redundancy test and it needs no training:
                 the decoder takes its node count from the MASK, so if changing only the
                 condition's N leaves the output unchanged, the explicit N conditioning
                 is carrying no information the mask does not already supply.
  NOISE ARM      condition fixed, a SECOND independent prior draw. This is the reference
                 scale: it is how much the output moves for reasons that have nothing to
                 do with conditioning.

Everything is reported as a ratio to the noise arm, which makes the three measurement
stages comparable even though they hold different kinds of object (a continuous tensor,
then a discrete graph, then a repaired discrete graph).

FAIR COMPARISON BETWEEN THE N AND QED DIRECTIONS
------------------------------------------------
Comparing "N 15 -> 30" against "QED 0.4 -> 0.8" directly is meaningless: they are
different units. Each arm therefore also records the norm of its perturbation in
STANDARDISED condition space, the space the conditioning network actually sees, and the
headline sensitivity is the latent displacement PER UNIT standardised condition change.
That is what makes "is N dominating the conditional representation?" a measurable
question rather than a rhetorical one.

WHAT THIS CAN AND CANNOT SETTLE
-------------------------------
It can localise the loss: a large latent displacement with an unchanged decoded molecule
indicts the decoder; a changed raw molecule with an unchanged corrected one indicts the
repair. It CANNOT tell us whether the size effect on decoded QED reported earlier is
conditioning or chemistry -- bigger molecules genuinely have different QED distributions,
so part of that 2.2x is real chemistry and not conditioner capacity. The N-condition arm
isolates the part attributable to conditioning, and may well overturn that earlier
reading.

Exit codes: 0 ok; 2 missing input; 3 provenance mismatch; 4 bad arguments; 1 unexpected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
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
from zinc_gnf.generation import decode_embeddings  # noqa: E402
from zinc_gnf.trainer_autoencoder import ResumeMismatchError, load_autoencoder  # noqa: E402
from zinc_gnf.trainer_flow import file_sha256, load_flow  # noqa: E402
from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402

PROBE_NOISE_SEED = 20261116
PROBE_NOISE_SEED_B = 20261117      # the independent second draw for the noise arm
DECODERS = ("raw", "degree")       # pre-correction and degree-constrained


def mean(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def median(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else None


def ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0.0):
        return None
    return numerator / denominator


# --------------------------------------------------------------------- sampling ---
def standardised(flow, scaler, *, n, qed, count):
    return standardise_condition(flow.config.condition_features, scaler=scaler,
                                 n=[n] * count, qed=[qed] * count)


@torch.no_grad()
def embeddings_for(flow, *, scaler, mask_n, condition_n, qed, eps, temperature, device):
    """Invert one prior draw under one condition. The MASK fixes the molecule size."""
    count = eps.shape[0]
    mask = torch.ones((count, mask_n), dtype=torch.bool, device=device)
    values_raw = standardised(flow, scaler, n=condition_n, qed=qed, count=count)
    values = torch.from_numpy(values_raw).to(device)
    condition = condition_embedding_for(flow, values, torch.ones_like(values),
                                        count, device)
    mu, sigma = flow.prior.parameters_for(count, condition, device)
    chunk = eps[:, :mask_n, :]
    z = (mu.unsqueeze(1) + float(temperature) * sigma.unsqueeze(1) * chunk) \
        * mask.unsqueeze(-1)
    return (flow.inverse(z, mask, condition).detach(), mask,
            np.asarray(values_raw, dtype=np.float64)[0])


@torch.no_grad()
def decode(autoencoder, embeddings, mask_n):
    from rdkit import Chem, rdBase
    from rdkit.Chem import QED, rdMolDescriptors

    records = decode_embeddings(autoencoder, embeddings,
                                [mask_n] * embeddings.shape[0], decoders=DECODERS)
    out: dict[str, list] = {d: [] for d in DECODERS}
    for decoder in DECODERS:
        for record in records:
            entry = record.get(decoder) or {}
            row: dict[str, Any] = {"valid": bool(entry.get("valid")),
                                   "smiles": entry.get("smiles"),
                                   "qed": None, "aromatic_rings": None,
                                   "heavy_atoms": None, "scorable": False}
            if entry.get("valid") and entry.get("smiles"):
                with rdBase.BlockLogs():
                    molecule = Chem.MolFromSmiles(str(entry["smiles"]))
                if molecule is not None:
                    row.update({
                        "qed": float(QED.qed(molecule)),
                        "aromatic_rings": float(
                            rdMolDescriptors.CalcNumAromaticRings(molecule)),
                        "heavy_atoms": float(molecule.GetNumHeavyAtoms()),
                        "scorable": True})
            out[decoder].append(row)
    return out


# -------------------------------------------------------------------- comparison ---
def latent_change(a, b, mask) -> dict[str, Any]:
    """Per-molecule displacement, normalised per element so sizes are comparable."""
    weights = mask.unsqueeze(-1).to(a.dtype)
    delta = ((b - a) * weights).flatten(1)
    base = (a * weights).flatten(1)
    elements = float(mask.sum(-1).float().mean()) * a.shape[-1]
    norms = delta.norm(dim=1)
    flat = delta / norms.clamp(min=1e-12).unsqueeze(1)
    cosines = (flat[:-1] * flat[1:]).sum(dim=1) if flat.shape[0] > 1 else None
    return {
        "rms_per_element": float((norms / max(elements, 1.0) ** 0.5).mean()),
        "norm_mean": float(norms.mean()),
        "relative_to_embedding_norm": float(
            (norms / base.norm(dim=1).clamp(min=1e-12)).mean()),
        # a consistent direction across molecules means the condition acts as a
        # translation; near-zero cosine means it acts per-molecule
        "direction_consistency": float(cosines.mean()) if cosines is not None else None,
    }


def decoded_change(before, after, *, decoder) -> dict[str, Any]:
    """Did the discrete output move, and did its QED move with it?"""
    a_rows, b_rows = before[decoder], after[decoder]
    pairs = list(zip(a_rows, b_rows))
    both = [(a, b) for a, b in pairs if a["scorable"] and b["scorable"]]
    identity_changed = [int(a["smiles"] != b["smiles"]) for a, b in pairs
                        if a["smiles"] and b["smiles"]]
    return {
        "validity_before": mean([float(a["valid"]) for a, _ in pairs]),
        "validity_after": mean([float(b["valid"]) for _, b in pairs]),
        "jointly_scorable": len(both),
        "smiles_change_rate": mean(identity_changed),
        "delta_qed_mean": mean([b["qed"] - a["qed"] for a, b in both]),
        "abs_delta_qed_mean": mean([abs(b["qed"] - a["qed"]) for a, b in both]),
        "delta_aromatic_rings": mean(
            [b["aromatic_rings"] - a["aromatic_rings"] for a, b in both]),
        "delta_heavy_atoms": mean([b["heavy_atoms"] - a["heavy_atoms"] for a, b in both]),
    }


def compare(label, first, second) -> dict[str, Any]:
    block: dict[str, Any] = {
        "arm": label,
        "condition_delta_standardised_norm": float(
            np.linalg.norm(second["condition_values"] - first["condition_values"])),
        "latent": latent_change(first["embeddings"], second["embeddings"],
                                first["mask"]),
        "decoded": {d: decoded_change(first["decoded"], second["decoded"], decoder=d)
                    for d in DECODERS},
    }
    norm = block["condition_delta_standardised_norm"]
    block["latent_rms_per_unit_condition"] = ratio(
        block["latent"]["rms_per_element"], norm if norm > 1e-9 else None)
    return block


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--ae-checkpoint", default=None)
    parser.add_argument("--data-dir",
                        default="data/processed/zinc250k_charge_aware_moflow_fixed")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-n", default="15,23,30")
    parser.add_argument("--qed-values", default="0.4,0.6,0.8")
    parser.add_argument("--reference-qed", type=float, default=0.6)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--allow-ae-mismatch", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        targets_n = [int(v) for v in args.target_n.split(",") if v.strip()]
        qed_values = [float(v) for v in args.qed_values.split(",") if v.strip()]
        if len(targets_n) < 2:
            raise ValueError("need at least two N values for the N-condition arm")
        if len(qed_values) < 2:
            raise ValueError("need at least two QED values for the QED arm")
        if args.samples < 4:
            raise ValueError("--samples must be >= 4")
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 4

    started = time.perf_counter()
    try:
        device = torch.device(args.device)
        scaler = load_condition_scaler(args.data_dir)
        flow, probe = load_flow(args.flow_checkpoint, device=device)
        flow.eval().requires_grad_(False)
        if not flow.config.conditional:
            raise ValueError("the probe needs a CONDITIONAL flow")
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
        print(f"flow step={probe.get('step')}  AE match={match}  "
              f"dim={flow.config.node_embedding_dim}", flush=True)

        width = max(targets_n)
        dimension = flow.config.node_embedding_dim
        payload: dict[str, Any] = {
            "framing": ("Read-only sensitivity probe. One prior-noise draw per N is "
                        "shared across every condition, so each comparison is "
                        "per-molecule. All effects are reported as ratios to a NOISE "
                        "arm, which is the scale of variation unrelated to "
                        "conditioning."),
            "settings": vars(args) | {"probe_noise_seeds":
                                       [PROBE_NOISE_SEED, PROBE_NOISE_SEED_B]},
            "flow": {"checkpoint": str(args.flow_checkpoint),
                     "step": probe.get("step"),
                     "sha256": file_sha256(args.flow_checkpoint)},
            "autoencoder": {"checkpoint": str(ae_path), "sha256_match": match},
            "by_n": {},
        }

        for n in targets_n:
            generator = torch.Generator(device=device).manual_seed(PROBE_NOISE_SEED)
            eps = torch.randn((args.samples, width, dimension), device=device,
                              generator=generator)
            generator_b = torch.Generator(device=device).manual_seed(PROBE_NOISE_SEED_B)
            eps_b = torch.randn((args.samples, width, dimension), device=device,
                                generator=generator_b)

            def state(*, condition_n, qed, noise):
                x, mask, values = embeddings_for(
                    flow, scaler=scaler, mask_n=n, condition_n=condition_n, qed=qed,
                    eps=noise, temperature=args.temperature, device=device)
                return {"embeddings": x, "mask": mask, "condition_values": values,
                        "decoded": decode(autoencoder, x, n)}

            reference = state(condition_n=n, qed=args.reference_qed, noise=eps)
            block: dict[str, Any] = {"mask_n": n, "arms": {}}

            # NOISE ARM: the reference scale
            noise_state = state(condition_n=n, qed=args.reference_qed, noise=eps_b)
            block["arms"]["noise"] = compare("noise", reference, noise_state)

            # QED ARM
            low, high = min(qed_values), max(qed_values)
            low_state = state(condition_n=n, qed=low, noise=eps)
            high_state = state(condition_n=n, qed=high, noise=eps)
            block["arms"]["qed_low_to_high"] = compare(
                f"qed {low}->{high}", low_state, high_state)

            # N-CONDITION ARM: mask stays at n, only the condition's N moves
            for other in targets_n:
                if other == n:
                    continue
                mismatched = state(condition_n=other, qed=args.reference_qed, noise=eps)
                block["arms"][f"condition_n_{n}_to_{other}"] = compare(
                    f"condition N {n}->{other} (mask stays {n})",
                    reference, mismatched)

            noise_rms = block["arms"]["noise"]["latent"]["rms_per_element"]
            for key, arm in block["arms"].items():
                arm["latent_snr_vs_noise"] = ratio(
                    arm["latent"]["rms_per_element"], noise_rms)
                for decoder in DECODERS:
                    arm["decoded"][decoder]["qed_snr_vs_noise"] = ratio(
                        arm["decoded"][decoder]["abs_delta_qed_mean"],
                        block["arms"]["noise"]["decoded"][decoder][
                            "abs_delta_qed_mean"])
            payload["by_n"][str(n)] = block

            qed_arm = block["arms"]["qed_low_to_high"]
            print(f"  N={n}: latent SNR(QED)={_f(qed_arm['latent_snr_vs_noise'])}  "
                  f"raw dQED={_f(qed_arm['decoded']['raw']['delta_qed_mean'])}  "
                  f"degree dQED={_f(qed_arm['decoded']['degree']['delta_qed_mean'])}  "
                  f"smiles change(raw)="
                  f"{_f(qed_arm['decoded']['raw']['smiles_change_rate'], '.3f')}",
                  flush=True)

        payload["synthesis"] = synthesise(payload)
        payload["elapsed_seconds"] = time.perf_counter() - started
        atomic_write_json(payload, out_dir / "condition_sensitivity.json")
        report = format_report(payload)
        (out_dir / "condition_sensitivity.txt").write_text(report)
        print("\n" + report)
        print(f"\nwrote {out_dir / 'condition_sensitivity.json'}")
        return 0
    except FileNotFoundError as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2
    except ResumeMismatchError as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 3
    except Exception:  # noqa: BLE001
        print("FATAL: unexpected\n" + traceback.format_exc(), file=sys.stderr)
        return 1


def synthesise(payload) -> dict[str, Any]:
    """Answer the five questions mechanically from the measured arms."""
    latent_qed, latent_n, raw_qed, degree_qed = [], [], [], []
    identity_qed, identity_n, size_n = [], [], []
    per_unit_qed, per_unit_n = [], []
    for block in payload["by_n"].values():
        qed_arm = block["arms"].get("qed_low_to_high") or {}
        latent_qed.append(qed_arm.get("latent_snr_vs_noise"))
        per_unit_qed.append(qed_arm.get("latent_rms_per_unit_condition"))
        raw_qed.append((qed_arm.get("decoded", {}).get("raw") or {}).get(
            "delta_qed_mean"))
        degree_qed.append((qed_arm.get("decoded", {}).get("degree") or {}).get(
            "delta_qed_mean"))
        identity_qed.append((qed_arm.get("decoded", {}).get("raw") or {}).get(
            "smiles_change_rate"))
        for key, arm in block["arms"].items():
            if not key.startswith("condition_n_"):
                continue
            latent_n.append(arm.get("latent_snr_vs_noise"))
            per_unit_n.append(arm.get("latent_rms_per_unit_condition"))
            identity_n.append((arm.get("decoded", {}).get("raw") or {}).get(
                "smiles_change_rate"))
            size_n.append((arm.get("decoded", {}).get("raw") or {}).get(
                "delta_heavy_atoms"))
    dominance = ratio(mean(per_unit_n), mean(per_unit_qed))
    return {
        "flow_responds_to_qed": {
            "latent_snr_vs_noise_mean": mean(latent_qed),
            "answer": _verdict_snr(mean(latent_qed)),
            "criterion": "SNR > 1 means the QED change moves the latent more than a "
                         "fresh noise draw does",
        },
        "decoder_preserves_the_response": {
            "raw_delta_qed_mean": mean(raw_qed),
            "identity_change_rate_mean": mean(identity_qed),
            "answer": _verdict_transmission(mean(latent_qed), mean(raw_qed),
                                            mean(identity_qed)),
        },
        "degree_correction_alters_the_response": {
            "raw_delta_qed_mean": mean(raw_qed),
            "degree_delta_qed_mean": mean(degree_qed),
            "attenuation": ratio(mean(degree_qed), mean(raw_qed)),
            "answer": _verdict_correction(mean(raw_qed), mean(degree_qed)),
        },
        "n_dominates_the_conditional_representation": {
            "latent_rms_per_unit_condition_n": mean(per_unit_n),
            "latent_rms_per_unit_condition_qed": mean(per_unit_qed),
            "n_over_qed_dominance": dominance,
            "answer": _verdict_dominance(dominance),
            "criterion": "per UNIT standardised condition change, so the two "
                         "directions are comparable",
        },
        "explicit_n_conditioning_is_redundant": {
            "identity_change_rate_from_condition_n_only": mean(identity_n),
            "heavy_atom_change_from_condition_n_only": mean(size_n),
            "answer": _verdict_redundancy(mean(identity_n), mean(size_n)),
            "criterion": "the mask fixes the node count, so if moving ONLY the "
                         "condition's N leaves the decoded molecule unchanged, the "
                         "explicit N channel carries nothing the mask does not",
        },
        "caveat": ("Three node counts and one checkpoint. The N-condition arm isolates "
                   "the conditioning contribution of N; it does NOT measure the "
                   "chemical effect of molecule size, which is real and separate. The "
                   "earlier 2.2x size-dominance reading conflated the two and should be "
                   "replaced by the per-unit numbers here."),
    }


def _verdict_snr(value):
    if value is None:
        return "not measurable"
    if value > 2.0:
        return "YES, strongly: the latent moves well beyond the noise scale"
    if value > 1.0:
        return "YES, weakly: above the noise scale but not by much"
    return "NO: the QED change moves the latent less than a fresh noise draw"


def _verdict_transmission(latent, raw_delta, identity):
    if latent is None or raw_delta is None:
        return "not measurable"
    if identity is not None and identity < 0.1:
        return ("NO: the decoded molecule barely changes at all, so the decoder "
                "collapses the latent displacement")
    if abs(raw_delta) < 0.02:
        return ("PARTIALLY: the molecule changes but its QED does not, so the "
                "displacement is QED-orthogonal at the decoder")
    return "YES: the raw decode carries a QED change"


def _verdict_correction(raw_delta, degree_delta):
    if raw_delta is None or degree_delta is None:
        return "not measurable"
    if abs(raw_delta) < 0.02 and abs(degree_delta) < 0.02:
        return "NOTHING TO ERASE: neither stage shows a QED response"
    attenuation = ratio(degree_delta, raw_delta)
    if attenuation is None:
        return "not measurable"
    if attenuation < 0.5:
        return f"YES: the correction attenuates the response to {attenuation:.0%} of raw"
    if attenuation > 1.5:
        return f"IT AMPLIFIES: the corrected response is {attenuation:.0%} of raw"
    return "NO: the correction roughly preserves the response"


def _verdict_dominance(dominance):
    if dominance is None:
        return "not measurable"
    if dominance > 2.0:
        return (f"YES: per unit of standardised condition change, N moves the latent "
                f"{dominance:.1f}x more than QED does")
    if dominance > 1.0:
        return f"MILDLY: N is {dominance:.1f}x stronger than QED per unit"
    return (f"NO: QED is the stronger direction per unit "
            f"({1 / dominance:.1f}x) \u2014 the dominance hypothesis is not supported")


def _verdict_redundancy(identity, size):
    if identity is None:
        return "not measurable"
    if identity < 0.1 and (size is None or abs(size) < 0.5):
        return ("YES: moving only the condition's N leaves the decoded molecule "
                "essentially unchanged, so the channel is redundant given the mask")
    if identity < 0.4:
        return ("PARTIALLY: the condition's N changes some molecules but not their size")
    return ("NO: the condition's N materially changes the output, so it is not "
            "redundant and removing it would be a real ablation")


def _f(value, spec=".4f"):
    return format(value, spec) if isinstance(value, (int, float)) else "n/a"


def format_report(payload) -> str:
    lines = ["Conditional sensitivity probe: latent vs raw decode vs degree correction",
             "=" * 78, "", payload["framing"], "",
             f"flow step {payload['flow']['step']}  "
             f"AE provenance match {payload['autoencoder']['sha256_match']}", ""]
    for n, block in payload["by_n"].items():
        lines.append(f"--- mask N = {n} " + "-" * 56)
        for key, arm in block["arms"].items():
            lines.append(
                f"  {arm['arm']:<34} |dc_std|="
                f"{_f(arm['condition_delta_standardised_norm'], '.3f')}  "
                f"latent rms={_f(arm['latent']['rms_per_element'])}  "
                f"SNR={_f(arm.get('latent_snr_vs_noise'), '.2f')}  "
                f"dir-consistency="
                f"{_f(arm['latent'].get('direction_consistency'), '.2f')}")
            for decoder in DECODERS:
                decoded = arm["decoded"][decoder]
                lines.append(
                    f"    [{decoder:<6}] validity "
                    f"{_f(decoded['validity_before'], '.3f')}->"
                    f"{_f(decoded['validity_after'], '.3f')}  "
                    f"smiles-change={_f(decoded['smiles_change_rate'], '.3f')}  "
                    f"dQED={_f(decoded['delta_qed_mean'])}  "
                    f"|dQED|={_f(decoded['abs_delta_qed_mean'])}  "
                    f"dArom={_f(decoded['delta_aromatic_rings'], '.3f')}  "
                    f"dAtoms={_f(decoded['delta_heavy_atoms'], '.2f')}  "
                    f"pairs={decoded['jointly_scorable']}")
        lines.append("")
    synthesis = payload["synthesis"]
    lines += ["-- ANSWERS " + "-" * 66]
    for key, label in (
            ("flow_responds_to_qed", "Does the flow respond to QED?"),
            ("decoder_preserves_the_response", "Does the decoder preserve it?"),
            ("degree_correction_alters_the_response", "Does degree correction erase it?"),
            ("n_dominates_the_conditional_representation", "Is N dominating?"),
            ("explicit_n_conditioning_is_redundant", "Is explicit N redundant?")):
        block = synthesis[key]
        lines += [f"  {label}", f"    {block['answer']}"]
        for field, value in block.items():
            if field in ("answer", "criterion"):
                continue
            lines.append(f"      {field}: {_f(value)}")
        if block.get("criterion"):
            lines.append(f"      criterion: {block['criterion']}")
    lines += ["", f"  CAVEAT: {synthesis['caveat']}"]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
