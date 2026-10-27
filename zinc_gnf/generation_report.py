"""Shared pieces for presentation reports built from SAVED generation outputs.

Factored out of `scripts/09_arm_comparison_figures.py` so that more than one report can
read the same files the same way (stage 9 compares the two architectures; stage 11
compares two training budgets of the same architecture). Behaviour is unchanged: these
functions only READ `metrics.json` / `samples.jsonl` and recompute chemistry from the
saved SMILES with RDKit. Nothing is trained, sampled or written here.

The one rule that matters scientifically: a metric can be genuinely UNDEFINED rather
than zero (uniqueness divides by the number of valid molecules), so missing values are
carried as NaN all the way to the figure and shown as `N/A` - never as 0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import QED
from rdkit.Chem.Draw import rdMolDraw2D

from zinc_gnf.metrics_generation import internal_diversity
from zinc_gnf.molecular_profile import profile_molecules

DECODERS = ("raw", "constrained", "degree")
DECODER_LABELS = {"raw": "raw\n(argmax, no chemistry)",
                  "constrained": "greedy\nvalence-aware",
                  "degree": "degree-constrained\n(+ greedy fallback)"}
DECODER_MEANING = {
    "raw": "argmax bond matrix, no chemistry applied",
    "constrained": "greedy valence-aware assembly with a connectivity pass",
    "degree": "degree-constrained integer program, greedy fallback when infeasible",
}

# Measured on OUR validation split by scripts/08_embedding_populations.py (job 332841)
# and scripts/07_latent_diagnostics.py (job 332840), 2,000 real molecules decoded through
# the frozen autoencoder's own pipeline. Used only when no --reference JSON is supplied,
# and always labelled with this provenance in the output.
REAL_ZINC_VALIDATION = {
    "qed_mean": 0.7314,
    "qed_at_least_0p7": 0.6595,
    "aromatic_rings_mean": 1.8305,
    "aromatic_ring_fraction": 0.631,
    "molecules_with_aromatic_ring": 0.934,
    "degree_fallback_rate": 0.044,
    "source": "measured on the ZINC250K validation split (job 332841 / 332840)",
}


# ---------------------------------------------------------------------------
# Reading saved outputs.
# ---------------------------------------------------------------------------
def resolve(path: Path) -> Path:
    """Accept either a metrics.json or the directory containing it."""
    if path.is_dir():
        return path / "metrics.json"
    return path


def load_run(path: Path, label: str) -> dict[str, Any]:
    """`metrics.json` plus the per-sample records beside it, if any."""
    metrics_path = resolve(path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"{label}: {metrics_path} not found")
    metrics = json.loads(metrics_path.read_text())
    samples_path = metrics_path.parent / "samples.jsonl"
    records: list[dict[str, Any]] = []
    if samples_path.exists():
        with samples_path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    return {"label": label, "metrics": metrics, "records": records,
            "metrics_path": str(metrics_path),
            "samples_path": str(samples_path) if records else None}


# ---------------------------------------------------------------------------
# Missing / undefined handling.
# ---------------------------------------------------------------------------
def number(value: Any) -> float:
    """Coerce to float, mapping missing / null / non-numeric to NaN."""
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def rate(block: Any) -> float:
    """metrics.json stores rates either bare or as {'value': ...}, possibly null."""
    if isinstance(block, dict):
        return number(block.get("value"))
    return number(block)


def undefined(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def show(value: Any, template: str = "{:.4f}", missing: str = "N/A") -> str:
    """Format a metric for display, as N/A when it is undefined."""
    return missing if undefined(value) else template.format(float(value))


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------
def decoder_numbers(run: dict[str, Any]) -> dict[str, Any]:
    """Per-decoder numbers from metrics.json, plus what only the records can give."""
    decoders = run["metrics"].get("decoders", {})
    headline = run["metrics"].get("headline_decoder", "degree")
    numbers: dict[str, Any] = {"headline_decoder": headline,
                               "samples": run["metrics"].get("samples")}
    for name in DECODERS:
        block = decoders.get(name)
        if block is None:
            continue
        numbers[name] = {
            "strict_validity": rate(block.get("strict_validity")),
            "sanitizable": rate(block.get("sanitizable")),
            "uniqueness": rate(block.get("uniqueness")),
            "novelty_vs_train": rate(block.get("novelty_vs_train")),
            "nuv": number(block.get("nuv")),
            "fallback_rate": rate(block.get("fallback_rate")),
            "internal_diversity": number((block.get("diversity") or {}).get(
                "internal_diversity")),
            "qed_mean": number((block.get("qed") or {}).get("mean")),
            "qed_at_least_0p7": rate(block.get("qed_at_least_0p7")),
        }
    head = numbers.get(headline)
    if head is not None and run["records"]:
        entries = [r[headline] for r in run["records"] if headline in r]
        solved = [e for e in entries if not e.get("fallback")]
        head["samples_solved_by_program"] = len(solved) / len(entries) if entries else float("nan")
        head["validity_among_solved"] = (
            sum(1 for e in solved if e.get("valid")) / len(solved) if solved else float("nan"))
        head["validity_among_fallback"] = (
            sum(1 for e in entries if e.get("fallback") and e.get("valid"))
            / max(1, len(entries) - len(solved)) if len(entries) - len(solved) else float("nan"))
        smiles = [e.get("smiles") if e.get("valid") else None for e in entries]
        profile = profile_molecules(smiles, name=run["label"], scaffolds=True)
        head["aromatic_rings_mean"] = profile["aromatic_rings"]["mean"]
        head["rings_mean"] = profile["rings"]["mean"]
        head["aromatic_ring_fraction"] = profile["aromatic_ring_fraction"]
        head["qed_mean_recomputed"] = profile["qed"]["mean"]
        head["qed_at_least_0p7_recomputed"] = profile["qed_at_least_0p7"]
        head["distinct_scaffolds_per_molecule"] = (profile.get("scaffolds") or {}).get(
            "per_molecule", float("nan"))
    return numbers


def recompute_decoder(records: list[dict[str, Any]], decoder: str, *,
                      diversity: bool = True) -> dict[str, Any]:
    """Everything derivable from `samples.jsonl` alone, recomputed with RDKit.

    Independent of `metrics.json`, so a saved run can be verified rather than trusted.
    Novelty is deliberately absent: it needs the training SMILES, which are not in the
    sample file, and guessing it would be worse than leaving it to metrics.json.
    """
    entries = [r[decoder] for r in records if decoder in r]
    total = len(entries)
    if not total:
        return {"samples": 0}
    valid = [e for e in entries if e.get("valid") and e.get("smiles")]
    valid_smiles = [str(e["smiles"]) for e in valid]
    fallback_entries = [e for e in entries if "fallback" in e]
    profile = profile_molecules([e["smiles"] for e in valid], name=decoder, scaffolds=True)
    result: dict[str, Any] = {
        "samples": total,
        "strict_validity": len(valid) / total,
        "sanitizable": sum(1 for e in entries if e.get("sanitizable")) / total,
        "valid_molecules": len(valid),
        "unique_molecules": len(set(valid_smiles)),
        "uniqueness": (len(set(valid_smiles)) / len(valid)) if valid else float("nan"),
        "fallback_rate": (sum(1 for e in fallback_entries if e.get("fallback"))
                          / len(fallback_entries)) if fallback_entries else float("nan"),
        "qed_mean": profile["qed"]["mean"],
        "qed_median": profile["qed"]["median"],
        "qed_at_least_0p7": profile["qed_at_least_0p7"],
        "rings_mean": profile["rings"]["mean"],
        "aromatic_rings_mean": profile["aromatic_rings"]["mean"],
        "aromatic_ring_fraction": profile["aromatic_ring_fraction"],
        "molecules_with_aromatic_ring": profile["molecules_with_aromatic_ring"],
        "heavy_atoms_mean": profile["heavy_atoms"]["mean"],
        "internal_diversity": float("nan"),
    }
    solved = [e for e in entries if "fallback" in e and not e.get("fallback")]
    if solved:
        result["samples_solved_by_program"] = len(solved) / total
        result["validity_among_solved"] = sum(1 for e in solved if e.get("valid")) / len(solved)
    if diversity and len(set(valid_smiles)) >= 2:
        result["internal_diversity"] = internal_diversity(valid_smiles)["internal_diversity"]
    return result


def load_references(paths: list[Path]) -> dict[str, dict[str, float]]:
    """Real / reconstruction reference values from earlier diagnostic outputs."""
    wanted = {"real_validation": "real ZINC250K (validation)", "real": "real ZINC250K (validation)",
              "ae_reconstruction": "frozen-AE reconstruction"}
    references: dict[str, dict[str, float]] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"reference not found: {path}")
        payload = json.loads(path.read_text())
        profiles = (payload.get("profiles") or {}).get("profiles") or {}
        for name, profile in profiles.items():
            if name not in wanted:
                continue
            label = wanted[name]
            references.setdefault(label, {
                "qed_mean": number((profile.get("qed") or {}).get("mean")),
                "qed_at_least_0p7": number(profile.get("qed_at_least_0p7")),
                "aromatic_rings_mean": number((profile.get("aromatic_rings") or {}).get("mean")),
                "aromatic_ring_fraction": number(profile.get("aromatic_ring_fraction")),
                "source": str(path),
            })
        if "degree_fallback_rate" in str(payload):
            for name, profile in profiles.items():
                if name in wanted and "degree_fallback_rate" in profile:
                    references[wanted[name]]["degree_fallback_rate"] = number(
                        profile["degree_fallback_rate"])
    return references


# ---------------------------------------------------------------------------
# Molecule grid.
# ---------------------------------------------------------------------------
def choose_molecules(records: list[dict[str, Any]], decoder: str, *, count: int,
                     mode: str) -> list[tuple[str, float]]:
    """Unique valid molecules with their QED, chosen by a stated, deterministic rule."""
    seen: dict[str, float] = {}
    for record in records:
        entry = record.get(decoder) or {}
        smiles = entry.get("smiles")
        if not entry.get("valid") or not smiles or smiles in seen:
            continue
        value = entry.get("qed")
        if value is None:
            with rdBase.BlockLogs():
                molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                continue
            value = float(QED.qed(molecule))
        seen[smiles] = float(value)
    ordered = sorted(seen.items(), key=lambda item: item[1])
    if not ordered:
        return []
    if mode == "best":
        return list(reversed(ordered[-count:]))
    if mode == "random":
        generator = np.random.default_rng(0)
        picked = generator.choice(len(ordered), size=min(count, len(ordered)), replace=False)
        return [ordered[int(i)] for i in sorted(picked)]
    # "spread": evenly spaced across the QED range, so the grid is not cherry-picked
    indices = np.unique(np.linspace(0, len(ordered) - 1, num=min(count, len(ordered))).round()
                        .astype(int))
    return [ordered[int(i)] for i in indices]


SELECTION_DESCRIPTION = {
    "spread": "evenly spaced across the QED range (not cherry-picked)",
    "best": "the highest-QED molecules",
    "random": "a seeded random sample",
}


def molecule_grid(chosen: list[tuple[str, float]], *, output: Path, subject: str, mode: str,
                  total_unique: int, columns: int = 4) -> None:
    """Draw the chosen molecules with their QED values, plus a caption file."""
    molecules, legends = [], []
    for smiles, value in chosen:
        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        molecules.append(molecule)
        legends.append(f"QED {value:.3f}")
    if not molecules:
        raise ValueError("no valid molecules to draw")
    rows = int(np.ceil(len(molecules) / columns))
    panel = 300
    description = SELECTION_DESCRIPTION[mode]
    for drawer, suffix in ((rdMolDraw2D.MolDraw2DCairo, ".png"),
                           (rdMolDraw2D.MolDraw2DSVG, ".svg")):
        # The four-argument form sets the PANEL size, which is what tiles the molecules;
        # with only the canvas size RDKit draws every molecule over the whole canvas.
        canvas = drawer(columns * panel, rows * panel, panel, panel)
        options = canvas.drawOptions()
        options.legendFontSize = 20
        try:
            canvas.DrawMolecules(molecules, legends=legends)
        except Exception:  # noqa: BLE001 - Cairo may be unavailable on a bare node
            if suffix == ".png":
                continue
            raise
        canvas.FinishDrawing()
        text = canvas.GetDrawingText()
        path = output.with_suffix(suffix)
        if isinstance(text, bytes):
            path.write_bytes(text)
        else:
            path.write_text(text, encoding="utf-8")
    caption = (f"{subject}: {len(molecules)} of {total_unique} unique valid molecules, "
               f"{description}. Decoder: degree-constrained assembly.")
    output.with_name(output.name + "_caption.txt").write_text(caption + "\n", encoding="utf-8")


__all__ = [
    "DECODERS",
    "DECODER_LABELS",
    "DECODER_MEANING",
    "REAL_ZINC_VALIDATION",
    "SELECTION_DESCRIPTION",
    "choose_molecules",
    "decoder_numbers",
    "load_references",
    "load_run",
    "molecule_grid",
    "number",
    "rate",
    "recompute_decoder",
    "resolve",
    "show",
    "undefined",
]
