#!/usr/bin/env python3
"""Stage 3c — summarise a same-budget autoencoder ablation (read-only, tiny).

Reads every `<root>/<variant>/seed_*/result.json` written by
`03b_reconstruction_experiment.py`, and reports per-variant mean +- std over
seeds, and the PAIRED difference of each variant against `--reference`
(same seed => same training subset, same initial seed, same validation
molecules). Uncertainty shown is across training seeds only.

Exit codes: 0 ok; 2 nothing found; 4 bad arguments.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from zinc_gnf.utils.runtime import atomic_write_json  # noqa: E402

METRICS = (
    ("headline_exact", "HEADLINE exact (own decoder)"),
    ("headline_validity", "headline validity"),
    ("headline_exact_among_decodable", "headline exact | decod."),
    ("headline_exact_among_unreachable", "headline exact | unreach."),
    ("degree_exact", "DEGREE-decoder exact"),
    ("degree_validity", "degree-decoder validity"),
    ("degree_exact_among_unreachable", "degree exact | unreach."),
    ("degree_accuracy", "degree accuracy"),
    ("constrained_exact", "constrained exact"),
    ("raw_exact", "raw exact"),
    ("constrained_validity", "constrained validity"),
    ("raw_validity", "raw validity"),
    ("exact_among_decodable", "exact | decodable"),
    ("exact_among_unreachable", "exact | unreachable"),
    ("atom_accuracy", "atom accuracy"),
    ("real_bond_accuracy", "real-bond accuracy"),
)


def load_runs(root: Path) -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = {}
    for path in sorted(root.glob("*/seed_*/result.json")):
        result = json.loads(path.read_text())
        runs.setdefault(path.parent.parent.name, {})[int(path.parent.name.split("_")[1])] = result
    return runs


def finite(values):
    return [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="directory holding <variant>/seed_*/result.json")
    parser.add_argument("--reference", default="baseline")
    parser.add_argument("--output", default=None)
    arguments = parser.parse_args(argv)

    runs = load_runs(Path(arguments.root))
    if not runs:
        print(f"no result.json files under {arguments.root}", file=sys.stderr)
        return 2
    if arguments.reference not in runs:
        print(f"reference variant {arguments.reference!r} not found; have {sorted(runs)}", file=sys.stderr)
        return 4

    summary: dict = {"root": str(arguments.root), "reference": arguments.reference, "variants": {}}
    lines = [f"Same-budget ablation under {arguments.root} (uncertainty = across training seeds only)"]
    for variant, by_seed in sorted(runs.items()):
        seeds = sorted(by_seed)
        entry = {"seeds": seeds, "metrics": {}, "paired_vs_reference": {}}
        first = by_seed[seeds[0]]
        lines.append(f"\n[{variant}] seeds={seeds} steps={first['steps']} train={first['train_molecules']:,} "
                     f"val={first['val_molecules']:,} params={first['parameters']:,}")
        for key, label in METRICS:
            values = finite([by_seed[s]["final"].get(key) for s in seeds])
            if not values:
                continue
            entry["metrics"][key] = {"mean": float(np.mean(values)), "std": float(np.std(values)),
                                     "values": values}
            lines.append(f"  {label:<22} {np.mean(values):.4f} +- {np.std(values):.4f}   {['%.3f' % v for v in values]}")
        if variant != arguments.reference:
            shared = sorted(set(seeds) & set(runs[arguments.reference]))
            for key, label in (METRICS[:4] + METRICS[8:13]):
                diffs = finite([by_seed[s]["final"].get(key) - runs[arguments.reference][s]["final"].get(key)
                                for s in shared
                                if by_seed[s]["final"].get(key) is not None
                                and runs[arguments.reference][s]["final"].get(key) is not None])
                if diffs:
                    entry["paired_vs_reference"][key] = {
                        "mean": float(np.mean(diffs)), "std": float(np.std(diffs)),
                        "per_seed": diffs, "seeds_where_better": int(sum(d > 0 for d in diffs)), "n": len(diffs)}
                    lines.append(f"  paired {label:<18} {np.mean(diffs):+.4f} +- {np.std(diffs):.4f}   "
                                 f"better in {sum(d > 0 for d in diffs)}/{len(diffs)} seeds")
        flows = [by_seed[s].get("flow_relevant") for s in seeds if by_seed[s].get("flow_relevant")]
        if flows:
            flow_entry = {
                "noise_variance_fraction": float(np.mean([f["embedding_statistics"]["noise_variance_fraction"] for f in flows])),
                "participation_ratio": float(np.mean([f["embedding_statistics"]["participation_ratio"] for f in flows])),
                "exact_across_noise_draws_std": float(np.mean([f["noise_draw_spread"]["std"] for f in flows])),
                "exact_under_embedding_error": {
                    str(sigma): float(np.mean([next(r["exact"] for r in f["decoder_robustness_to_embedding_error"]
                                                    if r["sigma"] == sigma) for f in flows]))
                    for sigma in (r["sigma"] for r in flows[0]["decoder_robustness_to_embedding_error"])},
                "dimension": flows[0]["embedding_statistics"]["dimension"],
            }
            entry["flow_relevant"] = flow_entry
            robust = "  ".join(f"s={k}:{v:.3f}" for k, v in flow_entry["exact_under_embedding_error"].items())
            lines.append(f"  flow-relevant: noise_variance_fraction={flow_entry['noise_variance_fraction']:.3f} "
                         f"participation_ratio={flow_entry['participation_ratio']:.1f}/{flow_entry['dimension']} "
                         f"draw-to-draw exact std={flow_entry['exact_across_noise_draws_std']:.4f}")
            lines.append(f"  exact under embedding error (sigma x per-dim std): {robust}")
        summary["variants"][variant] = entry
    text = "\n".join(lines)
    print(text)
    if arguments.output:
        out = Path(arguments.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(summary, out)
        out.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
        print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
