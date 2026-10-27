#!/usr/bin/env python3
"""Stage 6 — aggregate generation metrics across flow seeds and temperatures.

Reads every `<root>/<run>/T<temperature>/metrics.json` (as written by
`scripts/05_generate.py`), groups the runs by temperature and reports
mean +- std ACROSS FLOW SEEDS for the headline metrics. The uncertainty
covers flow seeds only: one frozen autoencoder is shared by every run, so
autoencoder variability is deliberately not included (project decision 3).

Writes `<output>` (default `<root>/generation_summary.json`) and a `.txt`
beside it.

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
    ("strict_validity", "strict validity"),
    ("uniqueness", "uniqueness"),
    ("novelty_vs_train", "novelty vs train"),
    ("nuv", "N.U.V."),
    ("internal_diversity", "internal diversity"),
    ("fallback_rate", "decoder fallback"),
    ("sanitizable", "sanitizable (raw sanity)"),
)


def load_runs(root: Path) -> dict[str, dict[str, dict]]:
    """{temperature: {run name: metrics}} for every metrics.json under `root`."""
    runs: dict[str, dict[str, dict]] = {}
    for path in sorted(root.glob("*/T*/metrics.json")):
        payload = json.loads(path.read_text())
        runs.setdefault(path.parent.name, {})[path.parent.parent.name] = payload
    return runs


def finite(values):
    return [float(v) for v in values
            if v is not None and not (isinstance(v, float) and np.isnan(v))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="directory holding <run>/T<temp>/metrics.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--decoder", default=None,
                        help="report this decoder instead of each run's headline")
    arguments = parser.parse_args(argv)

    root = Path(arguments.root)
    runs = load_runs(root)
    if not runs:
        print(f"no <run>/T*/metrics.json files under {root}", file=sys.stderr)
        return 2

    summary: dict = {"root": str(root), "decoder": arguments.decoder or "headline",
                     "uncertainty": "mean +- std across flow seeds; one frozen autoencoder",
                     "temperatures": {}}
    lines = [f"Generation summary under {root}",
             "uncertainty = across flow seeds only (the autoencoder is frozen and shared)"]

    for temperature in sorted(runs, key=lambda t: float(t.lstrip("T"))):
        by_run = runs[temperature]
        entry: dict = {"runs": sorted(by_run), "metrics": {}}
        first = by_run[sorted(by_run)[0]]
        samples = [r.get("samples") for r in by_run.values()]
        entry["samples_per_run"] = samples
        decoder = arguments.decoder or first.get("headline_decoder", "degree")
        entry["decoder"] = decoder
        lines.append(f"\n[{temperature}] runs={sorted(by_run)} decoder={decoder} "
                     f"samples/run={samples}")
        if len(set(samples)) > 1:
            lines.append("  WARNING: runs have different sample budgets; metrics are not comparable")
        for key, label in METRICS:
            values = []
            for payload in by_run.values():
                block = payload.get("decoders", {}).get(decoder, {})
                raw = block.get(key)
                if isinstance(raw, dict):
                    raw = raw.get("value")
                if key == "internal_diversity":
                    raw = block.get("diversity", {}).get("internal_diversity")
                values.append(raw)
            values = finite(values)
            if not values:
                continue
            entry["metrics"][key] = {"mean": float(np.mean(values)), "std": float(np.std(values)),
                                     "values": values, "n": len(values)}
            lines.append(f"  {label:<24} {np.mean(values):.4f} +- {np.std(values):.4f}   "
                         f"{['%.4f' % v for v in values]}")
        summary["temperatures"][temperature] = entry

    output = Path(arguments.output) if arguments.output else root / "generation_summary.json"
    atomic_write_json(summary, output)
    report = "\n".join(lines)
    output.with_suffix(".txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nreport: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
