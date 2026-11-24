#!/usr/bin/env python3
"""Audit EXISTING evaluation artefacts. Generates nothing, loads no flow, needs no GPU.

One command that answers the questions that must be settled before any result is
believed or any new generation is launched:

  completeness   every expected cell/repetition present, row counts equal to the budget,
                 status complete, settings fingerprint matching
  provenance     manifests present; decoder sha256 and data representation identical
                 across arms; arm labels correct
  denominators   recomputed from the saved attempts and compared with the stored metrics
  slopes         recomputed per N and PER SAMPLING REPETITION, not pooled only
  rings          generated and reference ring counts recomputed with ONE RDKit
                 definition, reporting NumRings, aromatic rings and ring-system counts
                 side by side, with sanitization behaviour stated
  reference      what "20,000 scanned" actually selected, and whether it is
                 representative of the sizes the models were asked for
  chemistry      size-matched AND size-plus-QED-matched references, sparse cells labelled
  grids          representative and extreme-ring molecules for both models and the real
                 reference, chosen by rank so nothing is cherry-picked
  slope_0.7      searches the supplied directories for any artefact recording a QED
                 slope near 0.7; reports UNTRACEABLE if none exists

Exit codes: 0 audit ran (findings may still be negative); 2 nothing to audit; 4 bad
arguments; 1 unexpected.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zinc_gnf.eval_support import (  # noqa: E402
    ARM_LABELS,
    DEFAULT_TOLERANCES,
    qed_cell_metrics,
    read_jsonl,
    response_fit,
    run_level_summary,
)

RING_NOTE = (
    "Ring counting is ambiguous and the ambiguity explains large apparent differences. "
    "This audit reports three quantities computed with ONE code path for generated and "
    "reference molecules alike: (a) rdkit RingInfo.NumRings(), the SSSR count, which "
    "counts every smallest ring and so grows quickly in fused/caged graphs; "
    "(b) aromatic rings, counted as SSSR rings whose every atom has GetIsAromatic(); "
    "(c) ring systems, the number of connected components of the ring-bond subgraph, "
    "which is closer to what a chemist means by 'rings' in a fused system. All three "
    "are computed AFTER Chem.MolFromSmiles, which sanitizes and perceives aromaticity, "
    "so both populations receive identical treatment.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/cluster.json")
    parser.add_argument("--target-set", default="grid")
    parser.add_argument("--decoder", default=None)
    parser.add_argument("--expected-attempts", type=int, default=None)
    parser.add_argument("--expected-repetitions", type=int, default=None)
    parser.add_argument("--reference-molecules", type=int, default=20000)
    parser.add_argument("--grid-molecules", type=int, default=9,
                        help="molecules per panel in the molecule grids")
    parser.add_argument("--search-dirs", nargs="*", default=None,
                        help="extra directories to search for the old slope~0.7 artefact")
    parser.add_argument("--no-grids", action="store_true")
    return parser.parse_args(argv)


def write_csv(path, rows):
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    names = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def ring_descriptors(smiles):
    """One code path for every molecule, generated or real."""
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)          # sanitizes + perceives aromaticity
    if molecule is None:
        return None
    from rdkit.Chem import Crippen, Descriptors

    info = molecule.GetRingInfo()
    sssr = int(info.NumRings())
    aromatic = int(sum(1 for ring in info.AtomRings()
                       if all(molecule.GetAtomWithIdx(i).GetIsAromatic() for i in ring)))
    ring_atoms = {i for ring in info.AtomRings() for i in ring}
    seen, systems = set(), 0
    for start in ring_atoms:
        if start in seen:
            continue
        systems += 1
        stack = [start]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            for bond in molecule.GetAtomWithIdx(current).GetBonds():
                if not bond.IsInRing():
                    continue
                other = bond.GetOtherAtomIdx(current)
                if other not in seen:
                    stack.append(other)
    return {"heavy_atoms": int(molecule.GetNumHeavyAtoms()),
            "rings_sssr": sssr, "aromatic_rings": aromatic, "ring_systems": systems,
            "molecular_weight": float(Descriptors.MolWt(molecule)),
            "logp": float(Crippen.MolLogP(molecule)),
            "qed_recomputed": _qed(molecule)}


def _qed(molecule):
    try:
        from rdkit.Chem import QED

        return float(QED.qed(molecule))
    except Exception:  # noqa: BLE001
        return None


def statistics(values):
    values = sorted(float(v) for v in values if v is not None)
    if not values:
        return {}
    mean = sum(values) / len(values)

    def quantile(q):
        if len(values) == 1:
            return values[0]
        position = q * (len(values) - 1)
        low = int(position)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (position - low)

    return {"n": len(values), "mean": mean,
            "sd": ((sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
                   if len(values) > 1 else 0.0),
            "min": values[0], "q05": quantile(0.05), "median": quantile(0.5),
            "q95": quantile(0.95), "max": values[-1]}


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        config = json.loads(Path(args.config).read_text())
        paths, run = config["paths"], config.get("run", {})
        decoder = args.decoder or run.get("headline_decoder", "degree")
        tolerances = tuple(run.get("tolerances", DEFAULT_TOLERANCES))
        expected_attempts = args.expected_attempts or run.get("attempts", 2000)
        expected_reps = args.expected_repetitions or run.get("repetitions", 5)
        results = Path(paths["results_dir"])
        root = results / "conditional" / args.target_set
        out = results / "audit" / args.target_set
        out.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError, KeyError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 4

    try:
        if not root.is_dir():
            print(f"FATAL: no artefacts under {root}", file=sys.stderr)
            return 2
        findings: dict[str, object] = {"root": str(root), "ring_note": RING_NOTE,
                                       "comparison_label": ARM_LABELS["comparison"]}

        # ---------------- 1. completeness ----------------
        completeness, cells_seen = [], {}
        for metrics_file in sorted(root.glob("*/*/rep*/metrics.json")):
            model = metrics_file.parents[2].name
            cell = metrics_file.parents[1].name
            payload = json.loads(metrics_file.read_text())
            samples = metrics_file.parent / "samples.jsonl"
            rows = (sum(1 for line in samples.open() if line.strip())
                    if samples.is_file() else 0)
            problems = []
            if payload.get("status") != "complete":
                problems.append(f"status={payload.get('status')!r}")
            if int(payload.get("attempted", -1)) != expected_attempts:
                problems.append(f"attempted={payload.get('attempted')}")
            if rows != expected_attempts:
                problems.append(f"samples_rows={rows}")
            if payload.get("artefact_fingerprint") is None:
                problems.append("no settings fingerprint (written before the fix)")
            cells_seen.setdefault(model, set()).add(cell)
            completeness.append({
                "model": model, "cell": cell,
                "repetition": payload.get("repetition"),
                "preset": payload.get("preset"),
                "attempted_recorded": payload.get("attempted"),
                "samples_rows": rows, "expected_attempts": expected_attempts,
                "status": payload.get("status"),
                "fingerprint": payload.get("artefact_fingerprint"),
                "sampling_seed": payload.get("sampling_seed"),
                "temperature": payload.get("temperature"),
                "ok": not problems, "problems": "; ".join(problems) or None})
        findings["cells_per_model"] = {m: len(c) for m, c in cells_seen.items()}
        findings["repetitions_expected"] = expected_reps
        findings["incomplete_cells"] = sum(1 for r in completeness if not r["ok"])
        write_csv(out / "completeness.csv", completeness)

        # ---------------- 2. provenance ----------------
        provenance = []
        for manifest_file in sorted(root.glob("manifest.json")):
            payload = json.loads(manifest_file.read_text())
            for model, block in (payload.get("models") or {}).items():
                flow_config = block.get("flow_config") or {}
                provenance.append({
                    "model": model, "label_recorded": block.get("label"),
                    "label_expected": ARM_LABELS.get(model),
                    "flow_checkpoint": block.get("flow_checkpoint"),
                    "flow_sha256": block.get("flow_sha256"),
                    "decoder_checkpoint": block.get("decoder_checkpoint"),
                    "decoder_sha256": block.get("decoder_sha256"),
                    "training_seed": block.get("training_seed"),
                    "condition_features": block.get("condition_features"),
                    "node_embedding_dim": flow_config.get("node_embedding_dim"),
                    "num_blocks": flow_config.get("num_blocks"),
                    "node_autoregressive": flow_config.get("node_autoregressive"),
                    "temperature": block.get("temperature"),
                    "attempts": block.get("attempts"),
                    "repetitions": block.get("repetitions"),
                    "data_representation": (block.get("data_fingerprint") or {})
                    .get("representation"),
                    "data_counts_digest": (block.get("data_fingerprint") or {})
                    .get("counts_digest")})
        decoder_hashes = {r["decoder_sha256"] for r in provenance if r["decoder_sha256"]}
        representations = {r["data_representation"] for r in provenance}
        digests = {r["data_counts_digest"] for r in provenance}
        architecture = {(r["node_embedding_dim"], r["num_blocks"],
                         r["node_autoregressive"]) for r in provenance}
        findings["provenance_checks"] = {
            "one_decoder_across_arms": len(decoder_hashes) <= 1,
            "decoder_sha256_values": sorted(h for h in decoder_hashes if h),
            "one_data_representation": len(representations) <= 1,
            "data_representations": sorted(str(r) for r in representations),
            "one_data_counts_digest": len(digests) <= 1,
            "one_architecture_across_arms": len(architecture) <= 1,
            "architectures_seen": [list(a) for a in sorted(architecture, key=str)],
            "note": ("a differing architecture means the comparison is not conditioning "
                     "alone; a differing decoder hash means the two arms were decoded by "
                     "different models")}
        write_csv(out / "provenance.csv", provenance)

        # ---------------- 3-5. recompute from saved attempts ----------------
        recomputed, points, chem_rows = [], [], []
        by_model_n_rep: dict[tuple, list] = {}
        by_cell_records: dict[tuple, list] = {}
        for samples_file in sorted(root.glob("*/*/rep*/samples.jsonl")):
            model = samples_file.parents[2].name
            records = read_jsonl(samples_file)
            if not records:
                continue
            requested_n = int(records[0]["requested_n"])
            requested_qed = float(records[0]["requested_qed"])
            repetition = int(records[0].get("repetition", -1))
            fresh = qed_cell_metrics(records, requested_qed=requested_qed,
                                     requested_n=requested_n, decoder=decoder,
                                     tolerances=tolerances, train_smiles=set())
            stored_path = samples_file.parent / "metrics.json"
            stored = (json.loads(stored_path.read_text())
                      if stored_path.is_file() else {})
            row = {"model": model, "cell": samples_file.parents[1].name,
                   "repetition": repetition, "requested_n": requested_n,
                   "requested_qed": requested_qed}
            for field in ("attempted", "valid", "scored", "validity_over_attempts",
                          "qed_mae", "qed_bias", "achieved_qed_mean",
                          "uniqueness_over_valid",
                          f"valid_hit_yield_within_0p1_over_all_attempts"):
                row[f"recomputed_{field}"] = fresh.get(field)
                row[f"stored_{field}"] = stored.get(field)
                left, right = fresh.get(field), stored.get(field)
                row[f"match_{field}"] = (
                    None if left is None or right is None
                    else abs(float(left) - float(right)) <= 1e-9)
            recomputed.append(row)
            by_cell_records[(model, requested_n, requested_qed, repetition)] = records
            for record in records:
                entry = record.get(decoder) or {}
                if entry.get("valid") and entry.get("qed") is not None:
                    points.append((model, requested_n, repetition, requested_qed,
                                   float(entry["qed"])))
                    by_model_n_rep.setdefault((model, requested_n, repetition),
                                              []).append(
                        (requested_qed, float(entry["qed"])))
        write_csv(out / "denominator_recheck.csv", recomputed)
        findings["denominator_mismatches"] = sum(
            1 for r in recomputed for k, v in r.items()
            if k.startswith("match_") and v is False)

        # slopes: per (model, N, repetition), then summarised across repetitions
        slope_rows = []
        for (model, n_value, repetition), pairs in sorted(by_model_n_rep.items()):
            fit = response_fit([p[0] for p in pairs], [p[1] for p in pairs])
            slope_rows.append({"model": model, "level": f"N={n_value}",
                               "repetition": repetition, "valid_samples": len(pairs),
                               **{k: v for k, v in fit.items() if k != "note"}})
        for model in sorted({p[0] for p in points}):
            for n_value in sorted({p[1] for p in points if p[0] == model}):
                values = [r["slope"] for r in slope_rows
                          if r["model"] == model and r["level"] == f"N={n_value}"
                          and r["slope"] is not None]
                if values:
                    summary = run_level_summary(values)
                    slope_rows.append({"model": model, "level": f"N={n_value}",
                                       "repetition": "ACROSS_REPS",
                                       "slope": summary["mean"],
                                       "slope_sd_across_repetitions": summary["sd"],
                                       "repetitions": summary["runs"]})
            pooled = [(p[3], p[4]) for p in points if p[0] == model]
            if pooled:
                fit = response_fit([p[0] for p in pooled], [p[1] for p in pooled])
                slope_rows.append({"model": model, "level": "POOLED over all N",
                                   "repetition": "ALL", "valid_samples": len(pooled),
                                   **{k: v for k, v in fit.items() if k != "note"}})
        write_csv(out / "slopes_by_n_and_repetition.csv", slope_rows)
        findings["slope_note"] = (
            "a pooled slope mixes sizes; within-N slopes are the ones the ZINC N-QED "
            "association cannot explain. The per-repetition spread shows whether a "
            "within-N slope is stable or an artefact of one sample")

        # ---------------- 6. rings, one definition, both populations ----------------
        generated_chem: dict[tuple, dict[str, list]] = {}
        extremes: dict[str, list] = {}
        try:
            for (model, n_value, qed_value, _), records in by_cell_records.items():
                for record in records:
                    entry = record.get(decoder) or {}
                    if not entry.get("valid") or not entry.get("smiles"):
                        continue
                    descriptors = ring_descriptors(entry["smiles"])
                    if descriptors is None:
                        continue
                    bucket = generated_chem.setdefault(
                        (model, int(descriptors["heavy_atoms"])), {})
                    for key, value in descriptors.items():
                        bucket.setdefault(key, []).append(value)
                    extremes.setdefault(model, []).append(
                        {"smiles": entry["smiles"], "requested_n": n_value,
                         "requested_qed": qed_value,
                         "stored_rings": entry.get("rings"), **descriptors})
            rdkit_available = True
        except ImportError as error:
            rdkit_available = False
            findings["rdkit_error"] = str(error)

        reference_chem: dict[int, dict[str, list]] = {}
        reference_audit = {"status": "not attempted"}
        if rdkit_available:
            try:
                from zinc_gnf.dataset import load_training_splits

                train_records, _, fingerprint = load_training_splits(paths["data_dir"])
                wanted_sizes = {size for _, size in generated_chem}
                requested_sizes = sorted({p[1] for p in points})
                scanned = kept = 0
                per_size_cap = max(1, args.reference_molecules // max(len(wanted_sizes), 1))
                counts_by_size: dict[int, int] = {}
                reference_qed: dict[int, list] = {}
                for record in train_records:
                    scanned += 1
                    descriptors = ring_descriptors(str(record.get("smiles", "")))
                    if descriptors is None:
                        continue
                    size = int(descriptors["heavy_atoms"])
                    if wanted_sizes and size not in wanted_sizes:
                        continue
                    if counts_by_size.get(size, 0) >= per_size_cap:
                        continue
                    counts_by_size[size] = counts_by_size.get(size, 0) + 1
                    kept += 1
                    bucket = reference_chem.setdefault(size, {})
                    for key, value in descriptors.items():
                        bucket.setdefault(key, []).append(value)
                    reference_qed.setdefault(size, []).append(
                        float(record.get("qed", float("nan"))))
                    if kept >= args.reference_molecules:
                        break
                reference_audit = {
                    "status": "computed",
                    "meaning_of_scanned": (
                        f"{scanned} training records were READ in dataset order and "
                        f"{kept} were KEPT. A record is kept only if its heavy-atom "
                        f"count is one the models actually produced and that size's "
                        f"quota is not full."),
                    "records_read": scanned, "records_kept": kept,
                    "per_size_cap": per_size_cap,
                    "kept_by_size": dict(sorted(counts_by_size.items())),
                    "sizes_requested_of_the_models": requested_sizes,
                    "selection_is_first_n_in_dataset_order": True,
                    "representativeness_warning": (
                        "selection takes the FIRST matching records in dataset order, "
                        "not a random sample, and caps each size equally. Equal caps "
                        "make the reference size-balanced, which is what a size-matched "
                        "comparison needs, but it is NOT the dataset's own size "
                        "distribution. Any statement about 'real ZINC' must say which "
                        "of the two is meant."),
                    "data_representation": fingerprint.get("representation"),
                    "total_training_molecules": len(train_records)}
                findings["reference_qed_by_size"] = {
                    str(size): statistics(values)
                    for size, values in sorted(reference_qed.items())}
            except Exception as error:  # noqa: BLE001
                reference_audit = {"status": f"skipped: {type(error).__name__}: {error}"}
        findings["reference_audit"] = reference_audit

        for (model, size), fields in sorted(generated_chem.items()):
            for field in ("rings_sssr", "aromatic_rings", "ring_systems",
                          "molecular_weight", "logp", "qed_recomputed"):
                if field not in fields:
                    continue
                row = {"model": model, "heavy_atoms": size, "property": field,
                       "match": "size-only"}
                row.update({f"generated_{k}": v
                            for k, v in statistics(fields[field]).items()})
                real = (reference_chem.get(size) or {}).get(field)
                if real:
                    row.update({f"reference_{k}": v for k, v in statistics(real).items()})
                    row["difference_of_means"] = (row.get("generated_mean", 0)
                                                  - row.get("reference_mean", 0))
                    row["reference_sparse"] = len(real) < 30
                chem_rows.append(row)
        write_csv(out / "chemistry_size_matched.csv", chem_rows)

        # size AND QED matched, where support allows
        matched_rows = []
        if rdkit_available and reference_chem:
            for (model, n_value, qed_value, _), records in sorted(by_cell_records.items()):
                reference_values = [
                    v for size, values in (findings.get("reference_qed_by_size") or {}).items()
                    for v in [values] if abs(int(size) - int(n_value)) <= 1]
                generated_rings = [
                    ring_descriptors(e["smiles"])["rings_sssr"]
                    for e in (r.get(decoder) or {} for r in records)
                    if e.get("valid") and e.get("smiles")
                    and ring_descriptors(e["smiles"]) is not None]
                reference_rings = [
                    v for size, fields in reference_chem.items()
                    if abs(size - int(n_value)) <= 1
                    for v, q in zip(fields.get("rings_sssr", []),
                                    fields.get("qed_recomputed", []))
                    if q is not None and abs(q - float(qed_value)) <= 0.05]
                row = {"model": model, "requested_n": n_value,
                       "requested_qed": qed_value, "property": "rings_sssr",
                       "match": "size+QED (+/-1 atom, +/-0.05 QED)",
                       "reference_molecules": len(reference_rings),
                       "reference_sparse": len(reference_rings) < 30,
                       "reference_qed_available": bool(reference_values)}
                row.update({f"generated_{k}": v
                            for k, v in statistics(generated_rings).items()})
                if reference_rings:
                    row.update({f"reference_{k}": v
                                for k, v in statistics(reference_rings).items()})
                matched_rows.append(row)
        write_csv(out / "chemistry_size_and_qed_matched.csv", matched_rows)

        # ---------------- 7. molecule grids, rank-selected ----------------
        grid_rows = []
        if rdkit_available and not args.no_grids:
            for model, entries in sorted(extremes.items()):
                entries.sort(key=lambda e: e["rings_sssr"])
                count = min(args.grid_molecules, len(entries))
                if count == 0:
                    continue
                picks = {
                    "median_rings": [entries[int(len(entries) * q)]
                                     for q in [0.5 - 0.02 * i for i in
                                               range(count // 2, -(count - count // 2), -1)]
                                     if 0 <= int(len(entries) * q) < len(entries)][:count],
                    "max_rings": entries[-count:],
                    "min_rings": entries[:count]}
                for panel, chosen in picks.items():
                    for rank, item in enumerate(chosen):
                        grid_rows.append({"model": model, "panel": panel, "rank": rank,
                                          "selection": "by ring-count rank, not by eye",
                                          **item})
            for size, fields in sorted(reference_chem.items()):
                order = sorted(range(len(fields.get("rings_sssr", []))),
                               key=lambda i: fields["rings_sssr"][i])
                for panel, indices in (("median_rings",
                                        order[len(order) // 2: len(order) // 2 + 3]),
                                       ("max_rings", order[-3:]),
                                       ("min_rings", order[:3])):
                    for rank, index in enumerate(indices):
                        grid_rows.append({
                            "model": "real_reference", "panel": panel, "rank": rank,
                            "heavy_atoms": size,
                            "rings_sssr": fields["rings_sssr"][index],
                            "aromatic_rings": fields.get("aromatic_rings", [None])[index]
                            if index < len(fields.get("aromatic_rings", [])) else None,
                            "selection": "by ring-count rank, not by eye"})
            write_csv(out / "molecule_grid_members.csv", grid_rows)
            try:
                from rdkit import Chem
                from rdkit.Chem import Draw

                figures = out / "figures"
                figures.mkdir(parents=True, exist_ok=True)
                for model in sorted({r["model"] for r in grid_rows}):
                    for panel in ("min_rings", "median_rings", "max_rings"):
                        members = [r for r in grid_rows
                                   if r["model"] == model and r["panel"] == panel
                                   and r.get("smiles")]
                        if not members:
                            continue
                        molecules = [Chem.MolFromSmiles(r["smiles"]) for r in members]
                        legends = [f"N={r.get('heavy_atoms')} SSSR={r.get('rings_sssr')} "
                                   f"arom={r.get('aromatic_rings')} "
                                   f"QED={r.get('qed_recomputed'):.2f}"
                                   if r.get("qed_recomputed") is not None else
                                   f"N={r.get('heavy_atoms')}" for r in members]
                        image = Draw.MolsToGridImage(
                            [m for m in molecules if m is not None],
                            molsPerRow=3, subImgSize=(260, 260),
                            legends=legends[:len([m for m in molecules if m])])
                        path = figures / f"grid_{model}_{panel}.png"
                        data = getattr(image, "data", None)
                        if data:
                            path.write_bytes(data)
                        else:
                            image.save(str(path))
            except Exception as error:  # noqa: BLE001
                findings["grid_images"] = f"skipped: {type(error).__name__}: {error}"

        # ---------------- 8. the old slope ~0.7 ----------------
        search = [results, Path(paths.get("baseline_output_dir", "outputs"))]
        search += [Path(d) for d in (args.search_dirs or [])]
        hits = []
        for directory in search:
            if not directory.is_dir():
                continue
            for found in list(directory.rglob("*.json")) + list(directory.rglob("*.csv")):
                try:
                    text = found.read_text(errors="ignore")
                except OSError:
                    continue
                for needle in ("0.7", "0.69", "0.70", "0.71", "0.72", "0.68"):
                    if f"slope" in text.lower() and needle in text:
                        hits.append({"file": str(found), "needle": needle,
                                     "size_bytes": found.stat().st_size})
                        break
        findings["old_slope_0p7"] = {
            "files_mentioning_a_slope_near_0.7": hits[:20],
            "verdict": ("UNTRACEABLE in the supplied directories: no artefact recording "
                        "a QED slope near 0.7 was found. It is not reconstructed or "
                        "guessed here." if not hits else
                        "candidate files listed; open them to confirm what the slope was "
                        "computed over (per-sample, per-target mean, within-N or pooled) "
                        "before comparing it with the current numbers"),
            "searched": [str(d) for d in search]}

        findings["condition_gap_caveat"] = (
            "A shuffled-(N,QED) likelihood gap permutes BOTH conditions, and N and QED "
            "are associated in ZINC, so that gap is not evidence about QED alone. Only a "
            "QED-only permutation that keeps each molecule's true N - or a within-size "
            "shuffle - speaks to QED.")

        (out / "audit.json").write_text(json.dumps(findings, indent=2, default=str))
        print(json.dumps({k: findings[k] for k in (
            "cells_per_model", "incomplete_cells", "denominator_mismatches",
            "provenance_checks", "old_slope_0p7")}, indent=2, default=str)[:4000])
        print(f"\nwrote {out}/audit.json and the CSVs beside it")
        if isinstance(reference_audit, dict) and reference_audit.get("records_kept"):
            print(f"\nreference selection: read {reference_audit['records_read']}, kept "
                  f"{reference_audit['records_kept']}, per-size cap "
                  f"{reference_audit['per_size_cap']}, by size "
                  f"{reference_audit['kept_by_size']}")
        return 0
    except Exception:  # noqa: BLE001
        print("FATAL: unexpected\n" + traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
