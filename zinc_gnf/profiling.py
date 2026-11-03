"""CPU profiler for the existing ZINC250K preprocessing path (Phase 0).

Question it answers: *why might the full preprocessing job be slow or time out
on an ISML compute node, and how long should the full run be expected to
take there?*

It measures, on a deterministic seeded sample of CSV rows:

  * RDKit parsing, canonicalization, kekulization, graph reconstruction, QED,
    and total time of the REAL `zinc_gnf.chemistry.encode_zinc_smiles`
    (phase timing wraps the production functions with timing proxies and
    always restores them; nothing is re-implemented);
  * clean (uninstrumented) per-molecule latency and throughput, including
    tail latencies and the slowest molecules;
  * the surrounding pipeline costs: full-CSV read, SHA-256 hashing, split,
    scaler fit, pickle write/read (to one or more scratch directories, so a
    network home directory can be compared with node-local storage);
  * a machine-speed calibration that is identical on every machine, so a
    sandbox number and a cluster number can be compared directly;
  * the CPUs and memory the job actually received.

It then extrapolates to the full dataset. It NEVER writes the production
processed dataset: all outputs go to a diagnostics directory that is checked
to be outside `data.processed_dir`, and scratch files are deleted.
"""

from __future__ import annotations

from array import array
from contextlib import contextmanager
import math
import os
from pathlib import Path
import pickle
import shutil
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Sequence

import numpy as np
from rdkit import Chem, rdBase

from zinc_gnf import chemistry
from zinc_gnf.diagnostics import (
    EXPECTED_ZINC250K_ROWS,
    EXPECTED_ZINC250K_SHA256,
    ZincTable,
    deterministic_sample_indices,
    indices_fingerprint,
    read_zinc_table,
    verify_dataset_identity,
)
from zinc_gnf.constants import MAX_NODES, MIN_NODES, REPRESENTATION_VERSION
from zinc_gnf.preprocessing import ConditionScaler, hash_file, random_split
from zinc_gnf.utils.runtime import (
    StopFlag,
    assert_output_outside,
    atomic_write_json,
    collect_environment,
    current_rss_mb,
    format_duration,
    peak_rss_mb,
    usable_cpu_count,
)

SCHEMA_VERSION = 1
PHASES = ("parse", "canonicalization", "kekulize", "reconstruction", "qed")
SAMPLE_RANGE_RECOMMENDED = (2000, 5000)
DEFAULT_SAMPLE_SIZE = 3000
WARMUP_MOLECULES = 50
SLOWEST_TO_REPORT = 5

# Fixed molecules for the cross-machine speed calibration (never read from
# the dataset, so the workload is identical everywhere).
CALIBRATION_SMILES = (
    "CCO", "c1ccccc1", "CC(=O)Nc1ccc(O)cc1", "O=C(Nc1ccc(F)cc1)c1cc(Cl)ccc1Br",
    "CN1CCC[C@H]1c1cccnc1", "C[NH3+]", "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "O=C1CCCN1c1ccc(S(=O)(=O)N2CCOCC2)cc1", "Brc1ccc2[nH]ccc2c1",
    "CC1=C(C(=O)OC)C(c2ccccc2[N+](=O)[O-])C(C(=O)OC)=C(C)N1",
)


# ---------------------------------------------------------------------------
# Phase timing by wrapping the real chemistry functions.
# ---------------------------------------------------------------------------
class PhaseTimer:
    """Accumulates wall time and call counts per named phase."""

    def __init__(self) -> None:
        self.seconds = {phase: 0.0 for phase in PHASES}
        self.calls = {phase: 0 for phase in PHASES}

    def reset(self) -> None:
        for phase in PHASES:
            self.seconds[phase] = 0.0
            self.calls[phase] = 0

    def wrap(self, phase: str, function: Callable[..., Any]) -> Callable[..., Any]:
        seconds, calls = self.seconds, self.calls

        def timed(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                seconds[phase] += time.perf_counter() - started
                calls[phase] += 1

        return timed


class _ChemProxy:
    """Delegates to `rdkit.Chem`, timing only parse and kekulize calls."""

    def __init__(self, original: Any, timer: PhaseTimer) -> None:
        self._original = original
        self._timer = timer
        self._wrapped = {
            "MolFromSmiles": timer.wrap("parse", original.MolFromSmiles),
            "Kekulize": timer.wrap("kekulize", original.Kekulize),
        }

    def __getattr__(self, name: str) -> Any:
        wrapped = self._wrapped.get(name)
        if wrapped is not None:
            return wrapped
        return getattr(self._original, name)


@contextmanager
def instrument_chemistry(timer: PhaseTimer) -> Iterator[None]:
    """Temporarily time the real functions `encode_zinc_smiles` calls.

    Patches module globals of `zinc_gnf.chemistry` only, and restores them in
    `finally`, so an exception can never leave the production module patched.
    """
    saved = {
        "Chem": chemistry.Chem,
        "QED": chemistry.QED,
        "canonical_connectivity_smiles": chemistry.canonical_connectivity_smiles,
        "graph_to_mol": chemistry.graph_to_mol,
    }
    try:
        chemistry.Chem = _ChemProxy(saved["Chem"], timer)
        chemistry.QED = SimpleNamespace(
            qed=timer.wrap("qed", saved["QED"].qed)
        )
        chemistry.canonical_connectivity_smiles = timer.wrap(
            "canonicalization", saved["canonical_connectivity_smiles"],
        )
        chemistry.graph_to_mol = timer.wrap(
            "reconstruction", saved["graph_to_mol"],
        )
        yield
    finally:
        chemistry.Chem = saved["Chem"]
        chemistry.QED = saved["QED"]
        chemistry.canonical_connectivity_smiles = saved[
            "canonical_connectivity_smiles"
        ]
        chemistry.graph_to_mol = saved["graph_to_mol"]


# ---------------------------------------------------------------------------
# Progress reporting.
# ---------------------------------------------------------------------------
class ProgressReporter:
    """Timestamped, flushed progress lines (count- and time-triggered)."""

    def __init__(
        self,
        logger: Any,
        label: str,
        total: int,
        *,
        every: int,
        heartbeat_seconds: float,
    ) -> None:
        self.logger = logger
        self.label = label
        self.total = total
        self.every = max(1, every)
        self.heartbeat = heartbeat_seconds
        self.started = time.perf_counter()
        self.last = self.started

    def update(self, done: int, extra: str = "") -> None:
        now = time.perf_counter()
        if (
            done % self.every == 0
            or now - self.last >= self.heartbeat
            or done == self.total
        ):
            elapsed = now - self.started
            rate = done / elapsed if elapsed > 0 else float("nan")
            remaining = (self.total - done) / rate if rate and rate > 0 else None
            self.logger.info(
                f"{self.label} {done:,}/{self.total:,} "
                f"({100.0 * done / max(1, self.total):.1f}%) "
                f"rate={rate:,.1f} mol/s elapsed={format_duration(elapsed)} "
                f"eta={format_duration(remaining)} "
                f"rss_mb={current_rss_mb()}{extra}"
            )
            self.last = now


# ---------------------------------------------------------------------------
# Statistics helpers.
# ---------------------------------------------------------------------------
def latency_statistics(seconds: Sequence[float] | np.ndarray) -> dict[str, Any]:
    data = np.asarray(seconds, dtype=np.float64)
    if data.size == 0:
        return {"n": 0}
    return {
        "n": int(data.size),
        "mean_ms": float(data.mean() * 1e3),
        "std_ms": float(data.std() * 1e3),
        "median_ms": float(np.median(data) * 1e3),
        "p90_ms": float(np.quantile(data, 0.90) * 1e3),
        "p99_ms": float(np.quantile(data, 0.99) * 1e3),
        "max_ms": float(data.max() * 1e3),
        "total_seconds": float(data.sum()),
    }


def extrapolate(
    per_item_seconds: np.ndarray, population: int,
) -> dict[str, float]:
    """Scale a sample's per-item time to `population` items.

    The interval is the normal-approximation 95% confidence interval of the
    sample mean; it reflects sampling noise only, not machine-load changes.
    """
    data = np.asarray(per_item_seconds, dtype=np.float64)
    if data.size == 0:
        return {"estimate_seconds": float("nan")}
    mean = float(data.mean())
    sem = float(data.std(ddof=1) / math.sqrt(data.size)) if data.size > 1 else 0.0
    return {
        "estimate_seconds": mean * population,
        "ci95_low_seconds": max(0.0, (mean - 1.96 * sem) * population),
        "ci95_high_seconds": (mean + 1.96 * sem) * population,
    }


def suggest_time_limit(estimated_seconds: float, *, safety: float = 2.0) -> str:
    """Round `safety * estimate` up to a quarter hour, as HH:MM:SS (max 24h)."""
    if not math.isfinite(estimated_seconds):
        return "n/a"
    seconds = max(900.0, estimated_seconds * safety)
    seconds = math.ceil(seconds / 900.0) * 900
    seconds = min(seconds, 24 * 3600)
    hours, remainder = divmod(int(seconds), 3600)
    minutes = remainder // 60
    return f"{hours:02d}:{minutes:02d}:00"


# ---------------------------------------------------------------------------
# Machine-speed calibration (same workload on every machine).
# ---------------------------------------------------------------------------
def machine_calibration(
    *, repetitions: int = 200, python_loop: int = 1_000_000,
) -> dict[str, Any]:
    started = time.perf_counter()
    accumulator = 0
    for value in range(python_loop):
        accumulator += value * value
    python_seconds = time.perf_counter() - started

    molecules = 0
    started = time.perf_counter()
    with rdBase.BlockLogs():
        for _ in range(repetitions):
            for smiles in CALIBRATION_SMILES:
                molecule = Chem.MolFromSmiles(smiles)
                if molecule is not None:
                    Chem.MolToSmiles(molecule)
                    molecules += 1
    rdkit_seconds = time.perf_counter() - started
    return {
        "python_loop_iterations": python_loop,
        "python_loop_seconds": python_seconds,
        "python_loop_checksum": accumulator,
        "rdkit_parse_and_write_molecules": molecules,
        "rdkit_parse_and_write_seconds": rdkit_seconds,
        "rdkit_parse_and_write_per_second": (
            molecules / rdkit_seconds if rdkit_seconds > 0 else float("nan")
        ),
        "note": (
            "Identical fixed workload on every machine; compare these "
            "numbers between the sandbox and a compute node to see how much "
            "slower the node is, independent of the dataset."
        ),
    }


# ---------------------------------------------------------------------------
# Pickle I/O benchmark.
# ---------------------------------------------------------------------------
def benchmark_pickle(
    payload: Any, scratch_dir: Path, *, label: str,
) -> dict[str, Any]:
    """Write then read `payload` under a fresh subdirectory, then delete it."""
    scratch_dir.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="zinc_profile_", dir=str(scratch_dir)))
    target = workdir / "profile_payload.pkl"
    result: dict[str, Any] = {"label": label, "directory": str(scratch_dir)}
    try:
        started = time.perf_counter()
        with target.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        result["write_seconds"] = time.perf_counter() - started
        result["bytes"] = target.stat().st_size

        started = time.perf_counter()
        with target.open("rb") as handle:
            pickle.load(handle)
        result["read_seconds"] = time.perf_counter() - started
    except OSError as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return result


# ---------------------------------------------------------------------------
# Passes over the sample.
# ---------------------------------------------------------------------------
def _encode_pass(
    smiles_list: Sequence[str],
    *,
    min_nodes: int,
    max_nodes: int,
    logger: Any,
    label: str,
    every: int,
    heartbeat_seconds: float,
    stop_flag: StopFlag,
    keep_records: bool,
) -> dict[str, Any]:
    latencies = array("d")
    outcomes: list[str] = []
    records: list[dict[str, Any]] = []
    progress = ProgressReporter(
        logger, label, len(smiles_list),
        every=every, heartbeat_seconds=heartbeat_seconds,
    )
    started = time.perf_counter()
    for position, smiles in enumerate(smiles_list, start=1):
        began = time.perf_counter()
        record, reason = chemistry.encode_zinc_smiles(
            smiles, min_nodes=min_nodes, max_nodes=max_nodes,
        )
        latencies.append(time.perf_counter() - began)
        outcomes.append("accepted" if record is not None else (reason or "unknown"))
        if keep_records and record is not None:
            records.append(record)
        progress.update(position)
        if stop_flag.requested:
            logger.warn(f"{label}: stopping early ({stop_flag.signal_name})")
            break
    return {
        "latencies": np.frombuffer(latencies, dtype=np.float64).copy(),
        "outcomes": outcomes,
        "records": records,
        "wall_seconds": time.perf_counter() - started,
    }


def _outcome_breakdown(
    latencies: np.ndarray, outcomes: Sequence[str],
) -> dict[str, Any]:
    groups: dict[str, list[float]] = {}
    for seconds, outcome in zip(latencies, outcomes):
        groups.setdefault(outcome, []).append(float(seconds))
    return {
        outcome: {
            "count": len(values),
            "mean_ms": float(np.mean(values) * 1e3),
            "max_ms": float(np.max(values) * 1e3),
        }
        for outcome, values in sorted(groups.items(), key=lambda kv: -len(kv[1]))
    }


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------
def summary_lines(report: dict[str, Any]) -> list[str]:
    sample = report.get("sample")
    clean = report.get("clean_pass")
    lines = [
        "ZINC250K preprocessing profile summary",
        f"  complete run:           {report['complete']}",
        f"  host / usable CPUs:     {report['environment']['hostname']} / "
        f"{report['environment']['usable_cpu_count']}",
        f"  cpu model:              {report['environment']['cpu_model']}",
    ]
    if sample:
        lines.append(
            f"  sample:                 {sample['size']:,} of "
            f"{sample['population']:,} rows (seed={sample['seed']}, "
            f"fingerprint={sample['index_fingerprint']})"
        )
    if clean:
        lines += [
            f"  clean throughput:       {clean['throughput_mol_per_s']:,.1f} mol/s "
            f"(mean {clean['latency']['mean_ms']:.3f} ms, "
            f"p99 {clean['latency']['p99_ms']:.2f} ms, "
            f"max {clean['latency']['max_ms']:.1f} ms)",
        ]
    breakdown = report.get("phase_breakdown")
    if breakdown:
        lines.append("  phase share of instrumented total:")
        for phase, block in breakdown["phases"].items():
            lines.append(
                f"    {phase:<16} {100 * block['share_of_total']:5.1f}% "
                f"({block['mean_ms_per_molecule']:.3f} ms/mol, "
                f"{block['calls_per_molecule']:.2f} calls/mol)"
            )
        lines.append(
            f"    {'other':<16} {100 * breakdown['other_share']:5.1f}%"
        )
    estimate = report.get("full_dataset_estimate")
    if estimate:
        lines += [
            "  estimated FULL-dataset serial run on this machine:",
            f"    encoding only:        "
            f"{format_duration(estimate['encoding']['estimate_seconds'])} "
            f"(95% CI {format_duration(estimate['encoding'].get('ci95_low_seconds'))}"
            f" to {format_duration(estimate['encoding'].get('ci95_high_seconds'))})",
            f"    whole preprocessing:  "
            f"{format_duration(estimate['total_serial_seconds'])}",
            f"    fits a 1 h limit:     {estimate['fits_1h_serial']}",
            f"    suggested --time:     {estimate['suggested_time_limit']}",
            f"    ideal on {estimate['usable_cpus']} CPUs:       "
            f"{format_duration(estimate['ideal_parallel_encoding_seconds'])} "
            "(upper bound, encoding only)",
        ]
    calibration = report.get("calibration")
    if calibration:
        lines.append(
            "  calibration: python loop "
            f"{calibration['python_loop_seconds']:.3f}s, rdkit "
            f"{calibration['rdkit_parse_and_write_per_second']:,.0f} mol/s"
        )
    for item in report.get("pickle_benchmarks", []):
        if "error" in item:
            lines.append(f"  pickle [{item['label']}]: ERROR {item['error']}")
        else:
            lines.append(
                f"  pickle [{item['label']}]: write {item['write_seconds']:.2f}s, "
                f"read {item['read_seconds']:.2f}s, "
                f"{item['bytes'] / 1e6:.1f} MB for the sample"
            )
    for warning in report.get("warnings", []):
        lines.append(f"  WARNING: {warning}")
    return lines


def run_preprocessing_profile(
    *,
    csv_path: str | Path,
    output_dir: str | Path,
    processed_dir: str | Path,
    logger: Any,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    sample_seed: int = 42,
    min_nodes: int = MIN_NODES,
    max_nodes: int = MAX_NODES,
    expected_sha256: str | None = EXPECTED_ZINC250K_SHA256,
    expected_rows: int | None = EXPECTED_ZINC250K_ROWS,
    scratch_dirs: Sequence[str | Path] | None = None,
    skip_io: bool = False,
    skip_calibration: bool = False,
    progress_every: int = 500,
    heartbeat_seconds: float = 15.0,
    split_seed: int = 42,
    repository_dir: str | Path | None = None,
    stop_flag: StopFlag | None = None,
) -> dict[str, Any]:
    """Profile preprocessing on a deterministic sample; write the JSON report."""
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    assert_output_outside(output_dir, processed_dir)
    scratch_paths = [Path(p) for p in (scratch_dirs or [tempfile.gettempdir()])]
    for path in scratch_paths:
        assert_output_outside(path, processed_dir)

    stop_flag = stop_flag or StopFlag()
    warnings: list[str] = []
    low, high = SAMPLE_RANGE_RECOMMENDED
    if not low <= sample_size <= high:
        warnings.append(
            f"sample_size={sample_size} is outside the recommended "
            f"{low}-{high} range"
        )

    run_started = time.perf_counter()
    logger.header("ZINC250K preprocessing profiler (Phase 0, CPU only)")
    environment = collect_environment(repository_dir)
    logger.info(
        f"host={environment['hostname']} usable_cpus="
        f"{environment['usable_cpu_count']} (os_cpu_count="
        f"{environment['os_cpu_count']}, affinity="
        f"{environment['affinity_cpu_count']}, "
        f"cgroup_quota={environment['cgroup'].get('cpu_quota_cpus')})"
    )
    logger.info(f"cpu_model={environment['cpu_model']}")
    logger.info(f"slurm={environment['slurm']}")
    logger.info(f"threads={environment['thread_env']}")
    logger.info(f"commit={environment['repository_commit']}")
    logger.info(f"load_average={environment['load_average_1_5_15']}")
    if environment["slurm"].get("SLURM_JOB_ID") is None:
        warnings.append(
            "not running under SLURM (no SLURM_JOB_ID); on the cluster this "
            "must run via sbatch, never on the master node"
        )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "preprocessing_profile",
        "complete": False,
        "stopped_by": None,
        "representation_version": REPRESENTATION_VERSION,
        "environment": environment,
        "warnings": warnings,
        "writes_production_dataset": False,
    }

    def finish() -> dict[str, Any]:
        report["complete"] = (
            report.get("clean_pass") is not None
            and report.get("phase_breakdown") is not None
            and not stop_flag.requested
        )
        report["stopped_by"] = stop_flag.signal_name
        report["wall_seconds"] = time.perf_counter() - run_started
        report["peak_rss_mb"] = peak_rss_mb()
        name = (
            "preprocessing_profile.json" if report["complete"]
            else "preprocessing_profile.partial.json"
        )
        path = atomic_write_json(report, output_dir / name)
        logger.info(f"wrote report: {path}")
        lines = summary_lines(report)
        for line in lines:
            logger.info(line)
        text_name = (
            "preprocessing_profile.txt" if report["complete"]
            else "preprocessing_profile.partial.txt"
        )
        (output_dir / text_name).parent.mkdir(parents=True, exist_ok=True)
        (output_dir / text_name).write_text(
            "\n".join(lines) + "\n", encoding="utf-8",
        )
        logger.info(
            f"done in {format_duration(time.perf_counter() - run_started)}; "
            f"complete={report['complete']}"
        )
        return report

    # ---- machine calibration ---------------------------------------------
    if not skip_calibration:
        logger.info("calibration: fixed workload (python loop + RDKit)")
        report["calibration"] = machine_calibration()
        logger.info(
            f"calibration done: python_loop="
            f"{report['calibration']['python_loop_seconds']:.3f}s rdkit="
            f"{report['calibration']['rdkit_parse_and_write_per_second']:,.0f} mol/s"
        )

    # ---- dataset identity + full CSV read (pipeline I/O cost) ---------------
    io_timing: dict[str, Any] = {}
    if skip_io:
        logger.warn("skip_io: not hashing the CSV or timing the full read")
        table = read_zinc_table(csv_path)  # still needed to draw the sample
        population = len(table)
    else:
        identity = verify_dataset_identity(
            csv_path, expected_sha256=expected_sha256, logger=logger,
        )
        io_timing["sha256_seconds"] = identity["hash_seconds"]
        io_timing["csv_bytes"] = identity["csv_bytes"]
        started = time.perf_counter()
        table = read_zinc_table(csv_path)
        io_timing["csv_read_seconds"] = time.perf_counter() - started
        population = len(table)
        logger.info(
            f"csv read: {population:,} records in "
            f"{io_timing['csv_read_seconds']:.2f}s rss_mb={current_rss_mb()}"
        )
        report["dataset"] = {**identity, "records_in_file": population}
        if expected_rows is not None and population != expected_rows:
            warnings.append(
                f"record count {population:,} differs from expected "
                f"{expected_rows:,}"
            )
    report["io_timing"] = io_timing

    indices = deterministic_sample_indices(population, sample_size, sample_seed)
    sample_smiles = [table.smiles[i] for i in indices]
    report["sample"] = {
        "size": len(indices),
        "population": population,
        "seed": sample_seed,
        "index_fingerprint": indices_fingerprint(indices),
        "first_indices": indices[:5],
    }
    logger.info(
        f"sample: {len(indices):,} of {population:,} rows, seed={sample_seed}, "
        f"fingerprint={report['sample']['index_fingerprint']}"
    )

    # ---- warm-up (RDKit lazy initialisation, QED tables) -------------------
    warm = sample_smiles[: min(WARMUP_MOLECULES, len(sample_smiles))]
    started = time.perf_counter()
    for smiles in warm:
        chemistry.encode_zinc_smiles(smiles, min_nodes=min_nodes, max_nodes=max_nodes)
    report["warmup"] = {
        "molecules": len(warm), "seconds": time.perf_counter() - started,
    }
    logger.info(
        f"warm-up: {len(warm)} molecules in {report['warmup']['seconds']:.2f}s "
        "(excluded from statistics)"
    )

    # ---- clean pass: honest throughput --------------------------------------
    rss_before = current_rss_mb()
    clean = _encode_pass(
        sample_smiles, min_nodes=min_nodes, max_nodes=max_nodes, logger=logger,
        label="clean-pass", every=progress_every,
        heartbeat_seconds=heartbeat_seconds, stop_flag=stop_flag,
        keep_records=True,
    )
    rss_after = current_rss_mb()
    latencies = clean["latencies"]
    outcomes = clean["outcomes"]
    accepted_records = clean["records"]
    report["clean_pass"] = {
        "molecules": int(latencies.size),
        "wall_seconds": clean["wall_seconds"],
        "throughput_mol_per_s": (
            latencies.size / latencies.sum() if latencies.sum() > 0 else float("nan")
        ),
        "latency": latency_statistics(latencies),
        "accepted": len(accepted_records),
        "accepted_fraction": (
            len(accepted_records) / latencies.size if latencies.size else None
        ),
        "outcomes": _outcome_breakdown(latencies, outcomes),
    }
    slowest = np.argsort(latencies)[::-1][:SLOWEST_TO_REPORT]
    report["clean_pass"]["slowest_molecules"] = [
        {
            "sample_position": int(position),
            "csv_row_index": int(indices[int(position)]),
            "seconds": float(latencies[position]),
            "outcome": outcomes[int(position)],
            "smiles": sample_smiles[int(position)],
        }
        for position in slowest
    ]
    logger.info(
        f"clean pass: {report['clean_pass']['throughput_mol_per_s']:,.1f} mol/s "
        f"mean={report['clean_pass']['latency']['mean_ms']:.3f}ms "
        f"p99={report['clean_pass']['latency']['p99_ms']:.2f}ms "
        f"accepted={len(accepted_records):,}/{latencies.size:,}"
    )
    if stop_flag.requested:
        return finish()

    # ---- instrumented pass: phase breakdown ---------------------------------
    timer = PhaseTimer()
    with instrument_chemistry(timer):
        instrumented = _encode_pass(
            sample_smiles, min_nodes=min_nodes, max_nodes=max_nodes,
            logger=logger, label="phase-pass", every=progress_every,
            heartbeat_seconds=heartbeat_seconds, stop_flag=stop_flag,
            keep_records=False,
        )
    # The production module must be restored exactly.
    assert chemistry.Chem is Chem, "chemistry.Chem was not restored"

    n_inst = int(instrumented["latencies"].size)
    total_inst = float(instrumented["latencies"].sum())
    phase_blocks = {}
    for phase in PHASES:
        seconds = timer.seconds[phase]
        phase_blocks[phase] = {
            "total_seconds": seconds,
            "mean_ms_per_molecule": 1e3 * seconds / n_inst if n_inst else float("nan"),
            "calls": timer.calls[phase],
            "calls_per_molecule": timer.calls[phase] / n_inst if n_inst else float("nan"),
            "share_of_total": seconds / total_inst if total_inst > 0 else float("nan"),
        }
    accounted = sum(timer.seconds.values())
    report["phase_breakdown"] = {
        "molecules": n_inst,
        "instrumented_total_seconds": total_inst,
        "phases": phase_blocks,
        "other_seconds": max(0.0, total_inst - accounted),
        "other_share": max(0.0, total_inst - accounted) / total_inst if total_inst > 0 else float("nan"),
        "instrumentation_overhead_ratio": (
            total_inst / float(latencies.sum()) if latencies.sum() > 0 else float("nan")
        ),
        "note": (
            "Phases are timed around the real functions used by "
            "encode_zinc_smiles. 'canonicalization' includes both the "
            "pre-encoding and the round-trip canonical SMILES; 'parse' "
            "includes both MolFromSmiles calls; 'reconstruction' is "
            "graph_to_mol (including its sanitization). 'other' is the "
            "remainder: fragment/atom checks, bond-matrix construction and "
            "valence checks."
        ),
    }
    logger.info(
        "phase breakdown: " + ", ".join(
            f"{p}={100 * phase_blocks[p]['share_of_total']:.1f}%" for p in PHASES
        ) + f", other={100 * report['phase_breakdown']['other_share']:.1f}%"
    )
    if stop_flag.requested:
        return finish()

    # ---- surrounding pipeline stages (split, scaler, pickle) -----------------
    stage_seconds: dict[str, float] = {}
    pickle_benchmarks: list[dict[str, Any]] = []
    if len(accepted_records) >= 10:
        started = time.perf_counter()
        splits = random_split(
            accepted_records, train_fraction=0.8, validation_fraction=0.1,
            seed=split_seed,
        )
        stage_seconds["split"] = time.perf_counter() - started

        started = time.perf_counter()
        scaler = ConditionScaler.fit(splits["train"])
        stage_seconds["scaler_fit"] = time.perf_counter() - started

        # Work on a copy so the original records stay untouched, mirroring
        # how production attaches the condition vector to each record.
        started = time.perf_counter()
        payload_records = []
        for subset in splits.values():
            for record in subset:
                payload_records.append(
                    {**record, "condition": scaler.transform(record["qed"], record["n_node"])}
                )
        stage_seconds["condition_transform"] = time.perf_counter() - started

        if not skip_io:
            for index, directory in enumerate(scratch_paths):
                pickle_benchmarks.append(
                    benchmark_pickle(
                        payload_records, directory,
                        label=f"scratch[{index}]={directory}",
                    )
                )
                last = pickle_benchmarks[-1]
                if "error" in last:
                    logger.warn(f"pickle benchmark failed: {last['error']}")
                else:
                    logger.info(
                        f"pickle [{last['label']}]: write={last['write_seconds']:.2f}s "
                        f"read={last['read_seconds']:.2f}s "
                        f"size={last['bytes'] / 1e6:.1f}MB"
                    )
    else:
        warnings.append(
            "fewer than 10 accepted molecules in the sample; split, scaler "
            "and pickle stages were skipped"
        )
    report["stage_seconds_for_sample"] = stage_seconds
    report["pickle_benchmarks"] = pickle_benchmarks

    # ---- memory ------------------------------------------------------------
    memory: dict[str, Any] = {
        "rss_mb_before_clean_pass": rss_before,
        "rss_mb_after_clean_pass": rss_after,
        "peak_rss_mb": peak_rss_mb(),
    }
    if rss_before is not None and rss_after is not None and accepted_records:
        per_record_mb = max(0.0, rss_after - rss_before) / len(accepted_records)
        memory["rough_rss_mb_per_retained_record"] = per_record_mb
    report["memory"] = memory

    # ---- full-dataset extrapolation -----------------------------------------
    usable = usable_cpu_count()
    accepted_fraction = report["clean_pass"]["accepted_fraction"] or 0.0
    encoding = extrapolate(latencies, population)
    scale_pickle = None
    pickle_estimates: list[dict[str, Any]] = []
    if accepted_records and population:
        scale_pickle = (accepted_fraction * population) / len(accepted_records)
        for item in pickle_benchmarks:
            if "error" in item:
                continue
            pickle_estimates.append({
                "label": item["label"],
                "write_seconds": item["write_seconds"] * scale_pickle,
                "read_seconds": item["read_seconds"] * scale_pickle,
                "gigabytes": item["bytes"] * scale_pickle / 1e9,
            })
        if memory.get("rough_rss_mb_per_retained_record") is not None:
            memory["rough_full_run_retained_records_mb"] = (
                memory["rough_rss_mb_per_retained_record"]
                * accepted_fraction * population
            )

    other_serial = (
        io_timing.get("sha256_seconds", 0.0)
        + io_timing.get("csv_read_seconds", 0.0)
    )
    if scale_pickle is not None:
        other_serial += sum(
            stage_seconds.get(k, 0.0) for k in ("split", "scaler_fit", "condition_transform")
        ) * scale_pickle
        if pickle_estimates:
            # Use the slowest scratch location as the conservative estimate.
            other_serial += max(e["write_seconds"] for e in pickle_estimates)
    total_serial = encoding["estimate_seconds"] + other_serial

    report["full_dataset_estimate"] = {
        "population_rows": population,
        "assumed_accepted_fraction": accepted_fraction,
        "encoding": encoding,
        "other_stages_seconds": other_serial,
        "pickle": pickle_estimates,
        "total_serial_seconds": total_serial,
        "fits_1h_serial": bool(total_serial < 3600.0),
        "suggested_time_limit": suggest_time_limit(total_serial),
        "usable_cpus": usable,
        "ideal_parallel_encoding_seconds": encoding["estimate_seconds"] / max(1, usable),
        "caveats": [
            "linear extrapolation from a seeded sample; assumes the sample "
            "is representative (compare the sample fingerprint across runs)",
            "reflects the machine load during this job only",
            "ideal_parallel_encoding_seconds is an upper bound on speed-up "
            "(no multiprocessing exists in the current pipeline)",
            "excludes the stage-1b plotting step",
        ],
    }
    if table.nonnumeric_qed_rows:
        warnings.append(
            f"{table.nonnumeric_qed_rows} CSV rows have non-numeric QED"
        )
    return finish()


__all__ = [
    "PHASES",
    "PhaseTimer",
    "ProgressReporter",
    "benchmark_pickle",
    "extrapolate",
    "instrument_chemistry",
    "latency_statistics",
    "machine_calibration",
    "run_preprocessing_profile",
    "suggest_time_limit",
    "summary_lines",
]
