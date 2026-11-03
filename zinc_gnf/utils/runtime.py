"""Runtime helpers shared by the Phase 0 diagnostic scripts.

These helpers exist so a diagnostic job that is slow, killed, or stuck on a
cluster node leaves enough evidence in its log and report to say why:

  * `collect_environment` records what the job actually got (usable CPUs
    after affinity and cgroup limits, memory limits, SLURM allocation, library
    versions, git commit) rather than what it asked for.
  * `StopFlag` turns SLURM's SIGTERM (sent before SIGKILL at the time limit,
    or on `scancel`) into a cooperative stop so a partial report can be
    written.
  * `json_safe` / `atomic_write_json` make reports strictly valid JSON
    (NaN/inf become null) and never leave a half-written file.
  * `assert_output_outside` guarantees a diagnostic can never write into the
    production processed-data directory.

Nothing here touches the network.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Mapping


# ---------------------------------------------------------------------------
# JSON.
# ---------------------------------------------------------------------------
def json_safe(value: Any) -> Any:
    """Recursively convert `value` into strictly JSON-serializable data.

    NaN and infinity become `None` (JSON has no representation for them);
    numpy scalars and arrays become Python numbers and lists; mapping keys
    become strings; sets become sorted lists.
    """
    import numpy as np

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return str(value)


def atomic_write_json(payload: Any, path: str | Path) -> Path:
    """Write `payload` as indented JSON via a temp file + rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            json_safe(payload), handle, indent=2, allow_nan=False,
        )
        handle.write("\n")
    temporary.replace(path)
    return path


# ---------------------------------------------------------------------------
# Output safety.
# ---------------------------------------------------------------------------
def assert_output_outside(
    output_dir: str | Path, protected_dir: str | Path,
) -> None:
    """Refuse to let a diagnostic write inside the production data directory.

    Diagnostics must never create, overwrite, or delete files in the
    processed-dataset directory (`data.processed_dir` in the config).
    """
    output = Path(output_dir).resolve()
    protected = Path(protected_dir).resolve()
    if output == protected or protected in output.parents:
        raise ValueError(
            f"refusing to write diagnostics inside the production "
            f"processed-data directory: output={output} protected={protected}"
        )


# ---------------------------------------------------------------------------
# Cooperative stop on SIGTERM / SIGINT.
# ---------------------------------------------------------------------------
class StopFlag:
    """Record SIGTERM/SIGINT so long loops can stop and write a partial report.

    SLURM sends SIGTERM when a job hits its time limit or is cancelled and
    SIGKILL shortly after, so a loop that polls `requested` can still flush
    a partial report in that short window.
    """

    def __init__(self, logger: Any = None) -> None:
        self.signal_name: str | None = None
        self._logger = logger
        self._previous: dict[int, Any] = {}

    @property
    def requested(self) -> bool:
        return self.signal_name is not None

    def _handler(self, signum: int, frame: Any) -> None:  # noqa: ARG002
        self.signal_name = signal.Signals(signum).name
        if self._logger is not None:
            self._logger.warn(
                f"received {self.signal_name}; finishing the current "
                "molecule and writing a partial report"
            )

    def __enter__(self) -> "StopFlag":
        for signum in (signal.SIGTERM, signal.SIGINT):
            try:
                self._previous[signum] = signal.signal(signum, self._handler)
            except ValueError:
                # Not the main thread; cooperative stop is unavailable.
                pass
        return self

    def __exit__(self, *exc_info: Any) -> None:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)
        self._previous.clear()


# ---------------------------------------------------------------------------
# Memory.
# ---------------------------------------------------------------------------
def current_rss_mb() -> float | None:
    """Resident set size of this process in MiB (Linux), else None."""
    try:
        with open("/proc/self/status", "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


def peak_rss_mb() -> float | None:
    """Peak resident set size of this process in MiB."""
    try:
        import resource

        peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return None
    # ru_maxrss is KiB on Linux and bytes on macOS.
    return peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0


# ---------------------------------------------------------------------------
# Environment capture.
# ---------------------------------------------------------------------------
SLURM_ENV_KEYS = (
    "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_PARTITION",
    "SLURM_JOB_NODELIST", "SLURM_NODELIST", "SLURM_CPUS_PER_TASK",
    "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE", "SLURM_NTASKS",
    "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_TIME_LIMIT",
    "SLURM_SUBMIT_DIR", "SLURM_SUBMIT_HOST", "SLURM_GPUS", "SLURM_JOB_GPUS",
    "CUDA_VISIBLE_DEVICES",
)

THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "PYTHONUNBUFFERED", "TMPDIR",
)


def _read_text(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return None


def read_cgroup_limits() -> dict[str, Any]:
    """Best-effort CPU and memory limits from cgroup v2 or v1."""
    limits: dict[str, Any] = {}

    candidate_dirs = ["/sys/fs/cgroup"]
    own = _read_text("/proc/self/cgroup")
    if own:
        for line in own.splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3 and parts[0] == "0" and parts[2] != "/":
                candidate_dirs.insert(0, "/sys/fs/cgroup" + parts[2])

    for directory in candidate_dirs:
        cpu_max = _read_text(f"{directory}/cpu.max")
        if cpu_max:
            limits["cgroup_v2_dir"] = directory
            limits["cpu_max"] = cpu_max
            fields = cpu_max.split()
            if len(fields) == 2 and fields[0] != "max":
                try:
                    limits["cpu_quota_cpus"] = float(fields[0]) / float(fields[1])
                except (ValueError, ZeroDivisionError):
                    pass
            limits["memory_max"] = _read_text(f"{directory}/memory.max")
            break

    if "cpu_max" not in limits:
        quota = _read_text("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period = _read_text("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if quota and period:
            limits["cgroup_v1_cfs_quota_us"] = quota
            limits["cgroup_v1_cfs_period_us"] = period
            try:
                if float(quota) > 0:
                    limits["cpu_quota_cpus"] = float(quota) / float(period)
            except (ValueError, ZeroDivisionError):
                pass
        limits["memory_max"] = _read_text(
            "/sys/fs/cgroup/memory/memory.limit_in_bytes"
        )
    return limits


def usable_cpu_count() -> int:
    """CPUs this process may actually run on (affinity, then cgroup quota)."""
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    quota = read_cgroup_limits().get("cpu_quota_cpus")
    if quota:
        count = min(count, max(1, int(math.floor(quota + 1e-9))))
    return max(1, count)


def _cpu_model() -> str | None:
    info = _read_text("/proc/cpuinfo")
    if not info:
        return platform.processor() or None
    for line in info.splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or None


def _git_commit(directory: str | Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(directory) if directory else None,
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and commit else None


def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except Exception:  # noqa: BLE001
        return None
    return getattr(module, "__version__", "unknown")


def collect_environment(repository_dir: str | Path | None = None) -> dict[str, Any]:
    """Describe the machine, allocation, and software this job really got."""
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    try:
        load_average = list(os.getloadavg())
    except (AttributeError, OSError):
        load_average = None

    cgroup = read_cgroup_limits()
    return {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "os_cpu_count": os.cpu_count(),
        "affinity_cpu_count": affinity,
        "cgroup": cgroup,
        "usable_cpu_count": usable_cpu_count(),
        "load_average_1_5_15": load_average,
        "libraries": {
            name: _module_version(name)
            for name in ("rdkit", "numpy", "scipy", "matplotlib", "yaml")
        },
        "slurm": {
            key: os.environ[key] for key in SLURM_ENV_KEYS if key in os.environ
        },
        "thread_env": {
            key: os.environ.get(key) for key in THREAD_ENV_KEYS
        },
        "stdout_is_tty": bool(getattr(sys.stdout, "isatty", lambda: False)()),
        "repository_commit": _git_commit(repository_dir),
        "utc_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def format_duration(seconds: float | None) -> str:
    """Human-readable duration, e.g. 3725 -> '1h 02m 05s'."""
    if seconds is None or not math.isfinite(seconds):
        return "n/a"
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


__all__ = [
    "StopFlag",
    "assert_output_outside",
    "atomic_write_json",
    "collect_environment",
    "current_rss_mb",
    "format_duration",
    "json_safe",
    "peak_rss_mb",
    "read_cgroup_limits",
    "usable_cpu_count",
]
