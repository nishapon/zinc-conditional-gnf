"""Lightweight structured logging for training and evaluation jobs.

Why a dedicated module (not just `print` or stdlib logging): every long
cluster job needs to produce an output that is (1) tailable in real time
from the SLURM `.log` file, (2) parseable after the fact for making plots
and tables, and (3) carries timing and shape context on every entry so
"what was running at second 3,217" is answerable without a reconstruction
of the whole run.

Design:
  * One line per event. No multi-line records.
  * Each line starts with an ISO timestamp and the elapsed wall-clock
    seconds since the logger was created, so you can read either the
    absolute time or the training-run time at a glance.
  * Metrics are logged as `key=value` pairs so a grep-and-awk one-liner
    can reconstruct any curve from the log file alone (useful when WandB
    is in offline mode on the cluster).
  * Output goes to BOTH stdout (so `tail -f logs/foo_output.log` works)
    AND a dedicated log file (so re-parsing is clean, with no SLURM
    header noise).
"""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import sys
import time
from typing import Any, Mapping


class JobLogger:
    """Structured per-job logger with stdout + file sinks.

    Instantiate once at the start of a script with a run name and an
    output directory; use the resulting instance for everything that
    would otherwise be a `print` call.
    """

    def __init__(
        self,
        name: str,
        output_dir: str | Path,
        *,
        filename: str = "run.log",
    ) -> None:
        self.name = str(name)
        self.start_time = time.monotonic()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._logger = logging.getLogger(f"zinc_gnf.{name}")
        self._logger.setLevel(logging.INFO)
        # Avoid duplicate handlers if a parent logger re-imports this.
        self._logger.handlers.clear()
        self._logger.propagate = False

        formatter = logging.Formatter(
            fmt="%(message)s",
        )

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        self._logger.addHandler(stdout_handler)

        file_path = output_dir / filename
        file_handler = logging.FileHandler(file_path, mode="a")
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

        self.file_path = file_path

    # ------------------------------------------------------------------
    # Low-level emit — all other methods go through this.
    # ------------------------------------------------------------------
    def _emit(self, level: str, message: str) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        elapsed_seconds = time.monotonic() - self.start_time
        self._logger.info(
            f"{timestamp} | +{elapsed_seconds:9.2f}s | {level:<5} | "
            f"{self.name} | {message}"
        )

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------
    def info(self, message: str) -> None:
        self._emit("INFO", message)

    def warn(self, message: str) -> None:
        self._emit("WARN", message)

    def error(self, message: str) -> None:
        self._emit("ERROR", message)

    def header(self, title: str) -> None:
        """Print a visually separable section header."""
        line = "=" * 70
        self._emit("INFO", line)
        self._emit("INFO", title)
        self._emit("INFO", line)

    def metrics(
        self,
        step: int,
        metrics: Mapping[str, Any],
        *,
        prefix: str = "metric",
    ) -> None:
        """Log a flat set of metric key=value pairs for one step.

        Formatting intentionally stable so downstream grep/awk tools can
        parse the run.log without a bespoke parser:

            metric step=1200 train_loss=0.4231 val_loss=0.4987 ...
        """
        parts = [f"step={int(step)}"]
        for key, value in metrics.items():
            if isinstance(value, float):
                parts.append(f"{key}={value:.6g}")
            else:
                parts.append(f"{key}={value}")
        self._emit("INFO", f"{prefix} " + " ".join(parts))

    def shapes(self, tensors: Mapping[str, Any]) -> None:
        """Log the shape of a dictionary of tensors.

        Used by the trainers' --dim-analysis flag (one of the professor's
        whiteboard checks). Accepts anything with a `.shape` attribute,
        including torch.Tensor and numpy.ndarray.
        """
        parts = []
        for key, tensor in tensors.items():
            shape = getattr(tensor, "shape", None)
            dtype = getattr(tensor, "dtype", None)
            if shape is None:
                parts.append(f"{key}=<no shape>")
            else:
                parts.append(f"{key}={tuple(shape)}:{dtype}")
        self._emit("INFO", "shapes " + " ".join(parts))


def null_logger() -> "JobLogger":
    """A logger that writes to /tmp — handy for unit tests that need to
    construct training objects but don't care about log output.
    """
    import tempfile

    return JobLogger(
        name="null",
        output_dir=tempfile.mkdtemp(prefix="zinc_gnf_test_"),
    )


__all__ = ["JobLogger", "null_logger"]
