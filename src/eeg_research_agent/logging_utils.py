"""Structured logging and execution bookkeeping.

Two streams are produced:

* a human readable log (console + ``pipeline.log``)
* a machine readable JSONL event log (``artifacts/execution_logs/events.jsonl``)
  which the verification agent uses to check that every number quoted in a
  report was actually computed during the run.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .paths import ensure_dir

__all__ = [
    "setup_logging",
    "get_logger",
    "EventLogger",
    "ExecutionManifest",
    "capture_environment",
]

_CONFIGURED = False


def setup_logging(
    level: str = "INFO",
    fmt: str = "%(asctime)s %(levelname)-8s %(name)s :: %(message)s",
    log_file: str | Path | None = None,
    quiet: bool = False,
) -> logging.Logger:
    """Configure the root logger once per process."""
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED:
        return root
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(fmt)

    if not quiet:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(formatter)
        root.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        ensure_dir(path.parent)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("mne").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def capture_environment() -> dict[str, Any]:
    """Record interpreter / platform facts for reproducibility."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cwd": os.getcwd(),
    }
    for module_name in ("numpy", "scipy", "sklearn", "pandas", "mne", "matplotlib", "pydantic", "langgraph"):
        try:
            module = __import__(module_name)
            info[f"{module_name}_version"] = getattr(module, "__version__", "unknown")
        except Exception:  # pragma: no cover - optional dependency
            info[f"{module_name}_version"] = None
    return info


class EventLogger:
    """Append-only JSONL event log used as an evidence trail for reports."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self._started = time.time()

    def emit(self, event: str, **payload: Any) -> dict[str, Any]:
        record = {
            "event": event,
            "timestamp": round(time.time(), 6),
            "elapsed_s": round(time.time() - self._started, 3),
            **payload,
        }
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        return record

    @contextmanager
    def stage(self, name: str, **payload: Any) -> Iterator[dict[str, Any]]:
        """Time a pipeline stage and log its outcome (including failures)."""
        started = time.time()
        self.emit("stage_start", stage=name, **payload)
        status: dict[str, Any] = {"stage": name}
        try:
            yield status
        except Exception as exc:
            self.emit(
                "stage_error",
                stage=name,
                duration_s=round(time.time() - started, 3),
                error_type=type(exc).__name__,
                error=str(exc),
                **payload,
            )
            raise
        self.emit(
            "stage_end",
            stage=name,
            duration_s=round(time.time() - started, 3),
            **{k: v for k, v in status.items() if k != "stage"},
            **payload,
        )


@dataclass
class ExecutionManifest:
    """Mutable record of one pipeline run; persisted next to the artifacts."""

    run_id: str
    objective: str
    data_origin: str = "unknown"
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    stages: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=capture_environment)

    def add_stage(self, name: str, duration_s: float, **extra: Any) -> None:
        self.stages.append({"stage": name, "duration_s": round(duration_s, 3), **extra})

    def to_dict(self) -> dict[str, Any]:
        data = {
            "run_id": self.run_id,
            "objective": self.objective,
            "data_origin": self.data_origin,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": (
                round(self.finished_at - self.started_at, 3) if self.finished_at else None
            ),
            "stages": self.stages,
            "metrics": self.metrics,
            "errors": self.errors,
            "environment": self.environment,
        }
        return data

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        ensure_dir(path.parent)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, default=str)
        return path
