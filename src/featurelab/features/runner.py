"""Execute one candidate feature over a whole cohort, in its own process.

Run as ``python -m featurelab.features.runner ...`` by :func:`run_feature`; never
imported into the orchestrator process.  It writes the feature column to
``out.npy`` and a JSON report with behavioural checks:

* probe: does it return a finite float on real epochs, fast enough?
* determinism: same epoch twice -> same value?
* scale invariance: does doubling the amplitude change it?  (Reported, not
  required; scale-invariant features transfer better to ear-EEG.)
"""

from __future__ import annotations

import argparse
import builtins
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len", "list", "max",
        "min", "pow", "range", "reversed", "round", "set", "sorted", "sum", "tuple", "zip",
        "isinstance", "map", "filter", "ValueError", "ZeroDivisionError", "Exception",
        "ArithmeticError", "FloatingPointError", "print", "slice", "divmod", "complex", "True",
        "False", "None", "str", "iter", "next", "frozenset", "hasattr",
    )
    if hasattr(builtins, name)
}
ALLOWED_ROOTS = {"numpy", "scipy", "math"}


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
    if level or name.split(".")[0] not in ALLOWED_ROOTS:
        raise ImportError(f"import of {name} is not allowed")
    return builtins.__import__(name, globals, locals, fromlist, level)


def load_function(code: str):
    namespace = {"__builtins__": {**SAFE_BUILTINS, "__import__": _guarded_import}, "__name__": "candidate"}
    exec(compile(code, "<candidate>", "exec"), namespace)  # noqa: S102 - gated + isolated process
    return namespace["feature"]


# ---- worker-process state (set by the pool initializer) -------------------- #
_W: dict = {}


def _init(code: str, cohort_dir: str) -> None:
    np.seterr(all="ignore")
    _W["fn"] = load_function(code)
    _W["X"] = np.load(Path(cohort_dir) / "X.npy", mmap_mode="r")
    _W["psd"] = np.load(Path(cohort_dir) / "psd.npy", mmap_mode="r")
    _W["freqs"] = np.load(Path(cohort_dir) / "freqs.npy").astype(np.float64)
    _W["fs"] = float(np.load(Path(cohort_dir) / "meta.npz")["sfreq"])


def _call(i: int) -> float:
    x = np.array(_W["X"][i], dtype=np.float64)
    p = np.array(_W["psd"][i], dtype=np.float64)
    try:
        value = float(_W["fn"](x, _W["fs"], _W["freqs"].copy(), p))
    except Exception:  # noqa: BLE001 - counted as a failure (NaN)
        return math.nan
    return value if math.isfinite(value) else math.nan


def _chunk(bounds: tuple[int, int]) -> np.ndarray:
    return np.array([_call(i) for i in range(*bounds)], dtype=np.float64)


def _probe(code: str, cohort_dir: str, n_probe: int = 64) -> dict:
    _init(code, cohort_dir)
    n = _W["X"].shape[0]
    idx = np.linspace(0, n - 1, num=min(n_probe, n)).astype(int)
    fn, fs, freqs = _W["fn"], _W["fs"], _W["freqs"]
    first_error = None
    values, t0 = [], time.perf_counter()
    for i in idx:
        x = np.array(_W["X"][i], dtype=np.float64)
        p = np.array(_W["psd"][i], dtype=np.float64)
        try:
            values.append(float(fn(x, fs, freqs.copy(), p)))
        except Exception as exc:  # noqa: BLE001
            values.append(math.nan)
            first_error = first_error or f"{type(exc).__name__}: {exc}"
    ms = 1000 * (time.perf_counter() - t0) / len(idx)
    values = np.asarray(values)
    x0 = np.array(_W["X"][idx[len(idx) // 2]], dtype=np.float64)
    p0 = np.array(_W["psd"][idx[len(idx) // 2]], dtype=np.float64)
    deterministic, scale_invariant = None, None
    try:
        a, b = float(fn(x0, fs, freqs.copy(), p0)), float(fn(x0.copy(), fs, freqs.copy(), p0.copy()))
        deterministic = bool(a == b or (math.isnan(a) and math.isnan(b)))
        c = float(fn(2 * x0, fs, freqs.copy(), 4 * p0))
        scale_invariant = bool(np.isclose(a, c, rtol=1e-3, atol=1e-9))
    except Exception:  # noqa: BLE001
        pass
    return {
        "probe_n": int(len(idx)),
        "probe_nan_fraction": float(np.mean(~np.isfinite(values))),
        "probe_first_error": first_error,
        "ms_per_epoch": round(ms, 3),
        "deterministic": deterministic,
        "scale_invariant": scale_invariant,
        "probe_unique_values": int(len(np.unique(values[np.isfinite(values)]))),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", required=True)
    parser.add_argument("--cohort-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-ms", type=float, default=25.0)
    args = parser.parse_args(argv)

    code = Path(args.code).read_text(encoding="utf-8")
    report: dict = {"status": "error"}
    try:
        report.update(_probe(code, args.cohort_dir))
        if report["probe_nan_fraction"] >= 0.5:
            report["reason"] = f"returned NaN/error on most probe epochs ({report['probe_first_error']})"
        elif report["ms_per_epoch"] > args.max_ms:
            report["reason"] = f"too slow: {report['ms_per_epoch']} ms/epoch > {args.max_ms}"
        elif report["deterministic"] is False:
            report["reason"] = "not deterministic: same epoch gave different values"
        else:
            from multiprocessing import get_context

            n = _W["X"].shape[0]
            workers = args.workers or os.cpu_count() or 1
            step = max(256, n // (workers * 8))
            bounds = [(s, min(n, s + step)) for s in range(0, n, step)]
            t0 = time.perf_counter()
            with get_context("spawn").Pool(workers, initializer=_init, initargs=(code, args.cohort_dir)) as pool:
                column = np.concatenate(pool.map(_chunk, bounds))
            np.save(args.out, column)
            report.update(
                status="ok",
                n=int(n),
                nan_fraction=float(np.mean(~np.isfinite(column))),
                seconds=round(time.perf_counter() - t0, 2),
            )
    except Exception as exc:  # noqa: BLE001 - the report carries the failure back to the agent
        report["reason"] = f"{type(exc).__name__}: {exc}"
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


def run_feature(code: str, cohort_dir: Path, workdir: Path, timeout_s: float = 900,
                max_ms: float = 25.0, workers: int = 0) -> tuple[np.ndarray | None, dict]:
    """Parent-side helper: run the candidate in a child process, return (column, report)."""
    workdir = Path(workdir).resolve()
    cohort_dir = Path(cohort_dir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    code_path = workdir / "feature.py"
    code_path.write_text(code, encoding="utf-8")
    out, report_path = workdir / "column.npy", workdir / "runner_report.json"
    keep = {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "PYTHONPATH", "OMP_NUM_THREADS", "SYSTEMROOT"}
    env = {k: v for k, v in os.environ.items() if k in keep}  # no API keys reach generated code
    env["OMP_NUM_THREADS"] = "1"
    src_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = src_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    def _limits():  # pragma: no cover - runs in the child
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (int(timeout_s), int(timeout_s) + 5))
        except Exception:
            pass

    command = [sys.executable, "-m", "featurelab.features.runner", "--code", str(code_path),
               "--cohort-dir", str(cohort_dir), "--out", str(out), "--report", str(report_path),
               "--workers", str(workers), "--max-ms", str(max_ms)]
    try:
        proc = subprocess.run(command, env=env, capture_output=True, text=True, timeout=timeout_s,
                              preexec_fn=_limits if os.name == "posix" else None, cwd=tempfile.gettempdir())
    except subprocess.TimeoutExpired:
        return None, {"status": "error", "reason": f"timeout after {timeout_s}s"}
    if not report_path.is_file():
        tail = (proc.stderr or "")[-800:]
        return None, {"status": "error", "reason": f"runner crashed (exit {proc.returncode}): {tail}"}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    column = np.load(out) if report.get("status") == "ok" and out.is_file() else None
    return column, report


if __name__ == "__main__":
    sys.exit(main())
