"""Evaluation steps shared by the agent loop and by human-submitted features.

A feature written by a teammate (``featurelab evaluate my_feature.py``) and a
feature invented by the LLM go through *exactly* the same code path.  One
implementation of the science, many entry points.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..features.gate import check_code


def gate_and_run(evaluator, code: str, workdir: Path) -> dict:
    """Static gate, then execute on the discovery cohort.  Returns a step result."""
    sb = evaluator.config["sandbox"]
    gate = check_code(code, int(sb["max_code_chars"]))
    if not gate.ok:
        return {"ok": False, "stage": "gate", "reason": "static check: " + gate.message()}
    column, report = evaluator.compute_column(code, "discovery", workdir)
    if column is None:
        return {"ok": False, "stage": "runtime", "reason": "runtime: " + str(report.get("reason")),
                "runner": report}
    path = workdir / "discovery_column.npy"
    np.save(path, column)
    return {"ok": True, "stage": "gate", "runner": report, "column_path": str(path)}


def confirm(evaluator, ledger, protocol: str, code: str, workdir: Path, discovery_column: np.ndarray,
            name: str) -> dict:
    """Stage C.  Each lockbox look spends part of a pre-declared alpha budget."""
    conf = evaluator.ev["confirm"]
    looks = ledger.count_lockbox_looks(protocol)
    max_looks = int(conf.get("max_looks", 10))
    if looks >= max_looks:
        return {"ok": False, "reason": f"lockbox budget exhausted ({looks}/{max_looks} looks used)",
                "exhausted": True}
    alpha = float(conf["alpha"]) / max_looks
    columns = {"discovery": discovery_column}
    runner = {}
    for role in ("lockbox", "external"):
        if evaluator.has_cohort(role):
            column, report = evaluator.compute_column(code, role, workdir)
            columns[role], runner[role] = column, report
    if columns.get("lockbox") is None:
        return {"ok": False, "reason": "lockbox cohort unavailable (run `featurelab prepare --cohort lockbox`)",
                "unavailable": True}
    result = evaluator.confirm(columns, name)
    lock = result["lockbox"]
    ok = lock["p_value"] < alpha and lock["mean_delta"] > float(conf["min_delta"])
    result["alpha_per_look"] = alpha
    result["look_number"] = looks + 1
    if "external" in result:
        ext = result["external"]
        result["generalises_external"] = bool(ext["mean_delta"] > 0 and ext["p_value"] < float(conf["alpha"]))
    reason = "confirmed on unseen subjects" if ok else (
        f"lockbox gain {lock['mean_delta']:+.4f}, p={lock['p_value']:.2g} (needs < {alpha:.2g})")
    return {"ok": ok, "reason": reason, "confirm": result, "runner": runner}
