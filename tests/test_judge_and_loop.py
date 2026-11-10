"""System tests: the judge must pass a planted effect and reject noise; the loop must run end to end."""

from pathlib import Path

import numpy as np

from featurelab.data import synthetic
from featurelab.evaluation.stages import Evaluator
from featurelab.orchestrator.graph import run_loop
from featurelab.orchestrator.ledger import Ledger


def _column(ev, code, tmp_path, name):
    column, report = ev.compute_column(code, "discovery", tmp_path / name)
    assert report["status"] == "ok", report
    return column


def test_positive_control_passes(world, tmp_path):
    ev = Evaluator(world)
    column = _column(ev, synthetic.POSITIVE_CONTROL, tmp_path, "pos")
    assert ev.screen(column)["passed"]
    result = ev.incremental(column)
    assert result["passed"], result["reason"]


def test_negative_control_fails(world, tmp_path):
    ev = Evaluator(world)
    column = _column(ev, synthetic.NEGATIVE_CONTROL, tmp_path, "neg")
    assert not ev.screen(column)["passed"]
    assert not ev.incremental(column)["passed"]


def test_runner_reports_crashing_code(world, tmp_path):
    ev = Evaluator(world)
    code = "def feature(x, fs, freqs, psd):\n    return 1.0 / 0\n"
    column, report = ev.compute_column(code, "discovery", tmp_path / "crash")
    assert column is None and "NaN" in report["reason"]


def test_scripted_loop_runs_every_path(world):
    config = {**world, "loop": {**world["loop"], "max_iterations": 3, "target_accepted": 5}}
    result = run_loop(config, provider="scripted", log=lambda *a: None)
    ledger = Ledger(Path(world["data"]["root"]) / "ledger.sqlite")
    rows = {r["name"]: r for r in ledger.rows("run_id = ?", (result["run_id"],))}
    assert set(rows) == {"frontal_eye_movement_index", "theta_burstiness", "sample_digit_entropy"}
    assert rows["theta_burstiness"]["status"] == "accepted"          # planted effect found + confirmed
    assert rows["sample_digit_entropy"]["status"] == "screen_failed"  # noise rejected
    assert "theta_burstiness" in result["accepted"]
    assert Path(result["report"]).is_file()
