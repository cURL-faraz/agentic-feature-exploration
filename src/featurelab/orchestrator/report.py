"""Run reports and exports - deterministic, rendered from the ledger.

There is no LLM "report writer" on purpose: a template cannot hallucinate a
number.  Use an LLM where judgement or creativity is needed, not where a
format string does the job.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

STATUS_LABEL = {
    "accepted": "ACCEPTED", "confirm_failed": "failed lockbox", "incremental_failed": "no added value",
    "screen_failed": "screened out", "gate_failed": "code rejected", "rejected_by_critic": "critic rejected",
    "duplicate": "duplicate", "proposed": "interrupted", "error": "error",
}


def _fmt(value, digits=4):
    return "n/a" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))


def hypothesis_markdown(row: dict) -> str:
    proposal = row.get("proposal") or {}
    lines = [f"### #{row['id']} `{row.get('name')}` - {STATUS_LABEL.get(row.get('status'), row.get('status'))}",
             f"source: {row.get('source')} | models: {json.dumps(row.get('models') or {})}", ""]
    for key in ("hypothesis", "formula", "target_confusion"):
        if proposal.get(key):
            lines.append(f"- **{key.replace('_', ' ')}**: {proposal[key]}")
    if proposal.get("prediction"):
        lines.append(f"- **pre-registered prediction**: higher in {proposal['prediction'].get('higher_in')}, "
                     f"lower in {proposal['prediction'].get('lower_in')}")
    lines.append(f"- **outcome**: {row.get('reason') or ''}")
    screen = row.get("screen") if isinstance(row.get("screen"), dict) else None
    if screen:
        aucs = ", ".join(f"{k} {v['auc']}" for k, v in (screen.get("pair_auc") or {}).items())
        lines.append(f"- screen: NaN {screen.get('nan_fraction', 0):.1%}; most similar baseline "
                     f"{(screen.get('max_redundancy') or {}).get('feature')} "
                     f"(|rho|={(screen.get('max_redundancy') or {}).get('abs_spearman')}); AUCs: {aucs}; "
                     f"prediction held: {screen.get('preregistered_prediction_held')}")
    inc = row.get("incremental") if isinstance(row.get("incremental"), dict) else None
    if inc:
        lines.append(f"- incremental (discovery CV): mean gain in N1/N2/REM F1 {inc['mean_delta']:+.4f} "
                     f"(95% CI {[round(v, 4) for v in inc.get('ci95', [])]}), p={inc['p_value']:.2g} "
                     f"vs alpha {inc['alpha_per_test']:.2g}; improved {inc['n_improved']}/{inc['n_subjects']} "
                     f"subjects; kappa {inc['baseline']['kappa']:.3f} -> {inc['candidate']['kappa']:.3f}; "
                     f"pair errors {inc['pair_errors_baseline']} -> {inc['pair_errors_candidate']}")
    conf = row.get("confirm") if isinstance(row.get("confirm"), dict) else None
    if conf:
        for role in ("lockbox", "external"):
            if role in conf:
                c = conf[role]
                lines.append(f"- {role}: mean gain {c['mean_delta']:+.4f}, p={c['p_value']:.2g}, "
                             f"improved {c['n_improved']}/{c['n_subjects']} subjects, kappa "
                             f"{c['baseline']['kappa']:.3f} -> {c['candidate']['kappa']:.3f}")
    code = proposal.get("code")
    if code:
        lines += ["", "```python", code.strip(), "```"]
    return "\n".join(lines) + "\n"


def write_run_report(ctx, state) -> Path:
    rows = ctx.ledger.rows("run_id = ?", (ctx.run_id,))
    base = ctx.baseline.get("summary", {})
    calls = [dict(r) for r in ctx.ledger.db.execute(
        "SELECT role, model, ok, cached FROM llm_calls WHERE run_id = ?", (ctx.run_id,))]
    funnel = {}
    for row in rows:
        funnel[row["status"]] = funnel.get(row["status"], 0) + 1
    lines = [
        f"# featurelab run {ctx.run_id}", "",
        f"- protocol: `{ctx.protocol}` | stop reason: {state.get('stop_reason') or 'completed'}",
        f"- baseline (discovery CV): kappa {_fmt(base.get('kappa'), 3)}, macro-F1 {_fmt(base.get('macro_f1'), 3)}, "
        f"N1/N2/REM F1 {_fmt(base.get('target_f1'), 3)}, per class {base.get('per_class_f1')}",
        f"- hypotheses: {len(rows)} | funnel: " + ", ".join(f"{STATUS_LABEL.get(k, k)}={v}" for k, v in funnel.items()),
        f"- accepted: {', '.join(state.get('accepted') or []) or 'none'}",
        f"- LLM calls: {len(calls)} ({sum(1 for c in calls if c['cached'])} cached, "
        f"{sum(1 for c in calls if not c['ok'])} failed) | models: "
        + ", ".join(sorted({c['model'] for c in calls if c['ok']})),
        f"- Stage-B tests used under this protocol: {ctx.ledger.count_stage_b(ctx.protocol)}"
        f"/{ctx.config['evaluation']['incremental']['hypothesis_budget']} | lockbox looks: "
        f"{ctx.ledger.count_lockbox_looks(ctx.protocol)}/{ctx.config['evaluation']['confirm'].get('max_looks', 10)}",
        "", "## Hypotheses", "",
    ]
    lines += [hypothesis_markdown(row) for row in rows]
    path = ctx.run_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def export_features(config: dict, out_dir: Path, include_status=("accepted",), log=print) -> list[Path]:
    """Accepted features -> one CSV per cohort (+ the feature code), for ML/DL teammates."""
    from ..config import protocol_hash
    from ..data.cohort import STAGES, available_cohorts
    from ..evaluation.stages import Evaluator
    from .ledger import Ledger

    root = Path(config["data"]["root"])
    ledger = Ledger(root / "ledger.sqlite")
    marks = ",".join("?" * len(include_status))
    rows = ledger.rows(f"protocol = ? AND status IN ({marks})", (protocol_hash(config), *include_status))
    evaluator = Evaluator(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    code_dir = out_dir / "features"
    code_dir.mkdir(exist_ok=True)
    for row in rows:
        (code_dir / f"{row['name']}.py").write_text(
            f'"""{row["name"]} (featurelab #{row["id"]}, status {row["status"]})\n\n'
            f'{(row.get("proposal") or {}).get("hypothesis", "")}\n"""\n' + (row["proposal"] or {}).get("code", ""),
            encoding="utf-8")
    written = []
    for name in available_cohorts(root):
        cohort = evaluator.cohort(name)
        frame = pd.DataFrame({"subject": cohort.subject, "recording": cohort.recording,
                              "epoch_idx": cohort.epoch_idx, "stage": [STAGES[c] for c in cohort.y]})
        frame = pd.concat([frame, evaluator.baseline(name).add_prefix("base_")], axis=1)
        for row in rows:
            column, report = evaluator.compute_column(row["proposal"]["code"], name,
                                                      out_dir / "work" / row["name"])
            frame[f"new_{row['name']}"] = column if column is not None else np.nan
        path = out_dir / f"{name}_features.csv.gz"
        frame.to_csv(path, index=False)
        written.append(path)
        log(f"export: {path} ({len(frame)} epochs, {len(rows)} new features)")
    return written
