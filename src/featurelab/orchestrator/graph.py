"""The discovery loop as an explicit LangGraph state machine.

    scout -> baseline -> invent -> critique -> gate -> screen -> incremental -> confirm -> record
                           ^         |  revise    |  repair    | fail       | fail          |
                           |---------+------------+            +------------+-------> record |
                           +------------------------------- decide (budget / target) <------+
                                                                  |
                                                                finish -> report

Why a graph and not a ``while`` loop with LLM calls inside?
* **Control flow is code, not a prompt.**  Routing decisions (retry, revise,
  stop) are small pure functions you can unit-test and read.  The LLM never
  decides whether its own idea passed.
* **Bounded autonomy.**  Every cycle has a counter: repairs, revisions,
  iterations, LLM calls, lockbox looks.  An agent loop without bounds is a
  bill, not a system.
* **State is small and serialisable.**  Big things (feature columns, reports)
  live on disk; the state carries their paths.  That keeps checkpoints cheap and
  makes a run resumable/inspectable.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from ..agents import roles
from ..agents.literature import build_inspiration
from ..agents.schemas import FeatureProposal, InspirationCard
from ..config import protocol_hash, runs_dir
from ..evaluation.stages import Evaluator
from ..llm.client import DailyCapReached, LLMError, build_client
from .ledger import Ledger, code_hash
from .report import write_run_report
from .steps import confirm as confirm_step
from .steps import gate_and_run


class LoopState(TypedDict, total=False):
    run_id: str
    iteration: int
    hypothesis_id: int
    proposal: dict
    feedback: str
    repairs: int
    revisions: int
    status: str
    reason: str
    step: dict
    accepted: list
    stop_reason: str
    cards_used: int


class Context:
    """Everything nodes share that is *not* state: services and configuration."""

    def __init__(self, config: dict, provider: str | None, log=print, skip_scout: bool = False):
        self.config = config
        self.log = log
        self.protocol = protocol_hash(config)
        self.run_id = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = runs_dir(config) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger(Path(config["data"]["root"]) / "ledger.sqlite")
        self.client = build_client(config, on_call=lambda rec: self.ledger.log_call(self.run_id, rec),
                                   provider=provider)
        self.evaluator = Evaluator(config)
        self.cards: list[InspirationCard] = []
        self.skip_scout = skip_scout
        self.baseline_features: list[str] = []
        self.baseline: dict = {}

    def workdir(self, state: LoopState) -> Path:
        return self.run_dir / f"h{state['hypothesis_id']:04d}"


def build_graph(ctx: Context):
    from langgraph.graph import END, StateGraph

    cfg_loop = ctx.config["loop"]
    log = ctx.log

    # ----------------------------------------------------------------- nodes #
    def scout(state: LoopState) -> dict:
        if ctx.skip_scout:
            return {}
        try:
            ctx.cards = build_inspiration(ctx.config, ctx.client, log=log)
        except Exception as exc:  # noqa: BLE001 - literature is inspiration, not a dependency
            log(f"scout unavailable ({exc}); continuing without literature")
        return {}

    def baseline(state: LoopState) -> dict:
        log("baseline: cross-validating the standard feature set (cached after the first run)...")
        result = ctx.evaluator.baseline_cv()
        ctx.baseline = result
        ctx.baseline_features = list(ctx.evaluator.baseline("discovery").columns)
        s = result["summary"]
        log(f"baseline: kappa={s['kappa']:.3f} macroF1={s['macro_f1']:.3f} "
            f"target(N1/N2/REM)F1={s['target_f1']:.3f} per-class={s['per_class_f1']}")
        return {"accepted": [], "iteration": 0}

    def invent(state: LoopState) -> dict:
        n_cards = 4
        start = (state.get("cards_used") or 0) % max(1, len(ctx.cards) or 1)
        cards = (ctx.cards[start:] + ctx.cards[:start])[:n_cards] if ctx.cards else []
        brief = roles.briefing(ctx.baseline["summary"], ctx.baseline["pair_errors"], cards,
                               ctx.ledger.history_for_inventor(ctx.protocol))
        previous = FeatureProposal.model_validate(state["proposal"]) if state.get("feedback") else None
        try:
            proposal, model = roles.invent(ctx.client, ctx.baseline_features, brief,
                                           feedback=state.get("feedback", ""), previous=previous)
        except DailyCapReached as exc:
            return {"stop_reason": str(exc)}
        except (LLMError, ValueError) as exc:
            log(f"inventor failed: {exc}")
            return {"stop_reason": f"inventor unavailable: {exc}"}
        update: dict[str, Any] = {"proposal": proposal.model_dump(), "feedback": "", "status": "proposed",
                                  "cards_used": start + n_cards}
        if previous is None:  # a brand-new hypothesis (not a repair/revision of the current one)
            iteration = int(state.get("iteration") or 0) + 1
            hid = ctx.ledger.add(run_id=ctx.run_id, protocol=ctx.protocol, source="llm", name=proposal.name,
                                 code_hash=code_hash(proposal.code), proposal=proposal.model_dump(),
                                 models={"inventor": model}, status="proposed")
            update.update(hypothesis_id=hid, iteration=iteration, repairs=0, revisions=0)
            log(f"\n[{iteration}/{cfg_loop['max_iterations']}] inventor ({model}) proposes "
                f"`{proposal.name}`: {proposal.hypothesis[:140]}")
        else:
            ctx.ledger.update(state["hypothesis_id"], name=proposal.name, code_hash=code_hash(proposal.code),
                              proposal=proposal.model_dump())
            log(f"    inventor revised `{proposal.name}`")
        seen = ctx.ledger.seen(ctx.protocol, code_hash(proposal.code))
        if seen and seen["id"] != update.get("hypothesis_id", state.get("hypothesis_id")):
            update.update(status="duplicate", reason=f"identical code already evaluated as #{seen['id']}")
        return update

    def critique(state: LoopState) -> dict:
        proposal = FeatureProposal.model_validate(state["proposal"])
        try:
            review, model = roles.critique(ctx.client, ctx.baseline_features, proposal)
        except DailyCapReached as exc:
            return {"stop_reason": str(exc)}
        except (LLMError, ValueError) as exc:
            log(f"    critic unavailable ({exc}); statistics will decide")
            return {"status": "approved"}
        log(f"    critic ({model}): {review.verdict} {('- ' + '; '.join(review.issues)[:200]) if review.issues else ''}")
        if review.verdict == "approve":
            return {"status": "approved"}
        if review.verdict == "revise" and int(state.get("revisions") or 0) < int(cfg_loop["max_revisions"]):
            return {"status": "revise", "revisions": int(state.get("revisions") or 0) + 1,
                    "feedback": "Reviewer: " + "; ".join(review.issues) + " " + review.suggestion}
        if review.verdict == "reject":
            return {"status": "rejected_by_critic", "reason": "; ".join(review.issues)[:400]}
        return {"status": "approved"}  # revisions exhausted: let the statistics decide

    def gate(state: LoopState) -> dict:
        proposal = FeatureProposal.model_validate(state["proposal"])
        step = gate_and_run(ctx.evaluator, proposal.code, ctx.workdir(state))
        if step["ok"]:
            r = step["runner"]
            log(f"    gate: ok ({r['ms_per_epoch']} ms/epoch, scale-invariant={r.get('scale_invariant')})")
            return {"status": "ran", "step": step}
        log(f"    gate: FAILED - {step['reason'][:200]}")
        if int(state.get("repairs") or 0) < int(cfg_loop["max_repairs"]):
            return {"status": "repair", "repairs": int(state.get("repairs") or 0) + 1,
                    "feedback": f"Your code failed the checks: {step['reason'][:1200]}", "step": step}
        return {"status": "gate_failed", "reason": step["reason"], "step": step}

    def screen(state: LoopState) -> dict:
        proposal = FeatureProposal.model_validate(state["proposal"])
        column = np.load(state["step"]["column_path"])
        result = ctx.evaluator.screen(column, proposal.prediction.model_dump())
        ctx.ledger.update(state["hypothesis_id"], screen=result, runner=state["step"].get("runner"))
        best = max(result.get("pair_auc", {}).items(), key=lambda kv: kv[1]["shift"], default=(None, {}))
        log(f"    screen: {'pass' if result['passed'] else 'FAIL'} - {result['reason'][:160]}"
            + (f" | best pair {best[0]} AUC={best[1].get('auc')}" if best[0] else ""))
        return {"status": "screened" if result["passed"] else "screen_failed", "reason": result["reason"]}

    def incremental(state: LoopState) -> dict:
        proposal = FeatureProposal.model_validate(state["proposal"])
        used = ctx.ledger.count_stage_b(ctx.protocol)
        budget = int(ctx.config["evaluation"]["incremental"]["hypothesis_budget"])
        if used >= budget:
            return {"status": "error", "reason": "hypothesis budget exhausted", "stop_reason":
                    f"Stage-B budget of {budget} tests used: start a new protocol (fresh data split)"}
        log("    incremental: baseline vs baseline+feature, subject-grouped CV (takes a few minutes)...")
        result = ctx.evaluator.incremental(np.load(state["step"]["column_path"]), proposal.name)
        ctx.ledger.update(state["hypothesis_id"], incremental=result)
        log(f"    incremental: {'pass' if result['passed'] else 'FAIL'} - mean gain {result['mean_delta']:+.4f} "
            f"(95% CI {result.get('ci95')}), p={result['p_value']:.2g}, improved in "
            f"{result['n_improved']}/{result['n_subjects']} subjects | pairs "
            f"{result['pair_errors_baseline']} -> {result['pair_errors_candidate']}")
        return {"status": "incremental_passed" if result["passed"] else "incremental_failed",
                "reason": result["reason"]}

    def confirm(state: LoopState) -> dict:
        proposal = FeatureProposal.model_validate(state["proposal"])
        log("    confirm: opening the lockbox (unseen subjects) + external cohort...")
        out = confirm_step(ctx.evaluator, ctx.ledger, ctx.protocol, proposal.code, ctx.workdir(state),
                           np.load(state["step"]["column_path"]), proposal.name)
        if out.get("confirm"):
            ctx.ledger.update(state["hypothesis_id"], confirm=out["confirm"])
        log(f"    confirm: {'ACCEPTED' if out['ok'] else 'not confirmed'} - {out['reason']}")
        update: dict[str, Any] = {"status": "accepted" if out["ok"] else "confirm_failed", "reason": out["reason"]}
        if out["ok"]:
            update["accepted"] = [*(state.get("accepted") or []), proposal.name]
        if out.get("exhausted"):
            update["stop_reason"] = out["reason"]
        return update

    def record(state: LoopState) -> dict:
        stage = {"rejected_by_critic": "critic", "gate_failed": "gate", "screen_failed": "screen",
                 "incremental_failed": "incremental", "confirm_failed": "confirm", "accepted": "confirm",
                 "duplicate": "dedup"}.get(state.get("status", ""), state.get("status"))
        ctx.ledger.update(state["hypothesis_id"], status=state.get("status"), stage_reached=stage,
                          reason=state.get("reason", ""))
        update: dict[str, Any] = {"feedback": "", "step": {}}
        if not state.get("stop_reason"):  # stop rules live in a node: routers must not mutate state
            if len(state.get("accepted") or []) >= int(cfg_loop["target_accepted"]):
                update["stop_reason"] = "target number of accepted features reached"
            elif int(state.get("iteration") or 0) >= int(cfg_loop["max_iterations"]):
                update["stop_reason"] = f"iteration budget ({cfg_loop['max_iterations']}) used"
        return update

    def finish(state: LoopState) -> dict:
        path = write_run_report(ctx, state)
        log(f"\nrun {ctx.run_id} finished: {state.get('stop_reason') or 'done'}\nreport: {path}")
        return {}

    # ---------------------------------------------------------------- routing #
    def after_invent(state: LoopState) -> str:
        if state.get("stop_reason"):
            return "finish"
        return "record" if state.get("status") == "duplicate" else "critique"

    def after_critique(state: LoopState) -> str:
        if state.get("stop_reason"):
            return "finish"
        return {"approved": "gate", "revise": "invent"}.get(state.get("status", ""), "record")

    def after_gate(state: LoopState) -> str:
        return {"ran": "screen", "repair": "invent"}.get(state.get("status", ""), "record")

    def after_screen(state: LoopState) -> str:
        return "incremental" if state.get("status") == "screened" else "record"

    def after_incremental(state: LoopState) -> str:
        return "confirm" if state.get("status") == "incremental_passed" else "record"

    def decide(state: LoopState) -> str:
        return "finish" if state.get("stop_reason") else "invent"

    graph = StateGraph(LoopState)
    for name, fn in [("scout", scout), ("baseline", baseline), ("invent", invent), ("critique", critique),
                     ("gate", gate), ("screen", screen), ("incremental", incremental), ("confirm", confirm),
                     ("record", record), ("finish", finish)]:
        graph.add_node(name, fn)
    graph.set_entry_point("scout")
    graph.add_edge("scout", "baseline")
    graph.add_edge("baseline", "invent")
    graph.add_conditional_edges("invent", after_invent, ["critique", "record", "finish"])
    graph.add_conditional_edges("critique", after_critique, ["gate", "invent", "record", "finish"])
    graph.add_conditional_edges("gate", after_gate, ["screen", "invent", "record"])
    graph.add_conditional_edges("screen", after_screen, ["incremental", "record"])
    graph.add_conditional_edges("incremental", after_incremental, ["confirm", "record"])
    graph.add_edge("confirm", "record")
    graph.add_conditional_edges("record", decide, ["invent", "finish"])
    graph.add_edge("finish", END)
    return graph.compile()


def run_loop(config: dict, provider: str | None = None, log=print, skip_scout: bool = False) -> dict:
    ctx = Context(config, provider, log=log, skip_scout=skip_scout)
    log(f"run {ctx.run_id} | protocol {ctx.protocol} | provider {provider or config['llm']['provider']}")
    app = build_graph(ctx)
    final = app.invoke({"run_id": ctx.run_id}, config={"recursion_limit": 40 * int(config["loop"]["max_iterations"]) + 20})
    return {"run_id": ctx.run_id, "accepted": final.get("accepted", []), "stop_reason": final.get("stop_reason"),
            "report": str(ctx.run_dir / "report.md")}


def evaluate_human(config: dict, code: str, name: str, meta: dict | None = None, log=print) -> dict:
    """Same judge, entered by a human: gate -> screen -> incremental -> (confirm)."""
    from .report import hypothesis_markdown

    ctx_root = Path(config["data"]["root"])
    protocol = protocol_hash(config)
    ledger = Ledger(ctx_root / "ledger.sqlite")
    evaluator = Evaluator(config)
    run_id = "human-" + time.strftime("%Y%m%d-%H%M%S")
    digest = code_hash(code)
    previous = ledger.seen(protocol, digest)
    if previous:
        log(f"identical code was already evaluated as #{previous['id']} ({previous['status']}); showing that result")
        return previous
    proposal = {"name": name, **(meta or {})}
    hid = ledger.add(run_id=run_id, protocol=protocol, source="human", name=name, code_hash=digest,
                     proposal=proposal, status="proposed")
    workdir = runs_dir(config) / run_id / f"h{hid:04d}"
    step = gate_and_run(evaluator, code, workdir)
    if not step["ok"]:
        ledger.update(hid, status="gate_failed", stage_reached="gate", reason=step["reason"])
        log("gate FAILED: " + step["reason"])
        return ledger.rows("id = ?", (hid,))[0]
    log(f"runner: {json.dumps(step['runner'])}")
    column = np.load(step["column_path"])
    screen = evaluator.screen(column, (meta or {}).get("prediction"))
    ledger.update(hid, screen=screen, runner=step["runner"])
    log(f"screen: {screen['reason']}  pair AUCs: {json.dumps(screen.get('pair_auc'))}")
    status, reason = ("screen_failed", screen["reason"])
    if screen["passed"]:
        log("incremental: running subject-grouped CV...")
        inc = evaluator.incremental(column, name)
        ledger.update(hid, incremental=inc)
        log(f"incremental: {inc['reason']} | mean gain {inc['mean_delta']:+.4f} p={inc['p_value']:.2g}")
        status, reason = ("incremental_failed", inc["reason"])
        if inc["passed"]:
            out = confirm_step(evaluator, ledger, protocol, code, workdir, column, name)
            if out.get("confirm"):
                ledger.update(hid, confirm=out["confirm"])
            status, reason = ("accepted" if out["ok"] else "confirm_failed", out["reason"])
            log("confirm: " + reason)
    ledger.update(hid, status=status, stage_reached=status, reason=reason)
    row = ledger.rows("id = ?", (hid,))[0]
    (workdir / "result.md").write_text(hypothesis_markdown(row), encoding="utf-8")
    log(f"result written to {workdir / 'result.md'}")
    return row
