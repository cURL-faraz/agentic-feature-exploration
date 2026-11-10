"""The three LLM agents: Scout, Inventor, Critic.

Each agent is a *function* from typed input to typed output, with one LLM call
(plus at most one repair call).  No agent decides control flow - the
orchestrator does.  That keeps every agent independently testable with a
scripted client, and keeps the system's behaviour inspectable.
"""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from ..llm.client import extract_json
from . import prompts
from .schemas import CriticReview, FeatureProposal, InspirationBatch, InspirationCard


def _structured(client, role: str, system: str, user: str, parse):
    """Ask, validate, and on failure ask once more with the exact error."""
    reply = client.chat(role, system, user)
    try:
        return parse(reply.text), reply
    except (ValueError, ValidationError) as exc:
        repair = (f"{user}\n\nYour previous reply could not be used: {str(exc)[:600]}\n"
                  "Reply again, following the output format exactly.")
        reply = client.chat(role, system, repair)
        return parse(reply.text), reply


# ------------------------------------------------------------------- scout #
def scout_cards(client, papers: list[dict]) -> list[InspirationCard]:
    listing = "\n\n".join(
        f"paper_id: {p['id']}\ntitle: {p['title']} ({p['year']}, {p['cited_by']} citations)\n"
        f"abstract: {p['abstract'][:1800]}" for p in papers
    )
    batch, _ = _structured(client, "scout", prompts.SCOUT, listing,
                           lambda text: InspirationBatch.model_validate(extract_json(text)))
    known = {p["id"] for p in papers}
    # Grounding check: a card that cites a paper we did not give it is a hallucination.
    return [card for card in batch.cards if card.paper_id in known]


# ---------------------------------------------------------------- inventor #
def _parse_proposal(text: str) -> FeatureProposal:
    code_match = re.findall(r"```python\s*(.*?)```", text, flags=re.S)
    meta = extract_json(re.sub(r"```python.*?```", "", text, flags=re.S))
    proposal = FeatureProposal.model_validate(meta)
    code = code_match[-1].strip() if code_match else str(meta.get("code", "")).strip()
    if "def feature" not in code:
        raise ValueError("no python code block defining `def feature(x, fs, freqs, psd)`")
    return proposal.model_copy(update={"code": code})


def invent(client, baseline_features: list[str], briefing: str, feedback: str = "",
           previous: FeatureProposal | None = None) -> tuple[FeatureProposal, str]:
    user = briefing
    if previous is not None and feedback:
        user += ("\n\n## Revise your previous proposal\n"
                 f"Previous proposal:\n```json\n{previous.model_dump_json(exclude={'code'}, indent=1)}\n```\n"
                 f"```python\n{previous.code}\n```\nProblem to fix:\n{feedback}\n"
                 "Return the corrected proposal in the same format.")
    proposal, reply = _structured(client, "inventor", prompts.inventor_system(baseline_features), user,
                                  _parse_proposal)
    return proposal, reply.model


def briefing(baseline_summary: dict, pair_errors: dict, cards: list[InspirationCard],
             history: list[dict]) -> str:
    """What the inventor is told about the world.  Coarse on purpose (see docs)."""
    lines = ["## Where the current single-channel model fails (discovery data, cross-validated)",
             f"Per-stage F1: {json.dumps(baseline_summary['per_class_f1'])}",
             f"Confusion counts between stage pairs (epochs): {json.dumps(pair_errors)}", ""]
    if cards:
        lines.append("## Inspiration from the literature (use, combine or ignore)")
        lines += [f"- [{c.paper_id}] {c.idea} (why: {c.mechanism}; targets {c.target_confusion})" for c in cards]
        lines.append("")
    if history:
        lines.append("## Previous attempts in this project (do not repeat them; learn from why they failed)")
        for h in history:
            lines.append(f"- {h['name']}: {h['outcome']}")
        lines.append("")
    lines.append("Invent ONE new feature now.")
    return "\n".join(lines)


# ------------------------------------------------------------------ critic #
def critique(client, baseline_features: list[str], proposal: FeatureProposal) -> tuple[CriticReview, str]:
    user = (f"Proposal:\n```json\n{proposal.model_dump_json(exclude={'code'}, indent=1)}\n```\n"
            f"Code:\n```python\n{proposal.code}\n```")
    review, reply = _structured(client, "critic", prompts.critic_system(baseline_features), user,
                                lambda text: CriticReview.model_validate(extract_json(text)))
    return review, reply.model
