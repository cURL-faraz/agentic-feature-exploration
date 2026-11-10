"""Typed contracts between agents.

Agents talk to each other through *validated objects*, never through free text.
If a model's reply does not parse into one of these, it is sent back once with
the validation error ("repair"), and dropped if it still fails.  This is what
stops one sloppy reply from corrupting everything downstream.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

STAGES = ["W", "N1", "N2", "N3", "REM"]


class InspirationCard(BaseModel):
    """One idea extracted from one real paper (grounded: paper_id must exist)."""

    paper_id: str
    idea: str = Field(description="the EEG phenomenon or measurement idea, 1-2 sentences")
    mechanism: str = Field(default="", description="why it should differ between stages")
    target_confusion: str = Field(default="", description="e.g. 'N1 vs REM'")


class InspirationBatch(BaseModel):
    cards: list[InspirationCard]


class Prediction(BaseModel):
    higher_in: str
    lower_in: str

    @field_validator("higher_in", "lower_in")
    @classmethod
    def _stage(cls, value: str) -> str:
        value = value.strip().upper().replace("STAGE ", "")
        value = {"WAKE": "W", "R": "REM", "S1": "N1", "S2": "N2"}.get(value, value)
        if value not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        return value


class FeatureProposal(BaseModel):
    """What the inventor must commit to *before* seeing any result (pre-registration)."""

    name: str
    hypothesis: str = Field(description="what physiological difference the feature captures")
    formula: str = Field(description="mathematical definition (LaTeX or plain math)")
    target_confusion: str = Field(description="which confused pair it should fix, e.g. 'N1 vs REM'")
    prediction: Prediction
    inspired_by: list[str] = Field(default_factory=list, description="paper_ids of inspiration cards")
    code: str = ""

    @field_validator("name")
    @classmethod
    def _snake(cls, value: str) -> str:
        value = re.sub(r"[^a-z0-9_]+", "_", value.strip().lower()).strip("_")
        if not value:
            raise ValueError("name must be a non-empty identifier")
        return value[:60]


class CriticReview(BaseModel):
    verdict: str = Field(description="approve | revise | reject")
    code_matches_formula: bool = True
    issues: list[str] = Field(default_factory=list)
    suggestion: str = ""

    @field_validator("verdict")
    @classmethod
    def _verdict(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"approve", "revise", "reject"}:
            raise ValueError("verdict must be approve, revise or reject")
        return value
