"""Pydantic schemas for structured LLM outputs and the human review decision.

Graph state itself holds plain dicts (JSON-friendly for the SQLite checkpointer); these models
are used at the boundaries where data comes back from a model or a person.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PlannedSubQuestion(BaseModel):
    question: str = Field(description="A focused, answerable sub-question.")
    search_query: str | None = Field(
        default=None, description="A short web search query (3-8 words) for this sub-question."
    )


class Plan(BaseModel):
    """Planner output: sub-questions to research plus the report outline."""

    title: str = Field(description="A short title for the report.")
    sub_questions: list[PlannedSubQuestion] = Field(min_length=1)
    outline: list[str] = Field(
        min_length=1, description="Section headings for the report, in order."
    )

    @field_validator("sub_questions", mode="before")
    @classmethod
    def _coerce_strings(cls, value: object) -> object:
        # Open models often return a list of strings instead of objects.
        if isinstance(value, list):
            return [{"question": v} if isinstance(v, str) else v for v in value]
        return value


class ExtractedNote(BaseModel):
    claim: str = Field(description="One factual claim, paraphrased in a single sentence.")
    quote: str = Field(description="An exact, verbatim quote from the page supporting the claim.")


class ExtractedNotes(BaseModel):
    """Researcher output for one page."""

    notes: list[ExtractedNote] = Field(default_factory=list)


Verdict = Literal["supported", "unsupported", "missing_citation"]


class SentenceVerdict(BaseModel):
    sentence_id: int
    verdict: Literal["supported", "unsupported"]
    reason: str = ""
    missing_evidence_query: str | None = Field(
        default=None,
        description="If unsupported because evidence is missing, a web search query that "
        "could find it.",
    )

    @field_validator("verdict", mode="before")
    @classmethod
    def _normalise_verdict(cls, value: object) -> object:
        # Map variants such as "Supported", "partially supported", "not supported".
        if isinstance(value, str):
            v = value.strip().lower().replace("-", " ").replace("_", " ")
            return "supported" if v in {"supported", "support", "yes", "true"} else "unsupported"
        if isinstance(value, bool):
            return "supported" if value else "unsupported"
        return value


class FactCheckOutput(BaseModel):
    verdicts: list[SentenceVerdict] = Field(default_factory=list)


class ReviewDecision(BaseModel):
    """What a human sends back when resuming from the review interrupt."""

    action: Literal["approve", "revise"]
    feedback: str = ""

    @field_validator("feedback")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()
