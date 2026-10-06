"""Graph state.

Values are plain JSON-friendly dicts and lists so every checkpoint can be stored in SQLite and
inspected by hand. Fields written by parallel researcher tasks use additive reducers.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class SubQuestion(TypedDict):
    id: str
    question: str
    search_query: str
    kind: str  # "plan" for planner sub-questions, "gap" for targeted fact-check follow-ups


class Note(TypedDict):
    note_id: str
    sub_question_id: str
    claim: str
    quote: str
    source_url: str
    source_title: str
    source_kind: str  # "page" (fetched and extracted) or "snippet" (search snippet fallback)


class ResearchState(TypedDict, total=False):
    thread_id: str
    question: str
    plan: dict[str, Any]
    # Written by parallel researcher tasks, so they accumulate.
    notes: Annotated[list[Note], operator.add]
    research_log: Annotated[list[dict[str, Any]], operator.add]
    # Written by aggregate (overwritten on every pass, ids stay stable).
    sources: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    # Writer / fact-checker loop.
    draft: str
    draft_version: int
    factcheck: dict[str, Any]
    factcheck_history: Annotated[list[dict[str, Any]], operator.add]
    revision_count: int
    gap_research_rounds: int
    revision_feedback: str
    gap_queries: list[str]
    route: str
    # Human review.
    review: dict[str, Any]
    human_revisions: int
    # Accounting and outcome.
    tokens_used: Annotated[int, operator.add]
    llm_calls: Annotated[int, operator.add]
    timings: Annotated[list[dict[str, Any]], operator.add]
    status: str
    stop_reason: str
    output_dir: str


class ResearchTask(TypedDict):
    """Payload sent to each parallel researcher via `Send`."""

    question: str
    sub_question: SubQuestion
