"""Planner: question -> focused sub-questions + report outline (structured output)."""

from __future__ import annotations

import logging
import re
from typing import Any

from research_agents.deps import Deps
from research_agents.llm import StructuredOutputError, structured_call, system_and_user
from research_agents.prompts import PLANNER_SYSTEM
from research_agents.schemas import Plan, PlannedSubQuestion
from research_agents.state import SubQuestion

logger = logging.getLogger(__name__)

DEFAULT_OUTLINE = ["Background", "Key findings", "Limitations of the evidence"]
_RESERVED_HEADINGS = re.compile(r"^(sources|references|bibliography|citations)$", re.IGNORECASE)


def fallback_plan(question: str) -> Plan:
    """Used when the model cannot produce a valid plan: research the question as-is."""
    return Plan(
        title=question.rstrip("?").strip()[:120] or "Research report",
        sub_questions=[PlannedSubQuestion(question=question)],
        outline=DEFAULT_OUTLINE,
    )


def _short_query(text: str, max_words: int = 12) -> str:
    words = re.sub(r"[\"?]", "", text).split()
    return " ".join(words[:max_words])


def normalize_plan(
    plan: Plan, question: str, max_sub_questions: int, origin: str
) -> dict[str, Any]:
    """Dedupe and cap sub-questions, assign ids q1..qn, and clean the outline."""
    seen: set[str] = set()
    sub_questions: list[SubQuestion] = []
    for item in plan.sub_questions:
        text = item.question.strip()
        key = re.sub(r"\W+", " ", text.lower()).strip()
        if not text or key in seen:
            continue
        seen.add(key)
        query = (item.search_query or "").strip() or _short_query(text)
        sub_questions.append(
            SubQuestion(
                id=f"q{len(sub_questions) + 1}",
                question=text,
                search_query=_short_query(query),
                kind="plan",
            )
        )
        if len(sub_questions) >= max_sub_questions:
            break
    if not sub_questions:
        sub_questions.append(
            SubQuestion(
                id="q1", question=question, search_query=_short_query(question), kind="plan"
            )
        )

    outline: list[str] = []
    for heading in plan.outline:
        heading = heading.strip().lstrip("#").strip()
        if heading and not _RESERVED_HEADINGS.match(heading) and heading not in outline:
            outline.append(heading)
    return {
        "title": plan.title.strip() or question,
        "sub_questions": sub_questions,
        "outline": outline[:8] or DEFAULT_OUTLINE,
        "origin": origin,
    }


def plan_question(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    question = state["question"].strip()
    settings = deps.settings
    system = PLANNER_SYSTEM.format(
        min_q=settings.min_sub_questions, max_q=settings.max_sub_questions
    )
    try:
        plan = structured_call(
            deps.llm, Plan, system_and_user(system, f"Research question: {question}")
        )
        origin = "model"
    except StructuredOutputError as exc:
        logger.warning("planner fell back to the single-question plan: %s", exc)
        plan = fallback_plan(question)
        origin = "fallback"
    return {
        "plan": normalize_plan(plan, question, settings.max_sub_questions, origin),
        "status": "researching",
    }
