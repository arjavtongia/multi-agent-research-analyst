"""Writer: drafts the markdown report from the evidence, citing every factual sentence."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

from research_agents.citations import normalize_citations, strip_sources_section, strip_title
from research_agents.deps import Deps
from research_agents.llm import OutputTruncatedError, chat_call, system_and_user
from research_agents.prompts import WRITER_SYSTEM
from research_agents.tools.untrusted import CLOSE_TAG, OPEN_TAG, sanitize

logger = logging.getLogger(__name__)

TRUNCATION_NOTE = (
    "\n\nEvidence gap: this draft was cut off at the model's output limit, so the report is "
    "incomplete. Raise LLM_MAX_OUTPUT_TOKENS and revise."
)

_WHOLE_FENCE_RE = re.compile(r"^\s*```(?:markdown|md)?\s*\n(.*)\n```\s*$", re.DOTALL | re.I)


def select_evidence(evidence: list[dict[str, Any]], max_items: int) -> list[dict[str, Any]]:
    """Round-robin across sources so one long page cannot crowd out the rest."""
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in evidence:
        by_source[item["source_id"]].append(item)
    selected: list[dict[str, Any]] = []
    queues = [list(items) for items in by_source.values()]
    while queues and len(selected) < max_items:
        for queue in list(queues):
            if not queue:
                queues.remove(queue)
                continue
            selected.append(queue.pop(0))
            if len(selected) >= max_items:
                break
    return selected


def format_evidence(evidence: list[dict[str, Any]], sources: list[dict[str, Any]]) -> str:
    """Evidence block for prompts, grouped by source id and wrapped as untrusted data."""
    titles = {s["id"]: (s["title"], s["url"], s.get("kind", "page")) for s in sources}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in evidence:
        grouped[item["source_id"]].append(item)
    lines = [f"{OPEN_TAG} id=evidence>>>"]
    for sid in sorted(grouped, key=lambda s: int(s[1:])):
        title, url, kind = titles.get(sid, ("unknown", "", "page"))
        suffix = " (search snippet only)" if kind == "snippet" else ""
        lines.append(f"[{sid}] {sanitize(title)} <{url}>{suffix}")
        for item in grouped[sid]:
            lines.append(f"  - claim: {sanitize(item['claim'])}")
            lines.append(f'    quote: "{sanitize(item["quote"])}"')
    lines.append(f"{CLOSE_TAG} id=evidence>>>")
    return "\n".join(lines)


def clean_draft(text: str, valid_ids: set[str]) -> tuple[str, list[str]]:
    """Strip fences, title and any model-written sources list; canonicalise citations."""
    fenced = _WHOLE_FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1)
    text = strip_sources_section(strip_title(text))
    return normalize_citations(text, valid_ids)


def no_evidence_draft(outline: list[str]) -> str:
    sections = [
        f"## {heading}\n\nEvidence gap: no usable sources were found for this section."
        for heading in outline
    ]
    return "\n\n".join(sections)


def write_report(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    plan = state["plan"]
    sources = state.get("sources", [])
    evidence = state.get("evidence", [])
    version = state.get("draft_version", 0) + 1

    if not evidence:
        return {
            "draft": no_evidence_draft(plan["outline"]),
            "draft_version": version,
            "revision_feedback": "",
            "status": "fact_checking",
        }

    selected = select_evidence(evidence, deps.settings.max_evidence_for_writer)
    outline = "\n".join(f"{i}. {h}" for i, h in enumerate(plan["outline"], start=1))
    prompt = (
        f"Research question: {state['question']}\n"
        f"Report title (do not repeat it): {plan['title']}\n\n"
        f"Outline:\n{outline}\n\n"
        f"Evidence:\n{format_evidence(selected, sources)}"
    )
    feedback = state.get("revision_feedback", "").strip()
    previous = state.get("draft", "")
    if feedback and previous:
        prompt += (
            "\n\nThis is a revision. Previous draft:\n"
            f"<<<DRAFT>>>\n{previous}\n<<<END_DRAFT>>>\n\n"
            "Fix every item below. Rewrite or remove sentences you cannot support with the "
            "evidence, add missing citations, and keep everything else that was fine.\n"
            f"{feedback}"
        )

    try:
        text = chat_call(deps.llm, system_and_user(WRITER_SYSTEM, prompt), raise_on_truncation=True)
    except OutputTruncatedError as exc:
        # A cut-off draft reads as complete to the fact-checker (every remaining sentence can
        # be supported), so it must never replace a finished one silently.
        logger.warning("writer output truncated at the token limit (draft v%d)", version)
        if feedback and previous:
            return {
                "draft": previous,
                "draft_version": version,
                "revision_feedback": "",
                "status": "fact_checking",
            }
        text = exc.text + TRUNCATION_NOTE
    draft, _invalid = clean_draft(text, {s["id"] for s in sources})
    return {
        "draft": draft,
        "draft_version": version,
        "revision_feedback": "",
        "status": "fact_checking",
    }
