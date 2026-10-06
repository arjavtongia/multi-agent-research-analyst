"""Researcher: one parallel task per sub-question. Search, fetch, extract grounded notes."""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlparse

from research_agents.deps import Deps
from research_agents.llm import StructuredOutputError, structured_call, system_and_user
from research_agents.nodes.aggregate import normalize_url
from research_agents.prompts import EXTRACTOR_SYSTEM
from research_agents.schemas import ExtractedNote, ExtractedNotes
from research_agents.state import Note, ResearchTask
from research_agents.tools.fetch import FetchError
from research_agents.tools.search import SearchError, SearchResult
from research_agents.tools.untrusted import wrap_untrusted

logger = logging.getLogger(__name__)

MIN_QUOTE_CHARS = 15
MIN_SNIPPET_CHARS = 80
_QUOTE_CHARS = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})
_ELLIPSIS_RE = re.compile(r"\s*(?:\.\.\.|…|\[\.\.\.\])\s*")


def _normalize_for_match(text: str) -> str:
    text = text.translate(_QUOTE_CHARS).replace(" ", " ").lower()
    return re.sub(r"\s+", " ", text).strip().strip("\"'")


def quote_in_text(quote: str, page_text: str) -> bool:
    """True if the quote (or every part around an ellipsis) appears verbatim in the page."""
    haystack = _normalize_for_match(page_text)
    parts = [p for p in _ELLIPSIS_RE.split(quote) if p.strip()]
    if not parts:
        return False
    normalized = [_normalize_for_match(p).strip(" .") for p in parts]
    if sum(len(p) for p in normalized) < MIN_QUOTE_CHARS:
        return False
    return all(p in haystack for p in normalized if p)


def ground_notes(
    notes: list[ExtractedNote], page_text: str
) -> tuple[list[ExtractedNote], list[ExtractedNote]]:
    """Split notes into (kept, dropped): a note is kept only if its quote is on the page."""
    kept, dropped = [], []
    for note in notes:
        if note.claim.strip() and quote_in_text(note.quote, page_text):
            kept.append(note)
        else:
            dropped.append(note)
    return kept, dropped


def _domain(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def select_results(results: list[SearchResult], blocked_domains: list[str]) -> list[SearchResult]:
    """Drop duplicate URLs and low-quality domains, keep search order."""
    blocked = {d.lower().removeprefix("www.") for d in blocked_domains}
    seen: set[str] = set()
    selected = []
    for result in results:
        domain = _domain(result.url)
        if not domain or any(domain == b or domain.endswith("." + b) for b in blocked):
            continue
        key = normalize_url(result.url)
        if key in seen:
            continue
        seen.add(key)
        selected.append(result)
    return selected


def extraction_prompt(task: ResearchTask, wrapped_page: str) -> str:
    sub_question = task["sub_question"]
    return (
        f"Overall research question: {task['question']}\n"
        f"Sub-question: {sub_question['question']}\n\n"
        f"{wrapped_page}"
    )


def research_sub_question(task: ResearchTask, deps: Deps) -> dict[str, Any]:
    settings = deps.settings
    sub_question = task["sub_question"]
    log: dict[str, Any] = {
        "sub_question_id": sub_question["id"],
        "question": sub_question["question"],
        "query": sub_question["search_query"],
        "kind": sub_question.get("kind", "plan"),
        "search_provider": getattr(deps.search, "name", "unknown"),
        "results": 0,
        "sources": [],
        "skipped": [],
        "notes": 0,
        "dropped_quotes": 0,
        "suspicious": [],
    }

    try:
        results = deps.search.search(
            sub_question["search_query"], max_results=settings.search_results_per_query
        )
    except SearchError as exc:
        log["error"] = str(exc)
        return {"notes": [], "research_log": [log]}
    log["results"] = len(results)

    documents: list[tuple[str, str, str, str]] = []  # (kind, url, title, text)
    failed: list[SearchResult] = []
    for result in select_results(results, settings.blocked_domains):
        if len(documents) >= settings.max_sources_per_question:
            break
        try:
            page = deps.fetcher.fetch(result.url)
        except FetchError as exc:
            log["skipped"].append({"url": result.url, "reason": str(exc)[:200]})
            failed.append(result)
            continue
        documents.append(("page", result.url, page.title or result.title, page.text))

    # Paywalled or script-rendered pages: fall back to the search snippet, flagged as such.
    for result in failed:
        if len(documents) >= settings.max_sources_per_question:
            break
        if len(result.snippet) >= MIN_SNIPPET_CHARS:
            documents.append(("snippet", result.url, result.title, result.snippet))

    system = EXTRACTOR_SYSTEM.format(max_notes=settings.max_notes_per_source)
    notes: list[Note] = []
    for kind, url, title, text in documents:
        # The title comes from the page too, so it goes inside the untrusted block.
        wrapped = wrap_untrusted(
            f"Title: {title}\n\n{text}", source=url, max_chars=settings.max_page_chars
        )
        if wrapped.suspicious_phrases:
            log["suspicious"].append({"url": url, "phrases": wrapped.suspicious_phrases})
        try:
            extracted = structured_call(
                deps.fast_llm,
                ExtractedNotes,
                system_and_user(system, extraction_prompt(task, wrapped.text)),
            )
        except StructuredOutputError as exc:
            log["skipped"].append({"url": url, "reason": f"extraction failed: {exc}"[:200]})
            continue
        kept, dropped = ground_notes(extracted.notes, text)
        log["dropped_quotes"] += len(dropped)
        kept = kept[: settings.max_notes_per_source]
        log["sources"].append({"url": url, "kind": kind, "notes": len(kept)})
        for note in kept:
            notes.append(
                Note(
                    note_id=f"{sub_question['id']}-n{len(notes) + 1}",
                    sub_question_id=sub_question["id"],
                    claim=note.claim.strip(),
                    quote=note.quote.strip(),
                    source_url=url,
                    source_title=title.strip() or _domain(url),
                    source_kind=kind,
                )
            )
    log["notes"] = len(notes)
    return {"notes": notes, "research_log": [log]}
