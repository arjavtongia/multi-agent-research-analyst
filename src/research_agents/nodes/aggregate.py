"""Aggregate: merge researcher notes, dedupe sources and assign stable ids [S1]..[Sn]."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse

_TRACKING_PARAMS = re.compile(r"^(utm_.*|fbclid|gclid|mc_cid|mc_eid|ref|ref_src|igshid)$", re.I)


def normalize_url(url: str) -> str:
    """Canonical key for deduping: no scheme, no www, no fragment, no tracking params."""
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if parsed.port and parsed.port not in (80, 443):
        host = f"{host}:{parsed.port}"
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if len(path) > 1:
        path = path.rstrip("/")
    query = urlencode(
        sorted((k, v) for k, v in parse_qsl(parsed.query) if not _TRACKING_PARAMS.match(k))
    )
    return f"{host}{path}" + (f"?{query}" if query else "")


def _quote_key(quote: str) -> str:
    return re.sub(r"\W+", " ", quote.lower()).strip()


def aggregate_notes(state: dict[str, Any]) -> dict[str, Any]:
    """Assign source ids. Existing ids never change, so a second pass (after targeted
    research) only appends new sources and citations in the current draft stay valid."""
    # Copy, never mutate the incoming state (including the nested id lists).
    sources: list[dict[str, Any]] = [
        {**s, "sub_question_ids": list(s.get("sub_question_ids", []))}
        for s in state.get("sources", [])
    ]
    by_key = {normalize_url(s["url"]): s for s in sources}
    evidence: list[dict[str, Any]] = []
    seen_quotes: set[tuple[str, str]] = set()

    for note in state.get("notes", []):
        key = normalize_url(note["source_url"])
        source = by_key.get(key)
        if source is None:
            source = {
                "id": f"S{len(sources) + 1}",
                "url": note["source_url"],
                "title": note["source_title"],
                "kind": note["source_kind"],
                "sub_question_ids": [],
            }
            sources.append(source)
            by_key[key] = source
        elif source.get("kind") == "snippet" and note["source_kind"] == "page":
            source["kind"] = "page"
        if note["sub_question_id"] not in source["sub_question_ids"]:
            source["sub_question_ids"].append(note["sub_question_id"])

        dedupe_key = (source["id"], _quote_key(note["quote"]))
        if dedupe_key in seen_quotes:
            continue
        seen_quotes.add(dedupe_key)
        evidence.append({**note, "source_id": source["id"]})

    counts: dict[str, int] = {}
    for item in evidence:
        counts[item["source_id"]] = counts.get(item["source_id"], 0) + 1
    for source in sources:
        source["note_count"] = counts.get(source["id"], 0)
    return {"sources": sources, "evidence": evidence, "status": "writing"}
