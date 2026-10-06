"""Fact-checker: verdict per factual sentence (supported / unsupported / missing citation).

Uncited factual sentences and citations to unknown sources are caught deterministically. Only
sentences with valid citations go to the model, together with the notes from the sources they
cite, in batches.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from research_agents.citations import Sentence, split_sentences
from research_agents.deps import Deps
from research_agents.llm import StructuredOutputError, structured_call, system_and_user
from research_agents.prompts import FACT_CHECKER_SYSTEM
from research_agents.schemas import FactCheckOutput
from research_agents.tools.untrusted import CLOSE_TAG, OPEN_TAG, sanitize

MAX_NOTES_PER_SOURCE_IN_CHECK = 8


def _verdict(sentence: Sentence, verdict: str, reason: str, query: str | None = None) -> dict:
    return {
        "sentence_id": sentence.id,
        "sentence": sentence.text,
        "section": sentence.section,
        "citations": sentence.citations,
        "verdict": verdict,
        "reason": reason,
        "missing_evidence_query": query,
    }


def build_check_prompt(batch: list[Sentence], evidence: list[dict[str, Any]]) -> str:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in evidence:
        by_source[item["source_id"]].append(item)
    cited = sorted({c for s in batch for c in s.citations}, key=lambda c: int(c[1:]))
    lines = ["Sentences to check:"]
    for sentence in batch:
        lines.append(f"{sentence.id}. {sentence.text}")
    lines.append("")
    lines.append(f"{OPEN_TAG} id=evidence>>>")
    for sid in cited:
        lines.append(f"[{sid}]")
        notes = by_source.get(sid, [])[:MAX_NOTES_PER_SOURCE_IN_CHECK]
        if not notes:
            lines.append("  (no notes collected from this source)")
        for note in notes:
            lines.append(f"  - claim: {sanitize(note['claim'])}")
            lines.append(f'    quote: "{sanitize(note["quote"])}"')
    lines.append(f"{CLOSE_TAG} id=evidence>>>")
    return "\n".join(lines)


def summarize(verdicts: list[dict[str, Any]], draft_version: int) -> dict[str, Any]:
    counts = {"supported": 0, "unsupported": 0, "missing_citation": 0}
    for v in verdicts:
        counts[v["verdict"]] += 1
    factual = len(verdicts)
    cited = counts["supported"] + counts["unsupported"]
    return {
        "draft_version": draft_version,
        "factual_sentences": factual,
        **counts,
        "citation_coverage": round(cited / factual, 4) if factual else None,
        "supported_rate": round(counts["supported"] / cited, 4) if cited else None,
    }


def fact_check(state: dict[str, Any], deps: Deps) -> dict[str, Any]:
    draft = state.get("draft", "")
    evidence = state.get("evidence", [])
    valid_ids = {s["id"] for s in state.get("sources", [])}
    version = state.get("draft_version", 0)

    verdicts: list[dict[str, Any]] = []
    to_check: list[Sentence] = []
    for sentence in split_sentences(draft):
        if not sentence.factual:
            continue
        unknown = [c for c in sentence.citations if c not in valid_ids]
        if not sentence.citations:
            verdicts.append(
                _verdict(sentence, "missing_citation", "Factual sentence without a citation.")
            )
        elif unknown:
            verdicts.append(
                _verdict(sentence, "unsupported", f"Cites unknown source(s): {', '.join(unknown)}.")
            )
        else:
            to_check.append(sentence)

    size = deps.settings.factcheck_batch_size
    for start in range(0, len(to_check), size):
        batch = to_check[start : start + size]
        try:
            output = structured_call(
                deps.llm,
                FactCheckOutput,
                system_and_user(FACT_CHECKER_SYSTEM, build_check_prompt(batch, evidence)),
            )
            by_id = {v.sentence_id: v for v in output.verdicts}
        except StructuredOutputError:
            by_id = {}
        for sentence in batch:
            result = by_id.get(sentence.id)
            if result is None:
                verdicts.append(
                    _verdict(sentence, "unsupported", "The fact-checker returned no verdict.")
                )
            else:
                verdicts.append(
                    _verdict(
                        sentence,
                        result.verdict,
                        result.reason.strip(),
                        (result.missing_evidence_query or "").strip() or None,
                    )
                )

    verdicts.sort(key=lambda v: v["sentence_id"])
    summary = summarize(verdicts, version)
    return {
        "factcheck": {"draft_version": version, "verdicts": verdicts, "summary": summary},
        "factcheck_history": [summary],
    }
