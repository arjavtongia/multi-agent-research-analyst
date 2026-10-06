"""Finalize: write report.md, sources.json and factcheck.json under runs/<thread_id>/."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research_agents.citations import find_citations
from research_agents.config import Settings
from research_agents.nodes.router import budget_exceeded
from research_agents.tracing import run_dir


def outcome(state: dict[str, Any], settings: Settings) -> tuple[str, str]:
    """(status, stop_reason) for the run."""
    if state.get("review", {}).get("action") == "approve":
        return "completed", ""
    if budget_exceeded(state, settings):
        return "partial", "token_budget"
    return "partial", "stopped_before_approval"


def partial_from_evidence(state: dict[str, Any]) -> str:
    """When the budget ran out before a draft existed, keep the collected evidence."""
    lines = ["## Collected evidence (not yet written up)", ""]
    for item in state.get("evidence", []):
        lines.append(f"- {item['claim']} [{item['source_id']}]")
    if len(lines) == 2:
        lines.append("No evidence was collected before the run stopped.")
    return "\n".join(lines)


def render_report(state: dict[str, Any], status: str, stop_reason: str, settings: Settings) -> str:
    plan = state.get("plan", {})
    title = plan.get("title") or state.get("question", "Research report")
    draft = state.get("draft") or partial_from_evidence(state)
    summary = state.get("factcheck", {}).get("summary", {})
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if status == "completed":
        status_line = "Approved by a human reviewer."
    elif stop_reason == "token_budget":
        status_line = (
            f"Partial report: the run stopped because it used {state.get('tokens_used', 0):,} "
            f"tokens, over the budget of {settings.token_budget:,}. Not reviewed."
        )
    else:
        status_line = "Partial report: the run stopped before human approval."

    meta = [
        f"> **Question:** {state.get('question', '')}",
        f"> **Status:** {status_line}",
        f"> **Generated:** {date}, thread `{state.get('thread_id', '')}`, "
        f"{len(state.get('sources', []))} sources, "
        f"{state.get('revision_count', 0)} automatic revision(s).",
    ]
    if summary.get("factual_sentences"):
        coverage = summary.get("citation_coverage")
        supported = summary.get("supported_rate")
        meta.append(
            "> **Fact-check:** "
            f"{summary.get('supported', 0)} supported, {summary.get('unsupported', 0)} "
            f"unsupported, {summary.get('missing_citation', 0)} missing a citation "
            f"(coverage {coverage:.0%}"
            + (f", supported rate {supported:.0%})." if supported is not None else ").")
        )

    source_lines = []
    for source in state.get("sources", []):
        note = " (search snippet only)" if source.get("kind") == "snippet" else ""
        source_lines.append(f"- **[{source['id']}]** [{source['title']}]({source['url']}){note}")
    sources_block = "\n".join(source_lines) or "No sources were collected."

    return f"# {title}\n\n" + "\n".join(meta) + f"\n\n{draft}\n\n## Sources\n\n{sources_block}\n"


def finalize(state: dict[str, Any], settings: Settings) -> dict[str, Any]:
    status, stop_reason = outcome(state, settings)
    out: Path = run_dir(settings.runs_dir, state["thread_id"])
    out.mkdir(parents=True, exist_ok=True)

    report = render_report(state, status, stop_reason, settings)
    (out / "report.md").write_text(report, encoding="utf-8")

    cited = set(find_citations(state.get("draft", "")))
    sources = [{**s, "cited": s["id"] in cited} for s in state.get("sources", [])]
    (out / "sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")

    factcheck = {
        "final": state.get("factcheck", {}),
        "history": state.get("factcheck_history", []),
        "revision_count": state.get("revision_count", 0),
        "human_revisions": state.get("human_revisions", 0),
    }
    (out / "factcheck.json").write_text(json.dumps(factcheck, indent=2), encoding="utf-8")
    return {"status": status, "stop_reason": stop_reason, "output_dir": str(out)}
