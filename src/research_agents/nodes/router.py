"""Router: decide what happens after a fact-check.

- no issues                         -> human review
- issues, revisions left, evidence
  missing and a gap round left       -> targeted research, then rewrite
- issues, revisions left             -> rewrite with the fact-check feedback
- issues, no revisions left          -> human review (remaining issues are shown to the person)
- token budget exhausted             -> finalize with a partial report
"""

from __future__ import annotations

from typing import Any

from research_agents.config import Settings

MAX_FEEDBACK_ITEMS = 25


def budget_exceeded(state: dict[str, Any], settings: Settings) -> bool:
    return state.get("tokens_used", 0) >= settings.token_budget


def format_feedback(issues: list[dict[str, Any]]) -> str:
    lines = ["Fact-check findings:"]
    for issue in issues[:MAX_FEEDBACK_ITEMS]:
        label = issue["verdict"].replace("_", " ")
        lines.append(f'- [{label}] "{issue["sentence"]}" -> {issue["reason"]}')
    if len(issues) > MAX_FEEDBACK_ITEMS:
        lines.append(f"- ...and {len(issues) - MAX_FEEDBACK_ITEMS} more of the same kind.")
    return "\n".join(lines)


def gap_queries(issues: list[dict[str, Any]], limit: int) -> list[str]:
    queries: list[str] = []
    for issue in issues:
        query = (issue.get("missing_evidence_query") or "").strip()
        if (
            issue["verdict"] == "unsupported"
            and query
            and query.lower() not in map(str.lower, queries)
        ):
            queries.append(query)
        if len(queries) >= limit:
            break
    return queries


def route_after_factcheck(state: dict[str, Any], settings: Settings) -> dict[str, Any]:
    if budget_exceeded(state, settings):
        return {"route": "budget"}

    verdicts = state.get("factcheck", {}).get("verdicts", [])
    issues = [v for v in verdicts if v["verdict"] != "supported"]
    revisions = state.get("revision_count", 0)
    if not issues or revisions >= settings.max_revisions:
        return {"route": "review", "status": "awaiting_review"}

    feedback = format_feedback(issues)
    rounds = state.get("gap_research_rounds", 0)
    queries = gap_queries(issues, settings.max_gap_queries)
    if queries and rounds < settings.max_gap_research_rounds:
        return {
            "route": "research",
            "gap_queries": queries,
            "gap_research_rounds": rounds + 1,
            "revision_count": revisions + 1,
            "revision_feedback": feedback,
            "status": "researching",
        }
    return {
        "route": "revise",
        "revision_count": revisions + 1,
        "revision_feedback": feedback,
        "status": "revising",
    }
