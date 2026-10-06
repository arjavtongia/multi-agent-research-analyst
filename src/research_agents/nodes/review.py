"""Human review: pause the graph with `interrupt` until a person approves or asks for changes."""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from research_agents.schemas import ReviewDecision

MAX_ISSUES_SHOWN = 15


def review_payload(state: dict[str, Any]) -> dict[str, Any]:
    """Everything a reviewer needs, JSON-serialisable (it is stored in the checkpoint)."""
    factcheck = state.get("factcheck", {})
    issues = [v for v in factcheck.get("verdicts", []) if v["verdict"] != "supported"]
    return {
        "type": "review_request",
        "question": state.get("question", ""),
        "title": state.get("plan", {}).get("title", ""),
        "draft": state.get("draft", ""),
        "draft_version": state.get("draft_version", 0),
        "factcheck_summary": factcheck.get("summary", {}),
        "open_issues": [
            {k: v[k] for k in ("sentence_id", "sentence", "verdict", "reason")}
            for v in issues[:MAX_ISSUES_SHOWN]
        ],
        "sources": [
            {"id": s["id"], "title": s["title"], "url": s["url"]} for s in state.get("sources", [])
        ],
        "revision_count": state.get("revision_count", 0),
        "instructions": 'Resume with {"action": "approve"} or '
        '{"action": "revise", "feedback": "what to change"}.',
    }


def human_review(state: dict[str, Any]) -> dict[str, Any]:
    # Nothing with side effects may run before interrupt(): the node re-executes on resume.
    raw = interrupt(review_payload(state))
    decision = ReviewDecision.model_validate(raw if isinstance(raw, dict) else {"action": raw})
    if decision.action == "revise":
        if not decision.feedback:
            raise ValueError("A revise decision needs feedback text.")
        return {
            "review": decision.model_dump(),
            "revision_feedback": (
                "Human reviewer feedback (highest priority, apply it while keeping every factual "
                f"sentence cited):\n{decision.feedback}"
            ),
            "human_revisions": state.get("human_revisions", 0) + 1,
            "status": "revising",
        }
    return {"review": decision.model_dump(), "status": "approved"}
