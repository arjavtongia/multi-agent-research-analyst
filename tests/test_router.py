from __future__ import annotations

import json

from fakes import (
    ScriptedChatModel,
    make_deps,
    make_settings,
    supported_checker,
    unsupported_checker,
)

from research_agents.nodes.router import route_after_factcheck
from research_agents.runner import start_run


def _state(verdicts: list[str], revisions: int = 0, query: str | None = None, **extra) -> dict:
    return {
        "factcheck": {
            "verdicts": [
                {
                    "sentence_id": i,
                    "sentence": f"Sentence {i}.",
                    "verdict": v,
                    "reason": "r",
                    "missing_evidence_query": query,
                }
                for i, v in enumerate(verdicts, start=1)
            ]
        },
        "revision_count": revisions,
        **extra,
    }


def test_router_sends_unsupported_claims_back_to_writer(tmp_path):
    settings = make_settings(tmp_path)
    update = route_after_factcheck(_state(["supported", "unsupported"]), settings)
    assert update["route"] == "revise"
    assert update["revision_count"] == 1
    assert "Sentence 2." in update["revision_feedback"]
    assert "Sentence 1." not in update["revision_feedback"]


def test_router_stops_at_max_revisions(tmp_path):
    settings = make_settings(tmp_path, max_revisions=2)
    update = route_after_factcheck(_state(["unsupported"], revisions=2), settings)
    assert update["route"] == "review"
    assert "revision_count" not in update


def test_router_goes_to_review_when_everything_is_supported(tmp_path):
    settings = make_settings(tmp_path)
    assert route_after_factcheck(_state(["supported"]), settings)["route"] == "review"


def test_router_requests_targeted_research_for_missing_evidence(tmp_path):
    settings = make_settings(tmp_path)
    update = route_after_factcheck(_state(["unsupported"], query="wind capacity 2024"), settings)
    assert update["route"] == "research"
    assert update["gap_queries"] == ["wind capacity 2024"]
    assert update["gap_research_rounds"] == 1
    # Gap rounds are capped too: the next time it is a plain rewrite.
    again = _state(["unsupported"], revisions=1, query="wind capacity 2024", gap_research_rounds=1)
    assert route_after_factcheck(again, settings)["route"] == "revise"


def test_router_missing_citation_triggers_revision(tmp_path):
    settings = make_settings(tmp_path)
    update = route_after_factcheck(_state(["missing_citation"]), settings)
    assert update["route"] == "revise"
    assert "[missing citation]" in update["revision_feedback"]


def test_router_stops_on_token_budget(tmp_path):
    settings = make_settings(tmp_path, token_budget=100)
    update = route_after_factcheck(_state(["unsupported"], tokens_used=500), settings)
    assert update == {"route": "budget"}


def test_graph_revision_loop_is_bounded(tmp_path):
    llm = ScriptedChatModel(handlers={"fact_checker": unsupported_checker})
    deps = make_deps(tmp_path, llm=llm, max_revisions=2)
    result = start_run("Loop?", deps=deps, auto_approve=True)
    values = result.values
    assert values["revision_count"] == 2
    assert llm.calls.count("writer") == 3  # first draft + 2 revisions, then it stops
    assert len(values["factcheck_history"]) == 3
    assert values["factcheck"]["summary"]["unsupported"] > 0
    assert result.status == "completed"  # a person (here: auto-approve) makes the final call


def test_graph_targeted_research_pass_when_evidence_is_missing(tmp_path):
    def checker(messages):
        # First check: one sentence lacks evidence. Afterwards everything is supported.
        data = json.loads(supported_checker(messages))
        if llm.calls.count("fact_checker") == 1:
            data["verdicts"][0].update(
                verdict="unsupported",
                reason="no note mentions this",
                missing_evidence_query="delta statistics 2024",
            )
        return json.dumps(data)

    llm = ScriptedChatModel(handlers={"fact_checker": checker})
    deps = make_deps(tmp_path, llm=llm)
    result = start_run("Gap?", deps=deps, auto_approve=True)
    values = result.values
    kinds = [entry["kind"] for entry in values["research_log"]]
    assert kinds.count("gap") == 1
    assert "delta statistics 2024" in deps.search.queries
    assert any(n["sub_question_id"] == "gap1-1" for n in values["notes"])
    assert values["revision_count"] == 1
    assert values["gap_research_rounds"] == 1
    assert llm.calls.count("writer") == 2
