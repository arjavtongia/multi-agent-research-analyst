from __future__ import annotations

from pathlib import Path

from fakes import ScriptedChatModel, make_deps

from research_agents.runner import start_run


def test_budget_exceeded_after_planning_stops_cleanly(tmp_path):
    llm = ScriptedChatModel()
    deps = make_deps(tmp_path, llm=llm, token_budget=50)
    result = start_run("Expensive question?", deps=deps)

    assert result.status == "partial"
    assert not result.awaiting_review  # no human review for a stopped run
    assert result.values["stop_reason"] == "token_budget"
    assert llm.calls == ["planner"]  # nothing ran after the planner
    report = (Path(result.output_dir) / "report.md").read_text(encoding="utf-8")
    assert "Partial report" in report
    assert "over the budget of 50" in report


def test_budget_exceeded_after_research_keeps_collected_evidence(tmp_path):
    llm = ScriptedChatModel()
    deps = make_deps(tmp_path, llm=llm, token_budget=1000)
    result = start_run("Expensive question?", deps=deps)

    assert result.status == "partial"
    assert "writer" not in llm.calls
    assert result.values["tokens_used"] >= 1000
    report = (Path(result.output_dir) / "report.md").read_text(encoding="utf-8")
    assert "Collected evidence" in report
    assert "[S1]" in report
