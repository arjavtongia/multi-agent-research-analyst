from __future__ import annotations

import json
import threading
from pathlib import Path

from fakes import FakeSearch, ScriptedChatModel, make_deps
from langgraph.types import Send

from research_agents.graph import gap_research_tasks, research_tasks
from research_agents.runner import start_run


def _plan(n: int) -> dict:
    return {
        "title": "T",
        "outline": ["A"],
        "sub_questions": [
            {"id": f"q{i}", "question": f"Q{i}?", "search_query": f"q{i}", "kind": "plan"}
            for i in range(1, n + 1)
        ],
    }


def test_fan_out_sends_one_research_task_per_sub_question():
    sends = research_tasks({"question": "Main?", "plan": _plan(4)})
    assert len(sends) == 4
    assert all(isinstance(s, Send) and s.node == "researcher" for s in sends)
    assert [s.arg["sub_question"]["id"] for s in sends] == ["q1", "q2", "q3", "q4"]
    assert all(s.arg["question"] == "Main?" for s in sends)


def test_gap_fan_out_builds_targeted_sub_questions():
    sends = gap_research_tasks(
        {"question": "Main?", "gap_queries": ["solar 2024 capacity"], "gap_research_rounds": 1}
    )
    assert len(sends) == 1
    sub_question = sends[0].arg["sub_question"]
    assert sub_question == {
        "id": "gap1-1",
        "question": "solar 2024 capacity",
        "search_query": "solar 2024 capacity",
        "kind": "gap",
    }


class BarrierSearch(FakeSearch):
    """Every researcher must be inside search() at the same time, or the barrier times out."""

    def __init__(self, parties: int) -> None:
        super().__init__()
        self.barrier = threading.Barrier(parties, timeout=10)

    def search(self, query, max_results):
        self.barrier.wait()
        return super().search(query, max_results)


def test_researchers_run_in_parallel(tmp_path):
    deps = make_deps(tmp_path, search=BarrierSearch(parties=3))
    result = start_run("Parallel?", deps=deps, auto_approve=True)
    log = result.values["research_log"]
    assert sorted(entry["sub_question_id"] for entry in log) == ["q1", "q2", "q3"]
    assert result.status == "completed"


def test_full_run_writes_report_sources_factcheck_and_trace(tmp_path):
    llm = ScriptedChatModel()
    deps = make_deps(tmp_path, llm=llm)
    result = start_run("What about alpha?", deps=deps, auto_approve=True, thread_id="t-full")

    assert result.status == "completed"
    out = Path(result.output_dir)
    report = (out / "report.md").read_text(encoding="utf-8")
    assert report.startswith("# Test report")
    assert "## Sources" in report and "[S1]" in report
    assert "Approved by a human reviewer" in report

    sources = json.loads((out / "sources.json").read_text(encoding="utf-8"))
    assert [s["id"] for s in sources] == [f"S{i}" for i in range(1, len(sources) + 1)]
    assert any(s["cited"] for s in sources)

    factcheck = json.loads((out / "factcheck.json").read_text(encoding="utf-8"))
    assert factcheck["final"]["summary"]["unsupported"] == 0
    assert factcheck["revision_count"] == 0

    trace = [json.loads(line) for line in (out / "trace.jsonl").read_text().splitlines()]
    nodes = [t["node"] for t in trace]
    for node in ["planner", "researcher", "aggregate", "writer", "fact_checker", "router"]:
        assert node in nodes
    assert nodes.count("researcher") == 3
    assert nodes[-1] == "finalize"
    assert any(t["status"] == "interrupted" for t in trace if t["node"] == "human_review")
    assert (
        sum(t["input_tokens"] + t["output_tokens"] for t in trace) == result.values["tokens_used"]
    )
    assert llm.calls.count("planner") == 1
