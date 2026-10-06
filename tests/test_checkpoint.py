from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from fakes import ScriptedChatModel, default_writer, make_deps

from research_agents.runner import (
    NotAwaitingReviewError,
    RunNotFoundError,
    get_run,
    resume_run,
    start_run,
)
from research_agents.schemas import ReviewDecision

TESTS_DIR = Path(__file__).parent


def test_run_pauses_for_human_review(tmp_path):
    deps = make_deps(tmp_path)
    result = start_run("Pause?", deps=deps, thread_id="pause-1")

    assert result.status == "awaiting_review"
    request = result.review_request
    assert request["type"] == "review_request"
    assert request["draft"]
    assert request["sources"]
    assert (tmp_path / "runs" / "checkpoints.sqlite").exists()
    assert (tmp_path / "runs" / "pause-1" / "review_draft.md").exists()
    assert not (tmp_path / "runs" / "pause-1" / "report.md").exists()


def test_resume_in_a_separate_process(tmp_path):
    deps = make_deps(tmp_path)
    paused = start_run("Cross-process?", deps=deps, thread_id="xproc-1")
    assert paused.awaiting_review

    script = textwrap.dedent(
        f"""
        from pathlib import Path
        from fakes import make_deps
        from research_agents.runner import resume_run
        from research_agents.schemas import ReviewDecision

        deps = make_deps(Path(r"{tmp_path}"))
        result = resume_run("xproc-1", ReviewDecision(action="approve"), deps=deps)
        print("STATUS=" + result.status)
        """
    )
    env = {**os.environ, "PYTHONPATH": str(TESTS_DIR)}
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=180
    )
    assert proc.returncode == 0, proc.stderr
    assert "STATUS=completed" in proc.stdout

    final = get_run("xproc-1", deps=make_deps(tmp_path))
    assert final.status == "completed"
    assert (tmp_path / "runs" / "xproc-1" / "report.md").exists()


def test_revise_feedback_reaches_the_writer_and_pauses_again(tmp_path):
    prompts: list[str] = []

    def writer(messages):
        prompts.append(str(messages[1].content))
        return default_writer(messages)

    llm = ScriptedChatModel(handlers={"writer": writer})
    deps = make_deps(tmp_path, llm=llm)
    start_run("Revise?", deps=deps, thread_id="rev-1")

    revised = resume_run(
        "rev-1", ReviewDecision(action="revise", feedback="Add a sentence on costs."), deps=deps
    )
    assert revised.awaiting_review
    assert revised.review_request["draft_version"] == 2
    assert "Add a sentence on costs." in prompts[-1]
    assert "Previous draft" in prompts[-1]

    done = resume_run("rev-1", ReviewDecision(action="approve"), deps=deps)
    assert done.status == "completed"
    assert done.values["human_revisions"] == 1


def test_resume_errors(tmp_path):
    deps = make_deps(tmp_path)
    with pytest.raises(RunNotFoundError):
        resume_run("missing", ReviewDecision(action="approve"), deps=deps)

    start_run("Done?", deps=deps, thread_id="done-1", auto_approve=True)
    with pytest.raises(NotAwaitingReviewError):
        resume_run("done-1", ReviewDecision(action="approve"), deps=deps)

    with pytest.raises(ValueError, match="feedback"):
        resume_run("done-1", ReviewDecision(action="revise", feedback="  "), deps=deps)

    with pytest.raises(ValueError, match="invalid thread id"):
        resume_run("../escape", ReviewDecision(action="approve"), deps=deps)


def test_crashed_run_is_reported_as_failed(tmp_path):
    class BrokenSearch:
        name = "broken"

        def search(self, query, max_results):
            raise RuntimeError("network down")

    deps = make_deps(tmp_path, search=BrokenSearch())
    with pytest.raises(RuntimeError, match="network down"):
        start_run("Crash?", deps=deps, thread_id="crash-1")
    assert get_run("crash-1", deps=deps).status == "failed"
    trace = (tmp_path / "runs" / "crash-1" / "trace.jsonl").read_text(encoding="utf-8")
    assert '"status": "error"' in trace and "network down" in trace
