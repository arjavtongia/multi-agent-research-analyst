"""Run, stream, pause and resume research threads. CLI, API, UI and eval all go through here.

Every run uses the SQLite checkpointer under `runs/`, so a thread paused at human review can be
resumed later from a different process with only its thread id.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from research_agents.deps import Deps, build_default_deps
from research_agents.graph import RECURSION_LIMIT, build_graph
from research_agents.schemas import ReviewDecision
from research_agents.tracing import run_dir, validate_thread_id

APPROVE = ReviewDecision(action="approve")


@dataclass
class RunEvent:
    """One streamed graph update: which node finished and what it wrote."""

    node: str
    update: dict[str, Any]


@dataclass
class RunResult:
    thread_id: str
    status: str  # awaiting_review | completed | partial | running | failed | unknown
    values: dict[str, Any] = field(default_factory=dict)
    review_request: dict[str, Any] | None = None

    @property
    def awaiting_review(self) -> bool:
        return self.review_request is not None

    @property
    def output_dir(self) -> str | None:
        return self.values.get("output_dir")


class RunNotFoundError(LookupError):
    pass


class NotAwaitingReviewError(RuntimeError):
    pass


def new_thread_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def run_config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}


@contextmanager
def open_graph(deps: Deps) -> Iterator[CompiledStateGraph]:
    """Compile the graph with a SQLite checkpointer for the duration of the block."""
    path = deps.settings.checkpoint_path
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False is safe: SqliteSaver serialises access with its own lock.
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
    try:
        # WAL lets the API read run status while a background run is writing, and
        # synchronous=NORMAL avoids an fsync per checkpoint (still safe against app crashes).
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        yield build_graph(deps, checkpointer=SqliteSaver(conn))
    finally:
        conn.close()


def snapshot_result(graph: CompiledStateGraph, thread_id: str) -> RunResult:
    snapshot = graph.get_state(run_config(thread_id))
    values = dict(snapshot.values or {})
    if not values:
        return RunResult(thread_id=thread_id, status="unknown")
    if snapshot.interrupts:
        return RunResult(
            thread_id=thread_id,
            status="awaiting_review",
            values=values,
            review_request=snapshot.interrupts[0].value,
        )
    if any(task.error for task in snapshot.tasks):
        return RunResult(thread_id=thread_id, status="failed", values=values)
    if snapshot.next:
        return RunResult(thread_id=thread_id, status="running", values=values)
    return RunResult(thread_id=thread_id, status=values.get("status", "completed"), values=values)


def iter_events(graph: CompiledStateGraph, graph_input: Any, thread_id: str) -> Iterator[RunEvent]:
    """Stream node-by-node updates until the graph finishes or pauses."""
    for chunk in graph.stream(graph_input, run_config(thread_id), stream_mode="updates"):
        for node, update in chunk.items():
            if node == "__interrupt__":
                continue
            yield RunEvent(node=node, update=update if isinstance(update, dict) else {})


def drive(
    graph: CompiledStateGraph,
    graph_input: Any,
    thread_id: str,
    *,
    auto_approve: bool = False,
    on_event: Callable[[RunEvent], None] | None = None,
) -> RunResult:
    """Run until completion or a review pause. With auto_approve, pauses are approved."""
    while True:
        for event in iter_events(graph, graph_input, thread_id):
            if on_event:
                on_event(event)
        result = snapshot_result(graph, thread_id)
        if result.awaiting_review and auto_approve:
            graph_input = Command(resume=APPROVE.model_dump())
            continue
        return result


def save_review_draft(deps: Deps, result: RunResult) -> None:
    """Write the paused draft to runs/<thread_id>/review_draft.md so a person can read it."""
    if not result.review_request:
        return
    out = run_dir(deps.settings.runs_dir, result.thread_id)
    out.mkdir(parents=True, exist_ok=True)
    request = result.review_request
    text = f"# {request.get('title', '')}\n\n{request.get('draft', '')}\n"
    (out / "review_draft.md").write_text(text, encoding="utf-8")


def describe_event(event: RunEvent) -> str:
    """One human-readable progress line per node update (used by the CLI and the UI)."""
    update = event.update
    timing = (update.get("timings") or [{}])[0]
    cost = f"{timing.get('duration_ms', 0) / 1000:.1f}s, {timing.get('tokens', 0):,} tokens"
    node = event.node
    if node == "planner":
        plan = update.get("plan", {})
        detail = f"{len(plan.get('sub_questions', []))} sub-questions ({plan.get('origin', '')})"
    elif node == "researcher":
        log = (update.get("research_log") or [{}])[0]
        detail = (
            f"[{log.get('sub_question_id', '?')}] {log.get('results', 0)} results, "
            f"{len(log.get('sources', []))} sources, {log.get('notes', 0)} notes"
        )
        if log.get("dropped_quotes"):
            detail += f", {log['dropped_quotes']} ungrounded quotes dropped"
        if log.get("suspicious"):
            detail += f", {len(log['suspicious'])} page(s) with injection-like text"
        if log.get("error"):
            detail += f", error: {log['error']}"
    elif node == "aggregate":
        sources, notes = len(update.get("sources", [])), len(update.get("evidence", []))
        detail = f"{sources} sources, {notes} notes"
    elif node == "writer":
        words = len(update.get("draft", "").split())
        detail = f"draft v{update.get('draft_version', '?')}, {words} words"
    elif node == "fact_checker":
        summary = update.get("factcheck", {}).get("summary", {})
        detail = (
            f"{summary.get('supported', 0)} supported, "
            f"{summary.get('unsupported', 0)} unsupported, "
            f"{summary.get('missing_citation', 0)} missing citation"
        )
    elif node == "router":
        detail = f"route: {update.get('route', '?')}"
    elif node == "human_review":
        detail = f"decision: {update.get('review', {}).get('action', '?')}"
    elif node == "finalize":
        detail = f"status: {update.get('status', '?')}"
    else:
        detail = ""
    return f"{node:<13} {detail} ({cost})"


def start_run(
    question: str,
    *,
    deps: Deps | None = None,
    thread_id: str | None = None,
    auto_approve: bool = False,
    on_event: Callable[[RunEvent], None] | None = None,
) -> RunResult:
    question = question.strip()
    if not question:
        raise ValueError("The research question is empty.")
    deps = deps or build_default_deps()
    thread_id = validate_thread_id(thread_id or new_thread_id())
    with open_graph(deps) as graph:
        if graph.get_state(run_config(thread_id)).values:
            raise ValueError(f"Thread {thread_id} already exists; use resume instead.")
        initial = {"thread_id": thread_id, "question": question, "status": "planning"}
        result = drive(graph, initial, thread_id, auto_approve=auto_approve, on_event=on_event)
    save_review_draft(deps, result)
    return result


def resume_run(
    thread_id: str,
    decision: ReviewDecision,
    *,
    deps: Deps | None = None,
    auto_approve: bool = False,
    on_event: Callable[[RunEvent], None] | None = None,
) -> RunResult:
    """Answer the pending review interrupt of a thread (possibly from another process)."""
    validate_thread_id(thread_id)
    if decision.action == "revise" and not decision.feedback:
        raise ValueError("--revise needs feedback text describing what to change.")
    deps = deps or build_default_deps()
    with open_graph(deps) as graph:
        current = snapshot_result(graph, thread_id)
        if current.status == "unknown":
            raise RunNotFoundError(f"No run found with thread id {thread_id}.")
        if not current.awaiting_review:
            raise NotAwaitingReviewError(
                f"Thread {thread_id} is not waiting for review (status: {current.status})."
            )
        result = drive(
            graph,
            Command(resume=decision.model_dump()),
            thread_id,
            auto_approve=auto_approve,
            on_event=on_event,
        )
    save_review_draft(deps, result)
    return result


def get_run(thread_id: str, *, deps: Deps) -> RunResult:
    validate_thread_id(thread_id)
    with open_graph(deps) as graph:
        return snapshot_result(graph, thread_id)
