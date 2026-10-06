"""Streamlit UI: a thin layer over `research_agents.runner`.

Run with:  uv run --extra ui streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from research_agents.deps import Deps, build_default_deps
from research_agents.llm import MissingAPIKeyError
from research_agents.runner import (
    RunEvent,
    RunResult,
    describe_event,
    get_run,
    new_thread_id,
    resume_run,
    start_run,
)
from research_agents.schemas import ReviewDecision
from research_agents.tools.search import SearchError

load_dotenv(override=False)
st.set_page_config(page_title="Research agents", layout="wide")


@st.cache_resource
def load_deps() -> Deps:
    return build_default_deps()


def run_with_progress(label: str, action: Callable[..., RunResult]) -> RunResult:
    """Run a graph action, writing one line per finished node into a status box."""
    with st.status(label, expanded=True) as status:

        def on_event(event: RunEvent) -> None:
            status.write(describe_event(event))

        result = action(on_event)
        status.update(label=f"{label}: {result.status}", state="complete", expanded=False)
    return result


def show_plan(values: dict) -> None:
    plan = values.get("plan") or {}
    if not plan:
        return
    st.subheader("Plan")
    st.markdown(f"**{plan.get('title', '')}**")
    for sq in plan.get("sub_questions", []):
        st.markdown(f"- `{sq['id']}` {sq['question']}  \n  search: *{sq['search_query']}*")
    st.caption("Outline: " + " / ".join(plan.get("outline", [])))


def show_sources(values: dict) -> None:
    sources = values.get("sources") or []
    if not sources:
        return
    st.subheader(f"Sources ({len(sources)})")
    frame = pd.DataFrame(
        [
            {
                "id": s["id"],
                "title": s["title"],
                "url": s["url"],
                "kind": s.get("kind", "page"),
                "notes": s.get("note_count", 0),
            }
            for s in sources
        ]
    )
    st.dataframe(frame, hide_index=True, column_config={"url": st.column_config.LinkColumn("url")})


def show_factcheck(values: dict) -> None:
    factcheck = values.get("factcheck") or {}
    verdicts = factcheck.get("verdicts") or []
    if not verdicts:
        return
    summary = factcheck.get("summary", {})
    st.subheader(f"Fact-check (draft v{factcheck.get('draft_version', '?')})")
    cols = st.columns(3)
    cols[0].metric("Supported", summary.get("supported", 0))
    cols[1].metric("Unsupported", summary.get("unsupported", 0))
    cols[2].metric("Missing citation", summary.get("missing_citation", 0))
    frame = pd.DataFrame(
        [
            {
                "id": v["sentence_id"],
                "verdict": v["verdict"],
                "sentence": v["sentence"],
                "reason": v["reason"],
            }
            for v in verdicts
        ]
    )
    st.dataframe(frame, hide_index=True)


def show_timings(values: dict) -> None:
    timings = values.get("timings") or []
    if not timings:
        return
    with st.expander("Node timings and tokens"):
        frame = pd.DataFrame(
            [
                {
                    "node": t["node"],
                    "sub-question": t.get("sub_question_id", ""),
                    "seconds": round(t["duration_ms"] / 1000, 2),
                    "tokens": t["tokens"],
                    "llm calls": t["llm_calls"],
                }
                for t in timings
            ]
        )
        st.dataframe(frame, hide_index=True)


def main() -> None:
    st.title("Multi-agent research analyst")
    st.caption(
        "Planner, parallel researchers, writer and fact-checker agents built with LangGraph. "
        "Every factual sentence must cite a source; you approve the report before it is saved."
    )
    try:
        deps = load_deps()
    except (MissingAPIKeyError, SearchError) as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.header("Resume a run")
        existing = st.text_input("Thread id", value=st.session_state.get("thread_id", ""))
        if st.button("Load thread") and existing.strip():
            st.session_state.thread_id = existing.strip()

    question = st.text_area("Research question", placeholder="What is known about ...?")
    auto_approve = st.checkbox("Auto-approve (skip the review step)")
    if st.button("Run research", type="primary", disabled=not question.strip()):
        thread_id = new_thread_id()
        st.session_state.thread_id = thread_id
        run_with_progress(
            "Researching",
            lambda on_event: start_run(
                question,
                deps=deps,
                thread_id=thread_id,
                auto_approve=auto_approve,
                on_event=on_event,
            ),
        )

    thread_id = st.session_state.get("thread_id")
    if not thread_id:
        return
    try:
        result = get_run(thread_id, deps=deps)
    except ValueError as exc:
        st.error(str(exc))
        return
    if result.status == "unknown":
        st.warning(f"No run found for thread {thread_id}.")
        return

    st.divider()
    st.markdown(
        f"Thread `{thread_id}`, status **{result.status}**, "
        f"{result.values.get('tokens_used', 0):,} tokens used."
    )
    show_plan(result.values)
    show_sources(result.values)
    show_factcheck(result.values)
    show_timings(result.values)

    st.subheader("Report draft")
    st.markdown(result.values.get("draft", "_No draft yet._"))

    if result.awaiting_review:
        st.subheader("Your review")
        feedback = st.text_area("Feedback for a revision (leave empty to approve)")
        approve_col, revise_col = st.columns(2)
        if approve_col.button("Approve and finalise", type="primary"):
            run_with_progress(
                "Finalising",
                lambda on_event: resume_run(
                    thread_id, ReviewDecision(action="approve"), deps=deps, on_event=on_event
                ),
            )
            st.rerun()
        if revise_col.button("Request revision", disabled=not feedback.strip()):
            run_with_progress(
                "Revising",
                lambda on_event: resume_run(
                    thread_id,
                    ReviewDecision(action="revise", feedback=feedback),
                    deps=deps,
                    on_event=on_event,
                ),
            )
            st.rerun()
    elif result.output_dir:
        st.success(f"Saved to {result.output_dir}")


main()
