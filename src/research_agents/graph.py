"""The LangGraph StateGraph that wires the research team together.

planner -> researcher x N (parallel, Send) -> aggregate -> writer -> fact_checker -> router
router -> writer | researcher (targeted gap pass) | human_review | finalize
human_review (interrupt) -> writer (revise) | finalize (approve)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer, Send

from research_agents.deps import Deps
from research_agents.nodes.aggregate import aggregate_notes
from research_agents.nodes.fact_checker import fact_check
from research_agents.nodes.finalize import finalize
from research_agents.nodes.planner import plan_question
from research_agents.nodes.researcher import research_sub_question
from research_agents.nodes.review import human_review
from research_agents.nodes.router import budget_exceeded, route_after_factcheck
from research_agents.nodes.writer import write_report
from research_agents.state import ResearchState, ResearchTask, SubQuestion
from research_agents.tracing import traced

RECURSION_LIMIT = 100


def research_tasks(state: dict[str, Any]) -> list[Send]:
    """One `Send` per planned sub-question; LangGraph runs them in the same step, in parallel."""
    return [
        Send("researcher", ResearchTask(question=state["question"], sub_question=sq))
        for sq in state["plan"]["sub_questions"]
    ]


def gap_research_tasks(state: dict[str, Any]) -> list[Send]:
    """Targeted researcher pass for evidence the fact-checker found missing."""
    round_no = state.get("gap_research_rounds", 1)
    return [
        Send(
            "researcher",
            ResearchTask(
                question=state["question"],
                sub_question=SubQuestion(
                    id=f"gap{round_no}-{i}", question=query, search_query=query, kind="gap"
                ),
            ),
        )
        for i, query in enumerate(state.get("gap_queries", []), start=1)
    ]


def build_graph(deps: Deps, checkpointer: Checkpointer = None) -> CompiledStateGraph:
    settings = deps.settings

    def node(name: str, fn: Callable[[dict[str, Any]], dict[str, Any]]) -> Any:
        return traced(name, fn, deps.tracer)

    def after_planner(state: dict[str, Any]) -> str | list[Send]:
        if budget_exceeded(state, settings):
            return "finalize"
        return research_tasks(state)

    def gate(next_node: str) -> Callable[[dict[str, Any]], str]:
        def check(state: dict[str, Any]) -> str:
            return "finalize" if budget_exceeded(state, settings) else next_node

        return check

    def after_router(state: dict[str, Any]) -> str | list[Send]:
        route = state.get("route", "review")
        if route == "research":
            return gap_research_tasks(state) or "writer"
        return {"revise": "writer", "review": "human_review", "budget": "finalize"}[route]

    def after_review(state: dict[str, Any]) -> str:
        if state.get("review", {}).get("action") != "revise":
            return "finalize"
        return "finalize" if budget_exceeded(state, settings) else "writer"

    builder = StateGraph(ResearchState)
    builder.add_node("planner", node("planner", lambda s: plan_question(s, deps)))
    builder.add_node("researcher", node("researcher", lambda s: research_sub_question(s, deps)))
    builder.add_node("aggregate", node("aggregate", aggregate_notes))
    builder.add_node("writer", node("writer", lambda s: write_report(s, deps)))
    builder.add_node("fact_checker", node("fact_checker", lambda s: fact_check(s, deps)))
    builder.add_node("router", node("router", lambda s: route_after_factcheck(s, settings)))
    builder.add_node("human_review", node("human_review", human_review))
    builder.add_node("finalize", node("finalize", lambda s: finalize(s, settings)))

    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", after_planner, ["researcher", "finalize"])
    builder.add_edge("researcher", "aggregate")
    builder.add_conditional_edges("aggregate", gate("writer"), ["writer", "finalize"])
    builder.add_conditional_edges("writer", gate("fact_checker"), ["fact_checker", "finalize"])
    builder.add_edge("fact_checker", "router")
    builder.add_conditional_edges(
        "router", after_router, ["researcher", "writer", "human_review", "finalize"]
    )
    builder.add_conditional_edges("human_review", after_review, ["writer", "finalize"])
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer)
