from __future__ import annotations

import json

from fakes import ScriptedChatModel, make_deps

from research_agents.nodes.planner import plan_question


def test_planner_assigns_ids_dedupes_and_caps(tmp_path):
    sub_questions = [{"question": f"Question {i}?", "search_query": f"q {i}"} for i in range(8)]
    sub_questions.insert(1, {"question": "question 0?"})  # duplicate, different case
    reply = json.dumps(
        {
            "title": "Report",
            "sub_questions": sub_questions,
            "outline": ["Intro", "## Findings", "Sources", "Intro"],
        }
    )
    deps = make_deps(tmp_path, llm=ScriptedChatModel(handlers={"planner": reply}))
    plan = plan_question({"question": "What happened?"}, deps)["plan"]

    ids = [sq["id"] for sq in plan["sub_questions"]]
    assert ids == ["q1", "q2", "q3", "q4", "q5", "q6"]  # capped at max_sub_questions
    assert plan["sub_questions"][1]["question"] == "Question 1?"  # duplicate removed
    assert plan["outline"] == ["Intro", "Findings"]  # reserved and repeated headings dropped
    assert plan["origin"] == "model"
    assert all(sq["kind"] == "plan" for sq in plan["sub_questions"])


def test_planner_fills_missing_search_query(tmp_path):
    reply = json.dumps(
        {"title": "R", "sub_questions": ["How big is the market?"], "outline": ["Market"]}
    )
    deps = make_deps(tmp_path, llm=ScriptedChatModel(handlers={"planner": reply}))
    plan = plan_question({"question": "Q"}, deps)["plan"]
    assert plan["sub_questions"][0]["search_query"] == "How big is the market"


def test_planner_falls_back_when_output_is_unusable(tmp_path):
    llm = ScriptedChatModel(handlers={"planner": ["I cannot do JSON", "still not JSON"]})
    deps = make_deps(tmp_path, llm=llm)
    plan = plan_question({"question": "Is coffee healthy?"}, deps)["plan"]
    assert plan["origin"] == "fallback"
    assert [sq["question"] for sq in plan["sub_questions"]] == ["Is coffee healthy?"]
    assert plan["outline"]
