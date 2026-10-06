from __future__ import annotations

import json
from typing import Any

import pytest
from fakes import ScriptedChatModel, make_settings
from langchain_core.runnables import RunnableLambda

from research_agents.llm import (
    MissingAPIKeyError,
    StructuredOutputError,
    chat_call,
    extract_json,
    get_chat_model,
    resolve_model_name,
    structured_call,
    system_and_user,
    track_usage,
)
from research_agents.prompts import PLANNER_SYSTEM
from research_agents.schemas import Plan

PLAN_JSON = json.dumps(
    {
        "title": "T",
        "sub_questions": [{"question": "A?", "search_query": "a"}],
        "outline": ["Intro"],
    }
)


def planner_messages():
    return system_and_user(PLANNER_SYSTEM.format(min_q=3, max_q=6), "Research question: x")


class NativeStructuredModel(ScriptedChatModel):
    """A model whose with_structured_output works, like a provider with tool calling."""

    def with_structured_output(self, schema: Any, **kwargs: Any):  # type: ignore[override]
        return RunnableLambda(lambda _messages: schema.model_validate_json(PLAN_JSON))


def test_structured_call_prefers_native_structured_output():
    llm = NativeStructuredModel()
    plan = structured_call(llm, Plan, planner_messages())
    assert plan.title == "T"
    assert llm.calls == []  # the plain-text JSON path was never needed


def test_structured_call_falls_back_to_json_with_fences_and_think_tags():
    reply = f"<think>let me plan</think>Sure, here it is:\n```json\n{PLAN_JSON}\n```"
    llm = ScriptedChatModel(handlers={"planner": reply})
    plan = structured_call(llm, Plan, planner_messages())
    assert plan.sub_questions[0].question == "A?"
    assert llm.calls == ["planner"]


def test_structured_call_repairs_once_after_invalid_json():
    llm = ScriptedChatModel(handlers={"planner": ["this is not json", PLAN_JSON]})
    plan = structured_call(llm, Plan, planner_messages())
    assert plan.outline == ["Intro"]
    assert llm.calls == ["planner", "planner"]


def test_structured_call_raises_after_failed_repair():
    llm = ScriptedChatModel(handlers={"planner": ['{"title": "no sub questions"}', "nope"]})
    with pytest.raises(StructuredOutputError):
        structured_call(llm, Plan, planner_messages())


def test_plan_accepts_plain_string_sub_questions():
    plan = Plan.model_validate({"title": "T", "sub_questions": ["One?", "Two?"], "outline": ["A"]})
    assert [s.question for s in plan.sub_questions] == ["One?", "Two?"]


def test_extract_json_finds_object_inside_prose():
    assert extract_json('Answer: {"a": 1} hope that helps') == {"a": 1}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_track_usage_counts_tokens_and_calls():
    llm = ScriptedChatModel(handlers={"planner": "x" * 400})
    with track_usage() as usage:
        chat_call(llm, planner_messages())
        chat_call(llm, planner_messages())
    assert usage.calls == 2
    assert usage.output_tokens == 200
    assert usage.input_tokens > 0
    # Outside the block nothing is counted any more.
    chat_call(llm, planner_messages())
    assert usage.calls == 2


def test_missing_nvidia_key_has_actionable_message(tmp_path):
    settings = make_settings(tmp_path, nvidia_api_key=None)
    with pytest.raises(MissingAPIKeyError, match="build.nvidia.com"):
        get_chat_model(settings=settings)


def test_missing_openai_key(tmp_path):
    settings = make_settings(tmp_path, llm_provider="openai", openai_api_key=None)
    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY"):
        get_chat_model(settings=settings)


def test_default_model_names(tmp_path):
    settings = make_settings(tmp_path)
    assert resolve_model_name(settings) == "nvidia/nemotron-3-super-120b-a12b"
    assert resolve_model_name(settings, fast=True) == "openai/gpt-oss-20b"
    override = make_settings(tmp_path, llm_model="my/model")
    assert resolve_model_name(override) == "my/model"


def test_nvidia_model_unknown_to_chatnvidia_uses_openai_compatible_client(tmp_path):
    from langchain_openai import ChatOpenAI

    settings = make_settings(tmp_path, nvidia_api_key="test-key", llm_fast_model="vendor/new-model")
    model = get_chat_model(fast=True, settings=settings)
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "vendor/new-model"
    assert "integrate.api.nvidia.com" in str(model.openai_api_base)


attempts: list[int] = []


class FailingNativeModel(ScriptedChatModel):
    """Native structured output always fails (like an endpoint without json_schema support)."""

    def with_structured_output(self, schema: Any, **kwargs: Any):  # type: ignore[override]
        attempts.append(1)
        raise NotImplementedError


def test_native_structured_output_is_skipped_after_repeated_failures(monkeypatch):
    from research_agents import llm as llm_module

    monkeypatch.setattr(llm_module, "_native_failures", {})
    attempts.clear()
    model = FailingNativeModel(handlers={"planner": PLAN_JSON})
    for _ in range(4):
        assert structured_call(model, Plan, planner_messages()).title == "T"
    assert len(attempts) == llm_module.NATIVE_FAILURE_LIMIT
    assert model.calls == ["planner"] * 4  # the JSON path answered every time
