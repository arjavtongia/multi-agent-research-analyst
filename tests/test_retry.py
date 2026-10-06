"""Transient provider failures are retried; real errors are not."""

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

import research_agents.llm as llm_module
from research_agents.llm import chat_call, is_transient


class ReadTimeout(Exception):
    """Stands in for requests/httpx ReadTimeout (matched by class name)."""


class FlakyModel(GenericFakeChatModel):
    failures: int = 0

    def _generate(self, *args, **kwargs):
        if self.failures > 0:
            self.failures -= 1
            raise ReadTimeout("read timed out")
        return super()._generate(*args, **kwargs)


def test_is_transient_classifies_errors():
    assert is_transient(ReadTimeout("x"))
    status_error = RuntimeError("busy")
    status_error.status_code = 503
    assert is_transient(status_error)
    assert not is_transient(ValueError("bad request"))


def test_chat_call_retries_transient_failures(monkeypatch):
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)
    model = FlakyModel(messages=iter([AIMessage(content="done")]), failures=2)
    assert chat_call(model, [HumanMessage(content="hi")]) == "done"


def test_chat_call_gives_up_after_the_last_retry(monkeypatch):
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)
    model = FlakyModel(messages=iter([AIMessage(content="never")]), failures=5)
    with pytest.raises(ReadTimeout):
        chat_call(model, [HumanMessage(content="hi")])
