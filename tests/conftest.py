from __future__ import annotations

import pytest
from fakes import ScriptedChatModel, make_deps

from research_agents.config import Settings
from research_agents.deps import Deps


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    """Keep real keys or provider settings from the shell out of the tests."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def llm() -> ScriptedChatModel:
    return ScriptedChatModel()


@pytest.fixture
def deps(tmp_path, llm) -> Deps:
    return make_deps(tmp_path, llm=llm)


@pytest.fixture(autouse=True)
def _no_retry_waits(monkeypatch):
    """Tests that simulate provider failures should not sit through real retry delays."""
    import research_agents.llm as llm_module

    monkeypatch.setattr(llm_module, "CHAT_RETRY_DELAYS_S", (0.0, 0.0))
