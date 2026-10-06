"""Dependencies injected into the graph: models, tools, settings and the tracer.

Building the graph from a `Deps` object keeps nodes free of global state, and lets tests swap
in a scripted model and fake search/fetch tools without patching anything.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel

from research_agents.config import Settings, get_settings
from research_agents.tools.fetch import Fetcher, HttpFetcher
from research_agents.tools.search import SearchTool, get_search_tool
from research_agents.tracing import Tracer


@dataclass
class Deps:
    settings: Settings
    llm: BaseChatModel
    fast_llm: BaseChatModel
    search: SearchTool
    fetcher: Fetcher
    tracer: Tracer


def build_default_deps(settings: Settings | None = None) -> Deps:
    """Real dependencies from settings. Raises MissingAPIKeyError if no key is configured."""
    from research_agents.llm import get_chat_model

    settings = settings or get_settings()
    return Deps(
        settings=settings,
        llm=get_chat_model(fast=False, settings=settings),
        fast_llm=get_chat_model(fast=True, settings=settings),
        search=get_search_tool(settings),
        fetcher=HttpFetcher(settings),
        tracer=Tracer(settings.runs_dir),
    )
