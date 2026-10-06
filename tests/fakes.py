"""Offline fakes: a scripted chat model that answers per agent role, and fake search/fetch."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr

from research_agents.config import Settings
from research_agents.deps import Deps
from research_agents.prompts import ROLE_PROMPTS
from research_agents.tools.fetch import FetchedPage, FetchError
from research_agents.tools.search import SearchResult
from research_agents.tracing import Tracer

Handler = Callable[[list[BaseMessage]], str] | str | list[str]


def detect_role(messages: Sequence[BaseMessage]) -> str:
    system = next((m for m in messages if isinstance(m, SystemMessage)), None)
    if system is None:
        return "unknown"
    first_line = str(system.content).strip().splitlines()[0]
    for role, template in ROLE_PROMPTS.items():
        if template.strip().splitlines()[0] == first_line:
            return role
    return "unknown"


def last_user_text(messages: Sequence[BaseMessage]) -> str:
    """The prompt body (the human message right after the system message)."""
    return str(messages[1].content) if len(messages) > 1 else ""


# ---- default per-role behaviour -----------------------------------------------------------


def default_planner(messages: list[BaseMessage]) -> str:
    return json.dumps(
        {
            "title": "Test report",
            "sub_questions": [
                {"question": "What is the current state of alpha?", "search_query": "alpha"},
                {"question": "How has beta changed recently?", "search_query": "beta"},
                {"question": "What are the risks of gamma?", "search_query": "gamma"},
            ],
            "outline": ["Overview", "Findings"],
        }
    )


_PAGE_RE = re.compile(r"<<<UNTRUSTED_WEB_CONTENT id=\w+ source=\"[^\"]*\">>>\n(.*?)\n<<<END", re.S)


def default_extractor(messages: list[BaseMessage]) -> str:
    match = _PAGE_RE.search(last_user_text(messages))
    text = match.group(1).split("\n\n", 1)[-1] if match else ""  # skip the "Title:" line
    first_sentence = re.split(r"(?<=\.)\s", text.strip())[0] if text.strip() else ""
    notes = (
        [{"claim": f"Claim: {first_sentence}", "quote": first_sentence}] if first_sentence else []
    )
    return json.dumps({"notes": notes})


def default_writer(messages: list[BaseMessage]) -> str:
    prompt = last_user_text(messages)
    outline = re.findall(r"^\d+\. (.+)$", prompt.split("Evidence:")[0], re.M)
    ids = re.findall(r"^\[(S\d+)\]", prompt, re.M)
    claims = re.findall(r"^  - claim: Claim: (.+)$", prompt, re.M)
    parts = []
    for i, heading in enumerate(outline or ["Findings"]):
        sid = ids[i % len(ids)] if ids else "S1"
        claim = claims[i % len(claims)] if claims else "Something happened in 2024"
        parts.append(f"## {heading}\n\n{claim.rstrip('.')} [{sid}].")
    return "\n\n".join(parts)


def supported_checker(messages: list[BaseMessage]) -> str:
    ids = re.findall(r"^(\d+)\. ", last_user_text(messages), re.M)
    return json.dumps(
        {"verdicts": [{"sentence_id": int(i), "verdict": "supported", "reason": "ok"} for i in ids]}
    )


def unsupported_checker(messages: list[BaseMessage]) -> str:
    ids = re.findall(r"^(\d+)\. ", last_user_text(messages), re.M)
    return json.dumps(
        {
            "verdicts": [
                {"sentence_id": int(i), "verdict": "unsupported", "reason": "number differs"}
                for i in ids
            ]
        }
    )


DEFAULT_HANDLERS: dict[str, Handler] = {
    "planner": default_planner,
    "extractor": default_extractor,
    "writer": default_writer,
    "fact_checker": supported_checker,
}


class ScriptedChatModel(BaseChatModel):
    """Answers based on which agent's system prompt it receives.

    A handler is a callable(messages) -> str, a fixed string, or a list of strings consumed in
    order. Safe to call from parallel researcher threads. Reports token usage like a real model.
    """

    handlers: dict[str, Any] = Field(default_factory=dict)
    _calls: list[str] = PrivateAttr(default_factory=list)
    _lock: Any = PrivateAttr(default_factory=threading.Lock)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    @property
    def calls(self) -> list[str]:
        return list(self._calls)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        role = detect_role(messages)
        with self._lock:
            self._calls.append(role)
            handler = self.handlers.get(role, DEFAULT_HANDLERS.get(role, ""))
            if isinstance(handler, list):
                text = handler.pop(0) if handler else ""
            elif isinstance(handler, str):
                text = handler
            else:
                text = None
        if text is None:
            text = handler(messages)
        prompt_chars = sum(len(str(m.content)) for m in messages)
        usage = {
            "input_tokens": prompt_chars // 4,
            "output_tokens": len(text) // 4,
            "total_tokens": prompt_chars // 4 + len(text) // 4,
        }
        message = AIMessage(content=text, usage_metadata=usage)
        return ChatResult(generations=[ChatGeneration(message=message)])


class FakeSearch:
    """Three results per query plus one URL shared by every query (to exercise dedupe)."""

    name = "fake"

    def __init__(self, per_query: int = 3, shared: bool = True) -> None:
        self.per_query = per_query
        self.shared = shared
        self.queries: list[str] = []
        self._lock = threading.Lock()

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        with self._lock:
            self.queries.append(query)
        slug = re.sub(r"\W+", "-", query.lower()).strip("-")
        results = [
            SearchResult(
                title=f"{query} page {i}",
                url=f"https://example.com/{slug}/{i}",
                snippet=f"Snippet {i} about {query}. " * 6,
            )
            for i in range(1, self.per_query + 1)
        ]
        if self.shared:
            results.append(
                SearchResult(
                    title="Shared overview",
                    url="https://www.shared.example.org/overview/?utm_source=search",
                    snippet="Shared snippet. " * 8,
                )
            )
        return results[:max_results]


class FakeFetcher:
    """Deterministic page text per URL. URLs listed in `fail` raise FetchError (paywall)."""

    def __init__(self, pages: dict[str, str] | None = None, fail: set[str] | None = None) -> None:
        self.pages = pages or {}
        self.fail = fail or set()
        self.fetched: list[str] = []
        self._lock = threading.Lock()

    def fetch(self, url: str) -> FetchedPage:
        with self._lock:
            self.fetched.append(url)
        if url in self.fail:
            raise FetchError("HTTP 402 (likely paywalled or blocked)")
        text = self.pages.get(url)
        if text is None:
            topic = url.rstrip("/").split("/")[-2] if url.count("/") > 3 else "topic"
            text = (
                f"The measured value for {topic} reached 42 percent in 2024 according to a survey. "
                f"Analysts expect {topic} to keep growing over the next five years. "
                + "Filler sentence for length. "
                * 20
            )
        return FetchedPage(url=url, title=f"Title of {url}", text=text)


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "runs_dir": tmp_path / "runs",
        "llm_provider": "nvidia",
        "search_provider": "duckduckgo",
        "min_page_chars": 0,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def make_deps(
    tmp_path: Path,
    *,
    llm: BaseChatModel | None = None,
    fast_llm: BaseChatModel | None = None,
    search: Any | None = None,
    fetcher: Any | None = None,
    **settings_overrides: Any,
) -> Deps:
    settings = make_settings(tmp_path, **settings_overrides)
    llm = llm or ScriptedChatModel()
    return Deps(
        settings=settings,
        llm=llm,
        fast_llm=fast_llm or llm,
        search=search or FakeSearch(),
        fetcher=fetcher or FakeFetcher(),
        tracer=Tracer(settings.runs_dir),
    )
