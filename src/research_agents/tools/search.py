"""Web search tools: DuckDuckGo through `ddgs` (no key) by default, Tavily when a key is set."""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

from research_agents.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class SearchTool(Protocol):
    name: str

    def search(self, query: str, max_results: int) -> list[SearchResult]: ...


class SearchError(RuntimeError):
    """Raised when a search provider fails after retries."""


_SEMAPHORES: dict[int, threading.BoundedSemaphore] = {}
_SEMAPHORES_LOCK = threading.Lock()


def _shared_semaphore(size: int) -> threading.BoundedSemaphore:
    """One semaphore per configured size, shared by every search tool in the process."""
    with _SEMAPHORES_LOCK:
        if size not in _SEMAPHORES:
            _SEMAPHORES[size] = threading.BoundedSemaphore(size)
        return _SEMAPHORES[size]


class DuckDuckGoSearch:
    """`ddgs` text search with retry, jittered backoff and a process-wide concurrency cap.

    Parallel researchers would otherwise fire several searches at once, which is the quickest
    way to get rate limited by a keyless endpoint.
    """

    name = "duckduckgo"

    def __init__(self, settings: Settings, client_factory: Any | None = None) -> None:
        self.settings = settings
        self._client_factory = client_factory
        self._semaphore = _shared_semaphore(settings.search_max_concurrency)

    def _client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory()
        from ddgs import DDGS

        return DDGS(timeout=int(self.settings.fetch_timeout_s))

    def backends(self) -> list[str]:
        """Backends to try in order, from the comma-separated `DDGS_BACKEND` setting."""
        names = [b.strip() for b in self.settings.ddgs_backend.split(",") if b.strip()]
        return names or ["duckduckgo"]

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        """Try each backend in turn; back off and repeat the round if all of them fail.

        Keyless engines soft-block bursts by answering "No results found" for queries that do
        have results. In a live check (2026-10-06) the duckduckgo backend alone answered 1 of 5
        ordinary queries, so "no results" from one backend is treated as "ask the next one",
        and an empty list is returned only when every backend says so in every round.
        """
        from ddgs.exceptions import DDGSException

        rounds = self.settings.search_max_retries + 1
        last_error: Exception | None = None
        only_empty = True
        for attempt in range(rounds):
            for backend in self.backends():
                try:
                    with self._semaphore:
                        raw = self._client().text(query, max_results=max_results, backend=backend)
                except DDGSException as exc:  # includes rate-limit and timeout subclasses
                    if "no results" not in str(exc).lower():
                        only_empty = False
                        last_error = exc
                    logger.info("search %r: backend %s failed (%s)", query, backend, exc)
                    continue
                results = [
                    SearchResult(
                        title=str(r.get("title", "")).strip(),
                        url=str(r.get("href") or r.get("url") or "").strip(),
                        snippet=str(r.get("body", "")).strip(),
                    )
                    for r in raw
                    if r.get("href") or r.get("url")
                ]
                if results:
                    return results
            if attempt + 1 < rounds:
                delay = min(2**attempt + random.uniform(0, 1), 20)
                logger.warning("search %r: no backend answered, retry in %.1fs", query, delay)
                time.sleep(delay)
        if only_empty:
            return []
        raise SearchError(f"Web search failed for {query!r}: {last_error}")


class TavilySearch:
    """Tavily search API (needs TAVILY_API_KEY and the `tavily` extra)."""

    name = "tavily"

    def __init__(self, settings: Settings) -> None:
        if settings.tavily_api_key is None or not settings.tavily_api_key.get_secret_value():
            raise SearchError("Set TAVILY_API_KEY in .env to use the Tavily search provider.")
        try:
            from tavily import TavilyClient
        except ImportError as exc:
            raise SearchError(
                "Tavily search needs the optional extra: uv sync --extra tavily"
            ) from exc
        self._client = TavilyClient(api_key=settings.tavily_api_key.get_secret_value())

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        try:
            response = self._client.search(query, max_results=max_results)
        except Exception as exc:  # noqa: BLE001 - report every client error the same way
            raise SearchError(f"Tavily search failed for {query!r}: {exc}") from exc
        return [
            SearchResult(
                title=str(r.get("title", "")).strip(),
                url=str(r.get("url", "")).strip(),
                snippet=str(r.get("content", "")).strip(),
            )
            for r in response.get("results", [])
            if r.get("url")
        ]


def get_search_tool(settings: Settings) -> SearchTool:
    """Pick the search provider: Tavily if configured (or requested), DuckDuckGo otherwise."""
    has_tavily = bool(settings.tavily_api_key and settings.tavily_api_key.get_secret_value())
    if settings.search_provider == "tavily" or (settings.search_provider == "auto" and has_tavily):
        return TavilySearch(settings)
    return DuckDuckGoSearch(settings)
