"""Tools used by the researcher: web search, page fetching and untrusted-content handling."""

from research_agents.tools.fetch import FetchedPage, Fetcher, FetchError, HttpFetcher
from research_agents.tools.search import (
    DuckDuckGoSearch,
    SearchError,
    SearchResult,
    SearchTool,
    TavilySearch,
    get_search_tool,
)
from research_agents.tools.untrusted import UNTRUSTED_RULES, wrap_untrusted

__all__ = [
    "DuckDuckGoSearch",
    "FetchError",
    "FetchedPage",
    "Fetcher",
    "HttpFetcher",
    "SearchError",
    "SearchResult",
    "SearchTool",
    "TavilySearch",
    "UNTRUSTED_RULES",
    "get_search_tool",
    "wrap_untrusted",
]
