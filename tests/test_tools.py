from __future__ import annotations

import httpx
import pytest
from fakes import make_settings

from research_agents.tools import search as search_module
from research_agents.tools.fetch import FetchError, HttpFetcher, check_url
from research_agents.tools.search import (
    DuckDuckGoSearch,
    SearchError,
    TavilySearch,
    get_search_tool,
)

ARTICLE = """<html><head><title>Grid storage &amp; you</title></head><body>
<nav>Home | About | Login</nav>
<article><h1>Grid storage</h1>
<p>Utility-scale battery storage capacity in the country grew to 16 gigawatts in 2023,
according to the national energy agency, more than double the figure from two years earlier.</p>
<p>Most new capacity was installed next to existing solar farms, which lets operators shift
midday solar output into the evening peak when electricity prices are highest.</p>
<p>Analysts expect lithium iron phosphate cells to dominate new installations because they are
cheaper and less prone to thermal runaway than nickel-based chemistries.</p>
</article><footer>Copyright</footer></body></html>"""


def make_fetcher(tmp_path, handler, **overrides) -> HttpFetcher:
    settings = make_settings(tmp_path, min_page_chars=100, **overrides)
    return HttpFetcher(settings, transport=httpx.MockTransport(handler))


def test_fetch_extracts_readable_text_and_sends_user_agent(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["ua"] = request.headers["user-agent"]
        return httpx.Response(200, html=ARTICLE)

    page = make_fetcher(tmp_path, handler).fetch("https://news.example.com/storage")
    assert page.title == "Grid storage & you"
    assert "16 gigawatts" in page.text
    assert "Login" not in page.text
    assert seen["ua"].startswith("research-agents/")


def test_fetch_reports_paywall_and_wrong_content_type(tmp_path):
    paywalled = make_fetcher(tmp_path, lambda r: httpx.Response(402, text="Subscribe"))
    with pytest.raises(FetchError, match="paywalled"):
        paywalled.fetch("https://paper.example.com/a")

    pdf = make_fetcher(
        tmp_path,
        lambda r: httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"}),
    )
    with pytest.raises(FetchError, match="content type"):
        pdf.fetch("https://example.com/report")


def test_fetch_caps_download_size(tmp_path):
    script = "<script>var filler = '" + "x" * 50_000 + "';</script>"
    big = ARTICLE.replace("<nav>", script + "<nav>")
    capped = make_fetcher(tmp_path, lambda r: httpx.Response(200, html=big), fetch_max_bytes=2000)
    with pytest.raises(FetchError, match="no readable text"):
        # The cap cuts the document inside the script, before the article text.
        capped.fetch("https://example.com/big")
    roomy = make_fetcher(tmp_path, lambda r: httpx.Response(200, html=big))
    assert "16 gigawatts" in roomy.fetch("https://example.com/big").text


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/file",
        "http://localhost:8000/admin",
        "http://127.0.0.1/",
        "http://10.0.0.5/internal",
        "http://169.254.169.254/latest/meta-data",
        "https://example.com/paper.pdf",
    ],
)
def test_check_url_rejects_unsafe_or_unsupported_urls(url):
    with pytest.raises(FetchError):
        check_url(url)


def test_check_url_accepts_public_pages():
    check_url("https://en.wikipedia.org/wiki/Battery_storage")


class StubDDGS:
    """Plays back outcomes in order (repeating the last one) and records the backends used."""

    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.backends = []

    def text(self, query, max_results, backend):
        self.backends.append(backend)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_duckduckgo_search_maps_results_and_retries_rate_limits(tmp_path, monkeypatch):
    from ddgs.exceptions import RatelimitException

    monkeypatch.setattr(search_module.time, "sleep", lambda s: None)
    stub = StubDDGS(
        [
            RatelimitException("slow down"),
            [
                {"title": "A", "href": "https://a.com", "body": "about a"},
                {"title": "no url", "body": "skipped"},
            ],
        ]
    )
    tool = DuckDuckGoSearch(make_settings(tmp_path), client_factory=lambda: stub)
    results = tool.search("query", max_results=5)
    assert [(r.title, r.url, r.snippet) for r in results] == [("A", "https://a.com", "about a")]


def test_duckduckgo_search_gives_up_after_retries(tmp_path, monkeypatch):
    from ddgs.exceptions import RatelimitException

    monkeypatch.setattr(search_module.time, "sleep", lambda s: None)
    stub = StubDDGS([RatelimitException("slow")])
    tool = DuckDuckGoSearch(
        make_settings(tmp_path, search_max_retries=2, ddgs_backend="duckduckgo"),
        client_factory=lambda: stub,
    )
    with pytest.raises(SearchError, match="rate|slow"):
        tool.search("query", max_results=5)
    assert len(stub.backends) == 3  # one backend, three rounds


def test_no_results_everywhere_is_empty_list(tmp_path, monkeypatch):
    from ddgs.exceptions import DDGSException

    monkeypatch.setattr(search_module.time, "sleep", lambda s: None)
    stub = StubDDGS([DDGSException("No results found.")])
    tool = DuckDuckGoSearch(
        make_settings(tmp_path, search_max_retries=1, ddgs_backend="yahoo,bing"),
        client_factory=lambda: stub,
    )
    assert tool.search("query", max_results=5) == []
    assert stub.backends == ["yahoo", "bing", "yahoo", "bing"]


def test_no_results_from_one_backend_falls_through_to_the_next(tmp_path):
    from ddgs.exceptions import DDGSException

    stub = StubDDGS(
        [
            DDGSException("No results found."),
            [{"title": "B", "href": "https://b.com", "body": "from bing"}],
        ]
    )
    tool = DuckDuckGoSearch(
        make_settings(tmp_path, ddgs_backend="yahoo,bing,duckduckgo"), client_factory=lambda: stub
    )
    assert [r.url for r in tool.search("query", max_results=5)] == ["https://b.com"]
    assert stub.backends == ["yahoo", "bing"]


def test_search_provider_selection(tmp_path):
    assert isinstance(
        get_search_tool(make_settings(tmp_path, search_provider="auto")), DuckDuckGoSearch
    )
    with pytest.raises(SearchError, match="TAVILY_API_KEY"):
        get_search_tool(make_settings(tmp_path, search_provider="tavily"))
    with pytest.raises(SearchError):
        TavilySearch(make_settings(tmp_path, tavily_api_key=""))
