"""Page fetching with httpx and readable-text extraction with trafilatura."""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from html import unescape
from typing import Protocol
from urllib.parse import urlparse

import httpx

from research_agents.config import Settings

_ALLOWED_TYPES = ("text/html", "application/xhtml+xml", "text/plain")
_SKIPPED_EXTENSIONS = (".pdf", ".zip", ".mp4", ".mp3", ".png", ".jpg", ".jpeg", ".gif", ".exe")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class FetchedPage:
    url: str
    title: str
    text: str


class FetchError(RuntimeError):
    """The page could not be fetched or had no readable text."""


class Fetcher(Protocol):
    def fetch(self, url: str) -> FetchedPage: ...


def check_url(url: str) -> None:
    """Reject non-http(s) URLs, local hosts and private IPs (URLs come from untrusted results)."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise FetchError(f"unsupported URL scheme: {parsed.scheme or 'none'}")
    host = (parsed.hostname or "").lower()
    if not host or host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise FetchError(f"refusing to fetch local host: {host or 'none'}")
    if parsed.path.lower().endswith(_SKIPPED_EXTENSIONS):
        raise FetchError("binary or PDF document skipped")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
        raise FetchError(f"refusing to fetch private address: {host}")


def extract_readable_text(html: str, url: str) -> tuple[str, str]:
    """Return (title, main text) for an HTML document using trafilatura."""
    import trafilatura

    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        favor_precision=True,
    )
    match = _TITLE_RE.search(html)
    title = unescape(re.sub(r"\s+", " ", match.group(1))).strip() if match else ""
    return title, (text or "").strip()


class HttpFetcher:
    """Polite fetcher: identifies itself, follows redirects, enforces timeouts and a size cap."""

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings
        self._client = httpx.Client(
            timeout=httpx.Timeout(settings.fetch_timeout_s),
            follow_redirects=True,
            max_redirects=5,
            headers={
                "User-Agent": settings.user_agent,
                "Accept": "text/html,application/xhtml+xml;q=0.9,text/plain;q=0.5",
                "Accept-Language": "en;q=0.9",
            },
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def fetch(self, url: str) -> FetchedPage:
        check_url(url)
        try:
            with self._client.stream("GET", url) as response:
                if response.status_code in (401, 402, 403):
                    raise FetchError(f"HTTP {response.status_code} (likely paywalled or blocked)")
                if response.status_code >= 400:
                    raise FetchError(f"HTTP {response.status_code}")
                content_type = response.headers.get("content-type", "").lower()
                if content_type and not content_type.startswith(_ALLOWED_TYPES):
                    raise FetchError(f"unsupported content type: {content_type.split(';')[0]}")
                body = bytearray()
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if len(body) > self.settings.fetch_max_bytes:
                        break
                encoding = response.encoding or "utf-8"
                final_url = str(response.url)
        except httpx.HTTPError as exc:
            raise FetchError(f"{type(exc).__name__}: {exc}") from exc

        html = bytes(body[: self.settings.fetch_max_bytes]).decode(encoding, errors="replace")
        if content_type.startswith("text/plain"):
            title, text = "", html.strip()
        else:
            title, text = extract_readable_text(html, final_url)
        if len(text) < self.settings.min_page_chars:
            raise FetchError(
                "no readable text (page may be paywalled, script-rendered or mostly navigation)"
            )
        return FetchedPage(url=final_url, title=title, text=text)
