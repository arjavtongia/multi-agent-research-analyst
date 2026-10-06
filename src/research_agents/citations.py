"""Citation parsing, sentence splitting and the citation-coverage metric.

These are deterministic so the fact-checker's routing and the evaluation numbers do not depend
on a model's mood.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CITATION_RE = re.compile(r"\[S(\d+)\]")
_WIDE_BRACKETS = str.maketrans({"【": "[", "】": "]", "［": "[", "］": "]"})
# "[S1, S2]", "[S1; S3]", "[s2,S4]" and the same in parentheses.
_GROUP_RE = re.compile(r"[\[(]\s*([Ss]\d+(?:\s*[,;&]\s*(?:and\s+)?[Ss]?\d+)*)\s*[\])]")
_ID_IN_GROUP_RE = re.compile(r"[Ss]?(\d+)")
_SPACE_BEFORE_CITATION_RE = re.compile(r"(?<=[^\s\[\]])(\[S\d+\])")
_DUPLICATE_CITATION_RE = re.compile(r"(\[S\d+\])(\s*\1)+")
_CITATION_AFTER_STOP_RE = re.compile(r"([.!?])((?:\s*\[S\d+\])+)")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'“(\[]?[A-Z0-9])")
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_SOURCES_HEADING_RE = re.compile(
    r"^\s{0,3}#{1,6}\s*(sources|references|bibliography|citations|works cited)\b",
    re.IGNORECASE | re.MULTILINE,
)
_ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "U.S.",
    "U.K.",
    "E.U.",
    "Dr.",
    "Mr.",
    "Ms.",
    "Mrs.",
    "No.",
    "Fig.",
    "St.",
    "Inc.",
    "Ltd.",
    "Corp.",
    "approx.",
)
_NON_FACTUAL_PREFIXES = (
    "this report",
    "this section",
    "in this report",
    "in this section",
    "the following",
    "the sections below",
    "below,",
    "evidence gap:",
    "note:",
)
MIN_FACTUAL_WORDS = 6


@dataclass
class Sentence:
    id: int
    text: str
    section: str
    citations: list[str] = field(default_factory=list)
    factual: bool = False


def source_id(n: int) -> str:
    return f"S{n}"


def find_citations(text: str) -> list[str]:
    """Source ids cited in `text`, in order of first appearance."""
    seen: dict[str, None] = {}
    for match in CITATION_RE.finditer(text):
        seen.setdefault(f"S{int(match.group(1))}", None)
    return list(seen)


def normalize_citations(text: str, valid_ids: set[str] | None = None) -> tuple[str, list[str]]:
    """Rewrite citation variants to the canonical `[S1][S2]` form.

    Returns the cleaned text and the list of invalid ids that were removed (ids not in
    `valid_ids`). When `valid_ids` is None every id is kept.
    """

    def expand(match: re.Match[str]) -> str:
        ids = [f"S{int(n)}" for n in _ID_IN_GROUP_RE.findall(match.group(1))]
        return "".join(f"[{i}]" for i in ids)

    # Some models write CJK-style brackets (【S3】). In one live eval run that made 41
    # properly sourced sentences count as uncited.
    text = text.translate(_WIDE_BRACKETS)
    text = _GROUP_RE.sub(expand, text)
    removed: list[str] = []
    if valid_ids is not None:

        def drop_invalid(match: re.Match[str]) -> str:
            sid = f"S{int(match.group(1))}"
            if sid in valid_ids:
                return f"[{sid}]"
            removed.append(sid)
            return ""

        text = CITATION_RE.sub(drop_invalid, text)
    text = _DUPLICATE_CITATION_RE.sub(r"\1", text)
    text = _SPACE_BEFORE_CITATION_RE.sub(r" \1", text)
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)
    return text, sorted(set(removed), key=lambda s: int(s[1:]))


def strip_sources_section(markdown: str) -> str:
    """Drop a model-written Sources/References section; the real one is generated."""
    match = _SOURCES_HEADING_RE.search(markdown)
    return markdown[: match.start()].rstrip() if match else markdown.strip()


def strip_title(markdown: str) -> str:
    """Remove leading H1 lines (the title is added by finalize)."""
    lines = markdown.strip().splitlines()
    while lines and (not lines[0].strip() or re.match(r"^\s{0,3}#\s+", lines[0])):
        lines.pop(0)
    return "\n".join(lines).strip()


def _split_paragraph(paragraph: str) -> list[str]:
    paragraph = _CITATION_AFTER_STOP_RE.sub(r"\2\1", paragraph)
    pieces = _SENTENCE_BOUNDARY_RE.split(paragraph)
    merged: list[str] = []
    for piece in pieces:
        if merged and merged[-1].endswith(_ABBREVIATIONS):
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return [p.strip() for p in merged if p.strip()]


def is_factual(text: str) -> bool:
    """Heuristic: does this sentence make a factual claim that needs a citation?

    A sentence counts if it carries a citation, or if it has at least six words and is not a
    question, a signpost ("This report covers...") or a declared evidence gap.
    """
    if CITATION_RE.search(text):
        return True
    plain = text.strip().strip("*_").strip()
    if plain.endswith("?") or plain.lower().startswith(_NON_FACTUAL_PREFIXES):
        return False
    words = re.findall(r"[A-Za-z0-9%$€£]+", plain)
    return len(words) >= MIN_FACTUAL_WORDS


def split_sentences(markdown: str) -> list[Sentence]:
    """Split report markdown into sentences, tracking the section each belongs to.

    Headings, code blocks, table separators and the Sources section are skipped. List items
    and table rows are treated as sentences of their own.
    """
    body = strip_sources_section(markdown)
    sentences: list[Sentence] = []
    section = ""
    in_code = False
    paragraph: list[str] = []

    def flush() -> None:
        if not paragraph:
            return
        for text in _split_paragraph(" ".join(paragraph)):
            sentences.append(
                Sentence(
                    id=len(sentences) + 1,
                    text=text,
                    section=section,
                    citations=find_citations(text),
                    factual=is_factual(text),
                )
            )
        paragraph.clear()

    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        if line.strip().startswith("```"):
            flush()
            in_code = not in_code
            continue
        if in_code:
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            flush()
            section = heading.group(2).strip()
            continue
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if stripped.startswith("|"):
            flush()
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue
            paragraph.append(" | ".join(c for c in cells if c))
            flush()
            continue
        if _LIST_PREFIX_RE.match(stripped):
            flush()
            paragraph.append(_LIST_PREFIX_RE.sub("", stripped))
            continue
        paragraph.append(stripped.lstrip("> ").strip())
    flush()
    return sentences


def citation_coverage(markdown: str) -> float | None:
    """Share of factual sentences that carry at least one citation (None if there are none)."""
    factual = [s for s in split_sentences(markdown) if s.factual]
    if not factual:
        return None
    return sum(1 for s in factual if s.citations) / len(factual)
