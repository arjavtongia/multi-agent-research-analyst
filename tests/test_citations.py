from __future__ import annotations

from fakes import ScriptedChatModel, make_deps

from research_agents.citations import (
    citation_coverage,
    find_citations,
    normalize_citations,
    split_sentences,
)
from research_agents.nodes.writer import clean_draft, write_report


def test_normalize_citation_variants():
    text, removed = normalize_citations("A [S1, S2]. B (S3). C [s4]. D[S1].")
    assert text == "A [S1][S2]. B [S3]. C [S4]. D [S1]."
    assert removed == []


def test_invalid_citations_are_removed_and_reported():
    text, removed = normalize_citations("Fact [S1][S9]. Other [S12].", {"S1"})
    assert text == "Fact [S1]. Other."
    assert removed == ["S9", "S12"]


def test_duplicate_adjacent_citations_collapse():
    text, _ = normalize_citations("Fact [S2] [S2][S3].")
    assert text == "Fact [S2][S3]."


def test_clean_draft_strips_fences_title_and_sources_section():
    raw = (
        "```markdown\n# My Title\n\n## Findings\n\nGrowth was 5% [S1, S2].\n\n"
        "## References\n- [S1] something\n```"
    )
    draft, removed = clean_draft(raw, {"S1", "S2"})
    assert draft == "## Findings\n\nGrowth was 5% [S1][S2]."
    assert removed == []


def test_writer_node_outputs_canonical_citations(tmp_path):
    reply = "# Title\n\n## Overview\n\nSales rose 4% in 2024 [S1; S2] and fell later [S7].\n"
    deps = make_deps(tmp_path, llm=ScriptedChatModel(handlers={"writer": reply}))
    state = {
        "question": "Q?",
        "plan": {"title": "T", "outline": ["Overview"]},
        "sources": [
            {"id": "S1", "title": "a", "url": "https://a.com", "kind": "page"},
            {"id": "S2", "title": "b", "url": "https://b.com", "kind": "page"},
        ],
        "evidence": [
            {"source_id": "S1", "claim": "c1", "quote": "q1"},
            {"source_id": "S2", "claim": "c2", "quote": "q2"},
        ],
    }
    update = write_report(state, deps)
    assert update["draft"] == "## Overview\n\nSales rose 4% in 2024 [S1][S2] and fell later."
    assert update["draft_version"] == 1
    assert find_citations(update["draft"]) == ["S1", "S2"]


def test_split_sentences_handles_citations_after_period_and_abbreviations():
    md = (
        "## Economy\n\n"
        "The U.S. economy grew 2.5% in 2024. [S1] Inflation fell to 3% by December [S2].\n\n"
        "- Unemployment stayed near 4% for the whole year [S3]\n"
        "## Sources\n- [S1] ignored"
    )
    sentences = split_sentences(md)
    assert [s.text for s in sentences] == [
        "The U.S. economy grew 2.5% in 2024 [S1].",
        "Inflation fell to 3% by December [S2].",
        "Unemployment stayed near 4% for the whole year [S3]",
    ]
    assert [s.citations for s in sentences] == [["S1"], ["S2"], ["S3"]]
    assert all(s.section == "Economy" for s in sentences)


def test_citation_coverage_counts_only_factual_sentences():
    md = (
        "## Findings\n\n"
        "This report covers three questions about energy markets.\n"
        "Solar capacity doubled between 2020 and 2024 [S1]. "
        "Wind capacity grew by about forty percent over the same period. "
        "Battery prices fell [S2]. "
        "Evidence gap: no source covered the cost of grid upgrades in detail. "
        "What happens next?"
    )
    # Factual: solar (cited), wind (uncited), battery (cited, short but cited) -> 2 of 3.
    assert citation_coverage(md) == 2 / 3


def test_citation_coverage_none_without_factual_sentences():
    assert citation_coverage("## Title only\n\nShort line.") is None


def test_wide_brackets_are_normalised():
    text, removed = normalize_citations(
        "Revenue rose 【S2】. Both held 【S1][S3】 and ［S1］.",
        {"S1", "S2", "S3"},
    )
    assert text == "Revenue rose [S2]. Both held [S1][S3] and [S1]."
    assert removed == []
