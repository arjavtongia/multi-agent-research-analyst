from __future__ import annotations

from fakes import FakeSearch, make_deps

from research_agents.nodes.aggregate import aggregate_notes, normalize_url
from research_agents.runner import start_run


def note(sq: str, url: str, quote: str, kind: str = "page") -> dict:
    return {
        "note_id": f"{sq}-n1",
        "sub_question_id": sq,
        "claim": f"claim from {url}",
        "quote": quote,
        "source_url": url,
        "source_title": f"title {url}",
        "source_kind": kind,
    }


def test_normalize_url_ignores_scheme_www_fragment_slash_and_tracking():
    variants = [
        "https://www.example.com/a/b/?utm_source=x#section",
        "http://example.com/a/b",
        "https://EXAMPLE.com/a/b/?fbclid=123",
    ]
    assert len({normalize_url(v) for v in variants}) == 1
    assert normalize_url("https://example.com/a?id=2") != normalize_url(
        "https://example.com/a?id=3"
    )


def test_aggregate_dedupes_sources_and_quotes():
    state = {
        "notes": [
            note("q1", "https://www.example.com/page/", "Same quote here."),
            note("q2", "https://example.com/page?utm_medium=email", "Same quote here!"),
            note("q2", "https://example.com/page", "A different quote."),
            note("q3", "https://other.org/x", "Other quote.", kind="snippet"),
        ]
    }
    update = aggregate_notes(state)
    sources = update["sources"]
    assert [s["id"] for s in sources] == ["S1", "S2"]
    assert sources[0]["sub_question_ids"] == ["q1", "q2"]
    assert sources[0]["note_count"] == 2  # duplicate quote dropped
    assert sources[1]["kind"] == "snippet"
    assert [e["source_id"] for e in update["evidence"]] == ["S1", "S1", "S2"]


def test_source_ids_stay_stable_when_new_notes_arrive():
    first = aggregate_notes({"notes": [note("q1", "https://a.com/1", "Quote one.")]})
    later_notes = [
        note("q1", "https://a.com/1", "Quote one."),
        note("gap1-1", "https://b.com/2", "Quote two."),
        note("gap1-1", "https://a.com/1/", "Quote three."),
    ]
    second = aggregate_notes({"sources": first["sources"], "notes": later_notes})
    ids = {s["url"]: s["id"] for s in second["sources"]}
    assert ids == {"https://a.com/1": "S1", "https://b.com/2": "S2"}
    assert first["sources"][0].get("sub_question_ids") == ["q1"]  # input not mutated


def test_shared_url_across_sub_questions_becomes_one_source(tmp_path):
    class SharedFirst(FakeSearch):
        def search(self, query, max_results):
            results = super().search(query, max_results)
            return [results[-1], *results[:-1]]  # shared overview ranks first everywhere

    deps = make_deps(tmp_path, search=SharedFirst(), max_sources_per_question=2)
    result = start_run("Q?", deps=deps, auto_approve=True)
    sources = result.values["sources"]
    shared = [s for s in sources if "shared.example.org" in s["url"]]
    assert len(shared) == 1
    assert shared[0]["id"] == "S1"
    assert sorted(shared[0]["sub_question_ids"]) == ["q1", "q2", "q3"]
    assert len(sources) == 4  # 1 shared + 1 unique page per sub-question
