from __future__ import annotations

import csv
from pathlib import Path

from fakes import make_deps

from research_agents.evaluation import load_questions, run_eval

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_bundled_question_set_is_well_formed():
    questions = load_questions(REPO_ROOT / "data" / "eval" / "questions.yaml")
    assert 8 <= len(questions) <= 12
    assert len({q.id for q in questions}) == len(questions)
    assert {q.topic for q in questions} == {"technology", "economics", "science"}
    assert all(q.question.endswith("?") for q in questions)


def test_eval_runs_questions_and_writes_reports(tmp_path):
    questions = tmp_path / "questions.yaml"
    questions.write_text(
        "questions:\n"
        "  - id: one\n    topic: test\n    question: First question?\n"
        "  - id: two\n    topic: test\n    question: Second question?\n"
        "  - id: three\n    topic: test\n    question: Third question?\n",
        encoding="utf-8",
    )
    deps = make_deps(tmp_path)
    rows, md_path, csv_path = run_eval(questions, tmp_path / "reports", deps=deps, limit=2)

    assert [r.id for r in rows] == ["one", "two"]
    assert all(r.status == "completed" for r in rows)
    assert all(r.citation_coverage == 1.0 and r.supported_rate == 1.0 for r in rows)
    assert all(r.unique_sources_cited >= 1 and r.tokens > 0 for r in rows)

    with csv_path.open(encoding="utf-8") as fh:
        records = list(csv.DictReader(fh))
    assert [r["id"] for r in records] == ["one", "two"]
    report = md_path.read_text(encoding="utf-8")
    assert "Mean citation coverage" in report
    assert "| one | test | completed |" in report


def test_eval_records_errors_without_stopping(tmp_path):
    class BrokenSearch:
        name = "broken"

        def search(self, query, max_results):
            raise RuntimeError("network down")

    questions = tmp_path / "q.yaml"
    questions.write_text("- Only question?\n", encoding="utf-8")
    deps = make_deps(tmp_path, search=BrokenSearch())
    rows, md_path, _ = run_eval(questions, tmp_path / "reports", deps=deps)
    assert rows[0].status == "error"
    assert "network down" in rows[0].error
    assert "## Errors" in md_path.read_text(encoding="utf-8")
