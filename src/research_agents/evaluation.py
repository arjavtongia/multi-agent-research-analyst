"""Evaluation: run each question with auto-approve and record quality and cost metrics.

Metrics per report:
- citation_coverage: share of factual sentences with at least one [S#] citation (deterministic,
  see `citations.is_factual`).
- supported_rate: share of cited factual sentences the fact-checker judged supported.
- unique_sources_cited: distinct source ids cited in the final report.
- revisions: automatic writer revisions triggered by the fact-checker.
- latency_s and tokens: wall-clock time and total tokens for the whole run.
"""

from __future__ import annotations

import csv
import re
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import yaml

from research_agents.citations import citation_coverage, find_citations
from research_agents.deps import Deps
from research_agents.llm import resolve_model_name
from research_agents.runner import new_thread_id, start_run


@dataclass
class EvalQuestion:
    id: str
    question: str
    topic: str = ""


@dataclass
class EvalRow:
    id: str
    topic: str
    question: str
    thread_id: str
    status: str
    citation_coverage: float | None
    supported_rate: float | None
    unique_sources_cited: int
    sources_collected: int
    revisions: int
    latency_s: float
    tokens: int
    error: str = ""


def load_questions(path: Path) -> list[EvalQuestion]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    items = data.get("questions", data) if isinstance(data, dict) else data
    questions = []
    for i, item in enumerate(items, start=1):
        if isinstance(item, str):
            item = {"question": item}
        questions.append(
            EvalQuestion(
                id=str(item.get("id", f"q{i:02d}")),
                question=str(item["question"]).strip(),
                topic=str(item.get("topic", "")),
            )
        )
    return questions


def evaluate_question(question: EvalQuestion, deps: Deps) -> EvalRow:
    safe_id = re.sub(r"[^A-Za-z0-9._-]", "-", question.id)[:24]
    thread_id = f"eval-{safe_id}-{new_thread_id()}"
    started = time.perf_counter()
    try:
        result = start_run(question.question, deps=deps, thread_id=thread_id, auto_approve=True)
    except Exception as exc:  # noqa: BLE001 - one failed question must not stop the eval
        return EvalRow(
            id=question.id,
            topic=question.topic,
            question=question.question,
            thread_id=thread_id,
            status="error",
            citation_coverage=None,
            supported_rate=None,
            unique_sources_cited=0,
            sources_collected=0,
            revisions=0,
            latency_s=round(time.perf_counter() - started, 2),
            tokens=0,
            error=f"{type(exc).__name__}: {exc}"[:300],
        )
    values = result.values
    draft = values.get("draft", "")
    summary = values.get("factcheck", {}).get("summary", {})
    return EvalRow(
        id=question.id,
        topic=question.topic,
        question=question.question,
        thread_id=thread_id,
        status=result.status,
        citation_coverage=citation_coverage(draft),
        supported_rate=summary.get("supported_rate"),
        unique_sources_cited=len(find_citations(draft)),
        sources_collected=len(values.get("sources", [])),
        revisions=values.get("revision_count", 0),
        latency_s=round(time.perf_counter() - started, 2),
        tokens=values.get("tokens_used", 0),
    )


def _mean(values: list[float | int | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return statistics.fmean(clean) if clean else None


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def write_csv(rows: list[EvalRow], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[f.name for f in fields(EvalRow)])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def render_markdown(rows: list[EvalRow], deps: Deps) -> str:
    settings = deps.settings
    ok = [r for r in rows if r.status != "error"]
    latencies = [r.latency_s for r in ok]
    lines = [
        "# Evaluation report",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `research-agents eval`.",
        "",
        "## Configuration",
        "",
        f"- Provider: `{settings.llm_provider}`, main model `{_model(deps, False)}`, "
        f"fast model `{_model(deps, True)}`",
        f"- Search: `{getattr(deps.search, 'name', 'unknown')}`, "
        f"{settings.max_sources_per_question} sources per sub-question",
        f"- Max revisions: {settings.max_revisions}, token budget: {settings.token_budget:,}",
        "- Human review: auto-approved",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Questions run | {len(rows)} ({len(rows) - len(ok)} errors) |",
        f"| Mean citation coverage | {_pct(_mean([r.citation_coverage for r in ok]))} |",
        f"| Mean supported-claim rate | {_pct(_mean([r.supported_rate for r in ok]))} |",
        f"| Mean unique sources cited | {_fmt(_mean([r.unique_sources_cited for r in ok]))} |",
        f"| Mean revisions used | {_fmt(_mean([r.revisions for r in ok]))} |",
        f"| Median latency (s) | {_fmt(statistics.median(latencies) if latencies else None)} |",
        f"| Mean tokens per report | {_fmt(_mean([r.tokens for r in ok]), 0)} |",
        "",
        "## Per question",
        "",
        "| id | topic | status | coverage | supported | sources cited | revisions | "
        "latency (s) | tokens |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.id} | {r.topic} | {r.status} | {_pct(r.citation_coverage)} | "
            f"{_pct(r.supported_rate)} | {r.unique_sources_cited} | {r.revisions} | "
            f"{r.latency_s:.1f} | {r.tokens:,} |"
        )
    errors = [r for r in rows if r.error]
    if errors:
        lines += ["", "## Errors", ""]
        lines += [f"- `{r.id}`: {r.error}" for r in errors]
    lines += [
        "",
        "## Metric definitions",
        "",
        "- **Citation coverage**: factual sentences with at least one `[S#]` citation divided by "
        "all factual sentences. A sentence is factual if it has a citation, or has six or more "
        "words and is not a question, signpost or declared evidence gap.",
        "- **Supported-claim rate**: cited factual sentences judged `supported` by the "
        "fact-checker on the final draft, divided by all cited factual sentences.",
        "- **Unique sources cited**: distinct source ids cited in the final report.",
        "- **Revisions used**: automatic rewrites triggered by the fact-checker "
        f"(capped at {settings.max_revisions}).",
        "- **Latency and tokens**: wall-clock seconds and total input+output tokens per run.",
        "",
        "The fact-checker is itself an LLM, so the supported-claim rate is a model judgement, "
        "not ground truth. Spot-check a few reports under `runs/` before trusting the numbers.",
    ]
    return "\n".join(lines) + "\n"


def _fmt(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:,.{digits}f}"


def _model(deps: Deps, fast: bool) -> str:
    try:
        return resolve_model_name(deps.settings, fast=fast)
    except Exception:  # noqa: BLE001 - only used for the report header
        return "unknown"


def run_eval(
    questions_path: Path,
    out_dir: Path,
    *,
    deps: Deps,
    limit: int | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[list[EvalRow], Path, Path]:
    questions = load_questions(questions_path)[: limit or None]
    rows: list[EvalRow] = []
    for i, question in enumerate(questions, start=1):
        if on_progress:
            on_progress(f"[{i}/{len(questions)}] {question.id}: {question.question}")
        rows.append(evaluate_question(question, deps))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "eval_report.md"
    csv_path = out_dir / "eval_results.csv"
    md_path.write_text(render_markdown(rows, deps), encoding="utf-8")
    write_csv(rows, csv_path)
    return rows, md_path, csv_path
