"""Command line interface: `research-agents run | resume | eval | serve`."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from research_agents.config import get_settings
from research_agents.deps import Deps, build_default_deps
from research_agents.llm import MissingAPIKeyError
from research_agents.runner import (
    NotAwaitingReviewError,
    RunEvent,
    RunNotFoundError,
    RunResult,
    describe_event,
    new_thread_id,
    resume_run,
    start_run,
)
from research_agents.schemas import ReviewDecision
from research_agents.tools.search import SearchError

app = typer.Typer(
    help="A team of LLM agents that researches a question on the web and writes a cited, "
    "fact-checked report, with a human approval step.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _deps() -> Deps:
    try:
        return build_default_deps(get_settings())
    except (MissingAPIKeyError, SearchError) as exc:
        console.print(f"[bold red]Configuration error:[/] {exc}")
        raise typer.Exit(code=2) from exc


def _report_failure(exc: Exception, thread_id: str) -> NoReturn:
    console.print(
        f"[bold red]Run failed:[/] {type(exc).__name__}: {exc}\n"
        f"Per-node trace: runs/{thread_id}/trace.jsonl"
    )
    raise typer.Exit(code=1) from exc


def _print_event(event: RunEvent) -> None:
    console.print(f"[cyan]>[/] {describe_event(event)}")


def _print_result(result: RunResult) -> None:
    values = result.values
    if result.awaiting_review:
        request = result.review_request or {}
        summary = request.get("factcheck_summary", {})
        console.print(
            Panel(
                Markdown(request.get("draft", "")),
                title=f"Draft v{request.get('draft_version', '?')}: {request.get('title', '')}",
            )
        )
        if summary:
            console.print(
                f"Fact-check: {summary.get('supported', 0)} supported, "
                f"{summary.get('unsupported', 0)} unsupported, "
                f"{summary.get('missing_citation', 0)} missing a citation."
            )
        for issue in request.get("open_issues", []):
            console.print(
                f"  [yellow]{issue['verdict']}[/]: {issue['sentence']} ({issue['reason']})"
            )
        console.print(
            f"\n[bold]Waiting for review.[/] Thread id: [bold]{result.thread_id}[/]\n"
            f"Draft saved to runs/{result.thread_id}/review_draft.md\n"
            f"  research-agents resume {result.thread_id} --approve\n"
            f'  research-agents resume {result.thread_id} --revise "what to change"'
        )
        return
    console.print(
        f"\n[bold green]Run {result.status}.[/] Thread id: {result.thread_id}. "
        f"Tokens used: {values.get('tokens_used', 0):,}."
    )
    if result.output_dir:
        console.print(f"Report: {Path(result.output_dir) / 'report.md'}")


@app.command()
def run(
    question: str = typer.Argument(..., help="The research question, in quotes."),
    auto_approve: bool = typer.Option(
        False, "--auto-approve", help="Approve the report automatically instead of pausing."
    ),
    thread_id: str | None = typer.Option(None, "--thread-id", help="Custom thread id."),
) -> None:
    """Research a question and draft a cited report (pauses for human review)."""
    deps = _deps()
    thread_id = thread_id or new_thread_id()
    try:
        result = start_run(
            question,
            deps=deps,
            thread_id=thread_id,
            auto_approve=auto_approve,
            on_event=_print_event,
        )
    except ValueError as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # noqa: BLE001 - provider/network errors: show, don't dump a trace
        _report_failure(exc, thread_id)
    _print_result(result)


@app.command()
def resume(
    thread_id: str = typer.Argument(..., help="Thread id printed when the run paused."),
    approve: bool = typer.Option(False, "--approve", help="Approve the draft and finalise."),
    revise: str | None = typer.Option(None, "--revise", help="Feedback for another revision."),
    auto_approve: bool = typer.Option(
        False, "--auto-approve", help="Approve automatically after the revision."
    ),
) -> None:
    """Resume a paused run with an approval or revision feedback."""
    if approve == bool(revise):
        console.print('[bold red]Pass exactly one of --approve or --revise "feedback".[/]')
        raise typer.Exit(code=1)
    decision = (
        ReviewDecision(action="approve")
        if approve
        else ReviewDecision(action="revise", feedback=revise or "")
    )
    deps = _deps()
    try:
        result = resume_run(
            thread_id, decision, deps=deps, auto_approve=auto_approve, on_event=_print_event
        )
    except (RunNotFoundError, NotAwaitingReviewError, ValueError) as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # noqa: BLE001
        _report_failure(exc, thread_id)
    _print_result(result)


@app.command(name="eval")
def eval_command(
    questions: Path = typer.Option(
        Path("data/eval/questions.yaml"), "--questions", help="YAML file of eval questions."
    ),
    out_dir: Path = typer.Option(Path("reports"), "--out", help="Where to write the report."),
    limit: int | None = typer.Option(None, "--limit", help="Only run the first N questions."),
) -> None:
    """Run every eval question with auto-approve and write reports/eval_report.md + CSV."""
    from research_agents.evaluation import run_eval

    deps = _deps()
    rows, md_path, csv_path = run_eval(
        questions,
        out_dir,
        deps=deps,
        limit=limit,
        on_progress=lambda msg: console.print(f"[cyan]>[/] {msg}"),
    )
    table = Table("id", "status", "coverage", "supported", "sources", "revisions", "latency s")
    for row in rows:
        table.add_row(
            row.id,
            row.status,
            "n/a" if row.citation_coverage is None else f"{row.citation_coverage:.0%}",
            "n/a" if row.supported_rate is None else f"{row.supported_rate:.0%}",
            str(row.unique_sources_cited),
            str(row.revisions),
            f"{row.latency_s:.1f}",
        )
    console.print(table)
    console.print(f"Wrote {md_path} and {csv_path}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address."),
    port: int = typer.Option(8000, help="Port."),
) -> None:
    """Start the HTTP API (FastAPI + uvicorn)."""
    import uvicorn

    uvicorn.run("research_agents.api:app", host=host, port=port)


def main() -> None:
    # Export .env values (for example LANGSMITH_TRACING) to the process environment so
    # LangChain's built-in LangSmith tracing sees them.
    load_dotenv(override=False)
    app()


if __name__ == "__main__":
    main()
