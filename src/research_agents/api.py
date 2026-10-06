"""HTTP API (FastAPI). Runs execute in the background; poll the thread for status.

POST /runs                      start a run            -> {thread_id}
GET  /runs/{thread_id}          status, review request, fact-check summary
POST /runs/{thread_id}/review   approve or request a revision
GET  /runs/{thread_id}/report   final report markdown
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from research_agents import __version__
from research_agents.deps import Deps, build_default_deps
from research_agents.llm import MissingAPIKeyError
from research_agents.runner import get_run, new_thread_id, resume_run, start_run
from research_agents.schemas import ReviewDecision
from research_agents.tracing import run_dir, validate_thread_id

app = FastAPI(title="research-agents", version=__version__)


@lru_cache(maxsize=1)
def _default_deps() -> Deps:
    return build_default_deps()


def get_deps() -> Deps:
    try:
        return _default_deps()
    except MissingAPIKeyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


DepsDep = Annotated[Deps, Depends(get_deps)]


class RunRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    auto_approve: bool = False


class RunCreated(BaseModel):
    thread_id: str
    status: str = "started"


def _thread(thread_id: str) -> str:
    try:
        return validate_thread_id(thread_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/runs", response_model=RunCreated, status_code=202)
def create_run(request: RunRequest, background: BackgroundTasks, deps: DepsDep) -> RunCreated:
    thread_id = new_thread_id()
    background.add_task(
        start_run,
        request.question,
        deps=deps,
        thread_id=thread_id,
        auto_approve=request.auto_approve,
    )
    return RunCreated(thread_id=thread_id)


@app.get("/runs/{thread_id}")
def read_run(thread_id: str, deps: DepsDep) -> dict[str, Any]:
    result = get_run(_thread(thread_id), deps=deps)
    if result.status == "unknown":
        raise HTTPException(status_code=404, detail="No run with this thread id (yet).")
    values = result.values
    return {
        "thread_id": thread_id,
        "status": result.status,
        "question": values.get("question"),
        "plan": values.get("plan"),
        "sources": values.get("sources", []),
        "factcheck": values.get("factcheck", {}).get("summary"),
        "review_request": result.review_request,
        "tokens_used": values.get("tokens_used", 0),
        "output_dir": result.output_dir,
    }


@app.post("/runs/{thread_id}/review", status_code=202)
def review_run(
    thread_id: str,
    decision: ReviewDecision,
    background: BackgroundTasks,
    deps: DepsDep,
) -> dict[str, str]:
    current = get_run(_thread(thread_id), deps=deps)
    if current.status == "unknown":
        raise HTTPException(status_code=404, detail="No run with this thread id.")
    if not current.awaiting_review:
        raise HTTPException(
            status_code=409, detail=f"Run is not awaiting review ({current.status})."
        )
    if decision.action == "revise" and not decision.feedback:
        raise HTTPException(status_code=422, detail="A revise decision needs feedback.")
    background.add_task(resume_run, thread_id, decision, deps=deps)
    return {"thread_id": thread_id, "status": "resumed"}


@app.get("/runs/{thread_id}/report", response_class=PlainTextResponse)
def read_report(thread_id: str, deps: DepsDep) -> str:
    path: Path = run_dir(deps.settings.runs_dir, _thread(thread_id)) / "report.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not written yet.")
    return path.read_text(encoding="utf-8")
