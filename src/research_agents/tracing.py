"""Per-node observability: timing, token usage and status appended to `trace.jsonl`.

LangSmith tracing needs no code here: LangChain picks it up from the environment when
LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphBubbleUp

from research_agents.llm import track_usage

_THREAD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

NodeFn = Callable[[dict[str, Any]], dict[str, Any]]


def validate_thread_id(thread_id: str) -> str:
    """Thread ids become directory names, so keep them to a safe character set."""
    if not _THREAD_ID_RE.match(thread_id):
        raise ValueError(
            f"invalid thread id {thread_id!r}: use 1-64 letters, digits, '.', '_' or '-'"
        )
    return thread_id


def run_dir(runs_dir: Path, thread_id: str) -> Path:
    return Path(runs_dir) / validate_thread_id(thread_id)


class Tracer:
    """Appends one JSON line per node execution to runs/<thread_id>/trace.jsonl."""

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = Path(runs_dir)
        self._lock = threading.Lock()

    def path(self, thread_id: str) -> Path:
        return run_dir(self.runs_dir, thread_id) / "trace.jsonl"

    def record(self, thread_id: str, record: dict[str, Any]) -> None:
        path = self.path(thread_id)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def traced(
    name: str, fn: NodeFn, tracer: Tracer
) -> Callable[[dict[str, Any], RunnableConfig], Any]:
    """Wrap a node so its duration, token usage and outcome are recorded.

    Token counts are also merged into the state update (`tokens_used`, `llm_calls`, `timings`),
    which is what the token budget and the eval report read.
    """

    def node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        thread_id = str(config.get("configurable", {}).get("thread_id", "no-thread"))
        started = time.perf_counter()
        status = "ok"
        error = ""
        with track_usage() as usage:
            try:
                update = fn(state) or {}
            except GraphBubbleUp:
                # interrupt() raises to pause the graph; that is not an error.
                status = "interrupted"
                raise
            except Exception as exc:
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
                record = {
                    "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    "thread_id": thread_id,
                    "node": name,
                    "status": status,
                    "duration_ms": elapsed_ms,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "llm_calls": usage.calls,
                }
                task = state.get("sub_question") if isinstance(state, dict) else None
                if isinstance(task, dict):
                    record["sub_question_id"] = task.get("id")
                if error:
                    record["error"] = error[:500]
                tracer.record(thread_id, record)

        timing = {
            "node": name,
            "duration_ms": elapsed_ms,
            "tokens": usage.total_tokens,
            "llm_calls": usage.calls,
        }
        if "sub_question_id" in record:
            timing["sub_question_id"] = record["sub_question_id"]
        update = dict(update)
        update["tokens_used"] = usage.total_tokens
        update["llm_calls"] = usage.calls
        update["timings"] = [timing]
        return update

    node.__name__ = name
    return node
