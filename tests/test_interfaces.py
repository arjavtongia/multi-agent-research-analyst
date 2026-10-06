"""CLI and HTTP API, both driven with fake dependencies."""

from __future__ import annotations

from fakes import make_deps
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from research_agents import api, cli


def test_cli_run_with_auto_approve(tmp_path, monkeypatch):
    deps = make_deps(tmp_path)
    monkeypatch.setattr(cli, "build_default_deps", lambda settings=None: deps)
    result = CliRunner().invoke(cli.app, ["run", "What is alpha?", "--auto-approve"])
    assert result.exit_code == 0, result.output
    assert "planner" in result.output and "finalize" in result.output
    assert "Run completed" in result.output


def test_cli_pause_then_resume(tmp_path, monkeypatch):
    deps = make_deps(tmp_path)
    monkeypatch.setattr(cli, "build_default_deps", lambda settings=None: deps)
    runner = CliRunner()
    paused = runner.invoke(cli.app, ["run", "What is alpha?", "--thread-id", "cli-1"])
    assert paused.exit_code == 0, paused.output
    assert "Waiting for review" in paused.output

    both = runner.invoke(cli.app, ["resume", "cli-1", "--approve", "--revise", "x"])
    assert both.exit_code == 1

    done = runner.invoke(cli.app, ["resume", "cli-1", "--approve"])
    assert done.exit_code == 0, done.output
    assert "Run completed" in done.output


def test_cli_reports_missing_key(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "get_settings", lambda: make_deps(tmp_path).settings)
    result = CliRunner().invoke(cli.app, ["run", "Anything?"])
    assert result.exit_code == 2
    assert "NVIDIA_API_KEY" in result.output


def test_api_run_review_and_report(tmp_path):
    deps = make_deps(tmp_path)
    api.app.dependency_overrides[api.get_deps] = lambda: deps
    try:
        client = TestClient(api.app)
        assert client.get("/health").json()["status"] == "ok"

        created = client.post("/runs", json={"question": "What is alpha?"})
        assert created.status_code == 202
        thread_id = created.json()["thread_id"]

        # TestClient runs background tasks before returning, so the run is now paused.
        state = client.get(f"/runs/{thread_id}").json()
        assert state["status"] == "awaiting_review"
        assert state["review_request"]["draft"]

        bad = client.post(f"/runs/{thread_id}/review", json={"action": "revise"})
        assert bad.status_code == 422
        ok = client.post(f"/runs/{thread_id}/review", json={"action": "approve"})
        assert ok.status_code == 202

        report = client.get(f"/runs/{thread_id}/report")
        assert report.status_code == 200
        assert "## Sources" in report.text
        assert client.get(f"/runs/{thread_id}").json()["status"] == "completed"
        assert client.get("/runs/does-not-exist").status_code == 404
    finally:
        api.app.dependency_overrides.clear()
