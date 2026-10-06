"""Drive the Streamlit UI headlessly with fake dependencies (skipped without the `ui` extra)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fakes import make_deps

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def _button(app, label: str):
    return next(b for b in app.button if b.label == label)


def test_ui_runs_pauses_for_review_and_approves(tmp_path, monkeypatch):
    import streamlit as st

    deps = make_deps(tmp_path)
    monkeypatch.setattr("research_agents.deps.build_default_deps", lambda settings=None: deps)
    st.cache_resource.clear()

    app = streamlit_testing.AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    assert not app.exception

    app.text_area[0].input("What is alpha?")
    app.run()  # the button enables once the question is committed
    _button(app, "Run research").click()
    app.run()
    assert not app.exception
    assert any("awaiting_review" in m.value for m in app.markdown)
    assert any(h.value == "Fact-check (draft v1)" for h in app.subheader)

    _button(app, "Approve and finalise").click()
    app.run()
    assert not app.exception
    assert any("Saved to" in s.value for s in app.success)
    st.cache_resource.clear()
