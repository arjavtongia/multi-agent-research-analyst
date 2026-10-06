"""A reply cut off at the output limit must never pass as a finished draft."""

from fakes import make_deps
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from research_agents.llm import was_truncated
from research_agents.nodes.writer import write_report


def cut_off(text: str) -> AIMessage:
    return AIMessage(content=text, response_metadata={"finish_reason": "length"})


def state(**extra):
    base = {
        "question": "How do passkeys work?",
        "plan": {"title": "Passkeys", "outline": ["How they work"]},
        "sources": [
            {"id": "S1", "title": "FIDO", "url": "https://fidoalliance.org", "kind": "page"}
        ],
        "evidence": [{"source_id": "S1", "claim": "Passkeys use key pairs.", "quote": "key pair"}],
    }
    base.update(extra)
    return base


def test_was_truncated_reads_provider_metadata():
    assert was_truncated(cut_off("x"))
    assert was_truncated(AIMessage(content="x", response_metadata={"stop_reason": "max_tokens"}))
    assert not was_truncated(AIMessage(content="x", response_metadata={"finish_reason": "stop"}))


def test_truncated_revision_keeps_the_previous_complete_draft(tmp_path):
    llm = GenericFakeChatModel(messages=iter([cut_off("## How they work\n\nPasskeys use [S1")]))
    previous = "## How they work\n\nPasskeys use key pairs [S1]."
    result = write_report(
        state(draft=previous, draft_version=1, revision_feedback="Cite sentence 2."),
        make_deps(tmp_path, llm=llm),
    )
    assert result["draft"] == previous
    assert result["draft_version"] == 2


def test_truncated_first_draft_is_marked_incomplete(tmp_path):
    llm = GenericFakeChatModel(messages=iter([cut_off("## How they work\n\nPasskeys use key")]))
    result = write_report(state(), make_deps(tmp_path, llm=llm))
    assert result["draft"].startswith("## How they work")
    assert "Evidence gap: this draft was cut off" in result["draft"]
