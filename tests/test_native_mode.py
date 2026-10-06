"""Which models use native structured output, and which go straight to the JSON path."""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI

from research_agents.llm import native_structured_output_ok


def test_nvidia_served_models_skip_native_mode() -> None:
    nim = ChatNVIDIA(model="nvidia/nemotron-3-super-120b-a12b", api_key="nvapi-test")
    assert native_structured_output_ok(nim) is False
    compat = ChatOpenAI(
        model="vendor/model", api_key="test", base_url="https://integrate.api.nvidia.com/v1"
    )
    assert native_structured_output_ok(compat) is False


def test_other_models_keep_native_mode() -> None:
    assert native_structured_output_ok(ChatOpenAI(model="gpt-5-mini", api_key="test")) is True
    assert native_structured_output_ok(GenericFakeChatModel(messages=iter([]))) is True
