"""LLM provider layer: model factory, structured output with fallbacks, and token accounting.

NVIDIA NIM is the default provider. Provider packages are imported lazily so a missing optional
dependency only fails for the provider that needs it.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.tracers.context import register_configure_hook
from pydantic import BaseModel, ValidationError

from research_agents.config import Settings, get_settings

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

DEFAULT_MODELS: dict[str, dict[str, str]] = {
    "nvidia": {
        "chat": "nvidia/nemotron-3-super-120b-a12b",
        "fast": "openai/gpt-oss-20b",
    },
    "openai": {"chat": "gpt-5-mini", "fast": "gpt-5-nano"},
    "anthropic": {"chat": "claude-sonnet-5-5", "fast": "claude-haiku-4-5-20251001"},
}


class MissingAPIKeyError(RuntimeError):
    """Raised when the selected provider has no API key configured."""


class StructuredOutputError(RuntimeError):
    """Raised when a model cannot produce output that validates against the schema."""


# --------------------------------------------------------------------------------------------
# Model factory
# --------------------------------------------------------------------------------------------


def _require(value: Any, message: str) -> Any:
    if value is None or (hasattr(value, "get_secret_value") and not value.get_secret_value()):
        raise MissingAPIKeyError(message)
    return value


def resolve_model_name(settings: Settings, fast: bool = False) -> str:
    """Return the model id (or Azure deployment name) for the main or fast model."""
    override = settings.llm_fast_model if fast else settings.llm_model
    if override:
        return override
    if settings.llm_provider == "azure":
        deployment = (
            settings.azure_openai_fast_deployment if fast else settings.azure_openai_chat_deployment
        )
        if fast and not deployment:
            deployment = settings.azure_openai_chat_deployment
        return _require(
            deployment,
            "Set AZURE_OPENAI_CHAT_DEPLOYMENT (and optionally AZURE_OPENAI_FAST_DEPLOYMENT) "
            "in .env to the names of your Azure OpenAI deployments.",
        )
    return DEFAULT_MODELS[settings.llm_provider]["fast" if fast else "chat"]


def get_chat_model(fast: bool = False, settings: Settings | None = None) -> BaseChatModel:
    """Build the chat model for the configured provider.

    `fast=True` returns the cheaper model used for high-volume steps (note extraction).
    """
    settings = settings or get_settings()
    provider = settings.llm_provider
    model = resolve_model_name(settings, fast=fast)
    common: dict[str, Any] = {"temperature": settings.llm_temperature}

    if provider == "nvidia":
        key = _require(
            settings.nvidia_api_key,
            "Set NVIDIA_API_KEY in .env, get a free key at build.nvidia.com",
        )
        return _nvidia_chat(model, key.get_secret_value(), settings)

    if provider == "openai":
        key = _require(
            settings.openai_api_key,
            "Set OPENAI_API_KEY in .env (or switch LLM_PROVIDER back to nvidia).",
        )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=key.get_secret_value(),
            timeout=settings.llm_timeout_s,
            max_retries=2,
            **common,
        )

    if provider == "azure":
        key = _require(settings.azure_openai_api_key, "Set AZURE_OPENAI_API_KEY in .env.")
        endpoint = _require(settings.azure_openai_endpoint, "Set AZURE_OPENAI_ENDPOINT in .env.")
        version = _require(
            settings.azure_openai_api_version,
            "Set AZURE_OPENAI_API_VERSION in .env (for example 2024-10-21).",
        )
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_deployment=model,
            azure_endpoint=endpoint,
            api_version=version,
            api_key=key.get_secret_value(),
            timeout=settings.llm_timeout_s,
            max_retries=2,
            **common,
        )

    if provider == "anthropic":
        key = _require(
            settings.anthropic_api_key,
            "Set ANTHROPIC_API_KEY in .env (or switch LLM_PROVIDER back to nvidia).",
        )
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - depends on installed extras
            raise RuntimeError(
                "The anthropic provider needs the optional extra: uv sync --extra anthropic"
            ) from exc
        return ChatAnthropic(
            model=model,
            api_key=key.get_secret_value(),
            max_tokens=settings.llm_max_output_tokens,
            timeout=settings.llm_timeout_s,
            max_retries=2,
            **common,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


def _nvidia_chat(model: str, api_key: str, settings: Settings) -> BaseChatModel:
    """ChatNVIDIA for model ids it knows; the OpenAI-compatible client for newer ids.

    ChatNVIDIA validates model ids against a static table shipped with the package. For ids
    newer than that table it makes an extra network call at construction and warns that
    inference may fail, so those ids go through `ChatOpenAI` pointed at the same endpoint.
    """
    from langchain_nvidia_ai_endpoints._statics import determine_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        known = determine_model(model) is not None

    if known:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        return ChatNVIDIA(
            model=model,
            api_key=api_key,
            base_url=settings.nvidia_base_url,
            temperature=settings.llm_temperature,
            max_completion_tokens=settings.llm_max_output_tokens,
            timeout=settings.llm_timeout_s,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=settings.nvidia_base_url,
        temperature=settings.llm_temperature,
        max_completion_tokens=settings.llm_max_output_tokens,
        timeout=settings.llm_timeout_s,
        max_retries=2,
    )


# --------------------------------------------------------------------------------------------
# Token accounting
# --------------------------------------------------------------------------------------------


class UsageCounter(BaseCallbackHandler):
    """Callback handler that sums token usage across every chat model call it sees."""

    def __init__(self) -> None:
        super().__init__()
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self._lock = threading.Lock()

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for generation in generations:
                if not isinstance(generation, ChatGeneration):
                    continue
                message = generation.message
                usage = getattr(message, "usage_metadata", None) or {}
                with self._lock:
                    self.input_tokens += int(usage.get("input_tokens", 0) or 0)
                    self.output_tokens += int(usage.get("output_tokens", 0) or 0)
        with self._lock:
            self.calls += 1


_usage_var: ContextVar[UsageCounter | None] = ContextVar("research_agents_usage", default=None)
register_configure_hook(_usage_var, inheritable=True)


@contextmanager
def track_usage() -> Iterator[UsageCounter]:
    """Count tokens for every LLM call made inside the block (including structured output).

    Uses a LangChain configure hook, so it adds a handler without replacing callbacks that
    are already attached (LangSmith tracing keeps working).
    """
    counter = UsageCounter()
    token = _usage_var.set(counter)
    try:
        yield counter
    finally:
        _usage_var.reset(token)


# --------------------------------------------------------------------------------------------
# Calling helpers
# --------------------------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json|markdown|md)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def message_text(message: BaseMessage) -> str:
    """Plain text of a model reply with any `<think>` reasoning block removed."""
    text = str(message.text) if hasattr(message, "text") else str(message.content)
    text = _THINK_RE.sub("", text)
    # Some reasoning models emit an unterminated think block when truncated.
    if "<think>" in text.lower() and "</think>" not in text.lower():
        text = text[: text.lower().index("<think>")]
    return text.strip()


class OutputTruncatedError(RuntimeError):
    """The model stopped at its output-token limit, so the reply is incomplete."""

    def __init__(self, text: str) -> None:
        super().__init__("model output hit the token limit; raise LLM_MAX_OUTPUT_TOKENS")
        self.text = text


def was_truncated(reply: BaseMessage) -> bool:
    """True when the provider reports the reply stopped at the output limit."""
    meta = getattr(reply, "response_metadata", None) or {}
    reason = meta.get("finish_reason") or meta.get("stop_reason") or ""
    return str(reason).lower() in {"length", "max_tokens"}


# Transient failures are retried here, on top of any retries inside the provider client
# (ChatNVIDIA has none). In a live run on 2026-10-06 one slow reply from the free NIM tier
# raised a ReadTimeout and ended a research run partway through.
CHAT_RETRY_DELAYS_S: tuple[float, ...] = (5.0, 20.0)
_TRANSIENT_ERROR_NAMES = {
    "ReadTimeout",
    "ConnectTimeout",
    "Timeout",
    "TimeoutError",
    "TimeoutException",
    "ReadError",
    "ConnectionError",
    "RemoteProtocolError",
    "APITimeoutError",
    "APIConnectionError",
    "RateLimitError",
    "InternalServerError",
}


def is_transient(exc: BaseException) -> bool:
    """Timeouts, dropped connections, 429s and 5xx answers: worth one more try."""
    if any(cls.__name__ in _TRANSIENT_ERROR_NAMES for cls in type(exc).__mro__):
        return True
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    return isinstance(status, int) and (status == 429 or status >= 500)


def chat_call(
    llm: BaseChatModel, messages: Sequence[BaseMessage], *, raise_on_truncation: bool = False
) -> str:
    """Invoke the model and return cleaned text, retrying transient provider failures.

    With `raise_on_truncation`, a reply cut off at the output limit raises
    `OutputTruncatedError` (carrying the partial text) instead of passing as complete.
    """
    for attempt, delay in enumerate((*CHAT_RETRY_DELAYS_S, None)):
        try:
            reply = llm.invoke(list(messages))
            break
        except Exception as exc:
            if delay is None or not is_transient(exc):
                raise
            logger.warning(
                "model call failed (%s), retry %d in %.0fs", type(exc).__name__, attempt + 1, delay
            )
            time.sleep(delay)
    text = message_text(reply)
    if raise_on_truncation and was_truncated(reply):
        raise OutputTruncatedError(text)
    return text


def extract_json(text: str) -> Any:
    """Parse the first JSON object or array in a model reply (handles code fences and prose)."""
    text = _THINK_RE.sub("", text).strip()
    candidates = [m.group(1).strip() for m in _FENCE_RE.finditer(text)] + [text]
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        start_positions = [i for i in (candidate.find("{"), candidate.find("[")) if i >= 0]
        if not start_positions:
            continue
        start = min(start_positions)
        try:
            obj, _ = json.JSONDecoder().raw_decode(candidate[start:])
            return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON object found in model reply")


def _json_instruction(schema: type[BaseModel]) -> str:
    return (
        "Reply with a single JSON object and nothing else: no prose, no markdown fences. "
        "The object must validate against this JSON schema:\n"
        + json.dumps(schema.model_json_schema(), indent=None)
    )


def _parse(schema: type[SchemaT], text: str) -> SchemaT:
    data = extract_json(text)
    if isinstance(data, list) and len(schema.model_fields) == 1:
        # Models sometimes return the bare list for a single-field wrapper object.
        data = {next(iter(schema.model_fields)): data}
    return schema.model_validate(data)


# Models that fail native structured output repeatedly skip it for the rest of the process,
# so a provider without tool calling or json_schema support does not cost an extra request on
# every call.
NATIVE_FAILURE_LIMIT = 2
_native_failures: dict[str, int] = {}
_native_lock = threading.Lock()


def _model_key(llm: BaseChatModel) -> str:
    name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or ""
    return f"{type(llm).__name__}:{name}"


def _native_enabled(key: str) -> bool:
    with _native_lock:
        return _native_failures.get(key, 0) < NATIVE_FAILURE_LIMIT


def _record_native_failure(key: str) -> None:
    with _native_lock:
        _native_failures[key] = _native_failures.get(key, 0) + 1


def native_structured_output_ok(llm: BaseChatModel) -> bool:
    """False for models served by NVIDIA NIM, which skip straight to the JSON path.

    Checked on 2026-10-06: guided-JSON decoding on `nvidia/nemotron-3-super-120b-a12b`
    returned schema-valid JSON with fields silently set to null that the "reply with JSON"
    path filled correctly, and faster. Valid-but-empty output is the worst failure mode,
    because nothing downstream raises, so NVIDIA-served models do not use native mode.
    """
    if type(llm).__name__ == "ChatNVIDIA":
        return False
    return "nvidia.com" not in str(getattr(llm, "openai_api_base", "") or "")


def structured_call(
    llm: BaseChatModel,
    schema: type[SchemaT],
    messages: Sequence[BaseMessage],
    *,
    use_native: bool | None = None,
) -> SchemaT:
    """Get output that validates against `schema`, with two fallbacks.

    1. Native structured output (`with_structured_output`), when the model supports it.
       Skipped when `use_native` is False, and by default for NVIDIA-served models (see
       `native_structured_output_ok`).
    2. Plain call asking for JSON matching the schema, parsed with pydantic.
    3. One repair retry that shows the model its invalid reply and the validation error.
    """
    if use_native is None:
        use_native = native_structured_output_ok(llm)
    messages = list(messages)
    key = _model_key(llm)
    if use_native and _native_enabled(key):
        try:
            result = llm.with_structured_output(schema).invoke(messages)
            if isinstance(result, schema):
                return result
            if isinstance(result, dict):
                return schema.model_validate(result)
            _record_native_failure(key)
        except Exception:  # noqa: BLE001 - any failure falls through to the JSON path
            _record_native_failure(key)

    json_messages = [*messages, HumanMessage(content=_json_instruction(schema))]
    reply = chat_call(llm, json_messages)
    try:
        return _parse(schema, reply)
    except (ValueError, ValidationError) as first_error:
        repair = [
            *json_messages,
            AIMessage(content=reply),
            HumanMessage(
                content=(
                    "Your previous reply could not be used. Error:\n"
                    f"{str(first_error)[:1500]}\n"
                    "Reply again with only the corrected JSON object."
                )
            ),
        ]
        reply = chat_call(llm, repair)
        try:
            return _parse(schema, reply)
        except (ValueError, ValidationError) as second_error:
            raise StructuredOutputError(
                f"{schema.__name__}: model output did not validate after one repair attempt: "
                f"{str(second_error)[:500]}"
            ) from second_error


def system_and_user(system: str, user: str) -> list[BaseMessage]:
    """Small helper for the common two-message prompt."""
    return [SystemMessage(content=system), HumanMessage(content=user)]
