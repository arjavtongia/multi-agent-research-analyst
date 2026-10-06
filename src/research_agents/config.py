"""Runtime configuration, read from environment variables and an optional `.env` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["nvidia", "azure", "openai", "anthropic"]
SearchProvider = Literal["auto", "duckduckgo", "tavily"]


class Settings(BaseSettings):
    """All tunables in one place. Every field maps to an upper-case env var of the same name."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        populate_by_name=True,
    )

    # --- LLM provider -------------------------------------------------------------------------
    llm_provider: LLMProvider = "nvidia"
    llm_model: str | None = None
    llm_fast_model: str | None = None
    llm_temperature: float = 0.2
    # Long reasoning outputs on NVIDIA's free tier can take minutes; see chat_call retries.
    llm_timeout_s: float = 300.0
    # Reasoning models spend part of this budget on hidden reasoning before the answer.
    llm_max_output_tokens: int = 16384

    nvidia_api_key: SecretStr | None = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    azure_openai_api_key: SecretStr | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_version: str | None = None
    azure_openai_chat_deployment: str | None = None
    azure_openai_fast_deployment: str | None = None

    # --- Web search and fetching ---------------------------------------------------------------
    search_provider: SearchProvider = "auto"
    tavily_api_key: SecretStr | None = None
    ddgs_backend: str = "yahoo,bing,duckduckgo"
    search_results_per_query: int = Field(default=8, ge=1, le=25)
    search_max_retries: int = Field(default=3, ge=0, le=10)
    search_max_concurrency: int = Field(default=2, ge=1, le=16)
    blocked_domains: list[str] = Field(
        default_factory=lambda: [
            "facebook.com",
            "instagram.com",
            "tiktok.com",
            "pinterest.com",
            "x.com",
            "twitter.com",
            "quora.com",
        ]
    )
    fetch_timeout_s: float = 15.0
    fetch_max_bytes: int = 2_000_000
    user_agent: str = (
        "research-agents/0.1 (+https://github.com/arjavtongia/multi-agent-research-analyst)"
    )

    # --- Research limits and guardrails --------------------------------------------------------
    min_sub_questions: int = Field(default=3, ge=1)
    max_sub_questions: int = Field(default=6, ge=1, le=10)
    max_sources_per_question: int = Field(default=3, ge=1, le=10)
    max_page_chars: int = Field(default=6000, ge=500)
    min_page_chars: int = Field(default=300, ge=0)
    max_notes_per_source: int = Field(default=5, ge=1)
    max_evidence_for_writer: int = Field(default=60, ge=5)
    max_revisions: int = Field(default=2, ge=0, le=10)
    max_gap_research_rounds: int = Field(default=1, ge=0, le=5)
    max_gap_queries: int = Field(default=2, ge=0, le=6)
    factcheck_batch_size: int = Field(default=12, ge=1)
    token_budget: int = Field(default=300_000, ge=1)

    # --- Storage -------------------------------------------------------------------------------
    runs_dir: Path = Path("runs")
    checkpoint_db: str = "checkpoints.sqlite"

    @property
    def checkpoint_path(self) -> Path:
        """SQLite file used by the LangGraph checkpointer."""
        return self.runs_dir / self.checkpoint_db


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings for the process. Tests build `Settings(...)` directly instead."""
    return Settings()
