"""Runtime configuration, read from environment variables with sensible defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    # Generation
    # Empty means "use the provider's default model" (see groundwork.llm.PROVIDERS).
    model: str = field(default_factory=lambda: _env("GROUNDWORK_MODEL", ""))
    judge_model: str = field(default_factory=lambda: _env("GROUNDWORK_JUDGE_MODEL", ""))
    max_tokens: int = field(default_factory=lambda: int(_env("GROUNDWORK_MAX_TOKENS", "1024")))

    # Retrieval
    embedder: str = field(default_factory=lambda: _env("GROUNDWORK_EMBEDDER", "hashing"))
    top_k: int = field(default_factory=lambda: int(_env("GROUNDWORK_TOP_K", "6")))
    retrieval_mode: str = field(default_factory=lambda: _env("GROUNDWORK_RETRIEVAL", "hybrid"))

    # Storage
    index_dir: str = field(default_factory=lambda: _env("GROUNDWORK_INDEX_DIR", ".index"))

    # Chunking
    chunk_chars: int = field(default_factory=lambda: int(_env("GROUNDWORK_CHUNK_CHARS", "900")))


def get_settings() -> Settings:
    return Settings()
