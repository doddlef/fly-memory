"""
flym.config
-----------
Loads and saves configuration from ~/.flym/config.json.

All settings have sensible defaults so the tool works out of the box
without any configuration file. When a config.json exists, values in it
override the defaults — only the keys present in the file are overridden,
the rest stay at their defaults.

Usage:
    from flym.config import load_config, save_config

    cfg = load_config()
    print(cfg.vault_path)          # Path object
    print(cfg.chunking.target_chars)

    cfg.chunking.target_chars = 2000
    save_config(cfg)
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field

# Where the config file lives. Can be overridden in tests by patching this.
CONFIG_PATH = Path.home() / ".flym" / "config.json"


# ---------------------------------------------------------------------------
# Sub-configs — one class per logical group
# ---------------------------------------------------------------------------

class EmbeddingConfig(BaseModel):
    provider: str = "ollama"        # "ollama" | "openai" (more added in Module 4)
    model: str = "nomic-embed-text"
    dimensions: int = 768           # must match the model's output size


class LLMConfig(BaseModel):
    provider: str = "ollama"        # "ollama" | "openai" | "anthropic"
    model: str = "qwen3.5:9b"


class ChunkingConfig(BaseModel):
    target_chars: int = 1500        # ~500 tokens at 3 chars/token average
    overlap_chars: int = 225        # 15% of target — overlap between adjacent chunks
    min_chars: int = 100            # chunks shorter than this are discarded


class SearchConfig(BaseModel):
    default_count: int = 5          # results returned to the user by default

    # BM25 early-return: if the top result is clearly dominant, skip the
    # expensive vector + reranking stages entirely.
    bm25_threshold: float = 0.85    # normalised top score must exceed this
    bm25_gap: float = 0.15          # gap between top and second must exceed this

    # Cache TTL and probabilistic eviction (see flym/cache.py, Module 11)
    cache_ttl_hours: int = 24
    cache_evict_probability: float = 0.01   # 1% of queries trigger cleanup
    cache_evict_limit: int = 100            # max rows deleted per cleanup pass


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class Config(BaseModel):
    vault_path: Path = Field(default=Path.home() / ".flym" / "vault")
    db_path: Path = Field(default=Path.home() / ".flym" / "flym.db")

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

def load_config() -> Config:
    """
    Return the active Config.

    If ~/.flym/config.json exists, its values override the defaults.
    Missing keys fall back to defaults — you never need a complete config file.
    """
    if CONFIG_PATH.exists():
        data = json.loads(CONFIG_PATH.read_text())
        return Config.model_validate(data)
    return Config()


def save_config(config: Config) -> None:
    """Persist config to ~/.flym/config.json, creating the directory if needed."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(config.model_dump_json(indent=2))
