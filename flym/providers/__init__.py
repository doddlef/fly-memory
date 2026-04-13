"""
flym.providers
--------------
Provider abstractions (Protocols) and concrete implementations.

Currently implemented:
    OllamaEmbedding  — local embedding via Ollama HTTP API
    OllamaLLM        — local text generation via Ollama HTTP API

Usage:
    from flym.providers.ollama import OllamaEmbedding, OllamaLLM
"""

from flym.providers.base import EmbeddingProvider, LLMProvider
from flym.providers.ollama import OllamaEmbedding, OllamaLLM

__all__ = ["EmbeddingProvider", "LLMProvider", "OllamaEmbedding", "OllamaLLM"]
