"""
flym.providers.ollama
----------------------
Concrete EmbeddingProvider and LLMProvider implementations backed by the
local Ollama Python SDK.

Prerequisites
~~~~~~~~~~~~~
    brew install ollama
    ollama serve                        # starts the local API on :11434
    ollama pull nomic-embed-text        # 274MB embedding model
    ollama pull llama3                  # or any chat model

Under the hood the SDK calls the same HTTP endpoints we used to call manually:
    POST /api/embed       → ollama.embed()
    POST /api/generate    → ollama.generate()

Reading the SDK source (github.com/ollama/ollama-python) shows exactly what
those calls look like if you want to understand the HTTP layer.
"""

from __future__ import annotations

import ollama


class OllamaEmbedding:
    """
    Embed texts via the local Ollama SDK.

    Passes the full list of texts in one call — the SDK sends them to
    /api/embed in a single request, so the model only loads once.

    Example
    -------
    >>> p = OllamaEmbedding(model="nomic-embed-text")
    >>> vecs = p.embed(["JWT authentication", "backpropagation"])
    >>> len(vecs), len(vecs[0])
    (2, 768)
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str | None = None,
    ) -> None:
        self.model = model
        # ollama.Client lets you point at a non-default host (e.g. a remote
        # Ollama server).  None falls back to http://localhost:11434.
        self._client = ollama.Client(host=host)
        self._dimensions: int | None = None

    # ------------------------------------------------------------------
    # EmbeddingProvider contract
    # ------------------------------------------------------------------

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            raise RuntimeError(
                "dimensions is not known until embed() has been called at least once"
            )
        return self._dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts in a single SDK call.

        Parameters
        ----------
        texts : list of strings (may be empty — returns [])

        Returns
        -------
        list of float vectors, same length as `texts`
        """
        if not texts:
            return []

        response = self._client.embed(model=self.model, input=texts)
        vectors: list[list[float]] = [list(v) for v in response.embeddings]

        if self._dimensions is None:
            self._dimensions = len(vectors[0])

        return vectors

    def __repr__(self) -> str:
        return f"OllamaEmbedding(model={self.model!r})"


class OllamaLLM:
    """
    Generate text via the local Ollama SDK.

    Example
    -------
    >>> llm = OllamaLLM(model="llama3")
    >>> print(llm.generate("List three synonyms for 'fast':"))
    quick, rapid, swift
    """

    def __init__(
        self,
        model: str = "qwen3.5:9b",
        host: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self._client = ollama.Client(host=host)

    # ------------------------------------------------------------------
    # LLMProvider contract
    # ------------------------------------------------------------------

    def generate(self, prompt: str, *, max_tokens: int = 512) -> str:
        """
        Generate a single text completion.

        Parameters
        ----------
        prompt     : the full prompt string
        max_tokens : maximum tokens to generate (passed as num_predict)

        Returns
        -------
        Stripped completion string.
        """
        response = self._client.generate(
            model=self.model,
            prompt=prompt,
            think=False,
            options={"num_predict": max_tokens},
        )
        return response.response.strip()

    def __repr__(self) -> str:
        return f"OllamaLLM(model={self.model!r})"
