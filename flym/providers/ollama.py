"""
flym.providers.ollama
----------------------
Concrete EmbeddingProvider and LLMProvider implementations backed by the
local Ollama HTTP API.

Prerequisites
~~~~~~~~~~~~~
    brew install ollama
    ollama serve                        # starts the local API on :11434
    ollama pull nomic-embed-text        # 274MB embedding model
    ollama pull llama3                  # or any chat model

API used
~~~~~~~~
    POST /api/embeddings                # single-text embedding
    POST /api/generate                  # text completion (non-streaming)

Both endpoints accept JSON and return JSON.  We use `httpx` (a modern
replacement for `requests`) because it has cleaner timeout handling and
will be needed for async calls in later modules.

Why not the Ollama Python SDK?
    The official SDK exists but it's an extra dependency for something that
    only needs two simple POST requests.  Rolling our own keeps the dependency
    list short and makes the HTTP contract visible in the code.

Timeout strategy
~~~~~~~~~~~~~~~~
Embedding calls are fast (< 1 s per text on CPU).  Generation can take tens
of seconds for long outputs.  We therefore use separate timeout values rather
than one global setting.
"""

from __future__ import annotations

import httpx


# Default Ollama server URL.  Override via OllamaEmbedding(base_url=...).
_DEFAULT_BASE_URL = "http://localhost:11434"

# Timeouts in seconds.
_EMBED_TIMEOUT   = 30.0   # per-text embedding request
_GENERATE_TIMEOUT = 120.0  # text generation can be slow on CPU


class OllamaEmbedding:
    """
    Embed texts via the local Ollama `/api/embeddings` endpoint.

    The endpoint only accepts one text per call, so `embed()` sends one
    request per text and collects the results.  In Module 5 (indexing) we
    will batch at a higher level and call embed() with reasonable batch sizes.

    Example
    -------
    >>> p = OllamaEmbedding(model="nomic-embed-text")
    >>> vecs = p.embed(["hello world", "how do neural networks learn?"])
    >>> len(vecs), len(vecs[0])
    (2, 768)
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = _DEFAULT_BASE_URL,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

        # `httpx.Client` is a connection pool — reuse it across calls.
        # `follow_redirects=True` handles any reverse-proxy setup.
        self._client = httpx.Client(
            base_url=self.base_url,
            follow_redirects=True,
        )

        # We discover dimensions lazily on the first embed() call.
        self._dimensions: int | None = None

    # ------------------------------------------------------------------
    # EmbeddingProvider contract
    # ------------------------------------------------------------------

    @property
    def dimensions(self) -> int:
        """
        Number of floats in each embedding vector.

        Raises RuntimeError if embed() has never been called (dimensions
        are not known until we see a response from the model).
        """
        if self._dimensions is None:
            raise RuntimeError(
                "dimensions is not known until embed() has been called at least once"
            )
        return self._dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts.  Sends one HTTP request per text.

        Parameters
        ----------
        texts : list of strings (may be empty — returns [])

        Returns
        -------
        list of float vectors, same length as `texts`

        Raises
        ------
        httpx.HTTPStatusError  — non-2xx response from Ollama
        httpx.TimeoutException — request exceeded _EMBED_TIMEOUT seconds
        """
        if not texts:
            return []

        vectors: list[list[float]] = []

        for text in texts:
            response = self._client.post(
                "/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=_EMBED_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()

            vec: list[float] = data["embedding"]
            vectors.append(vec)

            # Record dimensions from the first successful response.
            if self._dimensions is None:
                self._dimensions = len(vec)

        return vectors

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    # Support `with OllamaEmbedding() as p:` usage.
    def __enter__(self) -> "OllamaEmbedding":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"OllamaEmbedding(model={self.model!r}, base_url={self.base_url!r})"


class OllamaLLM:
    """
    Generate text via the local Ollama `/api/generate` endpoint.

    Uses non-streaming mode (`stream=False`) so the full response arrives
    in one JSON object.  Streaming would be better for interactive use, but
    flym only needs complete strings for query expansion.

    Example
    -------
    >>> llm = OllamaLLM(model="llama3")
    >>> print(llm.generate("List three synonyms for 'fast':"))
    quick, rapid, swift
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = _DEFAULT_BASE_URL,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

        self._client = httpx.Client(
            base_url=self.base_url,
            follow_redirects=True,
        )

    # ------------------------------------------------------------------
    # LLMProvider contract
    # ------------------------------------------------------------------

    def generate(self, prompt: str, *, max_tokens: int = 512) -> str:
        """
        Generate a single text completion.

        Parameters
        ----------
        prompt     : the full prompt string
        max_tokens : passed as `options.num_predict` to Ollama

        Returns
        -------
        Stripped completion string.

        Raises
        ------
        httpx.HTTPStatusError  — non-2xx response from Ollama
        httpx.TimeoutException — request exceeded _GENERATE_TIMEOUT seconds
        """
        response = self._client.post(
            "/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens},
            },
            timeout=_GENERATE_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return data["response"].strip()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> "OllamaLLM":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"OllamaLLM(model={self.model!r}, base_url={self.base_url!r})"
