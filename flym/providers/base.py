"""
flym.providers.base
--------------------
Abstract contracts for embedding and language model providers.

Why Protocol instead of ABC?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Python's `typing.Protocol` enables *structural subtyping* — a class satisfies
the contract if it has the right methods and properties, without needing to
explicitly inherit from the Protocol.  This is sometimes called "duck typing
with type safety".

    class MyCustomEmbedder:            # ← no 'extends EmbeddingProvider'
        dimensions = 384
        def embed(self, texts): ...    # ← just needs the right signature

    p: EmbeddingProvider = MyCustomEmbedder()  # type-checker: ✓

Contrast with ABC: with ABCs the class *must* inherit from the base and call
super().__init__().  Protocols work across libraries and let you wrap
third-party objects without subclassing.

The `runtime_checkable` decorator allows `isinstance(obj, EmbeddingProvider)`.
Without it, Protocol types can only be checked at static-analysis time.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """
    Contract for a text-embedding provider.

    Implementors must expose:
        dimensions : int     — length of every returned vector
        embed(texts)         — convert strings to float vectors

    The embed() method accepts a *batch* of texts.  Batching is important
    because embedding APIs charge per call (or have per-request overhead),
    so embedding 64 texts in one call is much cheaper than 64 separate calls.
    """

    #: Number of floats in each returned vector.
    #: Must be declared as a class attribute so Protocol can check it.
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts.

        Parameters
        ----------
        texts : list of strings to embed (may be empty)

        Returns
        -------
        list of float vectors, one per input text, each of length `dimensions`

        Raises
        ------
        httpx.HTTPError  — if the underlying HTTP request fails
        ValueError       — if the response shape is unexpected
        """
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """
    Contract for a text-generation (LLM) provider.

    Implementors must expose:
        model : str         — model identifier used in API calls
        generate(prompt)    — generate a completion string

    Kept intentionally minimal: we only need single-turn generation for
    query expansion and HyDE.  Chat history, streaming, and tool-use are
    out of scope for flym.
    """

    #: Model name passed to the underlying API (e.g. "llama3").
    model: str

    def generate(self, prompt: str, *, max_tokens: int = 512) -> str:
        """
        Generate a single text completion.

        Parameters
        ----------
        prompt     : the input prompt string
        max_tokens : maximum number of tokens to generate

        Returns
        -------
        Generated text string (stripped of leading/trailing whitespace).

        Raises
        ------
        httpx.HTTPError  — if the underlying HTTP request fails
        """
        ...
