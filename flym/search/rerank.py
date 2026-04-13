"""
flym.search.rerank
-------------------
Cross-encoder reranking via sentence-transformers.

Bi-encoder vs. cross-encoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The embedding model used in vector search is a *bi-encoder*: it encodes the
query and each document chunk independently into vectors, then compares them
with a distance function.  This is fast (embeddings are pre-computed and
stored), but the query and chunk never "see" each other — the model can't
reason about their interaction.

A *cross-encoder* takes the (query, chunk) pair as a single input and
produces a relevance score.  Because both texts are processed together, the
model can capture subtle interactions like negation, co-reference, and
domain-specific terminology.  The result is much more accurate ranking.

The cost: a cross-encoder must be run at query time for every candidate —
it can't use pre-computed vectors.  This is why we use it as a final
re-ranking step over a small candidate set (~25 chunks) rather than over
the entire corpus.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    - ~80 MB, runs on CPU in under a second for 25 candidates
    - Trained on MS MARCO passage ranking — excellent for short Q&A style queries
    - Output: raw logit (unbounded float); higher = more relevant

Lazy loading
~~~~~~~~~~~~
The CrossEncoder is loaded on the first call to rerank() and cached in a
module-level variable.  Loading takes ~1 s (reading weights from disk);
keeping it alive avoids paying that cost on every search.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_encoder: CrossEncoder | None = None


def _get_encoder() -> CrossEncoder:
    global _encoder
    if _encoder is None:
        _encoder = CrossEncoder(_MODEL_NAME)
    return _encoder


def rerank(
    query: str,
    excerpts: list[str],
    count: int,
) -> list[int]:
    """
    Score (query, excerpt) pairs and return indices sorted best-first.

    Parameters
    ----------
    query    : the user's query string
    excerpts : list of chunk texts to score
    count    : how many indices to return (top-K after scoring)

    Returns
    -------
    list of indices into `excerpts`, sorted by relevance descending,
    length = min(count, len(excerpts))

    Example
    -------
    >>> indices = rerank("JWT validation", ["text A", "text B", "text C"], 2)
    >>> indices  # e.g. [2, 0] — chunk C is most relevant, then A
    """
    if not excerpts:
        return []

    encoder = _get_encoder()
    pairs   = [(query, ex) for ex in excerpts]
    scores  = encoder.predict(pairs)   # returns numpy array of floats

    # argsort ascending; reverse for descending relevance.
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked[:count]
