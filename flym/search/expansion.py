"""
flym.search.expansion
----------------------
LLM-powered query expansion: rephrase and HyDE.

Two strategies
~~~~~~~~~~~~~~

**Rephrase** (keyword queries)
    Generate synonym-rich variants of the query and fold them into the BM25
    search as OR terms.  This bridges vocabulary gaps where the user says
    "fast" but the document says "efficient".

    Input:  "jwt token validation"
    Output: "jwt token validation OR JSON Web Token verify OR bearer token check"

**HyDE** — Hypothetical Document Embeddings (question queries)
    Instead of embedding the question, ask the LLM to write a short paragraph
    that *answers* the question, then embed that paragraph.

    Why this helps: a question and its answer live in different semantic
    spaces.  "How does backpropagation work?" is far from "Backpropagation
    computes gradients by applying the chain rule…" in embedding space, even
    though the answer is exactly what we want to retrieve.  By generating and
    embedding a plausible answer, we search in the *answer* space where the
    real documents live.

    The generated paragraph doesn't need to be factually correct — its
    embedding just needs to be close to real answers in the corpus.

Query classification
~~~~~~~~~~~~~~~~~~~~~
    - Ends with "?"          → HyDE
    - Starts with how/what/why/explain/describe/… → HyDE
    - Otherwise              → rephrase

LLM cache
~~~~~~~~~
LLM calls are deterministic for a given prompt (temperature=0) and
expensive.  We cache every prompt → response in the `cache` table with
`type='llm'` and `expires_at=NULL` (no TTL — the same prompt always gives
the same useful expansion).  Cache hits skip the Ollama round-trip entirely.
"""

from __future__ import annotations

import sqlite3

from flym.cache import cache_get, cache_set
from flym.providers.base import LLMProvider


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

_HYDE_STARTERS = {
    "how", "what", "why", "when", "where", "who",
    "explain", "describe", "define", "summarise", "summarize",
}


def classify(query: str) -> str:
    """
    Return "hyde" or "rephrase" based on query shape.

    HyDE is chosen when the query is phrased as a question or starts with
    an interrogative/explanatory word.  Everything else gets rephrased.
    """
    q = query.strip()
    if q.endswith("?"):
        return "hyde"
    first_word = q.split()[0].lower() if q else ""
    if first_word in _HYDE_STARTERS:
        return "hyde"
    return "rephrase"


# ---------------------------------------------------------------------------
# Rephrase
# ---------------------------------------------------------------------------

_REPHRASE_PROMPT = """\
Generate 4 short search query variations for the following query.
Each variation should use different words or synonyms but mean the same thing.
Return only the queries, one per line, with no numbering or punctuation.

Query: {query}
"""


def rephrase(query: str, llm: LLMProvider, conn: sqlite3.Connection) -> str:
    """
    Return an FTS5 OR-query combining the original and rephrased variants.

    Example
    -------
    Input:  "jwt token validation"
    Output: '"jwt token validation"* OR "JSON Web Token verify"* OR ...'
    """
    prompt = _REPHRASE_PROMPT.format(query=query)
    raw = _llm_cached(prompt, llm, conn)

    # Parse lines, drop empty ones, take up to 4 variants.
    variants = [line.strip() for line in raw.splitlines() if line.strip()][:4]

    # Build FTS5 OR expression: each term quoted + prefix wildcard.
    terms = [f'"{query}"*'] + [f'"{v}"*' for v in variants]
    return " OR ".join(terms)


# ---------------------------------------------------------------------------
# HyDE
# ---------------------------------------------------------------------------

_HYDE_PROMPT = """\
Write a short factual paragraph (3-5 sentences) that directly answers the
following question. Be specific and use domain terminology.

Question: {query}
"""


def hyde(query: str, llm: LLMProvider, conn: sqlite3.Connection) -> str:
    """
    Generate a hypothetical answer document for the query.

    Returns the generated paragraph as a string.  The caller should embed
    this paragraph and use the resulting vector for KNN search.
    """
    prompt = _HYDE_PROMPT.format(query=query)
    return _llm_cached(prompt, llm, conn)


# ---------------------------------------------------------------------------
# LLM cache
# ---------------------------------------------------------------------------

def _llm_cached(prompt: str, llm: LLMProvider, conn: sqlite3.Connection) -> str:
    """
    Return llm.generate(prompt), using the cache table as a persistent store.

    LLM entries are permanent (no TTL) because the same prompt always
    produces the same useful expansion.  Cache hit → no Ollama round-trip.
    """
    cached = cache_get(prompt, "llm", conn)
    if cached is not None:
        return str(cached)

    result = llm.generate(prompt)
    cache_set(prompt, result, "llm", conn, ttl_hours=None)
    return result
