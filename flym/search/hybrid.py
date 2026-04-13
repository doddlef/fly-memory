"""
flym.search.hybrid
-------------------
Hybrid search: fuse BM25 and vector results using Reciprocal Rank Fusion.

Why not just sum the scores?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
BM25 scores are normalised to [0,1] in our pipeline; vector scores are L2
distances (lower = better).  Even after normalising both to the same range,
their magnitudes encode different things:
  - BM25 captures lexical overlap (exact and stemmed word matches)
  - Vector captures semantic similarity (meaning, not words)

A document that appears in position 3 in BM25 and position 1 in vector
probably deserves a high combined rank — but their raw scores may be very
different numbers that don't add up meaningfully.

RRF (Reciprocal Rank Fusion) solves this by ignoring scores entirely and
only caring about *rank positions*:

    RRF(d) = Σ  1 / (k + rank_i(d))

where the sum is over each ranked list that contains document d, and k=60
is a smoothing constant that reduces the impact of top-ranked documents.

    k=60 origin: introduced in the original RRF paper (Cormack et al., 2009).
    It was tuned on TREC collections and has proven robust across many tasks.

Example with k=60:
    rank 1  → 1/61  ≈ 0.0164
    rank 10 → 1/70  ≈ 0.0143
    rank 50 → 1/110 ≈ 0.0091

A document ranked 1st in BM25 and 1st in vector scores:
    0.0164 + 0.0164 = 0.0328  (top of hybrid list)

A document ranked 1st in only one list and absent from the other scores:
    0.0164  (present in one list only)

This naturally promotes documents that both systems agree on.

The hybrid_search() function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Embed the query with the provider.
2. Run BM25 search (with early-return check disabled — we always want both).
3. Run vector KNN search.
4. Apply RRF across both ranked lists.
5. Fetch full metadata for the top results and return.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, replace

from flym.config import Config
from flym.providers.base import EmbeddingProvider
from flym.search.bm25 import BM25Result, bm25_search
from flym.search.vector import vector_search

# RRF smoothing constant — higher k reduces the impact of rank-1 documents.
_RRF_K = 60

# How many candidates to gather from each source before fusion.
# We over-fetch so RRF has enough overlap to work with.
_CANDIDATE_MULTIPLIER = 5


@dataclass
class HybridResult:
    """One result from the hybrid (RRF-fused) search."""
    chunk_id:     int
    doc_id:       int
    title:        str
    collection:   str
    section_path: str | None
    chunk_type:   str
    rrf_score:    float   # higher = better
    excerpt:      str
    bm25_rank:    int | None   # rank in BM25 list (1-based), None if absent
    vector_rank:  int | None   # rank in vector list (1-based), None if absent


def hybrid_search(
    query: str,
    conn: sqlite3.Connection,
    provider: EmbeddingProvider,
    config: Config,
    count: int,
    collection: str | None = None,
    bm25_query: str | None = None,
    query_vec: list[float] | None = None,
) -> list[HybridResult]:
    """
    Run hybrid BM25 + vector search and return RRF-fused results.

    Parameters
    ----------
    query      : raw user query string (used as fallback for BM25 + embedding)
    conn       : open database connection (virtual tables must be loaded)
    provider   : embedding provider (used to embed the query if query_vec absent)
    config     : active Config (chunking, search thresholds)
    count      : number of final results to return
    collection : if given, restrict to this collection only
    bm25_query : optional pre-expanded FTS5 query string (from rephrase expansion)
    query_vec  : optional pre-computed query embedding (from HyDE expansion)

    Returns
    -------
    list of HybridResult, ordered by RRF score descending, len <= count
    """
    candidates = count * _CANDIDATE_MULTIPLIER

    # --- BM25 candidates (ignore early_return — we always fuse here) ---------
    # Use the expanded query for BM25 if provided (rephrase path).
    bm25_results, _ = bm25_search(
        bm25_query or query, conn, config.search,
        count=candidates, collection=collection,
    )

    # --- Vector candidates ---------------------------------------------------
    # Use the pre-computed vector if provided (HyDE path), else embed the query.
    if query_vec is None:
        query_vec = provider.embed([query])[0]
    vector_results = vector_search(
        query_vec, conn, count=candidates, collection=collection
    )

    # --- Build rank maps: chunk_id → 1-based rank ----------------------------
    bm25_ranks  = {r.chunk_id: i + 1 for i, r in enumerate(bm25_results)}
    vector_ranks = {r.chunk_id: i + 1 for i, r in enumerate(vector_results)}

    # --- Collect all unique chunk IDs from both lists ------------------------
    all_ids = dict.fromkeys(
        [r.chunk_id for r in bm25_results] +
        [r.chunk_id for r in vector_results]
    )

    # --- Compute RRF score for each candidate --------------------------------
    rrf_scores: dict[int, float] = {}
    for chunk_id in all_ids:
        score = 0.0
        if chunk_id in bm25_ranks:
            score += 1.0 / (_RRF_K + bm25_ranks[chunk_id])
        if chunk_id in vector_ranks:
            score += 1.0 / (_RRF_K + vector_ranks[chunk_id])
        rrf_scores[chunk_id] = score

    # Sort by RRF score descending, take top `count`.
    ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
    top_ids = ranked_ids[:count]

    # --- Build result objects, using BM25 metadata (it carries text fields) --
    # Build lookup from chunk_id → BM25Result for fast access.
    bm25_by_id   = {r.chunk_id: r for r in bm25_results}
    vector_by_id = {r.chunk_id: r for r in vector_results}

    results: list[HybridResult] = []
    for chunk_id in top_ids:
        # Prefer BM25Result for metadata (has excerpt already).
        # Fall back to VectorResult if the chunk is only in the vector list.
        if chunk_id in bm25_by_id:
            r = bm25_by_id[chunk_id]
            excerpt      = r.excerpt
            title        = r.title
            collection_  = r.collection
            section_path = r.section_path
            chunk_type   = r.chunk_type
            doc_id       = r.doc_id
        else:
            r = vector_by_id[chunk_id]
            excerpt      = r.excerpt
            title        = r.title
            collection_  = r.collection
            section_path = r.section_path
            chunk_type   = r.chunk_type
            doc_id       = r.doc_id

        results.append(HybridResult(
            chunk_id     = chunk_id,
            doc_id       = doc_id,
            title        = title,
            collection   = collection_,
            section_path = section_path,
            chunk_type   = chunk_type,
            rrf_score    = rrf_scores[chunk_id],
            excerpt      = excerpt,
            bm25_rank    = bm25_ranks.get(chunk_id),
            vector_rank  = vector_ranks.get(chunk_id),
        ))

    return results
