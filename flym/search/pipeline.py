"""
flym.search.pipeline
---------------------
Assembles all search stages into one entry point: run_search().

Stage order
~~~~~~~~~~~
    1. BM25 (fast path)          — always runs; may short-circuit here
    2. Query expansion           — rephrase or HyDE via LLM
    3. Hybrid search (BM25+vec)  — RRF fusion of both ranked lists
    4. Cross-encoder reranking   — rescore top candidates accurately
    5. Context expansion         — attach neighbouring chunks for display

Top-K ladder
~~~~~~~~~~~~
Each stage works on a progressively smaller candidate set:

    hybrid stage  : 5 × count candidates  (wide net, cheap scoring)
    rerank stage  : 3 × count candidates  (accurate scoring, small set)
    final output  : count results

The ladder ensures reranking never processes more than ~25 chunks for a
default count of 5, keeping latency acceptable on CPU.

Context expansion
~~~~~~~~~~~~~~~~~
After reranking, for each result we fetch the immediately adjacent chunks
(seq - 1 and seq + 1) from the same document and concatenate their text as
a "context window".  This gives the user (and downstream LLMs) more
surrounding text without changing the ranking.

FinalResult
~~~~~~~~~~~
The unified output type for the full pipeline.  All prior intermediate types
(BM25Result, HybridResult) are internal to their respective modules; callers
of run_search() only see FinalResult.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass

from flym.cache import cache_get, cache_set
from flym.config import Config
from flym.providers.base import EmbeddingProvider, LLMProvider
from flym.search.bm25 import bm25_search
from flym.search.expansion import classify, hyde, rephrase
from flym.search.hybrid import hybrid_search
from flym.search.rerank import rerank


@dataclass
class FinalResult:
    """Unified result type returned by run_search()."""
    chunk_id:     int
    doc_id:       int
    title:        str
    collection:   str
    section_path: str | None
    chunk_type:   str
    score:        float         # rerank logit, or rrf_score if reranking skipped
    excerpt:      str           # the chunk text
    context:      str | None    # excerpt + neighbouring chunks (may be None)
    bm25_rank:    int | None
    vector_rank:  int | None


def run_search(
    query: str,
    conn: sqlite3.Connection,
    embed_provider: EmbeddingProvider,
    llm_provider: LLMProvider | None,
    config: Config,
    count: int,
    collection: str | None = None,
    expand: bool = True,
    rerank_results: bool = True,
) -> list[FinalResult]:
    """
    Run the full search pipeline and return FinalResult objects.

    Parameters
    ----------
    query          : raw user query
    conn           : open db connection (virtual tables loaded)
    embed_provider : embedding provider for vector search
    llm_provider   : LLM for query expansion; None or expand=False skips it
    config         : active Config
    count          : number of results to return
    collection     : optional collection filter
    expand         : if False, skip query expansion even if llm_provider given
    rerank_results : if False, skip cross-encoder reranking

    Returns
    -------
    list of FinalResult, ordered best-first, length <= count
    """
    # -------------------------------------------------------------------------
    # Cache lookup — skip the entire pipeline on a hit
    # -------------------------------------------------------------------------
    cache_key = _search_cache_key(query, collection, count, expand, rerank_results)
    cached    = cache_get(cache_key, "search", conn)
    if cached is not None:
        return [FinalResult(**r) for r in cached]

    # -------------------------------------------------------------------------
    # Stage 1: BM25 fast path
    # -------------------------------------------------------------------------
    bm25_results, early_return = bm25_search(
        query, conn, config.search, count=count, collection=collection,
    )

    if early_return:
        return [
            FinalResult(
                chunk_id     = r.chunk_id,
                doc_id       = r.doc_id,
                title        = r.title,
                collection   = r.collection,
                section_path = r.section_path,
                chunk_type   = r.chunk_type,
                score        = r.score,
                excerpt      = r.excerpt,
                context      = None,
                bm25_rank    = i + 1,
                vector_rank  = None,
            )
            for i, r in enumerate(bm25_results)
        ]

    # -------------------------------------------------------------------------
    # Stage 2: Query expansion
    # -------------------------------------------------------------------------
    bm25_query: str | None      = None
    query_vec:  list[float] | None = None

    if expand and llm_provider is not None:
        strategy = classify(query)
        if strategy == "hyde":
            hypo_doc  = hyde(query, llm_provider, conn)
            query_vec = embed_provider.embed([hypo_doc])[0]
        else:
            bm25_query = rephrase(query, llm_provider, conn)

    # -------------------------------------------------------------------------
    # Stage 3: Hybrid search — 5× candidates for the reranker
    # -------------------------------------------------------------------------
    hybrid_candidates = count * 5
    hybrid_results = hybrid_search(
        query, conn, embed_provider, config,
        count=hybrid_candidates, collection=collection,
        bm25_query=bm25_query,
        query_vec=query_vec,
    )

    if not hybrid_results:
        return []

    # -------------------------------------------------------------------------
    # Stage 4: Cross-encoder reranking — trim to 3× before reranking
    # -------------------------------------------------------------------------
    rerank_candidates = min(count * 3, len(hybrid_results))
    top_hybrid = hybrid_results[:rerank_candidates]

    if rerank_results:
        excerpts = [r.excerpt for r in top_hybrid]
        ranked_indices = rerank(query, excerpts, count)
        final_hybrid = [top_hybrid[i] for i in ranked_indices]
        # Score: use index position as a proxy (rerank returns sorted indices)
        scores = {top_hybrid[i].chunk_id: count - rank
                  for rank, i in enumerate(ranked_indices)}
    else:
        final_hybrid = top_hybrid[:count]
        scores = {r.chunk_id: r.rrf_score for r in final_hybrid}

    # -------------------------------------------------------------------------
    # Stage 5: Context expansion — fetch neighbouring chunks
    # -------------------------------------------------------------------------
    context_map = _fetch_context(
        [(r.chunk_id, r.bm25_rank) for r in final_hybrid],   # type: ignore[arg-type]
        conn,
    )

    results = [
        FinalResult(
            chunk_id     = r.chunk_id,
            doc_id       = r.doc_id,
            title        = r.title,
            collection   = r.collection,
            section_path = r.section_path,
            chunk_type   = r.chunk_type,
            score        = scores.get(r.chunk_id, r.rrf_score),
            excerpt      = r.excerpt,
            context      = context_map.get(r.chunk_id),
            bm25_rank    = r.bm25_rank,
            vector_rank  = r.vector_rank,
        )
        for r in final_hybrid
    ]

    # Store in cache with TTL.
    cache_set(
        cache_key, [asdict(r) for r in results],
        "search", conn,
        ttl_hours=config.search.cache_ttl_hours,
    )
    return results


# ---------------------------------------------------------------------------
# Context expansion helper
# ---------------------------------------------------------------------------

def _search_cache_key(
    query: str,
    collection: str | None,
    count: int,
    expand: bool,
    rerank_results: bool,
) -> str:
    """Canonical string that uniquely identifies a search request."""
    return json.dumps(
        [query, collection, count, expand, rerank_results],
        ensure_ascii=False,
    )


def _fetch_context(
    chunk_ids_and_ranks: list[tuple[int, int | None]],
    conn: sqlite3.Connection,
) -> dict[int, str]:
    """
    For each chunk_id, fetch the text of the previous and next chunk
    (by seq within the same content_hash) and return a context string.

    Returns a dict: chunk_id → context text (prev + chunk + next).
    Chunks without neighbours simply have a shorter context.
    """
    if not chunk_ids_and_ranks:
        return {}

    ids = [cid for cid, _ in chunk_ids_and_ranks]
    placeholders = ",".join("?" * len(ids))

    # Fetch the target chunks' seq + content_hash in one query.
    rows = conn.execute(
        f"SELECT id, content_hash, seq FROM chunks WHERE id IN ({placeholders})",
        ids,
    ).fetchall()

    context: dict[int, str] = {}

    for row in rows:
        cid   = row["id"]
        chash = row["content_hash"]
        seq   = row["seq"]

        # Fetch prev, current, and next chunk texts in one query.
        neighbours = conn.execute(
            """
            SELECT seq, substr(ct.doc, c.pos + 1, c.len) AS text
            FROM chunks c
            JOIN content ct ON ct.hash = c.content_hash
            WHERE c.content_hash = ?
              AND c.seq IN (?, ?, ?)
            ORDER BY c.seq
            """,
            (chash, seq - 1, seq, seq + 1),
        ).fetchall()

        context[cid] = "\n\n".join(r["text"] for r in neighbours)

    return context
