"""
flym.search.bm25
-----------------
Full-text search using SQLite's FTS5 BM25 ranking function.

How FTS5 BM25 works
~~~~~~~~~~~~~~~~~~~~
FTS5 exposes a `bm25()` function you call inside a query against a MATCH
table.  It returns a *negative* float: the more negative, the worse the
match.  A perfect match might be -5.0; a weak match might be -0.1.
Zero means "not matched at all" (but those rows aren't returned by MATCH).

    SELECT chunk_id, bm25(documents_fts) AS score
    FROM documents_fts
    WHERE chunk_text MATCH 'jwt token'
    ORDER BY score          -- ascending: most negative first = best first

Why we normalize
~~~~~~~~~~~~~~~~
Raw BM25 scores are not bounded and vary with corpus size, document length,
and query term frequency.  We can't threshold on them directly (-2.0 means
something different for a 10-doc corpus vs. a 10,000-doc corpus).

Normalization maps the raw scores to [0, 1]:

    normalised = (raw - worst) / (best - worst)

where best = min(raws) and worst = max(raws).
(Remember: scores are negative, so the *smallest* value is the *best*.)

After normalisation, top = 1.0 (best result) and bottom = 0.0 (worst).

Early return
~~~~~~~~~~~~
If the top normalised score exceeds `bm25_threshold` (default 0.85) AND the
gap between first and second result exceeds `bm25_gap` (default 0.15), the
top result is considered dominant and we skip the slower vector + reranking
stages.  The threshold values come from SearchConfig in config.py.

This optimisation is only worth having because BM25 is fast (pure SQL, no
model call).  The vector stage requires an embedding call; the reranker
requires a cross-encoder pass.  When BM25 is confident, we can skip both.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import sqlite_vec

from flym.config import SearchConfig
from flym.indexer import ensure_virtual_tables


@dataclass
class BM25Result:
    """One result row returned by bm25_search()."""
    chunk_id:     int
    doc_id:       int
    title:        str
    collection:   str
    section_path: str | None
    chunk_type:   str
    score:        float        # normalised to [0, 1]; 1.0 = best
    excerpt:      str          # the chunk text, for display


def bm25_search(
    query: str,
    conn: sqlite3.Connection,
    config: SearchConfig,
    count: int,
    collection: str | None = None,
) -> tuple[list[BM25Result], bool]:
    """
    Run a BM25 full-text search and return results + early-return flag.

    Parameters
    ----------
    query      : raw user query string
    conn       : open database connection (virtual tables must be loaded)
    config     : SearchConfig (thresholds, default_count)
    count      : number of results to return
    collection : if given, restrict to this collection only

    Returns
    -------
    (results, early_return)
        results      : list of BM25Result, ordered best-first, len <= count
        early_return : True if the top result is dominant (skip vector stage)

    Notes
    -----
    We fetch count * 5 candidates from FTS5 so normalisation has enough
    data points.  After normalisation we trim to `count`.
    """
    ensure_virtual_tables(conn)

    # Sanitise the query for FTS5: wrap in quotes to treat as a phrase,
    # but fall back to prefix search if it contains FTS5 special chars.
    fts_query = _make_fts_query(query)

    # Fetch more candidates than needed so normalisation is meaningful.
    fetch_limit = count * 5

    # Build the SQL, optionally filtering by collection.
    collection_filter = (
        "AND d.collection = :collection" if collection else ""
    )

    sql = f"""
        SELECT
            c.id          AS chunk_id,
            d.id          AS doc_id,
            d.title       AS title,
            d.collection  AS collection,
            c.section_path,
            c.chunk_type,
            bm25(documents_fts) AS raw_score,
            -- Retrieve the chunk text from content using pos+len.
            -- SQLite substr is 1-indexed, so we add 1 to the 0-based pos.
            substr(ct.doc, c.pos + 1, c.len) AS excerpt
        FROM documents_fts
        JOIN chunks      c  ON c.id         = documents_fts.rowid
        JOIN documents   d  ON d.hash       = c.content_hash
        JOIN content     ct ON ct.hash      = c.content_hash
        WHERE documents_fts MATCH :query
          AND d.active = 1
          {collection_filter}
        ORDER BY raw_score          -- ascending = best first (most negative)
        LIMIT :limit
    """

    params: dict = {"query": fts_query, "limit": fetch_limit}
    if collection:
        params["collection"] = collection

    rows = conn.execute(sql, params).fetchall()

    if not rows:
        return [], False

    # --- Normalise scores ----------------------------------------------------
    raw_scores = [row["raw_score"] for row in rows]
    best  = min(raw_scores)   # most negative = best
    worst = max(raw_scores)   # least negative = worst

    span = best - worst       # always negative or zero
    if span == 0:
        # All rows have the same score (single result or identical scores).
        norm_scores = [1.0] * len(raw_scores)
    else:
        norm_scores = [(r - worst) / span for r in raw_scores]

    # --- Build result objects ------------------------------------------------
    results: list[BM25Result] = []
    for row, score in zip(rows, norm_scores):
        results.append(BM25Result(
            chunk_id     = row["chunk_id"],
            doc_id       = row["doc_id"],
            title        = row["title"],
            collection   = row["collection"],
            section_path = row["section_path"],
            chunk_type   = row["chunk_type"],
            score        = score,
            excerpt      = row["excerpt"],
        ))

    # Trim to requested count after normalisation.
    results = results[:count]

    # --- Early-return decision -----------------------------------------------
    early_return = False
    if len(results) >= 2:
        gap = results[0].score - results[1].score
        early_return = (
            results[0].score >= config.bm25_threshold
            and gap >= config.bm25_gap
        )
    elif len(results) == 1:
        # Single result with a strong score is also a clear winner.
        early_return = results[0].score >= config.bm25_threshold

    return results, early_return


# ---------------------------------------------------------------------------
# FTS5 query helpers
# ---------------------------------------------------------------------------

def _make_fts_query(raw: str) -> str:
    """
    Convert a raw user string to an FTS5 query expression.

    Strategy:
    - Strip leading/trailing whitespace.
    - If the string contains FTS5 special characters (", *, -, OR, AND, NOT)
      pass it through unchanged — the user knows what they're doing.
    - Otherwise wrap in double quotes for an exact-phrase search, then add
      a wildcard on the last token to allow prefix matching:
          "jwt token"* → matches "jwt token", "jwt tokens", etc.

    This keeps simple queries simple and lets power users write FTS5 syntax.
    """
    raw = raw.strip()
    fts_special = {'"', '*', '-', '(', ')'}
    fts_keywords = {"OR", "AND", "NOT"}

    has_special = any(c in raw for c in fts_special)
    has_keyword = any(kw in raw.upper().split() for kw in fts_keywords)

    if has_special or has_keyword:
        return raw

    # Simple query: phrase + prefix wildcard on last token.
    return f'"{raw}"*'
