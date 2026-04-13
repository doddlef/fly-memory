"""
flym.cache
----------
Unified read/write interface for the `cache` table, plus probabilistic GC.

Cache types
~~~~~~~~~~~
type='llm'    expires_at=NULL   — LLM expansion results (rephrase, HyDE).
              Permanent because the same prompt always produces the same
              useful expansion.  No TTL needed.

type='search' expires_at=<iso>  — serialised search result lists.
              Expire after config.search.cache_ttl_hours (default 24 h).
              A repeated identical query within the TTL window is instant.

Cache key
~~~~~~~~~
SHA-256 of the input string (prompt for LLM, canonical query string for
search).  Collisions are astronomically unlikely; we treat the hash as the
primary key directly.

Probabilistic eviction
~~~~~~~~~~~~~~~~~~~~~~
Running a DELETE on every cache read would add latency to every search.
Instead, each read rolls a random float in [0, 1): if it falls below
EVICT_PROBABILITY (1%), the GC routine runs.  On average, one in every 100
reads cleans up expired rows — the cost is amortised invisibly.

    Expected cleanup frequency at 10 searches/day: once every 10 days.
    Expected cleanup frequency at 100 searches/day: roughly daily.

Content GC
~~~~~~~~~~
The same probabilistic trigger also runs a content table GC pass: it deletes
content rows that are no longer referenced by any document (active or
soft-deleted).  This reclaims space from documents that were purged.

Why NOT EXISTS instead of NOT IN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    DELETE FROM content WHERE hash NOT IN (SELECT hash FROM documents)

This looks correct but is subtly broken: if `documents.hash` ever contains a
NULL, the entire NOT IN condition evaluates to NULL (not TRUE), silently
skipping every deletion.  `NOT EXISTS` is NULL-safe:

    DELETE FROM content
    WHERE NOT EXISTS (SELECT 1 FROM documents WHERE documents.hash = content.hash)

It evaluates per row and returns TRUE when no matching document exists,
regardless of NULLs elsewhere in the table.
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from datetime import datetime, timedelta, timezone

EVICT_PROBABILITY = 0.01   # 1% of reads trigger GC
EVICT_LIMIT       = 100    # max expired rows deleted per GC pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cache_get(
    key: str,
    cache_type: str,
    conn: sqlite3.Connection,
) -> object | None:
    """
    Look up a cache entry.  Returns the deserialised value, or None on miss.

    Also triggers probabilistic GC (EVICT_PROBABILITY chance per call).

    Parameters
    ----------
    key        : raw string to hash (prompt text, query string, etc.)
    cache_type : 'llm' or 'search'
    conn       : open database connection
    """
    cache_key = _hash(key)
    now       = _now()

    row = conn.execute(
        """
        SELECT result FROM cache
        WHERE hash = ? AND type = ?
          AND (expires_at IS NULL OR expires_at > ?)
        """,
        (cache_key, cache_type, now),
    ).fetchone()

    # Probabilistic GC — runs ~1% of the time, independent of hit/miss.
    if random.random() < EVICT_PROBABILITY:
        _evict(conn)

    return json.loads(row["result"]) if row else None


def cache_set(
    key: str,
    value: object,
    cache_type: str,
    conn: sqlite3.Connection,
    ttl_hours: int | None = None,
) -> None:
    """
    Store a value in the cache.

    Parameters
    ----------
    key        : raw string to hash
    value      : any JSON-serialisable object
    cache_type : 'llm' or 'search'
    conn       : open database connection
    ttl_hours  : if None, entry is permanent (expires_at=NULL)
                 if set, entry expires after this many hours
    """
    cache_key  = _hash(key)
    expires_at = None
    if ttl_hours is not None:
        expires_at = (
            datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
        ).isoformat()

    conn.execute(
        """
        INSERT OR REPLACE INTO cache(hash, type, result, created_at, expires_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (cache_key, cache_type, json.dumps(value), _now(), expires_at),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# GC
# ---------------------------------------------------------------------------

def _evict(conn: sqlite3.Connection) -> None:
    """
    Delete expired cache rows and GC orphaned content.

    Called automatically with probability EVICT_PROBABILITY on each
    cache_get().  Safe to call manually too (e.g. from a maintenance script).
    """
    now = _now()

    # 1. Delete expired search cache rows (LLM rows have expires_at=NULL and
    #    are never deleted here — only manual cache_invalidate() would do that).
    conn.execute(
        """
        DELETE FROM cache
        WHERE expires_at IS NOT NULL
          AND expires_at < ?
        LIMIT ?
        """,
        (now, EVICT_LIMIT),
    )

    # 2. GC content rows no longer referenced by any document.
    #    Uses NOT EXISTS for NULL safety (see module docstring).
    conn.execute(
        """
        DELETE FROM content
        WHERE NOT EXISTS (
            SELECT 1 FROM documents WHERE documents.hash = content.hash
        )
        """
    )

    conn.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
