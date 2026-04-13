"""
flym.search.vector
-------------------
KNN vector search using the sqlite-vec virtual table.

How sqlite-vec KNN works
~~~~~~~~~~~~~~~~~~~~~~~~~
The `vectors_vec` table stores one `float[768]` blob per chunk.  To find
the K nearest neighbours of a query vector, sqlite-vec overloads the MATCH
operator and adds a special `k` constraint:

    SELECT rowid, distance
    FROM vectors_vec
    WHERE embedding MATCH :query_blob
      AND k = :k
    ORDER BY distance

`distance` is the L2 (Euclidean) distance between the query vector and each
stored vector.  Lower = more similar.

Why not cosine similarity?
    sqlite-vec currently supports L2 and inner-product distance.  L2 is fine
    for normalised vectors (nomic-embed-text normalises by default), where
    L2 and cosine similarity rank results identically.  We keep it simple.

The query vector must be serialised to the same binary format as the stored
vectors: 32-bit little-endian floats packed end-to-end.
sqlite_vec.serialize_float32() does exactly that.

Return value
~~~~~~~~~~~~
vector_search() returns a list of VectorResult, ordered by distance
(closest first).  VectorResult carries the same fields as BM25Result so the
hybrid layer can fuse both lists without special-casing.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import sqlite_vec

from flym.indexer import ensure_virtual_tables


@dataclass
class VectorResult:
    """One result row returned by vector_search()."""
    chunk_id:     int
    doc_id:       int
    title:        str
    collection:   str
    section_path: str | None
    chunk_type:   str
    distance:     float   # L2 distance; lower = more similar (not normalised)
    excerpt:      str


def vector_search(
    query_vec: list[float],
    conn: sqlite3.Connection,
    count: int,
    collection: str | None = None,
) -> list[VectorResult]:
    """
    Find the `count` chunks whose embeddings are closest to `query_vec`.

    Parameters
    ----------
    query_vec  : embedding of the user query (list of floats)
    conn       : open database connection (virtual tables must be loaded)
    count      : number of results to return
    collection : if given, restrict to this collection only

    Returns
    -------
    list of VectorResult ordered by distance (closest first)
    """
    ensure_virtual_tables(conn)

    query_blob = sqlite_vec.serialize_float32(query_vec)
    collection_filter = "AND d.collection = :collection" if collection else ""

    # We fetch count * 3 from the vector index then filter by collection.
    # The KNN index doesn't know about the collection filter, so we over-fetch
    # to ensure we have enough results after filtering.
    knn_limit = count * 3 if collection else count

    sql = f"""
        SELECT
            v.rowid       AS chunk_id,
            v.distance    AS distance,
            d.id          AS doc_id,
            d.title       AS title,
            d.collection  AS collection,
            c.section_path,
            c.chunk_type,
            substr(ct.doc, c.pos + 1, c.len) AS excerpt
        FROM vectors_vec v
        JOIN chunks    c  ON c.id   = v.rowid
        JOIN documents d  ON d.hash = c.content_hash
        JOIN content   ct ON ct.hash = c.content_hash
        WHERE v.embedding MATCH :query_blob
          AND v.k = :k
          AND d.active = 1
          {collection_filter}
        ORDER BY v.distance
        LIMIT :limit
    """

    params: dict = {"query_blob": query_blob, "k": knn_limit, "limit": count}
    if collection:
        params["collection"] = collection

    rows = conn.execute(sql, params).fetchall()

    return [
        VectorResult(
            chunk_id     = row["chunk_id"],
            doc_id       = row["doc_id"],
            title        = row["title"],
            collection   = row["collection"],
            section_path = row["section_path"],
            chunk_type   = row["chunk_type"],
            distance     = row["distance"],
            excerpt      = row["excerpt"],
        )
        for row in rows
    ]
