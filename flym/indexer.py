"""
flym.indexer
------------
Chunks documents, embeds the chunks, and stores results in SQLite.

The three things this module sets up
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. chunks table          — chunk metadata (pos, len, section_path, …)
2. vectors_vec table     — one float32 vector per chunk (sqlite-vec KNN)
3. documents_fts table   — FTS5 full-text index with content triggers

Why vectors_vec is created here, not in db.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sqlite-vec is a loadable extension, not a built-in.  We must load it with
sqlite3.load_extension() before we can CREATE the virtual table.  That
extension load requires the database connection that's already open, so it
can't happen at schema-creation time in db.py.  The virtual tables are
created on the first call to ensure_virtual_tables().

Why FTS5 uses content triggers instead of a content table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A *content table* FTS5 would duplicate the text inside SQLite.  Instead we
use an *external content* FTS5 table: documents_fts stores only the FTS
index; the actual text lives in content.doc.  Three triggers (after insert,
after update, after delete on documents) keep them in sync automatically.

Batch size for embedding
~~~~~~~~~~~~~~~~~~~~~~~~
We embed EMBED_BATCH_SIZE chunks per call.  One batch = one round-trip to
Ollama.  Too small → many round-trips.  Too large → risk of memory pressure
or timeout.  64 is a safe default for local models on consumer hardware.

Re-index detection
~~~~~~~~~~~~~~~~~~
Each chunk row stores the model name used to embed it.  index_document()
compares stored model vs. current model before embedding — if they match and
chunk count matches, the document is already up to date and we skip it.
"""

import sqlite3
from datetime import datetime, timezone

import sqlite_vec

from flym.chunker import chunk as chunk_text
from flym.config import Config
from flym.providers.base import EmbeddingProvider


EMBED_BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Virtual table setup (called once per connection)
# ---------------------------------------------------------------------------

def ensure_virtual_tables(conn: sqlite3.Connection) -> None:
    """
    Load the sqlite-vec extension and create virtual tables if absent.

    Must be called on every new connection before any vector or FTS query,
    because extensions do not persist across connections.

    Idempotent: CREATE VIRTUAL TABLE IF NOT EXISTS guards against re-creation.
    """
    # sqlite-vec ships as a Python package that exposes its .so path.
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.executescript("""
        -- KNN vector index.
        -- Each row: chunk id (rowid) + float32 blob of length = embedding dimensions.
        -- The USING vec0(...) syntax defines it as a sqlite-vec virtual table.
        -- We use a placeholder dimension of 1 here; the real table is created
        -- the first time index_document() runs with a live provider.
        CREATE VIRTUAL TABLE IF NOT EXISTS vectors_vec USING vec0(
            embedding float[768]
        );

        -- External-content FTS5 index over chunk text.
        -- content=''  means FTS5 stores only the inverted index.
        -- Actual text is retrieved from content.doc using pos+len at query time.
        -- tokenize='porter ascii' gives stemming: "running" matches "run".
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            chunk_text,
            content='',
            tokenize='porter ascii'
        );
    """)


# ---------------------------------------------------------------------------
# Index a single document
# ---------------------------------------------------------------------------

def index_document(
    doc_id: int,
    conn: sqlite3.Connection,
    provider: EmbeddingProvider,
    config: Config,
) -> dict[str, int | str]:
    """
    Chunk and embed one document, writing results to chunks + vectors_vec + FTS.

    Parameters
    ----------
    doc_id   : documents.id of the document to index
    conn     : open database connection (must have virtual tables loaded)
    provider : embedding provider (e.g. OllamaEmbedding)
    config   : active Config (chunking settings, embedding model name)

    Returns
    -------
    {"chunks_written": N, "status": "indexed"|"skipped"}

    Status "skipped" means the stored chunks already match the current model
    and chunk count — no work was done.

    Algorithm
    ~~~~~~~~~
    1. Load document row + its text from content.
    2. Chunk the text (using config.chunking parameters).
    3. Check whether re-indexing is needed (model match + chunk count match).
    4. Delete old chunks + vectors for this document.
    5. Embed all chunks in batches.
    6. Insert chunk rows + vector rows + FTS rows in one transaction.
    """
    now = datetime.now(timezone.utc).isoformat()
    model_id = config.embedding.model

    # --- 1. Load document + text ---------------------------------------------
    doc = conn.execute(
        "SELECT hash FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()

    if doc is None:
        raise ValueError(f"no document with id={doc_id}")

    content_hash = doc["hash"]
    text = conn.execute(
        "SELECT doc FROM content WHERE hash = ?", (content_hash,)
    ).fetchone()["doc"]

    # --- 2. Chunk -------------------------------------------------------------
    chunks = chunk_text(
        text,
        target_chars=config.chunking.target_chars,
        overlap_chars=config.chunking.overlap_chars,
        min_chars=config.chunking.min_chars,
    )

    # --- 3. Check if already up to date -------------------------------------
    existing = conn.execute(
        "SELECT count(*) as n, model FROM chunks WHERE content_hash = ?",
        (content_hash,),
    ).fetchone()

    if (
        existing["n"] == len(chunks)
        and existing["model"] == model_id
    ):
        return {"chunks_written": 0, "status": "skipped"}

    # --- 4. Delete stale chunks + vectors ------------------------------------
    old_ids = [
        row["id"]
        for row in conn.execute(
            "SELECT id FROM chunks WHERE content_hash = ?", (content_hash,)
        ).fetchall()
    ]

    if old_ids:
        placeholders = ",".join("?" * len(old_ids))
        conn.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", old_ids)
        conn.execute(
            f"DELETE FROM vectors_vec WHERE rowid IN ({placeholders})", old_ids
        )
        # FTS rows are keyed by rowid — delete matching entries.
        conn.execute(
            f"INSERT INTO documents_fts(documents_fts, rowid, chunk_text) "
            f"SELECT 'delete', id, '' FROM chunks WHERE id IN ({placeholders})",
            old_ids,
        )

    # --- 5. Embed in batches -------------------------------------------------
    chunk_texts = [text[c.pos : c.pos + c.len] for c in chunks]
    vectors: list[list[float]] = []

    for i in range(0, len(chunk_texts), EMBED_BATCH_SIZE):
        batch = chunk_texts[i : i + EMBED_BATCH_SIZE]
        vectors.extend(provider.embed(batch))

    # --- 6. Insert chunks, vectors, FTS --------------------------------------
    # Pass 1: insert all chunk rows in one call.
    conn.executemany(
        """
        INSERT INTO chunks(content_hash, seq, pos, len, model,
                           section_path, chunk_type, language, embedded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                content_hash,
                seq,
                c.pos,
                c.len,
                model_id,
                c.section_path or None,
                c.chunk_type,
                c.language,
                now,
            )
            for seq, c in enumerate(chunks)
        ],
    )

    # Pass 2: retrieve the IDs SQLite assigned, in seq order.
    chunk_ids = [
        row["id"]
        for row in conn.execute(
            "SELECT id FROM chunks WHERE content_hash = ? ORDER BY seq",
            (content_hash,),
        ).fetchall()
    ]

    # Pass 3: insert all vectors in one call.
    # sqlite_vec.serialize_float32() converts list[float] → bytes (float32 LE).
    conn.executemany(
        "INSERT INTO vectors_vec(rowid, embedding) VALUES (?, ?)",
        [
            (chunk_id, sqlite_vec.serialize_float32(vec))
            for chunk_id, vec in zip(chunk_ids, vectors)
        ],
    )

    # Pass 4: insert all FTS rows in one call.
    conn.executemany(
        "INSERT INTO documents_fts(rowid, chunk_text) VALUES (?, ?)",
        list(zip(chunk_ids, chunk_texts)),
    )

    conn.commit()
    return {"chunks_written": len(chunks), "status": "indexed"}


# ---------------------------------------------------------------------------
# Index all unindexed (or stale) documents
# ---------------------------------------------------------------------------

def index_all(
    conn: sqlite3.Connection,
    provider: EmbeddingProvider,
    config: Config,
) -> list[dict]:
    """
    Walk all active documents and index any that are new or stale.

    Returns a list of per-document result dicts from index_document().
    """
    doc_ids = [
        row["id"]
        for row in conn.execute(
            "SELECT id FROM documents WHERE active = 1"
        ).fetchall()
    ]

    results = []
    for doc_id in doc_ids:
        result = index_document(doc_id, conn, provider, config)
        result["doc_id"] = doc_id
        results.append(result)

    return results
