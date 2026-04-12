"""
flym.db
-------
Opens the SQLite database and ensures all core tables exist.

Design notes
~~~~~~~~~~~~
- Schema uses CREATE TABLE IF NOT EXISTS so running connect() multiple times
  is safe — tables are only created on the first call (idempotent migrations).
- WAL mode: allows one writer + many concurrent readers without blocking.
- Foreign keys: enforced explicitly because SQLite disables them by default.
- Row factory: rows are returned as sqlite3.Row objects, which support both
  index access (row[0]) and column-name access (row["title"]).

Virtual tables (FTS5 and sqlite-vec) are NOT created here. They require
extensions to be loaded first and are set up in flym/indexer.py (Module 5).

Usage:
    from flym.db import connect

    conn = connect()
    rows = conn.execute("SELECT id, title FROM documents").fetchall()
    for row in rows:
        print(row["title"])    # column-name access via Row factory
    conn.close()
"""

import sqlite3
from flym.config import load_config

# ---------------------------------------------------------------------------
# Schema — core tables only.
# Virtual tables (documents_fts, vectors_vec) added in Module 5.
# ---------------------------------------------------------------------------

_SCHEMA = """
-- Immutable content store.
-- One row per unique document body, addressed by its SHA-256 hash.
-- Multiple documents can point to the same content row (deduplication).
CREATE TABLE IF NOT EXISTS content (
    hash       TEXT PRIMARY KEY,   -- SHA-256 of the full document text
    doc        TEXT NOT NULL,      -- the complete document body (source of truth)
    created_at TEXT NOT NULL       -- ISO-8601 timestamp
);

-- Collection registry.
-- Maps a logical name ("work") to a directory on disk.
-- Documents always belong to a collection.
CREATE TABLE IF NOT EXISTS store_collections (
    name               TEXT PRIMARY KEY,
    path               TEXT NOT NULL,              -- absolute path on disk
    pattern            TEXT NOT NULL DEFAULT '**/*.md',
    ignore_patterns    TEXT,                       -- JSON array of globs
    include_by_default INTEGER NOT NULL DEFAULT 1, -- 0 = skip unless -c specified
    update_command     TEXT,                       -- shell command run by `flym update`
    context            TEXT                        -- JSON: path-prefix -> hint string
);

-- Document metadata.
-- One row per file per collection.  Text lives in content, not here.
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    collection  TEXT NOT NULL REFERENCES store_collections(name),
    path        TEXT NOT NULL,     -- path relative to collection root
    title       TEXT NOT NULL,     -- from YAML frontmatter or first heading
    hash        TEXT NOT NULL REFERENCES content(hash),
    metadata    TEXT,              -- JSON: full parsed YAML frontmatter
    active      INTEGER NOT NULL DEFAULT 1,  -- 0 = soft-deleted
    deleted_at  TEXT,              -- NULL = live; timestamp = when soft-deleted
    created_at  TEXT NOT NULL,
    modified_at TEXT NOT NULL,     -- file mtime recorded at index time
    UNIQUE(collection, path)       -- same file can't be in a collection twice
);

CREATE INDEX IF NOT EXISTS idx_documents_collection
    ON documents(collection, active);

CREATE INDEX IF NOT EXISTS idx_documents_hash
    ON documents(hash);

-- Chunk metadata.
-- Chunks do NOT store their text — pos+len are offsets into content.doc.
-- Retrieve chunk text with: substr(content.doc, chunk.pos + 1, chunk.len)
-- (SQLite substr is 1-indexed, hence the +1.)
CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,  -- rowid used by FTS5 + vectors
    content_hash TEXT NOT NULL REFERENCES content(hash),
    seq          INTEGER NOT NULL,    -- 0-based chunk index within the document
    pos          INTEGER NOT NULL,    -- character offset in content.doc (0-based)
    len          INTEGER NOT NULL,    -- character length of this chunk
    model        TEXT NOT NULL,       -- embedding model id (detect stale chunks)
    section_path TEXT,                -- e.g. "Installation > MacOS > Homebrew"
    chunk_type   TEXT NOT NULL DEFAULT 'prose',  -- 'prose' | 'code' | 'mixed'
    language     TEXT,                -- null, or 'python', 'typescript', etc.
    embedded_at  TEXT NOT NULL,       -- ISO-8601 timestamp
    UNIQUE(content_hash, seq)
);

CREATE INDEX IF NOT EXISTS idx_chunks_content
    ON chunks(content_hash, seq);

-- Unified cache table.
-- type='llm'    → expires_at IS NULL  (LLM responses are deterministic, no TTL)
-- type='search' → expires_at is set   (search results expire after N hours)
CREATE TABLE IF NOT EXISTS cache (
    hash       TEXT PRIMARY KEY,   -- hash of the input (prompt or query+filters)
    type       TEXT NOT NULL,      -- 'llm' | 'search'
    result     TEXT NOT NULL,      -- JSON-serialised result
    created_at TEXT NOT NULL,
    expires_at TEXT                -- NULL = permanent; ISO-8601 = has TTL
);

-- Partial index: only index rows that actually have an expiry.
-- Keeps the index small and the cleanup query fast.
CREATE INDEX IF NOT EXISTS idx_cache_expires
    ON cache(expires_at)
    WHERE expires_at IS NOT NULL;
"""


def connect() -> sqlite3.Connection:
    """
    Open (or create) the SQLite database and return a connection.

    On first call:
    - Creates ~/.flym/ and ~/.flym/vault/ directories if they don't exist.
    - Creates the database file and runs the schema.

    On subsequent calls:
    - The IF NOT EXISTS guards make the schema a no-op.
    - Returns a fresh connection each time (connections are not shared).

    The caller is responsible for closing the connection when done.
    """
    config = load_config()

    # Ensure directories exist before SQLite tries to open the file.
    config.db_path.parent.mkdir(parents=True, exist_ok=True)
    config.vault_path.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(config.db_path))

    # Return rows as sqlite3.Row: supports row["column"] access.
    conn.row_factory = sqlite3.Row

    # WAL mode: much better for concurrent reads (e.g. watch + search at once).
    conn.execute("PRAGMA journal_mode=WAL")

    # SQLite disables FK enforcement by default for backwards compat.
    conn.execute("PRAGMA foreign_keys=ON")

    # Apply schema. executescript commits any open transaction first.
    conn.executescript(_SCHEMA)

    return conn
