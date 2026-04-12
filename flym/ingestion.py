"""
flym.ingestion
--------------
Core logic for adding a document to the knowledge base.

The flow
~~~~~~~~
1. Read file → compute SHA-256 hash
2. Parse YAML frontmatter → extract title and metadata
3. Ensure the target collection exists
4. If --link: record original path; else copy into vault
5. INSERT OR IGNORE into content   (dedup: same hash → skip)
6. Upsert into documents           (same collection+path → update)

Why content and documents are separate tables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
content holds the raw text, keyed by hash.  If you add the same file to two
different collections, the text is stored once and both document rows point
to it.  When you update a file, the old content row stays until GC runs —
only the hash FK on the document row changes.

Why INSERT OR IGNORE for content
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If two documents have identical text, they share a content row.  OR IGNORE
means the second insert is silently skipped — no error, no duplicate.

Why ON CONFLICT DO UPDATE for documents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sqlite3's upsert syntax: if (collection, path) already exists, update the
mutable fields instead of erroring.  This makes `flym add` idempotent:
re-adding the same file re-hashes it and updates modified_at, but doesn't
create a second row.  The id and created_at stay stable.
"""

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import frontmatter  # python-frontmatter

from flym.config import Config, load_config
from flym.vault import copy_to_vault, ensure_collection, resolve_doc_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_hash(text: str) -> str:
    """Return the SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(text.encode()).hexdigest()


def extract_title(text: str, filename: str) -> tuple[str, dict[str, Any]]:
    """
    Parse YAML frontmatter and extract the document title.

    Priority order:
        1. 'title' key in YAML frontmatter
        2. First '# Heading' found in the body
        3. Filename without extension (last resort)

    Returns (title, metadata_dict).  metadata_dict contains all frontmatter
    keys (including 'title' if present) and is stored as JSON in documents.metadata.
    """
    post = frontmatter.loads(text)
    metadata: dict[str, Any] = dict(post.metadata)

    # 1. frontmatter title
    if title := metadata.get("title"):
        return str(title), metadata

    # 2. first ATX heading in the body
    for line in post.content.splitlines():
        stripped = line.lstrip("#").strip()
        if line.startswith("#") and stripped:
            return stripped, metadata

    # 3. filename fallback
    return Path(filename).stem, metadata


def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _mtime_iso(path: Path) -> str:
    """File modification time as an ISO-8601 string."""
    ts = path.stat().st_mtime
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def add_document(
    file: Path,
    collection: str,
    link: bool,
    conn: sqlite3.Connection,
    config: Config,
) -> dict[str, Any]:
    """
    Add *file* to the knowledge base under *collection*.

    Parameters
    ----------
    file       : absolute path to the source file
    collection : collection name (must exist, or "default" for auto-create)
    link       : if True, reference the original path instead of copying
    conn       : open database connection
    config     : active Config object

    Returns a dict with the result:
        {"status": "added"|"updated"|"unchanged", "title": ..., "hash": ...}
    """
    # --- 1. Read + hash -------------------------------------------------------
    text = file.read_text(encoding="utf-8")
    content_hash = compute_hash(text)

    # --- 2. Parse frontmatter -------------------------------------------------
    import json
    title, metadata = extract_title(text, file.name)
    metadata_json = json.dumps(metadata) if metadata else None

    # --- 3. Ensure collection exists ------------------------------------------
    ensure_collection(collection, conn, config)

    collection_row = conn.execute(
        "SELECT path FROM store_collections WHERE name = ?", (collection,)
    ).fetchone()
    collection_root = Path(collection_row["path"])

    # --- 4. Resolve paths -----------------------------------------------------
    doc_path = resolve_doc_path(file, collection_root)

    if link:
        # Link mode: record the original absolute path, no copy.
        # doc_path stores the absolute path so it can be re-read later.
        vault_or_source_path = str(file)
    else:
        # Import mode: copy into vault, derive relative doc_path from vault root.
        vault_abs = copy_to_vault(file, doc_path, collection, config)
        vault_or_source_path = str(vault_abs)

    # --- 5. Insert content (dedup) --------------------------------------------
    # OR IGNORE: if this hash already exists, skip — the text is already stored.
    existing = conn.execute(
        "SELECT hash FROM content WHERE hash = ?", (content_hash,)
    ).fetchone()

    conn.execute(
        "INSERT OR IGNORE INTO content(hash, doc, created_at) VALUES (?, ?, ?)",
        (content_hash, text, _now_iso()),
    )

    # --- 6. Upsert document ---------------------------------------------------
    # Check if a document at this (collection, path) already exists.
    existing_doc = conn.execute(
        "SELECT hash FROM documents WHERE collection = ? AND path = ?",
        (collection, doc_path),
    ).fetchone()

    now = _now_iso()
    mtime = _mtime_iso(file)

    conn.execute(
        """
        INSERT INTO documents(collection, path, title, hash, metadata,
                              active, created_at, modified_at)
        VALUES (?, ?, ?, ?, ?, 1, ?, ?)
        ON CONFLICT(collection, path) DO UPDATE SET
            title       = excluded.title,
            hash        = excluded.hash,
            metadata    = excluded.metadata,
            active      = 1,
            deleted_at  = NULL,
            modified_at = excluded.modified_at
        """,
        (collection, doc_path, title, content_hash, metadata_json, now, mtime),
    )
    conn.commit()

    # Determine what actually happened for the return status.
    if existing_doc is None:
        status = "added"
    elif existing_doc["hash"] != content_hash:
        status = "updated"
    else:
        status = "unchanged"

    return {"status": status, "title": title, "hash": content_hash, "path": doc_path}
