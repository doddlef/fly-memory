"""
flym.collections
-----------------
Business logic for managing collections: register, list, update.

A collection is a named pointer to a directory on disk.  It records:
  - name           : short identifier used in CLI flags (-c work)
  - path           : absolute directory path
  - pattern        : glob pattern for auto-discovery (default **/*.md)
  - update_command : optional shell command run before re-indexing
                     (e.g. "git pull", "rsync …")

Why update_command?
~~~~~~~~~~~~~~~~~~~
Some collections are mirrors of external sources (a git repo, an rsync
target, a shared folder).  Rather than building sync logic into flym, we
delegate to a shell command.  `flym collections update work` runs the
command, then re-indexes any files that changed.

Re-index detection
~~~~~~~~~~~~~~~~~~
update_collection() walks the directory, calls add_document() for each
matching file, and lets the hash comparison in ingestion.py decide whether
anything actually changed.  Only documents whose hash changed get new chunks
and vectors written.  Unchanged documents are a cheap no-op.
"""

from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path

from flym.config import Config


def register_collection(
    name: str,
    path: str | Path,
    conn: sqlite3.Connection,
    *,
    pattern: str = "**/*.md",
    update_command: str | None = None,
    include_by_default: bool = True,
) -> bool:
    """
    Register a new collection in store_collections.

    Returns True if the collection was newly created, False if it already
    existed (INSERT OR IGNORE semantics).

    Parameters
    ----------
    name               : unique collection identifier
    path               : absolute path to the collection root directory
    pattern            : glob for file discovery (default **/*.md)
    update_command     : optional shell command run before re-indexing
    include_by_default : if False, excluded from searches unless -c used
    conn               : SQLite connection
    """
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise ValueError(f"path does not exist: {root}")

    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO store_collections
            (name, path, pattern, update_command, include_by_default)
        VALUES (?, ?, ?, ?, ?)
        """,
        (name, str(root), pattern, update_command, int(include_by_default)),
    )
    conn.commit()
    return cursor.rowcount == 1


def list_collections(conn: sqlite3.Connection) -> list[dict]:
    """Return all registered collections as a list of dicts."""
    rows = conn.execute(
        "SELECT name, path, pattern, update_command, include_by_default "
        "FROM store_collections ORDER BY name"
    ).fetchall()
    return [dict(row) for row in rows]


def update_collection(
    name: str,
    conn: sqlite3.Connection,
    config: Config,
) -> dict:
    """
    Sync and re-index a collection.

    Steps:
    1. Look up the collection record.
    2. If update_command is set, run it in the collection directory.
    3. Walk the directory, calling add_document() for each matching file.
    4. Return a summary dict.

    The indexing step (embed + write vectors) is NOT done here — call
    `flym index` after updating, or let `flym watch` handle it automatically.
    """
    row = conn.execute(
        "SELECT path, pattern, update_command FROM store_collections WHERE name = ?",
        (name,),
    ).fetchone()

    if row is None:
        raise ValueError(f"collection '{name}' not found")

    root           = Path(row["path"])
    pattern        = row["pattern"]
    update_command = row["update_command"]
    added = updated = unchanged = 0

    # --- Step 1: run update command ------------------------------------------
    if update_command:
        subprocess.run(update_command, shell=True, cwd=root, check=True)

    # --- Step 2: walk files and ingest ---------------------------------------
    from flym.ingestion import add_document  # avoid circular import at module level

    for file in sorted(root.glob(pattern)):
        if not file.is_file():
            continue
        try:
            result = add_document(file, name, link=False, conn=conn, config=config)
            if result["status"] == "added":
                added += 1
            elif result["status"] == "updated":
                updated += 1
            else:
                unchanged += 1
        except Exception:
            pass  # skip unreadable files silently

    return {
        "collection": name,
        "added": added,
        "updated": updated,
        "unchanged": unchanged,
    }
