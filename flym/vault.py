"""
flym.vault
----------
Handles the physical placement of documents on disk.

Two modes
~~~~~~~~~
Import mode (default):
    The file is *copied* into ~/.flym/vault/<collection>/<relative-path>.
    flym owns this copy — the original can be moved or deleted without
    breaking anything.

Link mode (--link):
    The file stays in place; only its path is recorded.  Useful for large
    files or when you want the original to stay in sync.  If the original
    moves, the reference breaks (detected later via hash mismatch).

Collection root resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~
When a file lives inside the collection's root directory, its vault path
mirrors the relative directory structure:

    collection root : /Users/you/notes
    file            : /Users/you/notes/auth/jwt.md
    vault path      : ~/.flym/vault/work/auth/jwt.md   ← structure preserved

When a file is *outside* the collection root (e.g. added ad-hoc), we fall
back to just the filename:

    file            : /tmp/scratch.md
    vault path      : ~/.flym/vault/work/scratch.md
"""

import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from flym.config import Config


def ensure_collection(name: str, conn: sqlite3.Connection, config: Config) -> None:
    """
    Make sure a collection row exists in store_collections.

    For the built-in "default" collection we create it automatically,
    pointing its root at ~/.flym/vault/default/.  For any other name the
    caller is expected to have registered the collection already via
    `flym collections add`.

    This is intentionally lenient for the "default" case so that
    `flym add note.md` just works without any setup.
    """
    row = conn.execute(
        "SELECT name FROM store_collections WHERE name = ?", (name,)
    ).fetchone()

    if row is not None:
        return  # already registered, nothing to do

    if name == "default":
        # Auto-create a default collection backed by the vault itself.
        default_path = config.vault_path / "default"
        default_path.mkdir(parents=True, exist_ok=True)
        conn.execute(
            """
            INSERT INTO store_collections(name, path, pattern, include_by_default)
            VALUES (?, ?, '**/*.md', 1)
            """,
            (name, str(default_path)),
        )
        conn.commit()
    else:
        raise ValueError(
            f"Collection '{name}' does not exist. "
            f"Register it first with: flym collections add {name} <path>"
        )


def resolve_doc_path(src: Path, collection_root: Path) -> str:
    """
    Return the document path *relative to the collection root*.

    This string is stored in documents.path and used as a stable identifier
    for the file within its collection.

    Examples
    --------
    >>> resolve_doc_path(Path("/notes/auth/jwt.md"), Path("/notes"))
    'auth/jwt.md'
    >>> resolve_doc_path(Path("/tmp/scratch.md"), Path("/notes"))
    'scratch.md'   # outside root → fall back to filename only
    """
    try:
        return str(src.relative_to(collection_root))
    except ValueError:
        # File is not inside the collection root — use filename only.
        return src.name


def copy_to_vault(
    src: Path,
    doc_path: str,
    collection: str,
    config: Config,
) -> Path:
    """
    Copy *src* into the vault and return the absolute vault path.

    The vault destination mirrors the relative doc_path structure:
        ~/.flym/vault/<collection>/<doc_path>

    If a file already exists at the destination it is overwritten — this
    handles re-adding a changed file gracefully.
    """
    dest = config.vault_path / collection / doc_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)   # copy2 preserves mtime metadata
    return dest
