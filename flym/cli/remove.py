"""
flym.cli.remove
---------------
Implements `flym remove` (soft-delete) and `flym purge` (hard-delete + GC).

Soft-delete vs hard-delete
~~~~~~~~~~~~~~~~~~~~~~~~~~~
`flym remove` sets active=0 and deleted_at=<now> on the document row.
The content and chunks remain intact — the document is just invisible to
search.  This is reversible (you could set active=1 back manually).

`flym purge` is the irreversible GC step:
  1. Hard-delete document rows where deleted_at is older than the cutoff.
  2. Hard-delete chunk + vector rows for content hashes no longer referenced
     by any active or soft-deleted document.
  3. Hard-delete content rows with no remaining references.

Why keep soft-deleted documents before purging?
    You might remove a document by accident.  The soft-delete window gives
    you time to recover.  Purge is explicit and separate.

Duration syntax
~~~~~~~~~~~~~~~
`--older-than` accepts:  0d, 7d, 1w, 2w, 30d
  d = days, w = weeks
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timedelta, timezone

import click

from flym.db import connect
from flym.indexer import ensure_virtual_tables


@click.command("remove")
@click.argument("path")
@click.option("--collection", "-c", default=None,
              help="Collection the document belongs to (disambiguates if path appears in multiple).")
def remove(path: str, collection: str | None) -> None:
    """Soft-delete a document by its path."""
    conn = connect()
    try:
        count = _soft_delete(path, collection, conn)
    finally:
        conn.close()

    if count == 0:
        raise click.ClickException(f"no document found at path '{path}'")
    click.echo(f"[-] {count} document(s) marked as deleted (run `flym purge` to free space)")


@click.command("purge")
@click.option("--older-than", "older_than", default="7d", show_default=True,
              help="Remove documents deleted more than this long ago (e.g. 7d, 2w).")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show what would be deleted without actually deleting.")
def purge(older_than: str, dry_run: bool) -> None:
    """Hard-delete soft-deleted documents and garbage-collect orphaned content."""
    try:
        cutoff_delta = _parse_duration(older_than)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    cutoff = (datetime.now(timezone.utc) - cutoff_delta).isoformat()

    conn = connect()
    try:
        ensure_virtual_tables(conn)
        doc_count, chunk_count, content_count = _purge(cutoff, conn, dry_run)
    finally:
        conn.close()

    verb = "would delete" if dry_run else "deleted"
    click.echo(f"{verb}:")
    click.echo(f"  {doc_count} document(s)")
    click.echo(f"  {chunk_count} chunk(s) + vector rows")
    click.echo(f"  {content_count} content row(s)")


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

def _soft_delete(path: str, collection: str | None, conn: sqlite3.Connection) -> int:
    now = datetime.now(timezone.utc).isoformat()
    if collection:
        cursor = conn.execute(
            "UPDATE documents SET active=0, deleted_at=? "
            "WHERE path=? AND collection=? AND active=1",
            (now, path, collection),
        )
    else:
        cursor = conn.execute(
            "UPDATE documents SET active=0, deleted_at=? "
            "WHERE path=? AND active=1",
            (now, path),
        )
    conn.commit()
    return cursor.rowcount


def _purge(
    cutoff: str,
    conn: sqlite3.Connection,
    dry_run: bool,
) -> tuple[int, int, int]:
    # Documents to hard-delete.
    doomed_docs = conn.execute(
        "SELECT id, hash FROM documents WHERE active=0 AND deleted_at < ?",
        (cutoff,),
    ).fetchall()
    doc_count = len(doomed_docs)

    if doc_count == 0:
        return 0, 0, 0

    doomed_hashes = list({row["hash"] for row in doomed_docs})
    doc_ids       = [row["id"] for row in doomed_docs]

    # Content hashes that will become orphaned after these docs are removed.
    ph = ",".join("?" * len(doomed_hashes))
    orphan_hashes = [
        row["hash"]
        for row in conn.execute(
            f"SELECT hash FROM content WHERE hash IN ({ph}) "
            f"AND NOT EXISTS ("
            f"  SELECT 1 FROM documents "
            f"  WHERE hash = content.hash AND id NOT IN ({','.join('?' * len(doc_ids))})"
            f")",
            doomed_hashes + doc_ids,
        ).fetchall()
    ]

    # Count chunks to be removed.
    if orphan_hashes:
        oph = ",".join("?" * len(orphan_hashes))
        chunk_rows = conn.execute(
            f"SELECT id FROM chunks WHERE content_hash IN ({oph})", orphan_hashes
        ).fetchall()
        chunk_ids  = [r["id"] for r in chunk_rows]
        chunk_count = len(chunk_ids)
    else:
        chunk_ids   = []
        chunk_count = 0

    content_count = len(orphan_hashes)

    if dry_run:
        return doc_count, chunk_count, content_count

    # Hard-delete documents.
    dp = ",".join("?" * len(doc_ids))
    conn.execute(f"DELETE FROM documents WHERE id IN ({dp})", doc_ids)

    # Delete chunks + vectors + FTS for orphaned content.
    if chunk_ids:
        cp = ",".join("?" * len(chunk_ids))
        conn.execute(f"DELETE FROM chunks WHERE id IN ({cp})", chunk_ids)
        conn.execute(f"DELETE FROM vectors_vec WHERE rowid IN ({cp})", chunk_ids)
        fts_values = ", ".join("('delete', ?, '')" for _ in chunk_ids)
        conn.execute(
            f"INSERT INTO documents_fts(documents_fts, rowid, chunk_text) VALUES {fts_values}",
            chunk_ids,
        )

    # Delete orphaned content.
    if orphan_hashes:
        op = ",".join("?" * len(orphan_hashes))
        conn.execute(f"DELETE FROM content WHERE hash IN ({op})", orphan_hashes)

    conn.commit()
    return doc_count, chunk_count, content_count


def _parse_duration(s: str) -> timedelta:
    """Parse '7d' or '2w' into a timedelta."""
    m = re.fullmatch(r"(\d+)([dw])", s.strip())
    if not m:
        raise ValueError(f"invalid duration '{s}' — use e.g. 7d or 2w")
    n, unit = int(m.group(1)), m.group(2)
    return timedelta(days=n if unit == "d" else n * 7)
