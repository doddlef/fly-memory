"""
flym.cli.watch
--------------
Implements `flym watch <dir>` — a foreground filesystem watcher that
automatically ingests and indexes files as they change.

    flym watch ~/Documents/notes
    flym watch ~/Documents/notes -c work
    flym watch ~/Documents/notes -c work --pattern '*.md'

How it works
~~~~~~~~~~~~
`watchfiles` emits a set of (ChangeType, path) pairs each time files in the
watched directory change.  We process each changed path:

  ADDED / MODIFIED → add_document() then index_document()
  DELETED          → soft-delete the document

The watcher runs in the foreground and blocks until Ctrl-C.

Why foreground only?
    Background daemon management (PID files, systemd units, launchd plists)
    is OS-specific and adds significant complexity for limited value at this
    stage.  The foreground watcher is easy to reason about: kill the terminal,
    the watcher stops.

Debouncing
~~~~~~~~~~
`watchfiles` already debounces rapid successive changes (e.g. an editor
writing a temp file then renaming it).  We rely on its built-in debouncing
rather than implementing our own.
"""

from __future__ import annotations

from pathlib import Path

import click
from watchfiles import Change, watch

from flym.config import load_config
from flym.db import connect
from flym.indexer import ensure_virtual_tables, index_document
from flym.ingestion import add_document
from flym.providers.ollama import OllamaEmbedding


@click.command("watch")
@click.argument("directory", type=click.Path(exists=True, file_okay=False,
                                               resolve_path=True, path_type=Path))
@click.option("--collection", "-c", default="default", show_default=True,
              help="Collection to add watched files to.")
@click.option("--pattern", default="*.md", show_default=True,
              help="Only process files matching this glob pattern.")
def watch_cmd(directory: Path, collection: str, pattern: str) -> None:
    """Watch DIRECTORY and auto-ingest changed files. Press Ctrl-C to stop."""
    config   = load_config()
    provider = OllamaEmbedding(model=config.embedding.model)

    click.echo(f"Watching {directory}  (collection={collection}, pattern={pattern})")
    click.echo("Press Ctrl-C to stop.\n")

    try:
        for changes in watch(directory):
            conn = connect()
            try:
                ensure_virtual_tables(conn)
                for change_type, raw_path in changes:
                    path = Path(raw_path)

                    # Skip files that don't match the pattern.
                    if not path.match(pattern):
                        continue

                    if change_type in (Change.added, Change.modified):
                        _handle_upsert(path, collection, conn, provider, config)
                    elif change_type == Change.deleted:
                        _handle_delete(path, collection, conn)
            finally:
                conn.close()

    except KeyboardInterrupt:
        click.echo("\nWatcher stopped.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _handle_upsert(
    path: Path,
    collection: str,
    conn,
    provider: OllamaEmbedding,
    config,
) -> None:
    try:
        result = add_document(path, collection, link=False, conn=conn, config=config)
        icon   = {"added": "+", "updated": "~", "unchanged": "="}[result["status"]]
        click.echo(f"[{icon}] {result['title']}")

        if result["status"] in ("added", "updated"):
            doc_row = conn.execute(
                "SELECT id FROM documents WHERE collection=? AND path=?",
                (collection, result["path"]),
            ).fetchone()
            if doc_row:
                index_result = index_document(
                    doc_row["id"], conn, provider, config
                )
                click.echo(
                    f"    indexed {index_result['chunks_written']} chunks"
                )
    except Exception as exc:
        click.echo(f"[!] {path.name}: {exc}")


def _handle_delete(path: Path, collection: str, conn) -> None:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "UPDATE documents SET active=0, deleted_at=? "
        "WHERE path LIKE ? AND collection=? AND active=1",
        (now, f"%{path.name}", collection),
    )
    conn.commit()
    if cursor.rowcount:
        click.echo(f"[-] {path.name}  (soft-deleted)")
