"""
flym.cli.add
------------
Implements `flym add <path> [path ...]`.

Accepts any mix of files and directories.  Directories are walked recursively
using --pattern (default **/*.md).

Examples
--------
    flym add note.md                        # single file
    flym add note.md other.md               # multiple files
    flym add ~/Documents/notes/             # whole directory
    flym add ~/notes/ -c work               # directory into a collection
    flym add ~/notes/ --pattern '**/*.txt'  # different file type
    flym add note.md --link                 # reference only, do not copy
    flym add note.md --index                # add + embed in one step
    flym add ~/notes/ --index               # bulk add + embed
"""

from pathlib import Path

import click

from flym.config import load_config
from flym.db import connect
from flym.ingestion import add_document


@click.command("add")
@click.argument("paths", nargs=-1, required=True,
                type=click.Path(exists=True, resolve_path=True, path_type=Path))
@click.option("-c", "--collection", default="default", show_default=True,
              help="Collection to add the document(s) to.")
@click.option("--link", is_flag=True, default=False,
              help="Reference original paths instead of copying into the vault.")
@click.option("--pattern", default="**/*.md", show_default=True,
              help="Glob pattern used when a path is a directory.")
@click.option("--index", "do_index", is_flag=True, default=False,
              help="Embed and index documents immediately after adding.")
def add(
    paths: tuple[Path, ...],
    collection: str,
    link: bool,
    pattern: str,
    do_index: bool,
) -> None:
    """Add one or more files (or directories) to the knowledge base.

    Directories are walked recursively with --pattern.
    Use --index to embed immediately (otherwise run `flym index` separately).
    """
    config = load_config()
    files  = _collect_files(paths, pattern)

    if not files:
        raise click.ClickException(
            f"no files matched pattern '{pattern}' in the given paths"
        )

    # Lazily import indexing machinery — only needed when --index is passed.
    if do_index:
        from flym.indexer import ensure_virtual_tables, index_document
        from flym.providers.ollama import OllamaEmbedding
        provider = OllamaEmbedding(model=config.embedding.model)

    conn = connect()
    try:
        if do_index:
            ensure_virtual_tables(conn)

        counts      = {"added": 0, "updated": 0, "unchanged": 0}
        index_total = 0
        mode        = "linked" if link else "imported"

        for file in files:
            try:
                result = add_document(file, collection, link, conn, config)
            except ValueError as exc:
                click.echo(f"[!] {file.name}: {exc}")
                continue

            counts[result["status"]] += 1
            icon = {"added": "+", "updated": "~", "unchanged": "="}[result["status"]]

            if len(files) == 1:
                click.echo(f"[{icon}] {result['title']}")
                click.echo(f"    path       : {result['path']}")
                click.echo(f"    collection : {collection}")
                click.echo(f"    hash       : {result['hash'][:12]}...")
                click.echo(f"    status     : {result['status']} ({mode})")
            else:
                click.echo(f"[{icon}] {result['path']}")

            # Index immediately if requested and the document changed.
            if do_index and result["status"] in ("added", "updated"):
                doc_row = conn.execute(
                    "SELECT id FROM documents WHERE collection=? AND path=?",
                    (collection, result["path"]),
                ).fetchone()
                if doc_row:
                    idx = index_document(doc_row["id"], conn, provider, config)
                    index_total += idx["chunks_written"]
                    if len(files) == 1:
                        click.echo(f"    chunks     : {idx['chunks_written']} indexed")

        if len(files) > 1:
            summary = (
                f"\n{counts['added']} added, "
                f"{counts['updated']} updated, "
                f"{counts['unchanged']} unchanged  "
                f"({mode}, collection={collection})"
            )
            if do_index:
                summary += f"\n{index_total} chunks indexed"
            click.echo(summary)

    finally:
        conn.close()


def _collect_files(paths: tuple[Path, ...], pattern: str) -> list[Path]:
    """Expand directories using glob; pass files through directly."""
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(p for p in path.glob(pattern) if p.is_file()))
        else:
            files.append(path)
    return files
