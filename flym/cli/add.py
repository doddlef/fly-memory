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
def add(paths: tuple[Path, ...], collection: str, link: bool, pattern: str) -> None:
    """Add one or more files (or directories) to the knowledge base.

    Directories are walked recursively with --pattern.
    """
    config = load_config()
    files  = _collect_files(paths, pattern)

    if not files:
        raise click.ClickException(
            f"no files matched pattern '{pattern}' in the given paths"
        )

    conn = connect()
    try:
        counts = {"added": 0, "updated": 0, "unchanged": 0}
        mode   = "linked" if link else "imported"

        for file in files:
            try:
                result = add_document(file, collection, link, conn, config)
            except ValueError as exc:
                click.echo(f"[!] {file.name}: {exc}")
                continue

            counts[result["status"]] += 1
            icon = {"added": "+", "updated": "~", "unchanged": "="}[result["status"]]

            if len(files) == 1:
                # Single file: show full detail.
                click.echo(f"[{icon}] {result['title']}")
                click.echo(f"    path       : {result['path']}")
                click.echo(f"    collection : {collection}")
                click.echo(f"    hash       : {result['hash'][:12]}...")
                click.echo(f"    status     : {result['status']} ({mode})")
            else:
                # Multiple files: one line per file.
                click.echo(f"[{icon}] {result['path']}")

        if len(files) > 1:
            click.echo(
                f"\n{counts['added']} added, "
                f"{counts['updated']} updated, "
                f"{counts['unchanged']} unchanged  "
                f"({mode}, collection={collection})"
            )
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
