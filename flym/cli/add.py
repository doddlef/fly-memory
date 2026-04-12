"""
flym.cli.add
------------
Implements `flym add <file>`.

Examples
--------
    flym add note.md                  # import into default collection
    flym add note.md -c work          # import into 'work' collection
    flym add note.md --link           # reference only, do not copy
    flym add note.md -c work --link
"""

from pathlib import Path

import click

from flym.config import load_config
from flym.db import connect
from flym.ingestion import add_document


@click.command("add")
@click.argument("file", type=click.Path(exists=True, resolve_path=True, path_type=Path))
@click.option(
    "-c", "--collection",
    default="default",
    show_default=True,
    help="Collection to add the document to.",
)
@click.option(
    "--link",
    is_flag=True,
    default=False,
    help="Reference the original file path instead of copying into the vault.",
)
def add(file: Path, collection: str, link: bool) -> None:
    """Add FILE to the knowledge base.

    By default the file is copied into ~/.flym/vault/<collection>/.
    Use --link to reference the original path without copying.
    """
    config = load_config()
    conn = connect()

    try:
        result = add_document(file, collection, link, conn, config)
    except ValueError as exc:
        raise click.ClickException(str(exc))
    finally:
        conn.close()

    # Status icons make it easy to spot at a glance what happened.
    icon = {"added": "+", "updated": "~", "unchanged": "="}[result["status"]]
    mode = "linked" if link else "imported"

    click.echo(f"[{icon}] {result['title']}")
    click.echo(f"    path       : {result['path']}")
    click.echo(f"    collection : {collection}")
    click.echo(f"    hash       : {result['hash'][:12]}...")
    click.echo(f"    status     : {result['status']} ({mode})")
