"""
flym.cli.collections
---------------------
Implements `flym collections <subcommand>`.

    flym collections add work ~/Documents/notes
    flym collections add work ~/notes --pattern '**/*.{md,txt}'
    flym collections add work ~/notes --update-cmd 'git pull'
    flym collections list
    flym collections update work
"""

from __future__ import annotations

import click

from flym.collections import list_collections, register_collection, update_collection
from flym.config import load_config
from flym.db import connect


@click.group("collections")
def collections() -> None:
    """Manage document collections."""


@collections.command("add")
@click.argument("name")
@click.argument("path", type=click.Path(exists=True, file_okay=False,
                                         resolve_path=True))
@click.option("--pattern", default="**/*.md", show_default=True,
              help="Glob pattern for file discovery.")
@click.option("--update-cmd", default=None,
              help="Shell command to run before re-indexing (e.g. 'git pull').")
@click.option("--exclude/--include", "include_by_default",
              default=True, show_default=True,
              help="Exclude from searches unless -c is specified.")
def collections_add(
    name: str,
    path: str,
    pattern: str,
    update_cmd: str | None,
    include_by_default: bool,
) -> None:
    """Register a new collection NAME pointing at PATH."""
    conn = connect()
    try:
        created = register_collection(
            name, path, conn,
            pattern=pattern,
            update_command=update_cmd,
            include_by_default=include_by_default,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc))
    finally:
        conn.close()

    if created:
        click.echo(f"[+] collection '{name}' registered")
        click.echo(f"    path    : {path}")
        click.echo(f"    pattern : {pattern}")
    else:
        click.echo(f"[=] collection '{name}' already exists (no change)")


@collections.command("list")
def collections_list() -> None:
    """List all registered collections."""
    conn = connect()
    try:
        rows = list_collections(conn)
    finally:
        conn.close()

    if not rows:
        click.echo("No collections registered.")
        return

    for row in rows:
        flag = "" if row["include_by_default"] else "  [excluded by default]"
        click.echo(f"  {row['name']:<20} {row['path']}{flag}")
        if row["update_command"]:
            click.echo(f"    {'update_cmd':<18} {row['update_command']}")


@collections.command("update")
@click.argument("name")
def collections_update(name: str) -> None:
    """Sync and re-ingest all files in collection NAME."""
    config = load_config()
    conn = connect()
    try:
        result = update_collection(name, conn, config)
    except ValueError as exc:
        raise click.ClickException(str(exc))
    finally:
        conn.close()

    click.echo(f"Collection '{name}' updated:")
    click.echo(f"  added     : {result['added']}")
    click.echo(f"  updated   : {result['updated']}")
    click.echo(f"  unchanged : {result['unchanged']}")
    click.echo("Run `flym index` to embed new/updated documents.")
