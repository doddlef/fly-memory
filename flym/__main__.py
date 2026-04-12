"""
flym.__main__
-------------
Entry point for `python -m flym` and the `flym` command installed by pip.

This file only wires together the CLI group and sub-commands.
No business logic lives here — each command is implemented in its own module
under flym/cli/ and imported here as it is built in later modules.

Adding a new command:
    1. Create flym/cli/yourcommand.py with a @click.command() function.
    2. Import it here and register it with cli.add_command().

Current commands (grow with each module):
    Module 1:  (none yet — just `flym --help` and `flym db-check`)
    Module 2:  flym add
    Module 5:  flym index
    Module 6:  flym search
    Module 10: flym collections, flym remove, flym purge, flym watch
"""

import click
from flym.db import connect


@click.group()
def cli() -> None:
    """flym — personal document base.

    Search your notes and documents from the terminal.

    \b
    Quick start:
        flym collections add work ~/Documents/notes
        flym add ~/Documents/notes/auth.md
        flym search "JWT token"
    """


@cli.command("db-check")
def db_check() -> None:
    """Verify the database is reachable and show table row counts."""
    conn = connect()
    tables = [
        "content",
        "store_collections",
        "documents",
        "chunks",
        "cache",
    ]
    click.echo(f"Database: {conn.execute('PRAGMA database_list').fetchone()[2]}\n")
    for table in tables:
        try:
            count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            click.echo(f"  {table:<22} {count:>6} rows")
        except Exception as exc:
            click.echo(f"  {table:<22} ERROR: {exc}")
    conn.close()


# ---------------------------------------------------------------------------
# Commands added in later modules are imported and registered here.
# Uncomment each line as you reach the corresponding module.
# ---------------------------------------------------------------------------

# Module 2
# from flym.cli.add import add
# cli.add_command(add)

# Module 5
# from flym.cli.index import index
# cli.add_command(index)

# Module 6
# from flym.cli.search import search
# cli.add_command(search)

# Module 10
# from flym.cli.collections import collections
# from flym.cli.remove import remove, purge
# from flym.cli.watch import watch
# cli.add_command(collections)
# cli.add_command(remove)
# cli.add_command(purge)
# cli.add_command(watch)


if __name__ == "__main__":
    cli()
