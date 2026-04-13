"""
flym.cli.search
---------------
Implements `flym search "<query>"`.

Examples
--------
    flym search "JWT token"
    flym search "JWT token" --count 10
    flym search "JWT token" -c work
    flym search "JWT token" --json
"""

from __future__ import annotations

import json

import click

from flym.config import load_config
from flym.db import connect
from flym.indexer import ensure_virtual_tables
from flym.search.bm25 import bm25_search


@click.command("search")
@click.argument("query")
@click.option(
    "--count", "-n",
    default=None,
    type=int,
    help="Number of results to return (default from config).",
)
@click.option(
    "--collection", "-c",
    default=None,
    help="Restrict search to this collection.",
)
@click.option(
    "--json", "as_json",
    is_flag=True,
    default=False,
    help="Output results as a JSON array.",
)
def search(
    query: str,
    count: int | None,
    collection: str | None,
    as_json: bool,
) -> None:
    """Search the knowledge base for QUERY."""
    config = load_config()
    n = count if count is not None else config.search.default_count

    conn = connect()
    try:
        ensure_virtual_tables(conn)
        results, early_return = bm25_search(
            query,
            conn,
            config.search,
            count=n,
            collection=collection,
        )
    finally:
        conn.close()

    if not results:
        click.echo("No results found.")
        return

    if as_json:
        click.echo(json.dumps([
            {
                "score":        r.score,
                "title":        r.title,
                "collection":   r.collection,
                "section_path": r.section_path,
                "chunk_type":   r.chunk_type,
                "excerpt":      r.excerpt,
            }
            for r in results
        ], indent=2))
        return

    # --- Plain text output ---------------------------------------------------
    # Show a subtle indicator when the fast path triggered.
    if early_return:
        click.echo(f"  (BM25 early return — top result is dominant)\n")

    for i, r in enumerate(results):
        # Heading line: rank, score bar, title
        bar   = _score_bar(r.score)
        path  = f"  {r.section_path}" if r.section_path else ""
        click.echo(f"[{i+1}] {bar} {r.score:.2f}  {r.title}{path}")

        # Excerpt: first 200 chars, newlines replaced with spaces
        preview = r.excerpt[:200].replace("\n", " ").strip()
        click.echo(f"     {preview}")
        click.echo()


def _score_bar(score: float, width: int = 5) -> str:
    """Return a small ASCII bar, e.g. '████░' for score 0.8."""
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)
