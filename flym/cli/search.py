"""
flym.cli.search
---------------
Implements `flym search "<query>"`.

Search pipeline (two paths):
  Fast path  — BM25 only, returns immediately when top result is dominant.
  Hybrid path — BM25 + vector KNN, fused with RRF.  Used when BM25 is not
                confident enough to early-return.

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
from flym.providers.ollama import OllamaEmbedding
from flym.search.bm25 import BM25Result, bm25_search
from flym.search.hybrid import HybridResult, hybrid_search


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

        # Always run BM25 first — it's fast and may avoid an embedding call.
        bm25_results, early_return = bm25_search(
            query, conn, config.search, count=n, collection=collection,
        )

        if early_return or not bm25_results:
            # Fast path: BM25 result is dominant or corpus is empty.
            results = bm25_results
            path_used = "bm25"
        else:
            # Hybrid path: BM25 wasn't confident — fuse with vector search.
            provider = OllamaEmbedding(model=config.embedding.model)
            hybrid_results = hybrid_search(
                query, conn, provider, config, count=n, collection=collection,
            )
            results = hybrid_results
            path_used = "hybrid"
    finally:
        conn.close()

    if not results:
        click.echo("No results found.")
        return

    if as_json:
        click.echo(json.dumps([_to_dict(r) for r in results], indent=2))
        return

    # --- Plain text output ---------------------------------------------------
    click.echo(f"  ({path_used})\n")

    for i, r in enumerate(results):
        score  = r.score if isinstance(r, BM25Result) else r.rrf_score
        bar    = _score_bar(score)
        path   = f"  {r.section_path}" if r.section_path else ""
        ranks  = _rank_hint(r)
        click.echo(f"[{i+1}] {bar} {score:.4f}  {r.title}{path}{ranks}")

        preview = r.excerpt[:200].replace("\n", " ").strip()
        click.echo(f"     {preview}")
        click.echo()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dict(r: BM25Result | HybridResult) -> dict:
    if isinstance(r, BM25Result):
        return {
            "score":        r.score,
            "title":        r.title,
            "collection":   r.collection,
            "section_path": r.section_path,
            "chunk_type":   r.chunk_type,
            "excerpt":      r.excerpt,
        }
    return {
        "rrf_score":    r.rrf_score,
        "bm25_rank":    r.bm25_rank,
        "vector_rank":  r.vector_rank,
        "title":        r.title,
        "collection":   r.collection,
        "section_path": r.section_path,
        "chunk_type":   r.chunk_type,
        "excerpt":      r.excerpt,
    }


def _rank_hint(r: BM25Result | HybridResult) -> str:
    """Show BM25/vector rank positions for hybrid results."""
    if not isinstance(r, HybridResult):
        return ""
    bm25 = f"B{r.bm25_rank}" if r.bm25_rank else "B-"
    vec  = f"V{r.vector_rank}" if r.vector_rank else "V-"
    return f"  [{bm25} {vec}]"


def _score_bar(score: float, width: int = 5) -> str:
    """Return a small ASCII bar, e.g. '████░' for score 0.8."""
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)
