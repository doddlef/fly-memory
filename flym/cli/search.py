"""
flym.cli.search
---------------
Implements `flym search "<query>"`.

Delegates to flym.search.pipeline.run_search() which assembles all stages:
BM25 → expansion → hybrid → rerank → context expansion.

Examples
--------
    flym search "JWT token"
    flym search "how does backpropagation work"
    flym search "JWT token" --count 10
    flym search "JWT token" -c work
    flym search "JWT token" --json
    flym search "JWT token" --no-expand     # skip LLM expansion
    flym search "JWT token" --no-rerank     # skip cross-encoder reranking
"""

from __future__ import annotations

import json

import click

from flym.config import load_config
from flym.db import connect
from flym.indexer import ensure_virtual_tables
from flym.providers.ollama import OllamaEmbedding, OllamaLLM
from flym.search.pipeline import FinalResult, run_search


@click.command("search")
@click.argument("query")
@click.option("--count", "-n", default=None, type=int,
              help="Number of results (default from config).")
@click.option("--collection", "-c", default=None,
              help="Restrict to this collection.")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Output as JSON array.")
@click.option("--no-expand", is_flag=True, default=False,
              help="Skip LLM query expansion.")
@click.option("--no-rerank", is_flag=True, default=False,
              help="Skip cross-encoder reranking (faster).")
def search(
    query: str,
    count: int | None,
    collection: str | None,
    as_json: bool,
    no_expand: bool,
    no_rerank: bool,
) -> None:
    """Search the knowledge base for QUERY."""
    config = load_config()
    n = count if count is not None else config.search.default_count

    embed_provider = OllamaEmbedding(model=config.embedding.model)
    llm_provider   = None if no_expand else OllamaLLM(model=config.llm.model)

    conn = connect()
    try:
        ensure_virtual_tables(conn)
        results = run_search(
            query,
            conn,
            embed_provider,
            llm_provider,
            config,
            count          = n,
            collection     = collection,
            expand         = not no_expand,
            rerank_results = not no_rerank,
        )
    finally:
        conn.close()

    if not results:
        click.echo("No results found.")
        return

    if as_json:
        click.echo(json.dumps([_to_dict(r) for r in results], indent=2))
        return

    for i, r in enumerate(results):
        bar     = _score_bar(r.score)
        section = f"  {r.section_path}" if r.section_path else ""
        ranks   = _rank_hint(r)
        click.echo(f"[{i+1}] {bar}  {r.title}{section}{ranks}")

        preview = r.excerpt[:200].replace("\n", " ").strip()
        click.echo(f"     {preview}")
        click.echo()


def _to_dict(r: FinalResult) -> dict:
    return {
        "score":        r.score,
        "title":        r.title,
        "collection":   r.collection,
        "section_path": r.section_path,
        "chunk_type":   r.chunk_type,
        "excerpt":      r.excerpt,
        "context":      r.context,
        "bm25_rank":    r.bm25_rank,
        "vector_rank":  r.vector_rank,
    }


def _rank_hint(r: FinalResult) -> str:
    if r.bm25_rank is None and r.vector_rank is None:
        return ""
    bm25 = f"B{r.bm25_rank}" if r.bm25_rank else "B-"
    vec  = f"V{r.vector_rank}" if r.vector_rank else "V-"
    return f"  [{bm25} {vec}]"


def _score_bar(score: float, width: int = 5) -> str:
    # Rerank scores are raw logits (unbounded); normalise to [0,1] for display.
    # Typical MS-MARCO scores fall in roughly [-10, 10].
    clamped = max(-10.0, min(10.0, score))
    norm    = (clamped + 10.0) / 20.0
    filled  = round(norm * width)
    return "█" * filled + "░" * (width - filled)
