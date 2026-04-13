"""
flym.cli.index
--------------
Implements `flym index`.

Examples
--------
    flym index                    # index all documents with default model
    flym index --model mxbai-embed-large
    flym index --doc-id 3         # re-index a single document
"""

import click

from flym.config import load_config
from flym.db import connect
from flym.indexer import ensure_virtual_tables, index_all, index_document
from flym.providers.ollama import OllamaEmbedding


@click.command("index")
@click.option(
    "--model",
    default=None,
    help="Embedding model to use (overrides config). E.g. mxbai-embed-large.",
)
@click.option(
    "--doc-id",
    type=int,
    default=None,
    help="Index only this document id (from documents table).",
)
def index(model: str | None, doc_id: int | None) -> None:
    """Chunk and embed documents, writing results to the vector index."""
    config = load_config()

    if model:
        config.embedding.model = model

    provider = OllamaEmbedding(model=config.embedding.model)
    conn = connect()

    try:
        ensure_virtual_tables(conn)

        if doc_id is not None:
            result = index_document(doc_id, conn, provider, config)
            _print_result(result)
        else:
            results = index_all(conn, provider, config)
            total_chunks = sum(r["chunks_written"] for r in results)
            indexed = sum(1 for r in results if r["status"] == "indexed")
            skipped = sum(1 for r in results if r["status"] == "skipped")

            click.echo(f"Documents : {len(results)}")
            click.echo(f"  indexed : {indexed}")
            click.echo(f"  skipped : {skipped}  (already up to date)")
            click.echo(f"Chunks    : {total_chunks} written")
    finally:
        conn.close()


def _print_result(result: dict) -> None:
    icon = "+" if result["status"] == "indexed" else "="
    click.echo(
        f"[{icon}] doc_id={result.get('doc_id', '?')}  "
        f"chunks={result['chunks_written']}  ({result['status']})"
    )
