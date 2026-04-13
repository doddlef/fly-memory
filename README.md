# flym

A lightweight personal document base that lets you search your notes and documents by keyword or natural language from the terminal.

```
flym search "how does JWT validation work"
flym search "chunking algorithm"
```

---

## How it works

```
add document → chunk → embed → store
                                  ↓
query → BM25 → [expand] → hybrid (BM25 + vector) → rerank → results
```

1. **Chunking** — documents are split at semantic breakpoints (headings, paragraphs, code fences) rather than fixed character counts, keeping chunks coherent.
2. **Embedding** — each chunk is embedded by a local Ollama model into a float vector and stored in SQLite via `sqlite-vec`.
3. **BM25 fast path** — every search runs BM25 first. If the top result is dominant (score ≥ 0.85 and gap ≥ 0.15), it returns immediately without touching the embedding model.
4. **Query expansion** — keyword queries are rephrased with synonyms; question queries use HyDE (a hypothetical answer is generated and embedded instead of the question itself).
5. **Hybrid search** — BM25 and vector KNN results are fused with Reciprocal Rank Fusion, which ranks by position rather than raw score.
6. **Reranking** — a cross-encoder (`ms-marco-MiniLM-L-6-v2`) rescores the top candidates by reading query and chunk together, catching nuances the bi-encoder misses.

---

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) running locally

```bash
brew install ollama
ollama serve
ollama pull nomic-embed-text   # embedding model (~274 MB)
ollama pull qwen3.5:4b         # or any chat model for query expansion
```

---

## Installation

```bash
git clone <repo>
cd memory
pip install -e .
```

Verify:

```bash
flym --help
flym db-check
```

---

## Quick start

```bash
# Add a single file
flym add ~/Documents/notes/auth.md

# Add an entire directory
flym add ~/Documents/notes/

# Index (embed) all added documents
flym index

# Search
flym search "JWT token validation"
flym search "how does backpropagation work"
```

---

## Commands

### `flym add <path> [path ...]`

Add one or more files or directories to the knowledge base.

```bash
flym add note.md                        # single file → default collection
flym add note.md other.md               # multiple files
flym add ~/Documents/notes/             # entire directory (*.md)
flym add ~/notes/ -c work               # into a named collection
flym add ~/notes/ --pattern '**/*.txt'  # different file type
flym add note.md --link                 # reference in place, do not copy
```

### `flym index`

Chunk and embed all unindexed (or stale) documents.

```bash
flym index                              # index everything
flym index --doc-id 3                   # re-index one document
flym index --model mxbai-embed-large    # use a different embedding model
```

### `flym search <query>`

Search the knowledge base.

```bash
flym search "JWT token"
flym search "how does attention work"   # triggers HyDE expansion
flym search "JWT" -c work              # restrict to a collection
flym search "JWT" -n 10               # more results
flym search "JWT" --no-expand          # skip LLM expansion (faster)
flym search "JWT" --no-rerank          # skip cross-encoder (faster)
flym search "JWT" --json               # machine-readable output
```

Output format:
```
  (hybrid+hyde)

[1] █████  Authentication Guide  JWT Tokens  [B2 V1]
     Always validate the signature before trusting the payload...

[2] ████░  Authentication Guide  Validation  [B1 V3]
     Never decode without verifying the signature...
```

The `[B2 V1]` hint shows the result's rank in the BM25 list (B) and vector list (V) before fusion.

### `flym collections`

Manage named collections (named groups of documents).

```bash
flym collections add work ~/Documents/work-notes
flym collections add personal ~/notes --update-cmd 'git pull'
flym collections list
flym collections update work            # run update-cmd + re-ingest changed files
```

### `flym remove` / `flym purge`

```bash
flym remove auth.md                     # soft-delete (invisible to search)
flym remove auth.md -c work            # if the path exists in multiple collections
flym purge --dry-run                    # preview what would be deleted
flym purge --older-than 7d             # hard-delete + GC after 7 days
flym purge --older-than 0d             # hard-delete immediately
```

### `flym watch <directory>`

Watch a directory and automatically ingest + index changed files.

```bash
flym watch ~/Documents/notes           # foreground watcher, Ctrl-C to stop
flym watch ~/notes -c work
flym watch ~/notes --pattern '*.md'
```

### `flym db-check`

Show row counts for all tables — useful for verifying the database is healthy.

```bash
flym db-check
```

---

## Configuration

Settings live in `~/.flym/config.json`. All keys are optional — defaults work out of the box.

```json
{
  "embedding": {
    "provider": "ollama",
    "model": "nomic-embed-text",
    "dimensions": 768
  },
  "llm": {
    "provider": "ollama",
    "model": "qwen3.5:4b"
  },
  "chunking": {
    "target_chars": 1500,
    "overlap_chars": 225,
    "min_chars": 100
  },
  "search": {
    "default_count": 5,
    "bm25_threshold": 0.85,
    "bm25_gap": 0.15
  }
}
```

---

## Data layout

```
~/.flym/
  flym.db          # SQLite database (all metadata, vectors, FTS index)
  vault/           # copied document files (when not using --link)
    default/
    work/
  config.json      # optional configuration overrides
```

### Database tables

| Table | Contents |
|---|---|
| `content` | Raw document text, keyed by SHA-256 hash (one row per unique body) |
| `documents` | Document metadata: title, collection, path, hash, active flag |
| `store_collections` | Collection registry: name → directory path |
| `chunks` | Chunk metadata: pos, len, section\_path, chunk\_type, seq |
| `vectors_vec` | Embedding vectors (sqlite-vec virtual table) |
| `documents_fts` | Full-text search index (FTS5 virtual table) |
| `cache` | LLM expansion cache (permanent) and search result cache (TTL) |

---

## Architecture

```
flym/
  config.py          # Pydantic config, ~/.flym/config.json
  db.py              # SQLite connection, schema migrations
  ingestion.py       # SHA-256 hashing, content-addressable storage
  vault.py           # file copy / link mode
  chunker.py         # semantic breakpoint chunking
  indexer.py         # chunk → embed → store pipeline
  collections.py     # collection registry and bulk update
  providers/
    base.py          # EmbeddingProvider, LLMProvider Protocols
    ollama.py        # Ollama SDK implementations
  search/
    bm25.py          # FTS5 BM25 search + score normalisation
    vector.py        # sqlite-vec KNN search
    hybrid.py        # RRF fusion of BM25 + vector
    expansion.py     # query classification, rephrase, HyDE
    rerank.py        # cross-encoder reranking
    pipeline.py      # full pipeline: expand → hybrid → rerank → context
  cli/
    add.py           # flym add
    index.py         # flym index
    search.py        # flym search
    collections.py   # flym collections add/list/update
    remove.py        # flym remove, flym purge
    watch.py         # flym watch
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `click` | CLI framework |
| `pydantic` | Configuration validation |
| `python-frontmatter` | YAML frontmatter parsing |
| `ollama` | Local LLM and embedding via Ollama |
| `sqlite-vec` | Vector KNN search extension for SQLite |
| `sentence-transformers` | Cross-encoder reranking |
| `watchfiles` | Filesystem watching |
