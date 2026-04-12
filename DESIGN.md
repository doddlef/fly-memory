# flym — Design Document

A lightweight personal document base built for search. Drop in your markdown notes,
query them in plain language or by keyword — from the terminal.

---

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Storage Model](#storage-model)
4. [Database Schema](#database-schema)
5. [Chunking Algorithm](#chunking-algorithm)
6. [Search Pipeline](#search-pipeline)
7. [Query Expansion](#query-expansion)
8. [CLI Reference](#cli-reference)
9. [Configuration](#configuration)
10. [Provider Interface](#provider-interface)
11. [Future Work](#future-work)

---

## Overview

`flym` is a personal knowledge base CLI with a multi-stage search pipeline. It indexes
markdown documents into a local SQLite database, stores vector embeddings locally, and
retrieves results using a cascade of search strategies — from fast exact matching to
semantic reranking.

Design goals:
- **No server** — SQLite + local embedding model, runs entirely offline
- **Fast** — BM25 early exit, semantic query cache, metadata pre-filtering
- **Portable** — entire knowledge base lives in one directory
- **Markdown-first** — YAML frontmatter parsed as metadata, heading hierarchy preserved

---

## Core Concepts

### Vault

A managed directory (`~/.flym/vault/` by default) where all imported documents live.
The vault plus the SQLite database form a single portable unit — zip it, move it, or
back it up as one.

Documents can be added in two modes:

```bash
flym add note.md          # copy into vault (default)
flym add note.md --link   # reference original path, do not copy
```

Imported documents are owned by flym. Linked documents remain in place; if moved or
deleted externally, the reference breaks (detected via `content_hash` on next access).

### Collections

Named groupings of documents, each mapped to a directory on disk. Registered in
`store_collections`. A document belongs to exactly one collection.

```bash
flym search "JWT" -c work          # search only the 'work' collection
flym search "JWT"                  # search all default-included collections
```

Collections support per-collection glob patterns, ignore rules, and an optional
`update_command` (e.g. `git pull`) to sync before re-indexing.

### Content-Addressable Storage

Document text is stored once in the `content` table, keyed by SHA-256 hash. The
`documents` table holds metadata and references content by hash. This means:

- Duplicate files are stored only once
- Updating a document swaps the hash FK; old content is garbage-collected
- Content is immutable — a given hash always refers to the same text

### Chunks

Each document is split into overlapping chunks for retrieval. Chunks do **not** store
text — they store `(pos, len)` offsets into `content.doc`. Chunk text is recovered via
`substr(content.doc, pos, len)`.

---

## Storage Model

```
~/.flym/
  config.json       # provider config, vault path, chunking params
  flym.db           # SQLite database (schema below)
  vault/
    <collection>/
      document.md
```

`flym.db` contains everything: document metadata, chunk offsets, embeddings, FTS index,
and caches. The entire knowledge base is self-contained.

---

## Database Schema

### `content`

Immutable text store. One row per unique document body.

```sql
CREATE TABLE content (
  hash       TEXT PRIMARY KEY,   -- SHA-256 of full document text
  doc        TEXT NOT NULL,      -- full document body (source of truth)
  created_at TEXT NOT NULL
);
```

**Garbage collection:** orphaned rows (no document references the hash) are cleaned up
lazily. On 1% of queries and on every `flym purge`, run:

```sql
DELETE FROM content
WHERE NOT EXISTS (
  SELECT 1 FROM documents WHERE documents.hash = content.hash
);
```

`NOT EXISTS` is used over `NOT IN` to avoid NULL edge cases and for better performance.

---

### `store_collections`

Collection registry. Maps logical names to filesystem roots.

```sql
CREATE TABLE store_collections (
  name               TEXT PRIMARY KEY,
  path               TEXT NOT NULL,           -- absolute root on disk
  pattern            TEXT DEFAULT '**/*.md',  -- glob for files to index
  ignore_patterns    TEXT,                    -- JSON array of exclusion globs
  include_by_default INTEGER DEFAULT 1,       -- 0 = skip unless -c specified
  update_command     TEXT,                    -- e.g. "git -C /path pull"
  context            TEXT                     -- JSON: path prefix → hint
);
```

**Field notes:**

- `pattern` — a collection of code notes might use `**/*.md,**/*.txt`
- `ignore_patterns` — e.g. `[".git/**", "_drafts/**", "node_modules/**"]`
- `include_by_default = 0` — useful for archive/scratch collections that should not
  pollute everyday search results
- `update_command` — `flym update <name>` runs this command then re-indexes dirty files
- `context` — injects domain hints into LLM query expansion per path prefix:
  ```json
  { "auth/": "authentication and security notes", "ml/": "machine learning notes" }
  ```

---

### `documents`

One row per file per collection. Holds mutable metadata; immutable text is in `content`.

```sql
CREATE TABLE documents (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  collection  TEXT NOT NULL REFERENCES store_collections(name),
  path        TEXT NOT NULL,         -- path relative to collection root
  title       TEXT NOT NULL,         -- extracted from YAML frontmatter or first heading
  hash        TEXT NOT NULL REFERENCES content(hash),
  metadata    TEXT,                  -- JSON: parsed YAML frontmatter
  active      INTEGER NOT NULL DEFAULT 1,
  deleted_at  TEXT,                  -- NULL = live, timestamp = soft-deleted
  created_at  TEXT NOT NULL,
  modified_at TEXT NOT NULL,         -- file mtime at last index
  UNIQUE(collection, path)
);
```

**Soft delete:** `active = 0` and `deleted_at` set on `flym remove`. Hard delete via
`flym purge`. Both fields are kept — `active` for fast index filtering, `deleted_at`
for `flym purge --older-than 30d`.

**Update detection:** on re-scan, compare file mtime against `modified_at`. If changed,
re-hash and compare against `documents.hash`. Re-index only if hash differs (catches
mtime-only changes like touch).

---

### `chunks`

One row per chunk. Text is not stored here — use `substr(content.doc, pos, len)`.

```sql
CREATE TABLE chunks (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,  -- rowid for FTS5 and vectors
  content_hash TEXT NOT NULL REFERENCES content(hash),
  seq          INTEGER NOT NULL,         -- chunk index within document (0-based)
  pos          INTEGER NOT NULL,         -- character offset in content.doc
  len          INTEGER NOT NULL,         -- character length
  model        TEXT NOT NULL,            -- embedding model used (for re-embed detection)
  section_path TEXT,                     -- e.g. "Installation > MacOS > Homebrew"
  chunk_type   TEXT DEFAULT 'prose',     -- 'prose' | 'code' | 'mixed'
  language     TEXT,                     -- null, or 'python', 'typescript', etc.
  embedded_at  TEXT NOT NULL,
  UNIQUE(content_hash, seq)
);
```

**Field notes:**

- `model` — when the embedding model changes, find chunks to re-embed via
  `WHERE model != current_model`
- `section_path` — breadcrumb derived from the heading parse tree. Used in result
  display and boosts relevance scoring for section-matched queries
- `chunk_type` — controls which embedding model is used (`prose` → text model,
  `code` → code model) and result display (code chunks rendered with syntax highlighting)
- `language` — extracted from fenced code block annotation (` ```python `). Used to
  select the correct tree-sitter parser and to support `--language` filtering

---

### `cache`

Unified cache table for both LLM responses and search results.

```sql
CREATE TABLE cache (
  hash       TEXT PRIMARY KEY,   -- input hash (prompt hash or query+filters hash)
  type       TEXT NOT NULL,      -- 'llm' | 'search'
  result     TEXT NOT NULL,      -- serialized JSON result
  created_at TEXT NOT NULL,
  expires_at TEXT                -- NULL = permanent (LLM), timestamp = TTL (search)
);

CREATE INDEX idx_cache_expires ON cache(expires_at) WHERE expires_at IS NOT NULL;
```

**LLM cache (`type='llm'`):** caches query expansion and HyDE responses. `expires_at`
is NULL — LLM responses are deterministic, so no expiry is needed. The same prompt
always produces the same expansion.

**Search cache (`type='search'`):** caches full search results. `expires_at` is set to
`created_at + 24h`. Eviction is probabilistic: on 1% of queries, delete expired rows
(capped at 100 rows per cleanup to bound latency).

---

### Virtual Tables

```sql
-- Document-level full-text search (BM25 fast path)
CREATE VIRTUAL TABLE documents_fts USING fts5(
  filepath,        -- "collection/path"
  title,
  body,            -- full document text (copied from content.doc)
  tokenize='porter unicode61'
);
-- Kept in sync via triggers: documents_ai, documents_au, documents_ad

-- Vector embeddings (chunk-level)
CREATE VIRTUAL TABLE vectors_vec USING vec0(
  chunk_id  INTEGER PRIMARY KEY,   -- FK -> chunks.id
  embedding FLOAT[768]
);
```

---

### Indexes

```sql
CREATE INDEX idx_documents_collection ON documents(collection, active);
CREATE INDEX idx_documents_hash       ON documents(hash);
CREATE INDEX idx_chunks_content       ON chunks(content_hash, seq);
CREATE INDEX idx_cache_expires        ON cache(expires_at) WHERE expires_at IS NOT NULL;
```

---

### Schema Diagram

```
store_collections
      │
      │ name
      ▼
  documents ──── hash ────► content ◄──── hash ──── chunks
      │                                               │
      │ id                                            │ id
      ▼                                               ▼
  documents_fts                                  vectors_vec
  (virtual, FTS5)                               (virtual, vec0)

  cache
  (llm + search results)
```

---

## Chunking Algorithm

### Overview

Documents are split at **semantic breakpoints** rather than fixed token boundaries.
When the chunker reaches a target size boundary, it looks back within the current chunk
for the highest-scoring breakpoint.

### Breakpoint Scoring

Each candidate break position is scored by type and proximity to the target boundary:

```
score = breakpoint.score × (1 - normalizedDistance² × 0.7)
```

Where `normalizedDistance` is `(targetPos - breakPos) / targetPos`, ranging from 0
(at target) to 1 (at chunk start). The quadratic penalty strongly prefers nearby
breaks but allows a high-quality heading to override a nearby low-quality boundary.

Scores use a 1–100 integer scale. Headings are spread wide so they decisively beat
lower-quality breaks. When two patterns match at the same position, **max-score-wins**
— no score accumulation.

**Break patterns (ordered by score, more specific first):**

```typescript
export const BREAK_PATTERNS: [RegExp, number, string][] = [
  [/\n#{1}(?!#)/g,               100, 'h1'],       // # but not ##
  [/\n#{2}(?!#)/g,                90, 'h2'],       // ## but not ###
  [/\n#{3}(?!#)/g,                80, 'h3'],       // ### but not ####
  [/\n#{4}(?!#)/g,                70, 'h4'],       // #### but not #####
  [/\n(?:---|\*\*\*|___)\s*\n/g,  65, 'hr'],       // horizontal rule
  [/\n```/g,                      65, 'codeblock'],// code block boundary
  [/\n#{5}(?!#)/g,                60, 'h5'],       // ##### but not ######
  [/\n#{6}(?!#)/g,                50, 'h6'],       // ######
  [/[.!?]\s+(?=[A-Z])/g,          30, 'sentence'], // sentence boundary
  [/\n\n+/g,                      20, 'blank'],    // paragraph boundary
  [/\n[-*]\s/g,                    5, 'list'],     // unordered list item
  [/\n\d+\.\s/g,                   5, 'numlist'],  // ordered list item
  [/\n/g,                          1, 'newline'],  // minimal break
];
```

**Score rationale:**

| Type | Score | Notes |
|---|---|---|
| `h1` | 100 | Strongest semantic boundary |
| `h2` | 90 | |
| `h3` | 80 | |
| `h4` | 70 | |
| `hr` / `codeblock` | 65 | Between h4 and h5 — structural but not semantic headings |
| `h5` | 60 | |
| `h6` | 50 | |
| `sentence` | 30 | Fallback for dense paragraphs with no blank lines |
| `blank` | 20 | Paragraph boundary |
| `list` / `numlist` | 5 | Weak — list items are part of a unit, not boundaries |
| `newline` | 1 | Absolute minimum fallback |

### Hard Rules

- **Never split inside a fenced code block** — track fence open/close state before
  applying break patterns. Any break point found at a position where the fence count
  is odd (inside a block) is discarded:
  ```typescript
  function isInsideCodeBlock(text: string, pos: number): boolean {
    const fences = (text.slice(0, pos).match(/\n```/g) ?? []).length;
    return fences % 2 === 1;
  }
  ```
- **Never split inside a table** — only break at row boundaries
- **YAML frontmatter** — strip before applying patterns; the closing `---` would
  otherwise match the `hr` pattern:
  ```typescript
  const FRONTMATTER = /^---\n[\s\S]*?\n---\n/;
  const body = text.replace(FRONTMATTER, '');
  ```

### Code Blocks: tree-sitter Integration

For fenced code blocks with a known language annotation, tree-sitter identifies
additional breakpoints at AST node boundaries:

| Node type | Score |
|---|---|
| Class definition | 95 |
| Function/method definition | 85 |
| Import/require block | 75 |

These are merged with regex breakpoints using **max-score-wins**: if both sources
identify a breakpoint at the same position, the higher score wins. tree-sitter takes
precedence on ties (syntactically authoritative).

For unlabeled code blocks, the block boundary itself is a single breakpoint (score 65).
For code blocks exceeding the target chunk size, tree-sitter splits at top-level AST
node boundaries; never mid-statement.

### Overlap

After each chunk boundary is determined, the next chunk begins at:

```typescript
charPos = endPos - overlapChars;
const lastChunkPos = chunks.at(-1)!.pos;
if (charPos <= lastChunkPos) {
  charPos = endPos;   // no room for overlap — start exactly at end
} else {
  while (charPos < endPos && text[charPos] !== ' ') charPos++;  // snap to word
}
```

Default overlap: 15% of target chunk size.

### Default Parameters

| Parameter | Default | Notes |
|---|---|---|
| `target_chars` | 1500 | ~500 tokens at 3 chars/token |
| `overlap_chars` | 225 | 15% of target |
| `min_chars` | 100 | Discard chunks shorter than this |

---

## Search Pipeline

Queries pass through a progressive pipeline, exiting early when confidence is
sufficiently high.

```
Query + filters
     │
     ▼
[0] Metadata pre-filter
     │   Filter document IDs by collection, date, file type
     │   All downstream steps operate only within this set
     ▼
[1] BM25 fast path
     │   FTS5 full-text search on documents_fts
     │   Normalize scores: score / max_score_in_results
     │   Early return if: topScore >= 0.85 AND (topScore - secondScore) >= 0.15
     ▼
[2] Query expansion
     │   Classify query type (see below)
     │   Keyword query  → LLM rephrasing (synonyms, alternate terms)
     │   Question query → HyDE (generate hypothetical answer, embed it)
     ▼
[3] Hybrid search
     │   BM25 on expanded terms   ┐
     │   Vector search            ├─ fused via RRF
     │   (original + HyDE embed)  ┘
     │
     │   RRF: score(d) = Σ 1 / (k + rank_i(d)),  k = 60
     ▼
[4] Cross-encoder rerank
     │   Model: cross-encoder/ms-marco-MiniLM-L-6-v2
     │   Re-scores top (3 × count) candidates
     ▼
[5] Context expansion
     │   Fetch chunk[seq-1] and chunk[seq+1] for each result
     │   Provides richer surrounding context in output
     ▼
Return top `count` results
```

### Top-K Ladder

| Stage | Candidates |
|---|---|
| BM25 candidates | 5 × count |
| After hybrid / RRF | 3 × count |
| After rerank | count |
| Returned to user | count (default: 5) |

### BM25 Early Return

The early-return condition uses **normalized** BM25 scores (raw scores vary by corpus
size and IDF values and must not be compared as absolute values):

```python
normalized = [s / max(scores) for s in raw_scores]
if normalized[0] >= 0.85 and (normalized[0] - normalized[1]) >= 0.15:
    return early  # high confidence exact match
```

### Query Classification

| Signal | Strategy |
|---|---|
| Contains `?` or starts with how/what/why/when | HyDE |
| Code snippet or camelCase/snake_case terms | No expansion; code-aware BM25 |
| Default | LLM rephrase |

### Semantic Query Cache

Before the pipeline, normalize and hash the query + filters. Check `cache` table for
a recent result (`type='search'`). On hit, increment `hit_count` and return.

Probabilistic eviction: 1% of queries trigger cleanup of expired search cache rows
(LIMIT 100 per cleanup). Same trigger also GCs orphaned `content` rows.

---

## Query Expansion

### LLM Rephrase

For keyword queries, the LLM generates synonyms and alternate phrasings:

```
Input:  "JWT token"
Output: ["JSON Web Token", "bearer token", "JWT authentication", "token validation"]
```

The expanded terms are combined with the original and fed into the BM25 + vector search.

### HyDE (Hypothetical Document Embeddings)

For natural language questions, instead of embedding the query directly, the LLM
generates a hypothetical answer document and that is embedded:

```
Input:  "how does backpropagation work"
HyDE:   "Backpropagation is an algorithm for training neural networks by computing
         gradients of the loss function with respect to weights using the chain rule..."
```

Both the original query embedding and the HyDE embedding are used for vector search.
Results from both are merged before RRF fusion.

HyDE bridges the vocabulary gap between short queries and longer document chunks.

---

## CLI Reference

### Adding Documents

```bash
flym add note.md                    # import into vault (copy)
flym add note.md --link             # reference original, do not copy
flym add ~/notes/ --recursive       # import a directory
flym watch ~/notes/                 # auto-index on file changes (foreground)
```

### Searching

```bash
flym search "JWT token"
flym search "how does backpropagation work"

# Filters
flym search "JWT" -c work                    # specific collection
flym search "JWT" -c work/projects           # nested collection path
flym search "JWT" --since 2w                 # modified in last 2 weeks
flym search "JWT" --since 2025-01-01         # absolute date
flym search "JWT" --language typescript      # code chunks only
flym search "JWT" --count 10                 # return 10 results (default: 5)
flym search "JWT" --json                     # JSON output
```

### Filter Syntax

| Syntax | Meaning |
|---|---|
| `-c work` | collection = work |
| `--since 2w` | modified within last 2 weeks |
| `--language ts` | chunk language = typescript |
| `--count N` | return N results |
| `--json` | output as JSON |

### Document Management

```bash
flym list                          # list all documents
flym list -c work                  # list documents in collection
flym remove note.md                # soft delete
flym purge                         # hard delete all soft-deleted documents
flym purge --older-than 30d        # hard delete soft-deleted docs older than 30 days
flym purge --doc <id>              # hard delete a specific document
```

### Collection Management

```bash
flym collections                   # list all collections
flym collections add work ~/Documents/work-notes
flym update work                   # run update_command then re-index dirty files
```

### Output Format

**Plain text (default):**
```
Found 3 results for "JWT token"  (12ms)

1. [0.94]  auth/jwt-guide.md  §  Authentication > JWT
   ...tokens are signed using RSA or HMAC. The header contains
   the algorithm and token type...

2. [0.87]  work/api-notes.md  §  API Design > Security
   ...JWT tokens must be validated on every request. Set expiry
   to a short window and rotate refresh tokens...
```

**JSON (`--json`):**
```json
[
  {
    "score": 0.94,
    "matched_by": "bm25",
    "document": { "id": 1, "title": "jwt-guide", "path": "auth/jwt-guide.md" },
    "chunk": { "id": 42, "section_path": "Authentication > JWT", "content": "..." }
  }
]
```

`matched_by` values: `"bm25"` (early return), `"hybrid"`, `"vector"`.

---

## Configuration

Stored in `~/.flym/config.json`. Can also be partially stored in `store_config(key,
value)` inside the database for portability.

```json
{
  "vault_path": "~/.flym/vault",
  "db_path":    "~/.flym/flym.db",

  "embedding": {
    "provider":   "ollama",
    "model":      "nomic-embed-text",
    "dimensions": 768
  },
  "llm": {
    "provider": "ollama",
    "model":    "llama3.2"
  },

  "chunking": {
    "target_chars":  1500,
    "overlap_chars": 225,
    "min_chars":     100
  },

  "search": {
    "default_count":           5,
    "bm25_threshold":          0.85,
    "bm25_gap":                0.15,
    "cache_ttl_hours":         24,
    "cache_evict_probability": 0.01,
    "cache_evict_limit":       100
  }
}
```

---

## Provider Interface

Embedding and LLM providers are pluggable via a protocol interface. Start with Ollama
for local/offline use; switch to API providers by changing config.

```python
class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dimensions(self) -> int: ...

class LLMProvider(Protocol):
    def complete(self, prompt: str) -> str: ...
```

**Built-in implementations:**

| Class | Provider | Notes |
|---|---|---|
| `OllamaEmbedding` | Ollama | Default, fully offline |
| `OpenAIEmbedding` | OpenAI API | `text-embedding-3-small` |
| `OllamaLLM` | Ollama | Default, fully offline |
| `OpenAILLM` | OpenAI API | |
| `AnthropicLLM` | Anthropic API | |

To add a new provider, implement the protocol and register it in the provider factory.
No other changes required.

**Re-embedding after model change:** chunks store `model` (the embedding model ID used).
When the configured model changes, find stale chunks via:

```sql
SELECT * FROM chunks WHERE model != ?
```

Re-embed and update `vectors_vec` in batches. The `content_hash + seq` uniquely
identifies each chunk for upsert.

---

## Future Work

The following features are intentionally deferred. The schema supports them without
breaking changes.

### Tags

Add two tables:

```sql
CREATE TABLE tags (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
CREATE TABLE document_tags (
  document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
  tag_id      INTEGER REFERENCES tags(id)      ON DELETE CASCADE,
  PRIMARY KEY (document_id, tag_id)
);
```

Tags apply to documents and are inherited by all chunks at search time via join — no
changes to the chunks table. CLI filter syntax: `-t auth,security` (OR), `-t auth+security` (AND).

### Chunk-Level FTS5

Add a `chunks_fts` virtual table for more precise BM25 matching within documents.
Use `chunks.id` as rowid. Populate via trigger on chunk insert.

### `store_config` Table

```sql
CREATE TABLE store_config (key TEXT PRIMARY KEY, value TEXT);
```

Stores `schema_version`, `global_context`, and other runtime config inside the
database. Makes the database fully self-describing without an external config file.

### `flym watch` Background Daemon

Currently `flym watch` runs as a foreground process. Future: persistent background
daemon with a socket interface, so indexing happens continuously without a terminal.
