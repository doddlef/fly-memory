"""
flym — a lightweight personal document base.

Modules (build order):
  1. config, db          — scaffold + schema
  2. vault, ingestion    — flym add
  3. chunker             — semantic chunking
  4. providers/          — embedding + LLM protocols
  5. indexer             — chunk → embed → store
  6. search/bm25         — BM25 fast path
  7. search/vector,      — vector search + RRF hybrid
     search/hybrid
  8. search/expansion    — query expansion (rephrase + HyDE)
  9. search/rerank,      — cross-encoder + full pipeline
     search/pipeline
 10. collections,        — collection registry + full CLI
     cli/
 11. cache               — unified cache + GC
 12. chunker_code        — tree-sitter code chunking (optional)
"""
