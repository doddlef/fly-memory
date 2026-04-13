"""
flym.search
-----------
Search pipeline, assembled incrementally across modules:

    Module 6  — BM25 full-text search (fast path, early return)
    Module 7  — vector KNN + RRF hybrid fusion
    Module 8  — query expansion (rephrase + HyDE)
    Module 9  — cross-encoder reranking + full pipeline
"""
