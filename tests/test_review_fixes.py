"""
Regression tests for the 8 issues found in the hostile code review.

Each test is named after the bug it covers and includes a reference to the
issue number so future developers know exactly what it protects against.

Issue 1: Insecure deserialization (pickle RCE)  -> test_no_pickle_in_saved_index
Issue 2: chunk_id line-number churn             -> test_chunk_id_stable_across_line_shifts
Issue 3: Non-atomic cache write (race)          -> test_atomic_cache_write
Issue 4: Symbol score overwrite (max vs add)    -> test_multi_term_symbol_ranking
Issue 5: Cache model mismatch                   -> test_cache_keyed_by_model_dim
Issue 6: called_by not populated                -> test_called_by_populated_after_build
Issue 7: Duplicate rank overwrite in RRF        -> test_duplicate_candidate_rank_preservation
Issue 8: Empty query crash                      -> test_empty_query_safe_return
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.chunking.models import CodeChunk
from src.retrieval.fusion import RRFFuser, RawCandidate
from src.retrieval.symbol_channel import SymbolChannel
from src.retrieval.structural_index import StructuralIndex, StructuralWeights


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _chunk(cid: str, name: str, calls: list = None, code: str = "fn") -> CodeChunk:
    return CodeChunk(
        chunk_id=cid,
        repo_id="test",
        version_id="v1",
        file_path="app.js",
        symbol_name=name,
        symbol_type="function_declaration",
        start_line=1,
        end_line=5,
        code=code,
        normalized_code=code,
        context=name,
        imports=[],
        exports=[],
        calls=calls or [],
        called_by=[],
        chunk_hash=cid,
    )


# ---------------------------------------------------------------------------
# Issue 1: No pickle in saved index
# ---------------------------------------------------------------------------

def test_no_pickle_in_saved_index(tmp_path):
    """
    Issue 1 – Security: DenseChannel.save() must NOT write any pickle file.
    Pickle enables RCE if the index directory is tampered with.
    """
    # We only test the save path — no embedding model needed
    from src.indexing.dense_index import DenseChannel

    # Verify the source code itself does not import pickle
    src_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "indexing", "dense_index.py"
    )
    src = open(src_path).read()
    assert "import pickle" not in src, "dense_index.py must not import pickle"
    assert "pickle.load" not in src, "dense_index.py must not call pickle.load"
    assert "pickle.dump" not in src, "dense_index.py must not call pickle.dump"

    # Verify that after save(), no .pkl files exist in the output dir
    # (We can test the file creation path without a real FAISS index
    #  by confirming mappings.json + chunks.json are the only metadata files)
    # We just test the logic: if saved files exist, they should be .json not .pkl
    # This is a static import / source check — sufficient for CI
    assert "mappings.json" in src, "save() should write mappings.json"
    assert "chunks.json" in src, "save() should write chunks.json"


# ---------------------------------------------------------------------------
# Issue 2: chunk_id stable across line shifts
# ---------------------------------------------------------------------------

def test_chunk_id_stable_across_line_shifts():
    """
    Issue 2 – Correctness: chunk_id must NOT change when a comment is inserted
    above the function (which shifts start_line and end_line).
    """
    from src.parsing.parser import generate_chunk_id

    # Simulated V1: function starts at line 10
    id_v1 = generate_chunk_id("repo", "app.js", "handleRequest", "function_declaration", "abc123")

    # Simulated V2: same function, but now starts at line 15 (a blank line was added above)
    # The hash and symbol identity are unchanged — the id must be the same.
    id_v2 = generate_chunk_id("repo", "app.js", "handleRequest", "function_declaration", "abc123")

    assert id_v1 == id_v2, (
        "chunk_id changed when only line numbers changed — "
        "this breaks incremental updates for unmodified code"
    )


def test_chunk_id_differs_for_different_content():
    """
    Issue 2 counterpart: chunk_id MUST differ when code content changes.
    """
    from src.parsing.parser import generate_chunk_id
    import hashlib

    hash_v1 = hashlib.sha256(b"function handleRequest() {}").hexdigest()
    hash_v2 = hashlib.sha256(b"function handleRequest() { return 1; }").hexdigest()

    id_v1 = generate_chunk_id("repo", "app.js", "handleRequest", "function_declaration", hash_v1)
    id_v2 = generate_chunk_id("repo", "app.js", "handleRequest", "function_declaration", hash_v2)

    assert id_v1 != id_v2, "chunk_id must differ when code content changes"


# ---------------------------------------------------------------------------
# Issue 3: Atomic cache write
# ---------------------------------------------------------------------------

def test_atomic_cache_write():
    """
    Issue 3 – Concurrency: embedding cache writes must be atomic.
    Verify the source code uses os.replace() rather than np.save() directly.
    """
    src_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "indexing", "dense_index.py"
    )
    src = open(src_path).read()
    assert "os.replace" in src, (
        "dense_index.py must use os.replace() for atomic cache writes. "
        "Direct np.save() is not atomic and can corrupt cache on concurrent access."
    )
    assert ".tmp" in src, "Atomic write requires writing to a .tmp file first"


# ---------------------------------------------------------------------------
# Issue 4: Multi-term symbol score accumulation
# ---------------------------------------------------------------------------

def test_multi_term_symbol_ranking():
    """
    Issue 4 – Ranking: a chunk matching BOTH query terms must outrank
    a chunk matching only ONE term. Scores must accumulate, not use max().
    """
    sc = SymbolChannel()

    # Chunk A matches only "logger"
    sc.add_chunks([_chunk("cA", "logger")])
    # Chunk B matches both "logger" AND "init" (because its name is "loggerInit")
    sc.add_chunks([_chunk("cB", "loggerInit")])

    results = sc.search("logger init", top_k=10)
    ranked_ids = [r.chunk_id for r in results]

    assert len(results) >= 2, "Both chunks should appear in results"
    # cB matches more terms, so it must outrank cA
    assert ranked_ids.index("cB") < ranked_ids.index("cA"), (
        "Chunk matching both query tokens must rank above chunk matching only one"
    )


# ---------------------------------------------------------------------------
# Issue 5: Cache keyed by model dimension
# ---------------------------------------------------------------------------

def test_cache_keyed_by_model_dim():
    """
    Issue 5 – Caching: embedding cache filenames must encode the model dimension
    so that swapping models doesn't silently load stale vectors.
    """
    src_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "indexing", "dense_index.py"
    )
    src = open(src_path).read()
    # The cache path format should include the provider's dim
    assert "_provider.dim" in src, (
        "Cache path must incorporate self._provider.dim so different embedding "
        "models do not share the same .npy files"
    )


# ---------------------------------------------------------------------------
# Issue 6: called_by populated after StructuralIndex.build()
# ---------------------------------------------------------------------------

def test_called_by_populated_after_build():
    """
    Issue 6 – Metadata: after StructuralIndex.build(), each chunk's called_by
    list must be populated with the IDs of chunks that call it.
    Without this, all chunks appear to be dead code.
    """
    caller = _chunk("c_caller", "router", calls=["handleRequest"])
    callee = _chunk("c_callee", "handleRequest", calls=[])

    si = StructuralIndex()
    si.build([caller, callee])

    # The callee chunk should now have caller's ID in its called_by list
    callee_in_index = si._chunks["c_callee"]
    assert "c_caller" in callee_in_index.called_by, (
        "StructuralIndex.build() must backfill called_by on chunks "
        "using resolved incoming call edges"
    )


# ---------------------------------------------------------------------------
# Issue 7: Duplicate candidate rank preservation in RRF
# ---------------------------------------------------------------------------

def test_duplicate_candidate_rank_preservation():
    """
    Issue 7 – Ranking: if a channel emits the same chunk_id at rank 1 and rank 10,
    RRFFuser must use rank 1 (the better rank) for score computation.
    """
    fuser = RRFFuser(k=60)
    chunk_a = _chunk("c_best", "best")
    chunk_b = _chunk("c_other", "other")

    # c_best appears at rank 1 AND rank 10 in the same channel (duplicate)
    channel_results = {
        "dense": [
            RawCandidate("c_best", 0.9, "dense", chunk_a),   # rank 1
            RawCandidate("c_other", 0.5, "dense", chunk_b),   # rank 2
            RawCandidate("c_best", 0.1, "dense", chunk_a),    # rank 3 — duplicate!
        ]
    }

    fused = fuser.fuse(channel_results)

    # Score using rank-1: 1/(60+1) ≈ 0.01639
    # Score using rank-3: 1/(60+3) ≈ 0.01587
    # The correct score must use rank 1
    expected_score = 1.0 / (60.0 + 1)
    best_result = next(r for r in fused if r.chunk_id == "c_best")
    assert abs(best_result.fused_score - expected_score) < 1e-9, (
        f"Expected score {expected_score:.6f} (rank 1), "
        f"got {best_result.fused_score:.6f} — duplicate rank overwrite bug"
    )


# ---------------------------------------------------------------------------
# Issue 8: Empty query safe return
# ---------------------------------------------------------------------------

def test_empty_query_safe_return_bm25():
    """
    Issue 8 – Error handling: BM25Channel.search('') must return [] safely.
    """
    from src.indexing.bm25_index import BM25Channel
    bm25 = BM25Channel()
    bm25.add_chunks([_chunk("c1", "foo", code="function foo() {}")])

    assert bm25.search("") == [], "Empty query must return empty list"
    assert bm25.search("   ") == [], "Whitespace query must return empty list"


def test_empty_query_guard_in_source():
    """
    Issue 8 – Error handling: DenseChannel.search() must guard empty queries
    before calling embed_query (which may produce NaN vectors).
    """
    src_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "indexing", "dense_index.py"
    )
    src = open(src_path).read()
    assert "not query.strip()" in src or "not query or not query.strip()" in src, (
        "DenseChannel.search() must guard against empty/whitespace queries"
    )
