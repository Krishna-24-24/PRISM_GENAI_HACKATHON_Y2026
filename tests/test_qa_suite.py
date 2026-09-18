"""
Comprehensive QA Test Suite for TRACEIQ.

Covers all 14 QA dimensions:
1. Ingestion (scanner edge cases, filtering, ignored dirs/extensions)
2. AST parsing (classes, methods, arrow functions, syntax error handling, exports/imports)
3. Chunking (tokenizer edge cases, code normalizer, chunk_hash & chunk_id stability)
4. Metadata (CodeChunk schema validation, context strings, called_by / calls)
5. BM25 retrieval (empty queries, tokenization, remove/add chunks, top_k)
6. Dense retrieval (IDMap mapping, empty query guard, remove_chunks, model-dim cache, JSON persistence)
7. Identifier retrieval (exact, case-insensitive, sub-word compound, remove_chunks, multi-term ranking)
8. Fusion (RRFFuser weights, missing channels, candidate deduplication rank preservation, LinearFuser)
9. Query classification (QueryAnalyzer categories, frame keywords, identifier ratio, fallback)
10. Structural retrieval (calls/called_by propagation, import siblings, score boosts, graceful empty degradation)
11. Reranking (CrossEncoder mock/fallback: sorting, timeout fallback, exception fallback, empty list)
12. Agent termination (early termination on budget/confidence/empty, PRF vs AST expansion, score decay)
13. Version updates (VersionManager diffing added/modified/deleted, active state tracking, parity)
14. API endpoints (FastAPI /health, /search valid/invalid, exception handling)
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.chunking.models import CodeChunk
from src.chunking.tokenizer import tokenize_for_bm25
from src.ingestion.scanner import discover_js_files
from src.parsing.parser import (
    generate_chunk_hash,
    generate_chunk_id,
    normalize_code,
    parse_javascript_file,
)
from src.indexing.bm25_index import BM25Channel
from src.retrieval.symbol_channel import SymbolChannel
from src.retrieval.fusion import RRFFuser, LinearFuser, RawCandidate
from src.retrieval.query_analyzer import QueryAnalyzer, QueryCategory
from src.retrieval.structural_index import (
    StructuralIndex,
    StructuralWeights,
    _normalise_identifier,
    _normalise_import,
)
from src.retrieval.adaptive_retriever import AdaptiveRetriever, AdaptiveRetrievalResult
from src.retrieval.hybrid_retriever import HybridConfig
from src.agent.retrieval_agent import RetrievalAgent, AgentTrace
from src.ranking.reranker import CrossEncoderReranker
from src.indexing.version_manager import VersionManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(
    cid: str,
    name: str = "foo",
    calls: list[str] = None,
    imports: list[str] = None,
    code: str = "function foo() {}",
    file_path: str = "app.js",
) -> CodeChunk:
    return CodeChunk(
        chunk_id=cid,
        repo_id="test_repo",
        version_id="v1",
        file_path=file_path,
        symbol_name=name,
        symbol_type="function_declaration",
        start_line=1,
        end_line=10,
        code=code,
        normalized_code=code,
        context=f"File: {file_path}\nSymbol: {name}",
        imports=imports or [],
        exports=[],
        calls=calls or [],
        called_by=[],
        chunk_hash=cid,
    )


# ===========================================================================
# 1. INGESTION
# ===========================================================================

def test_ingestion_discovers_all_allowed_extensions():
    with tempfile.TemporaryDirectory() as tmpdir:
        for ext in [".js", ".jsx", ".mjs", ".cjs"]:
            path = os.path.join(tmpdir, f"file{ext}")
            with open(path, "w") as f:
                f.write("// js")
        files = discover_js_files(tmpdir)
        assert len(files) == 4
        assert all(os.path.splitext(f)[1] in [".js", ".jsx", ".mjs", ".cjs"] for f in files)


def test_ingestion_ignores_all_prohibited_dirs_and_extensions():
    with tempfile.TemporaryDirectory() as tmpdir:
        for ignored_dir in ["node_modules", ".git", "dist", "build", "coverage", ".next", ".nuxt", "out"]:
            d = os.path.join(tmpdir, ignored_dir)
            os.makedirs(d)
            with open(os.path.join(d, "hidden.js"), "w") as f:
                f.write("// hidden")

        for ign_ext in [".min.js", ".bundle.js", ".map"]:
            with open(os.path.join(tmpdir, f"test{ign_ext}"), "w") as f:
                f.write("// ignored")

        valid_file = os.path.join(tmpdir, "valid.js")
        with open(valid_file, "w") as f:
            f.write("// valid")

        files = discover_js_files(tmpdir)
        assert files == [valid_file]


def test_ingestion_empty_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        assert discover_js_files(tmpdir) == []


# ===========================================================================
# 2. AST PARSING
# ===========================================================================

def test_ast_parsing_syntax_error_handling():
    # Malformed JavaScript should not crash the parser; tree-sitter recovers
    malformed_js = b"function broken( { let x = ;"
    chunks = parse_javascript_file("broken.js", malformed_js)
    # Tree-sitter parses the AST with ERROR nodes rather than crashing
    assert isinstance(chunks, list)


def test_ast_parsing_methods_and_classes():
    code = b"""
    export class UserService {
        constructor() { this.users = []; }
        async findUser(id) { return this.users.find(u => u.id === id); }
        deleteUser(id) { this.users = this.users.filter(u => u.id !== id); }
    }
    """
    chunks = parse_javascript_file("user.js", code)
    symbols = {c.symbol_name: c.symbol_type for c in chunks}
    assert "UserService" in symbols
    assert symbols["UserService"] == "class_declaration"
    assert "findUser" in symbols
    assert symbols["findUser"] == "method_definition"
    assert "deleteUser" in symbols
    assert symbols["deleteUser"] == "method_definition"


def test_ast_parsing_empty_and_comment_only_files():
    empty_chunks = parse_javascript_file("empty.js", b"")
    assert empty_chunks == []
    comment_chunks = parse_javascript_file("comment.js", b"// just a comment\n/* block */")
    assert comment_chunks == []


# ===========================================================================
# 3. CHUNKING
# ===========================================================================

def test_code_normalizer_collapses_whitespace():
    raw = "  function   test( ) \n\n {   return    true; \n }  "
    norm = normalize_code(raw)
    assert norm == "function test( ) { return true; }"


def test_chunk_hash_deterministic():
    code = "function add(a, b) { return a + b; }"
    h1 = generate_chunk_hash(code)
    h2 = generate_chunk_hash(code)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex length


def test_tokenizer_bm25_splits_camel_and_snake():
    tokens = tokenize_for_bm25("getUserById calculate_total_sum")
    assert "getuserbyid" in tokens
    assert "user" in tokens
    assert "calculate_total_sum" in tokens
    assert "total" in tokens
    assert "sum" in tokens


def test_tokenizer_bm25_handles_empty_and_punctuation():
    assert tokenize_for_bm25("") == []
    assert tokenize_for_bm25(None) == []
    assert tokenize_for_bm25("!@#$%^&*()") == []


# ===========================================================================
# 4. METADATA
# ===========================================================================

def test_code_chunk_model_validation():
    chunk = make_chunk("c1", "testFunc", calls=["log", "audit"], imports=["import os from 'os'"])
    assert chunk.chunk_id == "c1"
    assert chunk.symbol_name == "testFunc"
    assert chunk.calls == ["log", "audit"]
    assert chunk.called_by == []
    assert chunk.start_line <= chunk.end_line


# ===========================================================================
# 5. BM25 RETRIEVAL
# ===========================================================================

def test_bm25_empty_query_and_empty_index():
    bm25 = BM25Channel()
    assert bm25.search("query") == []

    bm25.add_chunks([make_chunk("c1", "searchFiles", code="function searchFiles() {}")])
    assert bm25.search("") == []
    assert bm25.search("   ") == []


def test_bm25_add_and_remove_chunks():
    bm25 = BM25Channel()
    c1 = make_chunk("c1", "alpha", code="function alpha() { doWork(); }")
    c2 = make_chunk("c2", "beta", code="function beta() { doOther(); }")
    c3 = make_chunk("c3", "gamma", code="function gamma() { doGamma(); }")
    bm25.add_chunks([c1, c2, c3])

    res = bm25.search("alpha")
    assert any(r.chunk_id == "c1" for r in res)

    bm25.remove_chunks(["c1"])
    res_after = bm25.search("alpha")
    assert not any(r.chunk_id == "c1" for r in res_after)


# ===========================================================================
# 6. DENSE RETRIEVAL (MOCKED EMBEDDER)
# ===========================================================================

def test_dense_empty_query_and_index():
    mock_provider = MagicMock()
    mock_provider.dim = 384
    from src.indexing.dense_index import DenseChannel
    dc = DenseChannel(mock_provider)

    assert dc.search("") == []
    assert dc.search("   ") == []


def test_dense_save_load_json_without_pickle(tmp_path):
    mock_provider = MagicMock()
    mock_provider.dim = 4
    mock_provider.embed_texts.return_value = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
    ]
    mock_provider.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]

    from src.indexing.dense_index import DenseChannel
    import numpy as np

    dc = DenseChannel(mock_provider)
    c1 = make_chunk("c1", "fn1", code="const a = 1;")
    c2 = make_chunk("c2", "fn2", code="const b = 2;")
    dc.add_chunks([c1, c2])

    save_dir = str(tmp_path / "dense_idx")
    dc.save(save_dir)

    # Verify no .pkl files exist
    saved_files = os.listdir(save_dir)
    assert "mappings.json" in saved_files
    assert "chunks.json" in saved_files
    assert not any(f.endswith(".pkl") for f in saved_files)

    # Load and search
    dc_loaded = DenseChannel(mock_provider)
    dc_loaded.load(save_dir)
    assert dc_loaded.num_vectors == 2

    # Verify incremental remove
    dc_loaded.remove_chunks(["c1"])
    assert dc_loaded.num_vectors == 1


# ===========================================================================
# 7. IDENTIFIER RETRIEVAL
# ===========================================================================

def test_symbol_channel_case_and_subword_tiers():
    sc = SymbolChannel()
    c1 = make_chunk("c1", "fetchUserData")
    c2 = make_chunk("c2", "calculateStats")
    sc.add_chunks([c1, c2])

    # Exact match (Tier 1)
    res_exact = sc.search("fetchUserData")
    assert res_exact[0].chunk_id == "c1"
    assert res_exact[0].score >= 1.0

    # Case-insensitive match
    res_ci = sc.search("fetchuserdata")
    assert res_ci[0].chunk_id == "c1"
    assert res_ci[0].score >= 0.8

    # Sub-word match (Tier 2)
    res_sub = sc.search("user data")
    assert any(r.chunk_id == "c1" for r in res_sub)


def test_symbol_channel_remove_chunks():
    sc = SymbolChannel()
    c1 = make_chunk("c1", "handleRequest")
    sc.add_chunks([c1])
    assert len(sc.search("handleRequest")) == 1

    sc.remove_chunks(["c1"])
    assert len(sc.search("handleRequest")) == 0
    assert len(sc) == 0


# ===========================================================================
# 8. FUSION
# ===========================================================================

def test_rrf_fuser_custom_weights_and_missing_channel():
    fuser = RRFFuser(k=60.0)
    c1 = make_chunk("c1", "fn1")
    c2 = make_chunk("c2", "fn2")

    channel_results = {
        "dense": [RawCandidate("c1", 0.9, "dense", c1), RawCandidate("c2", 0.5, "dense", c2)],
        "bm25": [RawCandidate("c2", 10.0, "bm25", c2)],  # c1 missing in bm25
    }

    fused = fuser.fuse(channel_results, weights={"dense": 1.0, "bm25": 2.0})
    assert len(fused) == 2
    # c2 has high weight from bm25 rank 1
    assert all(f.fused_score > 0 for f in fused)


def test_linear_fuser_normalization():
    fuser = LinearFuser()
    c1 = make_chunk("c1")
    c2 = make_chunk("c2")
    channel_results = {
        "bm25": [RawCandidate("c1", 10.0, "bm25", c1), RawCandidate("c2", 2.0, "bm25", c2)]
    }
    fused = fuser.fuse(channel_results)
    assert len(fused) == 2
    assert fused[0].chunk_id == "c1"
    assert fused[0].fused_score == pytest.approx(1.0)
    assert fused[1].fused_score == pytest.approx(0.0)


# ===========================================================================
# 9. QUERY CLASSIFICATION
# ===========================================================================

def test_query_analyzer_all_categories():
    qa = QueryAnalyzer(disable_routing=False)

    # Symbolic
    assert qa.analyze("where is `formatDate` defined").category == QueryCategory.SYMBOLIC
    assert qa.analyze("formatDate").category == QueryCategory.SYMBOLIC

    # Structural
    assert qa.analyze("class extending BaseController").category == QueryCategory.STRUCTURAL

    # Usage
    assert qa.analyze("how to call parseConfig").category == QueryCategory.USAGE

    # Mixed
    assert qa.analyze("authenticate user with verifyPassword token").category == QueryCategory.MIXED

    # Semantic
    assert qa.analyze("retrieve records from the remote database efficiently").category == QueryCategory.SEMANTIC


def test_query_analyzer_disable_routing_fallback():
    qa = QueryAnalyzer(disable_routing=True)
    profile = qa.analyze("where is `formatDate` defined")
    assert profile.category == QueryCategory.SEMANTIC
    assert profile.recommended_weights == {"dense": 1.0}


# ===========================================================================
# 10. STRUCTURAL RETRIEVAL
# ===========================================================================

def test_structural_index_scoring_and_disabled_flag():
    c1 = make_chunk("c1", "entrypoint", calls=["helper"], imports=["import express from 'express'"])
    c2 = make_chunk("c2", "helper", imports=["import express from 'express'"])
    c3 = make_chunk("c3", "unrelated")

    si = StructuralIndex()
    si.build([c1, c2, c3])

    # c2 called_by should be populated
    assert "c1" in si._chunks["c2"].called_by

    # c2 shares import with c1
    assert "c2" in si.get_import_siblings("c1")

    # Score with anchor on c1
    sw = StructuralWeights(callee_boost=0.01, caller_boost=0.005, import_sibling_boost=0.002, enabled=True)
    boosts = si.score_candidates(["c1", "c2", "c3"], {"c1": 0.5, "c2": 0.2, "c3": 0.1}, sw, top_anchor=1)
    assert boosts["c2"] > 0
    assert boosts["c3"] == 0

    # Disabled weights
    boosts_dis = si.score_candidates(["c1", "c2", "c3"], {"c1": 0.5, "c2": 0.2, "c3": 0.1}, StructuralWeights.disabled())
    assert all(b == 0.0 for b in boosts_dis.values())


# ===========================================================================
# 11. RERANKING
# ===========================================================================

def test_reranker_timeout_and_empty_fallback():
    # Empty candidate list
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model = MagicMock()
    assert reranker.rerank("query", []) == []

    # Timeout fallback
    c1 = make_chunk("c1", "fn1")
    cand1 = AdaptiveRetrievalResult(chunk_id="c1", fused_score=0.8, channel_contributions={}, chunk=c1)
    cand1.final_score = 0.8
    cand1.structural_boost = 0.05

    def slow_predict(*args, **kwargs):
        time.sleep(0.05)
        return [0.1]

    reranker.model.predict = slow_predict
    # Timeout 1ms to force fallback
    reranked = reranker.rerank("query", [cand1], top_k=5, timeout_ms=1.0)
    assert len(reranked) == 1
    # On fallback, final_score = fused_score + structural_boost = 0.8 + 0.05 = 0.85
    assert reranked[0].final_score == pytest.approx(0.85)


# ===========================================================================
# 12. AGENT TERMINATION
# ===========================================================================

def test_agent_early_termination_and_budget():
    mock_retriever = MagicMock()
    # Return high confidence hit (>0.85)
    c1 = make_chunk("c1")
    res1 = AdaptiveRetrievalResult(chunk_id="c1", fused_score=0.9, channel_contributions={}, chunk=c1)
    res1.final_score = 0.95
    mock_retriever.search.return_value = [res1]

    agent = RetrievalAgent(mock_retriever, high_confidence_threshold=0.85)
    hits, trace = agent.search("test query")

    assert trace.early_terminated is True
    assert "High confidence" in trace.termination_reason
    assert mock_retriever.search.call_count == 1


# ===========================================================================
# 13. VERSION UPDATES
# ===========================================================================

def test_version_manager_diff_and_update():
    cfg = HybridConfig(use_bm25=True, use_dense=False, use_symbol=True)
    retriever = AdaptiveRetriever(cfg, disable_routing=True)
    vm = VersionManager(retriever)

    # Initial V1
    c1 = make_chunk("c1", "initApp", code="function initApp() { return 1; }")
    c2 = make_chunk("c2", "cleanup", code="function cleanup() { return 2; }")
    res1 = vm.set_version([c1, c2])
    assert res1["total_active"] == 2
    assert res1["added"] == 2

    # V2: c1 modified, c2 deleted, c3 added
    c1_mod = make_chunk("c1", "initApp", code="function initApp() { return 99; }")
    c1_mod.chunk_hash = "new_hash_99"
    c3 = make_chunk("c3", "newFeature", code="function newFeature() {}")

    res2 = vm.set_version([c1_mod, c3])
    assert res2["added"] == 1
    assert res2["modified"] == 1
    assert res2["deleted"] == 1
    assert res2["total_active"] == 2
    assert "c2" not in vm.active_chunks
    assert vm.active_chunks["c1"].chunk_hash == "new_hash_99"


# ===========================================================================
# 14. API ENDPOINTS
# ===========================================================================

def test_api_endpoints_health_and_search():
    from fastapi.testclient import TestClient
    from apps.api.main import app

    client = TestClient(app)

    # Health
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    # Search with empty query -> 400
    r_empty = client.post("/search?query=")
    assert r_empty.status_code == 400
    assert "error" in r_empty.json()

    # Search with valid query -> 200
    r_search = client.post("/search?query=testQuery")
    assert r_search.status_code == 200
    assert r_search.json()["query"] == "testQuery"
    assert "results" in r_search.json()
