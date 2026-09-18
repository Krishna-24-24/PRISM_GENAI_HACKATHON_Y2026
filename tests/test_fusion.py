"""
Unit tests for fusion strategies and HybridRetriever.
These tests run fast (no model inference) using mock/stub channels.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.retrieval.fusion import FusedCandidate, LinearFuser, RRFFuser, RawCandidate
from src.retrieval.symbol_channel import SymbolChannel, _split_identifier
from src.chunking.models import CodeChunk


def _rc(cid, score, channel="ch"):
    return RawCandidate(chunk_id=cid, score=score, channel=channel)


def _chunk(id_, name="fn"):
    return CodeChunk(
        chunk_id=id_, repo_id="t", version_id="1", file_path="t.js",
        symbol_name=name, symbol_type="function",
        start_line=1, end_line=3, code=f"function {name}(){{}}", normalized_code="",
        context="", imports=[], exports=[], calls=[], called_by=[], chunk_hash=id_,
    )


# ---------------------------------------------------------------------------
# _split_identifier
# ---------------------------------------------------------------------------

def test_split_camel_case():
    assert "http" in _split_identifier("handleHttpRequest")
    assert "handle" in _split_identifier("handleHttpRequest")
    assert "request" in _split_identifier("handleHttpRequest")


def test_split_snake_case():
    tokens = _split_identifier("get_user_by_id")
    assert "get" in tokens
    assert "user" in tokens


def test_split_single_word():
    assert _split_identifier("sort") == ["sort"]


# ---------------------------------------------------------------------------
# SymbolChannel
# ---------------------------------------------------------------------------

@pytest.fixture
def symbol_index():
    ch = SymbolChannel()
    chunks = [
        _chunk("c1", "bubbleSort"),
        _chunk("c2", "handleHttpRequest"),
        _chunk("c3", "getUserById"),
    ]
    ch.add_chunks(chunks)
    return ch


def test_symbol_exact_match(symbol_index):
    results = symbol_index.search("bubbleSort", top_k=5)
    assert results[0].chunk_id == "c1"
    assert results[0].score == 1.0


def test_symbol_case_insensitive(symbol_index):
    results = symbol_index.search("bubblesort", top_k=5)
    ids = [r.chunk_id for r in results]
    assert "c1" in ids


def test_symbol_subword_match(symbol_index):
    results = symbol_index.search("http request", top_k=5)
    ids = [r.chunk_id for r in results]
    assert "c2" in ids


def test_symbol_no_anonymous_chunks():
    ch = SymbolChannel()
    ch.add_chunks([_chunk("c1", "anonymous")])
    results = ch.search("anonymous", top_k=5)
    assert results == []


# ---------------------------------------------------------------------------
# RRFFuser
# ---------------------------------------------------------------------------

@pytest.fixture
def rrf():
    return RRFFuser(k=60)


def test_rrf_single_channel(rrf):
    results = rrf.fuse({"ch": [_rc("a", 0.9), _rc("b", 0.5)]})
    ids = [r.chunk_id for r in results]
    assert ids[0] == "a"   # rank 1 → higher RRF score


def test_rrf_two_channels_consensus(rrf):
    """Doc ranked #1 in both channels should win."""
    r = rrf.fuse({
        "bm25":  [_rc("x", 1.0), _rc("y", 0.5)],
        "dense": [_rc("x", 0.9), _rc("z", 0.4)],
    })
    assert r[0].chunk_id == "x"


def test_rrf_channel_contributions_present(rrf):
    r = rrf.fuse({
        "bm25":  [_rc("a", 1.0)],
        "dense": [_rc("b", 0.9)],
    })
    for entry in r:
        assert "bm25" in entry.channel_contributions
        assert "dense" in entry.channel_contributions


def test_rrf_uniform_weights(rrf):
    r_unweighted = rrf.fuse({"ch": [_rc("a", 1.0), _rc("b", 0.5)]},
                             weights={"ch": 1.0})
    r_weighted   = rrf.fuse({"ch": [_rc("a", 1.0), _rc("b", 0.5)]},
                             weights={"ch": 2.0})
    # Order should be the same; scores doubled
    assert [x.chunk_id for x in r_unweighted] == [x.chunk_id for x in r_weighted]
    assert r_weighted[0].fused_score == pytest.approx(2 * r_unweighted[0].fused_score)


def test_rrf_absent_doc_gets_worst_rank(rrf):
    """Doc present in only one channel still gets scored."""
    r = rrf.fuse({
        "bm25":  [_rc("shared", 1.0), _rc("bm25_only", 0.5)],
        "dense": [_rc("shared", 0.9)],
    })
    ids = [x.chunk_id for x in r]
    assert "bm25_only" in ids
    # bm25_only should score lower than shared (absent in dense = worst rank)
    shared_score = next(x.fused_score for x in r if x.chunk_id == "shared")
    bm25_only_score = next(x.fused_score for x in r if x.chunk_id == "bm25_only")
    assert shared_score > bm25_only_score


def test_rrf_empty_channels(rrf):
    assert rrf.fuse({}) == []
    assert rrf.fuse({"ch": []}) == []


def test_rrf_deterministic(rrf):
    ch = {"a": [_rc("x", 1.0), _rc("y", 0.5)],
          "b": [_rc("y", 0.8), _rc("x", 0.3)]}
    r1 = [x.chunk_id for x in rrf.fuse(ch)]
    r2 = [x.chunk_id for x in rrf.fuse(ch)]
    assert r1 == r2


def test_rrf_k_effect():
    """Smaller k → sharper rank discrimination."""
    rrf_small = RRFFuser(k=1)
    rrf_large = RRFFuser(k=100)
    ch = {"ch": [_rc("a", 1.0), _rc("b", 0.5)]}
    small = rrf_small.fuse(ch)
    large = rrf_large.fuse(ch)
    # Both rank a first
    assert small[0].chunk_id == large[0].chunk_id == "a"
    # Rank-1 vs rank-2 gap is larger with small k
    gap_small = small[0].fused_score - small[1].fused_score
    gap_large = large[0].fused_score - large[1].fused_score
    assert gap_small > gap_large


# ---------------------------------------------------------------------------
# LinearFuser
# ---------------------------------------------------------------------------

def test_linear_fuser_order():
    fuser = LinearFuser()
    r = fuser.fuse({"ch": [_rc("a", 1.0), _rc("b", 0.0)]})
    assert r[0].chunk_id == "a"


def test_linear_fuser_normalises():
    fuser = LinearFuser()
    r = fuser.fuse({"ch": [_rc("a", 100.0), _rc("b", 50.0)]},
                   weights={"ch": 1.0})
    # min-max normalisation: a=(100-50)/(100-50)=1.0, b=(50-50)/(100-50)=0.0
    a_score = next(x.fused_score for x in r if x.chunk_id == "a")
    b_score = next(x.fused_score for x in r if x.chunk_id == "b")
    assert a_score == pytest.approx(1.0)
    assert b_score == pytest.approx(0.0)
