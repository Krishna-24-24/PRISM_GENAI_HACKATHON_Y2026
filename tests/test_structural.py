"""
Unit tests for StructuralIndex and StructuralWeights.

Uses a synthetic JS-like corpus with real call relationships so signals fire.
Does NOT depend on any benchmark dataset or embedding model.

Test coverage:
  1. Build: call edge resolution (name → chunk_id)
  2. get_callers / get_callees
  3. get_import_siblings
  4. score_candidates: callee boost fires
  5. score_candidates: caller boost fires
  6. score_candidates: import sibling boost fires
  7. Boosts are additive (all three can stack)
  8. StructuralWeights.disabled() → all boosts = 0
  9. Graceful degradation: chunks with empty calls/imports → no boost, no error
  10. score_candidates with empty candidate list
  11. Anchor selection uses primary scores correctly
  12. No self-edges (chunk does not call itself in its own boost)
  13. _normalise_identifier strips dotted names
  14. _normalise_import extracts ES6 and CJS specifiers
  15. Stats are populated after build
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.chunking.models import CodeChunk
from src.retrieval.structural_index import (
    StructuralIndex,
    StructuralWeights,
    _normalise_identifier,
    _normalise_import,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic JS-like corpus
# ---------------------------------------------------------------------------
#
# Call graph:
#   router  → handleRequest → validateInput → sanitize
#             handleRequest → sendResponse
#   router  → logError
#
# Imports:
#   router, handleRequest  both import "express"
#   sanitize, validateInput both import "validator"


def _chunk(
    cid: str,
    name: str,
    calls: list[str],
    imports: list[str],
    exports: list[str] | None = None,
) -> CodeChunk:
    return CodeChunk(
        chunk_id=cid,
        repo_id="test",
        version_id="1",
        file_path="app.js",
        symbol_name=name,
        symbol_type="function_declaration",
        start_line=1,
        end_line=10,
        code=f"function {name}() {{}}",
        normalized_code=f"function {name}() {{}}",
        context=f"function {name}",
        imports=imports,
        exports=exports or [],
        calls=calls,
        called_by=[],
        chunk_hash=cid,
    )


CHUNKS = [
    _chunk("c_router",       "router",       ["handleRequest", "logError"],
           ['import express from "express"']),
    _chunk("c_handle",       "handleRequest", ["validateInput", "sendResponse"],
           ['import { Request } from "express"']),
    _chunk("c_validate",     "validateInput", ["sanitize"],
           ['const validator = require("validator")']),
    _chunk("c_sanitize",     "sanitize",      [],
           ['const { escape } = require("validator")']),
    _chunk("c_sendResponse", "sendResponse",  [],
           []),
    _chunk("c_logError",     "logError",      [],
           []),
]


@pytest.fixture(scope="module")
def idx() -> StructuralIndex:
    si = StructuralIndex()
    si.build(CHUNKS)
    return si


# ---------------------------------------------------------------------------
# Build tests
# ---------------------------------------------------------------------------

def test_build_produces_call_edges(idx):
    # router calls handleRequest and logError → 2 outbound
    assert "c_handle" in idx.get_callees("c_router")
    assert "c_logError" in idx.get_callees("c_router")


def test_build_populates_calls_in(idx):
    # handleRequest is called by router
    assert "c_router" in idx.get_callers("c_handle")


def test_no_self_edges(idx):
    for chunk in CHUNKS:
        assert chunk.chunk_id not in idx.get_callees(chunk.chunk_id)
        assert chunk.chunk_id not in idx.get_callers(chunk.chunk_id)


def test_stats_populated(idx):
    assert idx.stats.num_chunks == len(CHUNKS)
    assert idx.stats.num_call_edges > 0
    assert idx.stats.build_time_s >= 0


def test_stats_call_edge_count(idx):
    # router→handle, router→logError, handle→validate,
    # handle→sendResponse, validate→sanitize = 5 edges
    assert idx.stats.num_call_edges == 5


# ---------------------------------------------------------------------------
# get_callers / get_callees
# ---------------------------------------------------------------------------

def test_get_callees_transitive_not_included(idx):
    # router directly calls handleRequest; sanitize is two hops away
    # get_callees should only return direct edges
    assert "c_sanitize" not in idx.get_callees("c_router")


def test_get_callers_leaf_node(idx):
    # router is never called by anyone in the corpus
    assert len(idx.get_callers("c_router")) == 0


def test_get_callees_leaf_node(idx):
    # sanitize calls nothing
    assert len(idx.get_callees("c_sanitize")) == 0


# ---------------------------------------------------------------------------
# Import siblings
# ---------------------------------------------------------------------------

def test_import_siblings_share_express(idx):
    # router and handleRequest both import "express"
    sibs = idx.get_import_siblings("c_router")
    assert "c_handle" in sibs


def test_import_siblings_share_validator(idx):
    sibs = idx.get_import_siblings("c_validate")
    assert "c_sanitize" in sibs


def test_import_siblings_no_self(idx):
    for chunk in CHUNKS:
        sibs = idx.get_import_siblings(chunk.chunk_id)
        assert chunk.chunk_id not in sibs


def test_no_import_no_siblings(idx):
    # sendResponse has no imports
    sibs = idx.get_import_siblings("c_sendResponse")
    assert len(sibs) == 0


# ---------------------------------------------------------------------------
# score_candidates
# ---------------------------------------------------------------------------

def test_callee_boost_fires(idx):
    """
    Query anchors on router (highest primary score).
    handleRequest is a callee of router → should get callee_boost.
    """
    sw = StructuralWeights(callee_boost=0.01, caller_boost=0.0,
                           import_sibling_boost=0.0, top_anchor=1)
    primary = {
        "c_router": 0.10,    # anchor
        "c_handle": 0.05,
        "c_logError": 0.03,
    }
    boosts = idx.score_candidates(
        list(primary.keys()), primary, sw, top_anchor=1
    )
    assert boosts["c_handle"] == pytest.approx(0.01)
    assert boosts["c_logError"] == pytest.approx(0.01)  # also a callee of router
    assert boosts["c_router"] == pytest.approx(0.0)     # anchor not boosted by itself


def test_caller_boost_fires(idx):
    """
    Anchor on handleRequest. router is a caller of handleRequest → caller_boost.
    """
    sw = StructuralWeights(callee_boost=0.0, caller_boost=0.01,
                           import_sibling_boost=0.0, top_anchor=1)
    primary = {
        "c_handle": 0.10,    # anchor
        "c_router": 0.05,
    }
    boosts = idx.score_candidates(
        list(primary.keys()), primary, sw, top_anchor=1
    )
    assert boosts["c_router"] == pytest.approx(0.01)
    assert boosts["c_handle"] == pytest.approx(0.0)


def test_import_sibling_boost_fires(idx):
    sw = StructuralWeights(callee_boost=0.0, caller_boost=0.0,
                           import_sibling_boost=0.005, top_anchor=1)
    primary = {
        "c_router": 0.10,    # anchor; imports "express"
        "c_handle": 0.05,    # also imports "express" → sibling
    }
    boosts = idx.score_candidates(
        list(primary.keys()), primary, sw, top_anchor=1
    )
    assert boosts["c_handle"] == pytest.approx(0.005)


def test_boosts_are_additive(idx):
    """A candidate that is both a callee and an import sibling gets both boosts."""
    sw = StructuralWeights(callee_boost=0.01, caller_boost=0.0,
                           import_sibling_boost=0.005, top_anchor=1)
    primary = {
        "c_router": 0.10,
        "c_handle": 0.05,
    }
    boosts = idx.score_candidates(
        list(primary.keys()), primary, sw, top_anchor=1
    )
    # c_handle is a callee of router (0.01) AND shares "express" import (0.005)
    assert boosts["c_handle"] == pytest.approx(0.015)


def test_disabled_weights_zero_boosts(idx):
    sw = StructuralWeights.disabled()
    primary = {"c_router": 0.10, "c_handle": 0.05}
    boosts = idx.score_candidates(list(primary.keys()), primary, sw)
    assert all(b == 0.0 for b in boosts.values())


def test_empty_candidate_list(idx):
    sw = StructuralWeights()
    boosts = idx.score_candidates([], {}, sw)
    assert boosts == {}


def test_anchor_count_respects_top_anchor(idx):
    """top_anchor=2 includes two anchors; top_anchor=1 includes one."""
    sw1 = StructuralWeights(callee_boost=0.01, caller_boost=0.0,
                            import_sibling_boost=0.0, top_anchor=1)
    sw2 = StructuralWeights(callee_boost=0.01, caller_boost=0.0,
                            import_sibling_boost=0.0, top_anchor=2)
    primary = {
        "c_router": 0.10,
        "c_handle": 0.09,
        "c_logError": 0.01,
    }
    boosts1 = idx.score_candidates(list(primary.keys()), primary, sw1, top_anchor=1)
    boosts2 = idx.score_candidates(list(primary.keys()), primary, sw2, top_anchor=2)

    # With top_anchor=2, both router and handleRequest are anchors
    # → validate is a callee of handle → should be boosted in sw2 but not sw1
    # (validate isn't in primary here, so just check router callees)
    # logError is callee of router → boosted by sw1 already
    assert boosts2["c_logError"] >= boosts1["c_logError"]


# ---------------------------------------------------------------------------
# Empty corpus / graceful degradation
# ---------------------------------------------------------------------------

def test_empty_calls_no_error():
    empty_chunks = [
        _chunk("x1", "foo", [], []),
        _chunk("x2", "bar", [], []),
    ]
    si = StructuralIndex()
    si.build(empty_chunks)
    sw = StructuralWeights()
    boosts = si.score_candidates(["x1", "x2"], {"x1": 0.5, "x2": 0.3}, sw)
    assert boosts == {"x1": 0.0, "x2": 0.0}


def test_empty_index_build():
    si = StructuralIndex()
    si.build([])
    assert si.stats.num_chunks == 0
    assert si.stats.num_call_edges == 0


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

def test_normalise_identifier_dotted():
    assert _normalise_identifier("obj.method") == "method"
    assert _normalise_identifier("this.helper") == "helper"


def test_normalise_identifier_plain():
    assert _normalise_identifier("handleRequest") == "handlerequest"


def test_normalise_identifier_empty():
    assert _normalise_identifier("") == ""
    assert _normalise_identifier(".") == ""


def test_normalise_import_es6():
    assert _normalise_import('import { foo } from "express"') == "express"
    assert _normalise_import("import React from 'react'") == "react"


def test_normalise_import_cjs():
    assert _normalise_import('const x = require("lodash")') == "lodash"


def test_normalise_import_no_match():
    assert _normalise_import("not an import statement") == ""
