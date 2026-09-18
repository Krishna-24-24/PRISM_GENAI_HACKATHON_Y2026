"""
Unit tests for BM25Channel:
  - index builds without error
  - tokeniser splits camelCase and snake_case
  - search returns ranked results with expected chunk_ids
  - top_k is respected
  - results are deterministic (same input → same output)
  - symbol/identifier queries are handled
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.chunking.models import CodeChunk
from src.chunking.tokenizer import tokenize_for_bm25
from src.indexing.bm25_index import BM25Channel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _chunk(id_, code, name="fn") -> CodeChunk:
    return CodeChunk(
        chunk_id=id_,
        repo_id="test",
        version_id="1",
        file_path="test.js",
        symbol_name=name,
        symbol_type="function",
        start_line=1,
        end_line=5,
        code=code,
        normalized_code=code,
        context=f"{name} test",
        imports=[],
        exports=[],
        calls=[],
        called_by=[],
        chunk_hash=id_,
    )


CHUNKS = [
    _chunk("c1", "function bubbleSort(arr) { return arr.sort(); }", "bubbleSort"),
    _chunk("c2", "function handleHttpRequest(req, res) { res.send('ok'); }", "handleHttpRequest"),
    _chunk("c3", "const getUserById = (id) => db.users.find(id);", "getUserById"),
    _chunk("c4", "function calculateChecksum(data) { return crc32(data); }", "calculateChecksum"),
    _chunk("c5", "function renderTemplate(html, ctx) { return html.replace('{{name}}', ctx.name); }", "renderTemplate"),
]


@pytest.fixture(scope="module")
def built_index() -> BM25Channel:
    ch = BM25Channel()
    ch.add_chunks(CHUNKS)
    return ch


# ---------------------------------------------------------------------------
# Tokeniser tests
# ---------------------------------------------------------------------------

def test_tokeniser_camelcase_split():
    tokens = tokenize_for_bm25("handleHttpRequest")
    assert "handle" in tokens
    assert "http" in tokens
    assert "request" in tokens
    # Full identifier also preserved
    assert "handlehttprequest" in tokens


def test_tokeniser_snakecase_split():
    tokens = tokenize_for_bm25("get_user_by_id")
    assert "get" in tokens
    assert "user" in tokens
    assert "by" in tokens
    assert "id" in tokens


def test_tokeniser_extra_identifiers():
    tokens = tokenize_for_bm25("some code", identifiers=["myFunc", "MyClass"])
    assert "myfunc" in tokens
    assert "myclass" in tokens


def test_tokeniser_empty():
    assert tokenize_for_bm25("") == []


# ---------------------------------------------------------------------------
# Index tests
# ---------------------------------------------------------------------------

def test_index_builds(built_index):
    assert built_index.bm25 is not None
    assert built_index.build_time >= 0


def test_search_returns_results(built_index):
    results = built_index.search("sort array", top_k=3)
    assert len(results) > 0
    assert results[0].score > 0


def test_top_k_respected(built_index):
    results = built_index.search("function", top_k=2)
    assert len(results) <= 2


def test_search_rank_relevance(built_index):
    """bubbleSort chunk should rank highest for 'sort array'."""
    results = built_index.search("sort array", top_k=5)
    ids = [r.chunk_id for r in results]
    assert ids[0] == "c1"


def test_search_identifier_query(built_index):
    """Identifier query 'handleHttpRequest' should hit c2."""
    results = built_index.search("handleHttpRequest", top_k=5)
    ids = [r.chunk_id for r in results]
    assert "c2" in ids


def test_search_natural_language(built_index):
    """NL query 'get user from database' should hit getUserById."""
    results = built_index.search("get user from database", top_k=5)
    ids = [r.chunk_id for r in results]
    assert "c3" in ids


def test_results_deterministic(built_index):
    r1 = [r.chunk_id for r in built_index.search("checksum crc", top_k=5)]
    r2 = [r.chunk_id for r in built_index.search("checksum crc", top_k=5)]
    assert r1 == r2


def test_chunk_metadata_preserved(built_index):
    results = built_index.search("bubble sort", top_k=1)
    assert results[0].chunk is not None
    assert results[0].chunk.symbol_name == "bubbleSort"


def test_index_size_nonzero(built_index):
    assert built_index.get_index_size_bytes() > 0
