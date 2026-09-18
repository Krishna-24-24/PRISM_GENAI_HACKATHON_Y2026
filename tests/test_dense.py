"""
Unit tests for DenseChannel (FAISS) and EmbeddingProvider interface.

Tests run on a tiny synthetic corpus (5 chunks) with a real (but tiny)
BGE-small-en-v1.5 model — results are deterministic on CPU.

Slower than BM25 tests (~3-5s total for model load + encode) but
verified to run without GPU.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from src.chunking.models import CodeChunk
from src.indexing.dense_index import DenseChannel
from src.indexing.embedder import BGESmallProvider, EmbeddingProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def provider() -> BGESmallProvider:
    return BGESmallProvider(device="cpu")


def _chunk(id_: str, code: str, name: str = "fn") -> CodeChunk:
    return CodeChunk(
        chunk_id=id_,
        repo_id="test",
        version_id="1",
        file_path="test.js",
        symbol_name=name,
        symbol_type="function",
        start_line=1,
        end_line=3,
        code=code,
        normalized_code=code,
        context=f"{name} context",
        imports=[],
        exports=[],
        calls=[],
        called_by=[],
        chunk_hash=id_,
    )


CHUNKS = [
    _chunk("c1", "function sortArray(arr) { return arr.sort((a,b) => a-b); }", "sortArray"),
    _chunk("c2", "function fetchUser(id) { return db.find(id); }", "fetchUser"),
    _chunk("c3", "function renderHTML(template, ctx) { return template.replace('{{x}}', ctx.x); }", "renderHTML"),
    _chunk("c4", "function computeHash(data) { return sha256(data); }", "computeHash"),
    _chunk("c5", "function sendEmail(to, body) { mailer.send({to, body}); }", "sendEmail"),
]


@pytest.fixture(scope="module")
def built_channel(provider) -> DenseChannel:
    ch = DenseChannel(provider, batch_size=8)
    ch.build(CHUNKS, show_progress=False)
    return ch


# ---------------------------------------------------------------------------
# Provider interface tests
# ---------------------------------------------------------------------------

def test_provider_implements_interface(provider):
    assert isinstance(provider, EmbeddingProvider)


def test_provider_dim(provider):
    assert provider.dim == 384


def test_provider_model_name(provider):
    assert "bge" in provider.model_name.lower()


def test_embed_texts_shape(provider):
    vecs = provider.embed_texts(["hello world", "def foo(): pass"], batch_size=2)
    assert vecs.shape == (2, 384)
    assert vecs.dtype == np.float32


def test_embed_texts_l2_normalised(provider):
    vecs = provider.embed_texts(["test"], batch_size=1)
    norm = float(np.linalg.norm(vecs[0]))
    assert abs(norm - 1.0) < 1e-5


def test_embed_query_shape(provider):
    q = provider.embed_query("sort an array in ascending order")
    assert q.shape == (384,)
    assert q.dtype == np.float32


def test_embed_query_l2_normalised(provider):
    q = provider.embed_query("find user by id")
    norm = float(np.linalg.norm(q))
    assert abs(norm - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# DenseChannel build tests
# ---------------------------------------------------------------------------

def test_channel_builds(built_channel):
    assert built_channel.num_vectors == len(CHUNKS)
    assert built_channel.build_time > 0


def test_channel_embedding_throughput(built_channel):
    assert built_channel.embedding_throughput > 0


def test_channel_index_size(built_channel):
    assert built_channel.get_index_size_bytes() > 0


# ---------------------------------------------------------------------------
# DenseChannel search tests
# ---------------------------------------------------------------------------

def test_search_returns_results(built_channel):
    results = built_channel.search("sort an array", top_k=3)
    assert len(results) == 3


def test_search_top_k(built_channel):
    results = built_channel.search("code", top_k=2)
    assert len(results) == 2


def test_search_sort_ranks_first(built_channel):
    """sortArray should be the top result for 'sort numbers in ascending order'."""
    results = built_channel.search("sort numbers in ascending order", top_k=5)
    assert results[0].chunk_id == "c1"


def test_search_scores_descending(built_channel):
    results = built_channel.search("send an email", top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_deterministic(built_channel):
    r1 = [r.chunk_id for r in built_channel.search("hash data", top_k=5)]
    r2 = [r.chunk_id for r in built_channel.search("hash data", top_k=5)]
    assert r1 == r2


def test_chunk_metadata_in_results(built_channel):
    results = built_channel.search("find user", top_k=1)
    assert results[0].chunk is not None
    assert results[0].chunk.symbol_name is not None


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

def test_save_and_load_round_trip(provider):
    ch = DenseChannel(provider, batch_size=8)
    ch.build(CHUNKS, show_progress=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        ch.save(tmpdir)

        ch2 = DenseChannel(provider, batch_size=8)
        ch2.load(tmpdir)

        r1 = [r.chunk_id for r in ch.search("render html template", top_k=5)]
        r2 = [r.chunk_id for r in ch2.search("render html template", top_k=5)]
        assert r1 == r2


def test_embedding_cache_reuses_vectors(provider):
    """Second build with same cache dir should produce identical results without re-embedding."""
    with tempfile.TemporaryDirectory() as cache_dir:
        ch1 = DenseChannel(provider, batch_size=8)
        ch1.build(CHUNKS, cache_dir=cache_dir, show_progress=False)

        ch2 = DenseChannel(provider, batch_size=8)
        ch2.build(CHUNKS, cache_dir=cache_dir, show_progress=False)

        # Both should produce same ranking
        r1 = [r.chunk_id for r in ch1.search("compute hash checksum", top_k=5)]
        r2 = [r.chunk_id for r in ch2.search("compute hash checksum", top_k=5)]
        assert r1 == r2
