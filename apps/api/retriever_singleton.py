"""
TRACEIQ API — Retriever singleton.

Loads AdaptiveRetriever once at startup from the pre-built FAISS index.
Thread-safe lazy initialization.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

from src.chunking.models import CodeChunk
from src.indexing.dense_index import DenseChannel
from src.indexing.embedder import BGESmallProvider
from src.retrieval.adaptive_retriever import AdaptiveRetriever
from src.retrieval.hybrid_retriever import HybridConfig

_lock = threading.Lock()
_retriever: Optional[AdaptiveRetriever] = None
_chunks: list[CodeChunk] = []

FAISS_DIR = str(Path(__file__).parent.parent.parent / "data" / "faiss_index")
CHUNKS_JSON = str(Path(__file__).parent.parent.parent / "data" / "faiss_index" / "chunks.json")
CACHE_DIR = str(Path(__file__).parent.parent.parent / "data" / "embed_cache")


def _load_chunks_from_json() -> list[CodeChunk]:
    """Load CodeChunk objects from the persisted chunks.json."""
    p = Path(CHUNKS_JSON)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        raw: dict = json.load(f)
    chunks = []
    for item in raw.values():
        try:
            chunks.append(CodeChunk(**item))
        except Exception:
            pass
    return chunks


def get_retriever() -> AdaptiveRetriever:
    """Return the global AdaptiveRetriever, initializing on first call."""
    global _retriever, _chunks
    if _retriever is not None:
        return _retriever

    with _lock:
        if _retriever is not None:
            return _retriever

        provider = BGESmallProvider(device="cpu")
        cfg = HybridConfig(
            use_bm25=True,
            use_dense=True,
            use_symbol=True,
            embed_cache_dir=CACHE_DIR,
            faiss_index_dir=FAISS_DIR,
        )

        ret = AdaptiveRetriever(cfg, embedding_provider=provider, disable_routing=False)

        # Load pre-built FAISS index
        faiss_path = Path(FAISS_DIR) / "index.faiss"
        if faiss_path.exists():
            ret.load_dense_index(FAISS_DIR)
        else:
            raise RuntimeError(
                f"Pre-built FAISS index not found at {FAISS_DIR}. "
                "Run the ablation benchmark first to build the index."
            )

        # Load chunks for BM25 + symbol channel
        _chunks = _load_chunks_from_json()
        if _chunks:
            if ret._retriever._bm25 is not None:
                ret._retriever._bm25.add_chunks(_chunks)
            if ret._retriever._symbol is not None:
                ret._retriever._symbol.add_chunks(_chunks)

        _retriever = ret

    return _retriever


def get_chunks() -> list[CodeChunk]:
    """Return the loaded chunks (populated after get_retriever() is called)."""
    return _chunks
