"""
HybridRetriever — orchestrates BM25 + Dense + Symbol channels with RRF fusion.

This is the main retrieval entry point. All configuration is data-driven;
no weights are hardcoded. Weights are determined experimentally and passed in.

Usage:
    retriever = HybridRetriever(config)
    retriever.build(chunks)
    results = retriever.search(query, top_k=10)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.chunking.models import CodeChunk
from src.indexing.bm25_index import BM25Channel, ScoredCandidate as BM25Candidate
from src.indexing.dense_index import DenseChannel, DenseCandidate
from src.indexing.embedder import EmbeddingProvider
from src.retrieval.fusion import FusedCandidate, RRFFuser, RawCandidate
from src.retrieval.symbol_channel import SymbolCandidate, SymbolChannel


@dataclass
class HybridConfig:
    """
    All fusion parameters in one place.
    Values here are DEFAULTS — experiments must justify any deviations.
    """
    # Which channels to enable
    use_bm25: bool = True
    use_dense: bool = True
    use_symbol: bool = True

    # How many candidates each channel returns before fusion
    bm25_top_k: int = 100
    dense_top_k: int = 100
    symbol_top_k: int = 50

    # RRF damping constant (sweep via E-RRFK)
    rrf_k: float = 60.0

    # Channel weights for wRRF (uniform = no evidence yet)
    # Overwrite after running ablation experiments
    weights: Dict[str, float] = field(default_factory=lambda: {
        "bm25": 1.0,
        "dense": 1.0,
        "symbol": 1.0,
    })

    # Dense embedding cache dir (None = no caching)
    embed_cache_dir: Optional[str] = None

    # FAISS index persistence dir (None = in-memory only)
    faiss_index_dir: Optional[str] = None


@dataclass
class RetrievalResult:
    chunk_id: str
    fused_score: float
    channel_contributions: Dict[str, float]
    chunk: Optional[CodeChunk]
    latency_ms: float = 0.0


class HybridRetriever:
    """
    Multi-channel retriever with pluggable fusion.

    Building and searching are separate so the index can be persisted
    and reloaded without re-embedding.
    """

    def __init__(
        self,
        config: HybridConfig,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> None:
        self.config = config
        self._bm25: Optional[BM25Channel] = None
        self._dense: Optional[DenseChannel] = None
        self._symbol: Optional[SymbolChannel] = None
        self._fuser = RRFFuser(k=config.rrf_k)
        self._provider = embedding_provider

        if config.use_bm25:
            self._bm25 = BM25Channel()
        if config.use_dense:
            if embedding_provider is None:
                raise ValueError("embedding_provider required when use_dense=True")
            self._dense = DenseChannel(embedding_provider)
        if config.use_symbol:
            self._symbol = SymbolChannel()

    def build(self, chunks: List[CodeChunk], show_progress: bool = True) -> None:
        """Index all chunks across enabled channels."""
        if self._bm25 is not None:
            self._bm25.add_chunks(chunks)

        if self._dense is not None:
            self._dense.build(
                chunks,
                cache_dir=self.config.embed_cache_dir,
                show_progress=show_progress,
            )
            if self.config.faiss_index_dir:
                self._dense.save(self.config.faiss_index_dir)

        if self._symbol is not None:
            self._symbol.add_chunks(chunks)

    def load_dense_index(self, directory: str) -> None:
        """Load a pre-built FAISS index (skip re-embedding)."""
        if self._dense is None:
            raise RuntimeError("use_dense must be True to load a dense index.")
        self._dense.load(directory)

    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        t0 = time.perf_counter()
        channel_results: Dict[str, List[RawCandidate]] = {}

        if self._bm25 is not None:
            raw = self._bm25.search(query, top_k=self.config.bm25_top_k)
            channel_results["bm25"] = [
                RawCandidate(c.chunk_id, c.score, "bm25", c.chunk) for c in raw
            ]

        if self._dense is not None:
            raw_d = self._dense.search(query, top_k=self.config.dense_top_k)
            channel_results["dense"] = [
                RawCandidate(c.chunk_id, c.score, "dense", c.chunk) for c in raw_d
            ]

        if self._symbol is not None:
            raw_s = self._symbol.search(query, top_k=self.config.symbol_top_k)
            channel_results["symbol"] = [
                RawCandidate(c.chunk_id, c.score, "symbol", c.chunk) for c in raw_s
            ]

        # Filter to only enabled channels' weights
        active_weights = {
            ch: self.config.weights.get(ch, 1.0)
            for ch in channel_results
        }

        fused = self._fuser.fuse(channel_results, weights=active_weights)
        latency_ms = (time.perf_counter() - t0) * 1000

        return [
            RetrievalResult(
                chunk_id=fc.chunk_id,
                fused_score=fc.fused_score,
                channel_contributions=fc.channel_contributions,
                chunk=fc.chunk,
                latency_ms=latency_ms,
            )
            for fc in fused[:top_k]
        ]

    @property
    def index_size_mb(self) -> float:
        total = 0
        if self._bm25:
            total += self._bm25.get_index_size_bytes()
        if self._dense:
            total += self._dense.get_index_size_bytes()
        return total / (1024 ** 2)
