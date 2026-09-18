"""
AdaptiveRetriever — HybridRetriever with per-query channel routing
and optional structural signal boosting.

Pipeline per query:
  1. QueryAnalyzer classifies query → QueryProfile (weights, category)
  2. Active channels searched with profile-recommended weights
  3. wRRF fusion → ranked candidate list
  4. StructuralIndex scores each candidate (additive boost from call graph + imports)
  5. Re-sort by (fused_score + structural_boost) → final top-k

Structural signals are:
  - ADDITIVE only — never replace fused_score
  - BOUNDED — max boost is << typical primary score range
  - DISABLEABLE — StructuralWeights.disabled() zeroes all boosts

This file replaces the previous version of adaptive_retriever.py.

Evidence base:
  BM25 alone:    NDCG@10=0.0131  (ablation 2026-09-17)
  Dense alone:   NDCG@10=0.0498  (ablation 2026-09-17)  ← best
  BM25+Dense:    NDCG@10=0.0495  (BM25 dilutes Dense)
  Adaptive=Baseline on AppsRetrieval (91% semantic NL queries, no AST in corpus)
  Structural goal: ≥ Dense-only on AppsRetrieval; measurable gain on JS demo path
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.chunking.models import CodeChunk
from src.indexing.embedder import EmbeddingProvider
from src.retrieval.fusion import RRFFuser, RawCandidate
from src.retrieval.hybrid_retriever import HybridConfig, HybridRetriever, RetrievalResult
from src.retrieval.query_analyzer import QueryAnalyzer, QueryProfile
from src.retrieval.structural_index import (
    StructuralIndex,
    StructuralWeights,
)


@dataclass
class AdaptiveRetrievalResult(RetrievalResult):
    """RetrievalResult extended with routing and structural metadata."""
    query_profile: Optional[QueryProfile] = None
    structural_boost: float = 0.0
    final_score: float = 0.0   # fused_score + structural_boost


class AdaptiveRetriever:
    """
    Per-query adaptive channel weighting + optional structural signal boosting.

    All channels (BM25, Dense, Symbol) are always indexed at build time.
    QueryAnalyzer selects which channels to activate and at what weight per query.
    StructuralIndex adds call-graph and import-graph boosts after fusion.
    """

    def __init__(
        self,
        config: HybridConfig,
        embedding_provider: Optional[EmbeddingProvider] = None,
        disable_routing: bool = False,
        structural_weights: Optional[StructuralWeights] = None,
    ) -> None:
        self.disable_routing = disable_routing
        self._structural_weights = structural_weights or StructuralWeights()
        self._analyzer = QueryAnalyzer(disable_routing=disable_routing)
        self._retriever = HybridRetriever(config, embedding_provider=embedding_provider)
        self._fuser = self._retriever._fuser
        self._config = config
        self._structural = StructuralIndex()
        self._structural_built = False

    # ------------------------------------------------------------------
    # Build / load
    # ------------------------------------------------------------------

    def build(self, chunks: List[CodeChunk], show_progress: bool = True) -> None:
        """Index all chunks across retrieval and structural channels."""
        self._retriever.build(chunks, show_progress=show_progress)
        if self._structural_weights.enabled:
            self._structural.build(chunks)
            self._structural_built = True

    def load_dense_index(self, directory: str) -> None:
        self._retriever.load_dense_index(directory)

    def build_structural(self, chunks: List[CodeChunk]) -> None:
        """Build structural index separately (e.g., after loading a pre-built FAISS index)."""
        self._structural.build(chunks)
        self._structural_built = True

    @property
    def index_size_mb(self) -> float:
        retrieval_bytes = (
            (self._retriever._bm25.get_index_size_bytes() if self._retriever._bm25 else 0)
            + (self._retriever._dense.get_index_size_bytes() if self._retriever._dense else 0)
        )
        structural_bytes = (
            self._structural.stats.index_size_bytes
            if self._structural_built else 0
        )
        return (retrieval_bytes + structural_bytes) / (1024 ** 2)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        structural_candidates: int = 50,
    ) -> List[AdaptiveRetrievalResult]:
        """
        Args:
            query: Natural language query string.
            top_k: Final results to return.
            structural_candidates: How many post-fusion candidates to
                pass through structural scoring (larger = more recall,
                slower scoring; 50 is a safe default).
        """
        t0 = time.perf_counter()

        # Step 1: Query classification
        profile = self._analyzer.analyze(query)
        weights = profile.recommended_weights

        # Step 2: Channel retrieval
        channel_results: Dict[str, list] = {}
        r = self._retriever

        if r._bm25 is not None and weights.get("bm25", 0) > 0:
            raw = r._bm25.search(query, top_k=self._config.bm25_top_k)
            channel_results["bm25"] = [
                RawCandidate(c.chunk_id, c.score, "bm25", c.chunk) for c in raw
            ]

        if r._dense is not None and weights.get("dense", 0) > 0:
            raw_d = r._dense.search(query, top_k=self._config.dense_top_k)
            channel_results["dense"] = [
                RawCandidate(c.chunk_id, c.score, "dense", c.chunk) for c in raw_d
            ]

        if r._symbol is not None and weights.get("symbol", 0) > 0:
            sym_query = (
                " ".join(profile.extracted_identifiers)
                if profile.extracted_identifiers else query
            )
            raw_s = r._symbol.search(sym_query, top_k=self._config.symbol_top_k)
            channel_results["symbol"] = [
                RawCandidate(c.chunk_id, c.score, "symbol", c.chunk) for c in raw_s
            ]

        # Step 3: wRRF fusion
        active_weights = {ch: weights.get(ch, 1.0) for ch in channel_results}
        fused = self._fuser.fuse(channel_results, weights=active_weights)

        # Step 4: Structural boosting
        # Take top `structural_candidates` from fusion, score structurally,
        # re-sort by (fused_score + structural_boost), return top_k
        fused_pool = fused[:structural_candidates]
        candidate_ids = [fc.chunk_id for fc in fused_pool]
        primary_scores = {fc.chunk_id: fc.fused_score for fc in fused_pool}

        struct_boosts: Dict[str, float] = {}
        if self._structural_built and self._structural_weights.enabled:
            struct_boosts = self._structural.score_candidates(
                candidate_ids=candidate_ids,
                primary_scores=primary_scores,
                weights=self._structural_weights,
                top_anchor=self._structural_weights.top_anchor,
            )

        # Step 5: Re-sort by final score
        fc_lookup = {fc.chunk_id: fc for fc in fused_pool}
        reranked = sorted(
            candidate_ids,
            key=lambda cid: primary_scores[cid] + struct_boosts.get(cid, 0.0),
            reverse=True,
        )

        latency_ms = (time.perf_counter() - t0) * 1000

        return [
            AdaptiveRetrievalResult(
                chunk_id=cid,
                fused_score=primary_scores[cid],
                structural_boost=struct_boosts.get(cid, 0.0),
                final_score=primary_scores[cid] + struct_boosts.get(cid, 0.0),
                channel_contributions=fc_lookup[cid].channel_contributions,
                chunk=fc_lookup[cid].chunk,
                latency_ms=latency_ms,
                query_profile=profile,
            )
            for cid in reranked[:top_k]
        ]
