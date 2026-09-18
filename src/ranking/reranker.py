"""
Cross-Encoder Reranker for TRACEIQ.

This component applies a cross-attention transformer over the final (Query, CodeChunk) pairs.
While dual-encoders (DenseChannel) embed queries and documents independently for fast O(1) distance lookups,
cross-encoders process the pair jointly. This allows the model to map specific terms in the query
directly to variables, syntax, or logical flows in the code snippet.

Model selection rationale:
  'cross-encoder/ms-marco-MiniLM-L-6-v2' is only ~33M parameters (86MB RAM).
  On CPU, evaluating 50 pairs takes ~1-2 seconds, which is near the upper limit for acceptable latency,
  but typically yields the highest NDCG@10 delta for difficult queries.

Integration:
  Designed to be applied as a post-processing step on the top-K candidates produced by
  AdaptiveRetriever (or any other first-stage retriever).
"""
from __future__ import annotations

import time
from typing import List

from src.retrieval.adaptive_retriever import AdaptiveRetrievalResult


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        """
        Initializes the CrossEncoder model.
        Requires `sentence-transformers` to be installed.
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("sentence-transformers is required for reranking. Run `pip install sentence-transformers`.")

        self.model_name = model_name
        self.device = device
        # max_length=512 ensures we don't OOM on long chunks and limits CPU time.
        self.model = CrossEncoder(model_name, device=device, max_length=512)

    def rerank(
        self, 
        query: str, 
        candidates: List[AdaptiveRetrievalResult], 
        top_k: int = 10,
        timeout_ms: float = 1500.0,
    ) -> List[AdaptiveRetrievalResult]:
        """
        Reranks candidates using the cross-encoder.
        
        Includes fallback behavior: if reranking exceeds timeout_ms or fails,
        it gracefully returns the original candidates sorted by fused_score.
        """
        if not candidates:
            return []

        t0 = time.perf_counter()

        try:
            # Batching is handled internally by sentence_transformers predict()
            pairs = [[query, c.chunk.code] for c in candidates]
            
            # CPU-friendly inference: batch_size=16 is optimal for MiniLM on CPU
            scores = self.model.predict(pairs, batch_size=16)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            
            # Timeout/Fallback check
            if elapsed_ms > timeout_ms:
                print(f"[Reranker] Timeout exceeded ({elapsed_ms:.1f}ms > {timeout_ms}ms). Falling back to fusion scores.")
                raise TimeoutError("Reranking took too long")

            for c, score in zip(candidates, scores):
                c.final_score = float(score)

            candidates.sort(key=lambda x: x.final_score, reverse=True)

        except Exception as e:
            # Fallback behavior: restore original sorting based on fused_score + structural
            print(f"[Reranker] Fallback triggered due to: {e}")
            for c in candidates:
                c.final_score = c.fused_score + c.structural_boost
            candidates.sort(key=lambda x: x.final_score, reverse=True)

        latency_ms = (time.perf_counter() - t0) * 1000
        for c in candidates:
            c.latency_ms += latency_ms

        return candidates[:top_k]
