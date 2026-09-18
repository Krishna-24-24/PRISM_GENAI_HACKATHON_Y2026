"""
Reranking Experiment Matrix Benchmark.

Objective: Determine if a lightweight CrossEncoder reranking stage is justified
by the MRR/NDCG@10 improvements, relative to the added CPU latency overhead.

We benchmark four configurations on the CoIR AppsRetrieval dataset (sample):
  A. Baseline: Adaptive Hybrid (No Reranker)
  B. Hybrid -> Rerank Top 10
  C. Hybrid -> Rerank Top 50
  D. BM25-only Top 50 -> Rerank (Tests if we can bypass Dense entirely)

Usage:
    python benchmarks/run_reranker_benchmark.py --sample 200 --skip-build

DO NOT pre-fill any metric values.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.run_bm25_benchmark import corpus_to_chunks, load_apps_dataset
from src.evaluation.metrics import compute_metrics
from src.indexing.embedder import BGESmallProvider
from src.retrieval.adaptive_retriever import AdaptiveRetriever
from src.retrieval.hybrid_retriever import HybridConfig
from src.ranking.reranker import CrossEncoderReranker


def run_pipeline(
    name: str,
    retriever: AdaptiveRetriever,
    reranker: Optional[CrossEncoderReranker],
    rerank_top_k: int,
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, float]],
    eval_qids: List[str],
    final_top_k: int = 10,
) -> dict:
    print(f"\n{'='*65}", flush=True)
    print(f"  {name}", flush=True)
    print(f"{'='*65}", flush=True)

    results_per_query: Dict[str, List[str]] = {}
    latencies: List[float] = []

    for qid in eval_qids:
        query_text = queries[qid]
        t0 = time.perf_counter()
        
        # 1. First stage retrieval
        # If we have a reranker, we need to fetch `rerank_top_k` candidates first
        fetch_k = rerank_top_k if reranker else final_top_k
        hits = retriever.search(query_text, top_k=fetch_k)
        
        # 2. Second stage reranking
        if reranker and hits:
            hits = reranker.rerank(query_text, hits, top_k=final_top_k)
            
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        
        # We cap at final_top_k for metrics
        results_per_query[qid] = [h.chunk_id for h in hits[:final_top_k]]

    metrics = compute_metrics(results_per_query, qrels, k=final_top_k)
    lat_arr = np.array(latencies)

    row = {
        "configuration": name,
        "ndcg@10": metrics["ndcg@10"],
        "mrr": metrics["mrr"],
        "latency_avg_ms": round(float(np.mean(lat_arr)), 2),
        "latency_p50_ms": round(float(np.percentile(lat_arr, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(lat_arr, 95)), 2),
        "queries_evaluated": len(eval_qids),
    }

    print(f"  NDCG@10={row['ndcg@10']}  MRR={row['mrr']}  "
          f"P50={row['latency_p50_ms']}ms  P95={row['latency_p95_ms']}ms", flush=True)

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    print("Loading AppsRetrieval dataset...", flush=True)
    corpus, queries, qrels = load_apps_dataset()
    chunks = corpus_to_chunks(corpus)
    eval_qids = sorted(qid for qid in qrels if qid in queries)[:args.sample]
    print(f"  corpus={len(corpus):,}  eval_qids={len(eval_qids)}", flush=True)

    print("\nLoading BGE-small-en-v1.5 (Retriever) on CPU...", flush=True)
    provider = BGESmallProvider(device="cpu")

    print("Loading ms-marco-MiniLM-L-6-v2 (Reranker) on CPU...", flush=True)
    reranker = CrossEncoderReranker()

    cache_dir = str(Path(__file__).parent.parent / "data" / "embed_cache")
    index_dir = str(Path(__file__).parent.parent / "data" / "faiss_index")

    if not args.skip_build:
        from src.indexing.dense_index import DenseChannel
        print("\nBuilding shared dense index...", flush=True)
        dc = DenseChannel(provider, batch_size=32)
        dc.build(chunks, cache_dir=cache_dir, show_progress=True)
        dc.save(index_dir)

    # Base Hybrid config (Dense-only fallback via disable_routing)
    cfg_hybrid = HybridConfig(use_bm25=False, use_dense=True, use_symbol=False, embed_cache_dir=cache_dir)
    ret_hybrid = AdaptiveRetriever(cfg_hybrid, embedding_provider=provider, disable_routing=True)
    ret_hybrid.load_dense_index(index_dir)

    # BM25-only config
    cfg_bm25 = HybridConfig(use_bm25=True, use_dense=False, use_symbol=False)
    ret_bm25 = AdaptiveRetriever(cfg_bm25, disable_routing=True)
    ret_bm25.build(chunks, show_progress=False) # fast enough to build dynamically

    rows = []

    # Config A
    rows.append(run_pipeline(
        "A. Baseline: Adaptive Hybrid (No Reranker)",
        retriever=ret_hybrid, reranker=None, rerank_top_k=0,
        queries=queries, qrels=qrels, eval_qids=eval_qids
    ))

    # Config B
    rows.append(run_pipeline(
        "B. Hybrid -> Rerank Top 10",
        retriever=ret_hybrid, reranker=reranker, rerank_top_k=10,
        queries=queries, qrels=qrels, eval_qids=eval_qids
    ))

    # Config C
    rows.append(run_pipeline(
        "C. Hybrid -> Rerank Top 50",
        retriever=ret_hybrid, reranker=reranker, rerank_top_k=50,
        queries=queries, qrels=qrels, eval_qids=eval_qids
    ))

    # Config D
    rows.append(run_pipeline(
        "D. BM25 Top 50 -> Rerank",
        retriever=ret_bm25, reranker=reranker, rerank_top_k=50,
        queries=queries, qrels=qrels, eval_qids=eval_qids
    ))

    # -----------------------------------------------------------------------
    # Print comparison table
    # -----------------------------------------------------------------------
    cols = ["configuration", "ndcg@10", "mrr", "latency_p50_ms", "latency_p95_ms"]
    widths = [45, 9, 8, 14, 14]
    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    sep = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print("RERANKING EXPERIMENT MATRIX")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)
    for row in rows:
        line = "  ".join(str(row.get(c, "")).ljust(w) for c, w in zip(cols, widths))
        print(line)
    print(f"{'='*len(header)}")

    baseline_ndcg = rows[0]["ndcg@10"]
    baseline_mrr  = rows[0]["mrr"]
    print("\nDelta vs baseline (Adaptive Hybrid):")
    for row in rows[1:]:
        d_n = round(row["ndcg@10"] - baseline_ndcg, 4)
        d_m = round(row["mrr"] - baseline_mrr, 4)
        print(f"  {row['configuration'][:45]:<45}  NDCG@10 {d_n:+.4f}  MRR {d_m:+.4f}")

    # Save
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    path = results_dir / f"reranker_{ts}.json"
    path.write_text(json.dumps({"timestamp": datetime.utcnow().isoformat(), "results": rows}, indent=2))
    print(f"\nResults saved to {path}", flush=True)

if __name__ == "__main__":
    main()
