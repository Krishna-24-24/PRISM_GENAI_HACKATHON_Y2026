"""
Agentic Retrieval Loop Benchmark.

Objective: Determine if an autonomous, multi-pass retrieval loop 
outperforms single-pass configurations on the CoIR AppsRetrieval dataset,
and whether it adheres to strict CPU latency constraints.

Configurations compared:
  A. Static Hybrid (disable_routing=True)  <- Equivalent to Dense-only on this dataset
  B. Adaptive Retrieval                    <- Baseline from previous experiments
  C. Agentic Retrieval                     <- Multi-pass loop using PRF/AST pivots

Usage:
    python benchmarks/run_agentic_benchmark.py --sample 200 --skip-build

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
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.run_bm25_benchmark import corpus_to_chunks, load_apps_dataset
from src.evaluation.metrics import compute_metrics
from src.indexing.embedder import BGESmallProvider
from src.retrieval.adaptive_retriever import AdaptiveRetriever
from src.retrieval.hybrid_retriever import HybridConfig
from src.agent.retrieval_agent import RetrievalAgent


def run_pipeline(
    name: str,
    retriever, # AdaptiveRetriever or RetrievalAgent
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, float]],
    eval_qids: List[str],
    top_k: int = 10,
) -> dict:
    print(f"\n{'='*65}", flush=True)
    print(f"  {name}", flush=True)
    print(f"{'='*65}", flush=True)

    results_per_query: Dict[str, List[str]] = {}
    latencies: List[float] = []
    
    # Track agent traces
    early_term_count = 0
    pass2_count = 0

    for qid in eval_qids:
        query_text = queries[qid]
        t0 = time.perf_counter()
        
        if isinstance(retriever, RetrievalAgent):
            hits, trace = retriever.search(query_text, top_k=top_k)
            if trace.early_terminated:
                early_term_count += 1
            if len(trace.steps) > 1:
                pass2_count += 1
        else:
            hits = retriever.search(query_text, top_k=top_k)
            
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        
        results_per_query[qid] = [h.chunk_id for h in hits[:top_k]]

    metrics = compute_metrics(results_per_query, qrels, k=top_k)
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
          
    if isinstance(retriever, RetrievalAgent):
        print(f"  Agent Early Terminations: {early_term_count}/{len(eval_qids)}")
        print(f"  Agent 2nd-Pass Executions: {pass2_count}/{len(eval_qids)}")
        row["agent_stats"] = {
            "early_terminations": early_term_count,
            "pass2_executions": pass2_count
        }

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

    print("\nLoading BGE-small-en-v1.5 on CPU...", flush=True)
    provider = BGESmallProvider(device="cpu")

    cache_dir = str(Path(__file__).parent.parent / "data" / "embed_cache")
    index_dir = str(Path(__file__).parent.parent / "data" / "faiss_index")

    if not args.skip_build:
        from src.indexing.dense_index import DenseChannel
        print("\nBuilding shared dense index...", flush=True)
        dc = DenseChannel(provider, batch_size=32)
        dc.build(chunks, cache_dir=cache_dir, show_progress=True)
        dc.save(index_dir)

    # Base Hybrid config 
    cfg_hybrid = HybridConfig(use_bm25=True, use_dense=True, use_symbol=True, embed_cache_dir=cache_dir)
    
    # Config A: Static Hybrid (routing disabled)
    ret_static = AdaptiveRetriever(cfg_hybrid, embedding_provider=provider, disable_routing=True)
    ret_static.load_dense_index(index_dir)
    
    # Config B: Adaptive Hybrid
    ret_adaptive = AdaptiveRetriever(cfg_hybrid, embedding_provider=provider, disable_routing=False)
    ret_adaptive.load_dense_index(index_dir)
    
    # Needs BM25 and Symbol built for PRF/AST expansions to work
    ret_adaptive._retriever._bm25.add_chunks(chunks)
    ret_adaptive._retriever._symbol.add_chunks(chunks)
    
    # Config C: Agentic Retrieval
    agent = RetrievalAgent(retriever=ret_adaptive, max_iterations=2, max_latency_ms=800.0)

    rows = []

    # Config A
    rows.append(run_pipeline(
        "A. Static Hybrid",
        retriever=ret_static, queries=queries, qrels=qrels, eval_qids=eval_qids
    ))

    # Config B
    rows.append(run_pipeline(
        "B. Adaptive Retrieval",
        retriever=ret_adaptive, queries=queries, qrels=qrels, eval_qids=eval_qids
    ))

    # Config C
    rows.append(run_pipeline(
        "C. Agentic Retrieval",
        retriever=agent, queries=queries, qrels=qrels, eval_qids=eval_qids
    ))

    # -----------------------------------------------------------------------
    # Print comparison table
    # -----------------------------------------------------------------------
    cols = ["configuration", "ndcg@10", "mrr", "latency_p50_ms", "latency_p95_ms"]
    widths = [45, 9, 8, 14, 14]
    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    sep = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print("AGENTIC RETRIEVAL BENCHMARK")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)
    for row in rows:
        line = "  ".join(str(row.get(c, "")).ljust(w) for c, w in zip(cols, widths))
        print(line)
    print(f"{'='*len(header)}")

    baseline_ndcg = rows[1]["ndcg@10"]
    baseline_mrr  = rows[1]["mrr"]
    print("\nDelta vs Adaptive Retrieval:")
    for row in rows[2:]:
        d_n = round(row["ndcg@10"] - baseline_ndcg, 4)
        d_m = round(row["mrr"] - baseline_mrr, 4)
        print(f"  {row['configuration'][:45]:<45}  NDCG@10 {d_n:+.4f}  MRR {d_m:+.4f}")

    # Save
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    path = results_dir / f"agentic_{ts}.json"
    path.write_text(json.dumps({"timestamp": datetime.utcnow().isoformat(), "results": rows}, indent=2))
    print(f"\nResults saved to {path}", flush=True)

if __name__ == "__main__":
    main()
