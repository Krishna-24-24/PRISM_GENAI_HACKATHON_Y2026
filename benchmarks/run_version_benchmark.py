"""
Code Versions & Incremental Updates Benchmark.

Validates TRACEIQ's P1 constraint: "Retrieval across code versions".
Instead of rebuilding the entire index when small changes occur, the VersionManager
must efficiently diff chunk hashes and surgically update the indices.

We measure:
1. Full rebuild time (Baseline)
2. Incremental update time (Adding/Removing/Modifying chunks)
3. Consistency: Incremental V2 must yield the EXACT same MRR/NDCG@10 as a freshly built V2.

Usage:
    python benchmarks/run_version_benchmark.py --sample 200
"""
from __future__ import annotations

import argparse
import copy
import hashlib
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
from src.chunking.models import CodeChunk
from src.evaluation.metrics import compute_metrics
from src.indexing.embedder import BGESmallProvider
from src.indexing.version_manager import VersionManager
from src.retrieval.adaptive_retriever import AdaptiveRetriever
from src.retrieval.hybrid_retriever import HybridConfig


def generate_v2_corpus(v1_chunks: List[CodeChunk], seed: int = 42) -> List[CodeChunk]:
    """
    Simulates a repository update:
    - 5% chunks deleted
    - 5% chunks added (we hold them back from V1)
    - 5% chunks modified
    - 85% unchanged
    """
    np.random.seed(seed)
    n = len(v1_chunks)
    
    # Shuffle indices to randomly pick sets
    indices = np.random.permutation(n)
    
    del_idx = indices[:int(0.05 * n)]
    mod_idx = indices[int(0.05 * n):int(0.10 * n)]
    keep_idx = indices[int(0.10 * n):]
    
    v2 = []
    
    # Keep chunks
    for idx in keep_idx:
        v2.append(v1_chunks[idx])
        
    # Modify chunks
    for idx in mod_idx:
        c = copy.deepcopy(v1_chunks[idx])
        c.code = c.code + "\n// modified in V2"
        # update hash
        c.chunk_hash = hashlib.sha256(c.code.encode()).hexdigest()
        v2.append(c)
        
    return v2


def evaluate(name: str, retriever: AdaptiveRetriever, queries: dict, qrels: dict, qids: list) -> dict:
    results_per_query = {}
    latencies = []
    for qid in qids:
        t0 = time.perf_counter()
        hits = retriever.search(queries[qid], top_k=10)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        results_per_query[qid] = [h.chunk_id for h in hits]
        
    metrics = compute_metrics(results_per_query, qrels, k=10)
    return {
        "configuration": name,
        "ndcg@10": metrics["ndcg@10"],
        "mrr": metrics["mrr"],
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=200)
    args = parser.parse_args()

    print("Loading AppsRetrieval dataset...", flush=True)
    corpus, queries, qrels = load_apps_dataset()
    all_chunks = corpus_to_chunks(corpus)
    eval_qids = sorted(qid for qid in qrels if qid in queries)[:args.sample]
    
    # We hold back 5% of chunks entirely so they act as "brand new additions" in V2
    np.random.seed(42)
    shuffled = np.random.permutation(len(all_chunks))
    split_pt = int(0.95 * len(all_chunks))
    
    v1_pool_idx = shuffled[:split_pt]
    v2_new_idx = shuffled[split_pt:]
    
    v1_chunks = [all_chunks[i] for i in v1_pool_idx]
    v2_new_chunks = [all_chunks[i] for i in v2_new_idx]
    
    # Generate V2 base
    v2_chunks = generate_v2_corpus(v1_chunks)
    # Add the brand new ones
    v2_chunks.extend(v2_new_chunks)
    
    print(f"  V1 chunks: {len(v1_chunks)}")
    print(f"  V2 chunks: {len(v2_chunks)} (simulated updates/deletions/additions)", flush=True)

    print("\nLoading BGE-small-en-v1.5 on CPU...", flush=True)
    provider = BGESmallProvider(device="cpu")
    cache_dir = str(Path(__file__).parent.parent / "data" / "embed_cache")
    
    cfg = HybridConfig(use_bm25=True, use_dense=True, use_symbol=True, embed_cache_dir=cache_dir)
    
    # ---------------------------------------------------------
    # 1. Build V1 (Baseline)
    # ---------------------------------------------------------
    print("\n--- 1. Building V1 (Baseline) ---", flush=True)
    ret_v1 = AdaptiveRetriever(cfg, embedding_provider=provider)
    vm = VersionManager(ret_v1)
    
    t0 = time.perf_counter()
    res_v1 = vm.set_version(v1_chunks)
    build_time_v1 = time.perf_counter() - t0
    
    print(f"  Full Rebuild Time: {build_time_v1:.2f}s")
    print(f"  Active: {res_v1['total_active']}, Embedded: {res_v1['chunks_embedded_new']}")
    
    # Evaluate V1
    eval_v1 = evaluate("V1 Base", ret_v1, queries, qrels, eval_qids)
    print(f"  V1 Quality: NDCG@10={eval_v1['ndcg@10']:.4f}, MRR={eval_v1['mrr']:.4f}")

    # ---------------------------------------------------------
    # 2. Incrementally Update to V2
    # ---------------------------------------------------------
    print("\n--- 2. Incremental Update (V1 -> V2) ---", flush=True)
    t0 = time.perf_counter()
    res_v2_inc = vm.set_version(v2_chunks)
    build_time_inc = time.perf_counter() - t0
    
    print(f"  Incremental Update Time: {build_time_inc:.2f}s")
    print(f"  Added: {res_v2_inc['added']}, Modified: {res_v2_inc['modified']}, Deleted: {res_v2_inc['deleted']}")
    print(f"  Speedup: {build_time_v1 / build_time_inc:.1f}x faster than full rebuild")
    
    # Evaluate V2 Incremental
    eval_v2_inc = evaluate("V2 Incremental", ret_v1, queries, qrels, eval_qids)
    print(f"  V2 (Inc) Quality: NDCG@10={eval_v2_inc['ndcg@10']:.4f}, MRR={eval_v2_inc['mrr']:.4f}")

    # ---------------------------------------------------------
    # 3. Build V2 From Scratch (Fresh)
    # ---------------------------------------------------------
    print("\n--- 3. Fresh Rebuild (V2) ---", flush=True)
    ret_v2_fresh = AdaptiveRetriever(cfg, embedding_provider=provider)
    vm_fresh = VersionManager(ret_v2_fresh)
    
    t0 = time.perf_counter()
    res_v2_fresh = vm_fresh.set_version(v2_chunks)
    build_time_fresh = time.perf_counter() - t0
    
    print(f"  Fresh Rebuild Time: {build_time_fresh:.2f}s")
    
    # Evaluate V2 Fresh
    eval_v2_fresh = evaluate("V2 Fresh Rebuild", ret_v2_fresh, queries, qrels, eval_qids)
    print(f"  V2 (Fresh) Quality: NDCG@10={eval_v2_fresh['ndcg@10']:.4f}, MRR={eval_v2_fresh['mrr']:.4f}")

    # ---------------------------------------------------------
    # Verification
    # ---------------------------------------------------------
    assert eval_v2_inc['ndcg@10'] == eval_v2_fresh['ndcg@10'], "Mismatch in NDCG between incremental and fresh!"
    assert eval_v2_inc['mrr'] == eval_v2_fresh['mrr'], "Mismatch in MRR between incremental and fresh!"
    print("\n[VERIFIED] Incremental retrieval quality is perfectly mathematically equivalent to a fresh rebuild.")

    rows = [
        {"action": "Full Rebuild (V1)", "time_s": round(build_time_v1, 2), "ndcg@10": eval_v1['ndcg@10']},
        {"action": "Incremental Update (V2)", "time_s": round(build_time_inc, 2), "ndcg@10": eval_v2_inc['ndcg@10']},
        {"action": "Full Rebuild (V2)", "time_s": round(build_time_fresh, 2), "ndcg@10": eval_v2_fresh['ndcg@10']}
    ]

    # Save
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    path = results_dir / f"version_updates_{ts}.json"
    path.write_text(json.dumps({"timestamp": datetime.utcnow().isoformat(), "results": rows}, indent=2))
    print(f"\nResults saved to {path}", flush=True)

if __name__ == "__main__":
    main()
