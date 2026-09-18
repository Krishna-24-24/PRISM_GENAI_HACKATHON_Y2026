"""
Hybrid retrieval ablation experiment runner.

Evaluates every configuration under identical conditions:
  - Same dataset split (CoIR AppsRetrieval / test)
  - Same query sample (deterministic: sorted qids[:N])
  - Same top_k=100 candidate pool, metrics computed @10

Configurations tested:
  1. BM25 only
  2. Dense only          (BGE-small-en-v1.5, CPU)
  3. BM25 + Dense        (RRF k=60, uniform weights)
  4. BM25 + Dense + Symbol (RRF k=60, uniform weights)

This runner also sweeps RRF k values for the winning hybrid config.

Usage:
    python benchmarks/run_ablation.py --sample 200
    python benchmarks/run_ablation.py --sample 200 --skip-build   # reuse saved index

Output:
    results/ablation_<timestamp>.json   (machine-readable)
    results/ablation_<timestamp>.txt    (human-readable table)

DO NOT pre-fill any metric values. All numbers come from actual runs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.run_bm25_benchmark import corpus_to_chunks, load_apps_dataset
from src.chunking.models import CodeChunk
from src.evaluation.metrics import compute_metrics
from src.indexing.embedder import BGESmallProvider
from src.retrieval.hybrid_retriever import HybridConfig, HybridRetriever


# ---------------------------------------------------------------------------
# Shared dataset + chunk loading
# ---------------------------------------------------------------------------

def get_eval_data(sample: Optional[int] = None):
    print("Loading AppsRetrieval dataset...", flush=True)
    corpus, queries, qrels = load_apps_dataset()
    print(f"  corpus={len(corpus):,}  queries={len(queries):,}  qrels={len(qrels):,}", flush=True)

    chunks = corpus_to_chunks(corpus)

    eval_qids = sorted(qid for qid in qrels if qid in queries)
    if sample:
        eval_qids = eval_qids[:sample]

    return chunks, queries, qrels, eval_qids


# ---------------------------------------------------------------------------
# Run a single configuration
# ---------------------------------------------------------------------------

def run_config(
    name: str,
    config: HybridConfig,
    chunks: List[CodeChunk],
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, float]],
    eval_qids: List[str],
    provider: Optional[BGESmallProvider],
    shared_index_dir: Optional[str],
) -> Dict:
    print(f"\n{'='*60}", flush=True)
    print(f"  Configuration: {name}", flush=True)
    print(f"{'='*60}", flush=True)

    retriever = HybridRetriever(config, embedding_provider=provider)

    # Build or load
    build_start = time.perf_counter()
    if config.use_dense and shared_index_dir and os.path.exists(
        os.path.join(shared_index_dir, "index.faiss")
    ):
        print("  Loading pre-built FAISS index...", flush=True)
        if config.use_bm25:
            retriever._bm25.add_chunks(chunks)
        if config.use_symbol:
            retriever._symbol.add_chunks(chunks)
        retriever.load_dense_index(shared_index_dir)
    else:
        print("  Building index...", flush=True)
        retriever.build(chunks, show_progress=config.use_dense)

    build_time = time.perf_counter() - build_start
    index_mb = retriever.index_size_mb
    print(f"  Build: {build_time:.2f}s  index: {index_mb:.1f} MB", flush=True)

    # Query loop
    print(f"  Evaluating {len(eval_qids)} queries...", flush=True)
    results_per_query: Dict[str, List[str]] = {}
    latencies: List[float] = []

    for qid in eval_qids:
        t = time.perf_counter()
        hits = retriever.search(queries[qid], top_k=100)
        latencies.append((time.perf_counter() - t) * 1000)
        results_per_query[qid] = [h.chunk_id for h in hits]

    metrics = compute_metrics(results_per_query, qrels, k=10)
    lat = np.array(latencies)

    row = {
        "configuration": name,
        "channels": {
            "bm25": config.use_bm25,
            "dense": config.use_dense,
            "symbol": config.use_symbol,
        },
        "rrf_k": config.rrf_k,
        "weights": config.weights,
        "ndcg@10": metrics["ndcg@10"],
        "mrr": metrics["mrr"],
        "recall@1": metrics["recall@1"],
        "recall@10": metrics["recall@10"],
        "latency_avg_ms": round(float(np.mean(lat)), 2),
        "latency_p50_ms": round(float(np.percentile(lat, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(lat, 95)), 2),
        "latency_p99_ms": round(float(np.percentile(lat, 99)), 2),
        "index_size_mb": round(index_mb, 2),
        "index_build_time_s": round(build_time, 4),
        "queries_evaluated": len(eval_qids),
    }

    print(f"  NDCG@10={row['ndcg@10']}  MRR={row['mrr']}  "
          f"P50={row['latency_p50_ms']}ms  P95={row['latency_p95_ms']}ms", flush=True)
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=200,
                        help="Number of queries to evaluate (deterministic). Default: 200")
    parser.add_argument("--skip-build", action="store_true",
                        help="Reuse saved FAISS index from data/faiss_index/")
    parser.add_argument("--rrf-k-sweep", action="store_true",
                        help="Also sweep RRF k in {1,5,10,30,60,100} on best hybrid config")
    args = parser.parse_args()

    chunks, queries, qrels, eval_qids = get_eval_data(args.sample)

    # Single provider instance shared by all dense configs (model loaded once)
    print("\nLoading BGE-small-en-v1.5 on CPU...", flush=True)
    provider = BGESmallProvider(device="cpu")

    cache_dir = str(Path(__file__).parent.parent / "data" / "embed_cache")
    index_dir = str(Path(__file__).parent.parent / "data" / "faiss_index")

    # Pre-build dense index once — all configs that use dense reuse it
    if not args.skip_build:
        print("\nPre-building dense index (shared across configs)...", flush=True)
        from src.indexing.dense_index import DenseChannel
        dc = DenseChannel(provider, batch_size=32)
        dc.build(chunks, cache_dir=cache_dir, show_progress=True)
        dc.save(index_dir)
        print(f"Dense index saved to {index_dir}", flush=True)

    # -----------------------------------------------------------------------
    # Define configurations
    # -----------------------------------------------------------------------
    configs: List[Tuple[str, HybridConfig]] = [
        (
            "BM25 only",
            HybridConfig(use_bm25=True, use_dense=False, use_symbol=False),
        ),
        (
            "Dense only",
            HybridConfig(use_bm25=False, use_dense=True, use_symbol=False,
                         embed_cache_dir=cache_dir),
        ),
        (
            "BM25 + Dense (RRF k=60, uniform)",
            HybridConfig(use_bm25=True, use_dense=True, use_symbol=False,
                         rrf_k=60.0, embed_cache_dir=cache_dir,
                         weights={"bm25": 1.0, "dense": 1.0}),
        ),
        (
            "BM25 + Dense + Symbol (RRF k=60, uniform)",
            HybridConfig(use_bm25=True, use_dense=True, use_symbol=True,
                         rrf_k=60.0, embed_cache_dir=cache_dir,
                         weights={"bm25": 1.0, "dense": 1.0, "symbol": 1.0}),
        ),
    ]

    rows = []
    for name, cfg in configs:
        row = run_config(
            name, cfg, chunks, queries, qrels, eval_qids,
            provider=provider if cfg.use_dense else None,
            shared_index_dir=index_dir,
        )
        rows.append(row)

    # -----------------------------------------------------------------------
    # Optional RRF k sweep on best hybrid
    # -----------------------------------------------------------------------
    if args.rrf_k_sweep:
        print("\n--- RRF k sweep (BM25+Dense) ---", flush=True)
        for k in [1, 5, 10, 30, 60, 100]:
            cfg = HybridConfig(
                use_bm25=True, use_dense=True, use_symbol=False,
                rrf_k=float(k), embed_cache_dir=cache_dir,
                weights={"bm25": 1.0, "dense": 1.0},
            )
            row = run_config(
                f"BM25+Dense RRF k={k}", cfg, chunks, queries, qrels, eval_qids,
                provider=provider,
                shared_index_dir=index_dir,
            )
            rows.append(row)

    # -----------------------------------------------------------------------
    # Print table
    # -----------------------------------------------------------------------
    cols = ["configuration", "ndcg@10", "mrr", "recall@10",
            "latency_p50_ms", "latency_p95_ms", "index_size_mb"]
    widths = [45, 9, 8, 10, 14, 14, 13]

    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    sep = "-" * len(header)

    print(f"\n{'='*len(header)}", flush=True)
    print("ABLATION RESULTS", flush=True)
    print(f"{'='*len(header)}", flush=True)
    print(header, flush=True)
    print(sep, flush=True)
    for row in rows:
        line = "  ".join(
            str(row.get(c, "")).ljust(w) for c, w in zip(cols, widths)
        )
        print(line, flush=True)
    print(f"{'='*len(header)}", flush=True)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")

    json_path = results_dir / f"ablation_{ts}.json"
    json_path.write_text(json.dumps(rows, indent=2))

    txt_path = results_dir / f"ablation_{ts}.txt"
    lines = [header, sep] + [
        "  ".join(str(row.get(c, "")).ljust(w) for c, w in zip(cols, widths))
        for row in rows
    ]
    txt_path.write_text("\n".join(lines))

    print(f"\nResults saved to {json_path}", flush=True)
    print(f"Table saved to   {txt_path}", flush=True)


if __name__ == "__main__":
    main()
