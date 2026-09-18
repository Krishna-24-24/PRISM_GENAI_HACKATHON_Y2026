"""
Adaptive routing benchmark: baseline hybrid vs adaptive routing.

This script answers the only question that matters:
  Does adaptive routing improve NDCG@10 / MRR over the best baseline?

Best baseline (from ablation_20260917T123339.json):
  Dense only: NDCG@10=0.0498, MRR=0.0459, P50=164ms

Configurations compared:
  A. Baseline: Dense-only     (disable_routing=True, use_dense=True, use_bm25=False)
  B. Adaptive: routing ON     (QueryAnalyzer selects channels per query)
  C. Adaptive+BM25+Symbol:    all channels built, routing selects weights per query

Usage:
    python benchmarks/run_adaptive_benchmark.py --sample 200
    python benchmarks/run_adaptive_benchmark.py --sample 200 --skip-build

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
from src.chunking.models import CodeChunk
from src.evaluation.metrics import compute_metrics
from src.indexing.embedder import BGESmallProvider
from src.retrieval.adaptive_retriever import AdaptiveRetriever
from src.retrieval.hybrid_retriever import HybridConfig
from src.retrieval.query_analyzer import QueryAnalyzer, QueryCategory


# ---------------------------------------------------------------------------
# Category distribution analysis
# ---------------------------------------------------------------------------

def analyze_query_distribution(queries: Dict[str, str], eval_qids: List[str]) -> dict:
    """Classify all eval queries and report category distribution."""
    analyzer = QueryAnalyzer()
    counts: Dict[str, int] = {}
    for qid in eval_qids:
        cat = analyzer.analyze(queries[qid]).category.value
        counts[cat] = counts.get(cat, 0) + 1
    total = len(eval_qids)
    return {cat: {"count": c, "pct": round(100 * c / total, 1)} for cat, c in sorted(counts.items())}


# ---------------------------------------------------------------------------
# Run a single retriever config
# ---------------------------------------------------------------------------

def run_retriever(
    name: str,
    retriever: AdaptiveRetriever,
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, float]],
    eval_qids: List[str],
    top_k: int = 100,
) -> dict:
    print(f"\n{'='*55}", flush=True)
    print(f"  {name}", flush=True)
    print(f"{'='*55}", flush=True)

    results_per_query: Dict[str, List[str]] = {}
    latencies: List[float] = []
    category_hits: Dict[str, List[float]] = {}   # category → list of reciprocal ranks

    for qid in eval_qids:
        q_text = queries[qid]
        t = time.perf_counter()
        hits = retriever.search(q_text, top_k=top_k)
        latencies.append((time.perf_counter() - t) * 1000)
        results_per_query[qid] = [h.chunk_id for h in hits]

        # Track per-category MRR for diagnostic purposes
        if hits and hits[0].query_profile is not None:
            cat = hits[0].query_profile.category.value
            rel_ids = set(qrels.get(qid, {}).keys())
            rr = 0.0
            for rank, h in enumerate(hits[:10], 1):
                if h.chunk_id in rel_ids:
                    rr = 1.0 / rank
                    break
            category_hits.setdefault(cat, []).append(rr)

    metrics = compute_metrics(results_per_query, qrels, k=10)
    lat = np.array(latencies)

    # Per-category MRR
    per_cat_mrr = {
        cat: round(float(np.mean(rrs)), 4)
        for cat, rrs in category_hits.items()
    }

    row = {
        "configuration": name,
        "ndcg@10": metrics["ndcg@10"],
        "mrr": metrics["mrr"],
        "recall@1": metrics["recall@1"],
        "recall@10": metrics["recall@10"],
        "latency_avg_ms": round(float(np.mean(lat)), 2),
        "latency_p50_ms": round(float(np.percentile(lat, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(lat, 95)), 2),
        "queries_evaluated": len(eval_qids),
        "per_category_mrr": per_cat_mrr,
    }

    print(f"  NDCG@10={row['ndcg@10']}  MRR={row['mrr']}  "
          f"P50={row['latency_p50_ms']}ms  P95={row['latency_p95_ms']}ms", flush=True)
    if per_cat_mrr:
        print(f"  Per-category MRR: {per_cat_mrr}", flush=True)

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--skip-build", action="store_true",
                        help="Load pre-built FAISS index from data/faiss_index/")
    args = parser.parse_args()

    # Load data
    print("Loading AppsRetrieval dataset...", flush=True)
    corpus, queries, qrels = load_apps_dataset()
    chunks = corpus_to_chunks(corpus)
    eval_qids = sorted(qid for qid in qrels if qid in queries)[:args.sample]
    print(f"  corpus={len(corpus):,}  eval_qids={len(eval_qids)}", flush=True)

    # Query distribution analysis
    dist = analyze_query_distribution(queries, eval_qids)
    print(f"\nQuery category distribution (n={len(eval_qids)}):")
    for cat, info in dist.items():
        print(f"  {cat:<12} {info['count']:>4}  ({info['pct']}%)")

    # Provider (loaded once, shared across all configs)
    print("\nLoading BGE-small-en-v1.5 on CPU...", flush=True)
    provider = BGESmallProvider(device="cpu")

    cache_dir = str(Path(__file__).parent.parent / "data" / "embed_cache")
    index_dir = str(Path(__file__).parent.parent / "data" / "faiss_index")

    # Build/load shared dense index
    if not args.skip_build:
        from src.indexing.dense_index import DenseChannel
        print("\nBuilding shared dense index...", flush=True)
        dc = DenseChannel(provider, batch_size=32)
        dc.build(chunks, cache_dir=cache_dir, show_progress=True)
        dc.save(index_dir)
        print(f"Dense index saved: {index_dir}", flush=True)

    # -----------------------------------------------------------------------
    # Configuration A: Baseline — Dense only (routing disabled)
    # -----------------------------------------------------------------------
    cfg_baseline = HybridConfig(
        use_bm25=False, use_dense=True, use_symbol=False,
        embed_cache_dir=cache_dir,
    )
    retriever_a = AdaptiveRetriever(cfg_baseline, embedding_provider=provider,
                                    disable_routing=True)
    retriever_a.load_dense_index(index_dir)
    row_a = run_retriever("Baseline: Dense-only (routing OFF)",
                          retriever_a, queries, qrels, eval_qids)

    # -----------------------------------------------------------------------
    # Configuration B: Adaptive — Dense + Symbol, routing ON
    # (BM25 excluded: ablation shows it hurts Dense via RRF dilution)
    # -----------------------------------------------------------------------
    cfg_adaptive_ds = HybridConfig(
        use_bm25=False, use_dense=True, use_symbol=True,
        embed_cache_dir=cache_dir,
        # Default weights; QueryAnalyzer overrides per query
        weights={"dense": 1.0, "symbol": 1.0},
    )
    retriever_b = AdaptiveRetriever(cfg_adaptive_ds, embedding_provider=provider,
                                    disable_routing=False)
    retriever_b.load_dense_index(index_dir)
    retriever_b._retriever._symbol.add_chunks(chunks)  # build symbol index
    row_b = run_retriever("Adaptive: Dense+Symbol (routing ON, BM25 excluded)",
                          retriever_b, queries, qrels, eval_qids)

    # -----------------------------------------------------------------------
    # Configuration C: Adaptive — all channels, routing ON
    # (BM25 included but only activated on symbolic/structural queries)
    # -----------------------------------------------------------------------
    cfg_adaptive_all = HybridConfig(
        use_bm25=True, use_dense=True, use_symbol=True,
        embed_cache_dir=cache_dir,
        weights={"dense": 1.0, "bm25": 1.0, "symbol": 1.0},
    )
    retriever_c = AdaptiveRetriever(cfg_adaptive_all, embedding_provider=provider,
                                    disable_routing=False)
    retriever_c.load_dense_index(index_dir)
    retriever_c._retriever._bm25.add_chunks(chunks)
    retriever_c._retriever._symbol.add_chunks(chunks)
    row_c = run_retriever("Adaptive: Dense+BM25+Symbol (routing ON, all channels)",
                          retriever_c, queries, qrels, eval_qids)

    rows = [row_a, row_b, row_c]

    # -----------------------------------------------------------------------
    # Comparison table
    # -----------------------------------------------------------------------
    cols = ["configuration", "ndcg@10", "mrr", "recall@10",
            "latency_p50_ms", "latency_p95_ms"]
    widths = [50, 9, 8, 10, 14, 14]
    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    sep = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print("ADAPTIVE ROUTING BENCHMARK RESULTS")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)
    for row in rows:
        line = "  ".join(str(row.get(c, "")).ljust(w) for c, w in zip(cols, widths))
        print(line)
    print(f"{'='*len(header)}")

    # Delta vs baseline
    print(f"\nDelta vs baseline (Dense-only):")
    baseline_ndcg = row_a["ndcg@10"]
    baseline_mrr  = row_a["mrr"]
    for row in rows[1:]:
        d_ndcg = round(row["ndcg@10"] - baseline_ndcg, 4)
        d_mrr  = round(row["mrr"]     - baseline_mrr,  4)
        sign_n = "+" if d_ndcg >= 0 else ""
        sign_m = "+" if d_mrr  >= 0 else ""
        print(f"  {row['configuration'][:50]:<50} "
              f"NDCG@10 {sign_n}{d_ndcg:+.4f}   MRR {sign_m}{d_mrr:+.4f}")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")

    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "query_distribution": dist,
        "results": rows,
        "baseline": {
            "configuration": "Dense-only (ablation 2026-09-17)",
            "ndcg@10": 0.0498,
            "mrr": 0.0459,
            "source": "results/ablation_20260917T123339.json",
        },
    }
    out_path = results_dir / f"adaptive_{ts}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
