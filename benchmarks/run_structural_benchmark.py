"""
Structural signals ablation benchmark.

Compares four configurations — all under identical conditions:

  A. Baseline: Dense-only, no structural signals
  B. Dense + Structural (callee+caller+import, default weights)
  C. Dense + Structural (callee+caller only, no import)
  D. Dense + Structural (callee-only, most conservative)

Additionally reports:
  - Structural index build time and size
  - Fraction of candidates that received a non-zero boost
  - Per-category MRR breakdown

AppsRetrieval expectation:
  The corpus has NO real function call relationships (each document is an
  independent Python solution). Structural boosts will be 0.0 for all
  candidates. The benchmark should show ZERO delta vs baseline — confirming
  the code degrades gracefully when AST fields are absent.

JS demo expectation:
  Real call graphs → structural boosts activate → measurable re-ranking.

Usage:
    python benchmarks/run_structural_benchmark.py --sample 200 --skip-build

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
from src.retrieval.structural_index import StructuralIndex, StructuralWeights


def run_retriever(
    name: str,
    retriever: AdaptiveRetriever,
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, float]],
    eval_qids: List[str],
    top_k: int = 100,
) -> dict:
    print(f"\n{'='*60}", flush=True)
    print(f"  {name}", flush=True)
    print(f"{'='*60}", flush=True)

    results_per_query: Dict[str, List[str]] = {}
    latencies: List[float] = []
    boost_counts = 0
    total_candidates = 0

    for qid in eval_qids:
        t = time.perf_counter()
        hits = retriever.search(queries[qid], top_k=top_k)
        latencies.append((time.perf_counter() - t) * 1000)
        results_per_query[qid] = [h.chunk_id for h in hits]

        for h in hits:
            total_candidates += 1
            if h.structural_boost > 0:
                boost_counts += 1

    metrics = compute_metrics(results_per_query, qrels, k=10)
    lat = np.array(latencies)
    boost_rate = boost_counts / total_candidates if total_candidates > 0 else 0.0

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
        "structural_boost_rate": round(boost_rate, 4),
        "structural_index_size_mb": round(
            retriever._structural.stats.index_size_bytes / (1024**2), 3
        ),
        "structural_build_time_s": retriever._structural.stats.build_time_s,
        "structural_call_edges": retriever._structural.stats.num_call_edges,
    }

    print(f"  NDCG@10={row['ndcg@10']}  MRR={row['mrr']}  "
          f"P50={row['latency_p50_ms']}ms  P95={row['latency_p95_ms']}ms", flush=True)
    print(f"  Structural boost_rate={boost_rate:.2%}  "
          f"call_edges={row['structural_call_edges']}  "
          f"build={row['structural_build_time_s']:.4f}s", flush=True)

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

    # Shared dense-only config (routing disabled = dense-only)
    cfg = HybridConfig(
        use_bm25=False, use_dense=True, use_symbol=False,
        embed_cache_dir=cache_dir,
    )

    # --- Report structural index stats on this corpus ---
    print("\nBuilding structural index (for stats)...", flush=True)
    si = StructuralIndex()
    si.build(chunks)
    print(f"  call_edges={si.stats.num_call_edges}  "
          f"import_groups={si.stats.num_import_groups}  "
          f"build={si.stats.build_time_s:.4f}s  "
          f"size={si.stats.index_size_bytes/1024:.1f} KB", flush=True)
    if si.stats.num_call_edges == 0:
        print("  NOTE: 0 call edges — corpus has no resolved call relationships.", flush=True)
        print("  Structural boosts will be 0 for all candidates (expected on AppsRetrieval).", flush=True)

    def make_retriever(sw: StructuralWeights) -> AdaptiveRetriever:
        ret = AdaptiveRetriever(cfg, embedding_provider=provider,
                                disable_routing=True, structural_weights=sw)
        ret.load_dense_index(index_dir)
        ret.build_structural(chunks)
        return ret

    configs = [
        ("Baseline: Dense-only (structural OFF)",
         StructuralWeights.disabled()),
        ("Dense + Structural (callee+caller+import, default)",
         StructuralWeights(callee_boost=0.005, caller_boost=0.003,
                           import_sibling_boost=0.001, enabled=True)),
        ("Dense + Structural (callee+caller, no import)",
         StructuralWeights(callee_boost=0.005, caller_boost=0.003,
                           import_sibling_boost=0.0, enabled=True)),
        ("Dense + Structural (callee-only)",
         StructuralWeights(callee_boost=0.005, caller_boost=0.0,
                           import_sibling_boost=0.0, enabled=True)),
    ]

    rows = []
    for name, sw in configs:
        ret = make_retriever(sw)
        row = run_retriever(name, ret, queries, qrels, eval_qids)
        row["structural_weights"] = {
            "callee_boost": sw.callee_boost,
            "caller_boost": sw.caller_boost,
            "import_sibling_boost": sw.import_sibling_boost,
            "enabled": sw.enabled,
        }
        rows.append(row)

    # -----------------------------------------------------------------------
    # Print comparison table
    # -----------------------------------------------------------------------
    cols = ["configuration", "ndcg@10", "mrr", "latency_p50_ms",
            "latency_p95_ms", "structural_boost_rate"]
    widths = [50, 9, 8, 14, 14, 22]
    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    sep = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print("STRUCTURAL SIGNALS ABLATION")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)
    for row in rows:
        line = "  ".join(str(row.get(c, "")).ljust(w) for c, w in zip(cols, widths))
        print(line)
    print(f"{'='*len(header)}")

    baseline_ndcg = rows[0]["ndcg@10"]
    baseline_mrr  = rows[0]["mrr"]
    print("\nDelta vs baseline:")
    for row in rows[1:]:
        d_n = round(row["ndcg@10"] - baseline_ndcg, 4)
        d_m = round(row["mrr"] - baseline_mrr, 4)
        print(f"  {row['configuration'][:50]:<50}  NDCG@10 {d_n:+.4f}  MRR {d_m:+.4f}")

    # Save
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out = {
        "timestamp": datetime.utcnow().isoformat(),
        "corpus_structural_stats": {
            "call_edges": si.stats.num_call_edges,
            "import_groups": si.stats.num_import_groups,
            "build_time_s": si.stats.build_time_s,
        },
        "results": rows,
    }
    path = results_dir / f"structural_{ts}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nResults saved to {path}", flush=True)


if __name__ == "__main__":
    main()
