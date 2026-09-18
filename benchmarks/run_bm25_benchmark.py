"""
BM25 retrieval benchmark on CoIR AppsRetrieval (test split).

Usage:
    python benchmarks/run_bm25_benchmark.py            # full 3765-query eval
    python benchmarks/run_bm25_benchmark.py --sample 200   # fast smoke-test

Metrics reported: NDCG@10, MRR, Recall@1, Recall@10, latency P50/P99/avg.
Results are saved to results/bm25_<timestamp>.json.

IMPORTANT: No benchmark number is claimed here until this script is actually executed
and the JSON output is committed alongside the code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.chunking.models import CodeChunk
from src.evaluation.metrics import compute_metrics
from src.indexing.bm25_index import BM25Channel


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_apps_dataset(split: str = "test"):
    """Load CoIR-Retrieval/apps from HuggingFace (or local cache)."""
    from datasets import load_dataset

    corpus_ds = load_dataset("CoIR-Retrieval/apps", "corpus", split="corpus")
    queries_ds = load_dataset("CoIR-Retrieval/apps", "queries", split="queries")
    qrels_ds = load_dataset("CoIR-Retrieval/apps", "default", split=split)

    # corpus: {doc_id: {"text": ..., "title": ...}}
    corpus = {str(d["_id"]): d for d in corpus_ds}
    # queries: {q_id: text}
    queries = {str(d["_id"]): d["text"] for d in queries_ds}
    # qrels: {q_id: {doc_id: score}}
    qrels: dict[str, dict[str, float]] = {}
    for row in qrels_ds:
        qid = str(row["query-id"])
        did = str(row["corpus-id"])
        qrels.setdefault(qid, {})[did] = float(row["score"])

    return corpus, queries, qrels


def corpus_to_chunks(corpus: dict) -> list[CodeChunk]:
    chunks = []
    for doc_id, doc in corpus.items():
        text = doc.get("text", "")
        title = doc.get("title", "")
        chunks.append(CodeChunk(
            chunk_id=doc_id,
            repo_id="appsretrieval",
            version_id="1",
            file_path="corpus.py",
            symbol_name=title or doc_id,
            symbol_type="module",
            start_line=1,
            end_line=max(1, text.count("\n") + 1),
            code=text,
            normalized_code=text,
            context=title,
            imports=[],
            exports=[],
            calls=[],
            called_by=[],
            chunk_hash=doc_id,
        ))
    return chunks


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run(sample: int | None = None) -> dict:
    print("Loading AppsRetrieval dataset...")
    corpus, queries, qrels = load_apps_dataset()
    print(f"  corpus={len(corpus):,}  queries={len(queries):,}  qrels={len(qrels):,}")

    # --- Build index ---
    chunks = corpus_to_chunks(corpus)
    index = BM25Channel()
    print("Building BM25 index...", end=" ", flush=True)
    t0 = time.time()
    index.add_chunks(chunks)
    build_time = time.time() - t0
    index_mb = index.get_index_size_bytes() / (1024 ** 2)
    print(f"done. {build_time:.2f}s  {index_mb:.1f} MB")

    # --- Query loop ---
    eval_qids = [qid for qid in qrels if qid in queries]
    if sample:
        # Deterministic subset: sort → take first N
        eval_qids = sorted(eval_qids)[:sample]

    print(f"Evaluating {len(eval_qids)} queries (top_k=100)...")
    results_per_query: dict[str, list[str]] = {}
    latencies: list[float] = []

    for qid in eval_qids:
        q_text = queries[qid]
        t_q = time.time()
        hits = index.search(q_text, top_k=100)
        latencies.append(time.time() - t_q)
        results_per_query[qid] = [h.chunk_id for h in hits]

    # --- Metrics ---
    metrics = compute_metrics(results_per_query, qrels, k=10)

    lat_ms = [l * 1000 for l in latencies]
    summary = {
        "model": "BM25 (rank_bm25/BM25Okapi)",
        "tokeniser": "code-aware (camelCase + snake_case split)",
        "corpus_size": len(corpus),
        "queries_evaluated": len(eval_qids),
        "index_build_time_s": round(build_time, 4),
        "index_size_mb": round(index_mb, 2),
        "ndcg@10": metrics["ndcg@10"],
        "mrr": metrics["mrr"],
        "recall@1": metrics["recall@1"],
        "recall@10": metrics["recall@10"],
        "latency_avg_ms": round(float(np.mean(lat_ms)), 2),
        "latency_p50_ms": round(float(np.percentile(lat_ms, 50)), 2),
        "latency_p99_ms": round(float(np.percentile(lat_ms, 99)), 2),
        "timestamp": datetime.utcnow().isoformat(),
    }

    # --- Print ---
    print("\n=== BM25 BENCHMARK RESULTS ===")
    for k, v in summary.items():
        print(f"  {k:<30} {v}")

    # --- Save ---
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out_path = results_dir / f"bm25_{ts}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {out_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="Evaluate only the first N queries (deterministic). "
                             "Use for fast smoke-tests. Omit for full eval.")
    args = parser.parse_args()
    run(sample=args.sample)
