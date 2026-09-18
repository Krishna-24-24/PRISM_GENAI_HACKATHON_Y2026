"""
Dense retrieval benchmark on CoIR AppsRetrieval (test split).

Model: BAAI/bge-small-en-v1.5
  - 33M parameters, 384-dim embeddings
  - Designed for CPU inference: quantization-friendly, low memory footprint
  - BGE query prefix applied: "Represent this sentence for searching relevant passages: <query>"
  - Corpus encoded without prefix (asymmetric encoding per model card)

Usage:
    python benchmarks/run_dense_benchmark.py                        # full eval, embed from scratch
    python benchmarks/run_dense_benchmark.py --sample 200          # fast smoke-test
    python benchmarks/run_dense_benchmark.py --load-index data/faiss_index  # reuse built index

Metrics reported: NDCG@10, MRR, Recall@1, Recall@10,
                  embedding throughput, index build time, query latency, index size.

IMPORTANT: No numbers are pre-filled. All figures come from the actual run.
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

from benchmarks.run_bm25_benchmark import corpus_to_chunks, load_apps_dataset
from src.evaluation.metrics import compute_metrics
from src.indexing.dense_index import DenseChannel
from src.indexing.embedder import BGESmallProvider


def run(sample: int | None = None, load_index: str | None = None, save_index: str | None = None) -> dict:
    print("Loading AppsRetrieval dataset...")
    corpus, queries, qrels = load_apps_dataset()
    print(f"  corpus={len(corpus):,}  queries={len(queries):,}  qrels={len(qrels):,}")

    # --- Provider ---
    print("Initialising BGE-small-en-v1.5 on CPU...")
    provider = BGESmallProvider(device="cpu")
    channel = DenseChannel(provider, batch_size=32)

    cache_dir = str(Path(__file__).parent.parent / "data" / "embed_cache")
    index_dir = save_index or str(Path(__file__).parent.parent / "data" / "faiss_index")

    if load_index and os.path.exists(os.path.join(load_index, "index.faiss")):
        print(f"Loading pre-built index from {load_index}...")
        t0 = time.time()
        channel.load(load_index)
        build_time = time.time() - t0
        embed_throughput = 0.0  # not re-embedded
        print(f"  Loaded {channel.num_vectors:,} vectors in {build_time:.2f}s")
    else:
        # --- Build ---
        chunks = corpus_to_chunks(corpus)
        print(f"Embedding {len(chunks):,} chunks (cache: {cache_dir})...")
        channel.build(chunks, cache_dir=cache_dir, show_progress=True)
        build_time = channel.build_time
        embed_throughput = channel.embedding_throughput
        print(f"  Build: {build_time:.2f}s  throughput: {embed_throughput:.1f} chunks/s")

        # Save for reuse
        channel.save(index_dir)
        print(f"  Index saved to {index_dir}")

    index_mb = channel.get_index_size_bytes() / (1024 ** 2)
    print(f"  Index size: {index_mb:.1f} MB  vectors: {channel.num_vectors:,}")

    # --- Query loop ---
    eval_qids = [qid for qid in qrels if qid in queries]
    if sample:
        eval_qids = sorted(eval_qids)[:sample]

    print(f"Evaluating {len(eval_qids)} queries (top_k=100)...")
    results_per_query: dict[str, list[str]] = {}
    latencies: list[float] = []

    for qid in eval_qids:
        q_text = queries[qid]
        t_q = time.time()
        hits = channel.search(q_text, top_k=100)
        latencies.append(time.time() - t_q)
        results_per_query[qid] = [h.chunk_id for h in hits]

    # --- Metrics ---
    metrics = compute_metrics(results_per_query, qrels, k=10)
    lat_ms = [l * 1000 for l in latencies]

    summary = {
        "model": provider.model_name,
        "embedding_dim": provider.dim,
        "index_type": "FAISS IndexFlatIP (exact cosine)",
        "corpus_size": len(corpus),
        "queries_evaluated": len(eval_qids),
        "index_build_time_s": round(build_time, 4),
        "embedding_throughput_chunks_per_s": round(embed_throughput, 2),
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

    print("\n=== DENSE BENCHMARK RESULTS ===")
    for k, v in summary.items():
        print(f"  {k:<40} {v}")

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out_path = results_dir / f"dense_{ts}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {out_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="Evaluate only the first N queries (deterministic).")
    parser.add_argument("--load-index", type=str, default=None,
                        help="Path to pre-built FAISS index directory (skips embedding).")
    parser.add_argument("--save-index", type=str, default=None,
                        help="Override path to save the built index.")
    args = parser.parse_args()
    run(sample=args.sample, load_index=args.load_index, save_index=args.save_index)
