"""
Shared retrieval quality metrics.

NDCG@10 and MRR follow the standard IR definitions:
  - MRR   = mean of (1/rank_of_first_relevant) across queries
  - NDCG@10 = mean of (DCG@10 / IDCG@10) across queries

All values are in [0, 1]. Higher is better.
Results are NOT fabricated — they are computed from actual search output.
"""

import math
from typing import Dict, List, Tuple


def dcg_at_k(ranked_ids: List[str], relevant: Dict[str, float], k: int) -> float:
    """Discounted Cumulative Gain at rank k."""
    dcg = 0.0
    for i, doc_id in enumerate(ranked_ids[:k]):
        if doc_id in relevant:
            rel = relevant[doc_id]
            dcg += (2.0 ** rel - 1.0) / math.log2(i + 2)
    return dcg


def ideal_dcg_at_k(relevant: Dict[str, float], k: int) -> float:
    """Ideal DCG — corpus sorted by descending relevance."""
    sorted_rels = sorted(relevant.values(), reverse=True)
    idcg = 0.0
    for i, rel in enumerate(sorted_rels[:k]):
        idcg += (2.0 ** rel - 1.0) / math.log2(i + 2)
    return idcg


def ndcg_at_k(ranked_ids: List[str], relevant: Dict[str, float], k: int = 10) -> float:
    idcg = ideal_dcg_at_k(relevant, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(ranked_ids, relevant, k) / idcg


def reciprocal_rank(ranked_ids: List[str], relevant: Dict[str, float]) -> float:
    for i, doc_id in enumerate(ranked_ids):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def compute_metrics(
    results_per_query: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, float]],
    k: int = 10,
) -> Dict[str, float]:
    """
    Args:
        results_per_query: {query_id: [ranked_doc_id, ...]} — full ranked list, ≥ k entries
        qrels:             {query_id: {doc_id: relevance_score}}
        k:                 cutoff for NDCG

    Returns dict with ndcg@k, mrr, recall@1, recall@k, num_queries.
    """
    ndcg_sum = 0.0
    mrr_sum = 0.0
    recall1_sum = 0.0
    recallk_sum = 0.0
    count = 0

    for q_id, ranked in results_per_query.items():
        if q_id not in qrels:
            continue
        relevant = qrels[q_id]
        if not relevant:
            continue

        count += 1
        ndcg_sum += ndcg_at_k(ranked, relevant, k)
        mrr_sum += reciprocal_rank(ranked, relevant)

        top1_set = set(ranked[:1])
        topk_set = set(ranked[:k])
        rel_set = set(relevant.keys())
        recall1_sum += 1.0 if top1_set & rel_set else 0.0
        recallk_sum += 1.0 if topk_set & rel_set else 0.0

    if count == 0:
        return {"ndcg@10": 0.0, "mrr": 0.0, "recall@1": 0.0, "recall@10": 0.0, "num_queries": 0}

    return {
        f"ndcg@{k}": round(ndcg_sum / count, 4),
        "mrr": round(mrr_sum / count, 4),
        "recall@1": round(recall1_sum / count, 4),
        f"recall@{k}": round(recallk_sum / count, 4),
        "num_queries": count,
    }
