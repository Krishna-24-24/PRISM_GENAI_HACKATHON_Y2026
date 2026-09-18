"""
Candidate fusion strategies.

Two strategies are implemented:

1. ReciprocaLRankFusion (RRF)
   score(d) = Σ_c  w_c / (k + rank_c(d))

   - Rank-based: immune to score distribution differences between channels
   - k=60 is the Cormack et al. default (tunable via E-RRFK ablation)
   - Weighted: channel weights come from QueryProfile or are uniform (1.0)
   - Documents absent from a channel get rank = len(channel_results) + 1

2. LinearFusion
   score(d) = Σ_c  w_c * normalised_score_c(d)

   - Requires score normalisation (min-max per channel)
   - Sensitive to outlier scores; included for ablation (E1 in plan)
   - Not recommended as primary strategy without calibration evidence

Both implement the CandidateFuser protocol so they are drop-in swappable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class RawCandidate:
    """A scored document from any retrieval channel."""
    chunk_id: str
    score: float
    channel: str
    chunk: Optional[Any] = None  # CodeChunk, attached when available


@dataclass
class FusedCandidate:
    """Post-fusion ranked entry."""
    chunk_id: str
    fused_score: float
    channel_contributions: Dict[str, float] = field(default_factory=dict)
    chunk: Optional[Any] = None


class CandidateFuser(Protocol):
    def fuse(
        self,
        channel_results: Dict[str, List[RawCandidate]],
        weights: Optional[Dict[str, float]] = None,
    ) -> List[FusedCandidate]: ...


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

class RRFFuser:
    """
    Weighted Reciprocal Rank Fusion.

    score(d) = Σ_c  w_c / (k + rank_c(d))

    Args:
        k: damping constant (default 60, from Cormack et al. 2009).
           Smaller k → sharper rank discrimination; sweep via E-RRFK.
    """

    def __init__(self, k: float = 60.0) -> None:
        self.k = k

    def fuse(
        self,
        channel_results: Dict[str, List[RawCandidate]],
        weights: Optional[Dict[str, float]] = None,
    ) -> List[FusedCandidate]:
        if not channel_results:
            return []

        # Default uniform weights
        w = weights or {ch: 1.0 for ch in channel_results}

        # Build rank maps: channel → {chunk_id: rank (1-based)}
        rank_maps: Dict[str, Dict[str, int]] = {}
        chunk_store: Dict[str, Any] = {}
        for channel, candidates in channel_results.items():
            rank_maps[channel] = {}
            for rank, cand in enumerate(candidates, start=1):
                # Only record first (best) rank for duplicate chunk_ids in same channel
                if cand.chunk_id not in rank_maps[channel]:
                    rank_maps[channel][cand.chunk_id] = rank
                if cand.chunk is not None:
                    chunk_store[cand.chunk_id] = cand.chunk

        # Collect all unique chunk_ids across channels
        all_ids = set()
        for cands in channel_results.values():
            all_ids.update(c.chunk_id for c in cands)

        # Score each document
        fused: Dict[str, FusedCandidate] = {}
        for chunk_id in all_ids:
            total = 0.0
            contributions: Dict[str, float] = {}
            for channel, rank_map in rank_maps.items():
                channel_w = w.get(channel, 1.0)
                rank = rank_map.get(chunk_id, len(channel_results[channel]) + 1)
                contrib = channel_w / (self.k + rank)
                total += contrib
                contributions[channel] = contrib

            fused[chunk_id] = FusedCandidate(
                chunk_id=chunk_id,
                fused_score=total,
                channel_contributions=contributions,
                chunk=chunk_store.get(chunk_id),
            )

        return sorted(fused.values(), key=lambda x: x.fused_score, reverse=True)


# ---------------------------------------------------------------------------
# Linear (score-based) Fusion — for ablation only
# ---------------------------------------------------------------------------

class LinearFuser:
    """
    Weighted linear combination of min-max normalised channel scores.
    Use only in ablation experiments — requires well-calibrated scores.
    """

    def fuse(
        self,
        channel_results: Dict[str, List[RawCandidate]],
        weights: Optional[Dict[str, float]] = None,
    ) -> List[FusedCandidate]:
        if not channel_results:
            return []

        w = weights or {ch: 1.0 for ch in channel_results}

        # Normalise each channel's scores to [0, 1]
        normalised: Dict[str, Dict[str, float]] = {}
        chunk_store: Dict[str, Any] = {}
        for channel, candidates in channel_results.items():
            if not candidates:
                normalised[channel] = {}
                continue
            scores = [c.score for c in candidates]
            mn, mx = min(scores), max(scores)
            rng = mx - mn if mx != mn else 1.0
            normalised[channel] = {c.chunk_id: (c.score - mn) / rng for c in candidates}
            for c in candidates:
                if c.chunk is not None:
                    chunk_store[c.chunk_id] = c.chunk

        all_ids = set()
        for cands in channel_results.values():
            all_ids.update(c.chunk_id for c in cands)

        fused: Dict[str, FusedCandidate] = {}
        for chunk_id in all_ids:
            total = 0.0
            contributions: Dict[str, float] = {}
            for channel in channel_results:
                contrib = w.get(channel, 1.0) * normalised[channel].get(chunk_id, 0.0)
                total += contrib
                contributions[channel] = contrib
            fused[chunk_id] = FusedCandidate(
                chunk_id=chunk_id,
                fused_score=total,
                channel_contributions=contributions,
                chunk=chunk_store.get(chunk_id),
            )

        return sorted(fused.values(), key=lambda x: x.fused_score, reverse=True)
