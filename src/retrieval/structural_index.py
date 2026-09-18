"""
StructuralIndex — call graph and import graph built from CodeChunk AST fields.

What this does:
  Build a directed graph from `calls` and `called_by` lists extracted by the
  JS AST parser. Also index `imports` per chunk for import co-citation scoring.

What this does NOT do:
  - Generate answers
  - Explain code
  - Anything after retrieval ranking

The graph is used exclusively to compute structural proximity scores that are
added as a small additive boost to existing RRF-fused candidate scores.

Evidence basis for this approach:
  The call graph encodes relationships the embedding model cannot capture:
  - Function A is more relevant if it calls the exact function the query names
  - Function B is more relevant if it is called by other top-ranked candidates
  - Two chunks that import the same module are structurally co-located

Graph construction:
  - Nodes: chunk_id
  - Directed edges: chunk_id_A → chunk_id_B when A.calls contains B.symbol_name
  - Undirected co-import edges: chunk_id_A — chunk_id_B when they share ≥1 import

Complexity:
  - Build: O(E) where E = sum of len(calls) across all chunks
  - Query scoring: O(k * avg_degree) where k = number of candidates
  - Memory: ~200 bytes per edge, negligible for JS codebases ≤ 100k functions
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set

from src.chunking.models import CodeChunk


@dataclass
class StructuralStats:
    num_chunks: int = 0
    num_call_edges: int = 0
    num_import_groups: int = 0
    build_time_s: float = 0.0
    index_size_bytes: int = 0


class StructuralIndex:
    """
    Lightweight call graph + import co-citation index.

    After building, use `score_candidates()` to get per-chunk additive boosts.
    """

    def __init__(self) -> None:
        # symbol_name (lowercase) → set of chunk_ids that define it
        self._name_to_chunks: Dict[str, Set[str]] = defaultdict(set)

        # chunk_id → set of chunk_ids this chunk calls (resolved)
        self._calls_out: Dict[str, Set[str]] = defaultdict(set)

        # chunk_id → set of chunk_ids that call this chunk (resolved)
        self._calls_in: Dict[str, Set[str]] = defaultdict(set)

        # frozenset of import strings → set of chunk_ids sharing them
        # We use per-import indexing for O(1) lookup
        self._import_to_chunks: Dict[str, Set[str]] = defaultdict(set)

        self._chunks: Dict[str, CodeChunk] = {}
        self.stats = StructuralStats()

    def build(self, chunks: List[CodeChunk]) -> None:
        t0 = time.perf_counter()

        # Pass 1: index symbol names and imports
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
            name = (chunk.symbol_name or "").strip().lower()
            if name and name != "anonymous":
                self._name_to_chunks[name].add(chunk.chunk_id)

            for imp in chunk.imports:
                # Normalise: take the module specifier substring
                key = _normalise_import(imp)
                if key:
                    self._import_to_chunks[key].add(chunk.chunk_id)

        # Pass 2: resolve call edges
        for chunk in chunks:
            for callee_raw in chunk.calls:
                callee_name = _normalise_identifier(callee_raw)
                if not callee_name:
                    continue
                callee_ids = self._name_to_chunks.get(callee_name, set())
                for cid in callee_ids:
                    if cid != chunk.chunk_id:
                        self._calls_out[chunk.chunk_id].add(cid)
                        self._calls_in[cid].add(chunk.chunk_id)

        build_time = time.perf_counter() - t0
        total_edges = sum(len(v) for v in self._calls_out.values())

        self.stats = StructuralStats(
            num_chunks=len(chunks),
            num_call_edges=total_edges,
            num_import_groups=len(self._import_to_chunks),
            build_time_s=round(build_time, 4),
            index_size_bytes=self._estimate_size_bytes(),
        )

        # Backfill called_by on each chunk using resolved incoming edges.
        # Without this, all chunks appear to be dead code (called_by=[]).
        for cid, callers in self._calls_in.items():
            if cid in self._chunks:
                self._chunks[cid].called_by = list(callers)

    def _estimate_size_bytes(self) -> int:
        # Rough: 60 bytes per chunk_id string, 3 dicts
        n = len(self._chunks)
        edges = sum(len(v) for v in self._calls_out.values())
        imports = sum(len(v) for v in self._import_to_chunks.values())
        return (n * 60) + (edges * 120) + (imports * 60)

    def get_callers(self, chunk_id: str) -> Set[str]:
        """Return chunk_ids that call the given chunk."""
        return self._calls_in.get(chunk_id, set())

    def get_callees(self, chunk_id: str) -> Set[str]:
        """Return chunk_ids called by the given chunk."""
        return self._calls_out.get(chunk_id, set())

    def get_import_siblings(self, chunk_id: str) -> Set[str]:
        """Return chunk_ids sharing at least one import with the given chunk."""
        chunk = self._chunks.get(chunk_id)
        if not chunk:
            return set()
        siblings: Set[str] = set()
        for imp in chunk.imports:
            key = _normalise_import(imp)
            if key:
                siblings.update(self._import_to_chunks.get(key, set()))
        siblings.discard(chunk_id)
        return siblings

    def score_candidates(
        self,
        candidate_ids: List[str],
        primary_scores: Dict[str, float],
        weights: "StructuralWeights",
        top_anchor: int = 5,
    ) -> Dict[str, float]:
        """
        Compute structural boost scores for each candidate.

        Strategy:
          1. Identify "anchor" chunks: top `top_anchor` candidates by primary score
          2. Any candidate that is a caller/callee of an anchor gets a boost
          3. Any candidate that shares imports with an anchor gets a smaller boost
          4. Boosts are additive and scaled to be << primary_scores
             (structural never overrides semantic — it only nudges)

        Returns dict: chunk_id → structural_boost (always ≥ 0)
        """
        if not candidate_ids or not self._chunks:
            return {cid: 0.0 for cid in candidate_ids}

        # Anchor set: top-k by primary score
        sorted_by_primary = sorted(
            [c for c in candidate_ids if c in primary_scores],
            key=lambda c: primary_scores[c],
            reverse=True,
        )
        anchors = set(sorted_by_primary[:top_anchor])

        # Collect structural neighbours of anchors
        callee_of_anchor: Set[str] = set()
        caller_of_anchor: Set[str] = set()
        import_sibling_of_anchor: Set[str] = set()

        for anchor in anchors:
            callee_of_anchor.update(self.get_callees(anchor))
            caller_of_anchor.update(self.get_callers(anchor))
            if weights.import_sibling_boost > 0:
                import_sibling_of_anchor.update(self.get_import_siblings(anchor))

        boosts: Dict[str, float] = {}
        for cid in candidate_ids:
            b = 0.0
            if cid in callee_of_anchor:
                b += weights.callee_boost
            if cid in caller_of_anchor:
                b += weights.caller_boost
            if cid in import_sibling_of_anchor:
                b += weights.import_sibling_boost
            boosts[cid] = b

        return boosts


# ---------------------------------------------------------------------------
# Structural scoring weights
# ---------------------------------------------------------------------------

@dataclass
class StructuralWeights:
    """
    Additive boost magnitudes for each structural signal.

    These are expressed relative to RRF scores (typical range 0.01–0.03).
    Start conservative: max boost << typical primary score difference.

    Evidence: all weights start at 0 and are tuned in ablation.
    Values here are the first hypothesis — to be measured.
    """
    # A chunk that is called by a top-ranked candidate is more relevant
    callee_boost: float = 0.005

    # A chunk that calls a top-ranked candidate is more relevant
    caller_boost: float = 0.003

    # A chunk sharing an import with a top-ranked candidate is weakly relevant
    import_sibling_boost: float = 0.001

    # How many top candidates to treat as "anchors" for neighbour propagation
    top_anchor: int = 5

    # Master on/off switch
    enabled: bool = True

    @classmethod
    def disabled(cls) -> "StructuralWeights":
        return cls(callee_boost=0.0, caller_boost=0.0,
                   import_sibling_boost=0.0, enabled=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_identifier(raw: str) -> str:
    """
    Extract bare function name from call expressions.
    'obj.method' → 'method'
    'this.helper' → 'helper'
    'require("x")' → skip
    """
    if not raw:
        return ""
    # Take last component of dotted name
    parts = raw.strip().split(".")
    name = parts[-1].lower().strip()
    # Skip very short / numeric / common noise
    if len(name) < 2 or name.isdigit():
        return ""
    return name


def _normalise_import(raw: str) -> str:
    """
    Extract the module specifier from an import statement string.
    'import { foo } from "bar"' → 'bar'
    'const x = require("baz")' → 'baz'
    """
    import re
    # ES6: from "..." or from '...'
    m = re.search(r'from\s+["\']([^"\']+)["\']', raw)
    if m:
        return m.group(1).lower()
    # CJS: require("...")
    m = re.search(r'require\s*\(\s*["\']([^"\']+)["\']\s*\)', raw)
    if m:
        return m.group(1).lower()
    return ""
