"""
Identifier/Symbol channel — exact and fuzzy name matching.

For the AppsRetrieval benchmark this channel contributes near-zero
(queries are NL problem descriptions, not identifier lookups).
It is built here so the JS-demo path has it, and so E-CLASSIFIER
(hostile review) can be confirmed empirically.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.chunking.models import CodeChunk


class SymbolCandidate:
    __slots__ = ("chunk_id", "score", "chunk")

    def __init__(self, chunk_id: str, score: float, chunk: Optional[CodeChunk] = None):
        self.chunk_id = chunk_id
        self.score = score
        self.chunk = chunk


class SymbolChannel:
    """
    Index of symbol names → chunk_ids with three match tiers:

        1.0  exact case-sensitive match     (symbol_name == query_token)
        0.8  case-insensitive match
        0.5  camelCase/snake_case sub-word match
              e.g. query "http request" matches "handleHttpRequest"
    """

    def __init__(self) -> None:
        # lowercase_name → list of (original_name, chunk_id, chunk)
        self._exact: Dict[str, List[tuple]] = {}
        # split tokens → list of (chunk_id, chunk)
        self._token_map: Dict[str, List[tuple]] = {}
        self._chunks: Dict[str, CodeChunk] = {}

    def add_chunks(self, chunks: List[CodeChunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
            name = chunk.symbol_name or ""
            if not name or name == "anonymous":
                continue

            # Exact / case-insensitive
            key = name.lower()
            self._exact.setdefault(key, []).append((name, chunk.chunk_id, chunk))

            # Sub-word tokens
            for token in _split_identifier(name):
                self._token_map.setdefault(token, []).append((chunk.chunk_id, chunk))

    def remove_chunks(self, chunk_ids: List[str]) -> None:
        id_set = set(chunk_ids)
        for cid in chunk_ids:
            if cid in self._chunks:
                del self._chunks[cid]

        # Clean exact map
        for key in list(self._exact.keys()):
            new_list = [item for item in self._exact[key] if item[1] not in id_set]
            if new_list:
                self._exact[key] = new_list
            else:
                del self._exact[key]

        # Clean token map
        for key in list(self._token_map.keys()):
            new_list = [item for item in self._token_map[key] if item[0] not in id_set]
            if new_list:
                self._token_map[key] = new_list
            else:
                del self._token_map[key]

    def search(self, query: str, top_k: int = 50) -> List[SymbolCandidate]:
        if not query or not query.strip():
            return []

        scores: Dict[str, float] = {}

        query_tokens = query.strip().split()
        for qt in query_tokens:
            qt_lower = qt.lower()
            token_scores: Dict[str, float] = {}

            # Tier 1: exact match
            for orig, cid, _ in self._exact.get(qt_lower, []):
                boost = 1.0 if orig == qt else 0.8
                token_scores[cid] = max(token_scores.get(cid, 0.0), boost)

            # Tier 2: sub-word match (0.6 so multiple sub-word matches beat a single exact match)
            for cid, _ in self._token_map.get(qt_lower, []):
                token_scores[cid] = max(token_scores.get(cid, 0.0), 0.6)

            for cid, s in token_scores.items():
                scores[cid] = scores.get(cid, 0.0) + s

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            SymbolCandidate(cid, sc, self._chunks.get(cid))
            for cid, sc in ranked
        ]

    def __len__(self) -> int:
        return len(self._chunks)


def _split_identifier(name: str) -> List[str]:
    """Split camelCase and snake_case into lowercase tokens."""
    # Insert space before uppercase runs
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", name)
    spaced = re.sub(r"([a-z\d])([A-Z])", r"\1 \2", spaced)
    # Split on underscores and spaces
    return [t.lower() for t in re.split(r"[_\s]+", spaced) if t]
