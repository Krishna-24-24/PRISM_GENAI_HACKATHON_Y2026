"""
VersionManager - Handles incremental updates across code versions.

Computes a diff between the currently indexed version and a new version using
`chunk_hash` to efficiently detect unmodified, modified, added, and deleted chunks.
Pushes incremental updates (add/remove) to the underlying index channels without
requiring a full rebuild.
"""
from __future__ import annotations

import time
from typing import Dict, List, Set, Tuple

from src.chunking.models import CodeChunk
from src.retrieval.adaptive_retriever import AdaptiveRetriever


class VersionManager:
    def __init__(self, retriever: AdaptiveRetriever):
        """
        Wraps an AdaptiveRetriever to manage its internal state incrementally.
        """
        self.retriever = retriever
        # Tracks current active chunks: chunk_id -> CodeChunk
        self.active_chunks: Dict[str, CodeChunk] = {}

    def set_version(self, new_chunks: List[CodeChunk]) -> dict:
        """
        Diffs the new_chunks against active_chunks and incrementally updates
        the Dense, BM25, Symbol, and Structural indices.
        
        Returns a dict of metrics.
        """
        t0 = time.perf_counter()
        
        new_dict = {c.chunk_id: c for c in new_chunks}
        
        added_ids = set(new_dict.keys()) - set(self.active_chunks.keys())
        added_chunks = [new_dict[cid] for cid in added_ids]
        
        deleted_ids = set(self.active_chunks.keys()) - set(new_dict.keys())
        pure_deletions = len(deleted_ids)
        
        modified_chunks = []
        for cid in set(new_dict.keys()) & set(self.active_chunks.keys()):
            if new_dict[cid].chunk_hash != self.active_chunks[cid].chunk_hash:
                modified_chunks.append(new_dict[cid])
                deleted_ids.add(cid)
                
        chunks_to_add = added_chunks + modified_chunks
        
        # 1. Update Dense Index
        dense_chan = self.retriever._retriever._dense
        if dense_chan:
            if deleted_ids:
                dense_chan.remove_chunks(list(deleted_ids))
            if chunks_to_add:
                dense_chan.add_chunks(
                    chunks_to_add, 
                    cache_dir=self.retriever._retriever.config.embed_cache_dir,
                    show_progress=False
                )
                
        # 2. Update BM25 Index
        bm25_chan = self.retriever._retriever._bm25
        if bm25_chan:
            if deleted_ids:
                bm25_chan.remove_chunks(list(deleted_ids))
            if chunks_to_add:
                bm25_chan.add_chunks(chunks_to_add)
                
        # 3. Update Symbol Index
        symbol_chan = self.retriever._retriever._symbol
        if symbol_chan:
            if deleted_ids:
                symbol_chan.remove_chunks(list(deleted_ids))
            if chunks_to_add:
                symbol_chan.add_chunks(chunks_to_add)
                
        # Update our active chunks state
        for cid in deleted_ids:
            if cid in self.active_chunks:
                del self.active_chunks[cid]
        for c in chunks_to_add:
            self.active_chunks[c.chunk_id] = c
            
        # 4. Rebuild Structural Index
        # Structural graph rebuilds in O(N) very quickly (<10ms for 10K chunks), 
        # so we just recompute the active subgraph for stability rather than tracking edges.
        if self.retriever._structural_weights.enabled:
            self.retriever._structural.build(list(self.active_chunks.values()))
            self.retriever._structural_built = True

        return {
            "update_time_s": time.perf_counter() - t0,
            "added": len(added_chunks),
            "modified": len(modified_chunks),
            "deleted": pure_deletions,
            "total_active": len(self.active_chunks),
            "chunks_embedded_new": len(chunks_to_add)  # Some may hit .npy cache
        }
