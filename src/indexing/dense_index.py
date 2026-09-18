"""
DenseChannel — FAISS-backed dense retrieval over CodeChunk embeddings.

Uses faiss.IndexIDMap(faiss.IndexFlatIP) to support incremental updates
(adding and removing specific chunks) without a full rebuild.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from src.chunking.models import CodeChunk
from src.indexing.embedder import EmbeddingProvider


class DenseCandidate:
    __slots__ = ("chunk_id", "score", "chunk")

    def __init__(self, chunk_id: str, score: float, chunk: Optional[CodeChunk] = None) -> None:
        self.chunk_id = chunk_id
        self.score = score
        self.chunk = chunk


class DenseChannel:
    def __init__(self, provider: EmbeddingProvider, batch_size: int = 32) -> None:
        self._provider = provider
        self._batch_size = batch_size
        
        self._index: Optional[faiss.IndexIDMap] = None
        self._chunk_map: Dict[str, CodeChunk] = {}
        
        # Mapping string chunk_ids to FAISS 64-bit int IDs
        self._str_to_int: Dict[str, int] = {}
        self._int_to_str: Dict[int, str] = {}
        self._next_faiss_id: int = 1
        
        self.build_time: float = 0.0
        self.embedding_throughput: float = 0.0

    def _ensure_index(self):
        if self._index is None:
            base_index = faiss.IndexFlatIP(self._provider.dim)
            self._index = faiss.IndexIDMap(base_index)

    def build(
        self,
        chunks: List[CodeChunk],
        cache_dir: Optional[str] = None,
        show_progress: bool = True,
    ) -> None:
        """Build index from scratch."""
        self._ensure_index()
        self._index.reset()
        self._chunk_map.clear()
        self._str_to_int.clear()
        self._int_to_str.clear()
        self._next_faiss_id = 1
        self.add_chunks(chunks, cache_dir, show_progress)

    def add_chunks(
        self,
        chunks: List[CodeChunk],
        cache_dir: Optional[str] = None,
        show_progress: bool = True,
    ) -> None:
        """Incrementally add new chunks to the index."""
        if not chunks:
            return
            
        t0 = time.time()
        self._ensure_index()
        
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

        texts: List[str] = []
        faiss_ids: List[int] = []
        cached_vecs: Dict[int, np.ndarray] = {}

        for i, chunk in enumerate(chunks):
            self._chunk_map[chunk.chunk_id] = chunk
            fid = self._next_faiss_id
            self._next_faiss_id += 1
            
            self._str_to_int[chunk.chunk_id] = fid
            self._int_to_str[fid] = chunk.chunk_id
            faiss_ids.append(fid)

            # Try to load from cache
            if cache_dir:
                cache_path = os.path.join(cache_dir, f"{chunk.chunk_hash}_{self._provider.dim}.npy")
                if os.path.exists(cache_path):
                    cached_vecs[i] = np.load(cache_path).astype(np.float32)
                    texts.append(None)
                    continue
            texts.append(chunk.code)

        need_embed_indices = [i for i, t in enumerate(texts) if t is not None]
        need_embed_texts = [texts[i] for i in need_embed_indices]

        all_vecs = np.zeros((len(chunks), self._provider.dim), dtype=np.float32)

        for pos, vec in cached_vecs.items():
            all_vecs[pos] = vec

        if need_embed_texts:
            computed = self._provider.embed_texts(
                need_embed_texts,
                batch_size=self._batch_size,
                show_progress=show_progress,
            )
            for local_i, global_i in enumerate(need_embed_indices):
                all_vecs[global_i] = computed[local_i]
                if cache_dir:
                    chunk = chunks[global_i]
                    cache_path = os.path.join(cache_dir, f"{chunk.chunk_hash}_{self._provider.dim}.npy")
                    tmp_path = cache_path.replace(".npy", ".tmp.npy")
                    np.save(tmp_path, computed[local_i])
                    os.replace(tmp_path, cache_path)

        elapsed = time.time() - t0
        self.build_time += elapsed
        if len(need_embed_texts) > 0 and elapsed > 0:
            self.embedding_throughput = len(need_embed_texts) / elapsed

        # Add to FAISS IndexIDMap
        self._index.add_with_ids(all_vecs, np.array(faiss_ids, dtype=np.int64))

    def remove_chunks(self, chunk_ids: List[str]) -> None:
        """Incrementally remove chunks from the index."""
        if not chunk_ids or self._index is None:
            return
            
        ids_to_remove = []
        for cid in chunk_ids:
            if cid in self._str_to_int:
                fid = self._str_to_int[cid]
                ids_to_remove.append(fid)
                del self._int_to_str[fid]
                del self._str_to_int[cid]
            if cid in self._chunk_map:
                del self._chunk_map[cid]
                
        if ids_to_remove:
            self._index.remove_ids(np.array(ids_to_remove, dtype=np.int64))

    def save(self, directory: str) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, os.path.join(directory, "index.faiss"))
        mappings = {
            "str_to_int": self._str_to_int,
            "int_to_str": {str(k): v for k, v in self._int_to_str.items()},
            "next_faiss_id": self._next_faiss_id,
        }
        with open(os.path.join(directory, "mappings.json"), "w", encoding="utf-8") as f:
            json.dump(mappings, f)
        chunk_data = {cid: chunk.model_dump() for cid, chunk in self._chunk_map.items()}
        with open(os.path.join(directory, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(chunk_data, f)

    def load(self, directory: str) -> None:
        self._index = faiss.read_index(os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "mappings.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        self._str_to_int = data["str_to_int"]
        self._int_to_str = {int(k): v for k, v in data["int_to_str"].items()}
        self._next_faiss_id = data["next_faiss_id"]
        with open(os.path.join(directory, "chunks.json"), "r", encoding="utf-8") as f:
            chunk_data = json.load(f)
        from src.chunking.models import CodeChunk
        self._chunk_map = {cid: CodeChunk(**d) for cid, d in chunk_data.items()}

    def search(self, query: str, top_k: int = 10) -> List[DenseCandidate]:
        if not query or not query.strip():
            return []
        if self._index is None or self._index.ntotal == 0:
            return []

        q_vec = self._provider.embed_query(query).reshape(1, -1)
        scores, indices = self._index.search(q_vec, top_k)

        results: List[DenseCandidate] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk_id = self._int_to_str.get(int(idx))
            if chunk_id:
                results.append(DenseCandidate(
                    chunk_id=chunk_id,
                    score=float(score),
                    chunk=self._chunk_map.get(chunk_id),
                ))
        return results

    def get_index_size_bytes(self) -> int:
        if self._index is None:
            return 0
        vec_bytes = self._index.ntotal * self._provider.dim * 4
        meta_bytes = sys.getsizeof(self._str_to_int) + sys.getsizeof(self._chunk_map)
        return vec_bytes + meta_bytes

    @property
    def num_vectors(self) -> int:
        return self._index.ntotal if self._index else 0
