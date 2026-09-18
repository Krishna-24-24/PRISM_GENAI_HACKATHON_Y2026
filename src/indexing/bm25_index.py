import time
import sys
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
from src.chunking.models import CodeChunk
from src.chunking.tokenizer import tokenize_for_bm25

class ScoredCandidate:
    def __init__(self, chunk_id: str, score: float, channel: str = "bm25", chunk: Optional[CodeChunk] = None):
        self.chunk_id = chunk_id
        self.score = score
        self.channel = channel
        self.chunk = chunk

class BM25Channel:
    def __init__(self):
        self.chunks: Dict[str, CodeChunk] = {}
        self.chunk_list: List[CodeChunk] = []
        self._tokenized_docs: Dict[str, List[str]] = {}
        self.bm25: Optional[BM25Okapi] = None
        self.build_time: float = 0.0

    def add_chunks(self, chunks: List[CodeChunk]):
        start = time.perf_counter()
        
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
            text_to_tokenize = f"{chunk.symbol_name} {chunk.context} {chunk.code}"
            identifiers = [chunk.symbol_name] + chunk.calls + chunk.imports
            tokens = tokenize_for_bm25(text_to_tokenize, identifiers)
            self._tokenized_docs[chunk.chunk_id] = tokens
            
        self._rebuild_okapi()
        self.build_time += time.perf_counter() - start

    def remove_chunks(self, chunk_ids: List[str]) -> None:
        start = time.perf_counter()
        modified = False
        for cid in chunk_ids:
            if cid in self.chunks:
                del self.chunks[cid]
                del self._tokenized_docs[cid]
                modified = True
                
        if modified:
            self._rebuild_okapi()
        self.build_time += time.perf_counter() - start

    def _rebuild_okapi(self):
        self.chunk_list = list(self.chunks.values())
        corpus = [self._tokenized_docs[c.chunk_id] for c in self.chunk_list]
        if corpus:
            self.bm25 = BM25Okapi(corpus)
        else:
            self.bm25 = None
        
    def search(self, query: str, top_k: int = 10) -> List[ScoredCandidate]:
        if not query or not query.strip():
            return []
        if not self.bm25:
            return []
            
        query_tokens = tokenize_for_bm25(query)
        scores = self.bm25.get_scores(query_tokens)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for i in top_indices:
            if scores[i] > 0:
                results.append(ScoredCandidate(
                    chunk_id=self.chunk_list[i].chunk_id,
                    score=scores[i],
                    chunk=self.chunk_list[i]
                ))
                
        return results
        
    def get_index_size_bytes(self) -> int:
        if not self.bm25:
            return 0
        return sys.getsizeof(self.bm25) + sys.getsizeof(self.chunks) + sys.getsizeof(self.chunk_list)
