"""
Unit tests for RetrievalAgent.

Verifies:
1. Early termination on high confidence.
2. Early termination on empty hits.
3. 2nd-pass expansion triggered by AST calls.
4. 2nd-pass expansion triggered by PRF tokens.
5. Fused results retain the best scores with proper decay.
"""
from __future__ import annotations

import os
import sys
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.chunking.models import CodeChunk
from src.retrieval.adaptive_retriever import AdaptiveRetrievalResult, AdaptiveRetriever
from src.agent.retrieval_agent import RetrievalAgent, AgentTrace


class MockRetriever:
    """Mocks AdaptiveRetriever to return specific hits per query."""
    def __init__(self, responses: dict):
        self.responses = responses
        self.call_history = []

    def search(self, query: str, top_k: int = 10) -> List[AdaptiveRetrievalResult]:
        self.call_history.append(query)
        hits = self.responses.get(query, [])
        return hits[:top_k]


def test_early_termination_high_confidence():
    # Pass 1 returns a hit with final_score = 0.9 (above 0.85 threshold)
    chunk = CodeChunk(chunk_id="c1", repo_id="", version_id="", file_path="", 
                      symbol_name="", symbol_type="", start_line=1, end_line=2, 
                      code="function foo() {}", normalized_code="", context="", 
                      imports=[], exports=[], calls=[], called_by=[], chunk_hash="")
    hit = AdaptiveRetrievalResult(chunk_id="c1", fused_score=0.9, channel_contributions={}, chunk=chunk)
    hit.final_score = 0.9

    mock = MockRetriever({"test_query": [hit]})
    agent = RetrievalAgent(retriever=mock, high_confidence_threshold=0.85)

    final_hits, trace = agent.search("test_query")
    
    assert trace.early_terminated is True
    assert "High confidence" in trace.termination_reason
    assert len(mock.call_history) == 1
    assert final_hits[0].chunk_id == "c1"


def test_ast_calls_expansion():
    # Pass 1 returns a hit with score=0.5 (below threshold) and AST calls
    chunk = CodeChunk(chunk_id="c1", repo_id="", version_id="", file_path="", 
                      symbol_name="", symbol_type="", start_line=1, end_line=2, 
                      code="function foo() { logger(); validate(); }", normalized_code="", context="", 
                      imports=[], exports=[], calls=["logger", "validate"], called_by=[], chunk_hash="")
    hit1 = AdaptiveRetrievalResult(chunk_id="c1", fused_score=0.5, channel_contributions={}, chunk=chunk)
    hit1.final_score = 0.5
    
    # Pass 2 mock response for the expansion query "logger validate" or "validate logger"
    chunk2 = CodeChunk(chunk_id="c2", repo_id="", version_id="", file_path="", 
                      symbol_name="logger", symbol_type="", start_line=1, end_line=2, 
                      code="function logger() {}", normalized_code="", context="", 
                      imports=[], exports=[], calls=[], called_by=[], chunk_hash="")
    hit2 = AdaptiveRetrievalResult(chunk_id="c2", fused_score=0.7, channel_contributions={}, chunk=chunk2)
    hit2.final_score = 0.7

    # Since sets are unordered, we provide both permutations for the mock
    mock = MockRetriever({
        "test_query": [hit1],
        "logger validate": [hit2],
        "validate logger": [hit2]
    })
    
    agent = RetrievalAgent(retriever=mock, high_confidence_threshold=0.85)
    final_hits, trace = agent.search("test_query")
    
    assert trace.early_terminated is False
    assert len(mock.call_history) == 2
    assert "AST_CALLS_EXPANSION" in trace.steps[1].action
    assert len(final_hits) == 2
    
    # Check that hit2's score was decayed (0.7 * 0.8 = 0.56)
    assert final_hits[0].chunk_id == "c2" # 0.56
    assert final_hits[1].chunk_id == "c1" # 0.50


def test_prf_lexical_expansion():
    # Pass 1 returns a hit with NO AST calls, but with camelCase/snake_case tokens in code
    chunk = CodeChunk(chunk_id="c1", repo_id="", version_id="", file_path="", 
                      symbol_name="", symbol_type="", start_line=1, end_line=2, 
                      code="let myVar = get_user_id();", normalized_code="", context="", 
                      imports=[], exports=[], calls=[], called_by=[], chunk_hash="")
    hit1 = AdaptiveRetrievalResult(chunk_id="c1", fused_score=0.5, channel_contributions={}, chunk=chunk)
    hit1.final_score = 0.5
    
    # Pass 2 mock response
    hit2 = AdaptiveRetrievalResult(chunk_id="c2", fused_score=0.6, channel_contributions={}, chunk=chunk)
    hit2.final_score = 0.6

    mock = MockRetriever({
        "test_query": [hit1],
        "myVar get_user_id": [hit2],
        "get_user_id myVar": [hit2]
    })
    
    agent = RetrievalAgent(retriever=mock, high_confidence_threshold=0.85)
    final_hits, trace = agent.search("test_query")
    
    assert trace.early_terminated is False
    assert len(mock.call_history) == 2
    assert "LEXICAL_PRF_EXPANSION" in trace.steps[1].action


def test_no_results_early_termination():
    mock = MockRetriever({"test_query": []})
    agent = RetrievalAgent(retriever=mock)
    
    final_hits, trace = agent.search("test_query")
    assert trace.early_terminated is True
    assert len(final_hits) == 0
    assert len(mock.call_history) == 1
