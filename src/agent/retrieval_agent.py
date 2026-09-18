"""
TRACEIQ Agentic Retrieval Loop.

This component wraps the AdaptiveRetriever to form an autonomous multi-pass loop.
It enforces strict latency budgets and candidate limits to remain CPU-friendly.
No unrestricted LLM calls are used; the agent identifies missing signals via
deterministic AST analysis and Pseudo-Relevance Feedback (PRF).

Target Loop:
1. INITIAL SEARCH (Adaptive hybrid)
2. INSPECT (Check confidence and latency budgets)
3. IDENTIFY MISSING SIGNAL (Extract AST calls/imports or PRF lexical tokens)
4. REFINE (Formulate a targeted expansion query)
5. SECOND SEARCH (Execute expansion query)
6. FUSE (Combine Pass 1 and Pass 2 candidates with decay)
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from src.retrieval.adaptive_retriever import AdaptiveRetriever, AdaptiveRetrievalResult


@dataclass
class AgentStep:
    iteration: int
    action: str
    query_used: str
    candidates_retrieved: int
    latency_ms: float


@dataclass
class AgentTrace:
    original_query: str
    steps: List[AgentStep] = field(default_factory=list)
    final_candidate_count: int = 0
    total_latency_ms: float = 0.0
    early_terminated: bool = False
    termination_reason: str = ""

    def summary(self) -> str:
        s = f"AgentTrace(query='{self.original_query}', time={self.total_latency_ms:.1f}ms)\n"
        for step in self.steps:
            s += f"  [{step.iteration}] {step.action}: '{step.query_used}' -> {step.candidates_retrieved} hits ({step.latency_ms:.1f}ms)\n"
        s += f"  Terminated: {self.early_terminated} ({self.termination_reason})"
        return s


class RetrievalAgent:
    """
    Autonomous retrieval orchestrator enforcing hackathon constraints:
    - Maximum iterations (default 2)
    - Latency budget (default 800ms)
    - No execution, no hallucination, bounded loops
    """

    def __init__(
        self,
        retriever: AdaptiveRetriever,
        max_iterations: int = 2,
        max_latency_ms: float = 800.0,
        high_confidence_threshold: float = 0.85,
    ):
        self.retriever = retriever
        self.max_iterations = max_iterations
        self.max_latency_ms = max_latency_ms
        self.high_confidence_threshold = high_confidence_threshold

        # Regex for Pseudo-Relevance Feedback (PRF) extraction
        # Matches camelCase and snake_case tokens
        self._prf_regex = re.compile(r"\b[a-z]+[A-Z][a-zA-Z0-9]*\b|\b[a-z]+_[a-z0-9_]+\b")

    def search(
        self, query: str, top_k: int = 10
    ) -> tuple[List[AdaptiveRetrievalResult], AgentTrace]:
        """
        Execute the agentic retrieval loop.
        Returns the fused final ranking and the trace of agent actions.
        """
        t_start = time.perf_counter()
        trace = AgentTrace(original_query=query)

        # ---------------------------------------------------------
        # PASS 1: Initial Search
        # ---------------------------------------------------------
        pass1_hits = self.retriever.search(query, top_k=20)
        step1_lat = (time.perf_counter() - t_start) * 1000
        trace.steps.append(
            AgentStep(1, "Initial Adaptive Search", query, len(pass1_hits), step1_lat)
        )

        if not pass1_hits:
            trace.early_terminated = True
            trace.termination_reason = "No candidates found in Pass 1"
            self._finalize(pass1_hits, top_k, trace, t_start)
            return pass1_hits[:top_k], trace

        # ---------------------------------------------------------
        # INSPECTION & EARLY TERMINATION CHECKS
        # ---------------------------------------------------------
        # 1. Budget check
        elapsed = (time.perf_counter() - t_start) * 1000
        if elapsed > (self.max_latency_ms * 0.5):
            trace.early_terminated = True
            trace.termination_reason = "Latency budget nearing exhaustion"
            self._finalize(pass1_hits, top_k, trace, t_start)
            return pass1_hits[:top_k], trace

        # 2. Confidence check
        best_score = pass1_hits[0].final_score
        if best_score > self.high_confidence_threshold:
            trace.early_terminated = True
            trace.termination_reason = f"High confidence match found ({best_score:.2f} > {self.high_confidence_threshold})"
            self._finalize(pass1_hits, top_k, trace, t_start)
            return pass1_hits[:top_k], trace

        if self.max_iterations < 2:
            trace.early_terminated = True
            trace.termination_reason = "Max iterations reached (1)"
            self._finalize(pass1_hits, top_k, trace, t_start)
            return pass1_hits[:top_k], trace

        # ---------------------------------------------------------
        # IDENTIFY MISSING SIGNAL & REFINE
        # ---------------------------------------------------------
        expansion_query = ""
        action_name = ""

        # Strategy A: AST Pivot (Look for definitions of called functions)
        ast_calls: Set[str] = set()
        for h in pass1_hits[:3]:
            if h.chunk.calls:
                ast_calls.update(h.chunk.calls)
        
        if ast_calls:
            # Join calls to force the QueryAnalyzer into SYMBOLIC mode on Pass 2
            expansion_query = " ".join(list(ast_calls)[:5])
            action_name = "AST_CALLS_EXPANSION"
        else:
            # Strategy B: Lexical PRF Pivot (Extract identifiers from dense hits)
            # Useful for bridging pure semantic queries to exact lexical chunks
            top_code = pass1_hits[0].chunk.code
            tokens = self._prf_regex.findall(top_code)
            unique_tokens = list(dict.fromkeys(tokens)) # deduplicate preserving order
            
            if unique_tokens:
                expansion_query = " ".join(unique_tokens[:5])
                action_name = "LEXICAL_PRF_EXPANSION"

        if not expansion_query:
            trace.early_terminated = True
            trace.termination_reason = "No missing signal identified for expansion"
            self._finalize(pass1_hits, top_k, trace, t_start)
            return pass1_hits[:top_k], trace

        # ---------------------------------------------------------
        # PASS 2: Second Search
        # ---------------------------------------------------------
        t2_start = time.perf_counter()
        pass2_hits = self.retriever.search(expansion_query, top_k=20)
        step2_lat = (time.perf_counter() - t2_start) * 1000
        trace.steps.append(
            AgentStep(2, action_name, expansion_query, len(pass2_hits), step2_lat)
        )

        # ---------------------------------------------------------
        # FUSE RESULTS (Pass 1 + Pass 2)
        # ---------------------------------------------------------
        # Pass 2 results are exploratory/expanded, so we apply a decay penalty
        # to ensure they don't overriding strong semantic matches from Pass 1.
        DECAY_FACTOR = 0.8

        fused_dict: Dict[str, AdaptiveRetrievalResult] = {h.chunk_id: h for h in pass1_hits}
        for h in pass2_hits:
            if h.chunk_id in fused_dict:
                # If found in both, take the max (or add them)
                fused_dict[h.chunk_id].final_score = max(
                    fused_dict[h.chunk_id].final_score, h.final_score * DECAY_FACTOR
                )
            else:
                h.final_score = h.final_score * DECAY_FACTOR
                fused_dict[h.chunk_id] = h

        final_candidates = list(fused_dict.values())
        final_candidates.sort(key=lambda x: x.final_score, reverse=True)

        trace.early_terminated = False
        trace.termination_reason = "Completed all agentic iterations"
        self._finalize(final_candidates, top_k, trace, t_start)

        return final_candidates[:top_k], trace

    def _finalize(self, hits, top_k, trace, t_start):
        trace.final_candidate_count = min(len(hits), top_k)
        trace.total_latency_ms = (time.perf_counter() - t_start) * 1000
