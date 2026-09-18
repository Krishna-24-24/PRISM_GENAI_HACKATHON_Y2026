"""
TRACEIQ API — Pydantic request/response schemas for the search endpoint.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Natural language code search query")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return")
    show_trace: bool = Field(default=False, description="Include per-channel retrieval trace in response")


class ChannelContribution(BaseModel):
    channel: str
    score: float


class QuerySignals(BaseModel):
    has_backtick_identifier: bool
    has_camel_case_token: bool
    has_snake_case_token: bool
    has_dot_accessor: bool
    has_structural_keyword: bool
    has_usage_keyword: bool
    has_behavioral_keyword: bool
    identifier_count: int
    word_count: int
    extracted_identifiers: List[str]


class QueryProfileResponse(BaseModel):
    category: str
    routing_reason: str
    recommended_weights: Dict[str, float]
    signals: QuerySignals


class CodeResult(BaseModel):
    rank: int
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str
    symbol_type: str
    code: str
    relevance_score: float
    fused_score: float
    structural_boost: float
    channel_contributions: Dict[str, float]
    imports: List[str]
    calls: List[str]


class TraceStep(BaseModel):
    step: str
    detail: str
    count: Optional[int] = None


class SearchResponse(BaseModel):
    query: str
    results: List[CodeResult]
    query_profile: QueryProfileResponse
    channels_used: List[str]
    total_candidates: int
    returned_count: int
    latency_ms: float
    trace: Optional[List[TraceStep]] = None


class ExampleQuery(BaseModel):
    label: str
    category: str
    query: str
    description: str


class ExampleQueriesResponse(BaseModel):
    examples: List[ExampleQuery]


class SystemStatus(BaseModel):
    status: str
    environment: str
    index_ready: bool
    index_path: str
    chunk_count: int
    file_count: int
    index_size_bytes: int
    current_repository: Optional[str] = None
    current_version: Optional[str] = None
    dense_top_k: Optional[int] = None
    bm25_top_k: Optional[int] = None
    symbol_top_k: Optional[int] = None
    reason: Optional[str] = None


class RepositorySummary(BaseModel):
    repo_id: str
    version: str
    files: int
    chunks: int
    status: str
    index_size_bytes: int
    path: str


class RepositoryListResponse(BaseModel):
    repositories: List[RepositorySummary]


class EvaluationRun(BaseModel):
    configuration: str
    mrr: Optional[float] = None
    ndcg_at_10: Optional[float] = None
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    index_build_time_s: Optional[float] = None
    incremental_update_time_s: Optional[float] = None
    index_size_mb: Optional[float] = None
    queries_evaluated: Optional[int] = None


class EvaluationResponse(BaseModel):
    available: bool
    source: Optional[str] = None
    timestamp: Optional[str] = None
    sample_queries: Optional[int] = None
    corpus_size: Optional[int] = None
    results: List[EvaluationRun] = []
