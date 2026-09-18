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
