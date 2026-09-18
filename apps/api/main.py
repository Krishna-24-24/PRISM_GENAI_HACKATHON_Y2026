from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from apps.api.exceptions import (
    TraceIQException,
    global_exception_handler,
    traceiq_exception_handler,
)
from apps.api.schemas import (
    CodeResult,
    ExampleQueriesResponse,
    ExampleQuery,
    QueryProfileResponse,
    QuerySignals,
    SearchRequest,
    SearchResponse,
    TraceStep,
)
from src.config import settings
from src.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.app_name} in {settings.environment} mode")
    # Eagerly warm-up the retriever so the first query isn't slow.
    try:
        from apps.api.retriever_singleton import get_retriever
        get_retriever()
        logger.info("Retriever warm-up complete.")
    except Exception as exc:
        logger.warning(f"Retriever warm-up failed (demo will still work): {exc}")
    yield


app = FastAPI(
    title="TRACEIQ API",
    description="Adaptive Agentic Code Retrieval & Ranking Engine",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the Vite dev server (port 5173, 8080) and local access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8080", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:8080"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_exception_handler(TraceIQException, traceiq_exception_handler)
app.add_exception_handler(Exception, global_exception_handler)


# ---------------------------------------------------------------------------
# Example queries — one per retrieval category
# ---------------------------------------------------------------------------
_EXAMPLE_QUERIES = [
    ExampleQuery(
        label="Semantic",
        category="semantic",
        query="function that validates and sanitizes user form input",
        description="Natural language description of behavior — Dense retrieval dominant",
    ),
    ExampleQuery(
        label="Identifier",
        category="symbolic",
        query="`handleHttpRequest` middleware chain",
        description="Backtick-quoted exact symbol name — Symbol channel activated",
    ),
    ExampleQuery(
        label="Usage",
        category="usage",
        query="show me examples calling createConnection",
        description="Call-site lookup — Symbol + Dense channels fused",
    ),
    ExampleQuery(
        label="Structural",
        category="structural",
        query="class that extends EventEmitter and implements retry logic",
        description="Class hierarchy query — BM25 + Dense channels fused",
    ),
    ExampleQuery(
        label="Ambiguous",
        category="mixed",
        query="async state transitions with error recovery and backoff",
        description="Complex multi-concept query — Mixed routing, Dense fallback",
    ),
]


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "environment": settings.environment}


# ---------------------------------------------------------------------------
# Example queries
# ---------------------------------------------------------------------------
@app.get("/api/example-queries", response_model=ExampleQueriesResponse)
def example_queries() -> ExampleQueriesResponse:
    return ExampleQueriesResponse(examples=_EXAMPLE_QUERIES)


# ---------------------------------------------------------------------------
# Main search endpoint
# ---------------------------------------------------------------------------
@app.post("/api/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    if not request.query.strip():
        raise TraceIQException("Query cannot be empty", status_code=400)

    try:
        from apps.api.retriever_singleton import get_retriever
        retriever = get_retriever()
    except RuntimeError as exc:
        raise TraceIQException(str(exc), status_code=503)

    # Run retrieval
    raw_results = retriever.search(request.query, top_k=request.top_k)

    if not raw_results:
        latency = 0.0
        profile = None
    else:
        latency = raw_results[0].latency_ms
        profile = raw_results[0].query_profile

    # Determine channels actually used
    channels_used: list[str] = []
    if profile and profile.recommended_weights:
        channels_used = [ch for ch, w in profile.recommended_weights.items() if w > 0]
    if not channels_used:
        channels_used = ["dense"]

    # Build query profile response
    if profile:
        q_profile = QueryProfileResponse(
            category=profile.category.value,
            routing_reason=profile.routing_reason,
            recommended_weights=profile.recommended_weights,
            signals=QuerySignals(
                has_backtick_identifier=profile.has_backtick_identifier,
                has_camel_case_token=profile.has_camel_case_token,
                has_snake_case_token=profile.has_snake_case_token,
                has_dot_accessor=profile.has_dot_accessor,
                has_structural_keyword=profile.has_structural_keyword,
                has_usage_keyword=profile.has_usage_keyword,
                has_behavioral_keyword=profile.has_behavioral_keyword,
                identifier_count=profile.identifier_count,
                word_count=profile.word_count,
                extracted_identifiers=profile.extracted_identifiers,
            ),
        )
    else:
        q_profile = QueryProfileResponse(
            category="semantic",
            routing_reason="no profile available",
            recommended_weights={"dense": 1.0},
            signals=QuerySignals(
                has_backtick_identifier=False,
                has_camel_case_token=False,
                has_snake_case_token=False,
                has_dot_accessor=False,
                has_structural_keyword=False,
                has_usage_keyword=False,
                has_behavioral_keyword=False,
                identifier_count=0,
                word_count=len(request.query.split()),
                extracted_identifiers=[],
            ),
        )

    # Map to response schema
    code_results: list[CodeResult] = []
    for rank, r in enumerate(raw_results, start=1):
        chunk = r.chunk
        if chunk is None:
            continue

        # Adaptive results have final_score; plain hybrid results use fused_score
        final_score = getattr(r, "final_score", r.fused_score)
        structural_boost = getattr(r, "structural_boost", 0.0)

        code_results.append(
            CodeResult(
                rank=rank,
                chunk_id=chunk.chunk_id,
                file_path=chunk.file_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                symbol_name=chunk.symbol_name,
                symbol_type=chunk.symbol_type,
                code=chunk.code,
                relevance_score=round(final_score, 6),
                fused_score=round(r.fused_score, 6),
                structural_boost=round(structural_boost, 6),
                channel_contributions=r.channel_contributions,
                imports=chunk.imports,
                calls=chunk.calls,
            )
        )

    # Optional trace
    trace = None
    if request.show_trace and profile:
        trace = [
            TraceStep(step="1. Query Classification", detail=f"Category={profile.category.value}"),
            TraceStep(step="2. Signal Extraction", detail=f"identifiers={profile.extracted_identifiers[:5]}"),
            TraceStep(step="3. Channel Routing", detail=f"weights={profile.recommended_weights}"),
            TraceStep(step="4. Dense Retrieval", detail="FAISS IndexFlatIP cosine search", count=retriever._config.dense_top_k),
            TraceStep(step="5. Symbol Retrieval", detail="3-tier exact/prefix/subword match", count=retriever._config.symbol_top_k if "symbol" in profile.recommended_weights else 0),
            TraceStep(step="6. wRRF Fusion", detail=f"k=60, weights={profile.recommended_weights}"),
            TraceStep(step="7. Structural Boost", detail="Call-graph + import co-citation boosts"),
            TraceStep(step="8. Final Ranking", detail=f"Top {len(code_results)} returned", count=len(code_results)),
        ]

    return SearchResponse(
        query=request.query,
        results=code_results,
        query_profile=q_profile,
        channels_used=channels_used,
        total_candidates=retriever._config.dense_top_k,
        returned_count=len(code_results),
        latency_ms=round(latency, 2),
        trace=trace,
    )


# ---------------------------------------------------------------------------
# Legacy compatibility — keep old stub endpoint
# ---------------------------------------------------------------------------
@app.post("/search")
def search_legacy(query: str) -> Dict[str, Any]:
    if not query or not query.strip():
        raise TraceIQException("Query cannot be empty", status_code=400)
    return {"query": query, "results": [], "deprecated": True, "use": "/api/search"}
