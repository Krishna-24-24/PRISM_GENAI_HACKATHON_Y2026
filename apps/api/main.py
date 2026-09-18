from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from apps.api.exceptions import (
    TraceIQException,
    global_exception_handler,
    traceiq_exception_handler,
)
from apps.api.schemas import (
    CodeResult,
    EvaluationResponse,
    EvaluationRun,
    ExampleQueriesResponse,
    ExampleQuery,
    QueryProfileResponse,
    QuerySignals,
    RepositoryListResponse,
    RepositorySummary,
    SearchRequest,
    SearchResponse,
    SystemStatus,
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


def _index_dir_size(path) -> int:
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    total = 0
    for child in p.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _ensure_chunks_loaded() -> tuple[list, Optional[str]]:
    from apps.api.retriever_singleton import get_chunks, get_retriever

    chunks = get_chunks()
    if chunks:
        return chunks, None
    try:
        get_retriever()
        return get_chunks(), None
    except Exception as exc:
        return [], str(exc)


@app.get("/api/system", response_model=SystemStatus)
def system_status() -> SystemStatus:
    from apps.api.retriever_singleton import FAISS_DIR, get_retriever

    chunks, reason = _ensure_chunks_loaded()
    index_ready = False
    dense_top_k = bm25_top_k = symbol_top_k = None
    try:
        retriever = get_retriever()
        index_ready = True
        dense_top_k = retriever._config.dense_top_k
        bm25_top_k = retriever._config.bm25_top_k
        symbol_top_k = retriever._config.symbol_top_k
        reason = None
    except Exception as exc:
        reason = str(exc)

    repo_ids = sorted({c.repo_id for c in chunks if c.repo_id})
    version_ids = sorted({c.version_id for c in chunks if c.version_id})
    files = {c.file_path for c in chunks}

    return SystemStatus(
        status="ok" if index_ready else "degraded",
        environment=settings.environment,
        index_ready=index_ready,
        index_path=FAISS_DIR,
        chunk_count=len(chunks),
        file_count=len(files),
        index_size_bytes=_index_dir_size(FAISS_DIR),
        current_repository=repo_ids[0] if repo_ids else (None if not chunks else "indexed-corpus"),
        current_version=version_ids[0] if version_ids else None,
        dense_top_k=dense_top_k,
        bm25_top_k=bm25_top_k,
        symbol_top_k=symbol_top_k,
        reason=reason,
    )


@app.get("/api/repositories", response_model=RepositoryListResponse)
def list_repositories() -> RepositoryListResponse:
    from collections import defaultdict
    from apps.api.retriever_singleton import FAISS_DIR

    chunks, _ = _ensure_chunks_loaded()
    grouped: dict[str, list] = defaultdict(list)
    for chunk in chunks:
        key = chunk.repo_id or "indexed-corpus"
        grouped[key].append(chunk)

    size = _index_dir_size(FAISS_DIR)
    repos: list[RepositorySummary] = []
    for repo_id, items in sorted(grouped.items()):
        versions = sorted({c.version_id for c in items if c.version_id})
        files = {c.file_path for c in items}
        paths = sorted(files)
        common = paths[0] if len(paths) == 1 else (paths[0].rsplit("/", 1)[0] if paths else "")
        repos.append(
            RepositorySummary(
                repo_id=repo_id,
                version=versions[0] if versions else "—",
                files=len(files),
                chunks=len(items),
                status="Ready" if items else "Empty",
                index_size_bytes=size if len(grouped) == 1 else 0,
                path=common,
            )
        )
    return RepositoryListResponse(repositories=repos)


@app.get("/api/evaluation", response_model=EvaluationResponse)
def evaluation_results() -> EvaluationResponse:
    from pathlib import Path
    import json

    results_dir = Path(__file__).resolve().parents[2] / "results"
    if not results_dir.exists():
        return EvaluationResponse(available=False, results=[])

    files = sorted(results_dir.glob("ablation_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return EvaluationResponse(available=False, results=[])

    payload = json.loads(files[0].read_text(encoding="utf-8"))
    runs: list[EvaluationRun] = []
    for row in payload.get("results", []):
        runs.append(
            EvaluationRun(
                configuration=row.get("configuration", "unknown"),
                mrr=row.get("mrr"),
                ndcg_at_10=row.get("ndcg@10"),
                p50_latency_ms=row.get("p50_latency_ms"),
                p95_latency_ms=row.get("p95_latency_ms"),
                index_build_time_s=row.get("index_build_time_s"),
                incremental_update_time_s=row.get("incremental_update_time_s"),
                index_size_mb=row.get("index_size_mb"),
                queries_evaluated=row.get("queries_evaluated"),
            )
        )
    return EvaluationResponse(
        available=True,
        source=str(files[0].name),
        timestamp=payload.get("timestamp"),
        sample_queries=payload.get("sample_queries"),
        corpus_size=payload.get("corpus_size"),
        results=runs,
    )


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
