import { useEffect, useRef, useState } from "react";
import { Search, Loader2 } from "lucide-react";
import clsx from "clsx";
import { type ExampleQuery, fetchExampleQueries, searchCode, type SearchResponse } from "./api";
import { QueryInsightPanel } from "./components/QueryInsightPanel";
import { ResultCard } from "./components/ResultCard";

const DEFAULT_TOP_K = 10;

export default function App() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(DEFAULT_TOP_K);
  const [showTrace, setShowTrace] = useState(false);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [examples, setExamples] = useState<ExampleQuery[]>([]);
  const [activeExample, setActiveExample] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Fetch example queries on mount
  useEffect(() => {
    fetchExampleQueries()
      .then(setExamples)
      .catch(() => {
        // Use hard-coded fallback if API is unreachable
        setExamples([
          { label: "Semantic", category: "semantic", query: "function that validates and sanitizes user form input", description: "" },
          { label: "Identifier", category: "symbolic", query: "`handleHttpRequest` middleware chain", description: "" },
          { label: "Usage", category: "usage", query: "show me examples calling createConnection", description: "" },
          { label: "Structural", category: "structural", query: "class that extends EventEmitter and implements retry logic", description: "" },
          { label: "Ambiguous", category: "mixed", query: "async state transitions with error recovery and backoff", description: "" },
        ]);
      });
  }, []);

  async function handleSearch(q: string = query) {
    const trimmed = q.trim();
    if (!trimmed) return;
    setQuery(trimmed);
    setLoading(true);
    setError(null);
    setResponse(null);

    try {
      const data = await searchCode(trimmed, topK, showTrace);
      setResponse(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  function handleExampleClick(ex: ExampleQuery) {
    setQuery(ex.query);
    setActiveExample(ex.label);
    handleSearch(ex.query);
    inputRef.current?.focus();
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") handleSearch();
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-brand">
          <div className="header-logo">TQ</div>
          <div>
            <span className="header-title">TRACEIQ</span>
            <span className="header-subtitle"> · Adaptive Code Retrieval Engine</span>
          </div>
        </div>
        <div className="header-meta">
          <span className="header-badge">Samsung PRISM 2026–27</span>
          <span className="header-badge">Theme 01: Agentic Code Intelligence</span>
        </div>
      </header>

      <div className="main">
        {/* Search area */}
        <div className="search-area">
          <span className="search-label">Code Retrieval Query</span>

          <div className="search-row">
            <div className="search-input-wrap">
              <span className="search-input-icon">
                <Search size={16} />
              </span>
              <input
                ref={inputRef}
                className="search-input"
                type="text"
                placeholder='e.g. "function that validates form input" or `handleHttpRequest`'
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value);
                  setActiveExample(null);
                }}
                onKeyDown={handleKeyDown}
                disabled={loading}
                autoComplete="off"
                spellCheck={false}
              />
            </div>
            <button
              className="search-btn"
              onClick={() => handleSearch()}
              disabled={loading || !query.trim()}
            >
              {loading ? <Loader2 size={15} className="spin" /> : <Search size={15} />}
              {loading ? "Retrieving…" : "Search"}
            </button>
          </div>

          {/* Options */}
          <div className="search-options">
            <label className="search-option-label">
              Top-K:
              <select
                className="topk-select"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
              >
                {[5, 10, 20, 30].map((n) => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </label>
            <label className="search-option-label">
              <input
                type="checkbox"
                checked={showTrace}
                onChange={(e) => setShowTrace(e.target.checked)}
              />
              Show retrieval trace
            </label>
          </div>

          {/* Example queries */}
          {examples.length > 0 && (
            <div className="examples-row">
              <span className="examples-label">Examples:</span>
              {examples.map((ex) => (
                <button
                  key={ex.label}
                  className={clsx("example-chip", ex.label === activeExample && "active")}
                  onClick={() => handleExampleClick(ex)}
                  title={ex.description}
                  disabled={loading}
                >
                  {ex.label}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Body: sidebar + results */}
        <div className="body">
          {/* Left: Query insight panel */}
          <QueryInsightPanel
            profile={response?.query_profile ?? null}
            latencyMs={response?.latency_ms ?? 0}
            totalCandidates={response?.total_candidates ?? 0}
            returnedCount={response?.returned_count ?? 0}
            trace={response?.trace}
          />

          {/* Right: Results */}
          <div className="results-area">
            {/* Results header */}
            {response && !loading && (
              <div className="results-header">
                <span>
                  {response.returned_count} result{response.returned_count !== 1 ? "s" : ""} · {response.total_candidates} candidates · {response.latency_ms.toFixed(0)} ms
                </span>
                <span>
                  Channels: {response.channels_used.join(" + ")}
                </span>
              </div>
            )}

            {/* Loading */}
            {loading && (
              <div className="state-container">
                <div className="spinner" />
                <span className="state-title">Retrieving…</span>
                <span className="state-desc">Running adaptive multi-channel retrieval</span>
              </div>
            )}

            {/* Error */}
            {error && !loading && (
              <div className="state-container">
                <span className="state-icon">⚠</span>
                <span className="state-title">Retrieval error</span>
                <span className="error-text">{error}</span>
                <span className="state-desc">
                  Make sure the TRACEIQ API is running at <code>http://localhost:8000</code>
                </span>
              </div>
            )}

            {/* Empty state */}
            {!loading && !error && !response && (
              <div className="state-container">
                <span className="state-icon" style={{ fontSize: "2.5rem" }}>⌨</span>
                <span className="state-title">Enter a query to retrieve code snippets</span>
                <span className="state-desc">
                  TRACEIQ performs adaptive multi-channel retrieval using dense semantic search,
                  BM25 lexical matching, and AST-based identifier lookup.
                  Select an example query above to get started.
                </span>
              </div>
            )}

            {/* No results */}
            {!loading && !error && response && response.results.length === 0 && (
              <div className="state-container">
                <span className="state-icon" style={{ fontSize: "2rem" }}>○</span>
                <span className="state-title">No results found</span>
                <span className="state-desc">Try broadening your query or using different terminology.</span>
              </div>
            )}

            {/* Results */}
            {!loading && response?.results.map((r) => (
              <ResultCard key={r.chunk_id} result={r} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
