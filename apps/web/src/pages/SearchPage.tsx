import { useCallback, useEffect, useState } from "react";
import clsx from "clsx";
import {
  fetchExampleQueries,
  searchCode,
  type ExampleQuery,
  type SearchResponse,
} from "../api";
import { useWorkspace } from "../context/WorkspaceContext";
import { SearchInput } from "../components/search/SearchInput";
import { SearchResult } from "../components/search/SearchResult";
import { QueryMeta } from "../components/search/QueryMeta";
import { RetrievalTrace } from "../components/search/RetrievalTrace";
import { EmptyState } from "../components/ui/EmptyState";
import { ErrorState } from "../components/ui/ErrorState";
import { Kbd } from "../components/ui/Kbd";

export const QUERY_TYPES = ["semantic", "symbolic", "structural", "usage", "mixed"] as const;

const FALLBACK_EXAMPLES: ExampleQuery[] = [
  { label: "Semantic", category: "semantic", query: "function that validates and sanitizes user form input", description: "" },
  { label: "Identifier", category: "symbolic", query: "`handleHttpRequest` middleware chain", description: "" },
  { label: "Usage", category: "usage", query: "show me examples calling createConnection", description: "" },
  { label: "Structural", category: "structural", query: "class that extends EventEmitter and implements retry logic", description: "" },
  { label: "Ambiguous", category: "mixed", query: "async state transitions with error recovery and backoff", description: "" },
];

interface Props {
  pendingQuery?: string | null;
  onPendingConsumed?: () => void;
}

export function SearchPage({ pendingQuery, onPendingConsumed }: Props) {
  const { settings, system } = useWorkspace();
  const [query, setQuery] = useState("");
  const [queryType, setQueryType] = useState<string>("mixed");
  const [topK, setTopK] = useState(settings.defaultTopK);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [examples, setExamples] = useState<ExampleQuery[]>([]);
  const [activeExample, setActiveExample] = useState<string | null>(null);
  const [selected, setSelected] = useState(0);

  useEffect(() => {
    setTopK(settings.defaultTopK);
  }, [settings.defaultTopK]);

  useEffect(() => {
    fetchExampleQueries()
      .then(setExamples)
      .catch(() => setExamples(FALLBACK_EXAMPLES));
  }, []);

  const runSearch = useCallback(async (raw?: string) => {
    const trimmed = (raw ?? query).trim();
    if (!trimmed) return;
    setQuery(trimmed);
    setLoading(true);
    setError(null);
    setResponse(null);
    setSelected(0);
    try {
      const data = await searchCode(trimmed, topK, settings.alwaysShowTrace);
      setResponse(data);
      if (data.query_profile?.category) setQueryType(data.query_profile.category);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [query, settings.alwaysShowTrace, topK]);

  useEffect(() => {
    if (!pendingQuery) return;
    setQuery(pendingQuery);
    onPendingConsumed?.();
    void runSearch(pendingQuery);
  }, [pendingQuery, onPendingConsumed, runSearch]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement | null)?.tagName;
      const typing = tag === "INPUT" || tag === "TEXTAREA";
      if (typing) {
        if (e.key === "Escape") (e.target as HTMLElement).blur();
        return;
      }
      if (!response?.results.length) return;
      if (e.key === "ArrowDown" || e.key === "j") {
        e.preventDefault();
        setSelected((n) => Math.min(response.results.length - 1, n + 1));
      }
      if (e.key === "ArrowUp" || e.key === "k") {
        e.preventDefault();
        setSelected((n) => Math.max(0, n - 1));
      }
      if (e.key === "Enter") {
        document.getElementById(`result-${response.results[selected]?.rank}`)?.scrollIntoView({ block: "nearest" });
      }
      if (e.key === "c") {
        const code = response.results[selected]?.code;
        if (code) void navigator.clipboard.writeText(code);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [response, selected]);

  const filteredExamples = examples.filter((ex) => queryType === "mixed" || ex.category === queryType);

  return (
    <>
      <div className="page-head">
        <div>
          <div className="page-title">Search</div>
          <div className="page-desc">Retrieve ranked snippets from the indexed JavaScript corpus.</div>
        </div>
        <span style={{ color: "var(--text-muted)", fontSize: 12 }}>
          Focus <Kbd>/</Kbd> · results <Kbd>↑</Kbd><Kbd>↓</Kbd> · copy <Kbd>c</Kbd>
        </span>
      </div>

      <div className="search-workspace">
        <div>
          <div className="search-composer">
            <div className="field-label">Query</div>
            <SearchInput
              value={query}
              onChange={(v) => {
                setQuery(v);
                setActiveExample(null);
              }}
              onSubmit={() => void runSearch()}
              loading={loading}
            />
            <div className="search-toolbar">
              <div>
                <div className="field-label" style={{ marginBottom: 4 }}>Query type</div>
                <div className="seg">
                  {QUERY_TYPES.map((t) => (
                    <button
                      key={t}
                      type="button"
                      className={clsx(queryType === t && "active")}
                      onClick={() => setQueryType(t)}
                    >
                      {t}
                    </button>
                  ))}
                </div>
              </div>
              <label className="field">
                <span className="field-label">Top K</span>
                <select className="select" value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
                  {[5, 10, 20, 30].map((n) => (
                    <option key={n} value={n}>{n}</option>
                  ))}
                </select>
              </label>
            </div>
            {filteredExamples.length > 0 && (
              <div className="hint-row">
                <span className="label">Examples</span>
                {filteredExamples.map((ex) => (
                  <button
                    key={ex.label}
                    type="button"
                    className={clsx("example", activeExample === ex.label && "active")}
                    title={ex.description}
                    onClick={() => {
                      setActiveExample(ex.label);
                      setQueryType(ex.category);
                      setQuery(ex.query);
                      void runSearch(ex.query);
                    }}
                  >
                    {ex.query}
                  </button>
                ))}
              </div>
            )}
          </div>

          {response && !loading && (
            <div className="summary-strip">
              <div className="summary-cell">
                <div className="summary-k">Candidates searched</div>
                <div className="summary-v">{response.total_candidates}</div>
              </div>
              <div className="summary-cell">
                <div className="summary-k">Selected</div>
                <div className="summary-v">{response.returned_count}</div>
              </div>
              <div className="summary-cell">
                <div className="summary-k">Latency</div>
                <div className="summary-v">{response.latency_ms.toFixed(0)} ms</div>
              </div>
              <div className="summary-cell">
                <div className="summary-k">Strategy</div>
                <div className="summary-v">{response.channels_used.join(" + ") || "—"}</div>
              </div>
            </div>
          )}

          {loading && (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {[0, 1, 2].map((i) => (
                <div key={i} className="skeleton" style={{ height: 72 }} />
              ))}
            </div>
          )}

          {error && !loading && (
            <ErrorState
              title="RETRIEVAL FAILED"
              reason={error}
              context={[
                { label: "Endpoint", value: "/api/search" },
                { label: "Query", value: query || "—" },
              ]}
              action={
                <button className="btn" onClick={() => void runSearch()}>Retry</button>
              }
            />
          )}

          {!loading && !error && !response && (
            <EmptyState
              title={system && !system.index_ready ? "No repository indexed" : "Awaiting query"}
              description={
                system && !system.index_ready
                  ? system.reason ?? "The FAISS index is not loaded. Index a corpus before searching."
                  : "Type a natural-language, identifier, usage, or structural query. TRACEIQ routes retrieval across dense, BM25, and symbol channels."
              }
              hints={[
                "broader language for semantic intent",
                "a symbol name in backticks",
                "usage: “where is X called”",
                "structural relationship: extends / implements",
              ]}
            />
          )}

          {!loading && response && response.results.length === 0 && (
            <EmptyState
              title="No results"
              description="Nothing ranked above the cutoff for this query."
              hints={[
                "broader language",
                "symbol name",
                "usage description",
                "structural relationship",
              ]}
            />
          )}

          {!loading && response?.results.map((r, i) => (
            <SearchResult
              key={r.chunk_id}
              result={r}
              selected={i === selected}
              compact={settings.compactResults}
              onSelect={() => setSelected(i)}
            />
          ))}
        </div>

        <aside>
          {response ? (
            <>
              <QueryMeta
                query={response.query}
                profile={response.query_profile}
                channels={response.channels_used}
              />
              {response.query_profile.signals.extracted_identifiers.length > 0 && (
                <div className="diag" style={{ marginBottom: 12 }}>
                  <div className="diag-h">Identifiers</div>
                  <div className="diag-b">
                    <div className="kv-v">
                      {response.query_profile.signals.extracted_identifiers.join(", ")}
                    </div>
                  </div>
                </div>
              )}
              {response.trace && response.trace.length > 0 && (
                <RetrievalTrace steps={response.trace} />
              )}
            </>
          ) : (
            <div className="diag">
              <div className="diag-h">Retrieval trace</div>
              <div className="diag-b">
                <p style={{ color: "var(--text-muted)", fontSize: 12 }}>
                  Run a search to inspect classification, channel routing, fusion, and ranking.
                </p>
              </div>
            </div>
          )}
        </aside>
      </div>
    </>
  );
}
