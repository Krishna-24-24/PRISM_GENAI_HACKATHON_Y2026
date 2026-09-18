// TRACEIQ API client — typed wrappers around the FastAPI endpoints

export const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8080";

export interface QuerySignals {
  has_backtick_identifier: boolean;
  has_camel_case_token: boolean;
  has_snake_case_token: boolean;
  has_dot_accessor: boolean;
  has_structural_keyword: boolean;
  has_usage_keyword: boolean;
  has_behavioral_keyword: boolean;
  identifier_count: number;
  word_count: number;
  extracted_identifiers: string[];
}

export interface QueryProfile {
  category: string;
  routing_reason: string;
  recommended_weights: Record<string, number>;
  signals: QuerySignals;
}

export interface CodeResult {
  rank: number;
  chunk_id: string;
  file_path: string;
  start_line: number;
  end_line: number;
  symbol_name: string;
  symbol_type: string;
  code: string;
  relevance_score: number;
  fused_score: number;
  structural_boost: number;
  channel_contributions: Record<string, number>;
  imports: string[];
  calls: string[];
}

export interface TraceStep {
  step: string;
  detail: string;
  count?: number;
}

export interface SearchResponse {
  query: string;
  results: CodeResult[];
  query_profile: QueryProfile;
  channels_used: string[];
  total_candidates: number;
  returned_count: number;
  latency_ms: number;
  trace?: TraceStep[];
}

export interface ExampleQuery {
  label: string;
  category: string;
  query: string;
  description: string;
}

export interface ExampleQueriesResponse {
  examples: ExampleQuery[];
}

export async function searchCode(
  query: string,
  topK = 10,
  showTrace = false
): Promise<SearchResponse> {
  const res = await fetch(`${API_BASE}/api/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK, show_trace: showTrace }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: "Unknown error" }));
    throw new Error(err.error ?? `HTTP ${res.status}`);
  }
  return res.json();
}

export async function fetchExampleQueries(): Promise<ExampleQuery[]> {
  const res = await fetch(`${API_BASE}/api/example-queries`);
  if (!res.ok) throw new Error("Failed to fetch example queries");
  const data: ExampleQueriesResponse = await res.json();
  return data.examples;
}
