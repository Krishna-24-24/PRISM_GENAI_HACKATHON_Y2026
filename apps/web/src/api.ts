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

export interface SystemStatus {
  status: string;
  environment: string;
  index_ready: boolean;
  index_path: string;
  chunk_count: number;
  file_count: number;
  index_size_bytes: number;
  current_repository?: string | null;
  current_version?: string | null;
  dense_top_k?: number | null;
  bm25_top_k?: number | null;
  symbol_top_k?: number | null;
  reason?: string | null;
}

export interface RepositorySummary {
  repo_id: string;
  version: string;
  files: number;
  chunks: number;
  status: string;
  index_size_bytes: number;
  path: string;
}

export interface EvaluationRun {
  configuration: string;
  mrr?: number | null;
  ndcg_at_10?: number | null;
  p50_latency_ms?: number | null;
  p95_latency_ms?: number | null;
  index_build_time_s?: number | null;
  incremental_update_time_s?: number | null;
  index_size_mb?: number | null;
  queries_evaluated?: number | null;
}

export interface EvaluationResponse {
  available: boolean;
  source?: string | null;
  timestamp?: string | null;
  sample_queries?: number | null;
  corpus_size?: number | null;
  results: EvaluationRun[];
}

async function parseError(res: Response): Promise<string> {
  const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
  return err.error ?? err.detail ?? `HTTP ${res.status}`;
}

export async function searchCode(
  query: string,
  topK = 10,
  showTrace = true
): Promise<SearchResponse> {
  const res = await fetch(`${API_BASE}/api/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK, show_trace: showTrace }),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function fetchExampleQueries(): Promise<ExampleQuery[]> {
  const res = await fetch(`${API_BASE}/api/example-queries`);
  if (!res.ok) throw new Error("Failed to fetch example queries");
  const data: { examples: ExampleQuery[] } = await res.json();
  return data.examples;
}

export async function fetchSystemStatus(): Promise<SystemStatus> {
  const res = await fetch(`${API_BASE}/api/system`);
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function fetchRepositories(): Promise<RepositorySummary[]> {
  const res = await fetch(`${API_BASE}/api/repositories`);
  if (!res.ok) throw new Error(await parseError(res));
  const data: { repositories: RepositorySummary[] } = await res.json();
  return data.repositories;
}

export async function fetchEvaluation(): Promise<EvaluationResponse> {
  const res = await fetch(`${API_BASE}/api/evaluation`);
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function fetchHealth(): Promise<{ status: string; environment: string }> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}
