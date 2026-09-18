import { useEffect, useState } from "react";
import { fetchEvaluation, type EvaluationResponse } from "../api";
import { DataTable } from "../components/ui/DataTable";
import { EmptyState } from "../components/ui/EmptyState";
import { ErrorState } from "../components/ui/ErrorState";
import { Metric } from "../components/ui/Metric";
import { formatMs, formatScore } from "../lib/format";

export function EvaluationPage() {
  const [data, setData] = useState<EvaluationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());

  useEffect(() => {
    fetchEvaluation()
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load evaluation"));
  }, []);

  const rows = data?.results ?? [];

  const latest = data?.results.at(-1);

  return (
    <>
      <div className="page-head">
        <div>
          <div className="page-title">Evaluation</div>
          <div className="page-desc">Ablation comparison. Empty until a benchmark artifact is present on the API host.</div>
        </div>
      </div>

      {error && <ErrorState title="EVALUATION LOAD FAILED" reason={error} />}

      {data && !data.available && (
        <EmptyState
          title="No benchmark artifact"
          description="Place an ablation JSON under results/ (written by run_final_ablation.py). The UI will not invent MRR, NDCG, or latency."
        />
      )}

      {data?.available && latest && (
        <>
          <div className="metrics-row">
            <Metric label="MRR" value={formatScore(latest.mrr, 4)} />
            <Metric label="NDCG@10" value={formatScore(latest.ndcg_at_10, 4)} />
            <Metric label="P50" value={formatMs(latest.p50_latency_ms, 1)} />
            <Metric label="P95" value={formatMs(latest.p95_latency_ms, 1)} />
            <Metric label="Indexing" value={latest.index_build_time_s != null ? `${latest.index_build_time_s}s` : "—"} />
            <Metric label="Index size" value={latest.index_size_mb != null ? `${latest.index_size_mb} MB` : "—"} />
          </div>
          <p className="page-desc" style={{ marginBottom: 8 }}>
            Headline metrics are the last configuration in {data.source}. Select rows to compare a subset.
          </p>
          <DataTable
            columns={[
              {
                key: "sel",
                header: "",
                render: (r) => (
                  <input
                    type="checkbox"
                    checked={selected.has(r.configuration)}
                    onChange={(e) => {
                      e.stopPropagation();
                      setSelected((prev) => {
                        const next = new Set(prev);
                        if (next.has(r.configuration)) next.delete(r.configuration);
                        else next.add(r.configuration);
                        return next;
                      });
                    }}
                    onClick={(e) => e.stopPropagation()}
                  />
                ),
              },
              { key: "cfg", header: "Configuration", ui: true, render: (r) => r.configuration },
              { key: "mrr", header: "MRR", render: (r) => formatScore(r.mrr, 4) },
              { key: "ndcg", header: "NDCG@10", render: (r) => formatScore(r.ndcg_at_10, 4) },
              { key: "p95", header: "P95", render: (r) => formatMs(r.p95_latency_ms, 1) },
            ]}
            rows={rows}
            rowKey={(r) => r.configuration}
          />
        </>
      )}
    </>
  );
}
