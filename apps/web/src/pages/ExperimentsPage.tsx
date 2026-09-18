import { useEffect, useState } from "react";
import { fetchEvaluation, type EvaluationResponse, type EvaluationRun } from "../api";
import { DataTable } from "../components/ui/DataTable";
import { EmptyState } from "../components/ui/EmptyState";
import { ErrorState } from "../components/ui/ErrorState";
import { formatMs, formatScore } from "../lib/format";

export function ExperimentsPage() {
  const [data, setData] = useState<EvaluationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<EvaluationRun | null>(null);

  useEffect(() => {
    fetchEvaluation()
      .then((d) => {
        setData(d);
        setSelected(d.results[0] ?? null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load experiments"));
  }, []);

  return (
    <>
      <div className="page-head">
        <div>
          <div className="page-title">Experiments</div>
          <div className="page-desc">Benchmark runs written by the ablation pipeline. Values are loaded from disk, never synthesized.</div>
        </div>
      </div>

      {error && <ErrorState title="EXPERIMENT LOAD FAILED" reason={error} />}

      {data && !data.available && (
        <EmptyState
          title="No experiment runs"
          description="This table reads results/ablation_*.json produced by benchmarks/run_final_ablation.py. Until a run exists, TRACEIQ will not display metrics."
        />
      )}

      {data?.available && (
        <>
          <p className="page-desc" style={{ marginBottom: 12 }}>
            Source {data.source} · {data.timestamp ?? "untimed"} · {data.sample_queries ?? "—"} queries · corpus {data.corpus_size ?? "—"}
          </p>
          <DataTable
            columns={[
              { key: "run", header: "Run", render: (r) => r.configuration },
              { key: "ds", header: "Dataset", render: () => "CoIR AppsRetrieval" },
              { key: "mrr", header: "MRR", render: (r) => formatScore(r.mrr, 4) },
              { key: "ndcg", header: "NDCG@10", render: (r) => formatScore(r.ndcg_at_10, 4) },
              { key: "p50", header: "P50", render: (r) => formatMs(r.p50_latency_ms, 1) },
              { key: "p95", header: "P95", render: (r) => formatMs(r.p95_latency_ms, 1) },
              { key: "ts", header: "Timestamp", render: () => data.timestamp ?? "—" },
              { key: "st", header: "Status", render: () => "complete" },
            ]}
            rows={data.results}
            rowKey={(r) => r.configuration}
            selectedKey={selected?.configuration}
            onRowClick={setSelected}
          />
          {selected && (
            <div className="detail-panel">
              <div className="kv"><span className="kv-k">Configuration</span><span className="kv-v">{selected.configuration}</span></div>
              <div className="kv"><span className="kv-k">Queries evaluated</span><span className="kv-v">{selected.queries_evaluated ?? "—"}</span></div>
              <div className="kv"><span className="kv-k">Index build</span><span className="kv-v">{selected.index_build_time_s ?? "—"} s</span></div>
              <div className="kv"><span className="kv-k">Index size</span><span className="kv-v">{selected.index_size_mb ?? "—"} MB</span></div>
              <div className="kv"><span className="kv-k">Incremental update</span><span className="kv-v">{selected.incremental_update_time_s ?? "—"} s</span></div>
              <div className="kv"><span className="kv-k">Hardware</span><span className="kv-v">as recorded in the ablation host (not stored in JSON)</span></div>
            </div>
          )}
        </>
      )}
    </>
  );
}
