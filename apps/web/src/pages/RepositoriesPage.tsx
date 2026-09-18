import { useEffect, useState } from "react";
import { fetchRepositories, type RepositorySummary } from "../api";
import { DataTable } from "../components/ui/DataTable";
import { EmptyState } from "../components/ui/EmptyState";
import { ErrorState } from "../components/ui/ErrorState";
import { formatBytes, formatNumber } from "../lib/format";

export function RepositoriesPage() {
  const [rows, setRows] = useState<RepositorySummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<RepositorySummary | null>(null);

  useEffect(() => {
    fetchRepositories()
      .then((data) => {
        setRows(data);
        setSelected(data[0] ?? null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load repositories"))
      .finally(() => setLoading(false));
  }, []);

  return (
    <>
      <div className="page-head">
        <div>
          <div className="page-title">Repositories</div>
          <div className="page-desc">Indexed corpora currently loaded by the retrieval service.</div>
        </div>
      </div>

      {loading && <div className="skeleton" style={{ height: 160 }} />}
      {error && (
        <ErrorState title="REPOSITORY LIST FAILED" reason={error} context={[{ label: "Endpoint", value: "/api/repositories" }]} />
      )}
      {!loading && !error && rows.length === 0 && (
        <EmptyState
          title="No repository indexed"
          description="Index a JavaScript codebase to populate this table. Search is unavailable until chunks are loaded."
        />
      )}
      {!loading && rows.length > 0 && (
        <>
          <DataTable
            columns={[
              { key: "name", header: "Repository", ui: true, render: (r) => r.repo_id },
              { key: "path", header: "Path", render: (r) => r.path || "—" },
              { key: "version", header: "Version", render: (r) => r.version },
              { key: "files", header: "Files", render: (r) => formatNumber(r.files) },
              { key: "chunks", header: "Chunks", render: (r) => formatNumber(r.chunks) },
              { key: "status", header: "Status", render: (r) => r.status },
              { key: "size", header: "Index size", render: (r) => formatBytes(r.index_size_bytes) },
            ]}
            rows={rows}
            rowKey={(r) => r.repo_id}
            selectedKey={selected?.repo_id}
            onRowClick={setSelected}
          />
          {selected && (
            <div className="detail-panel">
              <div className="kv">
                <span className="kv-k">Repository</span>
                <span className="kv-v">{selected.repo_id}</span>
              </div>
              <div className="kv">
                <span className="kv-k">Version</span>
                <span className="kv-v">{selected.version}</span>
              </div>
              <div className="kv">
                <span className="kv-k">Status</span>
                <span className="kv-v">{selected.status}</span>
              </div>
              <div className="kv">
                <span className="kv-k">Files</span>
                <span className="kv-v">{formatNumber(selected.files)}</span>
              </div>
              <div className="kv">
                <span className="kv-k">Chunks</span>
                <span className="kv-v">{formatNumber(selected.chunks)}</span>
              </div>
              <div className="kv">
                <span className="kv-k">Index size</span>
                <span className="kv-v">{formatBytes(selected.index_size_bytes)}</span>
              </div>
            </div>
          )}
        </>
      )}
    </>
  );
}
