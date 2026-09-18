import { useEffect, useMemo, useState } from "react";
import { fetchRepositories, fetchSystemStatus, type RepositorySummary, type SystemStatus } from "../api";
import { EmptyState } from "../components/ui/EmptyState";
import { ErrorState } from "../components/ui/ErrorState";
import { formatNumber } from "../lib/format";

export function VersionsPage() {
  const [repos, setRepos] = useState<RepositorySummary[]>([]);
  const [system, setSystem] = useState<SystemStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([fetchRepositories(), fetchSystemStatus()])
      .then(([r, s]) => {
        setRepos(r);
        setSystem(s);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load versions"));
  }, []);

  const versions = useMemo(() => {
    const unique = Array.from(new Set(repos.map((r) => r.version).filter((v) => v && v !== "—")));
    return unique;
  }, [repos]);

  return (
    <>
      <div className="page-head">
        <div>
          <div className="page-title">Versions</div>
          <div className="page-desc">Incremental index comparison across corpus versions.</div>
        </div>
      </div>

      {error && <ErrorState title="VERSION LOAD FAILED" reason={error} />}

      {!error && versions.length < 2 && (
        <EmptyState
          title="Single version loaded"
          description="A second indexed version is required to compute file/chunk diffs, embedding reuse, and incremental update time. The live index reports only the currently loaded snapshot."
          hints={[
            system?.current_version ? `active version: ${system.current_version}` : "no version_id on loaded chunks",
            `${formatNumber(system?.chunk_count ?? 0)} chunks in the current snapshot`,
            "run VersionManager.set_version() to record an incremental update",
          ]}
        />
      )}

      {versions.length >= 2 && (
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th>Version</th>
                <th>Repository</th>
                <th>Files</th>
                <th>Chunks</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {repos.map((r) => (
                <tr key={`${r.repo_id}-${r.version}`}>
                  <td>{r.version}</td>
                  <td className="ui">{r.repo_id}</td>
                  <td>{formatNumber(r.files)}</td>
                  <td>{formatNumber(r.chunks)}</td>
                  <td>{r.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
