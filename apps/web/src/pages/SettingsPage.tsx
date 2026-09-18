import { useState } from "react";
import clsx from "clsx";
import { useWorkspace } from "../context/WorkspaceContext";
import { API_BASE } from "../api";

const SECTIONS = ["General", "Repository", "Retrieval", "Ranking", "Performance", "API", "Appearance"] as const;

export function SettingsPage() {
  const { settings, updateSettings, system } = useWorkspace();
  const [section, setSection] = useState<(typeof SECTIONS)[number]>("General");

  return (
    <>
      <div className="page-head">
        <div>
          <div className="page-title">Settings</div>
          <div className="page-desc">Local client preferences. Retrieval weights remain server-controlled.</div>
        </div>
      </div>

      <div className="settings-grid">
        <nav className="settings-nav">
          {SECTIONS.map((s) => (
            <button key={s} type="button" className={clsx(section === s && "active")} onClick={() => setSection(s)}>
              {s}
            </button>
          ))}
        </nav>

        <div className="settings-section">
          {section === "General" && (
            <>
              <div className="settings-row">
                <div>
                  <strong>Default top-K</strong>
                  <p>Applied when opening Search.</p>
                </div>
                <select
                  className="select"
                  value={settings.defaultTopK}
                  onChange={(e) => updateSettings({ defaultTopK: Number(e.target.value) })}
                >
                  {[5, 10, 20, 30].map((n) => <option key={n} value={n}>{n}</option>)}
                </select>
              </div>
              <div className="settings-row">
                <div>
                  <strong>Always include retrieval trace</strong>
                  <p>Sends show_trace on /api/search.</p>
                </div>
                <label className="check">
                  <input
                    type="checkbox"
                    checked={settings.alwaysShowTrace}
                    onChange={(e) => updateSettings({ alwaysShowTrace: e.target.checked })}
                  />
                  Enabled
                </label>
              </div>
            </>
          )}

          {section === "Repository" && (
            <div className="settings-row">
              <div>
                <strong>Active corpus</strong>
                <p>Derived from loaded chunks. Switching repositories is not exposed by the current API.</p>
              </div>
              <span className="mono">{system?.current_repository ?? "—"}</span>
            </div>
          )}

          {section === "Retrieval" && (
            <>
              <div className="settings-row">
                <div><strong>Dense top-K</strong><p>Server HybridConfig.</p></div>
                <span className="mono">{system?.dense_top_k ?? "—"}</span>
              </div>
              <div className="settings-row">
                <div><strong>BM25 top-K</strong><p>Server HybridConfig.</p></div>
                <span className="mono">{system?.bm25_top_k ?? "—"}</span>
              </div>
              <div className="settings-row">
                <div><strong>Symbol top-K</strong><p>Server HybridConfig.</p></div>
                <span className="mono">{system?.symbol_top_k ?? "—"}</span>
              </div>
            </>
          )}

          {section === "Ranking" && (
            <div className="settings-row">
              <div>
                <strong>Fusion</strong>
                <p>Weighted RRF and structural boosts are owned by the retriever. This client does not override them.</p>
              </div>
            </div>
          )}

          {section === "Performance" && (
            <div className="settings-row">
              <div>
                <strong>Compact result code</strong>
                <p>Collapse snippet bodies after 12 lines.</p>
              </div>
              <label className="check">
                <input
                  type="checkbox"
                  checked={settings.compactResults}
                  onChange={(e) => updateSettings({ compactResults: e.target.checked })}
                />
                Compact
              </label>
            </div>
          )}

          {section === "API" && (
            <div className="settings-row">
              <div>
                <strong>Base URL</strong>
                <p>Set VITE_API_URL at build time. Live value:</p>
              </div>
              <span className="mono">{API_BASE}</span>
            </div>
          )}

          {section === "Appearance" && (
            <div className="settings-row">
              <div>
                <strong>Theme</strong>
                <p>Dark-first engineering workstation. Light theme is not offered.</p>
              </div>
              <span className="mono">dark</span>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
