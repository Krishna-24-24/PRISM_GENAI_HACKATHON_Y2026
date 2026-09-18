import { useCallback, useEffect, useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { fetchExampleQueries, type ExampleQuery } from "./api";
import { WorkspaceProvider } from "./context/WorkspaceContext";
import { AppShell } from "./components/layout/AppShell";
import { CommandPalette } from "./components/ui/CommandPalette";
import { SearchPage } from "./pages/SearchPage";
import { RepositoriesPage } from "./pages/RepositoriesPage";
import { VersionsPage } from "./pages/VersionsPage";
import { ExperimentsPage } from "./pages/ExperimentsPage";
import { EvaluationPage } from "./pages/EvaluationPage";
import { SettingsPage } from "./pages/SettingsPage";

export default function App() {
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [examples, setExamples] = useState<ExampleQuery[]>([]);
  const [pendingQuery, setPendingQuery] = useState<string | null>(null);

  useEffect(() => {
    fetchExampleQueries().then(setExamples).catch(() => setExamples([]));
  }, []);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((v) => !v);
      }
      if (e.key === "Escape") setPaletteOpen(false);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const onPendingConsumed = useCallback(() => setPendingQuery(null), []);

  return (
    <WorkspaceProvider>
      <Routes>
        <Route element={<AppShell onCommand={() => setPaletteOpen(true)} />}>
          <Route
            path="/"
            element={
              <SearchPage pendingQuery={pendingQuery} onPendingConsumed={onPendingConsumed} />
            }
          />
          <Route path="/repositories" element={<RepositoriesPage />} />
          <Route path="/versions" element={<VersionsPage />} />
          <Route path="/experiments" element={<ExperimentsPage />} />
          <Route path="/evaluation" element={<EvaluationPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        examples={examples}
        onRunQuery={setPendingQuery}
      />
    </WorkspaceProvider>
  );
}
