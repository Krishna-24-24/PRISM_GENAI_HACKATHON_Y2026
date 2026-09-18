import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { fetchSystemStatus, type SystemStatus } from "../api";
import { loadSettings, saveSettings, type AppSettings } from "../lib/settings";

interface WorkspaceValue {
  system: SystemStatus | null;
  systemError: string | null;
  systemLoading: boolean;
  refreshSystem: () => Promise<void>;
  settings: AppSettings;
  updateSettings: (patch: Partial<AppSettings>) => void;
}

const WorkspaceContext = createContext<WorkspaceValue | null>(null);

export function WorkspaceProvider({ children }: { children: ReactNode }) {
  const [system, setSystem] = useState<SystemStatus | null>(null);
  const [systemError, setSystemError] = useState<string | null>(null);
  const [systemLoading, setSystemLoading] = useState(true);
  const [settings, setSettings] = useState<AppSettings>(loadSettings);

  const refreshSystem = useCallback(async () => {
    setSystemLoading(true);
    try {
      const data = await fetchSystemStatus();
      setSystem(data);
      setSystemError(null);
    } catch (e) {
      setSystem(null);
      setSystemError(e instanceof Error ? e.message : "API unreachable");
    } finally {
      setSystemLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshSystem();
    const id = window.setInterval(() => void refreshSystem(), 30_000);
    return () => window.clearInterval(id);
  }, [refreshSystem]);

  const updateSettings = useCallback((patch: Partial<AppSettings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...patch };
      saveSettings(next);
      return next;
    });
  }, []);

  const value = useMemo(
    () => ({ system, systemError, systemLoading, refreshSystem, settings, updateSettings }),
    [system, systemError, systemLoading, refreshSystem, settings, updateSettings]
  );

  return <WorkspaceContext.Provider value={value}>{children}</WorkspaceContext.Provider>;
}

export function useWorkspace(): WorkspaceValue {
  const ctx = useContext(WorkspaceContext);
  if (!ctx) throw new Error("useWorkspace must be used within WorkspaceProvider");
  return ctx;
}
