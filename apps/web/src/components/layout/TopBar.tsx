import { Menu } from "lucide-react";
import { useWorkspace } from "../../context/WorkspaceContext";
import { formatBytes } from "../../lib/format";
import { Kbd } from "../ui/Kbd";
import { StatusIndicator } from "../ui/StatusIndicator";

interface Props {
  onMenu: () => void;
  onCommand: () => void;
}

export function TopBar({ onMenu, onCommand }: Props) {
  const { system, systemError } = useWorkspace();
  const apiState = systemError ? "err" : system?.status === "ok" ? "ok" : "warn";

  return (
    <header className="topbar">
      <div className="topbar-cluster">
        <button className="btn btn-ghost mobile-toggle" onClick={onMenu} aria-label="Open navigation">
          <Menu size={14} />
        </button>
        <div className="topbar-item">
          <span className="topbar-label">Repository</span>
          <span className="topbar-value">{system?.current_repository ?? "—"}</span>
        </div>
        <div className="topbar-item">
          <span className="topbar-label">Version</span>
          <span className="topbar-value">{system?.current_version ?? "—"}</span>
        </div>
        <div className="topbar-item">
          <span className="topbar-label">Index</span>
          <span className="topbar-value">
            {system?.index_ready ? `${formatBytes(system.index_size_bytes)} · ready` : "not loaded"}
          </span>
        </div>
      </div>
      <div className="topbar-cluster">
        <StatusIndicator state={apiState} label={systemError ? "API unreachable" : "System"} />
        <button className="btn btn-ghost" onClick={onCommand} title="Command palette">
          Command <Kbd>⌘K</Kbd>
        </button>
      </div>
    </header>
  );
}
