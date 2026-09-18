import { NavLink } from "react-router-dom";
import {
  Beaker,
  FolderGit2,
  GitCompare,
  LineChart,
  Search,
  Settings,
} from "lucide-react";
import { useWorkspace } from "../../context/WorkspaceContext";
import { formatNumber } from "../../lib/format";
import { StatusIndicator } from "../ui/StatusIndicator";
import clsx from "clsx";

const NAV = [
  { to: "/", label: "Search", icon: Search, end: true },
  { to: "/repositories", label: "Repositories", icon: FolderGit2 },
  { to: "/versions", label: "Versions", icon: GitCompare },
  { to: "/experiments", label: "Experiments", icon: Beaker },
  { to: "/evaluation", label: "Evaluation", icon: LineChart },
  { to: "/settings", label: "Settings", icon: Settings },
];

interface Props {
  open: boolean;
  onNavigate?: () => void;
}

export function Sidebar({ open, onNavigate }: Props) {
  const { system, systemError } = useWorkspace();
  const apiState = systemError ? "err" : system?.index_ready ? "ok" : system ? "warn" : "idle";
  const apiLabel = systemError ? "API down" : system?.index_ready ? "Index ready" : "Index unavailable";

  return (
    <aside className={clsx("sidebar", open && "open")}>
      <div className="brand">
        <div className="brand-mark" aria-hidden>
          <span /><span /><span /><span />
        </div>
        <div className="brand-text">
          <span className="brand-name">TRACEIQ</span>
          <span className="brand-sub">Agentic Code Intelligence</span>
        </div>
      </div>

      <nav className="nav">
        {NAV.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) => clsx("nav-link", isActive && "active")}
            onClick={onNavigate}
          >
            <item.icon className="nav-icon" size={14} />
            {item.label}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-foot">
        <StatusIndicator state={apiState} label={apiLabel} />
        <div className="meta-row">
          <span>chunks</span>
          <span>{system ? formatNumber(system.chunk_count) : "—"}</span>
        </div>
        <div className="meta-row">
          <span>files</span>
          <span>{system ? formatNumber(system.file_count) : "—"}</span>
        </div>
        <div className="meta-row">
          <span>repository</span>
          <span title={system?.current_repository ?? undefined}>
            {system?.current_repository ?? "—"}
          </span>
        </div>
      </div>
    </aside>
  );
}
