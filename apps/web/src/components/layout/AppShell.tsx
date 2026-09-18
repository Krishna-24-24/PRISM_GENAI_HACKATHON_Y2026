import { useState } from "react";
import { Outlet } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";

interface Props {
  onCommand: () => void;
}

export function AppShell({ onCommand }: Props) {
  const [navOpen, setNavOpen] = useState(false);

  return (
    <div className="shell">
      <Sidebar open={navOpen} onNavigate={() => setNavOpen(false)} />
      <div className="workspace">
        <TopBar onMenu={() => setNavOpen((v) => !v)} onCommand={onCommand} />
        <main className="page">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
