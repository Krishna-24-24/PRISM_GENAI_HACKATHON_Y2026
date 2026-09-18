import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import clsx from "clsx";
import type { ExampleQuery } from "../../api";

interface Props {
  open: boolean;
  onClose: () => void;
  examples: ExampleQuery[];
  onRunQuery: (query: string) => void;
}

export function CommandPalette({ open, onClose, examples, onRunQuery }: Props) {
  const navigate = useNavigate();
  const [q, setQ] = useState("");
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const items = useMemo(() => {
    const nav = [
      { id: "search", label: "Go to Search", hint: "g s", run: () => navigate("/") },
      { id: "repos", label: "Go to Repositories", hint: "g r", run: () => navigate("/repositories") },
      { id: "versions", label: "Go to Versions", hint: "g v", run: () => navigate("/versions") },
      { id: "experiments", label: "Go to Experiments", hint: "g e", run: () => navigate("/experiments") },
      { id: "eval", label: "Go to Evaluation", hint: "g l", run: () => navigate("/evaluation") },
      { id: "settings", label: "Go to Settings", hint: "g ,", run: () => navigate("/settings") },
    ];
    const queries = examples.map((ex) => ({
      id: `q-${ex.label}`,
      label: `Search: ${ex.query}`,
      hint: ex.category,
      run: () => {
        navigate("/");
        onRunQuery(ex.query);
      },
    }));
    const all = [...nav, ...queries];
    const needle = q.trim().toLowerCase();
    if (!needle) return all;
    return all.filter((item) => item.label.toLowerCase().includes(needle) || item.hint.toLowerCase().includes(needle));
  }, [examples, navigate, onRunQuery, q]);

  useEffect(() => {
    if (open) {
      setQ("");
      setActive(0);
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  useEffect(() => {
    setActive(0);
  }, [q]);

  if (!open) return null;

  function run(i: number) {
    const item = items[i];
    if (!item) return;
    item.run();
    onClose();
  }

  return (
    <div className="palette-backdrop" onMouseDown={onClose}>
      <div className="palette" onMouseDown={(e) => e.stopPropagation()}>
        <input
          ref={inputRef}
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Go to page or run a query…"
          onKeyDown={(e) => {
            if (e.key === "Escape") onClose();
            if (e.key === "ArrowDown") {
              e.preventDefault();
              setActive((n) => Math.min(items.length - 1, n + 1));
            }
            if (e.key === "ArrowUp") {
              e.preventDefault();
              setActive((n) => Math.max(0, n - 1));
            }
            if (e.key === "Enter") {
              e.preventDefault();
              run(active);
            }
          }}
        />
        <div className="palette-list">
          {items.map((item, i) => (
            <button
              key={item.id}
              className={clsx("palette-item", i === active && "active")}
              onMouseEnter={() => setActive(i)}
              onClick={() => run(i)}
            >
              <span>{item.label}</span>
              <span className="mono" style={{ color: "var(--text-muted)" }}>{item.hint}</span>
            </button>
          ))}
          {items.length === 0 && (
            <div className="palette-item">No matching commands</div>
          )}
        </div>
      </div>
    </div>
  );
}
