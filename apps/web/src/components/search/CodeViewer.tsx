import { useState } from "react";
import { tokenizeLine } from "../../lib/highlight";
import { formatPath } from "../../lib/format";

interface Props {
  code: string;
  filePath: string;
  startLine: number;
  symbolName: string;
  compact?: boolean;
}

export function CodeViewer({ code, filePath, startLine, symbolName, compact }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState<"code" | "path" | null>(null);
  const lines = code.replace(/\n$/, "").split("\n");
  const visible = expanded || !compact ? lines : lines.slice(0, 12);
  const { dir, file } = formatPath(filePath);

  async function copy(kind: "code" | "path") {
    const text = kind === "code" ? code : filePath;
    await navigator.clipboard.writeText(text);
    setCopied(kind);
    window.setTimeout(() => setCopied(null), 1200);
  }

  return (
    <div className="code-viewer">
      <div className="code-toolbar">
        <span className="code-crumb" title={filePath}>
          {dir ? `${dir}/` : ""}{file} · {symbolName}
        </span>
        <div className="code-actions">
          <button type="button" onClick={() => void copy("path")}>
            {copied === "path" ? "Path copied" : "Copy path"}
          </button>
          <button type="button" onClick={() => void copy("code")}>
            {copied === "code" ? "Copied" : "Copy"}
          </button>
          {lines.length > 12 && (
            <button type="button" onClick={() => setExpanded((v) => !v)}>
              {expanded ? "Collapse" : "Expand"}
            </button>
          )}
        </div>
      </div>
      <div className={expanded ? "code-body expanded" : "code-body"}>
        {visible.map((line, i) => {
          const n = startLine + i;
          const tokens = tokenizeLine(line);
          return (
            <div className="code-line" key={n}>
              <span className="ln">{n}</span>
              <span className="lc">
                {tokens.length === 0 ? " " : tokens.map((t, ti) => (
                  <span key={ti} className={`tok-${t.kind}`}>{t.value}</span>
                ))}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
