import { type CodeResult } from "../api";
import CodeMirror from "@uiw/react-codemirror";
import { javascript } from "@codemirror/lang-javascript";
import { oneDark } from "@codemirror/theme-one-dark";
import { GitBranch, FileCode, ArrowUp } from "lucide-react";
import clsx from "clsx";

interface Props {
  result: CodeResult;
}

function scoreClass(s: number): "high" | "mid" | "low" {
  if (s >= 0.012) return "high";
  if (s >= 0.006) return "mid";
  return "low";
}

/** Guess the language from the file extension for the CodeMirror highlight hint. */
function isJsLike(filePath: string): boolean {
  return /\.(js|jsx|ts|tsx|mjs|cjs)$/.test(filePath);
}

export function ResultCard({ result }: Props) {
  const sc = result.relevance_score;
  const cls = scoreClass(sc);

  // Normalise score to 0–100 for the bar (use a soft log scale)
  const barPct = Math.min(100, (sc / 0.015) * 100);

  const channelEntries = Object.entries(result.channel_contributions).filter(([, v]) => v > 0);

  return (
    <div className="result-card">
      {/* Header */}
      <div className="result-header">
        <span className="result-rank">#{result.rank}</span>

        <div className="result-meta">
          {/* File path */}
          <span className="result-path" title={result.file_path}>
            <FileCode size={11} style={{ display: "inline", verticalAlign: "middle", marginRight: 4 }} />
            {result.file_path}
          </span>

          {/* Symbol name + type */}
          <div className="result-symbol-row">
            <span className="result-symbol">{result.symbol_name}</span>
            <span className="symbol-type-badge">{result.symbol_type}</span>
            <span className="result-lines">
              L{result.start_line}–{result.end_line}
            </span>
          </div>
        </div>

        {/* Score column */}
        <div className="result-score-col">
          <span className={clsx("score-value", cls)}>
            {sc.toFixed(4)}
          </span>
          <div className="score-bar-wrap">
            <div
              className={clsx("score-bar-fill", cls)}
              style={{ width: `${barPct}%` }}
            />
          </div>
        </div>
      </div>

      {/* Code block */}
      <div className="code-block">
        <CodeMirror
          value={result.code}
          extensions={isJsLike(result.file_path) ? [javascript({ jsx: true, typescript: true })] : []}
          theme={oneDark}
          editable={false}
          basicSetup={{
            lineNumbers: true,
            foldGutter: false,
            highlightActiveLine: false,
            highlightSelectionMatches: false,
          }}
          style={{ maxHeight: 280, overflow: "auto" }}
        />
      </div>

      {/* Footer: channel contributions + structural boost + calls */}
      <div className="result-footer">
        {channelEntries.map(([ch, contrib]) => (
          <span key={ch} className="channel-contribution">
            <span className={clsx("ch-dot", ch)} />
            {ch}: {Number(contrib).toFixed(3)}
          </span>
        ))}

        {result.structural_boost > 0 && (
          <span className="structural-boost-badge">
            +{result.structural_boost.toFixed(4)} struct
          </span>
        )}

        {result.calls.length > 0 && (
          <span className="calls-badge">
            <ArrowUp size={10} />
            calls:
            {result.calls.slice(0, 3).map((c) => (
              <code key={c}>{c}</code>
            ))}
            {result.calls.length > 3 && <span>+{result.calls.length - 3}</span>}
          </span>
        )}

        {result.imports.length > 0 && (
          <span className="calls-badge">
            <GitBranch size={10} />
            {result.imports.length} import{result.imports.length !== 1 ? "s" : ""}
          </span>
        )}
      </div>
    </div>
  );
}
