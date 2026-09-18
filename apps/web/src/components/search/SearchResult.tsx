import clsx from "clsx";
import type { CodeResult } from "../../api";
import { padRank, formatScore } from "../../lib/format";
import { CodeViewer } from "./CodeViewer";
import { SignalBreakdown } from "./SignalBreakdown";

interface Props {
  result: CodeResult;
  selected: boolean;
  compact?: boolean;
  onSelect: () => void;
}

export function SearchResult({ result, selected, compact, onSelect }: Props) {
  return (
    <article
      className={clsx("result", selected && "selected")}
      onClick={onSelect}
      id={`result-${result.rank}`}
    >
      <div className="result-head">
        <span className="result-rank">{padRank(result.rank)}</span>
        <span className="result-symbol">
          {result.symbol_name}
          {result.symbol_type ? `  ${result.symbol_type}` : ""}
        </span>
        <span className="result-score">{formatScore(result.relevance_score, 3)}</span>
      </div>
      <div className="result-file">
        {result.file_path}:{result.start_line}–{result.end_line}
      </div>
      <CodeViewer
        code={result.code}
        filePath={result.file_path}
        startLine={result.start_line}
        symbolName={result.symbol_name}
        compact={compact}
      />
      <SignalBreakdown
        contributions={result.channel_contributions}
        structuralBoost={result.structural_boost}
      />
    </article>
  );
}
