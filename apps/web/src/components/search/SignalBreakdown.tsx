const LABELS: Record<string, string> = {
  dense: "Semantic",
  bm25: "Lexical",
  symbol: "Symbol",
  structural: "Structural",
};

interface Props {
  contributions: Record<string, number>;
  structuralBoost?: number;
}

export function SignalBreakdown({ contributions, structuralBoost = 0 }: Props) {
  const entries = Object.entries(contributions).filter(([, v]) => v !== 0);
  if (structuralBoost > 0) entries.push(["structural", structuralBoost]);
  if (entries.length === 0) return null;

  return (
    <div className="signals">
      {entries.map(([key, value]) => (
        <span key={key}>
          {LABELS[key] ?? key}
          <b>{value.toFixed(3)}</b>
        </span>
      ))}
    </div>
  );
}
