import type { ReactNode } from "react";

interface Props {
  title: string;
  reason: string;
  context?: Array<{ label: string; value: string }>;
  action?: ReactNode;
}

export function ErrorState({ title, reason, context, action }: Props) {
  return (
    <div className="state error">
      <h2>{title}</h2>
      {context?.map((row) => (
        <div key={row.label} className="kv" style={{ marginBottom: 8 }}>
          <span className="kv-k">{row.label}</span>
          <span className="kv-v">{row.value}</span>
        </div>
      ))}
      <p>{reason}</p>
      {action}
    </div>
  );
}
