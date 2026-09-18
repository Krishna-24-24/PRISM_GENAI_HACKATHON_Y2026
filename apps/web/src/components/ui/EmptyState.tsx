import type { ReactNode } from "react";

interface Props {
  title: string;
  description?: string;
  hints?: string[];
  action?: ReactNode;
}

export function EmptyState({ title, description, hints, action }: Props) {
  return (
    <div className="state">
      <h2>{title}</h2>
      {description && <p>{description}</p>}
      {hints && hints.length > 0 && (
        <ul>
          {hints.map((h) => (
            <li key={h}>{h}</li>
          ))}
        </ul>
      )}
      {action && <div style={{ marginTop: 12 }}>{action}</div>}
    </div>
  );
}
