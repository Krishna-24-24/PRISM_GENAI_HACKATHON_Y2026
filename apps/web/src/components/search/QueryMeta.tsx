import type { QueryProfile } from "../../api";

interface Props {
  query: string;
  profile: QueryProfile;
  channels: string[];
}

export function QueryMeta({ query, profile, channels }: Props) {
  const strategy = channels.length ? channels.join(" + ") : Object.entries(profile.recommended_weights)
    .filter(([, w]) => w > 0)
    .map(([k]) => k)
    .join(" + ");

  return (
    <div className="diag" style={{ marginBottom: 12 }}>
      <div className="diag-h">Query</div>
      <div className="diag-b">
        <div className="kv">
          <span className="kv-k">Query</span>
          <span className="kv-v">{query}</span>
        </div>
        <div className="kv">
          <span className="kv-k">Type</span>
          <span className="kv-v">{profile.category.toUpperCase()}</span>
        </div>
        <div className="kv">
          <span className="kv-k">Strategy</span>
          <span className="kv-v">{strategy || "—"}</span>
        </div>
        <div className="kv">
          <span className="kv-k">Routing</span>
          <span className="kv-v">{profile.routing_reason}</span>
        </div>
      </div>
    </div>
  );
}
