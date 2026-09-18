import { type QueryProfile, type TraceStep } from "../api";
import { useState } from "react";
import {
  Cpu,
  Layers,
  Zap,
  ChevronDown,
  ChevronRight,
  Activity,
  Hash,
  Search,
} from "lucide-react";
import clsx from "clsx";

interface Props {
  profile: QueryProfile | null;
  latencyMs: number;
  totalCandidates: number;
  returnedCount: number;
  trace?: TraceStep[];
}

const CATEGORY_ICON: Record<string, React.ReactNode> = {
  semantic:   <Search size={11} />,
  symbolic:   <Hash size={11} />,
  usage:      <Zap size={11} />,
  structural: <Layers size={11} />,
  mixed:      <Activity size={11} />,
};

const SIGNAL_LABELS: [keyof NonNullable<QueryProfile["signals"]>, string][] = [
  ["has_backtick_identifier", "backtick id"],
  ["has_camel_case_token", "camelCase"],
  ["has_snake_case_token", "snake_case"],
  ["has_dot_accessor", "dot accessor"],
  ["has_structural_keyword", "struct keyword"],
  ["has_usage_keyword", "usage keyword"],
  ["has_behavioral_keyword", "behavioral kw"],
];

export function QueryInsightPanel({ profile, latencyMs, totalCandidates, returnedCount, trace }: Props) {
  const [traceOpen, setTraceOpen] = useState(false);

  const category = profile?.category ?? "semantic";
  const weights = profile?.recommended_weights ?? { dense: 1.0 };
  const signals = profile?.signals;
  const identifiers = signals?.extracted_identifiers ?? [];

  const channelEntries = Object.entries(weights).filter(([, w]) => w > 0);

  return (
    <aside className="insight-sidebar">
      <div className="panel">
        <div className="panel-title">
          <Cpu size={12} />
          Query Analysis
        </div>

        {/* Category */}
        <div className="insight-row">
          <span className="insight-key">Category</span>
          <span className={clsx("category-badge", category)}>
            {CATEGORY_ICON[category]}
            {category}
          </span>
        </div>

        {/* Channels */}
        <div className="insight-row">
          <span className="insight-key">Channels</span>
          <div className="channel-list">
            {channelEntries.map(([ch, w]) => (
              <div className="channel-row" key={ch}>
                <span className="channel-name">
                  <span className={clsx("ch-dot", ch)} />
                  {ch}
                </span>
                <span className="channel-weight">w={w.toFixed(1)}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Latency */}
        <div className="insight-row">
          <span className="insight-key">Latency</span>
          <div className="latency-badge">
            <span className="latency-ms">{latencyMs.toFixed(0)}</span>
            <span className="latency-unit">ms</span>
          </div>
        </div>

        {/* Candidate / returned counts */}
        <div className="insight-row">
          <span className="insight-key">Candidates</span>
          <div className="stats-row">
            <div className="stat-item">
              <span className="stat-val">{totalCandidates}</span>
              <span className="stat-key">retrieved</span>
            </div>
            <div className="stat-item">
              <span className="stat-val">{returnedCount}</span>
              <span className="stat-key">returned</span>
            </div>
          </div>
        </div>

        {/* Routing reason */}
        {profile?.routing_reason && (
          <div className="insight-row">
            <span className="insight-key">Routing Reason</span>
            <div className="routing-reason">{profile.routing_reason}</div>
          </div>
        )}

        {/* Signals */}
        {signals && (
          <div className="insight-row">
            <span className="insight-key">Signals</span>
            <div className="signals-grid">
              {SIGNAL_LABELS.map(([key, label]) => {
                const active = signals[key as keyof typeof signals] as boolean;
                return (
                  <div key={key} className={clsx("signal-item", active && "active")}>
                    <span className={clsx("signal-dot", active && "active")} />
                    {label}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Extracted identifiers */}
        {identifiers.length > 0 && (
          <div className="insight-row">
            <span className="insight-key">Identifiers ({identifiers.length})</span>
            <div className="idents-list">
              {identifiers.slice(0, 8).map((id) => (
                <span key={id} className="ident-chip">{id}</span>
              ))}
            </div>
          </div>
        )}

        {/* Trace accordion */}
        {trace && trace.length > 0 && (
          <div className="trace-accordion">
            <button className="trace-toggle" onClick={() => setTraceOpen(!traceOpen)}>
              <span>Retrieval Trace</span>
              {traceOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            </button>
            {traceOpen && (
              <div className="trace-steps">
                {trace.map((step, i) => (
                  <div key={i} className="trace-step">
                    <span className="trace-step-name">{step.step}</span>
                    <span className="trace-step-detail">
                      {step.detail}
                      {step.count !== undefined && step.count !== null
                        ? ` · ${step.count}`
                        : ""}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </aside>
  );
}
