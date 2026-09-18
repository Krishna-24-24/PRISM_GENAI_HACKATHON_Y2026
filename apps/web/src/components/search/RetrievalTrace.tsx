import type { TraceStep } from "../../api";

interface Props {
  steps: TraceStep[];
}

export function RetrievalTrace({ steps }: Props) {
  return (
    <div className="diag">
      <div className="diag-h">Retrieval trace</div>
      <div className="diag-b">
        <div className="trace">
          {steps.map((step, i) => (
            <div className="trace-step" key={`${step.step}-${i}`}>
              <div className="trace-rail">
                <div className="trace-node" />
              </div>
              <div>
                <div className="trace-name">{step.step}</div>
                <div className="trace-detail">
                  {step.detail}
                  {step.count != null ? ` · ${step.count}` : ""}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
