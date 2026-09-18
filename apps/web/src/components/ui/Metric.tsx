interface Props {
  label: string;
  value: string;
}

export function Metric({ label, value }: Props) {
  return (
    <div className="metric">
      <div className="metric-k">{label}</div>
      <div className="metric-v">{value}</div>
    </div>
  );
}
