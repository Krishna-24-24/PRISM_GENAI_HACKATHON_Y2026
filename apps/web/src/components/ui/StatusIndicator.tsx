import clsx from "clsx";

interface Props {
  state: "ok" | "warn" | "err" | "idle";
  label: string;
}

export function StatusIndicator({ state, label }: Props) {
  return (
    <span className="status">
      <span className={clsx("status-dot", state)} />
      {label}
    </span>
  );
}
