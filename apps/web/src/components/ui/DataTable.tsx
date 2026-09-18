import clsx from "clsx";
import type { ReactNode } from "react";

export interface Column<T> {
  key: string;
  header: string;
  ui?: boolean;
  render: (row: T) => ReactNode;
}

interface Props<T> {
  columns: Column<T>[];
  rows: T[];
  rowKey: (row: T) => string;
  selectedKey?: string | null;
  onRowClick?: (row: T) => void;
}

export function DataTable<T>({ columns, rows, rowKey, selectedKey, onRowClick }: Props<T>) {
  return (
    <div className="table-wrap">
      <table className="data">
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key}>{c.header}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const key = rowKey(row);
            return (
              <tr
                key={key}
                className={clsx(selectedKey === key && "selected")}
                onClick={() => onRowClick?.(row)}
              >
                {columns.map((c) => (
                  <td key={c.key} className={c.ui ? "ui" : undefined}>
                    {c.render(row)}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
