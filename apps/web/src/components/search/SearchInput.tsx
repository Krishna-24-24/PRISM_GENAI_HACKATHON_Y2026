import { useEffect, useRef } from "react";
import { Search } from "lucide-react";
import { Kbd } from "../ui/Kbd";

interface Props {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  loading?: boolean;
  disabled?: boolean;
}

export function SearchInput({ value, onChange, onSubmit, loading, disabled }: Props) {
  const ref = useRef<HTMLInputElement>(null);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement | null)?.tagName;
      const typing = tag === "INPUT" || tag === "TEXTAREA" || (e.target as HTMLElement)?.isContentEditable;
      if (e.key === "/" && !typing) {
        e.preventDefault();
        ref.current?.focus();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="search-box">
      <input
        ref={ref}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            onSubmit();
          }
        }}
        placeholder="How is the input preprocessed before going to the main function?"
        spellCheck={false}
        autoComplete="off"
        disabled={disabled}
      />
      <button className="btn btn-primary" onClick={onSubmit} disabled={loading || disabled || !value.trim()}>
        <Search size={13} />
        {loading ? "Retrieving" : "Search"}
        {!loading && <Kbd>↵</Kbd>}
      </button>
    </div>
  );
}
