import { useId } from "react";

/** A labelled native select for filter bars; the empty value means "any". */
export function FilterSelect<T extends string>({
  label,
  value,
  options,
  onChange,
  anyLabel = "Any",
}: {
  label: string;
  value: T | "";
  options: { value: T; label: string }[];
  onChange: (v: T | "") => void;
  anyLabel?: string;
}) {
  const id = useId();
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <label htmlFor={id} className="text-[13px] text-muted-foreground">{label}</label>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value as T | "")}
        className="h-9 min-w-36 rounded-md border border-border bg-card px-2 text-sm"
      >
        <option value="">{anyLabel}</option>
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </div>
  );
}

export function FilterCheckbox({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  const id = useId();
  return (
    <div className="flex h-9 items-center gap-2 self-end">
      <input id={id} type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="size-4 accent-[var(--saffron)]" />
      <label htmlFor={id} className="text-sm">{label}</label>
    </div>
  );
}
