"use client";

import { useId, useState } from "react";
import { ErrorNotice } from "@/components/notice";
import { Input } from "@/components/ui/input";
import { fromPct, previewCounts, ratiosValid, toPct } from "@/lib/splits";
import type { SplitConfig, SplitRatios } from "@/lib/types";

export interface SplitDataset {
  split_source: "dataset" | "none";
  given_splits: Record<string, number>;
  n_samples: number;
}

const FIELDS: { key: keyof SplitRatios; label: string }[] = [
  { key: "train", label: "Train %" },
  { key: "val", label: "Valid %" },
  { key: "test", label: "Test %" },
];

export function SplitFields({
  value,
  onChange,
  dataset,
  loadError,
}: {
  value: SplitConfig;
  onChange: (v: SplitConfig) => void;
  dataset: SplitDataset | null;
  /** Set when the train dataset's split info could not be loaded. */
  loadError?: string | null;
}) {
  // Text being typed per input, so a field can be emptied while editing.
  const [draft, setDraft] = useState<Partial<Record<keyof SplitRatios, string>>>({});
  const id = useId();
  const hasGiven = dataset?.split_source === "dataset";
  const error = ratiosValid(value.ratios);
  // Locked while the dataset's splits apply; unlocked whenever the ratios are invalid so they can be fixed.
  const locked = (hasGiven || !!loadError) && value.use_dataset_splits && !error;
  const g = dataset?.given_splits ?? {};
  const preview = dataset && !error ? previewCounts(g, dataset.n_samples, hasGiven ? value : { ...value, use_dataset_splits: false }) : null;

  // The typed text wins while it still matches the value; otherwise show the value (for example after a YAML edit).
  const shown = (key: keyof SplitRatios): string | number => {
    const t = draft[key];
    const v = value.ratios[key];
    if (t !== undefined) {
      const n = t === "" ? NaN : Number(t);
      if (Object.is(fromPct(n), v) || (Number.isNaN(n) && Number.isNaN(v))) return t;
    }
    return Number.isNaN(v) ? "" : toPct(v);
  };

  return (
    <fieldset className="space-y-2 rounded-md border border-border p-3">
      <legend className="px-1 text-sm font-medium">Data splits</legend>
      {loadError && <ErrorNotice>Cannot load the train dataset&apos;s splits ({loadError})</ErrorNotice>}
      {hasGiven && (
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            className="size-4 accent-[var(--saffron)]"
            checked={value.use_dataset_splits}
            onChange={(e) => onChange({ ...value, use_dataset_splits: e.target.checked })}
          />
          Use the dataset&apos;s splits
          <span className="font-mono text-muted-foreground">
            (train {g.train ?? 0} · valid {g.val ?? 0} · test {g.test ?? 0})
          </span>
        </label>
      )}
      <div className="grid grid-cols-3 gap-3">
        {FIELDS.map(({ key, label }) => (
          <div key={key} className="space-y-1">
            <label htmlFor={`${id}-${key}`} className="block text-[13px] text-muted-foreground">{label}</label>
            <Input
              id={`${id}-${key}`}
              type="number"
              min={0}
              max={100}
              step="any"
              disabled={locked}
              aria-invalid={!!error || undefined}
              aria-describedby={error ? `${id}-err` : undefined}
              value={shown(key)}
              onChange={(e) => {
                // Half-typed text such as "5." reads as empty; leave the browser's text alone.
                if (e.target.value === "" && e.target.validity.badInput) return;
                setDraft((d) => ({ ...d, [key]: e.target.value }));
                const n = e.target.value === "" ? NaN : e.target.valueAsNumber;
                onChange({ ...value, ratios: { ...value.ratios, [key]: fromPct(n) } });
              }}
              onBlur={() => setDraft((d) => ({ ...d, [key]: undefined }))}
            />
          </div>
        ))}
      </div>
      {error ? (
        <p id={`${id}-err`} className="text-[13px] text-status-failed">{error}</p>
      ) : (
        preview && (
          <p className="font-mono text-[13px] text-muted-foreground">
            {`≈ train ${preview.train} · valid ${preview.val} · test ${preview.test}`}
          </p>
        )
      )}
    </fieldset>
  );
}
