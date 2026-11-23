"use client";

import { useId } from "react";
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
}: {
  value: SplitConfig;
  onChange: (v: SplitConfig) => void;
  dataset: SplitDataset | null;
}) {
  const id = useId();
  const hasGiven = dataset?.split_source === "dataset";
  const locked = hasGiven && value.use_dataset_splits;
  const error = ratiosValid(value.ratios);
  const g = dataset?.given_splits ?? {};
  const preview = dataset && !error ? previewCounts(g, dataset.n_samples, hasGiven ? value : { ...value, use_dataset_splits: false }) : null;

  return (
    <fieldset className="space-y-2 rounded-md border border-border p-3">
      <legend className="px-1 text-sm font-medium">Data splits</legend>
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
              value={toPct(value.ratios[key])}
              onChange={(e) => {
                const n = e.target.valueAsNumber;
                if (Number.isNaN(n)) return;
                onChange({ ...value, ratios: { ...value.ratios, [key]: fromPct(n) } });
              }}
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
