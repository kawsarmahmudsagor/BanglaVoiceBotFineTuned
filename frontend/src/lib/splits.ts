import type { SplitConfig, SplitRatios } from "./types";

export const DEFAULT_SPLIT: SplitConfig = {
  use_dataset_splits: true,
  ratios: { train: 0.9, val: 0.05, test: 0.05 },
};

export type SplitCounts = { train: number; val: number; test: number };

/** Fraction (0.05) to the percentage shown in the UI (5), without float noise. */
export const toPct = (f: number): number => Math.round(f * 10000) / 100;
/** Percentage typed in the UI (5) to the stored fraction (0.05). */
export const fromPct = (p: number): number => p / 100;

/**
 * Approximates the server's per-run assignment (spec 15.5) for the live preview.
 * Pool = given train rows + unassigned rows; targets = train + every split with no given rows.
 */
export function previewCounts(given: Record<string, number>, n: number, cfg: SplitConfig): SplitCounts {
  const r = cfg.ratios;
  const g = cfg.use_dataset_splits
    ? { train: given.train ?? 0, val: given.val ?? 0, test: given.test ?? 0 }
    : { train: 0, val: 0, test: 0 };
  const assigned = g.train + g.val + g.test;
  const unassigned = cfg.use_dataset_splits ? (given.unassigned ?? Math.max(0, n - assigned)) : n;
  const pool = g.train + unassigned;
  const valIsTarget = g.val === 0;
  const testIsTarget = g.test === 0;
  const total = r.train + (valIsTarget ? r.val : 0) + (testIsTarget ? r.test : 0);
  if (total <= 0) return { train: pool, val: g.val, test: g.test };
  const val = valIsTarget ? Math.round((pool * r.val) / total) : g.val;
  const test = testIsTarget ? Math.round((pool * r.test) / total) : g.test;
  const train = pool - (valIsTarget ? val : 0) - (testIsTarget ? test : 0);
  return { train, val, test };
}

/** Mirrors the backend rules (ratios are fractions). Returns an error message, or null when valid. */
export function ratiosValid(r: SplitRatios): string | null {
  if (![r.train, r.val, r.test].every((x) => typeof x === "number" && Number.isFinite(x))) return "Ratios must add up to 100%";
  if (r.train <= 0) return "Train must be above 0%";
  if (r.val < 0 || r.test < 0) return "Ratios cannot be negative";
  if (Math.abs(r.train + r.val + r.test - 1) > 1e-6) return "Ratios must add up to 100%";
  return null;
}
