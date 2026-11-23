import type { RunMetric } from "./types";

export const MIN_COMPARE = 2;
export const MAX_COMPARE = 4;

export function parseCompareIds(q: string | null, known?: Set<string>): { ids: string[] } | { error: string } {
  const ids = [...new Set((q ?? "").split(",").map((s) => s.trim()).filter(Boolean))];
  if (ids.length < MIN_COMPARE) return { error: "Pick at least 2 runs to compare." };
  if (ids.length > MAX_COMPARE) return { error: "Compare up to 4 runs at a time." };
  const unknown = known ? ids.find((id) => !known.has(id)) : undefined;
  if (unknown) return { error: `Unknown run: ${unknown}` };
  return { ids };
}

const isPlainObject = (v: unknown): v is Record<string, unknown> =>
  v !== null && typeof v === "object" && !Array.isArray(v);

export function flatten(obj: Record<string, unknown>, prefix = ""): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(obj)) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (isPlainObject(v) && Object.keys(v).length > 0) Object.assign(out, flatten(v, key));
    else out[key] = v;
  }
  return out;
}

export interface DiffRow {
  key: string;
  values: Record<string, unknown>;
  differs: boolean;
}

/** The parts of a recipe spec worth comparing, flattened to dotted keys. */
function comparable(spec: Record<string, unknown>): Record<string, unknown> {
  const pick: Record<string, unknown> = {};
  for (const k of ["base_model", "method", "datasets", "hparams"]) if (k in spec) pick[k] = spec[k];
  return flatten(pick);
}

export function diffHparams(specs: Record<string, Record<string, unknown> | null>): DiffRow[] {
  const loaded = Object.entries(specs).filter((e): e is [string, Record<string, unknown>] => e[1] !== null);
  const flat = loaded.map(([id, s]) => [id, comparable(s)] as const);
  const keys = [...new Set(flat.flatMap(([, f]) => Object.keys(f)))];
  const rows = keys.map((key) => {
    const values: Record<string, unknown> = {};
    for (const [id, f] of flat) values[id] = f[key];
    const present = flat.every(([, f]) => key in f);
    const first = JSON.stringify(values[flat[0][0]]);
    const differs = !present || flat.some(([id]) => JSON.stringify(values[id]) !== first);
    return { key, values, differs };
  });
  return rows.sort((a, b) => Number(b.differs) - Number(a.differs) || a.key.localeCompare(b.key));
}

export type MergedRow = { step: number } & Record<string, number>;

/** One chart series per `split/name`; each row is a step with one column per run that logged it. */
export function mergeMetrics(byRun: Record<string, RunMetric[]>): Record<string, MergedRow[]> {
  const series: Record<string, Map<number, MergedRow>> = {};
  for (const [runId, metrics] of Object.entries(byRun)) {
    for (const m of metrics) {
      const key = `${m.split}/${m.name}`;
      const rows = (series[key] ??= new Map());
      const row = rows.get(m.step) ?? ({ step: m.step } as MergedRow);
      row[runId] = m.value;
      rows.set(m.step, row);
    }
  }
  return Object.fromEntries(
    Object.entries(series).map(([k, rows]) => [k, [...rows.values()].sort((a, b) => a.step - b.step)]),
  );
}

export function metricDirection(name: string): "min" | "max" {
  return /loss|wer|cer|error|rtf/i.test(name) ? "min" : "max";
}

export function bestRun(values: Record<string, number | undefined>, dir: "min" | "max"): string | null {
  let best: string | null = null;
  for (const [id, v] of Object.entries(values)) {
    if (v === undefined || Number.isNaN(v)) continue;
    if (best === null || (dir === "min" ? v < values[best]! : v > values[best]!)) best = id;
  }
  return best;
}
