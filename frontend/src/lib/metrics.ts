import type { Phase, RunMetric, RunState } from "./types";

export const LOG_LIMIT = 500;
export const PHASES: Phase[] = ["prepare", "train", "eval", "export"];
export type PhaseState = "done" | "active" | "pending" | "failed";
export type SeriesPoint = { step: number; value: number };

/** Group metrics by "<split>/<name>", sorted by step; the last value wins for a duplicate step (resumed runs). */
export function metricsToSeries(items: RunMetric[]): Record<string, SeriesPoint[]> {
  const byKey = new Map<string, Map<number, number>>();
  for (const it of items) {
    const key = `${it.split}/${it.name}`;
    let steps = byKey.get(key);
    if (!steps) byKey.set(key, (steps = new Map()));
    steps.set(it.step, it.value);
  }
  const out: Record<string, SeriesPoint[]> = {};
  for (const [key, steps] of byKey) {
    out[key] = [...steps].map(([step, value]) => ({ step, value })).sort((a, b) => a.step - b.step);
  }
  return out;
}

/** Append incoming metrics, dropping exact replays ((split, name, step, ts) already seen), e.g. after an SSE reconnect. */
export function mergeMetrics(existing: RunMetric[], incoming: RunMetric[]): RunMetric[] {
  const id = (x: RunMetric) => `${x.split}\u0000${x.name}\u0000${x.step}\u0000${x.ts}`;
  const seen = new Set(existing.map(id));
  const added = incoming.filter((x) => {
    const k = id(x);
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
  return added.length ? [...existing, ...added] : existing;
}

export function appendLog(lines: string[], line: string): string[] {
  const next = [...lines, line];
  return next.length > LOG_LIMIT ? next.slice(next.length - LOG_LIMIT) : next;
}

const STATE_PHASE: Partial<Record<RunState, Phase>> = {
  preparing: "prepare", training: "train", evaluating: "eval", exporting: "export",
};

/** Per-phase display state for [prepare, train, eval, export]. */
export function phaseStates(statusPhase: Phase | null | undefined, state: RunState): PhaseState[] {
  if (state === "done") return PHASES.map(() => "done");
  const current = statusPhase ?? STATE_PHASE[state];
  const idx = current ? PHASES.indexOf(current) : -1;
  const terminalBad = state === "failed" || state === "cancelled";
  return PHASES.map((_, i) => {
    if (i < idx) return "done";
    if (i === idx) return terminalBad ? "failed" : "active";
    return "pending";
  });
}

export type MetricRow = { name: string; value: number; base?: number; delta?: number };

/** One row per metric; `base/x` is paired with `x` (value, base, delta) and hidden from its own row. */
export function metricRows(metrics: Record<string, number>): MetricRow[] {
  const rows: MetricRow[] = [];
  for (const [name, value] of Object.entries(metrics)) {
    if (name.startsWith("base/")) continue;
    const base = metrics[`base/${name}`];
    rows.push(base === undefined ? { name, value } : { name, value, base, delta: Number((value - base).toPrecision(12)) });
  }
  return rows;
}
