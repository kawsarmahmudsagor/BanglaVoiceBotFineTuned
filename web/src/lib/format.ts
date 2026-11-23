import type { RunState } from "./types";

const UNITS = ["B", "KB", "MB", "GB", "TB"];

export function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "0 B";
  let i = 0;
  let v = n;
  while (v >= 1024 && i < UNITS.length - 1) {
    v /= 1024;
    i += 1;
  }
  return i === 0 ? `${Math.round(v)} B` : `${v.toFixed(1)} ${UNITS[i]}`;
}

export function formatDuration(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  if (h > 0) return `${h}h ${pad(m)}m`;
  if (m > 0) return `${m}m ${pad(sec)}s`;
  return `${sec}s`;
}

const STATE_COLORS: Record<RunState, string> = {
  queued: "bg-slate-500/15 text-slate-700 dark:text-slate-300",
  preparing: "bg-sky-500/15 text-sky-700 dark:text-sky-300",
  training: "bg-blue-500/15 text-blue-700 dark:text-blue-300",
  evaluating: "bg-violet-500/15 text-violet-700 dark:text-violet-300",
  exporting: "bg-cyan-500/15 text-cyan-700 dark:text-cyan-300",
  done: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300",
  failed: "bg-red-500/15 text-red-700 dark:text-red-300",
  cancelled: "bg-amber-500/15 text-amber-700 dark:text-amber-300",
};

export function stateColor(state: string): string {
  return STATE_COLORS[state as RunState] ?? "bg-muted text-muted-foreground";
}
