import { RUN_STATES, type RunListItem, type RunState, type RunStatus, type Task } from "./types";

const TASKS: Task[] = ["llm-sft", "asr", "tts"];

export interface RunFilter {
  state?: RunState[];
  task?: Task;
  recipe?: string;
  backend?: string;
  commercialOnly?: boolean;
  /** ISO dates (YYYY-MM-DD), inclusive, compared with the UTC date of created_at. */
  from?: string;
  to?: string;
}

export function applyRunFilter(runs: RunListItem[], f: RunFilter): RunListItem[] {
  return runs.filter((r) => {
    const day = r.created_at?.slice(0, 10) ?? "";
    return (
      (!f.state?.length || f.state.includes(r.state)) &&
      (!f.task || r.task === f.task) &&
      (!f.recipe || r.recipe === f.recipe) &&
      (!f.backend || r.backend === f.backend) &&
      (!f.commercialOnly || r.commercial_ok) &&
      (!f.from || (day !== "" && day >= f.from)) &&
      (!f.to || (day !== "" && day <= f.to))
    );
  });
}

export function filterToQuery(f: RunFilter): URLSearchParams {
  const q = new URLSearchParams();
  if (f.state?.length) q.set("state", f.state.join(","));
  if (f.task) q.set("task", f.task);
  if (f.recipe) q.set("recipe", f.recipe);
  if (f.backend) q.set("backend", f.backend);
  if (f.commercialOnly) q.set("commercial", "1");
  if (f.from) q.set("from", f.from);
  if (f.to) q.set("to", f.to);
  return q;
}

const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;

export function queryToFilter(q: URLSearchParams): RunFilter {
  const f: RunFilter = {};
  const states = (q.get("state") ?? "")
    .split(",")
    .filter((s): s is RunState => (RUN_STATES as readonly string[]).includes(s));
  if (states.length) f.state = states;
  const task = q.get("task");
  if (task && (TASKS as string[]).includes(task)) f.task = task as Task;
  const recipe = q.get("recipe");
  if (recipe) f.recipe = recipe;
  const backend = q.get("backend");
  if (backend) f.backend = backend;
  if (q.get("commercial") === "1") f.commercialOnly = true;
  const from = q.get("from");
  if (from && DATE_RE.test(from)) f.from = from;
  const to = q.get("to");
  if (to && DATE_RE.test(to)) f.to = to;
  return f;
}

export function compareHref(ids: string[]): string {
  return `/runs/compare?ids=${ids.map(encodeURIComponent).join(",")}`;
}

/** Fraction of the current phase done, for the matra progress fill. */
export function runProgress(status: RunStatus | null | undefined): number | undefined {
  if (!status || !status.total_steps) return undefined;
  return Math.min(1, Math.max(0, status.step / status.total_steps));
}
