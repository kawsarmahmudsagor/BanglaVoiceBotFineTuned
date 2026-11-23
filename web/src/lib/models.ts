import type { ModelVersion, Task } from "./types";

export const TASK_ORDER: Task[] = ["llm-sft", "asr", "tts"];

export function groupByTask(models: ModelVersion[]): { task: Task; models: ModelVersion[] }[] {
  return TASK_ORDER.map((task) => ({
    task,
    models: models
      .filter((m) => m.task === task)
      .sort((a, b) => a.name.localeCompare(b.name) || b.number - a.number),
  })).filter((g) => g.models.length > 0);
}

export function keyMetrics(metrics: Record<string, number>, max = 3): [string, number][] {
  return Object.entries(metrics)
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .slice(0, max);
}

export function formatMetric(v: number): string {
  return String(Number(v.toPrecision(4)));
}
