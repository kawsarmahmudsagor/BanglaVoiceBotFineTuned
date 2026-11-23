import type { CheckpointItem, InitFrom } from "./types";

/** A checkpoint directory name as the recipe's `checkpoint` field wants it: "final" or the int step. */
export function checkpointRef(name: string, step?: number): "final" | "latest" | number {
  if (name === "final") return "final";
  if (step !== undefined) return step;
  const m = /(\d+)$/.exec(name);
  return m ? Number(m[1]) : "latest";
}

export function initFromCheckpoint(runId: string, name: string, step?: number): InitFrom {
  return { run: runId, checkpoint: checkpointRef(name, step) };
}

export const initFromItem = (runId: string, c: CheckpointItem): InitFrom => initFromCheckpoint(runId, c.name, c.step);