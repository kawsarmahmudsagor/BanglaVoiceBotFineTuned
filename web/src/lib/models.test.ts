import { describe, expect, it } from "vitest";
import { formatMetric, groupByTask, keyMetrics, laneOrder } from "./models";
import type { ModelVersion } from "./types";

const mv = (id: number, task: ModelVersion["task"], name = "m", number = 1): ModelVersion => ({
  id, name, number, run_id: "r", task, base_model: "b", metrics: {}, exports: [],
  commercial_ok: true, champion: false, created_at: null,
});

describe("groupByTask", () => {
  it("orders sections llm-sft, asr, tts, omits empty ones, newest version first within name", () => {
    const out = groupByTask([mv(1, "tts"), mv(2, "llm-sft", "b", 1), mv(3, "llm-sft", "a", 1), mv(4, "llm-sft", "a", 2)]);
    expect(out.map((g) => g.task)).toEqual(["llm-sft", "tts"]);
    expect(out[0].models.map((m) => m.id)).toEqual([4, 3, 2]);
  });
  it("returns nothing for an empty list", () => {
    expect(groupByTask([])).toEqual([]);
  });
});

describe("keyMetrics", () => {
  it("returns up to 3 entries with sorted keys", () => {
    expect(keyMetrics({ wer: 0.2, cer: 0.1, loss: 1, acc: 0.9 })).toEqual([
      ["acc", 0.9], ["cer", 0.1], ["loss", 1],
    ]);
  });
});

describe("formatMetric", () => {
  it("uses 4 significant digits", () => {
    expect(formatMetric(0.123456)).toBe("0.1235");
    expect(formatMetric(12.5)).toBe("12.5");
    expect(formatMetric(3)).toBe("3");
  });
});

describe("laneOrder", () => {
  it("puts the champion first, then newest first", () => {
    const at = (id: number, created_at: string, champion = false): ModelVersion => ({ ...mv(id, "asr"), created_at, champion });
    const out = laneOrder([at(1, "2026-11-01T00:00:00Z"), at(2, "2026-11-03T00:00:00Z"), at(3, "2026-11-02T00:00:00Z", true)]);
    expect(out.map((m) => m.id)).toEqual([3, 2, 1]);
  });
  it("does not mutate its input", () => {
    const input = [mv(1, "asr"), mv(2, "asr")];
    laneOrder(input);
    expect(input.map((m) => m.id)).toEqual([1, 2]);
  });
});
