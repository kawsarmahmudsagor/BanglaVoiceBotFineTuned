import { describe, expect, it } from "vitest";
import { LOG_LIMIT, appendLog, mergeMetrics, metricsToSeries, phaseStates } from "./metrics";
import type { RunMetric } from "./types";

const m = (step: number, value: number, name = "loss", split = "train", ts = step): RunMetric => ({
  step, split, name, value, ts,
});

describe("metricsToSeries", () => {
  it("groups by split/name and sorts by step", () => {
    const out = metricsToSeries([m(2, 0.5), m(1, 0.9), m(1, 3, "wer", "eval")]);
    expect(Object.keys(out).sort()).toEqual(["eval/wer", "train/loss"]);
    expect(out["train/loss"]).toEqual([{ step: 1, value: 0.9 }, { step: 2, value: 0.5 }]);
  });

  it("last value wins for duplicate steps (resumed run)", () => {
    const out = metricsToSeries([m(1, 0.9, "loss", "train", 10), m(2, 0.8, "loss", "train", 11), m(2, 0.7, "loss", "train", 99)]);
    expect(out["train/loss"]).toEqual([{ step: 1, value: 0.9 }, { step: 2, value: 0.7 }]);
  });

  it("returns an empty object for no metrics", () => {
    expect(metricsToSeries([])).toEqual({});
  });
});

describe("mergeMetrics", () => {
  it("drops exact replays (same split, name, step, ts) after an SSE reconnect", () => {
    const a = [m(1, 1), m(2, 2)];
    const merged = mergeMetrics(a, [m(1, 1), m(2, 2), m(3, 3)]);
    expect(merged.map((x) => x.step)).toEqual([1, 2, 3]);
  });

  it("keeps a resumed re-log of the same step (different ts)", () => {
    const merged = mergeMetrics([m(2, 2, "loss", "train", 5)], [m(2, 1.5, "loss", "train", 50)]);
    expect(merged).toHaveLength(2);
  });
});

describe("appendLog", () => {
  it("keeps only the last LOG_LIMIT lines", () => {
    const lines = Array.from({ length: LOG_LIMIT }, (_, i) => `l${i}`);
    const out = appendLog(lines, "new");
    expect(out).toHaveLength(LOG_LIMIT);
    expect(out[0]).toBe("l1");
    expect(out.at(-1)).toBe("new");
  });
});

describe("phaseStates", () => {
  it("marks earlier phases done, current active, later pending", () => {
    expect(phaseStates("eval", "evaluating")).toEqual(["done", "done", "active", "pending"]);
  });
  it("falls back to the run state when status is null", () => {
    expect(phaseStates(null, "training")).toEqual(["done", "active", "pending", "pending"]);
    expect(phaseStates(null, "queued")).toEqual(["pending", "pending", "pending", "pending"]);
  });
  it("marks everything done for done runs", () => {
    expect(phaseStates("export", "done")).toEqual(["done", "done", "done", "done"]);
  });
  it("marks the current phase failed for failed/cancelled runs", () => {
    expect(phaseStates("train", "failed")).toEqual(["done", "failed", "pending", "pending"]);
    expect(phaseStates(null, "cancelled")).toEqual(["pending", "pending", "pending", "pending"]);
  });
});
