import { describe, expect, it } from "vitest";
import { bestRun, diffHparams, flatten, mergeMetrics, metricDirection, parseCompareIds } from "./compare";
import type { RunMetric } from "./types";

describe("parseCompareIds", () => {
  it("accepts 2 to 4 ids", () => {
    expect(parseCompareIds("a,b")).toEqual({ ids: ["a", "b"] });
    expect(parseCompareIds("a,b,c,d")).toEqual({ ids: ["a", "b", "c", "d"] });
  });
  it("rejects fewer than 2", () => {
    expect(parseCompareIds(null)).toEqual({ error: "Pick at least 2 runs to compare." });
    expect(parseCompareIds("a")).toEqual({ error: "Pick at least 2 runs to compare." });
  });
  it("rejects more than 4", () => {
    expect(parseCompareIds("a,b,c,d,e")).toEqual({ error: "Compare up to 4 runs at a time." });
  });
  it("removes duplicates and blanks before counting", () => {
    expect(parseCompareIds("a,a,b,,")).toEqual({ ids: ["a", "b"] });
    expect(parseCompareIds("a,a")).toEqual({ error: "Pick at least 2 runs to compare." });
  });
  it("names an unknown id", () => {
    expect(parseCompareIds("a,z", new Set(["a", "b"]))).toEqual({ error: "Unknown run: z" });
  });
});

describe("flatten", () => {
  it("dot-joins nested keys", () => {
    expect(flatten({ a: 1, b: { c: 2, d: { e: 3 } } })).toEqual({ a: 1, "b.c": 2, "b.d.e": 3 });
  });
  it("keeps arrays as values", () => {
    expect(flatten({ x: [1, 2] })).toEqual({ x: [1, 2] });
  });
});

describe("diffHparams", () => {
  const spec = (hparams: Record<string, unknown>, extra: Record<string, unknown> = {}) => ({
    base_model: "m",
    method: { type: "lora", r: 16 },
    datasets: { train: "d@v1", eval: null },
    hparams,
    ...extra,
  });

  it("flattens nested hparams and marks differing keys first", () => {
    const rows = diffHparams({
      r1: spec({ lr: 1, sched: { warmup: 10 } }),
      r2: spec({ lr: 2, sched: { warmup: 10 } }),
    });
    expect(rows[0]).toEqual({ key: "hparams.lr", values: { r1: 1, r2: 2 }, differs: true });
    const same = rows.find((r) => r.key === "hparams.sched.warmup")!;
    expect(same.differs).toBe(false);
    expect(rows.find((r) => r.key === "method.r")?.differs).toBe(false);
    expect(rows.find((r) => r.key === "base_model")?.differs).toBe(false);
    expect(rows.find((r) => r.key === "datasets.train")?.differs).toBe(false);
  });

  it("treats a key missing on one side as differing", () => {
    const rows = diffHparams({ r1: spec({ lr: 1, epochs: 2 }), r2: spec({ lr: 1 }) });
    expect(rows[0]).toEqual({ key: "hparams.epochs", values: { r1: 2, r2: undefined }, differs: true });
  });

  it("sorts differing rows first, then by key", () => {
    const rows = diffHparams({ r1: spec({ b: 1, a: 1, z: 1 }), r2: spec({ b: 2, a: 1, z: 2 }) });
    expect(rows.filter((r) => r.differs).map((r) => r.key)).toEqual(["hparams.b", "hparams.z"]);
    expect(rows.findIndex((r) => !r.differs)).toBe(2);
  });

  it("ignores a spec that failed to load", () => {
    const rows = diffHparams({ r1: spec({ lr: 1 }), r2: null, r3: spec({ lr: 1 }) });
    expect(rows.find((r) => r.key === "hparams.lr")).toEqual({ key: "hparams.lr", values: { r1: 1, r3: 1 }, differs: false });
  });
});

describe("mergeMetrics", () => {
  const m = (step: number, name: string, value: number, split = "train"): RunMetric => ({ step, split, name, value, ts: 0 });

  it("builds one series per split/name with each run as a column", () => {
    const merged = mergeMetrics({
      r1: [m(1, "loss", 2), m(2, "loss", 1)],
      r2: [m(2, "loss", 1.5), m(3, "loss", 1.2)],
    });
    expect(merged["train/loss"]).toEqual([
      { step: 1, r1: 2 },
      { step: 2, r1: 1, r2: 1.5 },
      { step: 3, r2: 1.2 },
    ]);
  });
  it("keeps metrics only one run has", () => {
    const merged = mergeMetrics({ r1: [m(1, "wer", 0.3, "eval")], r2: [m(1, "loss", 2)] });
    expect(Object.keys(merged).sort()).toEqual(["eval/wer", "train/loss"]);
    expect(merged["eval/wer"]).toEqual([{ step: 1, r1: 0.3 }]);
  });
  it("handles a run with zero metrics", () => {
    expect(mergeMetrics({ r1: [], r2: [] })).toEqual({});
  });
});

describe("metricDirection", () => {
  it.each(["loss", "eval_loss", "WER", "cer", "error_rate", "rtf"])("%s is lower-is-better", (n) => {
    expect(metricDirection(n)).toBe("min");
  });
  it.each(["accuracy", "judge_score", "mos"])("%s is higher-is-better", (n) => {
    expect(metricDirection(n)).toBe("max");
  });
});

describe("bestRun", () => {
  it("picks the minimum for min metrics", () => expect(bestRun({ a: 0.3, b: 0.1 }, "min")).toBe("b"));
  it("picks the maximum for max metrics", () => expect(bestRun({ a: 0.3, b: 0.1 }, "max")).toBe("a"));
  it("breaks ties by input order", () => expect(bestRun({ a: 1, b: 1 }, "min")).toBe("a"));
  it("skips missing values", () => expect(bestRun({ a: undefined, b: 2 }, "max")).toBe("b"));
  it("is null when nothing has a value", () => expect(bestRun({ a: undefined }, "max")).toBeNull());
});
