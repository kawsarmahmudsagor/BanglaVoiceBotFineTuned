import { describe, expect, it } from "vitest";
import { applyRunFilter, compareHref, filterToQuery, queryToFilter, runProgress, type RunFilter } from "./run-filters";
import type { RunListItem, RunStatus } from "./types";

function run(id: string, over: Partial<RunListItem> = {}): RunListItem {
  return {
    id,
    recipe: "smoke",
    recipe_number: 1,
    backend: "fake",
    task: "llm-sft",
    state: "done",
    commercial_ok: true,
    created_at: "2026-11-20T10:00:00Z",
    started_at: null,
    finished_at: null,
    ...over,
  };
}

const RUNS = [
  run("a"),
  run("b", { state: "failed", task: "asr", recipe: "whisper", backend: "nemo", commercial_ok: false, created_at: "2026-11-21T23:59:00Z" }),
  run("c", { state: "training", created_at: "2026-11-22T00:00:00Z" }),
];
const ids = (f: RunFilter) => applyRunFilter(RUNS, f).map((r) => r.id);

describe("applyRunFilter", () => {
  it("keeps everything with an empty filter", () => expect(ids({})).toEqual(["a", "b", "c"]));
  it("filters by any of several states", () => expect(ids({ state: ["failed", "training"] })).toEqual(["b", "c"]));
  it("filters by task", () => expect(ids({ task: "asr" })).toEqual(["b"]));
  it("filters by recipe", () => expect(ids({ recipe: "whisper" })).toEqual(["b"]));
  it("filters by backend", () => expect(ids({ backend: "fake" })).toEqual(["a", "c"]));
  it("filters to commercially usable runs", () => expect(ids({ commercialOnly: true })).toEqual(["a", "c"]));
  it("treats date bounds as inclusive whole days", () => {
    expect(ids({ from: "2026-11-21", to: "2026-11-21" })).toEqual(["b"]);
    expect(ids({ from: "2026-11-21" })).toEqual(["b", "c"]);
    expect(ids({ to: "2026-11-20" })).toEqual(["a"]);
  });
});

describe("query round trip", () => {
  it("round-trips every field", () => {
    const f: RunFilter = {
      state: ["failed", "done"], task: "tts", recipe: "r", backend: "b", commercialOnly: true, from: "2026-11-01", to: "2026-11-30",
    };
    expect(queryToFilter(filterToQuery(f))).toEqual(f);
  });
  it("round-trips the empty filter", () => {
    expect(filterToQuery({}).toString()).toBe("");
    expect(queryToFilter(new URLSearchParams(""))).toEqual({});
  });
  it("joins states with commas and drops unknown ones", () => {
    expect(filterToQuery({ state: ["done", "failed"] }).get("state")).toBe("done,failed");
    expect(queryToFilter(new URLSearchParams("state=done,bogus"))).toEqual({ state: ["done"] });
  });
  it("drops an unknown task", () => {
    expect(queryToFilter(new URLSearchParams("task=video"))).toEqual({});
  });
});

describe("compareHref", () => {
  it("builds the compare URL", () => expect(compareHref(["a", "b"])).toBe("/runs/compare?ids=a,b"));
});

describe("runProgress", () => {
  const status = (step: number, total: number | null): RunStatus => ({
    phase: "train", step, total_steps: total, eta_s: null, message: "", updated_at: 0,
  });
  it("is step over total", () => expect(runProgress(status(25, 100))).toBe(0.25));
  it("is undefined without a total", () => expect(runProgress(status(25, null))).toBeUndefined());
  it("is undefined without a status", () => expect(runProgress(null)).toBeUndefined());
  it("is capped at 1", () => expect(runProgress(status(120, 100))).toBe(1));
});
