import { describe, expect, it } from "vitest";
import { DEFAULT_SPLIT, fromPct, previewCounts, ratiosValid, toPct } from "./splits";
import type { SplitConfig } from "./types";

const R = DEFAULT_SPLIT.ratios;
const off: SplitConfig = { use_dataset_splits: false, ratios: R };

describe("previewCounts", () => {
  it("splits everything by ratios when nothing is given", () => {
    expect(previewCounts({}, 1000, DEFAULT_SPLIT)).toEqual({ train: 900, val: 50, test: 50 });
  });
  it("keeps given splits and splits the rest (train + test given)", () => {
    // given train 950 -> train:val 90:5 ; test as given
    const c = previewCounts({ train: 950, test: 40 }, 990, DEFAULT_SPLIT);
    expect(c.test).toBe(40);
    expect(c.train + c.val).toBe(950);
    expect(c.val).toBe(Math.round((950 * 5) / 95));
  });
  it("splits unassigned rows when only test is given", () => {
    const c = previewCounts({ test: 100, unassigned: 950 }, 1050, DEFAULT_SPLIT);
    expect(c).toEqual({ train: 900, val: 50, test: 100 });
  });
  it("uses given counts as-is when all are given", () => {
    expect(previewCounts({ train: 80, val: 10, test: 10 }, 100, DEFAULT_SPLIT)).toEqual({ train: 80, val: 10, test: 10 });
  });
  it("ignores given counts when use_dataset_splits is false", () => {
    expect(previewCounts({ train: 80, val: 10, test: 10 }, 1000, off)).toEqual({ train: 900, val: 50, test: 50 });
  });
});

describe("ratiosValid", () => {
  it("accepts the defaults", () => {
    expect(ratiosValid(R)).toBeNull();
  });
  it("rejects a sum other than 100%", () => {
    expect(ratiosValid({ train: 0.6, val: 0.2, test: 0.1 })).toBe("Ratios must add up to 100%");
  });
  it("rejects a train ratio of zero", () => {
    expect(ratiosValid({ train: 0, val: 0.5, test: 0.5 })).toBe("Train must be above 0%");
  });
});

describe("percent helpers", () => {
  it("round-trips without float noise", () => {
    expect(toPct(0.05)).toBe(5);
    expect(toPct(0.9)).toBe(90);
    expect(fromPct(5)).toBe(0.05);
  });
});

describe("ratiosValid non-finite", () => {
  it("rejects NaN ratios", () => {
    expect(ratiosValid({ train: 0.9, val: NaN, test: 0.05 })).toBe("Ratios must add up to 100%");
    expect(ratiosValid({ train: NaN, val: 0.05, test: 0.05 })).toBe("Ratios must add up to 100%");
  });
});
