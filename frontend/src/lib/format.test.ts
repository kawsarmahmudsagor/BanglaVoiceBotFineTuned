import { describe, expect, it } from "vitest";
import { formatBytes, formatDuration, stateColor } from "./format";

describe("formatBytes", () => {
  it("formats across units", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(1.5 * 1024 ** 3)).toBe("1.5 GB");
    expect(formatBytes(2 * 1024 ** 4)).toBe("2.0 TB");
  });
});

describe("formatDuration", () => {
  it("formats seconds, minutes and hours", () => {
    expect(formatDuration(45)).toBe("45s");
    expect(formatDuration(0)).toBe("0s");
    expect(formatDuration(125)).toBe("2m 05s");
    expect(formatDuration(3720)).toBe("1h 02m");
  });
  it("rounds fractional seconds", () => {
    expect(formatDuration(44.6)).toBe("45s");
  });
});

describe("stateColor", () => {
  it("returns a distinct class for each run state", () => {
    const states = ["queued", "preparing", "training", "evaluating", "exporting", "done", "failed", "cancelled"] as const;
    const classes = states.map(stateColor);
    expect(new Set(classes).size).toBe(states.length);
    for (const c of classes) expect(c).toMatch(/\S/);
  });
  it("falls back for an unknown state", () => {
    expect(stateColor("weird")).toMatch(/\S/);
  });
});
