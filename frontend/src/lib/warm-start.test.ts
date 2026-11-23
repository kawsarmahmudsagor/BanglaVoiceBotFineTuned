import { describe, expect, it } from "vitest";
import { initFromCheckpoint } from "./warm-start";

describe("initFromCheckpoint", () => {
  it("keeps final as final", () => {
    expect(initFromCheckpoint("r-1", "final")).toEqual({ run: "r-1", checkpoint: "final" });
  });
  it("turns checkpoint-N into the int step", () => {
    expect(initFromCheckpoint("r-1", "checkpoint-300")).toEqual({ run: "r-1", checkpoint: 300 });
  });
  it("falls back to latest for an unparseable name", () => {
    expect(initFromCheckpoint("r-1", "weird")).toEqual({ run: "r-1", checkpoint: "latest" });
  });
});