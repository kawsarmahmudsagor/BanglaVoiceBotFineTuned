import { describe, expect, it } from "vitest";
import { pushSample } from "./gpu-history";

describe("pushSample", () => {
  it("appends and caps at max, dropping the oldest", () => {
    expect(pushSample([1, 2, 3], 4, 3)).toEqual([2, 3, 4]);
    expect(pushSample([], 5)).toEqual([5]);
  });
  it("keeps 60 samples by default", () => {
    const h = Array.from({ length: 60 }, (_, i) => i);
    expect(pushSample(h, 99)).toHaveLength(60);
  });
  it("does not mutate its input", () => {
    const h = [1, 2];
    pushSample(h, 3, 2);
    expect(h).toEqual([1, 2]);
  });
});
