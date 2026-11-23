import { describe, expect, it } from "vitest";
import { wordDiff } from "./word-diff";

describe("wordDiff", () => {
  it("marks a deleted word", () => {
    expect(wordDiff("আমি ভাত খাই", "আমি খাই")).toEqual([
      { op: "eq", ref: "আমি", hyp: "আমি" },
      { op: "del", ref: "ভাত" },
      { op: "eq", ref: "খাই", hyp: "খাই" },
    ]);
  });
  it("marks substitutions and insertions", () => {
    expect(wordDiff("আমি ভাত", "আমি রুটি")[1]).toEqual({ op: "sub", ref: "ভাত", hyp: "রুটি" });
    expect(wordDiff("আমি", "আমি ভাত")[1]).toEqual({ op: "ins", hyp: "ভাত" });
  });
});
