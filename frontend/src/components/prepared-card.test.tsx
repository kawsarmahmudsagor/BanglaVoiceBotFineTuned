import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { PreparedCard } from "./prepared-card";
import type { PreparedRef } from "@/lib/types";

const REF: PreparedRef = {
  role: "train",
  spec: { family: "asr-nemo", version: 1, params: {} },
  spec_hash: "abcd1234abcd1234",
  dataset_hash: "ds1",
  counts: { train: 90, val: 10 },
  skipped: [{ sample_id: "s9", split: "train", reason: "undecodable audio: RuntimeError" }],
  files: [],
  built_at: null,
  build_s: 1,
  path: "/p",
  cache_hit: true,
};

describe("PreparedCard", () => {
  it("shows cache hit, hash, counts and skipped reasons", () => {
    render(<PreparedCard refs={[REF]} />);
    expect(screen.getByText("cache hit")).toBeInTheDocument();
    expect(screen.getByText("abcd1234abcd1234")).toBeInTheDocument();
    expect(screen.getByText(/undecodable audio: RuntimeError/)).toBeInTheDocument();
    expect(screen.getByText("train: 90")).toBeInTheDocument();
  });
  it("shows built badge and empty message", () => {
    const { rerender } = render(<PreparedCard refs={[{ ...REF, cache_hit: false }]} />);
    expect(screen.getByText("built")).toBeInTheDocument();
    rerender(<PreparedCard refs={[]} />);
    expect(screen.getByText("Prepared data appears after the prepare phase.")).toBeInTheDocument();
  });
});
