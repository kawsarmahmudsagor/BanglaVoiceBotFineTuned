import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SplitSummary } from "./split-summary";

describe("SplitSummary", () => {
  it("lists splits provided by the dataset", () => {
    render(<SplitSummary splitSource="dataset" given={{ train: 80, val: 10, test: 5, unassigned: 5 }} />);
    expect(screen.getByText("Splits provided by dataset: train 80 · valid 10 · test 5 (unassigned 5)")).toBeInTheDocument();
  });
  it("explains per-run assignment when there are none", () => {
    render(<SplitSummary splitSource="none" given={{}} />);
    expect(screen.getByText("No splits in dataset — assigned per run (default 90/5/5)")).toBeInTheDocument();
  });
});
