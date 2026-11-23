import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { HparamDiff } from "./hparam-diff";

const ROWS = [
  { key: "hparams.lr", values: { r1: 1, r2: 2 }, differs: true },
  { key: "hparams.epochs", values: { r1: 3, r2: undefined }, differs: true },
  { key: "base_model", values: { r1: "m", r2: "m" }, differs: false },
];

describe("HparamDiff", () => {
  it("marks differing rows with text and lists them first", () => {
    const { container } = render(<HparamDiff rows={ROWS} runIds={["r1", "r2"]} />);
    const marked = container.querySelectorAll('[data-differs="true"]');
    expect(marked).toHaveLength(2);
    expect(marked[0].textContent).toContain("hparams.lr");
    expect(screen.getAllByText("Differs")).toHaveLength(2);
  });

  it("puts identical rows in a collapsed details group", () => {
    const { container } = render(<HparamDiff rows={ROWS} runIds={["r1", "r2"]} />);
    const details = container.querySelector("details")!;
    expect(details.open).toBe(false);
    expect(details.querySelector("summary")?.textContent).toBe("Same in all runs (1)");
    expect(details.textContent).toContain("base_model");
  });

  it("shows a missing value as not set", () => {
    render(<HparamDiff rows={ROWS} runIds={["r1", "r2"]} />);
    expect(screen.getByText("not set")).toBeInTheDocument();
  });
});
