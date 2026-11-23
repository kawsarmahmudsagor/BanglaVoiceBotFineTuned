import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { SplitFields } from "./split-fields";
import { DEFAULT_SPLIT } from "@/lib/splits";
import type { SplitConfig } from "@/lib/types";

const GIVEN = { split_source: "dataset" as const, given_splits: { train: 80, val: 10, test: 10, unassigned: 0 }, n_samples: 100 };
const NONE = { split_source: "none" as const, given_splits: {}, n_samples: 1000 };

describe("SplitFields", () => {
  it("hides the toggle when the dataset has no splits", () => {
    render(<SplitFields value={{ ...DEFAULT_SPLIT, use_dataset_splits: false }} onChange={vi.fn()} dataset={NONE} />);
    expect(screen.queryByLabelText(/Use the dataset's splits/)).toBeNull();
    expect(screen.getByLabelText("Train %")).toBeEnabled();
    expect(screen.getByLabelText("Train %")).toHaveValue(90);
    expect(screen.getByLabelText("Valid %")).toHaveValue(5);
    expect(screen.getByLabelText("Test %")).toHaveValue(5);
  });

  it("shows the toggle with counts and disables inputs while on", () => {
    render(<SplitFields value={DEFAULT_SPLIT} onChange={vi.fn()} dataset={GIVEN} />);
    expect(screen.getByLabelText(/Use the dataset's splits/)).toBeChecked();
    expect(screen.getByText("(train 80 · valid 10 · test 10)")).toBeInTheDocument();
    expect(screen.getByLabelText("Train %")).toBeDisabled();
    expect(screen.getByLabelText("Valid %")).toBeDisabled();
    expect(screen.getByLabelText("Test %")).toBeDisabled();
    expect(screen.getByText("≈ train 80 · valid 10 · test 10")).toBeInTheDocument();
  });

  it("shows the live preview and emits fractions on edit", () => {
    const onChange = vi.fn();
    render(<SplitFields value={{ ...DEFAULT_SPLIT, use_dataset_splits: false }} onChange={onChange} dataset={NONE} />);
    expect(screen.getByText("≈ train 900 · valid 50 · test 50")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Valid %"), { target: { value: "10" } });
    expect(onChange).toHaveBeenCalledWith({ use_dataset_splits: false, ratios: { train: 0.9, val: 0.1, test: 0.05 } });
  });

  it("shows an inline error at 60/20/10", () => {
    const value = { use_dataset_splits: false, ratios: { train: 0.6, val: 0.2, test: 0.1 } };
    render(<SplitFields value={value} onChange={vi.fn()} dataset={NONE} />);
    expect(screen.getByText("Ratios must add up to 100%")).toBeInTheDocument();
  });

  it("toggling off emits use_dataset_splits false", () => {
    const onChange = vi.fn();
    render(<SplitFields value={DEFAULT_SPLIT} onChange={onChange} dataset={GIVEN} />);
    fireEvent.click(screen.getByLabelText(/Use the dataset's splits/));
    expect(onChange).toHaveBeenCalledWith({ ...DEFAULT_SPLIT, use_dataset_splits: false });
  });
});

describe("SplitFields editing", () => {
  it("allows clearing an input while typing and reports it as invalid", () => {
    function Host() {
      const [v, setV] = useState<SplitConfig>({ ...DEFAULT_SPLIT, use_dataset_splits: false });
      return <SplitFields value={v} onChange={setV} dataset={NONE} />;
    }
    render(<Host />);
    fireEvent.change(screen.getByLabelText("Valid %"), { target: { value: "" } });
    expect(screen.getByLabelText("Valid %")).toHaveValue(null);
    expect(screen.getByText("Ratios must add up to 100%")).toBeInTheDocument();
  });

  it("unlocks the inputs when the ratios are invalid even with the toggle on", () => {
    const value = { use_dataset_splits: true, ratios: { train: 0.6, val: 0.2, test: 0.1 } };
    render(<SplitFields value={value} onChange={vi.fn()} dataset={GIVEN} />);
    expect(screen.getByText("Ratios must add up to 100%")).toBeInTheDocument();
    expect(screen.getByLabelText("Train %")).toBeEnabled();
  });

  it("keeps inputs locked and shows the load error", () => {
    render(<SplitFields value={DEFAULT_SPLIT} onChange={vi.fn()} dataset={null} loadError="boom" />);
    expect(screen.getByText(/Cannot load the train dataset's splits \(boom\)/)).toBeInTheDocument();
    expect(screen.getByLabelText("Train %")).toBeDisabled();
  });
});
