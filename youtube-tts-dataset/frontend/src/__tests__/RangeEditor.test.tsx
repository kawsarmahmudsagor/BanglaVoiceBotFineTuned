import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { vi } from "vitest";
import { RangeEditor, mergeRanges } from "../components/RangeEditor";

test("mergeRanges sorts and merges overlaps", () => {
  expect(mergeRanges([[5, 8], [1, 3], [7, 10]])).toEqual([[1, 3], [5, 10]]);
});

function Harness({ playhead, onRanges }: { playhead: { v: number }; onRanges?: (r: [number, number][]) => void }) {
  const [ranges, setRanges] = useState<[number, number][]>([]);
  return (
    <RangeEditor
      duration={100}
      playhead={playhead.v}
      ranges={ranges}
      onChange={(r) => {
        setRanges(r);
        onRanges?.(r);
      }}
      onSeek={() => {}}
    />
  );
}

test("mark in at 2 then mark out at 1 swaps into [1, 2]", () => {
  const seen: [number, number][][] = [];
  const { rerender } = render(<Harness playhead={{ v: 2 }} onRanges={(r) => seen.push(r)} />);
  fireEvent.click(screen.getByRole("button", { name: "Mark in" }));
  rerender(<Harness playhead={{ v: 1 }} onRanges={(r) => seen.push(r)} />);
  fireEvent.click(screen.getByRole("button", { name: "Mark out" }));
  expect(seen.at(-1)).toEqual([[1, 2]]);
});

test("delete removes a range", () => {
  const onChange = vi.fn();
  render(
    <RangeEditor duration={100} playhead={0} ranges={[[1, 2], [5, 6]]} onChange={onChange} onSeek={() => {}} />,
  );
  fireEvent.click(screen.getAllByRole("button", { name: "Delete" })[0]);
  expect(onChange).toHaveBeenCalledWith([[5, 6]]);
});

test("typed end beyond the duration clamps", () => {
  const onChange = vi.fn();
  render(<RangeEditor duration={100} playhead={0} ranges={[[1, 2]]} onChange={onChange} onSeek={() => {}} />);
  fireEvent.change(screen.getByLabelText("End of range 1"), { target: { value: "250" } });
  expect(onChange).toHaveBeenCalledWith([[1, 100]]);
});

test("Play seeks to the range start and total shows merged kept time", () => {
  const onSeek = vi.fn();
  render(<RangeEditor duration={125} playhead={0} ranges={[[10, 30], [20, 40]]} onChange={() => {}} onSeek={onSeek} />);
  fireEvent.click(screen.getAllByRole("button", { name: "Play" })[0]);
  expect(onSeek).toHaveBeenCalledWith(10);
  expect(screen.getByText("Kept: 0:30 of 2:05")).toBeTruthy();
});
