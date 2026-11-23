import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SampleTable } from "./sample-table";

const ROWS = [
  { sample_id: "s1", split: "train", file: "a.wav", text: "আমি ভাত খাই", duration_s: 2.5 },
];

describe("SampleTable", () => {
  it("renders an audio row with a player and Bangla text", () => {
    const { container } = render(
      <SampleTable task="asr" dataset="ds" version={2} rows={ROWS} onEdit={vi.fn()} />,
    );
    const audio = container.querySelector("audio");
    expect(audio).not.toBeNull();
    expect(audio!.getAttribute("src")).toContain("/api/datasets/ds/versions/2/audio/s1");
    expect(audio!.getAttribute("preload")).toBe("none");
    expect(screen.getByText("আমি ভাত খাই")).toBeInTheDocument();
    expect(screen.getByText("a.wav")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Edit" })).toBeInTheDocument();
  });

  it("renders llm-sft messages compactly with no audio or edit", () => {
    const rows = [
      { sample_id: "m1", split: "val", messages_json: JSON.stringify([{ role: "user", content: "হ্যালো" }]) },
    ];
    const { container } = render(<SampleTable task="llm-sft" dataset="ds" version={1} rows={rows} />);
    expect(container.querySelector("audio")).toBeNull();
    expect(screen.getByText(/হ্যালো/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Edit" })).toBeNull();
  });
});
