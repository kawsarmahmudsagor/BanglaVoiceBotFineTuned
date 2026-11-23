import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { EvalSamples } from "./eval-samples";

const getEvalSamples = vi.fn();
vi.mock("@/lib/api", () => ({
  getEvalSamples: (...a: unknown[]) => getEvalSamples(...a),
  evalAudioUrl: (id: string, n: string) => `http://x/api/runs/${id}/eval/audio/${n}`,
}));

beforeEach(() => getEvalSamples.mockReset());

describe("EvalSamples", () => {
  it("renders audio elements for tts", async () => {
    getEvalSamples.mockResolvedValue({ items: [{ sample_id: "a", text: "হ্যালো", audio: "a.wav" }], total: 1 });
    const { container } = render(<EvalSamples runId="r1" task="tts" />);
    await waitFor(() => expect(container.querySelector("audio")).not.toBeNull());
    expect(container.querySelector("audio")!.getAttribute("src")).toBe("http://x/api/runs/r1/eval/audio/a.wav");
    expect(screen.getByText("হ্যালো")).toBeInTheDocument();
  });
  it("renders the diff for asr", async () => {
    getEvalSamples.mockResolvedValue({
      items: [{ sample_id: "a", reference: "আমি ভাত খাই", hypothesis: "আমি খাই", wer: 0.33 }],
      total: 1,
    });
    const { container } = render(<EvalSamples runId="r1" task="asr" />);
    await waitFor(() => expect(container.querySelector("del")).not.toBeNull());
    expect(container.querySelector("del")!.textContent).toBe("ভাত");
  });
  it("shows Load more while total exceeds loaded", async () => {
    getEvalSamples.mockResolvedValue({ items: [{ sample_id: "a", prompt: [{ role: "user", content: "প্রশ্ন" }], reference: "r", reply: "উত্তর" }], total: 3 });
    render(<EvalSamples runId="r1" task="llm-sft" />);
    await waitFor(() => expect(screen.getByRole("button", { name: "Load more" })).toBeInTheDocument());
    expect(screen.getByText("উত্তর")).toBeInTheDocument();
  });
});
