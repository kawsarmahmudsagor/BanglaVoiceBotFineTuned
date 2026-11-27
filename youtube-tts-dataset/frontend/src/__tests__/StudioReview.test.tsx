import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { vi } from "vitest";
import { StudioReview } from "../components/StudioReview";
import * as api from "../api";
import type { StudioClip } from "../types";

vi.mock("../components/Waveform", () => ({ Waveform: () => <div data-testid="waveform" /> }));
vi.mock("../api", async (orig) => {
  const real = await orig<typeof import("../api")>();
  return { ...real, getVideoClips: vi.fn(), patchClip: vi.fn(), exportDataset: vi.fn() };
});
vi.mock("../useJobEvents", () => ({
  useJobEvents: () => ({
    videos: [], connected: true, log: [], refresh: () => {},
    exportState: { status: "idle", done: 0, total: 0, summary: null, error: null }, markExportStarted: () => {},
  }),
}));

function mk(i: number, o: Partial<StudioClip>): StudioClip {
  return {
    dataset_id: "d1", id: `v1_${i}`, video_id: "v1", idx: i, start_s: 0, end_s: 16, dnsmos: 3, speaker_sim: 0.8,
    drop_reason: null, text_a: `a${i}`, text_b: `b${i}`, cer: null, text_en: null, text_translated: null,
    text_final: `final${i}`, text_source: "stt", status: "auto", no_punct: 0, flag: null, text_pick: "a",
    score_a: 0.9, score_b: 0.8, reason: "auto", orig_start_s: 65.25, orig_end_s: 81, ...o,
  };
}
const clips = [
  mk(0, {}),
  mk(1, { status: "flagged", flag: "disagree", reason: "disagree" }),
  mk(2, { status: "dropped", drop_reason: "bad_stt", reason: null }),
  mk(3, { status: "approved" }),
  mk(4, { status: "rejected" }),
  mk(5, { status: "flagged", flag: "incomplete_sentence", reason: "auto" }),
];

beforeEach(() => {
  vi.mocked(api.getVideoClips).mockResolvedValue(clips);
});

test("filters count clips and default to Needs attention", async () => {
  render(<StudioReview datasetId="d1" videoId="v1" />);
  await screen.findByRole("button", { name: "Needs attention (2)" });
  expect(screen.getByRole("button", { name: "All (6)" })).toBeTruthy();
  expect(screen.getByRole("button", { name: "Approved (1)" })).toBeTruthy();
  expect(screen.getByRole("button", { name: "Rejected (1)" })).toBeTruthy();
  expect(screen.getByRole("button", { name: "Dropped (1)" })).toBeTruthy();
  expect(screen.getAllByText("Models disagree").length).toBeGreaterThan(0);
  expect(screen.getByText("Incomplete sentence")).toBeTruthy();
  expect(screen.getAllByText("1:05.2–1:21.0").length).toBeGreaterThan(0);
  fireEvent.click(screen.getByRole("button", { name: "Dropped (1)" }));
  expect(screen.getByText("Broken transcript")).toBeTruthy();
});

test("Use model A calls patchClip and updates the text box", async () => {
  vi.mocked(api.patchClip).mockResolvedValue({ ...clips[1], text_pick: "a", text_final: "swapped A" } as never);
  render(<StudioReview datasetId="d1" videoId="v1" />);
  await screen.findByRole("button", { name: "Use model A" });
  fireEvent.click(screen.getByRole("button", { name: "Use model A" }));
  await waitFor(() => expect(api.patchClip).toHaveBeenCalledWith("d1", "v1_1", { text_pick: "a" }));
  await waitFor(() => expect((screen.getByRole("textbox") as HTMLTextAreaElement).value).toBe("swapped A"));
  expect(screen.getByText("Punctuation is not re-applied to the swapped text.")).toBeTruthy();
});

test("Export calls exportDataset", async () => {
  vi.mocked(api.exportDataset).mockResolvedValue({ ok: true } as never);
  render(<StudioReview datasetId="d1" videoId="v1" />);
  await screen.findByRole("button", { name: "Export dataset" });
  fireEvent.click(screen.getByRole("button", { name: "Export dataset" }));
  await waitFor(() => expect(api.exportDataset).toHaveBeenCalledWith("d1", {}));
});

test("an edit survives navigating away and back, and Approve sends it", async () => {
  vi.mocked(api.patchClip).mockImplementation(async (_d, id, patch) => {
    const c = clips.find((x) => x.id === id)!;
    return { ...c, ...patch } as never;
  });
  render(<StudioReview datasetId="d1" videoId="v1" />);
  const box = (await screen.findByRole("textbox")) as HTMLTextAreaElement;
  fireEvent.change(box, { target: { value: "edited text" } });
  fireEvent.click(screen.getByRole("button", { name: /Next/ }));
  await waitFor(() => expect(api.patchClip).toHaveBeenCalledWith("d1", "v1_1", { text_final: "edited text" }));
  await waitFor(() => expect((screen.getByRole("textbox") as HTMLTextAreaElement).value).toBe("final5"));
  fireEvent.click(screen.getByRole("button", { name: /Previous/ }));
  await waitFor(() => expect((screen.getByRole("textbox") as HTMLTextAreaElement).value).toBe("edited text"));
  fireEvent.click(screen.getByRole("button", { name: /^Approve/ }));
  await waitFor(() =>
    expect(api.patchClip).toHaveBeenCalledWith("d1", "v1_1", { text_final: "edited text", status: "approved" }),
  );
});
