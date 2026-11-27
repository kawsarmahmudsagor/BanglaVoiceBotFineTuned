import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { vi } from "vitest";
import { Studio } from "../pages/Studio";
import * as api from "../api";
import type { DatasetSummary, StudioVideo } from "../types";
import type { JobEvents } from "../useJobEvents";

vi.mock("../api", async (orig) => {
  const real = await orig<typeof import("../api")>();
  return {
    ...real,
    listStudioVideos: vi.fn(),
    addStudioUrl: vi.fn(),
    uploadStudioFile: vi.fn(),
    startStudioVideo: vi.fn(),
    studioMediaUrl: vi.fn(() => "/media/v1"),
    createDataset: vi.fn(),
    skipVideo: vi.fn(),
    retryVideo: vi.fn(),
  };
});
vi.mock("../useJobEvents", () => ({
  useJobEvents: (): JobEvents => ({
    videos: [], connected: true, log: [], refresh: () => {},
    exportState: { status: "idle", done: 0, total: 0, summary: null, error: null }, markExportStarted: () => {},
  }),
}));

const dataset: DatasetSummary = { id: "d1", name: "D", speaker: "s", created: 0, videos: 1 };
const stages = {} as StudioVideo["stages"];
const base: StudioVideo = {
  id: "v1", title: "Video one", url: "u", source: "youtube", status: "awaiting_cut", error: null, warning: null,
  duration_s: 100, media_ready: true, cuts: null, stages, clips: { total: 0, kept: 0, flagged: 0, dropped: 0 },
};

function renderStudio() {
  return render(
    <Studio dataset={dataset} datasets={[dataset]} onSelectDataset={() => {}} onDatasetCreated={() => {}} />,
  );
}

test("awaiting_cut shows the video and posts merged ranges on Start", async () => {
  vi.mocked(api.listStudioVideos).mockResolvedValue([{ ...base, cuts: [[5, 8], [1, 3], [7, 10]] }]);
  vi.mocked(api.startStudioVideo).mockResolvedValue({ ok: true, cuts: [] });
  const { container } = renderStudio();
  await screen.findByRole("button", { name: "Start processing" });
  expect(container.querySelector("video")?.getAttribute("src")).toBe("/media/v1");
  fireEvent.click(screen.getByRole("button", { name: "Start processing" }));
  await waitFor(() => expect(api.startStudioVideo).toHaveBeenCalledWith("d1", "v1", [[1, 3], [5, 10]]));
});

test("fetching row shows a progress bar", async () => {
  vi.mocked(api.listStudioVideos).mockResolvedValue([{ ...base, status: "fetching" }]);
  renderStudio();
  expect(await screen.findByRole("progressbar")).toBeTruthy();
});

test("dropping a file uploads it", async () => {
  vi.mocked(api.listStudioVideos).mockResolvedValue([]);
  vi.mocked(api.uploadStudioFile).mockResolvedValue({ video_id: "v9" });
  renderStudio();
  const file = new File(["x"], "clip.mp4", { type: "video/mp4" });
  fireEvent.drop(await screen.findByTestId("dropzone"), { dataTransfer: { files: [file] } });
  await waitFor(() => expect(api.uploadStudioFile).toHaveBeenCalledWith("d1", file, expect.any(Function)));
});
