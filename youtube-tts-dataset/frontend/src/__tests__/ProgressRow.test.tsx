import { render, screen } from "@testing-library/react";
import { ProgressRow } from "../components/ProgressRow";
import { applyEvent, toVideoState } from "../useJobEvents";
import { STAGES, type VideoDetail, type VideoState } from "../types";

function detail(over: Partial<VideoDetail> = {}): VideoDetail {
  return {
    dataset_id: "ds1",
    id: "abc123",
    url: "https://www.youtube.com/watch?v=abc123",
    title: "বাংলা খবর",
    channel: null,
    upload_date: null,
    duration_s: 754,
    license: null,
    status: "queued",
    error: null,
    warning: null,
    stages: Object.fromEntries(STAGES.map((s) => [s, false])) as VideoDetail["stages"],
    kept: 0,
    dropped: 0,
    ...over,
  };
}

function renderRow(video: VideoState) {
  return render(
    <table>
      <tbody>
        <ProgressRow video={video} onRetry={() => {}} onSkip={() => {}} onDelete={() => {}} />
      </tbody>
    </table>,
  );
}

test("ProgressRow shows download percent from progress event", () => {
  const [video] = applyEvent([toVideoState(detail())], {
    type: "progress",
    dataset_id: "ds1",
    video_id: "abc123",
    stage: "download",
    done: 42,
    total: 100,
    eta_s: 30,
  });
  renderRow(video);
  expect(screen.getByRole("progressbar", { name: /download/i })).toHaveAttribute("aria-valuenow", "42");
  expect(screen.getAllByText("42%").length).toBeGreaterThan(0);
  expect(screen.getByText("বাংলা খবর")).toBeInTheDocument();
});

test("ProgressRow shows retry only when failed", () => {
  const { unmount } = renderRow(toVideoState(detail({ status: "running" })));
  expect(screen.queryByRole("button", { name: /retry/i })).toBeNull();
  unmount();
  renderRow(toVideoState(detail({ status: "failed", error: "HTTP 403" })));
  expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();
  expect(screen.getByText(/HTTP 403/)).toBeInTheDocument();
});
