import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import RunDetailPage from "./page";

const api = vi.hoisted(() => ({
  getRun: vi.fn(), getLogs: vi.fn(), getMetrics: vi.fn(), getCheckpoints: vi.fn(),
  getPrepared: vi.fn(), getRecipeVersion: vi.fn(), cancelRun: vi.fn(), resumeRun: vi.fn(), promoteRun: vi.fn(),
  runEventsUrl: vi.fn(() => ""), runExportFileUrl: vi.fn(() => ""),
  ApiError: class extends Error {},
}));
vi.mock("@/lib/api", () => api);
vi.mock("next/navigation", () => ({ useParams: () => ({ id: "r-2" }), useRouter: () => ({ push: vi.fn() }) }));

const RUN = {
  id: "r-2", state: "done", attempt: 1, resume: false, cancel_requested: false, run_dir: "/x", commercial_ok: true,
  error: null, oom: false, git_sha: null, lock_hash: null, dataset_hashes: {}, mlflow_run_id: null, created_at: null,
  started_at: null, finished_at: null, recipe: { name: "cfg", number: 1 }, status: null, eval: null, exports: [], oom_hint: null,
  parent_run_id: "r-1", parent_checkpoint: "checkpoint-300",
};

beforeEach(() => {
  Object.values(api).forEach((f) => typeof f === "function" && "mockReset" in f && (f as ReturnType<typeof vi.fn>).mockReset());
  api.runEventsUrl.mockReturnValue("");
  api.getRun.mockResolvedValue(RUN);
  api.getLogs.mockResolvedValue({ stdout: [], stderr: [] });
  api.getMetrics.mockResolvedValue({ items: [], next_offset: 0 });
  api.getCheckpoints.mockResolvedValue({ items: [{ name: "final", step: 5, complete: true, init_meta: null }] });
});

describe("run detail warm start", () => {
  it("links to the parent run in the header", async () => {
    render(<RunDetailPage />);
    const link = await screen.findByRole("link", { name: "r-1" });
    expect(link).toHaveAttribute("href", "/runs/r-1");
    expect(screen.getByText("checkpoint-300")).toBeInTheDocument();
  });

  it("omits the parent line for a run without a parent", async () => {
    api.getRun.mockResolvedValue({ ...RUN, parent_run_id: null, parent_checkpoint: null });
    render(<RunDetailPage />);
    await screen.findByText("cfg");
    expect(screen.queryByText(/Started from/)).toBeNull();
  });

  it("shows the checkpoints tab with the action button", async () => {
    render(<RunDetailPage />);
    fireEvent.click(await screen.findByRole("tab", { name: "Checkpoints" }));
    expect(await screen.findByRole("button", { name: "New run from this checkpoint" })).toBeInTheDocument();
  });
});