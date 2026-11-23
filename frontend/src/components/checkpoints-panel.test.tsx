import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { CheckpointsPanel } from "./checkpoints-panel";

const api = vi.hoisted(() => ({ getCheckpoints: vi.fn() }));
vi.mock("@/lib/api", () => api);
const push = vi.hoisted(() => vi.fn());
vi.mock("next/navigation", () => ({ useRouter: () => ({ push }) }));

beforeEach(() => {
  api.getCheckpoints.mockReset();
  push.mockReset();
});

describe("CheckpointsPanel", () => {
  it("renders the button only on complete rows and navigates with the checkpoint", async () => {
    api.getCheckpoints.mockResolvedValue({
      items: [
        { name: "final", step: 900, complete: true, init_meta: null },
        { name: "checkpoint-300", step: 300, complete: false, init_meta: null },
        { name: "checkpoint-200", step: 200, complete: true, init_meta: null },
      ],
    });
    render(<CheckpointsPanel runId="r-1" />);
    expect(await screen.findByText("checkpoint-300")).toBeInTheDocument();
    const buttons = screen.getAllByRole("button", { name: "New run from this checkpoint" });
    expect(buttons).toHaveLength(2);
    expect(screen.getByText("incomplete")).toBeInTheDocument();
    fireEvent.click(buttons[1]);
    expect(push).toHaveBeenCalledWith("/train/new?from_run=r-1&checkpoint=checkpoint-200");
  });

  it("shows an empty state", async () => {
    api.getCheckpoints.mockResolvedValue({ items: [] });
    render(<CheckpointsPanel runId="r-1" />);
    expect(await screen.findByText(/No checkpoints yet/)).toBeInTheDocument();
  });

  it("shows fetch errors", async () => {
    api.getCheckpoints.mockRejectedValue(new Error("nope"));
    render(<CheckpointsPanel runId="r-1" />);
    expect(await screen.findByText(/Cannot load checkpoints \(nope\)/)).toBeInTheDocument();
  });
});