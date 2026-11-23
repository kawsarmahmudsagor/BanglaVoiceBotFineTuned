import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { StartFromPicker } from "./start-from-picker";

const api = vi.hoisted(() => ({ getInitSources: vi.fn() }));
vi.mock("@/lib/api", () => api);

const SOURCES = {
  runs: [
    {
      id: "r-1",
      recipe: "cfg",
      state: "done",
      finished_at: null,
      checkpoints: [
        { name: "final", step: 900, complete: true, init_meta: null },
        { name: "checkpoint-300", step: 300, complete: true, init_meta: null },
      ],
    },
  ],
  model_versions: [{ ref: "bn-llm@v2", run_id: "r-1", metrics: {} }],
};
const Q = { task: "llm-sft", backend: "unsloth", baseModel: "Qwen/Q" };

beforeEach(() => {
  api.getInitSources.mockReset();
  api.getInitSources.mockResolvedValue(SOURCES);
});

describe("StartFromPicker", () => {
  it("lists base model, run checkpoints and model versions", async () => {
    render(<StartFromPicker {...Q} value={null} onChange={() => {}} />);
    expect(await screen.findByRole("option", { name: "r-1 · final · step 900" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "r-1 · checkpoint-300 · step 300" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "bn-llm@v2" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Base model (default)" })).toBeInTheDocument();
    expect(api.getInitSources).toHaveBeenCalledWith({ task: "llm-sft", backend: "unsloth", base_model: "Qwen/Q" });
  });

  it("emits run+checkpoint, int step for intermediate checkpoints", async () => {
    const onChange = vi.fn();
    render(<StartFromPicker {...Q} value={null} onChange={onChange} />);
    await screen.findByRole("option", { name: "r-1 · final · step 900" });
    fireEvent.change(screen.getByLabelText("Start from"), { target: { value: "run|r-1|final" } });
    expect(onChange).toHaveBeenLastCalledWith({ run: "r-1", checkpoint: "final" });
    fireEvent.change(screen.getByLabelText("Start from"), { target: { value: "run|r-1|300" } });
    expect(onChange).toHaveBeenLastCalledWith({ run: "r-1", checkpoint: 300 });
    fireEvent.change(screen.getByLabelText("Start from"), { target: { value: "mv|bn-llm@v2" } });
    expect(onChange).toHaveBeenLastCalledWith({ model_version: "bn-llm@v2" });
    fireEvent.change(screen.getByLabelText("Start from"), { target: { value: "" } });
    expect(onChange).toHaveBeenLastCalledWith(null);
  });

  it("clears a selection that is no longer listed", async () => {
    const onChange = vi.fn();
    render(<StartFromPicker {...Q} value={{ run: "r-gone", checkpoint: "final" }} onChange={onChange} />);
    await waitFor(() => expect(onChange).toHaveBeenCalledWith(null));
  });

  it("keeps a listed selection", async () => {
    const onChange = vi.fn();
    render(<StartFromPicker {...Q} value={{ run: "r-1", checkpoint: 300 }} onChange={onChange} />);
    await screen.findByRole("option", { name: "r-1 · final · step 900" });
    expect(onChange).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Start from")).toHaveValue("run|r-1|300");
  });

  it("refetches when the base model changes and clears a stale choice", async () => {
    const onChange = vi.fn();
    const { rerender } = render(<StartFromPicker {...Q} value={{ run: "r-1", checkpoint: "final" }} onChange={onChange} />);
    await screen.findByRole("option", { name: "r-1 · final · step 900" });
    api.getInitSources.mockResolvedValue({ runs: [], model_versions: [] });
    rerender(<StartFromPicker {...Q} baseModel="Other/M" value={{ run: "r-1", checkpoint: "final" }} onChange={onChange} />);
    await waitFor(() => expect(api.getInitSources).toHaveBeenLastCalledWith({ task: "llm-sft", backend: "unsloth", base_model: "Other/M" }));
    await waitFor(() => expect(onChange).toHaveBeenCalledWith(null));
  });

  it("does not fetch until task, backend and base model are set", () => {
    render(<StartFromPicker task="llm-sft" backend="" baseModel="x" value={null} onChange={() => {}} />);
    expect(api.getInitSources).not.toHaveBeenCalled();
  });

  it("shows fetch errors and keeps the selection", async () => {
    api.getInitSources.mockRejectedValue(new Error("boom"));
    const onChange = vi.fn();
    render(<StartFromPicker {...Q} value={{ run: "r-1", checkpoint: "final" }} onChange={onChange} />);
    expect(await screen.findByText(/Cannot load start-from sources \(boom\)/)).toBeInTheDocument();
    expect(onChange).not.toHaveBeenCalled();
  });
});