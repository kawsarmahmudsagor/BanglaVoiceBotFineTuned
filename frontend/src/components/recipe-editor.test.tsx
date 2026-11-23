import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { RecipeEditor } from "./recipe-editor";

const api = vi.hoisted(() => ({
  listBackends: vi.fn(),
  listDatasets: vi.fn(),
  getDataset: vi.fn(),
  getVersion: vi.fn(),
  saveRecipe: vi.fn(),
  validateRecipe: vi.fn(),
}));
vi.mock("@/lib/api", () => api);
vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }) }));

const SPEC = {
  name: "r1",
  task: "llm-sft",
  backend: "fake",
  base_model: "m",
  base_model_meta: { license: "mit", commercial_ok: true },
  datasets: { train: "ds@v1", eval: null },
  method: { type: "none" },
  hparams: {},
  export: [],
};
const VERSION = { split_source: "dataset", given_splits: { train: 80, val: 10, test: 10, unassigned: 0 }, n_samples: 100 };

beforeEach(() => {
  Object.values(api).forEach((f) => f.mockReset());
  api.listBackends.mockResolvedValue([]);
  api.listDatasets.mockResolvedValue([{ name: "ds", task: "llm-sft", eval_only: false }]);
  api.getDataset.mockResolvedValue({ name: "ds", versions: [{ number: 1 }, { number: 2 }] });
  api.getVersion.mockResolvedValue(VERSION);
  api.validateRecipe.mockResolvedValue({ errors: [], spec: null });
});

describe("RecipeEditor data splits", () => {
  it("shows a notice and keeps inputs locked when the split info cannot load", async () => {
    api.getVersion.mockRejectedValue(new Error("boom"));
    render(<RecipeEditor initial={SPEC} />);
    expect(await screen.findByText(/Cannot load the train dataset's splits \(boom\)/)).toBeInTheDocument();
    expect(screen.getByLabelText("Train %")).toBeDisabled();
  });

  it("refetches and clears old counts when the train dataset changes", async () => {
    render(<RecipeEditor initial={SPEC} />);
    expect(await screen.findByText("(train 80 · valid 10 · test 10)")).toBeInTheDocument();
    api.getVersion.mockReturnValue(new Promise(() => {}));
    await waitFor(() => expect(screen.getByLabelText("Train dataset")).toContainHTML("ds@v2"));
    fireEvent.change(screen.getByLabelText("Train dataset"), { target: { value: "ds@v2" } });
    await waitFor(() => expect(api.getVersion).toHaveBeenLastCalledWith("ds", 2));
    expect(screen.queryByText("(train 80 · valid 10 · test 10)")).toBeNull();
  });

  it("disables save while the ratios are invalid", async () => {
    const spec = {
      ...SPEC,
      datasets: { ...SPEC.datasets, split: { use_dataset_splits: false, ratios: { train: 0.6, val: 0.2, test: 0.1 } } },
    };
    render(<RecipeEditor initial={spec} />);
    expect(await screen.findByText("Ratios must add up to 100%")).toBeInTheDocument();
    await waitFor(() => expect(api.validateRecipe).toHaveBeenCalled());
    await waitFor(() => expect(screen.queryByText("Validating�")).toBeNull());
    expect(screen.getByRole("button", { name: "Save recipe" })).toBeDisabled();
  });

  it("enables save when the ratios are valid (control)", async () => {
    const spec = {
      ...SPEC,
      datasets: { ...SPEC.datasets, split: { use_dataset_splits: false, ratios: { train: 0.9, val: 0.05, test: 0.05 } } },
    };
    render(<RecipeEditor initial={spec} />);
    await waitFor(() => expect(screen.getByRole("button", { name: "Save recipe" })).toBeEnabled());
  });
});
