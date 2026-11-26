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
  getInitSources: vi.fn(),
  listCatalog: vi.fn(),
  getCatalogEntry: vi.fn(),
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
  api.getInitSources.mockResolvedValue({ runs: [], model_versions: [] });
  api.listCatalog.mockResolvedValue({ entries: [], errors: [] });
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
    expect(screen.getByRole("button", { name: "Save training config" })).toBeDisabled();
  });

  it("enables save when the ratios are valid (control)", async () => {
    const spec = {
      ...SPEC,
      datasets: { ...SPEC.datasets, split: { use_dataset_splits: false, ratios: { train: 0.9, val: 0.05, test: 0.05 } } },
    };
    render(<RecipeEditor initial={spec} />);
    await waitFor(() => expect(screen.getByRole("button", { name: "Save training config" })).toBeEnabled());
  });
});

describe("RecipeEditor start from", () => {
  it("renders the picker and writes init_from into the YAML", async () => {
    api.getInitSources.mockResolvedValue({
      runs: [{ id: "r-1", recipe: "c", state: "done", finished_at: null, checkpoints: [{ name: "final", step: 9, complete: true, init_meta: null }] }],
      model_versions: [],
    });
    render(<RecipeEditor initial={SPEC} />);
    await screen.findByRole("option", { name: "r-1 · final · step 9" });
    fireEvent.change(screen.getByLabelText("Start from"), { target: { value: "run|r-1|final" } });
    const yaml = screen.getByLabelText("Training config YAML") as HTMLTextAreaElement;
    await waitFor(() => expect(yaml.value).toMatch(/init_from:\s+run: r-1\s+checkpoint: final/));
    fireEvent.change(screen.getByLabelText("Start from"), { target: { value: "" } });
    await waitFor(() => expect(yaml.value).not.toMatch(/init_from/));
  });
});
describe("RecipeEditor model picker", () => {
  const lic = { id: "apache-2.0", commercial_ok: true, url: "" };
  const CATALOG = [
    { id: "qwen3-asr-0-6b-bn", task: "asr", name: "Qwen3-ASR 0.6B Bengali", hf_repo: "SayedShaun/Qwen3-ASR-0.6B-Bengali", role: "primary", license: lic },
    { id: "whisper-large-v3-turbo", task: "asr", name: "Whisper large-v3-turbo", hf_repo: "openai/whisper-large-v3-turbo", role: "comparison", license: { ...lic, id: "mit" } },
    { id: "qwen3-4b", task: "llm-sft", name: "Qwen3 4B", hf_repo: "Qwen/Qwen3-4B-Instruct-2507", role: "primary", license: lic },
  ];
  const caps = (models: string[], methods = ["full"], exp = ["safetensors"]) =>
    ({ methods, export_formats: exp, schema: { properties: { base_model: { type: "string", enum: models } } } });
  const BACKENDS = [
    { name: "hf", label: "HF", tasks: { asr: caps(["openai/whisper-large-v3-turbo"], ["full", "lora"], ["ct2"]) } },
    { name: "qwen-asr", label: "Qwen", tasks: { asr: caps(["SayedShaun/Qwen3-ASR-0.6B-Bengali"]) } },
  ];
  const ASR = { ...SPEC, task: "asr", backend: "", base_model: "", method: { type: "" }, datasets: { train: "", eval: null } };

  beforeEach(() => {
    api.listCatalog.mockResolvedValue({ entries: CATALOG, errors: [] });
    api.listBackends.mockResolvedValue(BACKENDS);
    api.getCatalogEntry.mockResolvedValue({
      ...CATALOG[0], recipe_template: { backend: "qwen-asr", method: { type: "full" }, hparams: { learning_rate: 2e-5 }, export: ["safetensors"] },
    });
  });

  it("labels the types and lists only models of the selected type", async () => {
    render(<RecipeEditor initial={ASR} />);
    expect(screen.getByRole("option", { name: "ASR (speech-to-text)" })).toBeInTheDocument();
    await screen.findByRole("option", { name: "Qwen3-ASR 0.6B Bengali (SayedShaun/Qwen3-ASR-0.6B-Bengali)" });
    const model = screen.getByLabelText("Model");
    expect(model).toContainHTML("openai/whisper-large-v3-turbo");
    expect(model).not.toContainHTML("Qwen/Qwen3-4B-Instruct-2507");
  });

  it("picking a catalog model fills backend, method and license from the catalog", async () => {
    render(<RecipeEditor initial={ASR} />);
    await screen.findByRole("option", { name: /Qwen3-ASR 0.6B Bengali/ });
    fireEvent.change(screen.getByLabelText("Model"), { target: { value: "SayedShaun/Qwen3-ASR-0.6B-Bengali" } });
    const yaml = screen.getByLabelText("Training config YAML") as HTMLTextAreaElement;
    await waitFor(() => expect(yaml.value).toMatch(/backend: qwen-asr/));
    expect(yaml.value).toMatch(/base_model: SayedShaun\/Qwen3-ASR-0.6B-Bengali/);
    expect(yaml.value).toMatch(/license: apache-2.0/);
    expect(api.getCatalogEntry).toHaveBeenCalledWith("qwen3-asr-0-6b-bn");
  });

  it("changing the type clears the model", async () => {
    render(<RecipeEditor initial={{ ...ASR, base_model: "openai/whisper-large-v3-turbo" }} />);
    await screen.findByRole("option", { name: /Whisper large-v3-turbo/ });
    fireEvent.change(screen.getByLabelText("Type"), { target: { value: "llm-sft" } });
    await waitFor(() => expect((screen.getByLabelText("Model") as HTMLSelectElement).value).toBe(""));
    expect(screen.getByLabelText("Model")).toContainHTML("Qwen/Qwen3-4B-Instruct-2507");
  });

  it("custom lets you type a model that is not listed", async () => {
    render(<RecipeEditor initial={ASR} />);
    await screen.findByRole("option", { name: /Qwen3-ASR 0.6B Bengali/ });
    fireEvent.change(screen.getByLabelText("Model"), { target: { value: "__custom__" } });
    fireEvent.change(screen.getByLabelText("Custom model (Hugging Face repo or path)"), { target: { value: "me/my-whisper" } });
    const yaml = screen.getByLabelText("Training config YAML") as HTMLTextAreaElement;
    await waitFor(() => expect(yaml.value).toMatch(/base_model: me\/my-whisper/));
  });

  it("keeps a pre-filled model that is not in the list selectable", async () => {
    render(<RecipeEditor initial={{ ...ASR, base_model: "someone/other-model" }} />);
    await screen.findByRole("option", { name: /Qwen3-ASR 0.6B Bengali/ });
    expect((screen.getByLabelText("Model") as HTMLSelectElement).value).toBe("someone/other-model");
  });
});
