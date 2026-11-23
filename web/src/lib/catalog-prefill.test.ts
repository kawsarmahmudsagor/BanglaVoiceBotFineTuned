import { describe, expect, it } from "vitest";
import { NO_BACKEND_NOTICE, backendNotice, catalogToSpec } from "./catalog-prefill";
import type { CatalogEntry } from "./types";

const ENTRY = {
  id: "qwen3-4b",
  task: "llm-sft",
  name: "Qwen3 4B",
  hf_repo: "Qwen/Qwen3-4B-Instruct-2507",
  params: "4B",
  role: "primary",
  license: { id: "apache-2.0", commercial_ok: true, url: "https://x" },
  bangla: { status: "native", evidence: "listed" },
  runtime: { engine: "llama-server", convert: "gguf", streaming: true },
  description: "d",
  dataset: { format: "llm-sft", required_fields: ["messages"], optional_fields: [], example: "{}", recommended: {} },
  finetune: { methods: ["qlora"], frameworks: ["unsloth"], targets: [], hyperparams: {}, metrics: [], pitfalls: [] },
  recipe_template: { method: { type: "qlora" }, hparams: { learning_rate: 0.0002, epochs: 2 }, export: ["gguf"] },
  sources: [{ title: "card", url: "https://x" }],
  verified: "2026-11-23",
} as CatalogEntry;

describe("catalogToSpec", () => {
  it("maps a catalog entry onto a new recipe spec", () => {
    expect(catalogToSpec(ENTRY)).toEqual({
      name: "",
      task: "llm-sft",
      backend: "",
      base_model: "Qwen/Qwen3-4B-Instruct-2507",
      base_model_meta: { license: "apache-2.0", commercial_ok: true },
      datasets: { train: "", eval: null },
      method: { type: "qlora" },
      hparams: { learning_rate: 0.0002, epochs: 2 },
      export: ["gguf"],
    });
  });

  it("uses the template backend when set", () => {
    const e = { ...ENTRY, recipe_template: { ...ENTRY.recipe_template, backend: "unsloth" } };
    expect(catalogToSpec(e).backend).toBe("unsloth");
  });

  it("copies so editing the spec never mutates the entry", () => {
    const spec = catalogToSpec(ENTRY);
    (spec.hparams as Record<string, unknown>).epochs = 9;
    expect(ENTRY.recipe_template.hparams.epochs).toBe(2);
  });
});

describe("backendNotice", () => {
  it("warns when no backend is set", () => {
    expect(backendNotice("", ["fake"])).toBe(NO_BACKEND_NOTICE);
  });
  it("warns when the backend is not installed", () => {
    expect(backendNotice("unsloth", ["fake"])).toBe(NO_BACKEND_NOTICE);
  });
  it("is null when the backend is installed", () => {
    expect(backendNotice("fake", ["fake"])).toBeNull();
  });
  it("uses the spec wording", () => {
    expect(NO_BACKEND_NOTICE).toBe("No installed backend supports this model yet. Pick a backend or add one under backends/.");
  });
});
