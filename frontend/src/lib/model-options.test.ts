import { describe, expect, it } from "vitest";
import { applyModel, backendSupports, modelOptions, type BackendCapsLite } from "./model-options";
import type { CatalogEntry, CatalogSummary } from "./types";

type Template = CatalogEntry["recipe_template"];
const entry = (over: Partial<CatalogSummary> & { id: string; task: CatalogSummary["task"]; hf_repo: string }): CatalogSummary =>
  ({ name: over.id, role: "comparison", license: { id: "mit", commercial_ok: true, url: "" }, ...over }) as CatalogSummary;
const QWEN_T = { backend: "qwen-asr", method: { type: "full" }, hparams: { learning_rate: 2e-5 }, export: ["safetensors"] } as Template;
const HF_T = { backend: "hf", method: { type: "full" }, hparams: { learning_rate: 1e-5 }, export: ["ct2"] } as Template;

const ENTRIES = [
  entry({ id: "whisper-large-v3-turbo", task: "asr", hf_repo: "openai/whisper-large-v3-turbo" }),
  entry({ id: "whisper-large-v3-turbo-bn", task: "asr", hf_repo: "mozilla-ai/whisper-large-v3-turbo-bn", role: "primary",
    license: { id: "apache-2.0", commercial_ok: true, url: "" } as CatalogSummary["license"] }),
  entry({ id: "qwen3-asr-0-6b-bn", task: "asr", hf_repo: "SayedShaun/Qwen3-ASR-0.6B-Bengali", role: "primary" }),
  entry({ id: "qwen3-4b", task: "llm-sft", hf_repo: "Qwen/Qwen3-4B-Instruct-2507" }),
];

const enumSchema = (models: string[]) => ({ properties: { base_model: { type: "string", enum: models } } });
const BACKENDS: BackendCapsLite[] = [
  { name: "hf", tasks: { asr: { methods: ["full", "lora"], export_formats: ["ct2", "safetensors"],
    schema: enumSchema(["openai/whisper-large-v3-turbo", "mozilla-ai/whisper-large-v3-turbo-bn", "facebook/mms-1b-all"]) } } },
  { name: "qwen-asr", tasks: { asr: { methods: ["full"], export_formats: ["safetensors"],
    schema: enumSchema(["SayedShaun/Qwen3-ASR-0.6B-Bengali", "Qwen/Qwen3-ASR-0.6B-hf"]) } } },
  { name: "fake", tasks: { "llm-sft": { methods: ["none"], export_formats: [], schema: {} } } },
];

describe("modelOptions", () => {
  it("lists only the selected type, catalog primaries first, then backend-only models", () => {
    const opts = modelOptions("asr", ENTRIES, BACKENDS);
    expect(opts.map((o) => o.value)).toEqual([
      "SayedShaun/Qwen3-ASR-0.6B-Bengali",
      "mozilla-ai/whisper-large-v3-turbo-bn",
      "openai/whisper-large-v3-turbo",
      "facebook/mms-1b-all",
      "Qwen/Qwen3-ASR-0.6B-hf",
    ]);
    expect(opts.filter((o) => o.entry).length).toBe(3);
  });

  it("never mixes types", () => {
    expect(modelOptions("llm-sft", ENTRIES, BACKENDS).map((o) => o.value)).toEqual(["Qwen/Qwen3-4B-Instruct-2507"]);
    expect(modelOptions("tts", ENTRIES, BACKENDS)).toEqual([]);
  });
});

describe("backendSupports", () => {
  it("uses the backend's base_model enum; no enum means any model", () => {
    expect(backendSupports(BACKENDS[0], "asr", "openai/whisper-large-v3-turbo")).toBe(true);
    expect(backendSupports(BACKENDS[0], "asr", "SayedShaun/Qwen3-ASR-0.6B-Bengali")).toBe(false);
    expect(backendSupports(BACKENDS[2], "llm-sft", "anything")).toBe(true);
    expect(backendSupports(BACKENDS[2], "asr", "anything")).toBe(false);
  });
});

describe("applyModel", () => {
  const base = { name: "x", task: "asr", backend: "", base_model: "", base_model_meta: { license: "", commercial_ok: false },
    datasets: { train: "ds@v1", eval: null }, method: { type: "" }, hparams: {}, export: [] };

  it("fills license and the catalog recipe when no backend is chosen", () => {
    const opt = modelOptions("asr", ENTRIES, BACKENDS).find((o) => o.value === "SayedShaun/Qwen3-ASR-0.6B-Bengali")!;
    const s = applyModel(base, opt, BACKENDS, QWEN_T);
    expect(s).toMatchObject({ base_model: "SayedShaun/Qwen3-ASR-0.6B-Bengali", backend: "qwen-asr",
      method: { type: "full" }, hparams: { learning_rate: 2e-5 }, export: ["safetensors"],
      base_model_meta: { license: "mit", commercial_ok: true }, datasets: { train: "ds@v1", eval: null } });
  });

  it("keeps a compatible backend and its hparams, updating only model and license", () => {
    const spec = { ...base, backend: "hf", method: { type: "lora" }, hparams: { learning_rate: 3e-4 }, export: ["ct2"] };
    const opt = modelOptions("asr", ENTRIES, BACKENDS).find((o) => o.value === "mozilla-ai/whisper-large-v3-turbo-bn")!;
    expect(applyModel(spec, opt, BACKENDS, HF_T)).toMatchObject({ backend: "hf", method: { type: "lora" },
      hparams: { learning_rate: 3e-4 }, base_model_meta: { license: "apache-2.0", commercial_ok: true } });
  });

  it("switches an incompatible backend to the catalog's", () => {
    const spec = { ...base, backend: "hf", method: { type: "lora" } };
    const opt = modelOptions("asr", ENTRIES, BACKENDS).find((o) => o.value === "SayedShaun/Qwen3-ASR-0.6B-Bengali")!;
    expect(applyModel(spec, opt, BACKENDS, QWEN_T)).toMatchObject({ backend: "qwen-asr", method: { type: "full" } });
  });

  it("backend-only model: picks the first supporting backend, leaves license for the user", () => {
    const opt = modelOptions("asr", ENTRIES, BACKENDS).find((o) => o.value === "facebook/mms-1b-all")!;
    const s = applyModel(base, opt, BACKENDS);
    expect(s).toMatchObject({ base_model: "facebook/mms-1b-all", backend: "hf", method: { type: "full" },
      base_model_meta: { license: "", commercial_ok: false } });
  });

  it("does not switch to a catalog backend that is not installed", () => {
    const opt = { value: "x/y", label: "y", entry: entry({ id: "y", task: "asr", hf_repo: "x/y" }) };
    const nemo = { backend: "nemo", method: { type: "full" }, hparams: {}, export: [] } as Template;
    expect(applyModel(base, opt, BACKENDS, nemo)).toMatchObject({ base_model: "x/y", backend: "" });
  });

  it("catalog model without a fetched template falls back to a backend that lists it", () => {
    const opt = modelOptions("asr", ENTRIES, BACKENDS).find((o) => o.value === "SayedShaun/Qwen3-ASR-0.6B-Bengali")!;
    expect(applyModel(base, opt, BACKENDS)).toMatchObject({ base_model: "SayedShaun/Qwen3-ASR-0.6B-Bengali",
      backend: "qwen-asr", method: { type: "full" }, base_model_meta: { license: "mit", commercial_ok: true } });
  });
});
