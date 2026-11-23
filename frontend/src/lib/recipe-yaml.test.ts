import { describe, expect, it } from "vitest";
import { specToYaml, yamlToSpec } from "./recipe-yaml";

const SPEC = {
  name: "r1",
  task: "llm-sft",
  backend: "fake",
  base_model: "m",
  base_model_meta: { license: "mit", commercial_ok: true },
  datasets: { train: "ds@v1", eval: null },
  method: { type: "none" },
  hparams: { steps: 3, fail_phase: null },
  export: ["dummy"],
};

describe("recipe yaml", () => {
  it("round-trips spec -> yaml -> spec", () => {
    const r = yamlToSpec(specToYaml(SPEC));
    expect(r).toEqual({ ok: true, spec: SPEC });
  });

  it("reports parse errors without throwing", () => {
    const r = yamlToSpec("a: [unclosed");
    expect(r.ok).toBe(false);
  });

  it("rejects non-mapping documents", () => {
    expect(yamlToSpec("- 1\n- 2").ok).toBe(false);
    expect(yamlToSpec("").ok).toBe(false);
  });
});

describe("recipe yaml datasets.split", () => {
  it("round-trips datasets.split", () => {
    const spec = {
      ...SPEC,
      datasets: { train: "ds@v1", eval: null, split: { use_dataset_splits: false, ratios: { train: 0.9, val: 0.05, test: 0.05 } } },
    };
    expect(yamlToSpec(specToYaml(spec))).toEqual({ ok: true, spec });
  });
});

describe("recipe yaml init_from", () => {
  it("round-trips a run checkpoint and a model version", () => {
    for (const init_from of [{ run: "r-1", checkpoint: 300 }, { run: "r-1", checkpoint: "final" }, { model_version: "m@v2" }]) {
      const spec = { ...SPEC, init_from };
      expect(yamlToSpec(specToYaml(spec))).toEqual({ ok: true, spec });
    }
  });
});