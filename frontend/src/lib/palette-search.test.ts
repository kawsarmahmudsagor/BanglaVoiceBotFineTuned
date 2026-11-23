import { describe, expect, it } from "vitest";
import { NAV_GROUPS } from "./nav";
import { buildItems } from "./palette-search";
import type { CatalogSummary, DatasetListItem, RecipeListItem, RunListItem } from "./types";

const RUN: RunListItem = {
  id: "r-abc",
  recipe: "smoke",
  recipe_number: 2,
  backend: "fake",
  task: "llm-sft",
  state: "training",
  commercial_ok: true,
  created_at: null,
  started_at: null,
  finished_at: null,
};

const CAT: CatalogSummary = {
  id: "whisper-x",
  task: "asr",
  name: "Whisper X",
  hf_repo: "org/whisper-x",
  params: "809M",
  role: "primary",
  license: { id: "mit", commercial_ok: true, url: "https://x" },
  bangla_status: "finetuned",
  streaming: false,
};

const EMPTY = { runs: [], recipes: [], datasets: [], catalog: [] };

describe("buildItems", () => {
  it("covers every page in the nav", () => {
    const pages = buildItems(EMPTY).filter((i) => i.group === "Pages");
    expect(pages.map((p) => p.href)).toEqual(NAV_GROUPS.flatMap((g) => g.items.map((i) => i.href)));
  });

  it("gives only pages when sources are empty", () => {
    expect(new Set(buildItems(EMPTY).map((i) => i.group))).toEqual(new Set(["Pages"]));
  });

  it("labels runs with id, recipe and version, searchable by state", () => {
    const run = buildItems({ ...EMPTY, runs: [RUN] }).find((i) => i.group === "Runs")!;
    expect(run.label).toBe("r-abc · smoke v2");
    expect(run.href).toBe("/runs/r-abc");
    expect(run.keywords).toContain("training");
  });

  it("labels catalog entries by name with id, repo and task keywords", () => {
    const c = buildItems({ ...EMPTY, catalog: [CAT] }).find((i) => i.group === "Catalog")!;
    expect(c.label).toBe("Whisper X");
    expect(c.href).toBe("/catalog/whisper-x");
    expect(c.keywords).toEqual(["whisper-x", "org/whisper-x", "asr"]);
  });

  it("links recipes and datasets", () => {
    const recipe = { name: "smoke", latest: { number: 1, frozen: false, task: "llm-sft", backend: "fake", base_model: "m", updated_at: null } } as RecipeListItem;
    const dataset = { id: 1, name: "bn-chat", task: "llm-sft", eval_only: false, latest: null } as unknown as DatasetListItem;
    const items = buildItems({ ...EMPTY, recipes: [recipe], datasets: [dataset] });
    expect(items.find((i) => i.group === "Train")?.href).toBe("/train/smoke");
    expect(items.find((i) => i.group === "Datasets")?.href).toBe("/datasets/bn-chat");
  });
});
