import { describe, expect, it } from "vitest";
import { BANGLA_LABEL, SECTIONS, filterCatalog, sectionForTask } from "./catalog";
import type { CatalogSummary } from "./types";

function entry(id: string, over: Partial<CatalogSummary> = {}): CatalogSummary {
  return {
    id,
    task: "asr",
    name: id,
    hf_repo: `org/${id}`,
    params: "1B",
    role: "primary",
    license: { id: "mit", commercial_ok: true, url: "https://x" },
    bangla_status: "native",
    streaming: false,
    ...over,
  };
}

const ALL = [
  entry("a"),
  entry("b", { role: "comparison", bangla_status: "related" }),
  entry("c", { role: "benchmark", license: { id: "cc-by-nc-4.0", commercial_ok: false, url: "https://x" } }),
];

describe("filterCatalog", () => {
  it("returns everything with no filter", () => {
    expect(filterCatalog(ALL, {}).map((e) => e.id)).toEqual(["a", "b", "c"]);
  });
  it("filters by role", () => {
    expect(filterCatalog(ALL, { role: "comparison" }).map((e) => e.id)).toEqual(["b"]);
  });
  it("keeps only commercially usable models", () => {
    expect(filterCatalog(ALL, { commercialOnly: true }).map((e) => e.id)).toEqual(["a", "b"]);
  });
  it("filters by Bangla status", () => {
    expect(filterCatalog(ALL, { bangla: "related" }).map((e) => e.id)).toEqual(["b"]);
  });
  it("combines filters", () => {
    expect(filterCatalog(ALL, { role: "primary", bangla: "related" })).toEqual([]);
  });
});

describe("SECTIONS", () => {
  it("maps sections to tasks", () => {
    expect(SECTIONS).toEqual([
      { key: "llm", task: "llm-sft", label: "LLM" },
      { key: "stt", task: "asr", label: "STT" },
      { key: "tts", task: "tts", label: "TTS" },
    ]);
    expect(sectionForTask("asr").label).toBe("STT");
  });
  it("labels every Bangla status", () => {
    expect(BANGLA_LABEL).toEqual({
      native: "Bangla native",
      finetuned: "Bangla fine-tune",
      related: "Related languages only",
      none: "No Bangla",
    });
  });
});
