import type { BanglaStatus, CatalogRole, CatalogSummary, Task } from "./types";

export type SectionKey = "llm" | "stt" | "tts";

export const SECTIONS: { key: SectionKey; task: Task; label: string }[] = [
  { key: "llm", task: "llm-sft", label: "LLM" },
  { key: "stt", task: "asr", label: "STT" },
  { key: "tts", task: "tts", label: "TTS" },
];

export function sectionForTask(task: Task) {
  return SECTIONS.find((s) => s.task === task) ?? SECTIONS[0];
}

export const BANGLA_LABEL: Record<BanglaStatus, string> = {
  native: "Bangla native",
  finetuned: "Bangla fine-tune",
  related: "Related languages only",
  none: "No Bangla",
};

export const ROLE_LABEL: Record<CatalogRole, string> = {
  primary: "Primary",
  comparison: "Comparison",
  benchmark: "Benchmark only",
};

export interface CatalogFilter {
  role?: CatalogRole;
  commercialOnly?: boolean;
  bangla?: BanglaStatus;
}

export function filterCatalog(entries: CatalogSummary[], f: CatalogFilter): CatalogSummary[] {
  return entries.filter(
    (e) =>
      (!f.role || e.role === f.role) &&
      (!f.commercialOnly || e.license.commercial_ok) &&
      (!f.bangla || e.bangla_status === f.bangla),
  );
}
