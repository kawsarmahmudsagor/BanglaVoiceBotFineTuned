import type { CatalogEntry, CatalogSummary, Task } from "./types";

type RecipeTemplate = CatalogEntry["recipe_template"];

type Spec = Record<string, unknown>;

/** The parts of a backend's capabilities that model selection needs. */
export interface BackendCapsLite {
  name: string;
  tasks: Record<string, { methods: string[]; export_formats: string[]; schema?: { properties?: Record<string, unknown> } }>;
}

export interface ModelOption {
  value: string;
  label: string;
  /** Set for catalog models: the license comes from it; the recipe template is fetched with getCatalogEntry. */
  entry?: CatalogSummary;
}

export const TASK_LABELS: Record<Task, string> = {
  asr: "ASR (speech-to-text)",
  "llm-sft": "LLM (chat fine-tune)",
  tts: "TTS (text-to-speech)",
};

function schemaModels(b: BackendCapsLite, task: string): string[] | null {
  const prop = b.tasks?.[task]?.schema?.properties?.base_model as { enum?: unknown[] } | undefined;
  return Array.isArray(prop?.enum) ? prop.enum.filter((m): m is string => typeof m === "string") : null;
}

/** Whether `backend` trains `model` for `task`. A backend without a base_model enum accepts any model. */
export function backendSupports(b: BackendCapsLite, task: string, model: string): boolean {
  if (!b.tasks?.[task]) return false;
  const models = schemaModels(b, task);
  return models === null || models.includes(model);
}

/** Models for one task type: catalog entries (primary first, then by name), then models only a backend lists. */
export function modelOptions(task: Task, entries: CatalogSummary[], backends: BackendCapsLite[]): ModelOption[] {
  const catalog = entries
    .filter((e) => e.task === task)
    .sort((a, b) => Number(b.role === "primary") - Number(a.role === "primary") || a.name.localeCompare(b.name))
    .map((e) => ({ value: e.hf_repo, label: `${e.name} (${e.hf_repo})`, entry: e }));
  const seen = new Set(catalog.map((o) => o.value));
  const extra: ModelOption[] = [];
  for (const b of backends) {
    for (const m of schemaModels(b, task) ?? []) {
      if (!seen.has(m)) {
        seen.add(m);
        extra.push({ value: m, label: m });
      }
    }
  }
  return [...catalog, ...extra];
}

/**
 * The spec with `opt` as base model. A catalog model sets the license. The backend is kept when it supports the
 * model; otherwise the catalog's (installed) backend and its recipe `template` are applied. Without a template (a
 * model only a backend lists, or the catalog entry could not be fetched) the first backend whose base_model list
 * includes the model is chosen, with its first method.
 */
export function applyModel(spec: Spec, opt: ModelOption, backends: BackendCapsLite[], template?: RecipeTemplate): Spec {
  const task = String(spec.task ?? "");
  const next: Spec = { ...spec, base_model: opt.value };
  if (opt.entry) next.base_model_meta = { license: opt.entry.license.id, commercial_ok: opt.entry.license.commercial_ok };

  const current = backends.find((b) => b.name === spec.backend);
  if (current && backendSupports(current, task, opt.value)) return next;

  const fromCatalog = template?.backend ? backends.find((b) => b.name === template.backend) : undefined;
  if (template && fromCatalog && backendSupports(fromCatalog, task, opt.value)) {
    const t = structuredClone(template);
    return { ...next, backend: fromCatalog.name, method: t.method, hparams: t.hparams ?? {}, export: t.export ?? [] };
  }
  // The catalog names a backend that is not installed (or cannot train it): leave the choice to the user.
  if (template) return { ...next, backend: "" };

  const any = backends.find((b) => backendSupports(b, task, opt.value) && schemaModels(b, task) !== null);
  if (!any) return next;
  const cap = any.tasks[task];
  return { ...next, backend: any.name, method: { type: cap.methods[0] ?? "" },
    export: (Array.isArray(spec.export) ? (spec.export as string[]) : []).filter((f) => cap.export_formats.includes(f)) };
}
