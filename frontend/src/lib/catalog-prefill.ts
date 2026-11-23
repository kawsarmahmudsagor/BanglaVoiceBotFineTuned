import type { CatalogEntry } from "./types";

export const NO_BACKEND_NOTICE = "No installed backend supports this model yet. Pick a backend or add one under backend/workers/backends/.";

/** A new recipe spec pre-filled from a catalog entry. Deep-copied so edits never touch the entry. */
export function catalogToSpec(entry: CatalogEntry): Record<string, unknown> {
  const t = structuredClone(entry.recipe_template);
  return {
    name: "",
    task: entry.task,
    backend: t.backend ?? "",
    base_model: entry.hf_repo,
    base_model_meta: { license: entry.license.id, commercial_ok: entry.license.commercial_ok },
    datasets: { train: "", eval: null },
    method: t.method,
    hparams: t.hparams,
    export: t.export,
  };
}

export function backendNotice(backend: string, installed: string[]): string | null {
  return backend && installed.includes(backend) ? null : NO_BACKEND_NOTICE;
}
