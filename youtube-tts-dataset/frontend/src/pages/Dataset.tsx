import type { DatasetSummary } from "../types";

// Placeholder: Task 13 builds stats, split preview and export.
export function DatasetPage({ dataset }: { dataset: DatasetSummary | null }) {
  return <div className="panel empty">Stats and export for {dataset ? dataset.name : "the selected dataset"} come next.</div>;
}
