import type { DatasetSummary } from "../types";

// Placeholder: Task 13 builds the review queue.
export function Review({ dataset }: { dataset: DatasetSummary | null }) {
  return <div className="panel empty">Review queue for {dataset ? dataset.name : "the selected dataset"} comes next.</div>;
}
