import { DEFAULT_SPLIT, toPct } from "@/lib/splits";

export function SplitSummary({ splitSource, given }: { splitSource: "dataset" | "none"; given: Record<string, number> }) {
  if (splitSource === "dataset") {
    return (
      <div className="font-mono text-sm">
        {`Splits provided by dataset: train ${given.train ?? 0} · valid ${given.val ?? 0} · test ${given.test ?? 0} (unassigned ${given.unassigned ?? 0})`}
      </div>
    );
  }
  const d = DEFAULT_SPLIT.ratios;
  return (
    <div className="text-sm">{`No splits in dataset — assigned per run (default ${toPct(d.train)}/${toPct(d.val)}/${toPct(d.test)})`}</div>
  );
}
