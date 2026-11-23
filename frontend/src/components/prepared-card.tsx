import { Panel } from "@/components/matra";
import { Badge } from "@/components/ui/badge";
import type { PreparedRef } from "@/lib/types";

const MAX_SKIPPED = 50;

export function PreparedCard({ refs }: { refs: PreparedRef[] }) {
  if (refs.length === 0) {
    return (
      <Panel title="Prepared data">
        <p className="text-sm text-muted-foreground">Prepared data appears after the prepare phase.</p>
      </Panel>
    );
  }
  return (
    <div className="space-y-4">
      {refs.map((r) => (
        <Panel
          key={`${r.role}-${r.spec_hash}`}
          title={
            <span className="flex items-center gap-2">
              Prepared data <span className="font-mono text-[13px] text-muted-foreground">{r.role}</span>
            </span>
          }
          actions={<Badge variant={r.cache_hit ? "secondary" : "default"}>{r.cache_hit ? "cache hit" : "built"}</Badge>}
        >
          <dl className="grid gap-3 text-sm sm:grid-cols-2">
            <div className="min-w-0"><dt className="text-[13px] text-muted-foreground">Family</dt><dd className="font-mono text-[13px]">{r.spec.family}</dd></div>
            <div className="min-w-0"><dt className="text-[13px] text-muted-foreground">Spec hash</dt><dd className="font-mono text-[13px] break-all">{r.spec_hash}</dd></div>
            <div className="min-w-0 sm:col-span-2">
              <dt className="text-[13px] text-muted-foreground">Counts</dt>
              <dd className="flex flex-wrap gap-x-4 font-mono text-[13px]">
                {Object.entries(r.counts).map(([split, n]) => <span key={split}>{split}: {n}</span>)}
              </dd>
            </div>
          </dl>
          {r.skipped.length > 0 && (
            <div className="mt-3">
              <h3 className="text-[13px] text-muted-foreground">
                Skipped {r.skipped.length}{r.skipped.length > MAX_SKIPPED && ` (showing first ${MAX_SKIPPED})`}
              </h3>
              <ul className="mt-1 space-y-1 text-[13px]">
                {r.skipped.slice(0, MAX_SKIPPED).map((s, i) => (
                  <li key={i} className="min-w-0 break-words">
                    <span className="font-mono">{s.sample_id}</span> <span className="text-muted-foreground">({s.split})</span> {s.reason}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </Panel>
      ))}
    </div>
  );
}
