import Link from "next/link";
import type { ReportIssue, VersionReport } from "@/lib/types";

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

function group(issues: ReportIssue[]): [string, ReportIssue[]][] {
  const m = new Map<string, ReportIssue[]>();
  for (const i of issues) m.set(i.code, [...(m.get(i.code) ?? []), i]);
  return [...m.entries()];
}

function IssueGroups({
  issues, tone, dataset, version,
}: { issues: ReportIssue[]; tone: string; dataset?: string; version?: number }) {
  return (
    <ul className="space-y-1">
      {group(issues).map(([code, items]) => (
        <li key={code}>
          <details className="rounded-md border px-2 py-1 text-sm">
            <summary className="cursor-pointer">
              <span className={`font-mono ${tone}`}>{code}</span>{" "}
              <span className="text-muted-foreground">({items.length})</span>
            </summary>
            <ul className="mt-1 max-h-64 space-y-0.5 overflow-auto pl-3">
              {items.map((i, idx) => (
                <li key={idx} className="text-xs">
                  {i.sample_id ? (
                    dataset && version !== undefined ? (
                      <Link
                        href={`/datasets/${encodeURIComponent(dataset)}/v/${version}?warning=${encodeURIComponent(code)}`}
                        className="font-mono underline"
                      >
                        {i.sample_id}
                      </Link>
                    ) : (
                      <span className="font-mono">{i.sample_id}</span>
                    )
                  ) : null}
                  {i.message ? <span className="text-muted-foreground"> {i.message}</span> : null}
                </li>
              ))}
            </ul>
          </details>
        </li>
      ))}
    </ul>
  );
}

export function ReportView({
  report, dataset, version,
}: { report: VersionReport | null; dataset?: string; version?: number }) {
  if (!report) return <div className="text-sm text-muted-foreground">No report</div>;
  const errors = report.errors ?? [];
  const warnings = report.warnings ?? [];
  const stats = (report.stats ?? {}) as Record<string, unknown>;
  return (
    <div className="space-y-3">
      <div className="flex gap-4 text-sm font-semibold">
        <span className={errors.length ? "text-red-700 dark:text-red-300" : ""}>{plural(errors.length, "error")}</span>
        <span className={warnings.length ? "text-amber-700 dark:text-amber-300" : ""}>
          {plural(warnings.length, "warning")}
        </span>
      </div>
      {errors.length > 0 && (
        <IssueGroups issues={errors} tone="text-red-700 dark:text-red-300" dataset={dataset} version={version} />
      )}
      {warnings.length > 0 && (
        <IssueGroups issues={warnings} tone="text-amber-700 dark:text-amber-300" dataset={dataset} version={version} />
      )}
      {Object.keys(stats).length > 0 && (
        <dl className="flex flex-wrap gap-x-4 gap-y-1 text-xs">
          {Object.entries(stats).map(([k, v]) => (
            <div key={k} className="flex gap-1">
              <dt className="font-mono text-muted-foreground">{k}</dt>
              <dd>{typeof v === "object" ? JSON.stringify(v) : String(v)}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}
