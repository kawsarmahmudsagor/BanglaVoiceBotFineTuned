"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useState } from "react";
import { CompareCharts, RunLegend } from "@/components/compare-charts";
import { HparamDiff } from "@/components/hparam-diff";
import { PageHeader, Panel } from "@/components/matra";
import { ErrorNotice, WarningNotice } from "@/components/notice";
import { StatusBadge } from "@/components/status-badge";
import { ApiError, getMetrics, getRecipeVersion, getRun, listRuns } from "@/lib/api";
import { bestRun, diffHparams, mergeMetrics, metricDirection, parseCompareIds } from "@/lib/compare";
import type { RunDetail, RunMetric } from "@/lib/types";
import { cn } from "@/lib/utils";

const errText = (e: unknown) => (e instanceof ApiError ? e.detail : e instanceof Error ? e.message : String(e));

interface Loaded {
  run: RunDetail | null;
  metrics: RunMetric[];
  spec: Record<string, unknown> | null;
  problems: string[];
}

async function allMetrics(id: string): Promise<RunMetric[]> {
  const out: RunMetric[] = [];
  let offset = 0;
  for (;;) {
    const page = await getMetrics(id, offset);
    out.push(...page.items);
    if (page.next_offset <= offset) return out;
    offset = page.next_offset;
  }
}

/** Load one run; each part can fail on its own without hiding the others. */
async function loadRun(id: string): Promise<Loaded> {
  const problems: string[] = [];
  const [run, metrics] = await Promise.allSettled([getRun(id), allMetrics(id)]);
  const detail = run.status === "fulfilled" ? run.value : null;
  if (run.status === "rejected") problems.push(errText(run.reason));
  if (metrics.status === "rejected") problems.push(`metrics: ${errText(metrics.reason)}`);
  let spec: Record<string, unknown> | null = null;
  if (detail) {
    try {
      spec = (await getRecipeVersion(detail.recipe.name, detail.recipe.number)).spec as Record<string, unknown>;
    } catch (e) {
      problems.push(`training config: ${errText(e)}`);
    }
  }
  return { run: detail, metrics: metrics.status === "fulfilled" ? metrics.value : [], spec, problems };
}

function fmt(v: number) {
  return Number.isInteger(v) ? String(v) : v.toFixed(4);
}

function CompareView() {
  const params = useSearchParams();
  const raw = params.get("ids");
  const [known, setKnown] = useState<Set<string> | null>(null);
  const [knownError, setKnownError] = useState<string | null>(null);
  const [data, setData] = useState<Record<string, Loaded> | null>(null);

  useEffect(() => {
    let active = true;
    listRuns(undefined, 500)
      .then((runs) => active && setKnown(new Set(runs.map((r) => r.id))))
      .catch((e) => active && setKnownError(errText(e)));
    return () => {
      active = false;
    };
  }, []);

  // Without the run list, still try the ids; unknown ones surface as per-run load errors.
  const parsed = useMemo(() => parseCompareIds(raw, known ?? undefined), [raw, known]);
  const ids = "ids" in parsed ? parsed.ids : null;
  const idsKey = ids?.join(",") ?? "";

  useEffect(() => {
    if (!idsKey || (!known && !knownError)) return;
    let active = true;
    setData(null);
    const list = idsKey.split(",");
    Promise.all(list.map(loadRun)).then((all) => {
      if (active) setData(Object.fromEntries(list.map((id, i) => [id, all[i]])));
    });
    return () => {
      active = false;
    };
  }, [idsKey, known, knownError]);

  const merged = useMemo(
    () => (data ? mergeMetrics(Object.fromEntries(Object.entries(data).map(([id, d]) => [id, d.metrics]))) : {}),
    [data],
  );
  const diff = useMemo(
    () => (data ? diffHparams(Object.fromEntries(Object.entries(data).map(([id, d]) => [id, d.spec]))) : []),
    [data],
  );
  const evalKeys = useMemo(
    () => [...new Set(Object.values(data ?? {}).flatMap((d) => Object.keys(d.run?.eval?.metrics ?? {})))].sort(),
    [data],
  );

  if ("error" in parsed) {
    return (
      <div className="space-y-4">
        <PageHeader title="Compare runs" />
        <ErrorNotice>
          {parsed.error} <Link href="/runs" className="underline">Pick runs on the Runs page</Link>.
        </ErrorNotice>
      </div>
    );
  }

  const runIds = parsed.ids;
  return (
    <div className="space-y-5">
      <PageHeader title="Compare runs">
        <RunLegend runIds={runIds} />
      </PageHeader>
      {knownError && <WarningNotice>Cannot check the run list ({knownError}); loading the runs anyway.</WarningNotice>}
      {data &&
        Object.entries(data)
          .filter(([, d]) => d.problems.length > 0)
          .map(([id, d]) => (
            <ErrorNotice key={id}>
              Could not load <span className="font-mono">{id}</span>: {d.problems.join("; ")}
            </ErrorNotice>
          ))}
      {!data ? (
        <p className="text-sm text-muted-foreground">Loading runs…</p>
      ) : (
        <>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            {runIds.map((id) => {
              const r = data[id]?.run;
              return (
                <div key={id} className="min-w-0 rounded-[10px] border border-border bg-card p-3">
                  <Link href={`/runs/${encodeURIComponent(id)}`} className="block truncate font-mono text-[13px] hover:text-saffron" title={id}>{id}</Link>
                  {r ? (
                    <div className="mt-1 space-y-0.5 text-sm">
                      <StatusBadge state={r.state} />
                      <div className="truncate text-muted-foreground" title={`${r.recipe.name} v${r.recipe.number}`}>
                        {r.recipe.name} <span className="font-mono">v{r.recipe.number}</span>
                      </div>
                    </div>
                  ) : (
                    <p className="mt-1 text-sm text-muted-foreground">No data</p>
                  )}
                </div>
              );
            })}
          </div>

          <Panel title="Metrics">
            <CompareCharts merged={merged} runIds={runIds} />
          </Panel>

          <Panel title="Hyperparameters">
            {diff.length ? <HparamDiff rows={diff} runIds={runIds} /> : <p className="text-sm text-muted-foreground">No training configs could be loaded.</p>}
          </Panel>

          <Panel title="Final eval metrics">
            {evalKeys.length === 0 ? (
              <p className="text-sm text-muted-foreground">None of these runs has eval results yet.</p>
            ) : (
              <div className="overflow-x-auto rounded-md border border-border">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border">
                      <th scope="col" className="h-9 px-3 text-left text-[13px] font-medium text-muted-foreground">Metric</th>
                      {runIds.map((id) => (
                        <th key={id} scope="col" className="h-9 px-3 text-left font-mono text-[13px] font-medium text-muted-foreground">{id}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {evalKeys.map((k) => {
                      const dir = metricDirection(k);
                      const values = Object.fromEntries(runIds.map((id) => [id, data[id]?.run?.eval?.metrics[k]]));
                      const best = bestRun(values, dir);
                      return (
                        <tr key={k} className="border-b border-border last:border-0">
                          <th scope="row" className="px-3 py-1.5 text-left font-mono text-[13px] font-normal">
                            {k} <span className="font-sans text-[12px] text-muted-foreground">({dir === "min" ? "lower is better" : "higher is better"})</span>
                          </th>
                          {runIds.map((id) => {
                            const v = values[id];
                            const isBest = best === id;
                            return (
                              <td key={id} className={cn("px-3 py-1.5 font-mono text-[13px]", isBest && "font-semibold")}>
                                {v === undefined ? (
                                  <span className="font-sans text-muted-foreground">No data</span>
                                ) : (
                                  <>
                                    {fmt(v)}
                                    {isBest && <span className="ml-2 rounded bg-status-done/15 px-1.5 py-0.5 font-sans text-[12px] text-status-done">Best</span>}
                                  </>
                                )}
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </Panel>
        </>
      )}
    </div>
  );
}

export default function ComparePage() {
  return (
    <Suspense fallback={null}>
      <CompareView />
    </Suspense>
  );
}
