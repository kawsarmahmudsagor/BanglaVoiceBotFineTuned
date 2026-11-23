"use client";

import Link from "next/link";
import { Crown, HardDrive, Lock, LockOpen } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { MatraBar, PageHeader, Panel } from "@/components/matra";
import { MetricCharts } from "@/components/metric-charts";
import { EmptyState, ErrorNotice } from "@/components/notice";
import { StatusBadge } from "@/components/status-badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { getMetrics, getRun, getSystem, listDatasets, listModels, listRuns } from "@/lib/api";
import { mergeMetrics as mergeRunMetrics, metricsToSeries } from "@/lib/metrics";
import { sectionForTask } from "@/lib/catalog";
import { formatMetric, keyMetrics } from "@/lib/models";
import { runProgress } from "@/lib/run-filters";
import type { DatasetListItem, ModelVersion, RunDetail, RunListItem, RunMetric, RunState, SystemInfo, Task } from "@/lib/types";
import { cn } from "@/lib/utils";

const POLL_MS = 5000;
const TASKS: Task[] = ["llm-sft", "asr", "tts"];
const ACTIVE: RunState[] = ["preparing", "training", "evaluating", "exporting"];

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));

/** The running run's detail and loss curve, refreshed while it stays active. */
function useActiveRun(id: string | null) {
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [metrics, setMetrics] = useState<RunMetric[]>([]);
  const offset = useRef(0);
  useEffect(() => {
    setDetail(null);
    setMetrics([]);
    offset.current = 0;
    if (!id) return;
    let active = true;
    const tick = async () => {
      try {
        const [d, page] = await Promise.all([getRun(id), getMetrics(id, offset.current)]);
        if (!active) return;
        setDetail(d);
        if (page.items.length) setMetrics((cur) => mergeRunMetrics(cur, page.items));
        offset.current = Math.max(offset.current, page.next_offset);
      } catch {
        // The Now panel still shows the list entry; the next tick retries.
      }
    };
    void tick();
    const t = setInterval(() => void tick(), POLL_MS);
    return () => {
      active = false;
      clearInterval(t);
    };
  }, [id]);
  return { detail, metrics };
}

function DiskStatus({ disk }: { disk: SystemInfo["disk"] }) {
  const level = disk.block ? "block" : disk.warn ? "warn" : "ok";
  return (
    <div
      data-testid="disk-card"
      data-level={level}
      className={cn(
        "flex items-start gap-2 rounded-md border px-3 py-2 text-sm",
        level === "block" && "border-status-failed/60 bg-status-failed/10",
        level === "warn" && "border-saffron/60 bg-saffron/10",
        level === "ok" && "border-border",
      )}
    >
      <HardDrive className={cn("mt-0.5 size-4 shrink-0", level === "block" ? "text-status-failed" : level === "warn" ? "text-saffron" : "text-muted-foreground")} aria-hidden />
      <div>
        <div><span className="font-mono">{disk.free_gb.toFixed(1)} GB</span> free</div>
        <div className="text-[13px] text-muted-foreground">
          {level === "block" ? "Too low: uploads and new runs are blocked." : level === "warn" ? "Running low." : "OK"}
        </div>
      </div>
    </div>
  );
}

function NowPanel({ runs, system }: { runs: RunListItem[] | null; system: SystemInfo | null }) {
  const current = runs?.find((r) => ACTIVE.includes(r.state)) ?? null;
  const { detail, metrics } = useActiveRun(current?.id ?? null);
  const series = metricsToSeries(metrics);
  const lossKey = Object.keys(series).find((k) => /loss/i.test(k));
  const lock = system?.lock;
  return (
    <Panel title="Now" className="lg:col-span-2">
      <div className="space-y-4">
        {current ? (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="min-w-0">
                <Link href={`/runs/${encodeURIComponent(current.id)}`} className="block truncate font-mono text-[15px] hover:text-saffron">{current.id}</Link>
                <div className="text-sm text-muted-foreground">
                  {current.recipe} <span className="font-mono">v{current.recipe_number}</span>
                  {current.task && <> · {sectionForTask(current.task).label}</>}
                </div>
              </div>
              <StatusBadge state={detail?.state ?? current.state} />
            </div>
            <MatraBar progress={runProgress(detail?.status) ?? 0} />
            {detail?.status && (
              <div className="text-[13px] text-muted-foreground">
                {detail.status.phase} · step <span className="font-mono">{detail.status.step}{detail.status.total_steps ? ` / ${detail.status.total_steps}` : ""}</span>
              </div>
            )}
            {lossKey ? <MetricCharts series={{ [lossKey]: series[lossKey] }} compact /> : <p className="text-sm text-muted-foreground">Waiting for the first loss values.</p>}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">
            No run is using the GPU.{" "}
            {runs && runs.length > 0 && <Link href="/recipes" className="text-saffron underline">Start a run from a recipe</Link>}
          </p>
        )}
        <div className="flex flex-wrap gap-x-6 gap-y-2 border-t border-border pt-3 text-sm">
          <span><span className="font-mono">{system?.queue.queued ?? "–"}</span> queued</span>
          <span><span className="font-mono">{system?.queue.running ?? "–"}</span> running</span>
          <span className="inline-flex items-center gap-1.5">
            {lock?.holder_kind ? <Lock className="size-3.5 text-saffron" aria-hidden /> : <LockOpen className="size-3.5 text-muted-foreground" aria-hidden />}
            GPU lock:{" "}
            {lock?.holder_kind === "run" && lock.holder_id ? (
              <Link href={`/runs/${encodeURIComponent(lock.holder_id)}`} className="font-mono underline">{lock.holder_id}</Link>
            ) : lock?.holder_kind ? (
              <span className="font-mono">{`${lock.holder_kind} ${lock.holder_id ?? ""}`}</span>
            ) : (
              "free"
            )}
          </span>
        </div>
      </div>
    </Panel>
  );
}

function SystemPanel({ system }: { system: SystemInfo | null }) {
  const gpu = system?.gpu;
  return (
    <Panel title="Machine">
      {!system ? (
        <p className="text-sm text-muted-foreground">Waiting for the API…</p>
      ) : (
        <div className="space-y-3 text-sm">
          {gpu ? (
            <div className="space-y-1.5">
              <div className="font-medium">{gpu.name}</div>
              <MatraBar progress={gpu.vram_used_mb / Math.max(gpu.vram_total_mb, 1)} />
              <div className="text-[13px] text-muted-foreground">
                VRAM <span className="font-mono">{(gpu.vram_used_mb / 1024).toFixed(1)} / {(gpu.vram_total_mb / 1024).toFixed(1)} GB</span> ·
                util <span className="font-mono">{gpu.util_pct}%</span> · <span className="font-mono">{gpu.temp_c} °C</span>
              </div>
            </div>
          ) : (
            <div className="text-muted-foreground">GPU not visible</div>
          )}
          <DiskStatus disk={system.disk} />
        </div>
      )}
    </Panel>
  );
}

export default function Overview() {
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [runs, setRuns] = useState<RunListItem[] | null>(null);
  const [champions, setChampions] = useState<ModelVersion[] | null>(null);
  const [datasets, setDatasets] = useState<DatasetListItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      const [sys, rs, ms, ds] = await Promise.allSettled([getSystem(), listRuns(undefined, 10), listModels(), listDatasets()]);
      if (!active) return;
      if (sys.status === "fulfilled") setSystem(sys.value);
      if (rs.status === "fulfilled") setRuns(rs.value);
      if (ms.status === "fulfilled") setChampions(ms.value.filter((m) => m.champion));
      if (ds.status === "fulfilled") setDatasets(ds.value);
      const failed = [sys, rs, ms, ds].find((r) => r.status === "rejected");
      setError(failed && failed.status === "rejected" ? errText(failed.reason) : null);
    };
    void refresh();
    const timer = setInterval(() => void refresh(), POLL_MS);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  const recentDatasets = (datasets ?? [])
    .slice()
    .sort((a, b) => (b.latest?.created_at ?? "").localeCompare(a.latest?.created_at ?? ""))
    .slice(0, 5);

  return (
    <div className="space-y-5">
      <PageHeader title="Overview" />

      {error && (
        <ErrorNotice>Cannot reach the API ({error}). Retrying every 5 s; check that the SSH tunnel to port 8000 is up.</ErrorNotice>
      )}

      <div className="grid items-start gap-4 lg:grid-cols-3">
        <NowPanel runs={runs} system={system} />
        <SystemPanel system={system} />
      </div>

      <Panel title="Recent runs" actions={<Link href="/runs" className="text-sm text-saffron underline underline-offset-2">All runs</Link>}>
        {runs?.length === 0 ? (
          <EmptyState>
            No runs yet. Pick a model in <Link href="/catalog" className="text-saffron underline">Catalog</Link> to start a recipe.
          </EmptyState>
        ) : (
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead>Run</TableHead>
                <TableHead>Recipe</TableHead>
                <TableHead>State</TableHead>
                <TableHead>Created</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {runs?.map((r) => (
                <TableRow key={r.id}>
                  <TableCell className="max-w-44">
                    <Link href={`/runs/${encodeURIComponent(r.id)}`} className="block truncate font-mono text-[13px] hover:text-saffron" title={r.id}>{r.id}</Link>
                  </TableCell>
                  <TableCell className="max-w-48 truncate" title={r.recipe}>{r.recipe} <span className="font-mono text-muted-foreground">v{r.recipe_number}</span></TableCell>
                  <TableCell><StatusBadge state={r.state} /></TableCell>
                  <TableCell className="text-muted-foreground">{r.created_at ? new Date(r.created_at).toLocaleString() : ""}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </Panel>

      <div className="grid items-start gap-4 lg:grid-cols-2">
        <Panel title="Champions" actions={<Link href="/registry" className="text-sm text-saffron underline underline-offset-2">Registry</Link>}>
          <ul className="divide-y divide-border">
            {TASKS.map((task) => {
              const m = champions?.find((x) => x.task === task);
              return (
                <li key={task} className="flex min-w-0 flex-wrap items-center justify-between gap-2 py-2.5 first:pt-0 last:pb-0">
                  <span className="w-12 text-sm font-medium">{sectionForTask(task).label}</span>
                  {m ? (
                    <>
                      <Link href={`/registry/${m.id}`} className="inline-flex min-w-0 flex-1 items-center gap-1.5 truncate hover:text-saffron">
                        <Crown className="size-3.5 shrink-0 text-saffron" aria-hidden />
                        <span className="truncate">{m.name}</span> <span className="font-mono">v{m.number}</span>
                      </Link>
                      <span className="font-mono text-[13px] text-muted-foreground">
                        {keyMetrics(m.metrics, 2).map(([k, v]) => `${k} ${formatMetric(v)}`).join("  ")}
                      </span>
                    </>
                  ) : (
                    <span className="flex-1 text-sm text-muted-foreground">{champions ? "No champion yet" : "…"}</span>
                  )}
                </li>
              );
            })}
          </ul>
        </Panel>

        <Panel title="Recent datasets" actions={<Link href="/datasets" className="text-sm text-saffron underline underline-offset-2">All datasets</Link>}>
          {datasets?.length === 0 ? (
            <EmptyState>
              No datasets yet. <Link href="/datasets/new" className="text-saffron underline">Upload one</Link>.
            </EmptyState>
          ) : (
            <ul className="divide-y divide-border">
              {recentDatasets.map((d) => (
                <li key={d.id} className="flex min-w-0 items-center justify-between gap-2 py-2.5 first:pt-0 last:pb-0">
                  <Link href={`/datasets/${encodeURIComponent(d.name)}`} className="min-w-0 truncate hover:text-saffron" title={d.name}>{d.name}</Link>
                  <span className="shrink-0 font-mono text-[13px] text-muted-foreground">
                    {d.task}{d.latest ? ` · v${d.latest.number} · ${d.latest.n_samples} samples` : ""}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>
    </div>
  );
}
