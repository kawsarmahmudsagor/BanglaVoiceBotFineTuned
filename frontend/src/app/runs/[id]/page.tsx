"use client";

import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { EvalSamples } from "@/components/eval-samples";
import { LogView } from "@/components/log-view";
import { PageHeader, Panel } from "@/components/matra";
import { MetricCharts } from "@/components/metric-charts";
import { ErrorNotice, WarningNotice } from "@/components/notice";
import { PreparedCard } from "@/components/prepared-card";
import { PhaseTimeline } from "@/components/phase-timeline";
import { StatusBadge } from "@/components/status-badge";
import { Tabs } from "@/components/tabs";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  ApiError, cancelRun, getLogs, getMetrics, getPrepared, getRecipeVersion, getRun, promoteRun, resumeRun, runEventsUrl, runExportFileUrl,
} from "@/lib/api";
import { formatBytes, formatDuration } from "@/lib/format";
import { LOG_LIMIT, appendLog, mergeMetrics, metricRows, metricsToSeries } from "@/lib/metrics";
import { runProgress } from "@/lib/run-filters";
import type { PreparedRef, RunDetail, RunMetric, RunState, RunStatus } from "@/lib/types";

const TERMINAL: RunState[] = ["done", "failed", "cancelled"];
const isTerminal = (s: RunState) => TERMINAL.includes(s);

type TabKey = "metrics" | "logs" | "config" | "eval" | "artifacts";

const fmtMetric = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(4));

const errText = (e: unknown) => (e instanceof ApiError ? e.detail : e instanceof Error ? e.message : String(e));

export default function RunDetailPage() {
  const { id } = useParams<{ id: string }>();
  // keyed by id so run/metrics/logs state never leaks across routes
  return <RunDetailView key={id} id={id} />;
}

function RunDetailView({ id }: { id: string }) {
  const router = useRouter();
  const [run, setRun] = useState<RunDetail | null>(null);
  const [metrics, setMetrics] = useState<RunMetric[]>([]);
  const [logs, setLogs] = useState<{ stdout: string[]; stderr: string[] }>({ stdout: [], stderr: [] });
  const [loadError, setLoadError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [now, setNow] = useState(() => Date.now());
  const [tab, setTab] = useState<TabKey>("metrics");
  const [config, setConfig] = useState<{ yaml: string } | { error: string } | null>(null);

  const [prepared, setPrepared] = useState<PreparedRef[] | null>(null);
  const [preparedError, setPreparedError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const r = await getRun(id);
    setRun(r);
    return r;
  }, [id]);

  // Initial load: run detail, last LOG_LIMIT log lines, metrics from offset 0 following next_offset.
  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const [r, l] = await Promise.all([getRun(id), getLogs(id, LOG_LIMIT)]);
        if (!active) return;
        setRun(r);
        setLogs({ stdout: l.stdout.slice(-LOG_LIMIT), stderr: l.stderr.slice(-LOG_LIMIT) });
        let offset = 0;
        for (;;) {
          const page = await getMetrics(id, offset);
          if (!active) return;
          if (page.items.length) setMetrics((cur) => mergeMetrics(cur, page.items));
          if (page.next_offset <= offset) break;
          offset = page.next_offset;
        }
      } catch (e) {
        if (active) setLoadError(errText(e));
      }
    })();
    return () => {
      active = false;
    };
  }, [id]);

  // Live updates: only while the run is non-terminal. Re-opens when a resume flips it back to queued.
  const terminal = run ? isTerminal(run.state) : true;
  const loaded = run !== null;
  useEffect(() => {
    if (!loaded || terminal) return;
    const es = new EventSource(runEventsUrl(id));
    // The server replays every metric and log line from byte 0 on each (re)connection, so:
    //  - metrics are de-duplicated by (split, name, step, ts) in mergeMetrics;
    //  - for logs, the first line of each stream per connection replaces the buffer (replay refills it).
    let fresh = { stdout: true, stderr: true };
    es.onopen = () => {
      fresh = { stdout: true, stderr: true };
    };
    const parse = <T,>(e: MessageEvent): T | null => {
      try {
        return JSON.parse(e.data as string) as T;
      } catch {
        return null;
      }
    };
    es.addEventListener("state", (e) => {
      const d = parse<{ state: RunState }>(e as MessageEvent);
      if (!d) return;
      setRun((r) => (r ? { ...r, state: d.state } : r));
      if (isTerminal(d.state)) {
        es.close();
        refresh().catch((err) => setLoadError(errText(err)));
      }
    });
    es.addEventListener("status", (e) => {
      const d = parse<RunStatus>(e as MessageEvent);
      if (d) setRun((r) => (r ? { ...r, status: d } : r));
    });
    es.addEventListener("metric", (e) => {
      const d = parse<RunMetric>(e as MessageEvent);
      if (d) setMetrics((cur) => mergeMetrics(cur, [d]));
    });
    es.addEventListener("log", (e) => {
      const d = parse<{ stream: "stdout" | "stderr"; line: string }>(e as MessageEvent);
      if (!d) return;
      const replace = fresh[d.stream];
      fresh[d.stream] = false;
      setLogs((cur) => ({ ...cur, [d.stream]: replace ? [d.line] : appendLog(cur[d.stream], d.line) }));
    });
    return () => es.close();
  }, [id, loaded, terminal, refresh]);

  // Live elapsed time while running.
  const running = run !== null && !terminal;
  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [running]);

  const busyRef = useRef(false);
  const act = async (fn: () => Promise<unknown>) => {
    if (busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    setActionError(null);
    try {
      await fn();
    } catch (e) {
      setActionError(errText(e));
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  };

  const onCancel = () => act(async () => { await cancelRun(id); await refresh(); });
  const onResume = () => act(async () => { await resumeRun(id); await refresh(); });
  const onPromote = () =>
    act(async () => {
      const m = await promoteRun(id);
      router.push(`/registry/${m.id}`);
    });

  const series = useMemo(() => metricsToSeries(metrics), [metrics]);

  // The recipe version this run used, loaded once when the Config tab is first opened.
  const recipeName = run?.recipe.name;
  const recipeNumber = run?.recipe.number;
  useEffect(() => {
    if (tab !== "config" || config || !recipeName || recipeNumber === undefined) return;
    getRecipeVersion(recipeName, recipeNumber)
      .then((v) => setConfig({ yaml: v.yaml }))
      .catch((e) => setConfig({ error: errText(e) }));
  }, [tab, config, recipeName, recipeNumber]);

  // Prepared-data refs: refetched on each Config tab open and whenever the run state changes (refs appear after prepare).
  const runState = run?.state;
  useEffect(() => {
    if (tab !== "config") return;
    let active = true;
    getPrepared(id)
      .then((p) => { if (active) { setPrepared(p.refs); setPreparedError(null); } })
      .catch((e) => { if (active) setPreparedError(errText(e)); });
    return () => { active = false; };
  }, [tab, id, runState]);

  if (!run) {
    return loadError ? (
      <div className="space-y-4">
        <PageHeader title={id} />
        <ErrorNotice>Cannot load run ({loadError})</ErrorNotice>
      </div>
    ) : (
      <PageHeader title={<span className="font-mono">{id}</span>}>Loading run…</PageHeader>
    );
  }

  const status = run.status;
  const elapsed = run.started_at
    ? formatDuration(((run.finished_at ? Date.parse(run.finished_at) : now) - Date.parse(run.started_at)) / 1000)
    : null;
  const canCancel = !terminal;
  const canResume = run.state === "failed" || run.state === "cancelled";
  const progress = run.state === "done" ? 1 : terminal ? undefined : runProgress(status);
  const evalCount = run.eval ? Object.keys(run.eval.metrics).length : 0;

  return (
    <div className="space-y-5">
      <PageHeader
        title={<span className="font-mono text-[22px]">{run.id}</span>}
        progress={progress}
        actions={
          <>
            {canCancel && (
              <Button variant="destructive" size="lg" className="px-3" disabled={busy} onClick={onCancel}>
                {run.cancel_requested ? "Cancelling…" : "Cancel"}
              </Button>
            )}
            {canResume && <Button size="lg" className="px-3" disabled={busy} onClick={onResume}>Resume</Button>}
            {run.state === "done" && <Button size="lg" className="px-3" disabled={busy} onClick={onPromote}>Promote to registry</Button>}
          </>
        }
      >
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
          <StatusBadge state={run.state} />
          <Link href={`/recipes/${encodeURIComponent(run.recipe.name)}`} className="hover:text-saffron">
            {run.recipe.name} <span className="font-mono">v{run.recipe.number}</span>
          </Link>
          <span>attempt <span className="font-mono">{run.attempt}</span></span>
          <span className={run.commercial_ok ? "text-status-done" : "text-status-failed"}>
            {run.commercial_ok ? "commercial OK" : "non-commercial"}
          </span>
          {elapsed && <span className="font-mono">{elapsed}</span>}
        </div>
      </PageHeader>

      {(actionError || loadError) && <ErrorNotice>{actionError ?? loadError}</ErrorNotice>}
      {run.error && <ErrorNotice className="font-mono text-xs whitespace-pre-wrap">{run.error}</ErrorNotice>}
      {run.oom_hint && <WarningNotice>{run.oom_hint}</WarningNotice>}

      <PhaseTimeline phase={status?.phase} state={run.state} />

      {status && !terminal && (
        <div className="flex flex-wrap justify-between gap-2 text-sm">
          <span>
            <span className="font-mono">{status.total_steps ? `${status.step} / ${status.total_steps}` : status.step}</span> steps
            {status.message && <span className="ml-2 text-muted-foreground">{status.message}</span>}
          </span>
          {status.eta_s != null && !terminal && <span>ETA <span className="font-mono">{formatDuration(status.eta_s)}</span></span>}
        </div>
      )}

      <Tabs<TabKey>
        label="Run details"
        value={tab}
        onChange={setTab}
        tabs={[
          { key: "metrics", label: "Metrics" },
          { key: "logs", label: "Logs" },
          { key: "config", label: "Config" },
          { key: "eval", label: <>Eval {evalCount > 0 && <span className="font-mono text-[12px] text-muted-foreground">{evalCount}</span>}</> },
          { key: "artifacts", label: <>Artifacts {run.exports.length > 0 && <span className="font-mono text-[12px] text-muted-foreground">{run.exports.length}</span>}</> },
        ]}
      >
        {tab === "metrics" && <MetricCharts series={series} />}
        {tab === "logs" && <LogView stdout={logs.stdout} stderr={logs.stderr} />}
        {tab === "config" && (
          <div className="space-y-4">
          <Panel title={`${run.recipe.name} v${run.recipe.number}`}>
            {config === null && <p className="text-sm text-muted-foreground">Loading recipe…</p>}
            {config && "error" in config && <ErrorNotice>Cannot load the recipe version ({config.error})</ErrorNotice>}
            {config && "yaml" in config && (
              <pre aria-label="Recipe YAML" className="overflow-auto rounded-md bg-ink p-3 font-mono text-[13px] leading-relaxed">{config.yaml}</pre>
            )}
          </Panel>
          {preparedError && <ErrorNotice>Cannot load prepared data ({preparedError})</ErrorNotice>}
          {!preparedError && <PreparedCard refs={prepared ?? []} />}
          </div>
        )}
        {tab === "eval" &&
          (run.eval ? (
            <div className="space-y-4">
              <Panel title="Eval metrics">
                <Table>
                  <TableHeader>
                    <TableRow className="hover:bg-transparent">
                      <TableHead>Metric</TableHead>
                      <TableHead>Value</TableHead>
                      <TableHead>base/</TableHead>
                      <TableHead>Delta</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {metricRows(run.eval.metrics).map((m) => (
                      <TableRow key={m.name}>
                        <TableCell className="font-mono">{m.name}</TableCell>
                        <TableCell className="font-mono">{fmtMetric(m.value)}</TableCell>
                        <TableCell className="font-mono">{m.base === undefined ? "" : fmtMetric(m.base)}</TableCell>
                        <TableCell className="font-mono">{m.delta === undefined ? "" : `${m.delta > 0 ? "+" : ""}${fmtMetric(m.delta)}`}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Panel>
              <EvalSamples runId={run.id} task={run.task ?? null} />
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No eval results yet. They appear after the eval phase finishes.</p>
          ))}
        {tab === "artifacts" && (
          <div className="space-y-4">
            <Panel title="Exports">
              {run.exports.length === 0 ? (
                <p className="text-sm text-muted-foreground">No exports yet. They appear after the export phase finishes.</p>
              ) : (
                <ul className="space-y-2 text-sm">
                  {run.exports.map((x) => (
                    <li key={x.path} className="flex min-w-0 flex-wrap items-center gap-2">
                      <span className="rounded-md bg-raised px-2 py-0.5 font-mono text-[13px]">{x.format}</span>
                      <a href={runExportFileUrl(run.id, x.path)} download className="min-w-0 truncate font-mono underline" title={x.path}>{x.path}</a>
                      <span className="font-mono text-muted-foreground">{formatBytes(x.size_bytes)}</span>
                      {x.serve_with && <span className="text-muted-foreground">serve with {x.serve_with}</span>}
                    </li>
                  ))}
                </ul>
              )}
            </Panel>
            <Panel title="Provenance">
              <dl className="grid gap-3 text-sm sm:grid-cols-2">
                <div className="min-w-0"><dt className="text-[13px] text-muted-foreground">Run directory</dt><dd className="font-mono text-[13px] break-all">{run.run_dir}</dd></div>
                <div className="min-w-0"><dt className="text-[13px] text-muted-foreground">Git commit</dt><dd className="font-mono text-[13px] break-all">{run.git_sha ?? "unknown"}</dd></div>
                <div className="min-w-0"><dt className="text-[13px] text-muted-foreground">Lockfile hash</dt><dd className="font-mono text-[13px] break-all">{run.lock_hash ?? "unknown"}</dd></div>
                <div className="min-w-0"><dt className="text-[13px] text-muted-foreground">MLflow run</dt><dd className="font-mono text-[13px] break-all">{run.mlflow_run_id ?? "not logged"}</dd></div>
                {Object.entries(run.dataset_hashes).map(([k, v]) => (
                  <div key={k} className="min-w-0"><dt className="text-[13px] text-muted-foreground">Dataset {k}</dt><dd className="font-mono text-[13px] break-all">{v}</dd></div>
                ))}
              </dl>
            </Panel>
          </div>
        )}
      </Tabs>
    </div>
  );
}
