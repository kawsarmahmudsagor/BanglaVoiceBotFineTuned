"use client";

import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { LogView } from "@/components/log-view";
import { MetricCharts } from "@/components/metric-charts";
import { PhaseTimeline } from "@/components/phase-timeline";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  ApiError, cancelRun, getLogs, getMetrics, getRun, promoteRun, resumeRun, runEventsUrl, runFileUrl,
} from "@/lib/api";
import { formatBytes, formatDuration, stateColor } from "@/lib/format";
import { LOG_LIMIT, appendLog, mergeMetrics, metricsToSeries } from "@/lib/metrics";
import type { RunDetail, RunMetric, RunState, RunStatus } from "@/lib/types";
import { cn } from "@/lib/utils";

const TERMINAL: RunState[] = ["done", "failed", "cancelled"];
const isTerminal = (s: RunState) => TERMINAL.includes(s);

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
      router.push(`/models/${m.id}`);
    });

  const series = useMemo(() => metricsToSeries(metrics), [metrics]);

  if (!run) {
    return loadError ? (
      <div role="alert" className="rounded-md border border-red-500/50 bg-red-500/10 px-3 py-2 text-sm">
        Cannot load run ({loadError})
      </div>
    ) : (
      <p className="text-sm text-muted-foreground">Loading run...</p>
    );
  }

  const status = run.status;
  const pct = status?.total_steps ? Math.min(100, (status.step / status.total_steps) * 100) : null;
  const elapsed = run.started_at
    ? formatDuration(((run.finished_at ? Date.parse(run.finished_at) : now) - Date.parse(run.started_at)) / 1000)
    : null;
  const canCancel = !terminal;
  const canResume = run.state === "failed" || run.state === "cancelled";

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="font-mono text-lg font-semibold">{run.id}</h1>
        <Badge className={cn(stateColor(run.state))}>{run.state}</Badge>
        <span className="text-sm text-muted-foreground">attempt {run.attempt}</span>
        <Badge variant={run.commercial_ok ? "secondary" : "outline"}>
          {run.commercial_ok ? "commercial OK" : "non-commercial"}
        </Badge>
        <span className="text-sm text-muted-foreground">
          {run.recipe.name} v{run.recipe.number}
          {elapsed && ` · ${elapsed}`}
        </span>
        <div className="ml-auto flex gap-2">
          {canCancel && (
            <Button variant="destructive" disabled={busy} onClick={onCancel}>
              {run.cancel_requested ? "Cancelling..." : "Cancel"}
            </Button>
          )}
          {canResume && (
            <Button disabled={busy} onClick={onResume}>Resume</Button>
          )}
          {run.state === "done" && (
            <Button disabled={busy} onClick={onPromote}>Promote</Button>
          )}
        </div>
      </div>

      {(actionError || loadError) && (
        <div role="alert" className="rounded-md border border-red-500/50 bg-red-500/10 px-3 py-2 text-sm">
          {actionError ?? loadError}
        </div>
      )}
      {run.error && (
        <div role="alert" className="whitespace-pre-wrap rounded-md border border-red-500/50 bg-red-500/10 px-3 py-2 font-mono text-xs">
          {run.error}
        </div>
      )}
      {run.oom_hint && (
        <div role="status" className="rounded-md border border-amber-500/50 bg-amber-500/10 px-3 py-2 text-sm">
          {run.oom_hint}
        </div>
      )}

      <PhaseTimeline phase={status?.phase} state={run.state} />

      {status && (
        <div className="space-y-1">
          <div className="flex justify-between text-sm">
            <span>
              {status.total_steps ? `${status.step} / ${status.total_steps} steps` : `${status.step} steps`}
              {status.message && <span className="ml-2 text-muted-foreground">{status.message}</span>}
            </span>
            {status.eta_s != null && !terminal && <span>ETA {formatDuration(status.eta_s)}</span>}
          </div>
          {pct != null && <Progress value={pct} aria-label="Progress" />}
        </div>
      )}

      <section className="space-y-2">
        <h2 className="font-semibold">Metrics</h2>
        <MetricCharts series={series} />
      </section>

      <section className="space-y-2">
        <h2 className="font-semibold">Logs</h2>
        <LogView stdout={logs.stdout} stderr={logs.stderr} />
      </section>

      {run.eval && (
        <section className="space-y-2">
          <h2 className="font-semibold">Eval metrics</h2>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Metric</TableHead>
                <TableHead>Value</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {Object.entries(run.eval.metrics).map(([k, v]) => (
                <TableRow key={k}>
                  <TableCell className="font-mono">{k}</TableCell>
                  <TableCell>{Number.isInteger(v) ? v : v.toFixed(4)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </section>
      )}

      {run.exports.length > 0 && (
        <section className="space-y-2">
          <h2 className="font-semibold">Exports</h2>
          <ul className="space-y-1 text-sm">
            {run.exports.map((x) => (
              <li key={x.path} className="flex flex-wrap items-center gap-2">
                <Badge variant="outline">{x.format}</Badge>
                <a href={runFileUrl(run.id, x.path)} download className="font-mono underline">{x.path}</a>
                <span className="text-muted-foreground">{formatBytes(x.size_bytes)}</span>
                {x.serve_with && <span className="text-muted-foreground">serve with {x.serve_with}</span>}
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
