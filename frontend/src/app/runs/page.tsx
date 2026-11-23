"use client";

import Link from "next/link";
import { GitCompareArrows } from "lucide-react";
import { Suspense, useEffect, useMemo, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { FilterCheckbox, FilterSelect } from "@/components/filter-select";
import { PageHeader, Panel } from "@/components/matra";
import { EmptyState, ErrorNotice } from "@/components/notice";
import { StatusBadge } from "@/components/status-badge";
import { buttonVariants } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { listRuns } from "@/lib/api";
import { formatDuration } from "@/lib/format";
import { applyRunFilter, compareHref, filterToQuery, queryToFilter, type RunFilter } from "@/lib/run-filters";
import { RUN_STATES, type RunListItem, type RunState, type Task } from "@/lib/types";
import { cn } from "@/lib/utils";

const POLL_MS = 5000;
const MAX_COMPARE = 4;
const TASKS: { value: Task; label: string }[] = [
  { value: "llm-sft", label: "LLM (llm-sft)" },
  { value: "asr", label: "STT (asr)" },
  { value: "tts", label: "TTS (tts)" },
];

function duration(r: RunListItem, now: number): string {
  if (!r.started_at) return "";
  const end = r.finished_at ? Date.parse(r.finished_at) : now;
  return formatDuration((end - Date.parse(r.started_at)) / 1000);
}

const uniq = (xs: (string | null)[]) => [...new Set(xs.filter((x): x is string => !!x))].sort();

function DateInput({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-[13px] text-muted-foreground">{label}</span>
      <input type="date" value={value} onChange={(e) => onChange(e.target.value)} className="h-9 rounded-md border border-border bg-card px-2 text-sm" />
    </label>
  );
}

function RunsList() {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  const filter = useMemo(() => queryToFilter(new URLSearchParams(params.toString())), [params]);
  const [items, setItems] = useState<RunListItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const [selected, setSelected] = useState<string[]>([]);

  useEffect(() => {
    let active = true;
    const load = () =>
      listRuns(undefined, 500)
        .then((d) => {
          if (!active) return;
          setItems(d);
          setError(null);
        })
        .catch((e) => active && setError(e instanceof Error ? e.message : String(e)));
    load();
    const poll = setInterval(load, POLL_MS);
    const tick = setInterval(() => setNow(Date.now()), 1000);
    return () => {
      active = false;
      clearInterval(poll);
      clearInterval(tick);
    };
  }, []);

  const update = (patch: Partial<RunFilter>) => {
    const next: RunFilter = { ...filter, ...patch };
    for (const k of Object.keys(next) as (keyof RunFilter)[]) if (!next[k]) delete next[k];
    const q = filterToQuery(next).toString();
    router.replace(q ? `${pathname}?${q}` : pathname, { scroll: false });
  };

  const rows = useMemo(() => applyRunFilter(items ?? [], filter), [items, filter]);
  const recipes = useMemo(() => uniq((items ?? []).map((r) => r.recipe)), [items]);
  const backends = useMemo(() => uniq((items ?? []).map((r) => r.backend)), [items]);
  const toggle = (id: string) =>
    setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : s.length >= MAX_COMPARE ? s : [...s, id]));
  const canCompare = selected.length >= 2 && selected.length <= MAX_COMPARE;
  const filtered = Object.keys(filter).length > 0;

  return (
    <div className="space-y-5">
      <PageHeader
        title="Runs"
        actions={
          canCompare ? (
            <Link href={compareHref(selected)} className={cn(buttonVariants({ size: "lg" }), "gap-2 px-3")}>
              <GitCompareArrows className="size-4" aria-hidden />
              Compare ({selected.length})
            </Link>
          ) : (
            <span className={cn(buttonVariants({ size: "lg", variant: "outline" }), "pointer-events-none gap-2 px-3 opacity-60")} aria-disabled="true">
              <GitCompareArrows className="size-4" aria-hidden />
              Compare ({selected.length})
            </span>
          )
        }
      >
        Select 2 to 4 runs to compare their metrics and hyperparameters.
      </PageHeader>

      <div className="flex flex-wrap items-end gap-3">
        <FilterSelect
          label="State"
          value={filter.state?.length === 1 ? filter.state[0] : ""}
          options={RUN_STATES.map((s) => ({ value: s, label: s }))}
          onChange={(v) => update({ state: v ? [v as RunState] : undefined })}
        />
        <FilterSelect label="Task" value={filter.task ?? ""} options={TASKS} onChange={(v) => update({ task: (v || undefined) as Task | undefined })} />
        <FilterSelect label="Config" value={filter.recipe ?? ""} options={recipes.map((r) => ({ value: r, label: r }))} onChange={(v) => update({ recipe: v || undefined })} />
        <FilterSelect label="Backend" value={filter.backend ?? ""} options={backends.map((b) => ({ value: b, label: b }))} onChange={(v) => update({ backend: v || undefined })} />
        <DateInput label="Created from" value={filter.from ?? ""} onChange={(v) => update({ from: v || undefined })} />
        <DateInput label="Created to" value={filter.to ?? ""} onChange={(v) => update({ to: v || undefined })} />
        <FilterCheckbox label="Commercial use only" checked={!!filter.commercialOnly} onChange={(v) => update({ commercialOnly: v || undefined })} />
        {filtered && (
          <button type="button" onClick={() => router.replace(pathname)} className="h-9 self-end px-2 text-sm text-saffron underline underline-offset-2">
            Clear filters
          </button>
        )}
      </div>

      {error && <ErrorNotice>Cannot load runs ({error})</ErrorNotice>}

      <Panel>
        {items && rows.length === 0 ? (
          <EmptyState>
            {filtered ? (
              "No runs match these filters."
            ) : (
              <>No runs yet. <Link href="/train" className="text-saffron underline">Start one from Train</Link>.</>
            )}
          </EmptyState>
        ) : (
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead className="w-10"><span className="sr-only">Select for compare</span></TableHead>
                <TableHead>Run</TableHead>
                <TableHead>State</TableHead>
                <TableHead>Config</TableHead>
                <TableHead>Task</TableHead>
                <TableHead>Backend</TableHead>
                <TableHead>License</TableHead>
                <TableHead>Created</TableHead>
                <TableHead>Duration</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((r) => {
                const checked = selected.includes(r.id);
                const full = !checked && selected.length >= MAX_COMPARE;
                return (
                  <TableRow key={r.id} data-state={checked ? "selected" : undefined}>
                    <TableCell>
                      <input
                        type="checkbox"
                        className="size-4 accent-[var(--saffron)]"
                        aria-label={`Select run ${r.id} for compare`}
                        checked={checked}
                        disabled={full}
                        title={full ? "Compare up to 4 runs at a time" : undefined}
                        onChange={() => toggle(r.id)}
                      />
                    </TableCell>
                    <TableCell className="max-w-44">
                      <Link href={`/runs/${encodeURIComponent(r.id)}`} className="block truncate font-mono text-[13px] hover:text-saffron" title={r.id}>{r.id}</Link>
                    </TableCell>
                    <TableCell><StatusBadge state={r.state} /></TableCell>
                    <TableCell className="max-w-48 truncate" title={`${r.recipe} v${r.recipe_number}`}>
                      {r.recipe} <span className="font-mono text-muted-foreground">v{r.recipe_number}</span>
                    </TableCell>
                    <TableCell className="font-mono text-[13px]">{r.task ?? ""}</TableCell>
                    <TableCell className="font-mono text-[13px]">{r.backend ?? ""}</TableCell>
                    <TableCell className={r.commercial_ok ? "text-status-done" : "text-status-failed"}>
                      {r.commercial_ok ? "commercial OK" : "non-commercial"}
                    </TableCell>
                    <TableCell className="text-muted-foreground">{r.created_at ? new Date(r.created_at).toLocaleString() : ""}</TableCell>
                    <TableCell className="font-mono text-[13px]">{duration(r, now)}</TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        )}
      </Panel>
    </div>
  );
}

export default function RunsPage() {
  return (
    <Suspense fallback={<p className="text-sm text-muted-foreground">Loading runs…</p>}>
      <RunsList />
    </Suspense>
  );
}
