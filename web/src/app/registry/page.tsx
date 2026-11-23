"use client";

import Link from "next/link";
import { Crown, GitCompareArrows } from "lucide-react";
import { useEffect, useState } from "react";
import { PageHeader, Panel } from "@/components/matra";
import { EmptyState, ErrorNotice } from "@/components/notice";
import { buttonVariants } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { listModels } from "@/lib/api";
import { sectionForTask } from "@/lib/catalog";
import { TASK_ORDER, formatMetric, keyMetrics, laneOrder } from "@/lib/models";
import { compareHref } from "@/lib/run-filters";
import type { ModelVersion } from "@/lib/types";
import { cn } from "@/lib/utils";

const MAX_COMPARE = 4;

export default function RegistryPage() {
  const [items, setItems] = useState<ModelVersion[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<number[]>([]);

  useEffect(() => {
    let active = true;
    listModels()
      .then((d) => active && setItems(d))
      .catch((e) => active && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      active = false;
    };
  }, []);

  const toggle = (id: number) =>
    setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : s.length >= MAX_COMPARE ? s : [...s, id]));
  const runIds = [...new Set((items ?? []).filter((m) => selected.includes(m.id)).map((m) => m.run_id))];
  const canCompare = runIds.length >= 2 && runIds.length <= MAX_COMPARE;

  return (
    <div className="space-y-5">
      <PageHeader
        title="Registry"
        actions={
          canCompare ? (
            <Link href={compareHref(runIds)} className={cn(buttonVariants({ size: "lg" }), "gap-2 px-3")}>
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
        Promoted model versions, one lane per model type. The champion is the version the voice bot serves.
      </PageHeader>
      {error && <ErrorNotice>Cannot load models ({error})</ErrorNotice>}
      {items?.length === 0 && (
        <EmptyState>No models yet. Promote a finished run from its run page.</EmptyState>
      )}
      {items &&
        items.length > 0 &&
        TASK_ORDER.map((task) => {
          const lane = laneOrder(items.filter((m) => m.task === task));
          return (
            <Panel key={task} title={sectionForTask(task).label} actions={<span className="font-mono text-[13px] text-muted-foreground">{task}</span>}>
              {lane.length === 0 ? (
                <p className="text-sm text-muted-foreground">No {sectionForTask(task).label} models promoted yet.</p>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow className="hover:bg-transparent">
                      <TableHead className="w-10"><span className="sr-only">Select for compare</span></TableHead>
                      <TableHead>Version</TableHead>
                      <TableHead>Key metrics</TableHead>
                      <TableHead>License</TableHead>
                      <TableHead>Source run</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {lane.map((m) => {
                      const checked = selected.includes(m.id);
                      return (
                        <TableRow key={m.id} data-state={checked ? "selected" : undefined} className={cn(m.champion && "bg-saffron/5")}>
                          <TableCell>
                            <input
                              type="checkbox"
                              className="size-4 accent-[var(--saffron)]"
                              aria-label={`Select ${m.name} v${m.number} for compare`}
                              checked={checked}
                              disabled={!checked && selected.length >= MAX_COMPARE}
                              onChange={() => toggle(m.id)}
                            />
                          </TableCell>
                          <TableCell className="max-w-64">
                            <Link href={`/registry/${m.id}`} className="inline-flex max-w-full items-center gap-1.5 hover:text-saffron">
                              <span className="truncate font-medium" title={m.name}>{m.name}</span>
                              <span className="font-mono">v{m.number}</span>
                            </Link>
                            {m.champion && (
                              <span className="ml-2 inline-flex items-center gap-1 text-[13px] text-saffron">
                                <Crown className="size-3.5" aria-hidden />
                                Champion
                              </span>
                            )}
                          </TableCell>
                          <TableCell className="font-mono text-[13px]">
                            {keyMetrics(m.metrics).map(([k, v]) => `${k} ${formatMetric(v)}`).join("  ") || <span className="text-muted-foreground">none</span>}
                          </TableCell>
                          <TableCell className={m.commercial_ok ? "text-status-done" : "text-status-failed"}>
                            {m.commercial_ok ? "commercial OK" : "non-commercial"}
                          </TableCell>
                          <TableCell className="max-w-40">
                            <Link href={`/runs/${encodeURIComponent(m.run_id)}`} className="block truncate font-mono text-[13px] hover:text-saffron" title={m.run_id}>{m.run_id}</Link>
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              )}
            </Panel>
          );
        })}
    </div>
  );
}
