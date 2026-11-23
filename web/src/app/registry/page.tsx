"use client";

import Link from "next/link";
import { Suspense, useEffect, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { listModels } from "@/lib/api";
import { TASK_ORDER, formatMetric, groupByTask, keyMetrics } from "@/lib/models";
import type { ModelVersion, Task } from "@/lib/types";

function ModelsList() {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  const raw = params.get("task") ?? "";
  const task = (TASK_ORDER as string[]).includes(raw) ? (raw as Task) : undefined;
  const [items, setItems] = useState<ModelVersion[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    listModels(task)
      .then((d) => {
        if (!active) return;
        setItems(d);
        setError(null);
      })
      .catch((e) => active && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      active = false;
    };
  }, [task]);

  const onFilter = (value: string) =>
    router.replace(value ? `${pathname}?task=${encodeURIComponent(value)}` : pathname);
  const groups = items ? groupByTask(items) : [];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-semibold">Models</h1>
        <label className="flex items-center gap-2 text-sm">
          Task
          <select
            aria-label="Task filter"
            value={task ?? ""}
            onChange={(e) => onFilter(e.target.value)}
            className="h-8 rounded-md border bg-background px-2"
          >
            <option value="">All</option>
            {TASK_ORDER.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </label>
      </div>
      {error && (
        <div role="alert" className="rounded-md border border-red-500/50 bg-red-500/10 px-3 py-2 text-sm">
          Cannot load models ({error})
        </div>
      )}
      {items?.length === 0 && <p className="text-sm text-muted-foreground">No models yet. Promote a finished run.</p>}
      {groups.map((g) => (
        <section key={g.task} className="space-y-2">
          <h2 className="font-mono text-sm font-medium">{g.task}</h2>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>Version</TableHead>
                <TableHead>Base model</TableHead>
                <TableHead>Key metrics</TableHead>
                <TableHead>Commercial</TableHead>
                <TableHead>Champion</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {g.models.map((m) => (
                <TableRow key={m.id}>
                  <TableCell>
                    <Link href={`/registry/${m.id}`} className="underline">{m.name}</Link>
                  </TableCell>
                  <TableCell>v{m.number}</TableCell>
                  <TableCell className="font-mono">{m.base_model}</TableCell>
                  <TableCell className="font-mono text-xs">
                    {keyMetrics(m.metrics).map(([k, v]) => `${k}=${formatMetric(v)}`).join("  ")}
                  </TableCell>
                  <TableCell>
                    <Badge variant={m.commercial_ok ? "secondary" : "outline"}>
                      {m.commercial_ok ? "commercial OK" : "non-commercial"}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    {m.champion && <span role="img" aria-label="champion" title="champion">★</span>}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </section>
      ))}
    </div>
  );
}

export default function ModelsPage() {
  return (
    <Suspense fallback={<p className="text-sm text-muted-foreground">Loading models...</p>}>
      <ModelsList />
    </Suspense>
  );
}
