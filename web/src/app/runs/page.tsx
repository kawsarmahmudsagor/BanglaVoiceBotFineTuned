"use client";

import Link from "next/link";
import { Suspense, useEffect, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { listRecipes, listRuns } from "@/lib/api";
import { formatDuration, stateColor } from "@/lib/format";
import { RUN_STATES, type RunListItem } from "@/lib/types";
import { cn } from "@/lib/utils";

const POLL_MS = 5000;

function duration(r: RunListItem, now: number): string {
  if (!r.started_at) return "";
  const end = r.finished_at ? Date.parse(r.finished_at) : now;
  return formatDuration((end - Date.parse(r.started_at)) / 1000);
}

function RunsList() {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  const state = params.get("state") ?? "";
  const [items, setItems] = useState<RunListItem[] | null>(null);
  const [backends, setBackends] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    let active = true;
    const load = () =>
      listRuns(state || undefined)
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
  }, [state]);

  // The list endpoint carries no backend; use each recipe's latest version's backend.
  useEffect(() => {
    let active = true;
    listRecipes()
      .then((rs) => active && setBackends(Object.fromEntries(rs.map((r) => [r.name, r.latest.backend]))))
      .catch(() => {});
    return () => {
      active = false;
    };
  }, []);

  const onFilter = (value: string) =>
    router.replace(value ? `${pathname}?state=${encodeURIComponent(value)}` : pathname);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-semibold">Runs</h1>
        <label className="flex items-center gap-2 text-sm">
          State
          <select
            aria-label="State filter"
            value={state}
            onChange={(e) => onFilter(e.target.value)}
            className="h-8 rounded-md border bg-background px-2"
          >
            <option value="">All</option>
            {RUN_STATES.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </label>
      </div>
      {error && (
        <div role="alert" className="rounded-md border border-red-500/50 bg-red-500/10 px-3 py-2 text-sm">
          Cannot load runs ({error})
        </div>
      )}
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Run</TableHead>
            <TableHead>State</TableHead>
            <TableHead>Recipe</TableHead>
            <TableHead>Backend</TableHead>
            <TableHead>Commercial</TableHead>
            <TableHead>Started</TableHead>
            <TableHead>Duration</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {items?.length === 0 && (
            <TableRow>
              <TableCell colSpan={7} className="text-muted-foreground">No runs</TableCell>
            </TableRow>
          )}
          {items?.map((r) => (
            <TableRow key={r.id}>
              <TableCell>
                <Link href={`/runs/${encodeURIComponent(r.id)}`} className="font-mono underline">{r.id}</Link>
              </TableCell>
              <TableCell><Badge className={cn(stateColor(r.state))}>{r.state}</Badge></TableCell>
              <TableCell>{r.recipe}</TableCell>
              <TableCell className="font-mono">{backends[r.recipe] ?? ""}</TableCell>
              <TableCell>
                <Badge variant={r.commercial_ok ? "secondary" : "outline"}>
                  {r.commercial_ok ? "commercial OK" : "non-commercial"}
                </Badge>
              </TableCell>
              <TableCell>{r.started_at ? new Date(r.started_at).toLocaleString() : ""}</TableCell>
              <TableCell>{duration(r, now)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

export default function RunsPage() {
  return (
    <Suspense fallback={<p className="text-sm text-muted-foreground">Loading runs...</p>}>
      <RunsList />
    </Suspense>
  );
}
