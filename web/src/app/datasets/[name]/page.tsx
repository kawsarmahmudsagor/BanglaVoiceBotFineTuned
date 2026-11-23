"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { buttonVariants } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { getDataset, patchDataset } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { DatasetDetail } from "@/lib/types";

export default function DatasetPage() {
  const { name: rawName } = useParams<{ name: string }>();
  const name = decodeURIComponent(rawName);
  const [ds, setDs] = useState<DatasetDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    getDataset(name)
      .then((d) => active && setDs(d))
      .catch((e) => active && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      active = false;
    };
  }, [name]);

  async function toggleEvalOnly(value: boolean) {
    setError(null);
    try {
      const updated = await patchDataset(name, value);
      setDs((d) => (d ? { ...d, eval_only: updated.eval_only } : d));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  const enc = encodeURIComponent(name);
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-semibold">
          {name} {ds && <span className="font-mono text-sm text-muted-foreground">{ds.task}</span>}
        </h1>
        {ds && (
          <Link href={`/datasets/new?name=${enc}&task=${ds.task}`} className={cn(buttonVariants())}>
            Upload new version
          </Link>
        )}
      </div>
      {error && (
        <div role="alert" className="rounded-md border border-red-500/50 bg-red-500/10 px-3 py-2 text-sm">
          {error}
        </div>
      )}
      {ds && (
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={ds.eval_only} onChange={(e) => void toggleEvalOnly(e.target.checked)} />
          Eval only (excluded from training)
        </label>
      )}
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Version</TableHead>
            <TableHead>Samples</TableHead>
            <TableHead>Created</TableHead>
            <TableHead>Commercial</TableHead>
            <TableHead>Parent</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {ds?.versions.length === 0 && (
            <TableRow>
              <TableCell colSpan={5} className="text-muted-foreground">No versions yet</TableCell>
            </TableRow>
          )}
          {ds?.versions
            .slice()
            .reverse()
            .map((v) => (
              <TableRow key={v.id}>
                <TableCell>
                  <Link href={`/datasets/${enc}/v/${v.number}`} className="underline">v{v.number}</Link>
                </TableCell>
                <TableCell>{v.n_samples}</TableCell>
                <TableCell className="text-muted-foreground">
                  {v.created_at ? new Date(v.created_at).toLocaleString() : ""}
                </TableCell>
                <TableCell>{v.commercial_ok ? "OK" : "No"}</TableCell>
                <TableCell>{v.parent_number !== null ? `v${v.parent_number}` : ""}</TableCell>
              </TableRow>
            ))}
        </TableBody>
      </Table>
    </div>
  );
}
