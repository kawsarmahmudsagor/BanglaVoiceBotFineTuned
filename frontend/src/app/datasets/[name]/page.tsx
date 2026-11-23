"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { PageHeader, Panel } from "@/components/matra";
import { EmptyState, ErrorNotice } from "@/components/notice";
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
    <div className="space-y-5">
      <PageHeader
        title={name}
        actions={
          ds && (
            <Link href={`/datasets/new?name=${enc}&task=${ds.task}`} className={cn(buttonVariants({ size: "lg" }), "px-3")}>
              Upload new version
            </Link>
          )
        }
      >
        {ds && <span className="font-mono">{ds.task}</span>}
      </PageHeader>
      {error && <ErrorNotice>{error}</ErrorNotice>}
      {ds && (
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" className="size-4 accent-[var(--saffron)]" checked={ds.eval_only} onChange={(e) => void toggleEvalOnly(e.target.checked)} />
          Eval only (excluded from training)
        </label>
      )}
      <Panel title="Versions">
        {ds?.versions.length === 0 ? (
          <EmptyState>No versions yet.</EmptyState>
        ) : (
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead>Version</TableHead>
                <TableHead>Samples</TableHead>
                <TableHead>Created</TableHead>
                <TableHead>License</TableHead>
                <TableHead>Parent</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {ds?.versions
                .slice()
                .reverse()
                .map((v) => (
                  <TableRow key={v.id}>
                    <TableCell>
                      <Link href={`/datasets/${enc}/v/${v.number}`} className="font-mono hover:text-saffron">v{v.number}</Link>
                    </TableCell>
                    <TableCell className="font-mono">{v.n_samples}</TableCell>
                    <TableCell className="text-muted-foreground">{v.created_at ? new Date(v.created_at).toLocaleString() : ""}</TableCell>
                    <TableCell className={v.commercial_ok ? "text-status-done" : "text-status-failed"}>
                      {v.commercial_ok ? "commercial OK" : "non-commercial"}
                    </TableCell>
                    <TableCell className="font-mono">{v.parent_number !== null ? `v${v.parent_number}` : ""}</TableCell>
                  </TableRow>
                ))}
            </TableBody>
          </Table>
        )}
      </Panel>
    </div>
  );
}
