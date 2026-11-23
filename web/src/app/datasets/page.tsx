"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { buttonVariants } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { listDatasets } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { DatasetListItem } from "@/lib/types";

export default function DatasetsPage() {
  const [items, setItems] = useState<DatasetListItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    listDatasets()
      .then((d) => active && setItems(d))
      .catch((e) => active && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-semibold">Datasets</h1>
        <Link href="/datasets/new" className={cn(buttonVariants())}>
          Upload dataset
        </Link>
      </div>
      {error && (
        <div role="alert" className="rounded-md border border-red-500/50 bg-red-500/10 px-3 py-2 text-sm">
          Cannot load datasets ({error})
        </div>
      )}
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Name</TableHead>
            <TableHead>Task</TableHead>
            <TableHead>Latest</TableHead>
            <TableHead>Samples</TableHead>
            <TableHead>Commercial</TableHead>
            <TableHead>Flags</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {items?.length === 0 && (
            <TableRow>
              <TableCell colSpan={6} className="text-muted-foreground">No datasets yet</TableCell>
            </TableRow>
          )}
          {items?.map((d) => (
            <TableRow key={d.id}>
              <TableCell>
                <Link href={`/datasets/${encodeURIComponent(d.name)}`} className="underline">{d.name}</Link>
              </TableCell>
              <TableCell className="font-mono">{d.task}</TableCell>
              <TableCell>{d.latest ? `v${d.latest.number}` : ""}</TableCell>
              <TableCell>{d.latest?.n_samples ?? ""}</TableCell>
              <TableCell>{d.latest ? (d.latest.commercial_ok ? "OK" : "No") : ""}</TableCell>
              <TableCell>{d.eval_only && <Badge variant="outline">eval only</Badge>}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
