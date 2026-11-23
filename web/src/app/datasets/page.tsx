"use client";

import Link from "next/link";
import { Upload } from "lucide-react";
import { useEffect, useState } from "react";
import { PageHeader, Panel } from "@/components/matra";
import { EmptyState, ErrorNotice } from "@/components/notice";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useUploadDrawer } from "@/components/upload-drawer";
import { listDatasets } from "@/lib/api";
import { sectionForTask } from "@/lib/catalog";
import type { DatasetListItem } from "@/lib/types";

export default function DatasetsPage() {
  const [items, setItems] = useState<DatasetListItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { openUpload } = useUploadDrawer();

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
    <div className="space-y-5">
      <PageHeader
        title="Datasets"
        actions={
          <Button size="lg" className="gap-2 px-3" onClick={() => openUpload()}>
            <Upload className="size-4" aria-hidden />
            Upload dataset
          </Button>
        }
      >
        Versioned training and eval data. Each upload is validated and becomes a new immutable version.
      </PageHeader>
      {error && <ErrorNotice>Cannot load datasets ({error})</ErrorNotice>}
      <Panel>
        {items?.length === 0 ? (
          <EmptyState>
            No datasets yet. Check a model&apos;s training-data format in the{" "}
            <Link href="/catalog" className="text-saffron underline">catalog</Link>, then upload.
          </EmptyState>
        ) : (
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead>Name</TableHead>
                <TableHead>Task</TableHead>
                <TableHead>Latest</TableHead>
                <TableHead>Samples</TableHead>
                <TableHead>License</TableHead>
                <TableHead>Use</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {items?.map((d) => (
                <TableRow key={d.id}>
                  <TableCell className="max-w-64">
                    <Link href={`/datasets/${encodeURIComponent(d.name)}`} className="block truncate font-medium hover:text-saffron" title={d.name}>{d.name}</Link>
                  </TableCell>
                  <TableCell>
                    {sectionForTask(d.task).label} <span className="font-mono text-[13px] text-muted-foreground">{d.task}</span>
                  </TableCell>
                  <TableCell className="font-mono">{d.latest ? `v${d.latest.number}` : ""}</TableCell>
                  <TableCell className="font-mono">{d.latest?.n_samples ?? ""}</TableCell>
                  <TableCell className={d.latest ? (d.latest.commercial_ok ? "text-status-done" : "text-status-failed") : ""}>
                    {d.latest ? (d.latest.commercial_ok ? "commercial OK" : "non-commercial") : ""}
                  </TableCell>
                  <TableCell className="text-muted-foreground">{d.eval_only ? "Eval only" : "Train and eval"}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </Panel>
    </div>
  );
}
