"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Panel } from "@/components/matra";
import { ErrorNotice } from "@/components/notice";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { getCheckpoints } from "@/lib/api";
import type { CheckpointItem } from "@/lib/types";

const errMsg = (e: unknown) => (e instanceof Error ? e.message : String(e));

export function CheckpointsPanel({ runId }: { runId: string }) {
  const router = useRouter();
  const [items, setItems] = useState<CheckpointItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setItems(null);
    setError(null);
    getCheckpoints(runId)
      .then((r) => active && setItems(r.items))
      .catch((e) => active && setError(errMsg(e)));
    return () => {
      active = false;
    };
  }, [runId]);

  if (error) return <ErrorNotice>Cannot load checkpoints ({error})</ErrorNotice>;
  if (!items) return <p className="text-sm text-muted-foreground">Loading checkpoints…</p>;
  if (items.length === 0) return <p className="text-sm text-muted-foreground">No checkpoints yet. They appear while the run trains.</p>;
  return (
    <Panel title="Checkpoints">
      <Table>
        <TableHeader>
          <TableRow className="hover:bg-transparent">
            <TableHead>Name</TableHead>
            <TableHead>Step</TableHead>
            <TableHead>Status</TableHead>
            <TableHead><span className="sr-only">Actions</span></TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {items.map((c) => (
            <TableRow key={c.name}>
              <TableCell className="font-mono">{c.name}</TableCell>
              <TableCell className="font-mono">{c.step}</TableCell>
              <TableCell>{c.complete ? "complete" : "incomplete"}</TableCell>
              <TableCell>
                {c.complete && (
                  <Button
                    size="sm"
                    onClick={() => router.push(`/train/new?from_run=${encodeURIComponent(runId)}&checkpoint=${encodeURIComponent(c.name)}`)}
                  >
                    New run from this checkpoint
                  </Button>
                )}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Panel>
  );
}