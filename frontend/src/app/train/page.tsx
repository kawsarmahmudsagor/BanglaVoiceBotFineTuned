"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { PageHeader, Panel } from "@/components/matra";
import { EmptyState, ErrorNotice } from "@/components/notice";
import { buttonVariants } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { listRecipes } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { RecipeListItem } from "@/lib/types";

export default function RecipesPage() {
  const [items, setItems] = useState<RecipeListItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    listRecipes()
      .then((d) => active && setItems(d))
      .catch((e) => active && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="space-y-5">
      <PageHeader
        title="Train"
        actions={
          <>
            <Link href="/catalog" className={cn(buttonVariants({ variant: "outline", size: "lg" }), "px-3")}>Start from catalog</Link>
            <Link href="/train/new" className={cn(buttonVariants({ size: "lg" }), "px-3")}>New training config</Link>
          </>
        }
      >
        A training config pins the base model, dataset versions, method and hyperparameters for a training run.
      </PageHeader>
      {error && <ErrorNotice>Cannot load training configs ({error})</ErrorNotice>}
      <Panel>
        {items?.length === 0 ? (
          <EmptyState>
            No training configs yet. <Link href="/catalog" className="text-saffron underline">Pick a model in the catalog</Link> to start one.
          </EmptyState>
        ) : (
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead>Name</TableHead>
                <TableHead>Latest</TableHead>
                <TableHead>Task</TableHead>
                <TableHead>Backend</TableHead>
                <TableHead>Base model</TableHead>
                <TableHead>Updated</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {items?.map((r) => (
                <TableRow key={r.name}>
                  <TableCell className="max-w-56">
                    <Link href={`/train/${encodeURIComponent(r.name)}`} className="block truncate font-medium hover:text-saffron" title={r.name}>{r.name}</Link>
                  </TableCell>
                  <TableCell>
                    <span className="font-mono">v{r.latest.number}</span>
                    {r.latest.frozen && <span className="ml-2 text-[13px] text-muted-foreground">frozen</span>}
                  </TableCell>
                  <TableCell className="font-mono">{r.latest.task}</TableCell>
                  <TableCell className="font-mono">{r.latest.backend}</TableCell>
                  <TableCell className="max-w-64 truncate font-mono text-[13px]" title={r.latest.base_model}>{r.latest.base_model}</TableCell>
                  <TableCell className="text-muted-foreground">
                    {r.latest.updated_at ? new Date(r.latest.updated_at).toLocaleString() : ""}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </Panel>
    </div>
  );
}
