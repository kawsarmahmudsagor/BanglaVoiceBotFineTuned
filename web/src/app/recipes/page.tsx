"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
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
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-semibold">Recipes</h1>
        <Link href="/recipes/new" className={cn(buttonVariants())}>New recipe</Link>
      </div>
      {error && (
        <div role="alert" className="rounded-md border border-red-500/50 bg-red-500/10 px-3 py-2 text-sm">
          Cannot load recipes ({error})
        </div>
      )}
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Name</TableHead>
            <TableHead>Latest</TableHead>
            <TableHead>Task</TableHead>
            <TableHead>Backend</TableHead>
            <TableHead>Base model</TableHead>
            <TableHead>Updated</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {items?.length === 0 && (
            <TableRow>
              <TableCell colSpan={6} className="text-muted-foreground">No recipes yet</TableCell>
            </TableRow>
          )}
          {items?.map((r) => (
            <TableRow key={r.name}>
              <TableCell>
                <Link href={`/recipes/${encodeURIComponent(r.name)}`} className="underline">{r.name}</Link>
              </TableCell>
              <TableCell>
                v{r.latest.number} {r.latest.frozen && <Badge variant="secondary">frozen</Badge>}
              </TableCell>
              <TableCell className="font-mono">{r.latest.task}</TableCell>
              <TableCell>{r.latest.backend}</TableCell>
              <TableCell>{r.latest.base_model}</TableCell>
              <TableCell className="text-muted-foreground">
                {r.latest.updated_at ? new Date(r.latest.updated_at).toLocaleString() : ""}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
