"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { RecipeEditor } from "@/components/recipe-editor";
import { getRecipe, getRecipeVersion } from "@/lib/api";

function NewRecipe() {
  const from = useSearchParams().get("from");
  const [initial, setInitial] = useState<Record<string, unknown> | undefined>();
  const [ready, setReady] = useState(!from);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!from) return;
    let active = true;
    getRecipe(from)
      .then((r) => getRecipeVersion(from, Math.max(...r.versions.map((v) => v.number))))
      .then((v) => {
        if (!active) return;
        setInitial({ ...v.spec, name: "" });
        setReady(true);
      })
      .catch((e) => {
        if (!active) return;
        setError(e instanceof Error ? e.message : String(e));
        setReady(true);
      });
    return () => {
      active = false;
    };
  }, [from]);

  return (
    <div className="space-y-4">
      <h1 className="text-lg font-semibold">New recipe{from ? ` (from ${from})` : ""}</h1>
      {error && (
        <div role="alert" className="rounded-md border border-red-500/50 bg-red-500/10 px-3 py-2 text-sm">
          Cannot preload {from} ({error})
        </div>
      )}
      {ready && <RecipeEditor initial={initial} />}
    </div>
  );
}

export default function NewRecipePage() {
  return (
    <Suspense fallback={null}>
      <NewRecipe />
    </Suspense>
  );
}
