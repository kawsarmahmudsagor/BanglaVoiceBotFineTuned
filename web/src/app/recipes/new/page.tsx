"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { PageHeader } from "@/components/matra";
import { ErrorNotice } from "@/components/notice";
import { RecipeEditor } from "@/components/recipe-editor";
import { getCatalogEntry, getRecipe, getRecipeVersion, listBackends } from "@/lib/api";
import { backendNotice, catalogToSpec } from "@/lib/catalog-prefill";

type Spec = Record<string, unknown>;

const errMsg = (e: unknown) => (e instanceof Error ? e.message : String(e));

function NewRecipe() {
  const params = useSearchParams();
  const from = params.get("from");
  const catalog = params.get("catalog");
  const [initial, setInitial] = useState<Spec | undefined>();
  const [notice, setNotice] = useState<string | null>(null);
  const [title, setTitle] = useState("New recipe");
  const [ready, setReady] = useState(!from && !catalog);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!from && !catalog) return;
    let active = true;
    const load = async () => {
      if (catalog) {
        const entry = await getCatalogEntry(catalog);
        const spec = catalogToSpec(entry);
        const backends = (await listBackends()) as { name?: unknown }[];
        const installed = backends.map((b) => String(b.name ?? ""));
        return { spec, title: `New recipe (from catalog: ${entry.name})`, notice: backendNotice(String(spec.backend), installed) };
      }
      const r = await getRecipe(from!);
      const v = await getRecipeVersion(from!, Math.max(...r.versions.map((x) => x.number)));
      return { spec: { ...v.spec, name: "" }, title: `New recipe (from ${from})`, notice: null };
    };
    load()
      .then((r) => {
        if (!active) return;
        setInitial(r.spec);
        setTitle(r.title);
        setNotice(r.notice);
        setReady(true);
      })
      .catch((e) => {
        if (!active) return;
        setError(errMsg(e));
        setReady(true);
      });
    return () => {
      active = false;
    };
  }, [from, catalog]);

  return (
    <div className="space-y-4">
      <PageHeader title={title} />
      {error && <ErrorNotice>Cannot preload {catalog ?? from} ({error})</ErrorNotice>}
      {ready && <RecipeEditor initial={initial} notice={notice} />}
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
