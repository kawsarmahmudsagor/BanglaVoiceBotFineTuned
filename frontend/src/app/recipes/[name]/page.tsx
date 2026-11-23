"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { PageHeader, Panel } from "@/components/matra";
import { ErrorNotice } from "@/components/notice";
import { Button, buttonVariants } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cloneRecipe, createRun, getRecipe, getRecipeVersion } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { RecipeDetail, RecipeVersionDetail } from "@/lib/types";

const errMsg = (e: unknown) => (e instanceof Error ? e.message : String(e));

export default function RecipePage() {
  const { name: rawName } = useParams<{ name: string }>();
  const name = decodeURIComponent(rawName);
  const router = useRouter();
  const [recipe, setRecipe] = useState<RecipeDetail | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [version, setVersion] = useState<RecipeVersionDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cloning, setCloning] = useState(false);
  const [newName, setNewName] = useState("");
  const [cloneError, setCloneError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    let active = true;
    getRecipe(name)
      .then((r) => {
        if (!active) return;
        setRecipe(r);
        setSelected(r.versions.length ? Math.max(...r.versions.map((v) => v.number)) : null);
      })
      .catch((e) => active && setError(errMsg(e)));
    return () => {
      active = false;
    };
  }, [name]);

  useEffect(() => {
    if (selected === null) return;
    let active = true;
    setVersion(null);
    getRecipeVersion(name, selected)
      .then((v) => active && setVersion(v))
      .catch((e) => active && setError(errMsg(e)));
    return () => {
      active = false;
    };
  }, [name, selected]);

  async function clone() {
    setCloneError(null);
    try {
      const r = await cloneRecipe(name, newName.trim());
      router.push(`/recipes/${encodeURIComponent(r.name)}`);
    } catch (e) {
      setCloneError(errMsg(e));
    }
  }

  async function startRun() {
    setError(null);
    setStarting(true);
    try {
      const r = await createRun({ recipe: name, version: selected ?? undefined });
      router.push(`/runs/${encodeURIComponent(r.id)}`);
    } catch (e) {
      setError(errMsg(e));
      setStarting(false);
    }
  }

  return (
    <div className="space-y-5">
      <PageHeader
        title={name}
        actions={
          <>
            <Link href={`/recipes/new?from=${encodeURIComponent(name)}`} className={cn(buttonVariants({ variant: "outline", size: "lg" }), "px-3")}>
              Edit as new
            </Link>
            <Button variant="outline" size="lg" className="px-3" aria-expanded={cloning} onClick={() => setCloning((c) => !c)}>Clone</Button>
            <Button size="lg" className="px-3" onClick={() => void startRun()} disabled={starting || selected === null}>
              {starting ? "Starting…" : "Start run"}
            </Button>
          </>
        }
      />
      {error && <ErrorNotice>{error}</ErrorNotice>}
      {cloning && (
        <Panel title="Clone recipe">
          <div className="flex max-w-md flex-wrap gap-2">
            <Input aria-label="New recipe name" placeholder="new-recipe-name" value={newName} onChange={(e) => setNewName(e.target.value)} className="min-w-0 flex-1" />
            <Button onClick={() => void clone()} disabled={!newName.trim()}>Create clone</Button>
          </div>
          {cloneError && <ErrorNotice className="mt-2">{cloneError}</ErrorNotice>}
        </Panel>
      )}
      <div className="grid gap-4 md:grid-cols-[12rem_minmax(0,1fr)]">
        <Panel title="Versions">
          <ul aria-label="Versions" className="space-y-1">
            {recipe?.versions
              .slice()
              .sort((a, b) => b.number - a.number)
              .map((v) => (
                <li key={v.number}>
                  <button
                    type="button"
                    aria-current={selected === v.number ? "true" : undefined}
                    onClick={() => setSelected(v.number)}
                    className={cn(
                      "flex h-9 w-full items-center justify-between rounded-md px-2 text-left text-sm hover:bg-raised",
                      selected === v.number && "bg-raised font-medium",
                    )}
                  >
                    <span className="font-mono">v{v.number}</span>
                    {v.frozen && <span className="text-[13px] text-muted-foreground">frozen</span>}
                  </button>
                </li>
              ))}
          </ul>
        </Panel>
        <Panel title={selected ? `v${selected} YAML` : "YAML"}>
          <pre aria-label="Recipe YAML" className="overflow-auto rounded-md bg-ink p-3 font-mono text-[13px] leading-relaxed">
            {version?.yaml ?? ""}
          </pre>
        </Panel>
      </div>
    </div>
  );
}
