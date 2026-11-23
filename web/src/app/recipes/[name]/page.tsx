"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cloneRecipe, createRun, getRecipe, getRecipeVersion } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { RecipeDetail, RecipeVersionDetail } from "@/lib/types";

const errMsg = (e: unknown) => (e instanceof Error ? e.message : String(e));

function Alert({ msg }: { msg: string }) {
  return (
    <div role="alert" className="rounded-md border border-red-500/50 bg-red-500/10 px-3 py-2 text-sm">{msg}</div>
  );
}

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
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-semibold">{name}</h1>
        <div className="flex gap-2">
          <Link href={`/recipes/new?from=${encodeURIComponent(name)}`} className={cn(buttonVariants({ variant: "outline" }))}>
            Edit as new
          </Link>
          <Button variant="outline" onClick={() => setCloning((c) => !c)}>Clone</Button>
          <Button onClick={() => void startRun()} disabled={starting || selected === null}>Start run</Button>
        </div>
      </div>
      {error && <Alert msg={error} />}
      {cloning && (
        <div className="space-y-2">
          <div className="flex max-w-md gap-2">
            <Input aria-label="New recipe name" placeholder="new-recipe-name" value={newName} onChange={(e) => setNewName(e.target.value)} />
            <Button onClick={() => void clone()} disabled={!newName.trim()}>Create clone</Button>
          </div>
          {cloneError && <Alert msg={cloneError} />}
        </div>
      )}
      <div className="grid gap-4 md:grid-cols-[12rem_1fr]">
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
                    "flex w-full items-center justify-between rounded-md px-2 py-1.5 text-left text-sm hover:bg-muted",
                    selected === v.number && "bg-muted font-medium",
                  )}
                >
                  <span>v{v.number}</span>
                  {v.frozen && <Badge variant="secondary">frozen</Badge>}
                </button>
              </li>
            ))}
        </ul>
        <pre aria-label="Recipe YAML" className="overflow-auto rounded-md bg-muted p-3 font-mono text-xs">
          {version?.yaml ?? ""}
        </pre>
      </div>
    </div>
  );
}
