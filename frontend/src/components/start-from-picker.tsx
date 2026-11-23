"use client";

import { useEffect, useId, useState } from "react";
import { ErrorNotice } from "@/components/notice";
import { getInitSources } from "@/lib/api";
import type { InitFrom, InitSources } from "@/lib/types";
import { initFromItem } from "@/lib/warm-start";

const SELECT_CLASS = "h-9 w-full rounded-md border border-input bg-card px-2 text-sm";
const errMsg = (e: unknown) => (e instanceof Error ? e.message : String(e));

/** The option value that stands for an InitFrom. "latest" has no step to list, so it keeps its own key. */
function keyOf(v: InitFrom | null): string {
  if (!v) return "";
  if ("model_version" in v && v.model_version) return `mv|${v.model_version}`;
  if ("run" in v) return `run|${v.run}|${v.checkpoint}`;
  return "";
}

function parseKey(key: string): InitFrom | null {
  const [kind, a, b] = key.split("|");
  if (kind === "mv") return { model_version: a };
  if (kind === "run") return { run: a, checkpoint: b === "final" || b === "latest" ? b : Number(b) };
  return null;
}

export function StartFromPicker({
  task,
  backend,
  baseModel,
  value,
  onChange,
}: {
  task: string;
  backend: string;
  baseModel: string;
  value: InitFrom | null;
  onChange: (v: InitFrom | null) => void;
}) {
  const id = useId();
  const [sources, setSources] = useState<InitSources | null>(null);
  const [error, setError] = useState<string | null>(null);
  const ready = !!task && !!backend && !!baseModel;
  const current = keyOf(value);

  useEffect(() => {
    setSources(null);
    setError(null);
    if (!ready) return;
    let active = true;
    getInitSources({ task, backend, base_model: baseModel })
      .then((s) => active && setSources(s))
      .catch((e) => active && setError(errMsg(e)));
    return () => {
      active = false;
    };
  }, [ready, task, backend, baseModel]);

  const options: { key: string; label: string }[] = [];
  for (const r of sources?.runs ?? []) {
    for (const c of r.checkpoints) {
      const step = c.name === "final" ? "final" : c.step;
      options.push({ key: `run|${r.id}|${step}`, label: `${r.id} · ${c.name} · step ${c.step}` });
    }
  }
  for (const m of sources?.model_versions ?? []) options.push({ key: `mv|${m.ref}`, label: m.ref });

  // "latest" is not listed as such; it stays valid while its run is listed.
  const listed = (k: string) => {
    if (options.some((o) => o.key === k)) return true;
    const [kind, runId, which] = k.split("|");
    return kind === "run" && which === "latest" && !!sources?.runs.some((r) => r.id === runId);
  };
  const stale = !!sources && !!current && !listed(current);
  useEffect(() => {
    if (stale) onChange(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stale]);

  const extra = current && !options.some((o) => o.key === current) && !stale ? current : null;

  return (
    <div className="space-y-1">
      <label htmlFor={id} className="block text-[13px] text-muted-foreground">Start from</label>
      <select id={id} className={SELECT_CLASS} value={stale ? "" : current} onChange={(e) => onChange(parseKey(e.target.value))}>
        <option value="">Base model (default)</option>
        {extra && <option value={extra}>{extra.startsWith("mv|") ? extra.slice(3) : `${extra.split("|")[1]} · ${extra.split("|")[2]}`}</option>}
        {options.map((o) => <option key={o.key} value={o.key}>{o.label}</option>)}
      </select>
      {error && <ErrorNotice>Cannot load start-from sources ({error})</ErrorNotice>}
    </div>
  );
}