"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  EMPTY_PROVENANCE, FailedReport, IngestErrorNotice, IngestingNote, NameFields, ProvenanceFields,
  field, label, provenanceReady, useIngestRun, type UploadStatus,
} from "@/components/ingest-shared";
import { importDirs, ingestLocal } from "@/lib/api";
import { isValidSlug } from "@/lib/slug";
import type { ImportDir, Provenance, Task } from "@/lib/types";

const SPEECH_TASKS: Task[] = ["asr", "tts"];

export function ServerPathForm({ defaultTask, onStatus }: { defaultTask?: Task; onStatus?: (s: UploadStatus) => void } = {}) {
  const [task, setTask] = useState<Task>(SPEECH_TASKS.includes(defaultTask as Task) ? (defaultTask as Task) : "asr");
  const [name, setName] = useState("");
  const [evalOnly, setEvalOnly] = useState(false);
  const [prov, setProv] = useState<Provenance>(EMPTY_PROVENANCE);
  const [path, setPath] = useState("");
  const [dirs, setDirs] = useState<ImportDir[]>([]);
  const run = useIngestRun(onStatus);
  const { phase, setError, setIngest, error, ingest, busy } = run;

  useEffect(() => {
    let live = true;
    importDirs()
      .then((r) => {
        if (live) setDirs(r.items);
      })
      .catch(() => {
        /* the free-text path still works without the list */
      });
    return () => {
      live = false;
    };
  }, []);

  function choose(value: string) {
    setPath(value);
    const hint = dirs.find((d) => d.path === value)?.task_hint;
    if (hint) setTask(hint);
  }

  const ready = path.trim() !== "" && isValidSlug(name) && provenanceReady(prov);

  function submit(ev: React.FormEvent) {
    ev.preventDefault();
    if (!ready) return;
    setError(null);
    setIngest(null);
    void run.begin(() =>
      ingestLocal({ path: path.trim(), name, task, eval_only: evalOnly, provenance: prov }),
    );
  }

  return (
    <form onSubmit={submit} className="space-y-4">
      <IngestErrorNotice error={error} />

      <div className={field}>
        <label htmlFor="import-dir" className={label}>Import directory</label>
        <select
          id="import-dir"
          value={dirs.some((d) => d.path === path) ? path : ""}
          disabled={busy}
          onChange={(e) => choose(e.target.value)}
          className="h-9 w-full rounded-md border border-border bg-card px-2"
        >
          <option value="">Choose a directory...</option>
          {dirs.map((d) => (
            <option key={d.path} value={d.path}>{d.path}</option>
          ))}
        </select>
      </div>

      <div className={field}>
        <label htmlFor="server-path" className={label}>Path</label>
        <Input id="server-path" value={path} disabled={busy} placeholder="relative to the import root" onChange={(e) => setPath(e.target.value)} />
      </div>

      <div className={field}>
        <label htmlFor="task" className={label}>Task</label>
        <select
          id="task"
          value={task}
          disabled={busy}
          onChange={(e) => setTask(e.target.value as Task)}
          className="h-9 w-full rounded-md border border-border bg-card px-2"
        >
          {SPEECH_TASKS.map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
      </div>

      <NameFields name={name} setName={setName} evalOnly={evalOnly} setEvalOnly={setEvalOnly} disabled={busy} />
      <ProvenanceFields value={prov} onChange={setProv} disabled={busy} />

      <IngestingNote phase={phase} ingest={ingest} />

      <Button type="submit" size="lg" className="px-4" disabled={!ready || busy}>
        Ingest from server path
      </Button>

      <FailedReport ingest={ingest} />
    </form>
  );
}
