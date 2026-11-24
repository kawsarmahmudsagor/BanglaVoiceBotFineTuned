"use client";

import { useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import * as tus from "tus-js-client";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  EMPTY_PROVENANCE, FailedReport, IngestErrorNotice, IngestingNote, NameFields, ProvenanceFields,
  field, label, provenanceReady, useIngestRun, type UploadStatus,
} from "@/components/ingest-shared";
import { ServerPathForm } from "@/components/server-path-form";
import { API_URL, startIngest } from "@/lib/api";
import { formatBytes } from "@/lib/format";
import { isValidSlug } from "@/lib/slug";
import { describeUploadError, uploadIdFromUrl } from "@/lib/upload";
import type { Provenance, Task } from "@/lib/types";

export type { UploadStatus };

const ACCEPT: Record<Task, string> = { "llm-sft": ".jsonl", asr: ".zip", tts: ".zip" };
const TASKS: Task[] = ["llm-sft", "asr", "tts"];

type Mode = "upload" | "server";
const MODES: [Mode, string][] = [["upload", "Upload file"], ["server", "From server path"]];

export function UploadForm({ defaultTask, onStatus }: { defaultTask?: Task; onStatus?: (s: UploadStatus) => void } = {}) {
  const [mode, setMode] = useState<Mode>("upload");
  const [busy, setBusy] = useState(false);
  const report = (s: UploadStatus) => {
    setBusy(s.busy);
    onStatus?.(s);
  };
  return (
    <div className="max-w-2xl space-y-4">
      <div role="group" aria-label="Ingest source" className="inline-flex rounded-md border border-border p-0.5">
        {MODES.map(([m, text]) => (
          <button
            key={m}
            type="button"
            aria-pressed={mode === m}
            disabled={busy}
            onClick={() => setMode(m)}
            className={`h-8 rounded px-3 text-sm ${mode === m ? "bg-raised font-semibold text-foreground" : "text-muted-foreground"}`}
          >
            {text}
          </button>
        ))}
      </div>
      {mode === "upload" ? (
        <UploadFileForm defaultTask={defaultTask} onStatus={report} />
      ) : (
        <ServerPathForm defaultTask={defaultTask} onStatus={report} />
      )}
    </div>
  );
}

function UploadFileForm({ defaultTask, onStatus }: { defaultTask?: Task; onStatus?: (s: UploadStatus) => void }) {
  const params = useSearchParams();
  const initialTask = defaultTask ?? params.get("task");
  const [task, setTask] = useState<Task>(TASKS.includes(initialTask as Task) ? (initialTask as Task) : "llm-sft");
  const [name, setName] = useState(params.get("name") ?? "");
  const [evalOnly, setEvalOnly] = useState(false);
  const [prov, setProv] = useState<Provenance>(EMPTY_PROVENANCE);
  const [file, setFile] = useState<File | null>(null);
  const run = useIngestRun(onStatus);
  const { phase, setPhase, progress, setProgress, error, setError, ingest, setIngest, busy, alive } = run;
  const upload = useRef<tus.Upload | null>(null);

  useEffect(() => {
    return () => {
      void upload.current?.abort();
    };
  }, []);

  const ready = file !== null && isValidSlug(name) && provenanceReady(prov);

  async function finishUpload(url: string) {
    if (!alive.current) return;
    const uploadId = uploadIdFromUrl(url);
    await run.begin(() => startIngest({ upload_id: uploadId, name, task, eval_only: evalOnly, provenance: prov }));
  }

  function submit(ev: React.FormEvent) {
    ev.preventDefault();
    if (!file || !ready) return;
    setError(null);
    setIngest(null);
    setPhase("uploading");
    setProgress(0);
    const up = new tus.Upload(file, {
      endpoint: `${API_URL}/api/uploads`,
      chunkSize: 8 * 1024 * 1024,
      retryDelays: [0, 1000, 3000, 5000],
      metadata: { filename: file.name },
      onProgress: (sent, total) => setProgress(total ? (sent / total) * 100 : 0),
      onError: (err) => {
        setPhase("idle");
        setError(describeUploadError(err));
      },
      onSuccess: () => void finishUpload(up.url ?? ""),
    });
    upload.current = up;
    void up.findPreviousUploads().then((prev) => {
      if (prev.length > 0) up.resumeFromPreviousUpload(prev[0]);
      up.start();
    });
  }

  return (
    <form onSubmit={submit} className="space-y-4">
      <IngestErrorNotice error={error} />

      <div className={field}>
        <label htmlFor="task" className={label}>Task</label>
        <select
          id="task"
          value={task}
          disabled={busy}
          onChange={(e) => {
            setTask(e.target.value as Task);
            setFile(null);
          }}
          className="h-9 w-full rounded-md border border-border bg-card px-2"
        >
          {TASKS.map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
      </div>

      <div className={field}>
        <label htmlFor="file" className={label}>
          File ({ACCEPT[task]}){file ? `, ${formatBytes(file.size)}` : ""}
        </label>
        <input
          id="file"
          key={task}
          type="file"
          accept={ACCEPT[task]}
          disabled={busy}
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="block w-full text-sm file:mr-3 file:h-9 file:rounded-md file:border file:border-border file:bg-raised file:px-3 file:text-sm file:text-foreground"
        />
      </div>

      <NameFields name={name} setName={setName} evalOnly={evalOnly} setEvalOnly={setEvalOnly} disabled={busy} />
      <ProvenanceFields value={prov} onChange={setProv} disabled={busy} />

      {phase === "uploading" && progress !== null && (
        <div className="space-y-1 text-sm">
          <Progress value={progress} aria-label="Upload progress" />
          <div className="text-muted-foreground">Uploading <span className="font-mono">{progress.toFixed(0)}%</span>. You can close this panel; the upload continues.</div>
        </div>
      )}
      <IngestingNote phase={phase} ingest={ingest} />

      <Button type="submit" size="lg" className="px-4" disabled={!ready || busy}>
        Upload and ingest
      </Button>

      <FailedReport ingest={ingest} />
    </form>
  );
}
