"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import * as tus from "tus-js-client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { ReportView } from "@/components/report-view";
import { API_URL, ApiError, getIngest, startIngest } from "@/lib/api";
import { formatBytes } from "@/lib/format";
import { isValidSlug } from "@/lib/slug";
import type { IngestStatus, Task } from "@/lib/types";

const POLL_MS = 2000;
const ACCEPT: Record<Task, string> = { "llm-sft": ".jsonl", asr: ".zip", tts: ".zip" };
const TASKS: Task[] = ["llm-sft", "asr", "tts"];

const field = "space-y-1 text-sm";
const label = "block font-medium";

function describeUploadError(err: Error): string {
  const res = (err as { originalResponse?: { getStatus(): number; getBody(): string } }).originalResponse;
  const status = res?.getStatus();
  if (status === 507) return "Server disk is full (507). Free space on the VM before uploading.";
  if (status === 409) return `Conflict (409): ${res?.getBody() || err.message}`;
  return status ? `Upload failed (${status}): ${res?.getBody() || err.message}` : `Upload failed: ${err.message}`;
}

export function UploadForm() {
  const router = useRouter();
  const params = useSearchParams();
  const initialTask = params.get("task");
  const [task, setTask] = useState<Task>(TASKS.includes(initialTask as Task) ? (initialTask as Task) : "llm-sft");
  const [name, setName] = useState(params.get("name") ?? "");
  const [evalOnly, setEvalOnly] = useState(false);
  const [source, setSource] = useState("");
  const [license, setLicense] = useState("");
  const [consent, setConsent] = useState("");
  const [commercialOk, setCommercialOk] = useState(false);
  const [notes, setNotes] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [phase, setPhase] = useState<"idle" | "uploading" | "ingesting">("idle");
  const [error, setError] = useState<string | null>(null);
  const [ingest, setIngest] = useState<IngestStatus | null>(null);
  const upload = useRef<tus.Upload | null>(null);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      if (timer.current) clearInterval(timer.current);
      void upload.current?.abort();
    };
  }, []);

  const nameError = name !== "" && !isValidSlug(name) ? "Use lowercase letters, digits and single hyphens (e.g. bn-chat-v1)." : null;
  const ready =
    file !== null && isValidSlug(name) && source.trim() !== "" && license.trim() !== "" && consent.trim() !== "";

  function poll(id: number) {
    const tick = async () => {
      try {
        const st = await getIngest(id);
        if (!alive.current) return;
        setIngest(st);
        if (st.state === "done" && st.version) {
          if (timer.current) clearInterval(timer.current);
          router.push(`/datasets/${encodeURIComponent(st.dataset)}/v/${st.version.number}`);
        } else if (st.state === "failed") {
          if (timer.current) clearInterval(timer.current);
          setPhase("idle");
          setError(st.error ?? "Ingest failed");
        }
      } catch (e) {
        if (alive.current) setError(e instanceof Error ? e.message : String(e));
      }
    };
    void tick();
    timer.current = setInterval(() => void tick(), POLL_MS);
  }

  async function finishUpload(url: string) {
    const uploadId = url.split("/").filter(Boolean).pop() ?? "";
    setPhase("ingesting");
    try {
      const { ingest_id } = await startIngest({
        upload_id: uploadId,
        name,
        task,
        eval_only: evalOnly,
        provenance: { source, license, consent, commercial_ok: commercialOk, notes },
      });
      if (alive.current) poll(ingest_id);
    } catch (e) {
      setPhase("idle");
      setError(e instanceof ApiError ? `${e.status}: ${e.detail}` : e instanceof Error ? e.message : String(e));
    }
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

  const busy = phase !== "idle";

  return (
    <form onSubmit={submit} className="max-w-2xl space-y-4">
      {error && (
        <div role="alert" className="whitespace-pre-wrap rounded-md border border-red-500/50 bg-red-500/10 px-3 py-2 text-sm">
          {error}
        </div>
      )}

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
          className="h-8 rounded-md border bg-transparent px-2"
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
        />
      </div>

      <div className={field}>
        <label htmlFor="name" className={label}>Dataset name</label>
        <Input id="name" value={name} disabled={busy} onChange={(e) => setName(e.target.value)} aria-invalid={!!nameError} />
        {nameError && <p className="text-xs text-red-700 dark:text-red-300">{nameError}</p>}
      </div>

      <label className="flex items-center gap-2 text-sm">
        <input type="checkbox" checked={evalOnly} disabled={busy} onChange={(e) => setEvalOnly(e.target.checked)} />
        Eval only (never used for training)
      </label>

      <fieldset className="space-y-3 rounded-md border p-3">
        <legend className="px-1 text-sm font-semibold">Provenance</legend>
        <div className={field}>
          <label htmlFor="source" className={label}>Source</label>
          <Input id="source" value={source} disabled={busy} onChange={(e) => setSource(e.target.value)} />
        </div>
        <div className={field}>
          <label htmlFor="license" className={label}>License</label>
          <Input id="license" value={license} disabled={busy} onChange={(e) => setLicense(e.target.value)} />
        </div>
        <div className={field}>
          <label htmlFor="consent" className={label}>Consent</label>
          <Input id="consent" value={consent} disabled={busy} onChange={(e) => setConsent(e.target.value)} />
        </div>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={commercialOk} disabled={busy} onChange={(e) => setCommercialOk(e.target.checked)} />
          Commercial use allowed
        </label>
        <div className={field}>
          <label htmlFor="notes" className={label}>Notes (optional)</label>
          <Input id="notes" value={notes} disabled={busy} onChange={(e) => setNotes(e.target.value)} />
        </div>
      </fieldset>

      {phase === "uploading" && progress !== null && (
        <div className="space-y-1 text-sm">
          <Progress value={progress} aria-label="Upload progress" />
          <div className="text-muted-foreground">Uploading {progress.toFixed(0)}%</div>
        </div>
      )}
      {phase === "ingesting" && (
        <div className="text-sm text-muted-foreground">
          Validating and ingesting{ingest ? ` (${ingest.state})` : ""}...
        </div>
      )}

      <Button type="submit" disabled={!ready || busy}>
        Upload and ingest
      </Button>

      {ingest?.state === "failed" && (
        <section className="space-y-2">
          <h2 className="text-sm font-semibold">Validation report</h2>
          <ReportView report={ingest.report} />
        </section>
      )}
    </form>
  );
}
