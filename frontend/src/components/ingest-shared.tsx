"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { ErrorNotice } from "@/components/notice";
import { Input } from "@/components/ui/input";
import { ReportView } from "@/components/report-view";
import { ApiError, getIngest } from "@/lib/api";
import { isValidSlug } from "@/lib/slug";
import { ingestOutcome } from "@/lib/upload";
import type { IngestStatus, Provenance } from "@/lib/types";

const POLL_MS = 2000;

export const field = "space-y-1 text-sm";
export const label = "block text-[13px] text-muted-foreground";

export interface UploadStatus {
  busy: boolean;
  phase: "idle" | "uploading" | "ingesting" | "done";
  progress: number | null;
}

export type IngestPhase = "idle" | "uploading" | "ingesting";

export const EMPTY_PROVENANCE: Provenance = { source: "", license: "", consent: "", commercial_ok: false, notes: "" };

export const provenanceReady = (p: Provenance) =>
  p.source.trim() !== "" && p.license.trim() !== "" && p.consent.trim() !== "";

export function describeApiError(e: unknown): string {
  return e instanceof ApiError ? `${e.status}: ${e.detail}` : e instanceof Error ? e.message : String(e);
}

/** Shared phase/error/ingest state, status reporting and ingest polling for both ingest modes. */
export function useIngestRun(onStatus?: (s: UploadStatus) => void) {
  const router = useRouter();
  const [phase, setPhase] = useState<IngestPhase>("idle");
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ingest, setIngest] = useState<IngestStatus | null>(null);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  const alive = useRef(true);
  const report = useRef(onStatus);
  report.current = onStatus;

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      if (timer.current) clearInterval(timer.current);
    };
  }, []);

  const busy = phase !== "idle";
  useEffect(() => {
    report.current?.({ busy, phase, progress });
  }, [busy, phase, progress]);

  function poll(id: number) {
    const tick = async () => {
      try {
        const st = await getIngest(id);
        if (!alive.current) return;
        setIngest(st);
        setError(null);
        const outcome = ingestOutcome(st);
        if (outcome === "continue") return;
        if (timer.current) clearInterval(timer.current);
        if (outcome === "done" && st.version) {
          report.current?.({ busy: false, phase: "done", progress: 100 });
          router.push(`/datasets/${encodeURIComponent(st.dataset)}/v/${st.version.number}`);
        } else if (outcome === "failed") {
          setPhase("idle");
          setError(st.error ?? "Ingest failed");
        } else {
          setPhase("idle");
          setError("Ingest finished but reported no version; check the dataset page.");
        }
      } catch (e) {
        if (alive.current) setError(e instanceof Error ? e.message : String(e));
      }
    };
    void tick();
    timer.current = setInterval(() => void tick(), POLL_MS);
  }

  /** Starts an ingest via `start` and polls it; API errors are shown to the user. */
  async function begin(start: () => Promise<{ ingest_id: number }>) {
    setPhase("ingesting");
    try {
      const { ingest_id } = await start();
      if (alive.current) poll(ingest_id);
    } catch (e) {
      if (!alive.current) return;
      setPhase("idle");
      setError(describeApiError(e));
    }
  }

  return { phase, setPhase, progress, setProgress, error, setError, ingest, setIngest, busy, alive, begin };
}

export function NameFields({
  name, setName, evalOnly, setEvalOnly, disabled,
}: { name: string; setName: (v: string) => void; evalOnly: boolean; setEvalOnly: (v: boolean) => void; disabled: boolean }) {
  const nameError = name !== "" && !isValidSlug(name) ? "Use lowercase letters, digits and single hyphens (e.g. bn-chat-v1)." : null;
  return (
    <>
      <div className={field}>
        <label htmlFor="name" className={label}>Dataset name</label>
        <Input id="name" value={name} disabled={disabled} onChange={(e) => setName(e.target.value)} aria-invalid={!!nameError} />
        {nameError && <p className="text-[13px] text-status-failed">{nameError}</p>}
      </div>
      <label className="flex items-center gap-2 text-sm">
        <input type="checkbox" className="size-4 accent-[var(--saffron)]" checked={evalOnly} disabled={disabled} onChange={(e) => setEvalOnly(e.target.checked)} />
        Eval only (never used for training)
      </label>
    </>
  );
}

export function ProvenanceFields({
  value, onChange, disabled,
}: { value: Provenance; onChange: (p: Provenance) => void; disabled: boolean }) {
  const set = <K extends keyof Provenance>(k: K, v: Provenance[K]) => onChange({ ...value, [k]: v });
  return (
    <fieldset className="space-y-3 rounded-md border border-border p-3">
      <legend className="px-1 text-sm font-semibold">Provenance</legend>
      <div className={field}>
        <label htmlFor="source" className={label}>Source</label>
        <Input id="source" value={value.source} disabled={disabled} onChange={(e) => set("source", e.target.value)} />
      </div>
      <div className={field}>
        <label htmlFor="license" className={label}>License</label>
        <Input id="license" value={value.license} disabled={disabled} onChange={(e) => set("license", e.target.value)} />
      </div>
      <div className={field}>
        <label htmlFor="consent" className={label}>Consent</label>
        <Input id="consent" value={value.consent} disabled={disabled} onChange={(e) => set("consent", e.target.value)} />
      </div>
      <label className="flex items-center gap-2 text-sm">
        <input type="checkbox" className="size-4 accent-[var(--saffron)]" checked={value.commercial_ok} disabled={disabled} onChange={(e) => set("commercial_ok", e.target.checked)} />
        Commercial use allowed
      </label>
      <div className={field}>
        <label htmlFor="notes" className={label}>Notes (optional)</label>
        <Input id="notes" value={value.notes} disabled={disabled} onChange={(e) => set("notes", e.target.value)} />
      </div>
    </fieldset>
  );
}

export function IngestErrorNotice({ error }: { error: string | null }) {
  return error ? <ErrorNotice className="whitespace-pre-wrap">{error}</ErrorNotice> : null;
}

export function IngestingNote({ phase, ingest }: { phase: IngestPhase; ingest: IngestStatus | null }) {
  if (phase !== "ingesting") return null;
  const p = ingest?.progress;
  if (p && p.total > 0) {
    return (
      <div className="text-sm text-muted-foreground">
        Validating {p.done.toLocaleString()} of {p.total.toLocaleString()} files...
      </div>
    );
  }
  return (
    <div className="text-sm text-muted-foreground">Validating and ingesting{ingest ? ` (${ingest.state})` : ""}...</div>
  );
}

export function FailedReport({ ingest }: { ingest: IngestStatus | null }) {
  return ingest?.state === "failed" ? (
    <section className="space-y-2">
      <h2 className="text-sm font-semibold">Validation report</h2>
      <ReportView report={ingest.report} />
    </section>
  ) : null;
}
