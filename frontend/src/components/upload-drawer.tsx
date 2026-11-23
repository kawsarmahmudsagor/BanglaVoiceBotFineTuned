"use client";

import { Loader, Upload } from "lucide-react";
import { createContext, Suspense, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { Drawer } from "@/components/drawer";
import { UploadForm, type UploadStatus } from "@/components/upload-form";
import type { Task } from "@/lib/types";

interface UploadDrawerApi {
  openUpload: (task?: Task) => void;
  status: UploadStatus;
}

const IDLE: UploadStatus = { busy: false, phase: "idle", progress: null };
const Ctx = createContext<UploadDrawerApi>({ openUpload: () => {}, status: IDLE });

export const useUploadDrawer = () => useContext(Ctx);

/**
 * Owns the dataset upload drawer at app level, so an upload keeps running and stays visible
 * while the user moves between pages. Only a full page reload stops it (and asks first).
 */
export function UploadDrawerProvider({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const [task, setTask] = useState<Task | undefined>();
  const [formKey, setFormKey] = useState(0);
  const [status, setStatus] = useState<UploadStatus>(IDLE);

  const openUpload = useCallback(
    (t?: Task) => {
      // A fresh form for a new upload; never reset one that is still running.
      if (!status.busy && t !== task) {
        setTask(t);
        setFormKey((k) => k + 1);
      }
      setOpen(true);
    },
    [status.busy, task],
  );

  const onStatus = useCallback((s: UploadStatus) => {
    setStatus(s);
    if (s.phase === "done") {
      setOpen(false);
      setTask(undefined);
      setFormKey((k) => k + 1);
      setStatus(IDLE);
    }
  }, []);

  useEffect(() => {
    if (!status.busy) return;
    const warn = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [status.busy]);

  const api = useMemo(() => ({ openUpload, status }), [openUpload, status]);
  return (
    <Ctx.Provider value={api}>
      {children}
      <Drawer open={open} onOpenChange={setOpen} title="Upload dataset">
        <Suspense fallback={null}>
          <UploadForm key={formKey} defaultTask={task} onStatus={onStatus} />
        </Suspense>
      </Drawer>
    </Ctx.Provider>
  );
}

/** Top-bar chip shown while an upload runs; it reopens the drawer. */
export function UploadChip() {
  const { status, openUpload } = useUploadDrawer();
  if (!status.busy) return null;
  const label = status.phase === "uploading" ? `Uploading ${Math.round(status.progress ?? 0)}%` : "Ingesting dataset";
  return (
    <button
      type="button"
      onClick={() => openUpload()}
      className="inline-flex h-7 shrink-0 items-center gap-1.5 rounded-md border border-saffron/60 bg-card px-2 text-[13px] hover:bg-raised"
      aria-label={`${label}. Show upload`}
    >
      {status.phase === "uploading" ? <Upload className="size-3.5 text-saffron" aria-hidden /> : <Loader className="size-3.5 text-saffron" aria-hidden />}
      <span className="hidden font-mono sm:inline">{label}</span>
    </button>
  );
}
