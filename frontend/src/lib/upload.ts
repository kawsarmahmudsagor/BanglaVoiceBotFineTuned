import type { IngestStatus } from "./types";

export function uploadIdFromUrl(url: string): string {
  const path = url.split(/[?#]/)[0];
  return path.split("/").filter(Boolean).pop() ?? "";
}

const MAX_BODY = 300;

export function describeUploadError(err: Error): string {
  const res = (err as { originalResponse?: { getStatus(): number; getBody(): string } }).originalResponse;
  const status = res?.getStatus();
  const body = (res?.getBody() ?? "").slice(0, MAX_BODY);
  if (status === 507) return "Server disk is full (507). Free space on the VM before uploading.";
  if (status === 409) return `Conflict (409): ${body || err.message}`;
  return status ? `Upload failed (${status}): ${body || err.message}` : `Upload failed: ${err.message}`;
}

export type IngestOutcome = "continue" | "done" | "failed" | "broken";

export function ingestOutcome(st: IngestStatus): IngestOutcome {
  if (st.state === "failed") return "failed";
  if (st.state === "done") return st.version ? "done" : "broken";
  return "continue";
}
