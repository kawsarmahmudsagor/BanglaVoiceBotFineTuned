import { useCallback, useEffect, useRef, useState } from "react";
import { getDataset, getReview, patchClip, retryLlm } from "../api";
import { REASON_LABEL, ReviewCard } from "../components/ReviewCard";
import type { ClipStatus, DatasetSummary, ReviewClip, ReviewFilters, ReviewQueue, ReviewReason, Video } from "../types";

const QUEUE: ReadonlySet<ClipStatus> = new Set(["flagged", "needs_review", "pending_llm"]);

const STATUS_OPTIONS: { value: NonNullable<ReviewFilters["status"]> | ""; label: string }[] = [
  { value: "", label: "To review" },
  { value: "flagged", label: "Flagged" },
  { value: "needs_review", label: "Needs review" },
  { value: "pending_llm", label: "Waiting for LLM" },
  { value: "approved", label: "Approved" },
  { value: "rejected", label: "Rejected" },
  { value: "auto", label: "Auto-accepted" },
];

/** Whether the server would list this clip under these filters. */
function matches(c: ReviewClip, f: ReviewFilters): boolean {
  if (f.status ? c.status !== f.status : !QUEUE.has(c.status)) return false;
  if (f.reason && c.reason !== f.reason) return false;
  return !f.video || c.video_id === f.video;
}

/** The queue after a PATCH: the clip is replaced or, if it no longer matches the filters, removed. */
function withClip(q: ReviewQueue, updated: ReviewClip, f: ReviewFilters): ReviewQueue {
  const old = q.clips.find((c) => c.id === updated.id);
  if (!old) return q;
  const clips = matches(updated, f)
    ? q.clips.map((c) => (c.id === updated.id ? updated : c))
    : q.clips.filter((c) => c.id !== updated.id);
  const delta = Number(QUEUE.has(old.status)) - Number(QUEUE.has(updated.status)); // 1: left the queue
  return { clips, remaining: q.remaining - delta, reviewed: q.reviewed + delta };
}

function preview(c: ReviewClip): string {
  return c.text_final || c.text_a || c.text_translated || c.text_en || "No text";
}

export function Review({ dataset }: { dataset: DatasetSummary | null }) {
  const id = dataset?.id ?? null;
  const [filters, setFilters] = useState<ReviewFilters>({});
  const [queue, setQueue] = useState<ReviewQueue | null>(null);
  const [index, setIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [videos, setVideos] = useState<Video[]>([]);
  const [retrying, setRetrying] = useState(false);
  const [retryNote, setRetryNote] = useState<{ text: string; error: boolean } | null>(null);
  const listRef = useRef<HTMLOListElement>(null);
  const filtersRef = useRef(filters);
  filtersRef.current = filters;

  useEffect(() => {
    setVideos([]);
    setRetryNote(null);
    setFilters((f) => (Object.keys(f).length ? {} : f));
    if (!id) return;
    let live = true;
    getDataset(id).then(
      (d) => live && setVideos(d.videos),
      () => {},
    );
    return () => {
      live = false;
    };
  }, [id]);

  const load = useCallback(() => {
    if (!id) return () => {};
    let live = true;
    setError(null);
    getReview(id, filters).then(
      (q) => {
        if (!live) return;
        setQueue(q);
        setIndex(0);
      },
      (e: unknown) => live && setError(e instanceof Error ? e.message : String(e)),
    );
    return () => {
      live = false;
    };
  }, [id, filters]);

  useEffect(() => {
    setQueue(null);
    return load();
  }, [load]);

  const clips = queue?.clips ?? [];
  const at = Math.min(index, Math.max(0, clips.length - 1));
  const clip = clips[at] ?? null;

  useEffect(() => {
    listRef.current?.querySelector('[aria-current="true"]')?.scrollIntoView?.({ block: "nearest" });
  }, [at, clip?.id]);

  const apply = (updated: ReviewClip) => setQueue((q) => (q ? withClip(q, updated, filtersRef.current) : q));

  async function decide(c: ReviewClip, patch: { text_final?: string; status: "approved" | "rejected" }) {
    const updated = await patchClip(c.dataset_id, c.id, patch);
    apply(updated);
    // a clip that stays in the list (e.g. "Approved" filter) is stepped over; a removed one makes room
    if (matches(updated, filtersRef.current)) setIndex((i) => i + 1);
  }

  const next = () => setIndex((i) => Math.min(i + 1, Math.max(0, clips.length - 1)));
  const prev = () => setIndex((i) => Math.max(0, Math.min(i, clips.length - 1) - 1));
  const setFilter = <K extends keyof ReviewFilters>(k: K, v: ReviewFilters[K] | "") =>
    setFilters((f) => ({ ...f, [k]: v || undefined }));

  if (!dataset) {
    return <div className="panel empty">Pick a dataset above, or start one on the Jobs page, to review its clips.</div>;
  }

  const hasPendingLlm = clips.some((c) => c.status === "pending_llm");

  async function retry() {
    if (!id) return;
    setRetrying(true);
    setRetryNote(null);
    try {
      await retryLlm(id);
      setRetryNote({
        text: "Retry queued. Clips leave “Waiting for LLM” as the LLM answers them.",
        error: false,
      });
      load();
    } catch (e) {
      setRetryNote({ text: `Could not retry: ${e instanceof Error ? e.message : String(e)}`, error: true });
    } finally {
      setRetrying(false);
    }
  }

  const total = queue ? queue.reviewed + queue.remaining : 0;
  const pct = total ? Math.round((queue!.reviewed / total) * 100) : 0;
  const titleOf = (videoId: string) => videos.find((v) => v.id === videoId)?.title || videoId;

  return (
    <>
      <div className="review-bar">
        <div className="review-filters" role="group" aria-label="Filters">
          <label className="field">
            Reason
            <select className="select" value={filters.reason ?? ""}
              onChange={(e) => setFilter("reason", e.target.value as ReviewReason | "")}>
              <option value="">All reasons</option>
              {(Object.keys(REASON_LABEL) as ReviewReason[]).map((r) => (
                <option key={r} value={r}>
                  {REASON_LABEL[r]}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            Video
            <select className="select" value={filters.video ?? ""} onChange={(e) => setFilter("video", e.target.value)}>
              <option value="">All videos</option>
              {videos.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.title || v.id}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            Status
            <select className="select" value={filters.status ?? ""}
              onChange={(e) => setFilter("status", e.target.value as ReviewFilters["status"] | "")}>
              {STATUS_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>
        </div>
        {hasPendingLlm && (
          <div className="review-retry">
            <button type="button" className="btn" disabled={retrying} onClick={() => void retry()}>
              {retrying ? "Retrying…" : "Retry LLM"}
            </button>
          </div>
        )}
        <div className="review-count" aria-live="polite">
          <p className="num">
            <strong>{queue?.reviewed ?? 0}</strong> reviewed, <strong>{queue?.remaining ?? 0}</strong> remaining
          </p>
          <div className="bar" role="progressbar" aria-label="Reviewed" aria-valuemin={0} aria-valuemax={100}
            aria-valuenow={pct}>
            <span className="bar__fill" style={{ width: `${pct}%` }} />
          </div>
        </div>
      </div>

      {retryNote && (
        <p className={`notice review-retry__note${retryNote.error ? " notice--error" : ""}`}
          role={retryNote.error ? "alert" : "status"}>
          {retryNote.text}
        </p>
      )}
      {error && (
        <div className="banner" role="alert">
          <p>Could not load the review queue: {error}</p>
          <button type="button" className="btn btn--small" onClick={load}>
            Try again
          </button>
        </div>
      )}

      {queue && clips.length === 0 ? (
        <div className="panel empty">
          {filters.status || filters.reason || filters.video
            ? "No clips match these filters."
            : "Nothing to review. Every clip in this dataset was accepted automatically or has been reviewed. To check the automatic ones, set Status to Auto-accepted."}
        </div>
      ) : (
        <div className="review">
          <nav className="queue" aria-label="Clips in this list">
            <p className="queue__count muted num">
              {clips.length ? `${at + 1} of ${clips.length}` : "Loading…"}
            </p>
            <ol ref={listRef} className="queue__list">
              {clips.map((c, i) => (
                <li key={c.id}>
                  <button type="button" className="queue__item" aria-current={i === at ? "true" : undefined}
                    onClick={() => setIndex(i)}>
                    <span className="queue__top">
                      <span className={`dot dot--${c.reason}`} aria-hidden="true" />
                      <span className="queue__reason">{REASON_LABEL[c.reason]}</span>
                      <span className="queue__cer num">{c.cer == null ? "" : `CER ${Math.round(c.cer * 100)}%`}</span>
                    </span>
                    <span className="queue__text" lang="bn">
                      {preview(c)}
                    </span>
                  </button>
                </li>
              ))}
            </ol>
          </nav>
          {clip && (
            <ReviewCard
              key={clip.id}
              clip={clip}
              videoTitle={titleOf(clip.video_id)}
              onApprove={(text) => decide(clip, { text_final: text, status: "approved" })}
              onReject={() => decide(clip, { status: "rejected" })}
              onEdit={async (text) => apply(await patchClip(clip.dataset_id, clip.id, { text_final: text }))}
              onNext={next}
              onPrev={prev}
            />
          )}
        </div>
      )}
    </>
  );
}
