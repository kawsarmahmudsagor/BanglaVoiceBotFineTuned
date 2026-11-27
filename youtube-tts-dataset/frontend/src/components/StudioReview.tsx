import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { exportDataset, getVideoClips, patchClip } from "../api";
import { summaryText, unreviewedOf, UnreviewedDialog } from "../pages/Dataset";
import type { ClipFlag, ReviewClip, StudioClip } from "../types";
import { useJobEvents } from "../useJobEvents";
import { ReviewCard } from "./ReviewCard";

type Filter = "needs" | "all" | "approved" | "rejected" | "dropped";

const FLAG_LABEL: Record<ClipFlag, string> = {
  incomplete_sentence: "Incomplete sentence",
  unsplit: "Not split by LLM",
  disagree: "Models disagree",
};
const DROP_LABEL: Record<string, string> = {
  bad_stt: "Broken transcript",
  length: "Too short or too long",
  dnsmos: "Low audio quality",
  other_speaker: "Another speaker",
};
const ATTENTION = ["flagged", "needs_review", "pending_llm"];

const PREDICATE: Record<Filter, (c: StudioClip) => boolean> = {
  needs: (c) => (c.status != null && ATTENTION.includes(c.status)) || c.flag != null,
  all: () => true,
  approved: (c) => c.status === "approved",
  rejected: (c) => c.status === "rejected",
  dropped: (c) => c.status === "dropped",
};
const LABEL: Record<Filter, string> = {
  needs: "Needs attention",
  all: "All",
  approved: "Approved",
  rejected: "Rejected",
  dropped: "Dropped",
};
const ORDER: Filter[] = ["needs", "all", "approved", "rejected", "dropped"];

/** Seconds as m:ss.s (truncated to a tenth). */
export function formatOrig(s: number): string {
  const t = Math.floor(Math.max(0, s) * 10) / 10;
  const m = Math.floor(t / 60);
  return `${m}:${(t - m * 60).toFixed(1).padStart(4, "0")}`;
}

function toReviewClip(c: StudioClip): ReviewClip | null {
  if (c.status == null) return null;
  return { ...c, status: c.status, reason: c.reason ?? (c.flag === "disagree" ? "disagree" : "auto") };
}

function errText(e: unknown) {
  return e instanceof Error ? e.message : String(e);
}

export function StudioReview({ datasetId, videoId }: { datasetId: string; videoId: string }) {
  const [clips, setClips] = useState<StudioClip[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const [unreviewed, setUnreviewed] = useState<number | null>(null);
  const [starting, setStarting] = useState(false);
  const { exportState, markExportStarted } = useJobEvents(datasetId);
  const exportBtn = useRef<HTMLButtonElement>(null);

  const load = useCallback(() => {
    setLoadError(null);
    getVideoClips(datasetId, videoId).then(setClips, (e: unknown) => setLoadError(errText(e)));
  }, [datasetId, videoId]);
  useEffect(() => {
    setClips(null);
    load();
  }, [load]);

  const counts = useMemo(() => {
    const out = {} as Record<Filter, number>;
    for (const f of ORDER) out[f] = (clips ?? []).filter(PREDICATE[f]).length;
    return out;
  }, [clips]);
  const active: Filter = filter ?? (counts.needs > 0 ? "needs" : "all");
  const shown = useMemo(() => (clips ?? []).filter(PREDICATE[active]), [clips, active]);
  const current = shown.find((c) => c.id === selected) ?? shown[0] ?? null;

  function replace(updated: Partial<StudioClip> & { id: string }) {
    setClips((cs) => cs && cs.map((c) => (c.id === updated.id ? { ...c, ...updated } : c)));
  }
  const merge = (c: StudioClip, r: ReviewClip) => ({
    id: c.id,
    status: r.status,
    text_final: r.text_final,
    text_pick: r.text_pick,
  });
  function step(d: number) {
    if (!current) return;
    const i = shown.findIndex((c) => c.id === current.id);
    const n = shown[Math.min(shown.length - 1, Math.max(0, i + d))];
    if (n) setSelected(n.id);
  }

  async function startExport(skip: boolean) {
    setUnreviewed(null);
    setExportError(null);
    setStarting(true);
    try {
      await exportDataset(datasetId, skip ? { skip_unreviewed_translated: true } : {});
      markExportStarted();
    } catch (e) {
      const n = unreviewedOf(e);
      if (n !== null) setUnreviewed(n);
      else setExportError(errText(e));
    } finally {
      setStarting(false);
    }
  }

  const running = exportState.status === "running";
  const pct = exportState.total > 0 ? Math.round((exportState.done / exportState.total) * 100) : 0;
  const readOnly = current != null && (current.status === "dropped" || current.status == null);

  return (
    <section className="studio-review" data-video-id={videoId} aria-label="Review clips">
      <div className="studio-review__bar">
        <div className="studio-review__filters" role="group" aria-label="Filter clips">
          {ORDER.map((f) => (
            <button key={f} type="button" className={`btn btn--small${f === active ? " btn--primary" : ""}`}
              aria-pressed={f === active} onClick={() => setFilter(f)}>
              {`${LABEL[f]} (${counts[f]})`}
            </button>
          ))}
        </div>
        <button ref={exportBtn} type="button" className="btn btn--primary" disabled={running || starting}
          onClick={() => void startExport(false)}>
          Export dataset
        </button>
      </div>
      {exportError && (
        <p className="notice notice--error" role="alert">
          {exportError}
        </p>
      )}
      {exportState.status !== "idle" && (
        <div className="ds-export__status" aria-live="polite">
          <div className="bar" role="progressbar" aria-label="Export progress" aria-valuemin={0} aria-valuemax={100}
            aria-valuenow={exportState.status === "done" ? 100 : pct}>
            <span className={`bar__fill${exportState.status === "failed" ? " bar__fill--failed" : ""}`}
              style={{ width: `${exportState.status === "done" ? 100 : pct}%` }} />
          </div>
          {running && (
            <p className="num">
              {exportState.total ? `Writing clip ${exportState.done} of ${exportState.total}` : "Preparing…"}
            </p>
          )}
          {exportState.status === "done" &&
            exportState.summary &&
            summaryText(exportState.summary).map((l, i) => (
              <p key={l}>
                {l}
                {i === 0 && <span className="ds-export__out">{exportState.summary!.out_dir}</span>}
              </p>
            ))}
          {exportState.status === "failed" && (
            <p className="notice--error" role="alert">
              Export failed: {exportState.error}
            </p>
          )}
        </div>
      )}
      {loadError && (
        <div className="banner" role="alert">
          <p>Could not load the clips: {loadError}</p>
          <button type="button" className="btn btn--small" onClick={load}>
            Try again
          </button>
        </div>
      )}
      {clips === null && !loadError && <p className="muted">Loading clips…</p>}
      {clips !== null && shown.length === 0 && <p className="empty">No clips in this filter.</p>}

      <ul className="studio-review__list">
        {shown.map((c) => (
          <li key={c.id}>
            <button type="button" className="btn btn--small" aria-current={current?.id === c.id}
              onClick={() => setSelected(c.id)}>
              <span className="num">{`${formatOrig(c.orig_start_s)}–${formatOrig(c.orig_end_s)}`}</span>
              {c.flag && <span className="pill">{FLAG_LABEL[c.flag]}</span>}
              {c.status && <span className={`pill pill--${c.status}`}>{c.status}</span>}
            </button>
          </li>
        ))}
      </ul>

      {current && readOnly && (
        <div className="panel">
          {current.status === "dropped" ? (
            <p>
              Dropped: <strong>{DROP_LABEL[current.drop_reason ?? ""] ?? current.drop_reason ?? "unknown reason"}</strong>
            </p>
          ) : (
            <p className="muted">Processing…</p>
          )}
          {current.text_a && (
            <p lang="bn" className="muted">
              {current.text_a}
            </p>
          )}
        </div>
      )}
      {current && !readOnly && (
        <ReviewCard key={current.id} clip={toReviewClip(current)!}
          onApprove={async (text) =>
            replace(merge(current, await patchClip(datasetId, current.id, { text_final: text, status: "approved" })))}
          onReject={async () =>
            replace(merge(current, await patchClip(datasetId, current.id, { status: "rejected" })))}
          onEdit={async (text) => {
            // only text_final: a late response must not write an older status back
            const r = await patchClip(datasetId, current.id, { text_final: text });
            replace({ id: current.id, text_final: r.text_final });
          }}
          onSwapped={(r) => replace(merge(current, r))}
          onNext={() => step(1)} onPrev={() => step(-1)} />
      )}

      {unreviewed !== null && (
        <UnreviewedDialog count={unreviewed}
          onClose={() => {
            setUnreviewed(null);
            exportBtn.current?.focus();
          }}
          onSkip={() => void startExport(true)} />
      )}
    </section>
  );
}
