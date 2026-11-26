import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import { ApiError, exportDataset, getStats } from "../api";
import type { DatasetStats, DatasetSummary, ExportSummary } from "../types";
import { useJobEvents } from "../useJobEvents";

const DROP_LABEL: Record<string, string> = {
  dnsmos: "Low audio quality (DNSMOS)",
  other_speaker: "Another speaker",
  length: "Too short or too long",
  rejected: "Rejected in review",
  unreviewed: "Not reviewed yet",
  other: "Other",
};
// per-video column: dropped/rejected clips counted by drop_reason or status
const DROP_SHORT: Record<string, string> = {
  dnsmos: "low quality",
  other_speaker: "other speaker",
  length: "length",
  rejected: "rejected",
};
const DROP_ORDER = ["dnsmos", "other_speaker", "length", "rejected", "unreviewed"];
const SPLITS = ["train", "val", "test"] as const;

export function formatHours(h: number): string {
  if (!Number.isFinite(h) || h <= 0) return "0 min";
  return h < 1 ? `${(h * 60).toFixed(1)} min` : `${h.toFixed(2)} h`;
}

function plural(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

function dropText(dropped: Record<string, number>): string {
  const parts = Object.entries(dropped)
    .filter(([, n]) => n > 0)
    .map(([k, n]) => `${n} ${DROP_SHORT[k] ?? k}`);
  return parts.length ? parts.join(", ") : "none";
}

function summaryText(s: ExportSummary): string[] {
  const lines = [`Exported ${plural(s.clips, "clip")}${s.hours > 0 ? ` (${formatHours(s.hours)})` : ""} to:`];
  const left = [
    s.process_error && `${s.process_error} could not be processed`,
    s.empty_text_excluded && `${s.empty_text_excluded} had no text`,
    s.validation_dropped && `${s.validation_dropped} failed the audio checks`,
  ].filter(Boolean);
  if (left.length) lines.push(`Left out: ${left.join(", ")}.`);
  if (s.unreviewed_translated_skipped)
    lines.push(`${plural(s.unreviewed_translated_skipped, "unreviewed translated clip")} left out.`);
  return lines;
}

function unreviewedOf(e: unknown): number | null {
  if (!(e instanceof ApiError) || e.status !== 409) return null;
  const n = (e.body as { unreviewed?: unknown } | null)?.unreviewed;
  return typeof n === "number" ? n : null;
}

function UnreviewedDialog({ count, onSkip, onClose }: { count: number; onSkip: () => void; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    if (d && !d.open) d.showModal?.();
    d?.querySelector<HTMLElement>("[data-initial-focus]")?.focus();
  }, []);
  return (
    <dialog ref={ref} className="dialog" aria-labelledby="unrev-title" aria-describedby="unrev-body"
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}>
      <h2 id="unrev-title">{plural(count, "translated clip")} not reviewed</h2>
      <p id="unrev-body">
        Their text is the LLM's translation of the English transcript, and nobody has checked it yet. Review them
        first, or export now without them.
      </p>
      <div className="dialog__actions">
        <a className="btn btn--primary" href="#/review" onClick={onClose}>
          Review clips
        </a>
        <button type="button" className="btn" onClick={onSkip}>
          Export without them
        </button>
        <button type="button" className="btn" onClick={onClose} data-initial-focus>
          Cancel
        </button>
      </div>
    </dialog>
  );
}

export function DatasetPage({ dataset }: { dataset: DatasetSummary | null }) {
  const id = dataset?.id ?? null;
  const { exportState, markExportStarted } = useJobEvents(id);
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [statsError, setStatsError] = useState<string | null>(null);
  const [outDir, setOutDir] = useState("");
  const [starting, setStarting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [unreviewed, setUnreviewed] = useState<number | null>(null);
  const exportBtn = useRef<HTMLButtonElement>(null);

  const loadStats = useCallback(() => {
    if (!id) return;
    setStatsError(null);
    getStats(id).then(setStats, (e: unknown) => setStatsError(e instanceof Error ? e.message : String(e)));
  }, [id]);

  useEffect(() => {
    setStats(null);
    setExportError(null);
    setUnreviewed(null);
    loadStats();
  }, [loadStats]);

  useEffect(() => {
    if (exportState.status === "done") loadStats();
  }, [exportState.status, loadStats]);

  async function startExport(skipUnreviewed: boolean) {
    if (!id) return;
    setUnreviewed(null);
    setExportError(null);
    setStarting(true);
    try {
      await exportDataset(id, { out_dir: outDir.trim() || undefined, skip_unreviewed_translated: skipUnreviewed || undefined });
      markExportStarted();
    } catch (e) {
      const n = unreviewedOf(e);
      if (n !== null) setUnreviewed(n);
      else setExportError(e instanceof Error ? e.message : String(e));
    } finally {
      setStarting(false);
    }
  }

  // focus goes back to Export once the modal dialog is gone (it is inert while the dialog is open)
  const refocus = useRef(false);
  useEffect(() => {
    if (unreviewed === null && refocus.current) {
      refocus.current = false;
      exportBtn.current?.focus();
    }
  }, [unreviewed]);

  function closeDialog() {
    refocus.current = true;
    setUnreviewed(null);
  }

  if (!dataset) {
    return <div className="panel empty">Pick a dataset above to see what it holds and export it.</div>;
  }

  const running = exportState.status === "running";
  const pct = exportState.total > 0 ? Math.round((exportState.done / exportState.total) * 100) : 0;
  const droppedKeys = stats
    ? [...DROP_ORDER, ...Object.keys(stats.dropped).filter((k) => !DROP_ORDER.includes(k))]
    : [];
  const maxDrop = stats ? Math.max(1, ...Object.values(stats.dropped)) : 1;
  const splitHours = stats ? SPLITS.reduce((n, s) => n + stats.splits[s].hours, 0) : 0;

  return (
    <>
      {statsError && (
        <div className="banner" role="alert">
          <p>Could not load the dataset stats: {statsError}</p>
          <button type="button" className="btn btn--small" onClick={loadStats}>
            Try again
          </button>
        </div>
      )}

      <div className="ds-grid">
        <section className="panel ds-kept" aria-labelledby="kept-title">
          <h2 id="kept-title">Kept for training</h2>
          <p className="overall__figure num">{stats ? formatHours(stats.hours_kept) : "–"}</p>
          <p className="muted num">{stats ? plural(stats.clips_kept, "clip") : "Loading…"}</p>
        </section>

        <section className="panel" aria-labelledby="dropped-title">
          <h2 id="dropped-title">Not kept</h2>
          <ul className="drops">
            {droppedKeys.map((k) => {
              const n = stats?.dropped[k] ?? 0;
              return (
                <li key={k} className="drops__row">
                  <span>{DROP_LABEL[k] ?? k}</span>
                  <span className="drops__bar" aria-hidden="true">
                    <span style={{ width: `${(n / maxDrop) * 100}%` }} />
                  </span>
                  <span className="num">{n}</span>
                </li>
              );
            })}
          </ul>
        </section>

        <section className="panel ds-splits" aria-labelledby="splits-title">
          <h2 id="splits-title">Split preview</h2>
          <div className="splitbar" aria-hidden="true">
            {SPLITS.map((s) => (
              <span key={s} className={`splitbar__part splitbar__part--${s}`}
                style={{ width: `${stats && splitHours > 0 ? (stats.splits[s].hours / splitHours) * 100 : 0}%` }} />
            ))}
          </div>
          <table className="table table--compact">
            <thead>
              <tr>
                <th scope="col">Split</th>
                <th scope="col">Clips</th>
                <th scope="col">Audio</th>
                <th scope="col">Share</th>
              </tr>
            </thead>
            <tbody className="num">
              {SPLITS.map((s) => (
                <tr key={s}>
                  <th scope="row">
                    <span className={`swatch swatch--${s}`} aria-hidden="true" />
                    {s}
                  </th>
                  <td>{stats?.splits[s].clips ?? "–"}</td>
                  <td>{stats ? formatHours(stats.splits[s].hours) : "–"}</td>
                  <td>{stats && splitHours > 0 ? `${Math.round((stats.splits[s].hours / splitHours) * 100)}%` : "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>

      <section className="panel ds-export" aria-labelledby="export-title">
        <h2 id="export-title">Export</h2>
        <form className="ds-export__form" onSubmit={(e: FormEvent) => {
          e.preventDefault();
          void startExport(false);
        }}>
          <label className="field ds-export__path">
            <span>
              Output folder{" "}
              <span className="field__hint">
                leave empty for <span lang="bn">{dataset.name}</span> in the default export folder (YTDS_EXPORT_DIR)
              </span>
            </span>
            <input className="input" value={outDir} onChange={(e) => setOutDir(e.target.value)} spellCheck={false}
              autoComplete="off" placeholder={`…\\var\\import\\${dataset.name}`}
              aria-describedby={exportError ? "export-error" : undefined} />
          </label>
          <button ref={exportBtn} type="submit" className="btn btn--primary" disabled={running || starting}>
            {running ? "Exporting…" : starting ? "Starting…" : "Export"}
          </button>
        </form>
        <p className="field__hint ds-export__note">
          Replaces a previous export in the same folder. A folder that holds anything else is left alone.
        </p>
        {exportError && (
          <p id="export-error" className="notice notice--error" role="alert">
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
            {exportState.status === "done" && exportState.summary && (
              <>
                {summaryText(exportState.summary).map((l, i) => (
                  <p key={l}>
                    {l}
                    {i === 0 && <span className="ds-export__out">{exportState.summary!.out_dir}</span>}
                  </p>
                ))}
              </>
            )}
            {exportState.status === "failed" && (
              <p className="notice--error" role="alert">
                Export failed: {exportState.error}
              </p>
            )}
          </div>
        )}
      </section>

      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr>
              <th scope="col">Video</th>
              <th scope="col">Status</th>
              <th scope="col">Clips kept</th>
              <th scope="col">Audio kept</th>
              <th scope="col">Not kept</th>
            </tr>
          </thead>
          <tbody>
            {stats?.videos.map((v) => (
              <tr key={v.video_id}>
                <td className="prow__title">
                  <span lang="bn">{v.title || v.video_id}</span>
                  <span className="prow__meta">{v.video_id}</span>
                </td>
                <td>
                  <span className={`pill pill--${v.status}`}>{v.status}</span>
                </td>
                <td className="num kept">{v.clips_kept}</td>
                <td className="num">{formatHours(v.hours_kept)}</td>
                <td className="muted">{dropText(v.dropped)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {stats && stats.videos.length === 0 && (
          <p className="empty">No videos yet. Add YouTube URLs on the Jobs page.</p>
        )}
      </div>

      {unreviewed !== null && (
        <UnreviewedDialog count={unreviewed} onClose={closeDialog} onSkip={() => void startExport(true)} />
      )}
    </>
  );
}
