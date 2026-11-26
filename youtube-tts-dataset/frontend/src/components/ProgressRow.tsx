import { STAGES, type Stage, type VideoState } from "../types";

/** Stages whose progress counts clips (n/N); the others report a percentage. */
const CLIP_STAGES: ReadonlySet<Stage> = new Set(["transcribe", "fallback", "punctuate"]);

export function formatDuration(s: number | null | undefined): string {
  if (s == null || !Number.isFinite(s)) return "–";
  const t = Math.round(s);
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const sec = String(t % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${sec}` : `${m}:${sec}`;
}

function stagePercent(v: VideoState, s: Stage): number {
  if (v.stages[s]) return 100;
  const p = v.progress[s];
  if (!p || p.total <= 0) return 0;
  return Math.min(100, Math.round((p.done / p.total) * 100));
}

function stageMeasure(v: VideoState, s: Stage): string {
  const p = v.progress[s];
  if (v.stages[s]) return "done";
  if (!p || p.total <= 0) return v.status === "running" ? "working" : "";
  return CLIP_STAGES.has(s) ? `${p.done}/${p.total} clips` : `${stagePercent(v, s)}%`;
}

interface Props {
  video: VideoState;
  onRetry: (videoId: string) => void;
  onStop: (videoId: string) => void;
  onDelete: (videoId: string) => void;
  busy?: boolean;
}

export function ProgressRow({ video: v, onRetry, onStop, onDelete, busy = false }: Props) {
  const title = v.title || v.id;
  const live = v.status === "running" || v.status === "queued";
  const current = live ? v.currentStage : null;
  return (
    <tr className={`prow prow--${v.status}`}>
      <td className="prow__title">
        <a href={v.url} target="_blank" rel="noreferrer" lang="bn">
          {title}
        </a>
        <span className="prow__meta">
          <span className="num">{formatDuration(v.duration_s)}</span>
          <span>{v.id}</span>
        </span>
        {v.warning && <span className="prow__warning">{v.warning}</span>}
      </td>
      <td className="prow__current">
        {current ? (
          <>
            <span className="prow__stage">{current}</span>
            <span className="num">{stageMeasure(v, current)}</span>
          </>
        ) : (
          <span className="muted">–</span>
        )}
      </td>
      <td>
        <div className="strip">
          {STAGES.map((s) => {
            const pct = stagePercent(v, s);
            const state = v.stages[s] ? "done" : s === current ? (v.status === "running" ? "active" : "next") : "idle";
            return (
              <div
                key={s}
                className={`strip__cell strip__cell--${state}`}
                role="progressbar"
                aria-label={s}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={pct}
                aria-valuetext={stageMeasure(v, s) || `${pct}%`}
                title={`${s}: ${stageMeasure(v, s) || "waiting"}`}
              >
                <span className="strip__fill" style={{ width: `${pct}%` }} />
              </div>
            );
          })}
        </div>
      </td>
      <td className="num">
        <span className="kept">{v.kept}</span>
        <span className="muted"> / </span>
        <span className="dropped">{v.dropped}</span>
      </td>
      <td className="num">{v.status === "running" && v.eta_s != null ? formatDuration(v.eta_s) : "–"}</td>
      <td>
        <span className={`pill pill--${v.status}`}>{v.status === "skipped" ? "stopped" : v.status}</span>
        {v.status === "failed" && v.error && <span className="prow__error">{v.error}</span>}
      </td>
      <td className="prow__actions">
        {v.status === "failed" && (
          <button type="button" className="btn btn--small" disabled={busy} aria-label={`Retry ${title}`}
            onClick={() => onRetry(v.id)}>
            Retry
          </button>
        )}
        {v.status === "skipped" && (
          <button type="button" className="btn btn--small" disabled={busy} aria-label={`Resume ${title}`}
            onClick={() => onRetry(v.id)}>
            Resume
          </button>
        )}
        {live && (
          <button type="button" className="btn btn--small" disabled={busy} aria-label={`Stop ${title}`}
            title="Stops the running stage now; Resume continues from the last finished stage"
            onClick={() => onStop(v.id)}>
            Stop
          </button>
        )}
        <button type="button" className="btn btn--small btn--danger" disabled={busy} aria-label={`Delete ${title}`}
          onClick={() => onDelete(v.id)}>
          Delete
        </button>
      </td>
    </tr>
  );
}
