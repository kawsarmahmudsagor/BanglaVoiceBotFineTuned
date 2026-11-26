import { useCallback, useEffect, useState, type FormEvent } from "react";
import { ApiError, addVideos, createDataset, deleteVideo, getHealth, retryVideo, skipVideo } from "../api";
import { ProgressRow, formatDuration } from "../components/ProgressRow";
import { STAGES, type AddVideosResult, type DatasetSummary, type Health, type UrlError, type VideoState } from "../types";
import { useJobEvents } from "../useJobEvents";

const HEALTH_POLL_MS = 60_000;

/** Job completion over non-skipped videos: whole stages done plus the running stage's fraction. */
export function overallProgress(videos: VideoState[]): number {
  const active = videos.filter((v) => v.status !== "skipped");
  if (!active.length) return 0;
  let sum = 0;
  for (const v of active) {
    if (v.status === "done") {
      sum += 1;
      continue;
    }
    let units = STAGES.filter((s) => v.stages[s]).length;
    const cur = v.currentStage && !v.stages[v.currentStage] ? v.progress[v.currentStage] : undefined;
    if (cur && cur.total > 0) units += Math.min(1, cur.done / cur.total);
    sum += units / STAGES.length;
  }
  return Math.round((sum / active.length) * 100);
}

function plural(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

function resultText(r: AddVideosResult): string {
  const parts = [`Added ${plural(r.added, "video")}`];
  if (r.duplicates) parts.push(`${r.duplicates} already in the dataset`);
  if (r.errors.length) parts.push(`${plural(r.errors.length, "URL")} could not be read`);
  return parts.join(", ") + ".";
}

function errorsOf(e: unknown): UrlError[] {
  if (!(e instanceof ApiError)) return [];
  const detail = (e.body as { detail?: { errors?: UrlError[] } } | null)?.detail;
  return Array.isArray(detail?.errors) ? detail.errors : [];
}

function useHealth(): [Health | null, () => void] {
  const [health, setHealth] = useState<Health | null>(null);
  const check = useCallback(() => {
    getHealth().then(setHealth, () => setHealth(null));
  }, []);
  useEffect(() => {
    check();
    const t = setInterval(check, HEALTH_POLL_MS);
    return () => clearInterval(t);
  }, [check]);
  return [health, check];
}

interface Props {
  dataset: DatasetSummary | null;
  onDatasetCreated: (id: string) => void | Promise<void>;
  onVideosChanged: () => void;
}

export function Jobs({ dataset, onDatasetCreated, onVideosChanged }: Props) {
  const { videos, connected, log, refresh } = useJobEvents(dataset?.id ?? null);
  const [health, recheckHealth] = useHealth();
  const [mode, setMode] = useState<"new" | "add">(dataset ? "add" : "new");
  const [name, setName] = useState("");
  const [speaker, setSpeaker] = useState("");
  const [urls, setUrls] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<{ text: string; errors: UrlError[]; failed: boolean } | null>(null);
  const [busyVideo, setBusyVideo] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const adding = mode === "add" && dataset !== null;
  const datasetId = dataset?.id;

  useEffect(() => {
    setMode(datasetId ? "add" : "new");
  }, [datasetId]);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setResult(null);
    try {
      const r = adding
        ? await addVideos(dataset.id, urls)
        : await createDataset({ name: name.trim(), speaker: speaker.trim() || undefined, urls });
      setResult({ text: resultText(r), errors: r.errors, failed: false });
      setUrls("");
      if (adding) {
        refresh();
        onVideosChanged();
      } else {
        setName("");
        setSpeaker("");
        await onDatasetCreated(r.dataset_id);
      }
    } catch (err) {
      setResult({ text: err instanceof Error ? err.message : String(err), errors: errorsOf(err), failed: true });
    } finally {
      setSubmitting(false);
    }
  }

  async function act(videoId: string, fn: () => Promise<unknown>) {
    setBusyVideo(videoId);
    setActionError(null);
    try {
      await fn();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyVideo(null);
      refresh();
    }
  }

  const id = dataset?.id ?? "";
  const onRetry = (v: string) => act(v, () => retryVideo(id, v));
  const onStop = (v: string) => act(v, () => skipVideo(id, v));
  const onDelete = (v: string) => {
    const title = videos.find((x) => x.id === v)?.title || v;
    if (!window.confirm(`Delete "${title}" and its clips from this dataset?`)) return;
    void act(v, async () => {
      await deleteVideo(id, v);
      onVideosChanged();
    });
  };

  const pct = overallProgress(videos);
  const count = (s: VideoState["status"]) => videos.filter((v) => v.status === s).length;
  const kept = videos.reduce((n, v) => n + v.kept, 0);
  const dropped = videos.reduce((n, v) => n + v.dropped, 0);
  const totalS = videos.reduce((n, v) => n + (v.duration_s ?? 0), 0);
  const canSubmit = urls.trim() !== "" && (adding || name.trim() !== "") && !submitting;

  return (
    <>
      {health && !health.llm_ok && (
        <div className="banner" role="alert">
          <p>
            The LLM is not reachable. Downloads, separation and transcription keep running; punctuation and
            translation wait until it is back. Check the host and model in Settings.
          </p>
          <button type="button" className="btn btn--small" onClick={recheckHealth}>
            Check again
          </button>
        </div>
      )}

      <div className="jobs-top">
        <section className="panel" aria-labelledby="jobs-form-title">
          <h2 id="jobs-form-title">{adding ? `Add videos to ${dataset.name}` : "New dataset"}</h2>
          <form className="jobs-form" onSubmit={submit}>
            {!adding && (
              <>
                <label className="field">
                  Dataset name
                  <input className="input" value={name} onChange={(e) => setName(e.target.value)} required
                    placeholder="e.g. news-anchor-1" autoComplete="off" />
                </label>
                <label className="field">
                  <span>
                    Speaker <span className="field__hint">(optional, defaults to the dataset name)</span>
                  </span>
                  <input className="input" value={speaker} onChange={(e) => setSpeaker(e.target.value)} autoComplete="off" />
                </label>
              </>
            )}
            <label className="field span-2">
              <span>
                YouTube URLs <span className="field__hint">one per line; playlists and channels are expanded</span>
              </span>
              <textarea className="textarea" value={urls} onChange={(e) => setUrls(e.target.value)} required
                spellCheck={false} placeholder="https://www.youtube.com/watch?v=…" />
            </label>
            <div className="jobs-form__actions span-2">
              <button type="submit" className="btn btn--primary" disabled={!canSubmit}>
                {submitting ? "Reading URLs…" : adding ? "Add videos" : "Start"}
              </button>
              {dataset && (
                <button type="button" className="btn btn--link" onClick={() => setMode(adding ? "new" : "add")}>
                  {adding ? "Start a new dataset instead" : `Add to ${dataset.name} instead`}
                </button>
              )}
            </div>
          </form>
          {result && (
            <div className={`notice${result.failed ? " notice--error" : ""}`} role="status">
              {result.text}
              {result.errors.length > 0 && (
                <ul>
                  {result.errors.map((e) => (
                    <li key={e.url}>
                      {e.url}: {e.error}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </section>

        <section className="panel overall" aria-labelledby="overall-title">
          <h2 id="overall-title">{dataset ? dataset.name : "No dataset selected"}</h2>
          <p className="overall__figure num">
            {pct}% <small>complete</small>
          </p>
          <div className="bar" role="progressbar" aria-label="Overall progress" aria-valuemin={0} aria-valuemax={100}
            aria-valuenow={pct}>
            <span className="bar__fill" style={{ width: `${pct}%` }} />
          </div>
          <dl className="num">
            <dt>Videos</dt>
            <dd>
              {count("done")} done, {count("running")} running, {count("queued")} queued
              {count("failed") > 0 && `, ${count("failed")} failed`}
            </dd>
            <dt>Audio</dt>
            <dd>{formatDuration(totalS)}</dd>
            <dt>Clips</dt>
            <dd>
              {kept} kept, {dropped} dropped
            </dd>
            <dt>Updates</dt>
            <dd>
              <span className={`conn${connected ? " conn--live" : ""}`}>
                {dataset ? (connected ? "Live" : "Reconnecting…") : "Idle"}
              </span>
            </dd>
          </dl>
        </section>
      </div>

      {actionError && (
        <p className="notice notice--error" role="alert">
          {actionError}
        </p>
      )}

      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr>
              <th scope="col">Video</th>
              <th scope="col">Current stage</th>
              <th scope="col">
                <span className="visually-hidden">Stages: </span>
                <span className="strip-legend" aria-hidden="true" title={STAGES.join(", ")}>
                  {STAGES.map((s) => (
                    <span key={s} title={s}>
                      {s.slice(0, 3)}
                    </span>
                  ))}
                </span>
              </th>
              <th scope="col">Kept / dropped</th>
              <th scope="col">ETA</th>
              <th scope="col">Status</th>
              <th scope="col">
                <span className="visually-hidden">Actions</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {videos.map((v) => (
              <ProgressRow key={v.id} video={v} busy={busyVideo === v.id} onRetry={onRetry} onStop={onStop}
                onDelete={onDelete} />
            ))}
          </tbody>
        </table>
        {videos.length === 0 && (
          <p className="empty">
            {dataset ? "No videos in this dataset. Paste YouTube URLs above to add some." : "Paste YouTube URLs above and press Start to build a dataset."}
          </p>
        )}
      </div>

      <details className="log">
        <summary>Live log ({log.length})</summary>
        <pre aria-live="off">{log.length ? log.join("\n") : "No events yet."}</pre>
      </details>
    </>
  );
}
