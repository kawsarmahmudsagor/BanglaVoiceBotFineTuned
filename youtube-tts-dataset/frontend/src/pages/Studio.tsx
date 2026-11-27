import { useCallback, useEffect, useMemo, useRef, useState, type DragEvent, type FormEvent } from "react";
import {
  ApiError,
  UploadError,
  addStudioUrl,
  createDataset,
  listStudioVideos,
  retryVideo,
  skipVideo,
  startStudioVideo,
  studioMediaUrl,
  uploadStudioFile,
} from "../api";
import { formatDuration } from "../components/ProgressRow";
import { RangeEditor, mergeRanges } from "../components/RangeEditor";
import { STAGES, type DatasetSummary, type Stage, type StudioVideo, type VideoState } from "../types";
import { useJobEvents } from "../useJobEvents";

const POLL_MS = 5000;
const STAGE_LABELS: Record<Stage, string> = {
  download: "Download",
  cut: "Cut ranges",
  separate: "Separate vocals",
  pieces: "Find speech pieces",
  transcribe_a: "Transcribe (model A)",
  transcribe_b: "Transcribe (model B)",
  punctuate_pieces: "Punctuate pieces",
  group: "Group into clips",
  quality: "Quality checks",
  select: "Pick best text",
  punctuate: "Punctuate clips",
};

interface Props {
  dataset: DatasetSummary | null;
  datasets: DatasetSummary[];
  onSelectDataset: (id: string | null) => void;
  onDatasetCreated: (id: string) => void | Promise<void>;
}

interface Upload {
  name: string;
  sent: number;
  total: number;
}

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));
const duplicateId = (e: unknown): string | null => {
  if (!(e instanceof ApiError) || e.status !== 409) return null;
  const d = (e.body as { detail?: { video_id?: string } } | null)?.detail;
  return typeof d?.video_id === "string" ? d.video_id : null;
};

function Progress({ done, total, label }: { done?: number; total?: number; label: string }) {
  const known = total !== undefined && total > 0;
  const pct = known ? Math.min(100, Math.round(((done ?? 0) / total) * 100)) : undefined;
  return (
    <div className="bar" role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct}>
      <span className="bar__fill" style={{ width: `${pct ?? 5}%` }} />
    </div>
  );
}

/** Slot for the review panel of a finished video (filled in by the review task). */
function ReviewSlot({ video }: { video: StudioVideo }) {
  return (
    <div className="studio-review" data-video-id={video.id}>
      <p className="muted">Processing finished: {video.clips.total} clips. The review panel appears here.</p>
    </div>
  );
}

function CutPanel({ datasetId, video, onStarted }: { datasetId: string; video: StudioVideo; onStarted: () => void }) {
  const [ranges, setRanges] = useState<[number, number][]>(video.cuts ?? []);
  const [playhead, setPlayhead] = useState(0);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const duration = video.duration_s ?? 0;

  async function start() {
    setSaving(true);
    setError(null);
    try {
      await startStudioVideo(datasetId, video.id, mergeRanges(ranges));
      onStarted();
    } catch (e) {
      setError(errText(e));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="studio-cut">
      {video.media_ready ? (
        <video
          ref={videoRef}
          className="studio-video"
          controls
          src={studioMediaUrl(datasetId, video.id)}
          onTimeUpdate={(e) => setPlayhead(e.currentTarget.currentTime)}
        />
      ) : (
        <p className="notice">Preview unavailable — type the ranges.</p>
      )}
      <RangeEditor
        duration={duration}
        playhead={playhead}
        ranges={ranges}
        onChange={setRanges}
        onSeek={(t) => {
          const el = videoRef.current;
          if (el) {
            el.currentTime = t;
            try {
              void el.play()?.catch(() => {});
            } catch {
              // playback is optional
            }
          }
          setPlayhead(t);
        }}
      />
      {error && (
        <p className="notice notice--error" role="alert">
          {error}
        </p>
      )}
      <button type="button" className="btn btn--primary" disabled={saving} onClick={() => void start()}>
        Start processing
      </button>
    </div>
  );
}

export function Studio({ dataset, datasets, onSelectDataset, onDatasetCreated }: Props) {
  const datasetId = dataset?.id ?? null;
  const { videos: live } = useJobEvents(datasetId);
  const [rows, setRows] = useState<StudioVideo[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [upload, setUpload] = useState<Upload | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [newName, setNewName] = useState("");
  const [newSpeaker, setNewSpeaker] = useState("");
  const [actionError, setActionError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const seq = useRef(0);
  const datasetRef = useRef<string | null>(datasetId);
  datasetRef.current = datasetId;

  const reload = useCallback(async () => {
    if (!datasetId) return;
    const mine = ++seq.current;
    try {
      const list = await listStudioVideos(datasetId);
      if (mine === seq.current && datasetRef.current === datasetId) setRows(list);
    } catch (e) {
      if (mine === seq.current && datasetRef.current === datasetId) setMessage(errText(e));
    }
  }, [datasetId]);

  useEffect(() => {
    seq.current += 1; // drop in-flight lists of the previous dataset
    setRows([]);
    setSelected(null);
    setMessage(null);
    setUpload(null);
    void reload();
  }, [reload]);

  // Refresh when a stage finishes or a video completes (seen as a change in the live snapshot).
  const liveSig = live.map((v) => `${v.id}:${v.status}:${STAGES.filter((s) => v.stages[s]).length}`).join("|");
  const firstSig = useRef(true);
  useEffect(() => {
    if (firstSig.current) {
      firstSig.current = false;
      return;
    }
    void reload();
  }, [liveSig, reload]);

  const active = rows.some((r) => r.status === "fetching" || r.status === "queued" || r.status === "running");
  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => void reload(), POLL_MS);
    return () => clearInterval(t);
  }, [active, reload]);

  // Open the newest video by default.
  useEffect(() => {
    if (selected === null && !upload && rows.length > 0) setSelected(rows[rows.length - 1].id);
  }, [rows, selected, upload]);

  const liveById = useMemo(() => new Map<string, VideoState>(live.map((v) => [v.id, v])), [live]);
  const current = rows.find((r) => r.id === selected) ?? null;

  async function createNew(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setMessage(null);
    try {
      const r = await createDataset({ name: newName.trim(), speaker: newSpeaker.trim() || undefined, urls: "", allow_empty: true });
      setNewName("");
      setNewSpeaker("");
      await onDatasetCreated(r.dataset_id);
    } catch (err) {
      setMessage(errText(err));
    } finally {
      setBusy(false);
    }
  }

  async function addUrl(e: FormEvent) {
    e.preventDefault();
    if (!datasetId || !url.trim()) return;
    setBusy(true);
    setMessage(null);
    try {
      const r = await addStudioUrl(datasetId, url.trim());
      setUrl("");
      setSelected(r.video_id);
      await reload();
    } catch (err) {
      const dup = duplicateId(err);
      if (dup) {
        setUrl("");
        setSelected(dup);
        await reload();
      } else setMessage(errText(err));
    } finally {
      setBusy(false);
    }
  }

  async function doUpload(file: File) {
    if (!datasetId) return;
    const forDataset = datasetId;
    const stale = () => datasetRef.current !== forDataset;
    setMessage(null);
    setSelected(null);
    setUpload({ name: file.name, sent: 0, total: file.size });
    try {
      const r = await uploadStudioFile(forDataset, file, (sent, total) => {
        if (!stale()) setUpload({ name: file.name, sent, total });
      });
      if (!stale()) setSelected(r.video_id);
    } catch (err) {
      if (stale()) return;
      if (err instanceof UploadError && err.status === 409 && err.video_id) setSelected(err.video_id);
      else setMessage(errText(err));
    } finally {
      if (!stale()) {
        setUpload(null);
        await reload();
      }
    }
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer?.files?.[0];
    if (file) void doUpload(file);
  }

  async function act(fn: () => Promise<unknown>) {
    setActionError(null);
    try {
      await fn();
    } catch (err) {
      setActionError(errText(err));
    }
    await reload();
  }

  function renderMain() {
    if (!dataset) return <p className="empty">Create or select a dataset first.</p>;
    if (upload && !current) {
      return (
        <div>
          <h2>Uploading {upload.name}</h2>
          <Progress done={upload.sent} total={upload.total} label="Upload progress" />
        </div>
      );
    }
    if (!current) return <p className="empty">Add a YouTube URL or drop a video to begin.</p>;
    const lv = liveById.get(current.id);
    const title = current.title || current.id;
    return (
      <div>
        <h2>{title}</h2>
        {current.warning && <p className="notice">{current.warning}</p>}
        {current.status === "fetching" && (
          <div>
            <p className="muted">Fetching the video…</p>
            <Progress done={lv?.progress.download?.done} total={lv?.progress.download?.total} label="Download progress" />
          </div>
        )}
        {current.status === "awaiting_cut" && (
          <CutPanel key={current.id} datasetId={dataset.id} video={current} onStarted={() => void reload()} />
        )}
        {(current.status === "queued" || current.status === "running") && (
          <div>
            <ol className="studio-stages">
              {STAGES.map((s) => {
                const done = Boolean(current.stages?.[s]);
                const p = lv?.progress[s];
                return (
                  <li key={s} className={done ? "studio-stage studio-stage--done" : "studio-stage"}>
                    <span>{STAGE_LABELS[s]}</span>
                    <span className="muted">{done ? "done" : p && p.total > 0 ? `${Math.round((p.done / p.total) * 100)}%` : ""}</span>
                  </li>
                );
              })}
            </ol>
            <button type="button" className="btn btn--danger" onClick={() => void act(() => skipVideo(dataset.id, current.id))}>
              Stop
            </button>
          </div>
        )}
        {current.status === "failed" && (
          <div>
            <p className="notice notice--error" role="alert">
              {current.error || "Processing failed."}
            </p>
            <button type="button" className="btn" onClick={() => void act(() => retryVideo(dataset.id, current.id))}>
              Retry
            </button>
          </div>
        )}
        {current.status === "skipped" && <p className="muted">Stopped.</p>}
        {current.status === "done" && <ReviewSlot video={current} />}
        {actionError && (
          <p className="notice notice--error" role="alert">
            {actionError}
          </p>
        )}
      </div>
    );
  }

  return (
    <div className="studio">
      <aside className="studio__side panel">
        <label className="field">
          Dataset
          <select className="select" value={datasetId ?? ""} onChange={(e) => onSelectDataset(e.target.value || null)}>
            {datasets.length === 0 && <option value="">No datasets yet</option>}
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name}
              </option>
            ))}
          </select>
        </label>
        <details className="studio__new">
          <summary>New dataset</summary>
          <form onSubmit={(e) => void createNew(e)}>
            <label className="field">
              Name
              <input className="input" value={newName} onChange={(e) => setNewName(e.target.value)} />
            </label>
            <label className="field">
              Speaker
              <input className="input" value={newSpeaker} onChange={(e) => setNewSpeaker(e.target.value)} />
            </label>
            <button type="submit" className="btn btn--small" disabled={busy || !newName.trim()}>
              Create dataset
            </button>
          </form>
        </details>

        <form className="studio__url" onSubmit={(e) => void addUrl(e)}>
          <label className="field">
            YouTube URL
            <input
              className="input"
              value={url}
              placeholder="https://www.youtube.com/watch?v=..."
              onChange={(e) => setUrl(e.target.value)}
              disabled={!dataset}
            />
          </label>
          <button type="submit" className="btn btn--primary" disabled={!dataset || busy || !url.trim()}>
            Add
          </button>
        </form>

        <div
          className={dragOver ? "dropzone dropzone--over" : "dropzone"}
          data-testid="dropzone"
          role="button"
          tabIndex={0}
          aria-label="Drop a video file or click to choose one"
          onClick={() => fileInput.current?.click()}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") fileInput.current?.click();
          }}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
        >
          Drop a video or audio file here, or click to choose
          <input
            ref={fileInput}
            type="file"
            accept="video/*,audio/*"
            className="visually-hidden"
            aria-label="Choose a video file"
            tabIndex={-1}
            onClick={(e) => e.stopPropagation()}
            onChange={(e) => {
              const f = e.target.files?.[0];
              e.target.value = "";
              if (f) void doUpload(f);
            }}
          />
        </div>
        {message && (
          <p className="notice notice--error" role="alert">
            {message}
          </p>
        )}

        <ul className="studio__list">
          {rows.map((r) => {
            const lv = liveById.get(r.id);
            const stage = lv?.currentStage;
            const p = stage ? lv?.progress[stage] : undefined;
            const extra =
              r.status === "running" && stage ? ` · ${STAGE_LABELS[stage]}${p && p.total > 0 ? ` ${Math.round((p.done / p.total) * 100)}%` : ""}` : "";
            return (
              <li key={r.id}>
                <button
                  type="button"
                  className="studio__item"
                  aria-current={r.id === selected ? "true" : undefined}
                  onClick={() => setSelected(r.id)}
                >
                  <span className="studio__title">{r.title || r.id}</span>
                  <span className={`pill pill--${r.status}`}>{r.status.replace("_", " ")}</span>
                  <span className="muted">
                    {formatDuration(r.duration_s)}
                    {extra}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      </aside>
      <section className="studio__main panel">{renderMain()}</section>
    </div>
  );
}
