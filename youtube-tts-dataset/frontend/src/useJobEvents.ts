import { useCallback, useEffect, useRef, useState } from "react";
import { eventsUrl, getDataset } from "./api";
import { STAGES, type ExportState, type ExportSummary, type JobEvent, type Stage, type VideoDetail, type VideoState } from "./types";

const BACKOFF_S = [1, 2, 4, 8, 10];
const LOG_MAX = 500;

export function toVideoState(v: VideoDetail): VideoState {
  return { ...v, progress: {}, currentStage: STAGES.find((s) => !v.stages[s]) ?? null, eta_s: null };
}

const isStage = (s: unknown): s is Stage => STAGES.includes(s as Stage);

/** Applies one WebSocket event to the video list (pure). Events without a known video are ignored. */
export function applyEvent(videos: VideoState[], e: JobEvent): VideoState[] {
  if (!e.video_id || !videos.some((v) => v.id === e.video_id)) return videos;
  return videos.map((v) => {
    if (v.id !== e.video_id) return v;
    switch (e.type) {
      case "progress":
        if (!isStage(e.stage)) return v;
        return {
          ...v,
          status: v.status === "queued" ? "running" : v.status,
          currentStage: e.stage,
          progress: { ...v.progress, [e.stage]: { done: e.done ?? 0, total: e.total ?? 0 } },
          eta_s: e.eta_s ?? null,
        };
      case "stage_done": {
        if (!isStage(e.stage)) return v;
        const stages = { ...v.stages, [e.stage]: true };
        return { ...v, stages, currentStage: STAGES.find((s) => !stages[s]) ?? null, eta_s: null };
      }
      case "stage_failed":
        return { ...v, status: "failed", error: e.error ?? "stage failed", eta_s: null };
      case "video_done":
        return { ...v, status: "done", warning: e.message ?? null, currentStage: null, eta_s: null };
      default:
        return v;
    }
  });
}

export const EXPORT_IDLE: ExportState = { status: "idle", done: 0, total: 0, summary: null, error: null };

/** Applies one WebSocket event to the export state (pure). Only events with stage "export" count. */
export function applyExportEvent(state: ExportState, e: JobEvent): ExportState {
  if (e.stage !== "export") return state;
  switch (e.type) {
    case "progress":
      return { status: "running", done: e.done ?? 0, total: e.total ?? 0, summary: null, error: null };
    case "stage_done":
      return { ...state, status: "done", done: state.total, summary: (e.summary as unknown as ExportSummary) ?? null };
    case "stage_failed":
      return { ...state, status: "failed", error: e.error ?? "export failed" };
    default:
      return state;
  }
}

function logLine(e: JobEvent, titles: Map<string, string>): string | null {
  const who = e.video_id ? titles.get(e.video_id) ?? e.video_id : "dataset";
  const time = new Date().toLocaleTimeString();
  switch (e.type) {
    case "stage_done":
      return `${time}  ${who}: ${e.stage} done`;
    case "stage_failed":
      return `${time}  ${who}: ${e.stage} failed: ${e.error ?? "unknown error"}`;
    case "video_done":
      return `${time}  ${who}: finished${e.message ? ` (${e.message})` : ""}`;
    case "log":
    case "llm_status":
      return `${time}  ${who}: ${e.message ?? ""}`;
    default:
      return null;
  }
}

export interface JobEvents {
  videos: VideoState[];
  connected: boolean;
  log: string[];
  /** Refetch the dataset (after actions that publish no event, such as skip or delete). */
  refresh: () => void;
  exportState: ExportState;
  /** Show the export as running as soon as the POST returns 202, before its first event. */
  markExportStarted: () => void;
}

/**
 * Live state of one dataset. On every (re)connect it fetches GET /api/datasets/{id}, then applies
 * WebSocket events on top; events that arrive before the snapshot are buffered and replayed.
 * Reconnects with backoff 1, 2, 4, 8, max 10 s.
 */
export function useJobEvents(datasetId: string | null): JobEvents {
  const [videos, setVideos] = useState<VideoState[]>([]);
  const [connected, setConnected] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const [exportState, setExportState] = useState<ExportState>(EXPORT_IDLE);
  const loadRef = useRef<() => void>(() => {});

  useEffect(() => {
    setVideos([]);
    setLog([]);
    setExportState(EXPORT_IDLE);
    setConnected(false);
    if (!datasetId) return;

    let disposed = false;
    let ws: WebSocket | null = null;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let attempt = 0;
    let pending: JobEvent[] | null = null; // buffered while a snapshot is loading
    let loadSeq = 0;
    const titles = new Map<string, string>();

    const apply = (e: JobEvent) => {
      if (e.type === "video_done") load(); // kept/dropped counts change at the end of a video
      setVideos((vs) => applyEvent(vs, e));
      const line = logLine(e, titles);
      if (line) setLog((l) => [...l, line].slice(-LOG_MAX));
    };

    const load = () => {
      const seq = ++loadSeq;
      pending = pending ?? [];
      getDataset(datasetId)
        .then((d) => {
          if (disposed || seq !== loadSeq) return;
          for (const v of d.videos) titles.set(v.id, v.title || v.id);
          const buffered = pending ?? [];
          pending = null;
          setVideos(d.videos.map(toVideoState));
          for (const e of buffered) setVideos((vs) => applyEvent(vs, e));
        })
        .catch((err: unknown) => {
          if (disposed || seq !== loadSeq) return;
          pending = null;
          setLog((l) => [...l, `could not load dataset: ${err instanceof Error ? err.message : err}`]);
        });
    };
    loadRef.current = load;

    const connect = () => {
      ws = new WebSocket(eventsUrl(datasetId));
      ws.onopen = () => {
        attempt = 0;
        setConnected(true);
        load();
      };
      ws.onmessage = (msg: MessageEvent) => {
        const e = JSON.parse(String(msg.data)) as JobEvent;
        setExportState((s) => applyExportEvent(s, e));
        if (pending) {
          pending.push(e);
          const line = logLine(e, titles);
          if (line) setLog((l) => [...l, line].slice(-LOG_MAX));
        } else apply(e);
      };
      ws.onclose = () => {
        ws = null;
        if (disposed) return;
        setConnected(false);
        const delay = BACKOFF_S[Math.min(attempt, BACKOFF_S.length - 1)];
        attempt += 1;
        timer = setTimeout(connect, delay * 1000);
      };
    };
    connect();

    return () => {
      disposed = true;
      clearTimeout(timer);
      loadRef.current = () => {};
      if (ws) {
        ws.onclose = null;
        ws.close();
      }
    };
  }, [datasetId]);

  const refresh = useCallback(() => loadRef.current(), []);
  const markExportStarted = useCallback(() => setExportState({ ...EXPORT_IDLE, status: "running" }), []);
  return { videos, connected, log, refresh, exportState, markExportStarted };
}
