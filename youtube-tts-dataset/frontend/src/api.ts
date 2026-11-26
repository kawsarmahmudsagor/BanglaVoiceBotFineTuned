// One typed function per backend route (youtube-tts-dataset/backend/ytds/api).
import type {
  AddVideosResult,
  AppSettings,
  ClipPatch,
  CreateDatasetBody,
  Dataset,
  DatasetDetail,
  DatasetStats,
  DatasetSummary,
  ExportBody,
  ExportStarted,
  Health,
  LlmTestResult,
  ReviewClip,
  ReviewFilters,
  ReviewQueue,
  SettingsUpdate,
} from "./types";

/** A non-2xx response. `body` is the parsed JSON (FastAPI errors carry `detail`). */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly body: unknown,
  ) {
    super(errorMessage(status, body));
    this.name = "ApiError";
  }
}

function errorMessage(status: number, body: unknown): string {
  if (body && typeof body === "object") {
    const b = body as Record<string, unknown>;
    const detail = b.detail ?? b.error;
    if (typeof detail === "string") return detail;
    if (detail && typeof detail === "object" && typeof (detail as { error?: unknown }).error === "string") {
      return (detail as { error: string }).error;
    }
  }
  return `request failed with status ${status}`;
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await res.text();
  let data: unknown = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }
  }
  if (!res.ok) throw new ApiError(res.status, data);
  return data as T;
}

const ds = (id: string) => `/api/datasets/${encodeURIComponent(id)}`;
const vid = (id: string, videoId: string) => `${ds(id)}/videos/${encodeURIComponent(videoId)}`;
const clip = (id: string, clipId: string) => `${ds(id)}/clips/${encodeURIComponent(clipId)}`;

// datasets and videos
export const listDatasets = () => request<DatasetSummary[]>("GET", "/api/datasets");
export const createDataset = (body: CreateDatasetBody) => request<AddVideosResult>("POST", "/api/datasets", body);
export const getDataset = (id: string) => request<DatasetDetail>("GET", ds(id));
export const updateDataset = (id: string, speaker: string) => request<Dataset>("PUT", ds(id), { speaker });
export const addVideos = (id: string, urls: string) => request<AddVideosResult>("POST", `${ds(id)}/videos`, { urls });
export const retryVideo = (id: string, videoId: string) => request<{ ok: true }>("POST", `${vid(id, videoId)}/retry`);
export const skipVideo = (id: string, videoId: string) => request<{ ok: true }>("POST", `${vid(id, videoId)}/skip`);
export const deleteVideo = (id: string, videoId: string) => request<{ ok: true }>("DELETE", vid(id, videoId));

// review
export function getReview(id: string, filters: ReviewFilters = {}): Promise<ReviewQueue> {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(filters)) if (v) q.set(k, v);
  const qs = q.toString();
  return request<ReviewQueue>("GET", `${ds(id)}/review${qs ? `?${qs}` : ""}`);
}
export const patchClip = (id: string, clipId: string, patch: ClipPatch) =>
  request<ReviewClip>("PATCH", clip(id, clipId), patch);
/** URL for an <audio> element; the backend slices the vocals stem on demand. */
export const clipAudioUrl = (id: string, clipId: string) => `${clip(id, clipId)}/audio`;
/** The clip's WAV bytes, for drawing its waveform. */
export async function getClipAudio(id: string, clipId: string): Promise<ArrayBuffer> {
  const res = await fetch(clipAudioUrl(id, clipId));
  if (!res.ok) throw new ApiError(res.status, null);
  return res.arrayBuffer();
}

// dataset stats and export
export const getStats = (id: string) => request<DatasetStats>("GET", `${ds(id)}/stats`);
/**
 * Starts an export (202). A 409 throws ApiError whose body is `{unreviewed: n}` when translated clips
 * are still unreviewed, or `{error}` otherwise. Progress arrives on the WebSocket with stage "export".
 */
export const exportDataset = (id: string, body: ExportBody = {}) =>
  request<ExportStarted>("POST", `${ds(id)}/export`, body);

// settings and health
export const getSettings = () => request<AppSettings>("GET", "/api/settings");
export const putSettings = (body: SettingsUpdate) => request<AppSettings>("PUT", "/api/settings", body);
export const testLlm = () => request<LlmTestResult>("POST", "/api/settings/llm-test");
export const getHealth = () => request<Health>("GET", "/api/health");

// websocket
export function eventsUrl(id: string): string {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}/ws/datasets/${encodeURIComponent(id)}`;
}
