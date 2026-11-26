// Mirrors the backend shapes in youtube-tts-dataset/backend/ytds (store.py, api/*.py).

export const STAGES = [
  "download",
  "separate",
  "segment",
  "quality",
  "transcribe",
  "fallback",
  "punctuate",
] as const;
export type Stage = (typeof STAGES)[number];

export type VideoStatus = "queued" | "running" | "done" | "failed" | "skipped";
export type ClipStatus =
  | "dropped"
  | "auto"
  | "flagged"
  | "needs_review"
  | "pending_llm"
  | "approved"
  | "rejected";
export type TextSource = "stt" | "translated";
export type DropReason = "length" | "dnsmos" | "other_speaker";
export type ReviewReason = "cer" | "translated" | "pending_llm";

export interface Dataset {
  id: string;
  name: string;
  speaker: string;
  created: number;
}

/** GET /api/datasets row. */
export interface DatasetSummary extends Dataset {
  videos: number;
}

export interface Video {
  dataset_id: string;
  id: string;
  url: string;
  title: string | null;
  channel: string | null;
  upload_date: string | null;
  duration_s: number | null;
  license: string | null;
  status: VideoStatus;
  error: string | null;
  warning: string | null;
}

/** A video as returned by GET /api/datasets/{id}. */
export interface VideoDetail extends Video {
  stages: Record<Stage, boolean>;
  kept: number;
  dropped: number;
}

export interface DatasetDetail extends Dataset {
  videos: VideoDetail[];
}

export interface CreateDatasetBody {
  name: string;
  speaker?: string;
  urls: string;
}

export interface UrlError {
  url: string;
  error: string;
}

/** POST /api/datasets and POST /api/datasets/{id}/videos. */
export interface AddVideosResult {
  dataset_id: string;
  added: number;
  duplicates: number;
  errors: UrlError[];
}

export interface Clip {
  dataset_id: string;
  id: string;
  video_id: string;
  idx: number;
  start_s: number;
  end_s: number;
  dnsmos: number | null;
  speaker_sim: number | null;
  drop_reason: DropReason | null;
  text_a: string | null;
  text_b: string | null;
  cer: number | null;
  text_en: string | null;
  text_translated: string | null;
  text_final: string | null;
  text_source: TextSource | null;
  status: ClipStatus;
  no_punct: number;
}

export interface ReviewClip extends Clip {
  reason: ReviewReason;
}

export interface ReviewFilters {
  reason?: ReviewReason;
  video?: string;
  status?: "flagged" | "needs_review" | "pending_llm" | "approved" | "rejected";
}

export interface ReviewQueue {
  clips: ReviewClip[];
  remaining: number;
  reviewed: number;
}

export interface ClipPatch {
  text_final?: string;
  status?: "approved" | "rejected";
}

export interface SplitStats {
  clips: number;
  hours: number;
}

export interface VideoStats {
  video_id: string;
  title: string | null;
  status: VideoStatus;
  clips_kept: number;
  hours_kept: number;
  dropped: Record<string, number>;
}

export interface DatasetStats {
  clips_kept: number;
  hours_kept: number;
  dropped: Record<string, number>;
  videos: VideoStats[];
  splits: Record<"train" | "val" | "test", SplitStats>;
}

export interface ExportBody {
  out_dir?: string;
  /** Export without the unreviewed translated clips instead of answering 409. */
  skip_unreviewed_translated?: boolean;
}

export interface ExportStarted {
  out_dir: string;
}

/** `summary` of the export stage_done event (exporter.export_dataset). */
export interface ExportSummary {
  out_dir: string;
  clips: number;
  hours: number;
  process_error: number;
  empty_text_excluded: number;
  validation_dropped: number;
  unreviewed_translated_skipped: number;
}

/** The dataset's export as seen through WebSocket events with stage "export". */
export interface ExportState {
  status: "idle" | "running" | "done" | "failed";
  done: number;
  total: number;
  summary: ExportSummary | null;
  error: string | null;
}

export interface Thresholds {
  cer_auto: number;
  cer_fallback: number;
  dnsmos_min: number;
  speaker_sim_min: number;
  clip_min_s: number;
  clip_max_s: number;
  merge_gap_s: number;
  demucs_segment_s: number;
}

export interface StageModels {
  separate: string;
  vad: string;
  dnsmos: string;
  speaker: string;
  model_a: string;
  model_b: string;
  translate: string;
  llm: string;
}

export interface AppSettings {
  thresholds: Thresholds;
  cookies_path: string;
  llm_host: string;
  llm_model: string;
  models: StageModels;
}

export interface SettingsUpdate {
  thresholds?: Thresholds;
  /** "" clears it. */
  cookies_path?: string;
  llm_host?: string;
  llm_model?: string;
}

export interface LlmTestResult {
  ok: boolean;
  error?: string;
}

export interface Health {
  llm_ok: boolean;
  cuda: boolean;
  ffmpeg: boolean;
}

export type JobEventType =
  | "progress"
  | "stage_done"
  | "stage_failed"
  | "video_done"
  | "log"
  | "llm_status";

/** WebSocket event on /ws/datasets/{id}. Export events use stage "export" and no video_id. */
export interface JobEvent {
  type: JobEventType;
  dataset_id: string;
  video_id?: string;
  stage?: Stage | "export";
  done?: number;
  total?: number;
  eta_s?: number | null;
  message?: string | null;
  error?: string;
  summary?: Record<string, unknown>;
}

export interface StageProgress {
  done: number;
  total: number;
}

/** A video in the Jobs table: the REST snapshot plus live progress from events. */
export interface VideoState extends VideoDetail {
  progress: Partial<Record<Stage, StageProgress>>;
  currentStage: Stage | null;
  eta_s: number | null;
}
