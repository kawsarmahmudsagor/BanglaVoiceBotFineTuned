// Response shapes of the FastAPI routers in backend/app/routers.

export type Task = "llm-sft" | "asr" | "tts";

export const RUN_STATES = [
  "queued", "preparing", "training", "evaluating", "exporting", "done", "failed", "cancelled",
] as const;
export type RunState = (typeof RUN_STATES)[number];

export type Phase = "prepare" | "train" | "eval" | "export";

// system
export interface GpuInfo {
  name: string;
  vram_used_mb: number;
  vram_total_mb: number;
  util_pct: number;
  temp_c: number;
}
export interface SystemInfo {
  gpu: GpuInfo | null;
  lock: { holder_kind: string | null; holder_id: string | null; acquired_at: string | null };
  disk: { free_gb: number; warn: boolean; block: boolean };
  queue: { queued: number; running: number };
}

// datasets
export interface Provenance {
  source: string;
  license: string;
  consent: string;
  commercial_ok: boolean;
  notes: string;
}
export interface DatasetLatest {
  number: number;
  n_samples: number;
  commercial_ok: boolean;
  created_at: string | null;
}
export interface Dataset {
  id: number;
  name: string;
  task: Task;
  eval_only: boolean;
  created_at: string | null;
}
export interface DatasetListItem extends Dataset {
  latest: DatasetLatest | null;
}
export interface VersionSummary {
  id: number;
  number: number;
  hash: string;
  n_samples: number;
  commercial_ok: boolean;
  parent_number: number | null;
  created_at: string | null;
}
export interface DatasetDetail extends Dataset {
  versions: VersionSummary[];
}
export interface ReportIssue {
  code: string;
  message?: string;
  sample_id?: string | null;
  [key: string]: unknown;
}
export interface VersionReport {
  warnings?: ReportIssue[];
  errors?: ReportIssue[];
  [key: string]: unknown;
}
export interface VersionDetail extends VersionSummary {
  dataset: string;
  task: Task;
  report: VersionReport | null;
  provenance: Provenance | null;
  split_ratios: Record<string, number> | null;
  split_source: "dataset" | "none";
  given_splits: Record<string, number>;
  default_split_counts: Record<string, number>;
}
export interface SplitRatios {
  train: number;
  val: number;
  test: number;
}
/** Per-run split settings; ratios are fractions (0.9), not percentages. */
export interface SplitConfig {
  use_dataset_splits: boolean;
  ratios: SplitRatios;
}
export interface SampleRow {
  sample_id: string;
  split: string;
  file?: string;
  text?: string;
  [key: string]: unknown;
}
export interface SamplesPage {
  total: number;
  rows: SampleRow[];
}
export interface SamplesQuery {
  offset?: number;
  limit?: number;
  split?: string;
  warning?: string;
}
export interface IngestRequest {
  upload_id: string;
  name: string;
  task: Task;
  eval_only?: boolean;
  provenance: Provenance;
}
export interface LocalIngestRequest {
  path: string;
  name: string;
  task: Task;
  eval_only?: boolean;
  provenance: Provenance;
}
export interface ImportDir {
  path: string;
  task_hint: Task | null;
  has_curation: boolean;
}
export type IngestState = "pending" | "running" | "done" | "failed";
export interface IngestStatus {
  id: number;
  state: IngestState;
  report: VersionReport | null;
  version: VersionSummary | null;
  error: string | null;
  dataset: string;
  created_at: string | null;
  /** Files validated so far; null until validation starts (and on older API responses). */
  progress?: { done: number; total: number } | null;
}

// recipes
export interface RecipeSpec {
  name: string;
  task: Task;
  backend: string;
  base_model: string;
  [key: string]: unknown;
}
export type RecipeBody = { yaml: string; spec?: undefined } | { spec: Record<string, unknown>; yaml?: undefined };
export interface ValidateResult {
  errors: string[];
  spec: RecipeSpec | null;
}
export interface SavedRecipe {
  name: string;
  number: number;
}
export interface RecipeListItem {
  name: string;
  latest: {
    number: number;
    frozen: boolean;
    task: Task;
    backend: string;
    base_model: string;
    updated_at: string | null;
  };
}
export interface RecipeDetail {
  name: string;
  versions: { number: number; frozen: boolean; created_at: string | null }[];
}
export interface RecipeVersionDetail {
  name: string;
  number: number;
  frozen: boolean;
  spec: RecipeSpec;
  yaml: string;
}
// Backend capabilities (shared Capabilities model, dumped by alias).
export type Backend = Record<string, unknown>;

// runs
export interface CreateRunBody {
  recipe: string;
  version?: number;
}
export interface RunListItem {
  id: string;
  recipe: string;
  recipe_number: number;
  backend: string | null;
  task: Task | null;
  state: RunState;
  commercial_ok: boolean;
  created_at: string | null;
  started_at: string | null;
  finished_at: string | null;
}
export interface RunStatus {
  phase: Phase;
  step: number;
  total_steps: number | null;
  eta_s: number | null;
  message: string;
  updated_at: number;
}
export interface ExportArtifact {
  format: string;
  path: string;
  size_bytes: number;
  serve_with: string | null;
}
export interface EvalResults {
  metrics: Record<string, number>;
  samples_dir: string | null;
}
export interface RunDetail {
  id: string;
  state: RunState;
  attempt: number;
  resume: boolean;
  cancel_requested: boolean;
  run_dir: string;
  commercial_ok: boolean;
  error: string | null;
  oom: boolean;
  git_sha: string | null;
  lock_hash: string | null;
  dataset_hashes: Record<string, string>;
  mlflow_run_id: string | null;
  created_at: string | null;
  started_at: string | null;
  finished_at: string | null;
  recipe: { name: string; number: number };
  status: RunStatus | null;
  task?: Task | null;
  eval: EvalResults | null;
  exports: ExportArtifact[];
  oom_hint: string | null;
  parent_run_id?: string | null;
  parent_checkpoint?: string | null;
}
// warm start (spec 14)
export type InitFrom =
  | { run: string; checkpoint: "final" | "latest" | number }
  | { model_version: string };
export interface CheckpointItem {
  name: string;
  step: number;
  complete: boolean;
  init_meta: Record<string, unknown> | null;
}
export interface InitSourceRun {
  id: string;
  recipe: string;
  state: RunState;
  finished_at: string | null;
  checkpoints: CheckpointItem[];
}
export interface InitSources {
  runs: InitSourceRun[];
  model_versions: { ref: string; run_id: string; metrics: Record<string, number> }[];
}
export interface RunMetric {
  step: number;
  split: string;
  name: string;
  value: number;
  ts: number;
}
export interface RunMetrics {
  items: RunMetric[];
  next_offset: number;
}
export interface RunLogs {
  stdout: string[];
  stderr: string[];
}
export interface CancelResult {
  state: RunState;
  cancel_requested?: boolean;
}
export interface ResumeResult {
  state: RunState;
  attempt: number;
}

// models
export interface ModelVersion {
  id: number;
  name: string;
  number: number;
  run_id: string;
  task: Task;
  base_model: string;
  metrics: Record<string, number>;
  exports: ExportArtifact[];
  commercial_ok: boolean;
  champion: boolean;
  created_at: string | null;
}

// catalog
export type CatalogRole = "primary" | "comparison" | "benchmark";
export type BanglaStatus = "native" | "finetuned" | "related" | "none";
export interface CatalogLicense {
  id: string;
  commercial_ok: boolean;
  url: string;
  notes?: string | null;
}
export interface CatalogSummary {
  id: string;
  task: Task;
  name: string;
  hf_repo: string;
  params: string;
  role: CatalogRole;
  license: CatalogLicense;
  bangla_status: BanglaStatus;
  streaming: boolean;
}
export interface CatalogLoadError {
  file: string;
  message: string;
}
export interface CatalogList {
  entries: CatalogSummary[];
  errors: CatalogLoadError[];
}
export interface CatalogEntry {
  id: string;
  task: Task;
  name: string;
  hf_repo: string;
  derived_from?: string | null;
  params: string;
  role: CatalogRole;
  license: CatalogLicense;
  bangla: { status: BanglaStatus; evidence: string; url?: string | null };
  runtime: { engine: string; convert?: string | null; streaming: boolean };
  description: string;
  dataset: {
    format: Task;
    required_fields: string[];
    optional_fields: string[];
    example: string;
    recommended: Record<string, string>;
    notes?: string | null;
  };
  finetune: {
    methods: string[];
    frameworks: string[];
    targets: string[];
    hyperparams: Record<string, unknown>;
    metrics: string[];
    pitfalls: string[];
  };
  recipe_template: {
    backend?: string | null;
    method: Record<string, unknown> & { type: string };
    hparams: Record<string, unknown>;
    export: string[];
  };
  sources: { title: string; url: string }[];
  verified: string;
}

export interface PreparedSkip {
  sample_id: string;
  split: string;
  reason: string;
}
export interface PreparedRef {
  role: "train" | "eval";
  spec: { family: string; version: number | string; params: Record<string, unknown> };
  spec_hash: string;
  dataset_hash: string;
  counts: Record<string, number>;
  skipped: PreparedSkip[];
  files: string[];
  built_at: number; // unix seconds (float), as the backend writes it
  build_s: number | null;
  path: string;
  cache_hit: boolean;
}
export interface EvalSamplesPage {
  items: Record<string, unknown>[];
  total: number;
}
