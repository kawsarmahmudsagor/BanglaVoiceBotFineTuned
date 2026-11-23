import type {
  Backend, CancelResult, CatalogEntry, CatalogList, CreateRunBody, DatasetDetail, DatasetListItem, Dataset, IngestRequest,
  EvalSamplesPage, IngestStatus, ModelVersion, PreparedRef, RecipeBody, RecipeDetail, RecipeListItem, RecipeVersionDetail,
  ResumeResult, RunDetail, RunListItem, RunLogs, RunMetrics, SamplesPage, SamplesQuery, SavedRecipe,
  SystemInfo, Task, ValidateResult, VersionDetail, VersionSummary,
} from "./types";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(public status: number, public detail: string) {
    super(detail);
    this.name = "ApiError";
  }
}

function describeDetail(body: unknown, fallback: string): string {
  if (body && typeof body === "object") {
    const b = body as { detail?: unknown; errors?: unknown };
    if (Array.isArray(b.errors)) return b.errors.map(String).join("\n");
    if (typeof b.detail === "string") return b.detail;
    if (Array.isArray(b.detail)) {
      return b.detail
        .map((d) => {
          if (d && typeof d === "object" && "msg" in d) {
            const e = d as { loc?: unknown[]; msg: string };
            return e.loc?.length ? `${e.loc.join(".")}: ${e.msg}` : e.msg;
          }
          return String(d);
        })
        .join("\n");
    }
  }
  return fallback;
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const init: RequestInit & { headers: Record<string, string> } = { method, headers: {} };
  if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  const res = await fetch(`${API_URL}${path}`, init);
  if (!res.ok) {
    let parsed: unknown = null;
    try {
      parsed = await res.json();
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, describeDetail(parsed, res.statusText || `HTTP ${res.status}`));
  }
  return (await res.json()) as T;
}

export const apiGet = <T>(path: string) => request<T>("GET", path);
export const apiPost = <T>(path: string, body?: unknown) => request<T>("POST", path, body);
export const apiPatch = <T>(path: string, body?: unknown) => request<T>("PATCH", path, body);

function qs(params: Record<string, string | number | undefined>): string {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined) sp.set(k, String(v));
  const s = sp.toString();
  return s ? `?${s}` : "";
}

const enc = encodeURIComponent;

// system
export const getSystem = () => apiGet<SystemInfo>("/api/system");

// catalog
export const listCatalog = (task?: Task) => apiGet<CatalogList>(`/api/catalog${qs({ task })}`);
export const getCatalogEntry = (id: string) => apiGet<CatalogEntry>(`/api/catalog/${enc(id)}`);

// datasets
export const listDatasets = () => apiGet<DatasetListItem[]>("/api/datasets");
export const getDataset = (name: string) => apiGet<DatasetDetail>(`/api/datasets/${enc(name)}`);
export const patchDataset = (name: string, evalOnly: boolean) =>
  apiPatch<Dataset>(`/api/datasets/${enc(name)}`, { eval_only: evalOnly });
export const getVersion = (name: string, n: number) =>
  apiGet<VersionDetail>(`/api/datasets/${enc(name)}/versions/${n}`);
export const getSamples = (name: string, n: number, q: SamplesQuery = {}) =>
  apiGet<SamplesPage>(
    `/api/datasets/${enc(name)}/versions/${n}/samples${qs({ offset: q.offset, limit: q.limit, split: q.split, warning: q.warning })}`,
  );
export const startIngest = (body: IngestRequest) => apiPost<{ ingest_id: number }>("/api/datasets/ingest", body);
export const getIngest = (id: number) => apiGet<IngestStatus>(`/api/ingests/${id}`);
export const editTranscript = (name: string, n: number, sampleId: string, text: string) =>
  apiPost<VersionSummary>(`/api/datasets/${enc(name)}/versions/${n}/edits`, { sample_id: sampleId, text });

// recipes
export const listBackends = () => apiGet<Backend[]>("/api/backends");
export const validateRecipe = (body: RecipeBody) => apiPost<ValidateResult>("/api/recipes/validate", body);
export const saveRecipe = (body: RecipeBody) => apiPost<SavedRecipe>("/api/recipes", body);
export const listRecipes = () => apiGet<RecipeListItem[]>("/api/recipes");
export const getRecipe = (name: string) => apiGet<RecipeDetail>(`/api/recipes/${enc(name)}`);
export const getRecipeVersion = (name: string, n: number) =>
  apiGet<RecipeVersionDetail>(`/api/recipes/${enc(name)}/versions/${n}`);
export const cloneRecipe = (name: string, newName: string) =>
  apiPost<SavedRecipe>(`/api/recipes/${enc(name)}/clone`, { new_name: newName });

// runs
export const createRun = (body: CreateRunBody) => apiPost<{ id: string }>("/api/runs", body);
export const listRuns = (state?: string, limit?: number) =>
  apiGet<RunListItem[]>(`/api/runs${qs({ state, limit })}`);
export const getRun = (id: string) => apiGet<RunDetail>(`/api/runs/${enc(id)}`);
export const cancelRun = (id: string) => apiPost<CancelResult>(`/api/runs/${enc(id)}/cancel`);
export const resumeRun = (id: string) => apiPost<ResumeResult>(`/api/runs/${enc(id)}/resume`);
export const getMetrics = (id: string, offset = 0) =>
  apiGet<RunMetrics>(`/api/runs/${enc(id)}/metrics${qs({ offset })}`);
export const getLogs = (id: string, tail?: number) => apiGet<RunLogs>(`/api/runs/${enc(id)}/logs${qs({ tail })}`);
export const getPrepared = (id: string) => apiGet<{ refs: PreparedRef[] }>(`/api/runs/${enc(id)}/prepared`);
export const getEvalSamples = (id: string, offset = 0, limit = 50) =>
  apiGet<EvalSamplesPage>(`/api/runs/${enc(id)}/eval/samples${qs({ offset, limit })}`);
export const evalAudioUrl = (id: string, name: string) => `${API_URL}/api/runs/${enc(id)}/eval/audio/${enc(name)}`;
export const promoteRun = (id: string) => apiPost<ModelVersion>(`/api/runs/${enc(id)}/promote`);

// models
export const listModels = (task?: Task) => apiGet<ModelVersion[]>(`/api/models${qs({ task })}`);
export const getModel = (id: number) => apiGet<ModelVersion>(`/api/models/${id}`);
export const setChampion = (id: number) => apiPost<ModelVersion>(`/api/models/${id}/champion`);

// plain URLs (EventSource, <audio>, downloads)
export const runEventsUrl = (id: string) => `${API_URL}/api/runs/${enc(id)}/events`;
export const runFileUrl = (id: string, path: string) =>
  `${API_URL}/api/runs/${enc(id)}/files/${path.split("/").map(enc).join("/")}`;
// export manifest paths are relative to the run's export/ directory
export const runExportFileUrl = (id: string, path: string) => runFileUrl(id, `export/${path}`);
export const audioUrl = (name: string, n: number, sampleId: string) =>
  `${API_URL}/api/datasets/${enc(name)}/versions/${n}/audio/${enc(sampleId)}`;
export const exportUrl = (modelId: number, index: number) => `${API_URL}/api/models/${modelId}/exports/${index}`;
