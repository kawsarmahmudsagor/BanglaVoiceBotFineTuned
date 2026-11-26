import { useCallback, useEffect, useState } from "react";
import { listDatasets } from "./api";
import type { DatasetSummary } from "./types";
import { Jobs } from "./pages/Jobs";
import { Review } from "./pages/Review";
import { DatasetPage } from "./pages/Dataset";
import { Settings } from "./pages/Settings";

// Hash routes: the backend serves dist with StaticFiles(html=True), which 404s on deep paths.
const PAGES = [
  { path: "jobs", label: "Jobs" },
  { path: "review", label: "Review" },
  { path: "dataset", label: "Dataset" },
  { path: "settings", label: "Settings" },
] as const;
type PagePath = (typeof PAGES)[number]["path"];

const THEMES = ["system", "light", "dark"] as const;
type Theme = (typeof THEMES)[number];

function readStored(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}
function writeStored(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // storage unavailable: the choice just won't persist
  }
}

function currentPage(): PagePath {
  const p = window.location.hash.replace(/^#\/?/, "");
  return PAGES.find((x) => x.path === p)?.path ?? "jobs";
}

function useHashPage(): PagePath {
  const [page, setPage] = useState<PagePath>(currentPage);
  useEffect(() => {
    const on = () => setPage(currentPage());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return page;
}

function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(() => {
    const t = readStored("ytds.theme");
    return THEMES.includes(t as Theme) ? (t as Theme) : "system";
  });
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") delete root.dataset.theme;
    else root.dataset.theme = theme;
    writeStored("ytds.theme", theme);
  }, [theme]);
  const cycle = () => setTheme((t) => THEMES[(THEMES.indexOf(t) + 1) % THEMES.length]);
  return [theme, cycle];
}

export function App() {
  const page = useHashPage();
  const [theme, cycleTheme] = useTheme();
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [datasetId, setDatasetId] = useState<string | null>(() => readStored("ytds.dataset"));
  const [loadError, setLoadError] = useState<string | null>(null);

  const selectDataset = useCallback((id: string | null) => {
    setDatasetId(id);
    writeStored("ytds.dataset", id ?? "");
  }, []);

  const reloadDatasets = useCallback(async () => {
    try {
      const list = await listDatasets();
      setDatasets(list);
      setLoadError(null);
      return list;
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : String(e));
      return null;
    }
  }, []);

  useEffect(() => {
    void reloadDatasets().then((list) => {
      if (!list) return;
      setDatasetId((id) => (id && list.some((d) => d.id === id) ? id : list.at(-1)?.id ?? null));
    });
  }, [reloadDatasets]);

  const onDatasetCreated = useCallback(
    async (id: string) => {
      await reloadDatasets();
      selectDataset(id);
    },
    [reloadDatasets, selectDataset],
  );

  const dataset = datasets.find((d) => d.id === datasetId) ?? null;
  const title = PAGES.find((p) => p.path === page)!.label;

  return (
    <div className="shell">
      <nav className="nav" aria-label="Main">
        <div className="nav__brand">
          TTS dataset builder
          <small lang="bn">ইউটিউব থেকে বাংলা ভয়েস</small>
        </div>
        {PAGES.map((p) => (
          <a key={p.path} className="nav__link" href={`#/${p.path}`} aria-current={p.path === page ? "page" : undefined}>
            {p.label}
          </a>
        ))}
        <div className="nav__foot">
          <button type="button" className="btn btn--small" onClick={cycleTheme} aria-label={`Theme: ${theme}. Change theme`}>
            Theme: {theme}
          </button>
        </div>
      </nav>
      <main className="main">
        <header className="topbar">
          <h1>{title}</h1>
          <label className="field topbar__picker">
            Dataset
            <select
              className="select"
              value={dataset?.id ?? ""}
              onChange={(e) => selectDataset(e.target.value || null)}
            >
              {datasets.length === 0 && <option value="">No datasets yet</option>}
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} ({d.videos} videos)
                </option>
              ))}
            </select>
          </label>
        </header>
        {loadError && (
          <div className="banner" role="alert">
            <p>Could not reach the backend: {loadError}</p>
            <button type="button" className="btn btn--small" onClick={() => void reloadDatasets()}>
              Try again
            </button>
          </div>
        )}
        {page === "jobs" && <Jobs dataset={dataset} onDatasetCreated={onDatasetCreated} onVideosChanged={reloadDatasets} />}
        {page === "review" && <Review dataset={dataset} />}
        {page === "dataset" && <DatasetPage dataset={dataset} />}
        {page === "settings" && <Settings />}
      </main>
    </div>
  );
}
