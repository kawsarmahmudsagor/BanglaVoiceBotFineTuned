import { NAV_GROUPS } from "./nav";
import type { CatalogSummary, DatasetListItem, RecipeListItem, RunListItem } from "./types";

export type PaletteGroup = "Pages" | "Runs" | "Catalog" | "Train" | "Datasets";

export interface PaletteItem {
  group: PaletteGroup;
  label: string;
  href: string;
  keywords: string[];
}

export interface PaletteSources {
  runs: RunListItem[];
  recipes: RecipeListItem[];
  datasets: DatasetListItem[];
  catalog: CatalogSummary[];
}

const enc = encodeURIComponent;

export function buildItems(src: PaletteSources): PaletteItem[] {
  const pages: PaletteItem[] = NAV_GROUPS.flatMap((g) =>
    g.items.map((i) => ({ group: "Pages" as const, label: i.label, href: i.href, keywords: g.label ? [g.label] : [] })),
  );
  const runs: PaletteItem[] = src.runs.map((r) => ({
    group: "Runs",
    label: `${r.id} · ${r.recipe} v${r.recipe_number}`,
    href: `/runs/${enc(r.id)}`,
    keywords: [r.state, r.task ?? "", r.backend ?? ""].filter(Boolean),
  }));
  const catalog: PaletteItem[] = src.catalog.map((c) => ({
    group: "Catalog",
    label: c.name,
    href: `/catalog/${enc(c.id)}`,
    keywords: [c.id, c.hf_repo, c.task],
  }));
  const recipes: PaletteItem[] = src.recipes.map((r) => ({
    group: "Train",
    label: r.name,
    href: `/train/${enc(r.name)}`,
    keywords: [r.latest.task, r.latest.base_model],
  }));
  const datasets: PaletteItem[] = src.datasets.map((d) => ({
    group: "Datasets",
    label: d.name,
    href: `/datasets/${enc(d.name)}`,
    keywords: [d.task],
  }));
  return [...pages, ...runs, ...catalog, ...recipes, ...datasets];
}
