import {
  Boxes, Database, FileCode2, GitCompareArrows, LayoutDashboard, Library, Play, type LucideIcon,
} from "lucide-react";

export interface NavItem {
  href: string;
  label: string;
  icon: LucideIcon;
}
export interface NavGroup {
  label: string | null;
  items: NavItem[];
}

export const NAV_GROUPS: NavGroup[] = [
  { label: null, items: [{ href: "/", label: "Overview", icon: LayoutDashboard }] },
  {
    label: "Prepare",
    items: [
      { href: "/catalog", label: "Catalog", icon: Library },
      { href: "/datasets", label: "Datasets", icon: Database },
      { href: "/recipes", label: "Recipes", icon: FileCode2 },
    ],
  },
  {
    label: "Train",
    items: [
      { href: "/runs", label: "Runs", icon: Play },
      { href: "/runs/compare", label: "Compare", icon: GitCompareArrows },
    ],
  },
  { label: "Ship", items: [{ href: "/registry", label: "Registry", icon: Boxes }] },
];

const COMPARE = "/runs/compare";

function under(pathname: string, href: string): boolean {
  return pathname === href || pathname.startsWith(`${href}/`);
}

export function isActive(pathname: string, href: string): boolean {
  if (href === "/") return pathname === "/";
  if (href === "/runs" && under(pathname, COMPARE)) return false;
  return under(pathname, href);
}

const LABELS: Record<string, string> = {
  catalog: "Catalog",
  datasets: "Datasets",
  recipes: "Recipes",
  runs: "Runs",
  compare: "Compare",
  registry: "Registry",
  new: "New",
  v: "Versions",
};

export function breadcrumbs(pathname: string): { href: string; label: string }[] {
  const parts = pathname.split("/").filter(Boolean);
  if (parts.length === 0) return [{ href: "/", label: "Overview" }];
  const decode = (p: string) => {
    try {
      return decodeURIComponent(p);
    } catch {
      return p;
    }
  };
  const crumbs: { href: string; label: string }[] = [];
  for (let i = 0; i < parts.length; i++) {
    const p = parts[i];
    // /datasets/<name>/v/<n> has no page at .../v, so show "v<n>" as one crumb.
    if (p === "v" && parts[0] === "datasets" && i + 1 < parts.length) {
      crumbs.push({ href: `/${parts.slice(0, i + 2).join("/")}`, label: `v${decode(parts[i + 1])}` });
      i++;
      continue;
    }
    crumbs.push({ href: `/${parts.slice(0, i + 1).join("/")}`, label: LABELS[p] ?? decode(p) });
  }
  return crumbs;
}
