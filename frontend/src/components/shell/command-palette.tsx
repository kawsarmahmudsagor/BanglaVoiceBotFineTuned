"use client";

import { Command } from "cmdk";
import { Search } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { listCatalog, listDatasets, listRecipes, listRuns } from "@/lib/api";
import { buildItems, type PaletteGroup, type PaletteSources } from "@/lib/palette-search";

const GROUPS: PaletteGroup[] = ["Pages", "Runs", "Catalog", "Train", "Datasets"];
const EMPTY: PaletteSources = { runs: [], recipes: [], datasets: [], catalog: [] };

function useIsMac() {
  const [mac, setMac] = useState(false);
  useEffect(() => setMac(/Mac|iPhone|iPad/.test(navigator.platform)), []);
  return mac;
}

/** Ctrl+K / Cmd+K: jump to any page, run, catalog entry, training config or dataset. */
export function CommandPalette() {
  const router = useRouter();
  const mac = useIsMac();
  const [open, setOpen] = useState(false);
  const [sources, setSources] = useState<PaletteSources>(EMPTY);
  const [failed, setFailed] = useState<string[]>([]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((o) => !o);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    if (!open) return;
    let alive = true;
    Promise.allSettled([listRuns(undefined, 200), listRecipes(), listDatasets(), listCatalog()]).then(
      ([runs, recipes, datasets, catalog]) => {
        if (!alive) return;
        const miss: string[] = [];
        const pick = <T,>(r: PromiseSettledResult<T>, name: string, empty: T): T => {
          if (r.status === "fulfilled") return r.value;
          miss.push(name);
          return empty;
        };
        setSources({
          runs: pick(runs, "runs", []),
          recipes: pick(recipes, "recipes", []),
          datasets: pick(datasets, "datasets", []),
          catalog: pick(catalog, "catalog", { entries: [], errors: [] }).entries,
        });
        setFailed(miss);
      },
    );
    return () => {
      alive = false;
    };
  }, [open]);

  const items = useMemo(() => buildItems(sources), [sources]);

  const go = (href: string) => {
    setOpen(false);
    router.push(href);
  };

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        aria-label="Open command palette"
        className="inline-flex h-8 shrink-0 items-center gap-2 rounded-md border border-border bg-card px-2 text-[13px] text-muted-foreground hover:bg-raised hover:text-foreground"
      >
        <Search className="size-3.5" aria-hidden />
        <span className="hidden lg:inline">Search</span>
        <kbd className="hidden font-mono text-[12px] sm:inline">{mac ? "⌘K" : "Ctrl K"}</kbd>
      </button>
      <Command.Dialog
        open={open}
        onOpenChange={setOpen}
        label="Command palette"
        overlayClassName="fixed inset-0 z-50 bg-ink/70"
        contentClassName="fixed top-[12vh] left-1/2 z-50 w-[min(640px,calc(100vw-32px))] -translate-x-1/2 overflow-hidden rounded-[10px] border border-border bg-popover text-popover-foreground"
      >
        <div className="flex items-center gap-2 border-b border-border px-3">
          <Search className="size-4 text-muted-foreground" aria-hidden />
          <Command.Input
            placeholder="Go to a page, run, model, training config or dataset"
            className="h-11 w-full bg-transparent text-sm outline-none placeholder:text-muted-foreground"
          />
        </div>
        {failed.length > 0 && (
          <p className="border-b border-border px-3 py-2 text-[13px] text-muted-foreground">
            Could not load {failed.join(", ")}; showing the rest.
          </p>
        )}
        <Command.List className="max-h-[min(420px,60vh)] overflow-y-auto p-1.5">
          <Command.Empty className="px-3 py-6 text-center text-sm text-muted-foreground">No matches.</Command.Empty>
          {GROUPS.map((g) => {
            const groupItems = items.filter((i) => i.group === g);
            if (groupItems.length === 0) return null;
            return (
              <Command.Group
                key={g}
                heading={g}
                className="[&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:pt-2 [&_[cmdk-group-heading]]:pb-1 [&_[cmdk-group-heading]]:text-[13px] [&_[cmdk-group-heading]]:text-muted-foreground"
              >
                {groupItems.map((i) => (
                  <Command.Item
                    key={`${g}:${i.href}`}
                    value={`${g} ${i.label} ${i.href}`}
                    keywords={i.keywords}
                    onSelect={() => go(i.href)}
                    className="flex min-h-9 cursor-pointer items-center rounded-md px-2 text-sm data-[selected=true]:bg-raised data-[selected=true]:text-foreground"
                  >
                    <span className={g === "Runs" ? "truncate font-mono text-[13px]" : "truncate"}>{i.label}</span>
                  </Command.Item>
                ))}
              </Command.Group>
            );
          })}
        </Command.List>
      </Command.Dialog>
    </>
  );
}
