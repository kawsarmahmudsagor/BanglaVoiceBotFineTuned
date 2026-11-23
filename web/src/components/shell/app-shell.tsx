"use client";

import { X } from "lucide-react";
import { usePathname } from "next/navigation";
import { useCallback, useEffect, useState, type ReactNode } from "react";
import { getSystem } from "@/lib/api";
import { pushSample } from "@/lib/gpu-history";
import { applyTheme, readTheme, writeTheme, type Theme } from "@/lib/theme";
import type { SystemInfo } from "@/lib/types";
import { Brand, NavLinks, Sidebar } from "./sidebar";
import { TopBar } from "./top-bar";

const POLL_MS = 5000;
const RAIL_KEY = "rail";

function readRail(): boolean {
  try {
    return window.localStorage.getItem(RAIL_KEY) === "collapsed";
  } catch {
    return false;
  }
}

function writeRail(collapsed: boolean) {
  try {
    window.localStorage.setItem(RAIL_KEY, collapsed ? "collapsed" : "expanded");
  } catch {
    // Blocked storage: the rail resets on reload.
  }
}

function useSystem() {
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<number[]>([]);
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const s = await getSystem();
        if (!alive) return;
        setSystem(s);
        setError(null);
        const util = s.gpu?.util_pct;
        if (util !== undefined) setHistory((h) => pushSample(h, util));
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : String(e));
      }
    };
    tick();
    const id = setInterval(tick, POLL_MS);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);
  return { system, error, history };
}

export function AppShell({ children, palette }: { children: ReactNode; palette?: ReactNode }) {
  const { system, error, history } = useSystem();
  const [collapsed, setCollapsed] = useState(false);
  const [theme, setTheme] = useState<Theme>("dark");
  const [menuOpen, setMenuOpen] = useState(false);
  const pathname = usePathname();

  useEffect(() => {
    setCollapsed(readRail());
    setTheme(readTheme());
  }, []);

  useEffect(() => setMenuOpen(false), [pathname]);

  useEffect(() => {
    if (!menuOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMenuOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [menuOpen]);

  const toggleRail = useCallback(() => {
    setCollapsed((c) => {
      writeRail(!c);
      return !c;
    });
  }, []);

  const toggleTheme = useCallback(() => {
    const next: Theme = theme === "dark" ? "light" : "dark";
    writeTheme(next);
    applyTheme(next);
    setTheme(next);
  }, [theme]);

  return (
    <div className="flex min-h-screen">
      <Sidebar collapsed={collapsed} onToggle={toggleRail} />
      {menuOpen && (
        <div className="fixed inset-0 z-40 md:hidden" role="dialog" aria-modal="true" aria-label="Navigation">
          <div className="absolute inset-0 bg-ink/70" onClick={() => setMenuOpen(false)} aria-hidden />
          <div className="absolute inset-y-0 left-0 flex w-[260px] max-w-[85vw] flex-col border-r border-border bg-card">
            <div className="flex items-center justify-between pr-2">
              <Brand collapsed={false} />
              <button
                type="button"
                onClick={() => setMenuOpen(false)}
                aria-label="Close navigation"
                autoFocus
                className="inline-flex size-9 items-center justify-center rounded-md text-muted-foreground hover:bg-raised"
              >
                <X className="size-4" aria-hidden />
              </button>
            </div>
            <nav aria-label="Main" className="flex-1 overflow-y-auto px-2 py-3">
              <NavLinks collapsed={false} onNavigate={() => setMenuOpen(false)} />
            </nav>
          </div>
        </div>
      )}
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar
          system={system}
          error={error}
          history={history}
          theme={theme}
          onToggleTheme={toggleTheme}
          onOpenMenu={() => setMenuOpen(true)}
          palette={palette}
        />
        <main className="min-w-0 flex-1 px-4 py-6 md:px-6">{children}</main>
      </div>
    </div>
  );
}
