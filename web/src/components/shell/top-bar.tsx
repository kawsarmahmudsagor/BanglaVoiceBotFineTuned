"use client";

import { Menu, Moon, Sun } from "lucide-react";
import type { Theme } from "@/lib/theme";
import type { SystemInfo } from "@/lib/types";
import { Breadcrumbs } from "./breadcrumbs";
import { UploadChip } from "@/components/upload-drawer";
import { CommandPalette } from "./command-palette";
import { StatusChips } from "./status-chips";

export const ICON_BUTTON =
  "inline-flex size-9 shrink-0 items-center justify-center rounded-md text-muted-foreground hover:bg-raised hover:text-foreground";

export function TopBar({
  system,
  error,
  history,
  theme,
  onToggleTheme,
  onOpenMenu,
}: {
  system: SystemInfo | null;
  error: string | null;
  history: number[];
  theme: Theme;
  onToggleTheme: () => void;
  onOpenMenu: () => void;
}) {
  const next = theme === "dark" ? "light" : "dark";
  return (
    <header className="sticky top-0 z-30 flex h-12 items-center gap-2 border-b border-border bg-ink px-3 md:px-6">
      <button type="button" onClick={onOpenMenu} aria-label="Open navigation" className={`${ICON_BUTTON} md:hidden`}>
        <Menu className="size-4" aria-hidden />
      </button>
      <div className="min-w-0 flex-1">
        <Breadcrumbs />
      </div>
      <UploadChip />
      <StatusChips system={system} error={error} history={history} />
      <CommandPalette />
      <button type="button" onClick={onToggleTheme} aria-label={`Switch to ${next} theme`} className={ICON_BUTTON}>
        {theme === "dark" ? <Sun className="size-4" aria-hidden /> : <Moon className="size-4" aria-hidden />}
      </button>
    </header>
  );
}
