"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { PanelLeftClose, PanelLeftOpen } from "lucide-react";
import { NAV_GROUPS, isActive } from "@/lib/nav";
import { cn } from "@/lib/utils";

export function NavLinks({ collapsed, onNavigate }: { collapsed: boolean; onNavigate?: () => void }) {
  const pathname = usePathname() ?? "/";
  return (
    <div className="flex flex-col gap-4">
      {NAV_GROUPS.map((g, gi) => (
        <div key={g.label ?? gi}>
          {g.label &&
            (collapsed ? (
              <div className="mx-3 mb-1 h-px bg-rule" aria-hidden />
            ) : (
              <div className="px-3 pb-1.5 text-[13px] text-muted-foreground">{g.label}</div>
            ))}
          <ul className="flex flex-col gap-0.5">
            {g.items.map(({ href, label, icon: Icon }) => {
              const active = isActive(pathname, href);
              return (
                <li key={href}>
                  <Link
                    href={href}
                    onClick={onNavigate}
                    aria-current={active ? "page" : undefined}
                    title={collapsed ? label : undefined}
                    className={cn(
                      "relative flex h-9 items-center gap-2.5 rounded-md px-3 text-sm transition-colors duration-150",
                      "hover:bg-raised",
                      active ? "bg-raised font-medium text-foreground" : "text-muted-foreground",
                      collapsed && "justify-center px-0",
                    )}
                  >
                    {/* The item hangs from its own matra, which turns saffron when active. */}
                    <span
                      aria-hidden
                      className={cn(
                        "absolute top-0 h-0.5",
                        collapsed ? "right-2 left-2" : "right-3 left-3",
                        active ? "bg-saffron" : "bg-transparent",
                      )}
                    />
                    <Icon className={cn("size-4 shrink-0", active && "text-saffron")} aria-hidden />
                    {collapsed ? <span className="sr-only">{label}</span> : <span className="truncate">{label}</span>}
                  </Link>
                </li>
              );
            })}
          </ul>
        </div>
      ))}
    </div>
  );
}

export function Brand({ collapsed }: { collapsed: boolean }) {
  return (
    <Link href="/" className="flex h-12 shrink-0 items-center gap-2 px-3" aria-label="Bangla Voice MLOps, overview">
      <span
        aria-hidden
        className="flex size-7 shrink-0 items-start justify-center rounded-md bg-saffron pt-1 text-[15px] leading-none font-semibold text-primary-foreground"
      >
        <span className="border-t-2 border-current px-0.5 pt-0.5">ব</span>
      </span>
      {!collapsed && <span className="truncate text-sm font-semibold">Bangla Voice MLOps</span>}
    </Link>
  );
}

export function Sidebar({ collapsed, onToggle }: { collapsed: boolean; onToggle: () => void }) {
  return (
    <aside
      className={cn(
        "sticky top-0 hidden h-screen shrink-0 flex-col border-r border-border bg-card md:flex",
        collapsed ? "w-14" : "w-[220px]",
      )}
    >
      <Brand collapsed={collapsed} />
      <nav aria-label="Main" className="flex-1 overflow-y-auto px-2 py-3">
        <NavLinks collapsed={collapsed} />
      </nav>
      <button
        type="button"
        onClick={onToggle}
        aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        aria-expanded={!collapsed}
        className="m-2 flex h-9 items-center justify-center gap-2 rounded-md text-sm text-muted-foreground hover:bg-raised hover:text-foreground"
      >
        {collapsed ? <PanelLeftOpen className="size-4" aria-hidden /> : <PanelLeftClose className="size-4" aria-hidden />}
        {!collapsed && <span>Collapse</span>}
      </button>
    </aside>
  );
}
