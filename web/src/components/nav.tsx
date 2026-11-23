"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

export const NAV_ITEMS = [
  { href: "/", label: "Dashboard" },
  { href: "/datasets", label: "Datasets" },
  { href: "/recipes", label: "Recipes" },
  { href: "/runs", label: "Runs" },
  { href: "/models", label: "Models" },
] as const;

export function isActive(pathname: string, href: string): boolean {
  return href === "/" ? pathname === "/" : pathname === href || pathname.startsWith(`${href}/`);
}

export function Nav() {
  const pathname = usePathname() ?? "/";
  return (
    <nav aria-label="Main" className="w-48 shrink-0 border-r bg-sidebar p-3 text-sidebar-foreground">
      <div className="px-2 pb-3 text-sm font-semibold">Bangla Voice MLOps</div>
      <ul className="space-y-0.5">
        {NAV_ITEMS.map(({ href, label }) => {
          const active = isActive(pathname, href);
          return (
            <li key={href}>
              <Link
                href={href}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "block rounded-md px-2 py-1.5 text-sm hover:bg-sidebar-accent",
                  active && "bg-sidebar-accent font-medium text-sidebar-accent-foreground",
                )}
              >
                {label}
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
