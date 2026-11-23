"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { breadcrumbs } from "@/lib/nav";

export function Breadcrumbs() {
  const crumbs = breadcrumbs(usePathname() ?? "/");
  return (
    <nav aria-label="Breadcrumb" className="min-w-0">
      <ol className="flex min-w-0 items-center gap-1.5 text-sm text-muted-foreground">
        {crumbs.map((c, i) => {
          const last = i === crumbs.length - 1;
          return (
            <li key={c.href} className={last ? "min-w-0 truncate text-foreground" : "hidden shrink-0 sm:block"}>
              {last ? (
                <span aria-current="page" title={c.label}>{c.label}</span>
              ) : (
                <>
                  <Link href={c.href} className="hover:text-foreground">{c.label}</Link>
                  <span aria-hidden className="pl-1.5">/</span>
                </>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
