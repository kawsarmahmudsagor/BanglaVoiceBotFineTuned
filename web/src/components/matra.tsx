import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

/**
 * The matra line: a 2 px rule that headings hang from, like Bengali letters from মাত্রা.
 * With `progress`, a saffron fill runs along it.
 */
export function MatraBar({ progress, className }: { progress?: number; className?: string }) {
  const pct = progress === undefined ? undefined : Math.round(Math.min(1, Math.max(0, progress)) * 100);
  return (
    <div
      className={cn("relative h-0.5 w-full bg-rule", className)}
      {...(pct !== undefined
        ? { role: "progressbar", "aria-label": "Progress", "aria-valuenow": pct, "aria-valuemin": 0, "aria-valuemax": 100 }
        : { "aria-hidden": true })}
    >
      {pct !== undefined && <div className="absolute inset-y-0 left-0 bg-saffron" style={{ width: `${pct}%` }} />}
    </div>
  );
}

export function PageHeader({
  title,
  actions,
  progress,
  children,
}: {
  title: ReactNode;
  actions?: ReactNode;
  progress?: number;
  children?: ReactNode;
}) {
  return (
    <header className="mb-6">
      <MatraBar progress={progress} />
      <div className="flex flex-wrap items-start justify-between gap-3 pt-2">
        <h1 className="min-w-0 break-words text-[26px] leading-tight font-semibold">{title}</h1>
        {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
      </div>
      {children && <div className="mt-2 text-sm text-muted-foreground">{children}</div>}
    </header>
  );
}

export function Panel({
  title,
  actions,
  children,
  className,
}: {
  title?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("min-w-0 rounded-[10px] border border-border bg-card", className)}>
      {(title || actions) && (
        <div className="px-4 pt-3">
          <MatraBar />
          <div className="flex flex-wrap items-center justify-between gap-2 pt-1.5">
            {title && <h2 className="text-base font-semibold">{title}</h2>}
            {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
          </div>
        </div>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}
