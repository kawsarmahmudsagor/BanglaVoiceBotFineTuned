import { AlertTriangle, XCircle } from "lucide-react";
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

/** An error that stopped something loading or saving. Icon + text, never colour alone. */
export function ErrorNotice({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div
      role="alert"
      className={cn("flex items-start gap-2 rounded-md border border-status-failed/50 bg-status-failed/10 px-3 py-2 text-sm", className)}
    >
      <XCircle className="mt-0.5 size-4 shrink-0 text-status-failed" aria-hidden />
      <div className="min-w-0 break-words">{children}</div>
    </div>
  );
}

/** Something needs attention but the page still works. */
export function WarningNotice({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div
      role="status"
      className={cn("flex items-start gap-2 rounded-md border border-saffron/50 bg-saffron/10 px-3 py-2 text-sm", className)}
    >
      <AlertTriangle className="mt-0.5 size-4 shrink-0 text-saffron" aria-hidden />
      <div className="min-w-0 break-words">{children}</div>
    </div>
  );
}

/** An empty list that points at the next step. */
export function EmptyState({ children }: { children: ReactNode }) {
  return <div className="rounded-md border border-dashed border-border px-4 py-8 text-center text-sm text-muted-foreground">{children}</div>;
}
