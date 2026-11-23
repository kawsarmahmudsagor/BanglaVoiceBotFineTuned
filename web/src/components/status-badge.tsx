import { Ban, CheckCircle2, Clock, Loader, XCircle, type LucideIcon } from "lucide-react";
import type { RunState } from "@/lib/types";
import { cn } from "@/lib/utils";

const STYLE: Record<RunState, { icon: LucideIcon; color: string }> = {
  queued: { icon: Clock, color: "text-muted-foreground" },
  preparing: { icon: Loader, color: "text-status-running" },
  training: { icon: Loader, color: "text-status-running" },
  evaluating: { icon: Loader, color: "text-status-running" },
  exporting: { icon: Loader, color: "text-status-running" },
  done: { icon: CheckCircle2, color: "text-status-done" },
  failed: { icon: XCircle, color: "text-status-failed" },
  cancelled: { icon: Ban, color: "text-muted-foreground" },
};

/** Run state as icon + label + colour, never colour alone. */
export function StatusBadge({ state, className }: { state: RunState; className?: string }) {
  const { icon: Icon, color } = STYLE[state] ?? STYLE.queued;
  return (
    <span className={cn("inline-flex items-center gap-1.5 text-sm font-medium whitespace-nowrap", color, className)}>
      <Icon className="size-3.5 shrink-0" aria-hidden />
      {state}
    </span>
  );
}
