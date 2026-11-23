import { Check, Circle, Loader, X } from "lucide-react";
import { PHASES, phaseStates, type PhaseState } from "@/lib/metrics";
import type { Phase, RunState } from "@/lib/types";
import { cn } from "@/lib/utils";

const STYLE: Record<PhaseState, { cls: string; icon: typeof Check; label: string }> = {
  done: { cls: "border-status-done/50 text-status-done", icon: Check, label: "done" },
  active: { cls: "border-status-running bg-status-running/10 font-medium text-status-running", icon: Loader, label: "in progress" },
  failed: { cls: "border-status-failed/60 bg-status-failed/10 text-status-failed", icon: X, label: "failed" },
  pending: { cls: "border-border text-muted-foreground", icon: Circle, label: "pending" },
};

export function PhaseTimeline({ phase, state }: { phase: Phase | null | undefined; state: RunState }) {
  const states = phaseStates(phase, state);
  return (
    <ol aria-label="Phases" className="grid grid-cols-2 gap-2 sm:grid-cols-4">
      {PHASES.map((p, i) => {
        const s = STYLE[states[i]];
        const Icon = s.icon;
        return (
          <li
            key={p}
            data-phase={p}
            data-state={states[i]}
            aria-current={states[i] === "active" ? "step" : undefined}
            className={cn("flex items-center justify-center gap-1.5 rounded-md border px-3 py-1.5 text-sm", s.cls)}
          >
            <Icon className="size-3.5 shrink-0" aria-hidden />
            {p}
            <span className="sr-only">({s.label})</span>
          </li>
        );
      })}
    </ol>
  );
}
