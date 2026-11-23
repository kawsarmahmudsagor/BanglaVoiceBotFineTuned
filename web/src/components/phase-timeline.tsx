import { PHASES, phaseStates, type PhaseState } from "@/lib/metrics";
import type { Phase, RunState } from "@/lib/types";
import { cn } from "@/lib/utils";

const STYLE: Record<PhaseState, string> = {
  done: "border-emerald-500/50 bg-emerald-500/15 text-emerald-700 dark:text-emerald-300",
  active: "border-blue-500 bg-blue-500/15 font-medium text-blue-700 dark:text-blue-300",
  failed: "border-red-500/60 bg-red-500/15 text-red-700 dark:text-red-300",
  pending: "border-border text-muted-foreground",
};

export function PhaseTimeline({ phase, state }: { phase: Phase | null | undefined; state: RunState }) {
  const states = phaseStates(phase, state);
  return (
    <ol aria-label="Phases" className="flex gap-2">
      {PHASES.map((p, i) => (
        <li
          key={p}
          data-phase={p}
          data-state={states[i]}
          aria-current={states[i] === "active" ? "step" : undefined}
          className={cn("flex-1 rounded-md border px-3 py-1.5 text-center text-sm", STYLE[states[i]])}
        >
          {p}
        </li>
      ))}
    </ol>
  );
}
