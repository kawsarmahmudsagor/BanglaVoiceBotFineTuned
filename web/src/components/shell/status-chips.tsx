import Link from "next/link";
import { AlertTriangle, Cpu, Lock, LockOpen, MemoryStick, Plug } from "lucide-react";
import type { ReactNode } from "react";
import type { SystemInfo } from "@/lib/types";
import { cn } from "@/lib/utils";

function Chip({ children, title, className }: { children: ReactNode; title?: string; className?: string }) {
  return (
    <span
      title={title}
      className={cn(
        "inline-flex h-7 max-w-48 items-center gap-1.5 rounded-md border border-border bg-card px-2 text-[13px] whitespace-nowrap",
        className,
      )}
    >
      {children}
    </span>
  );
}

/** A tiny bar strip of recent GPU utilisation (0-100). */
function Spark({ history }: { history: number[] }) {
  if (history.length < 2) return null;
  const w = 60;
  const h = 14;
  const bw = w / 60;
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-hidden className="hidden text-saffron lg:block">
      {history.map((v, i) => {
        const bh = Math.max(1, (Math.min(100, Math.max(0, v)) / 100) * h);
        return <rect key={i} x={w - (history.length - i) * bw} y={h - bh} width={bw} height={bh} fill="currentColor" />;
      })}
    </svg>
  );
}

export function StatusChips({
  system,
  error,
  history,
}: {
  system: SystemInfo | null;
  error: string | null;
  history: number[];
}) {
  if (error) {
    return (
      <Chip title={error} className="border-status-failed text-status-failed">
        <Plug className="size-3.5" aria-hidden />
        <span>API unreachable</span>
      </Chip>
    );
  }
  if (!system) return null;
  const { gpu, lock } = system;
  return (
    <div className="flex min-w-0 items-center gap-2">
      {gpu ? (
        <>
          <Chip title={`${gpu.name}, ${gpu.temp_c} °C`}>
            <Cpu className="size-3.5 text-muted-foreground" aria-hidden />
            <span className="sr-only">GPU utilisation</span>
            <span className="font-mono">{`${gpu.util_pct}%`}</span>
            <Spark history={history} />
          </Chip>
          <Chip title="GPU memory used / total" className="hidden sm:inline-flex">
            <MemoryStick className="size-3.5 text-muted-foreground" aria-hidden />
            <span className="font-mono">{`${(gpu.vram_used_mb / 1024).toFixed(1)}/${(gpu.vram_total_mb / 1024).toFixed(1)} GB`}</span>
          </Chip>
        </>
      ) : (
        <Chip title="No NVIDIA GPU visible to the API">
          <AlertTriangle className="size-3.5 text-muted-foreground" aria-hidden />
          <span>No GPU</span>
        </Chip>
      )}
      {lock.holder_kind === "run" && lock.holder_id ? (
        <Link
          href={`/runs/${encodeURIComponent(lock.holder_id)}`}
          className="inline-flex h-7 max-w-48 items-center gap-1.5 rounded-md border border-border bg-card px-2 text-[13px] whitespace-nowrap hover:bg-raised"
          title="GPU lock holder"
        >
          <Lock className="size-3.5 text-saffron" aria-hidden />
          <span className="truncate font-mono">{lock.holder_id}</span>
        </Link>
      ) : lock.holder_kind ? (
        <Chip title="GPU lock holder">
          <Lock className="size-3.5 text-saffron" aria-hidden />
          <span className="truncate">{`${lock.holder_kind} ${lock.holder_id ?? ""}`.trim()}</span>
        </Chip>
      ) : (
        <Chip className="hidden sm:inline-flex">
          <LockOpen className="size-3.5 text-muted-foreground" aria-hidden />
          <span>GPU free</span>
        </Chip>
      )}
    </div>
  );
}
