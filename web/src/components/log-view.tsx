"use client";

import { useLayoutEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

type Stream = "stdout" | "stderr";

const BOTTOM_SLACK_PX = 24;

export function LogView({ stdout, stderr }: { stdout: string[]; stderr: string[] }) {
  const [tab, setTab] = useState<Stream>("stdout");
  const lines = tab === "stdout" ? stdout : stderr;
  const box = useRef<HTMLDivElement>(null);
  const atBottom = useRef(true);

  // Follow new lines only while the user is already at the bottom.
  useLayoutEffect(() => {
    const el = box.current;
    if (el && atBottom.current) el.scrollTop = el.scrollHeight;
  }, [lines, tab]);

  return (
    <div className="space-y-2">
      <div role="tablist" aria-label="Log stream" className="flex gap-1">
        {(["stdout", "stderr"] as const).map((s) => (
          <button
            key={s}
            role="tab"
            type="button"
            aria-selected={tab === s}
            onClick={() => setTab(s)}
            className={cn(
              "rounded-md px-3 py-1 text-sm",
              tab === s ? "bg-muted font-medium" : "text-muted-foreground hover:bg-muted/50",
            )}
          >
            {s} ({(s === "stdout" ? stdout : stderr).length})
          </button>
        ))}
      </div>
      <div
        ref={box}
        role="log"
        aria-label={`${tab} log`}
        onScroll={(e) => {
          const el = e.currentTarget;
          atBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight <= BOTTOM_SLACK_PX;
        }}
        className="h-72 overflow-auto rounded-md border bg-muted/30 p-2 font-mono text-xs"
      >
        {lines.length === 0 ? (
          <div className="text-muted-foreground">No output</div>
        ) : (
          lines.map((l, i) => (
            <div key={i} className="whitespace-pre-wrap break-all">{l}</div>
          ))
        )}
      </div>
    </div>
  );
}
