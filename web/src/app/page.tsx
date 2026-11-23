"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { getSystem, listModels, listRuns } from "@/lib/api";
import { stateColor } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { ModelVersion, RunListItem, SystemInfo, Task } from "@/lib/types";

const POLL_MS = 5000;
const TASKS: Task[] = ["llm-sft", "asr", "tts"];

function GpuCard({ gpu }: { gpu: SystemInfo["gpu"] }) {
  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle>GPU</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        {gpu ? (
          <>
            <div className="font-medium">{gpu.name}</div>
            <Progress value={(gpu.vram_used_mb / Math.max(gpu.vram_total_mb, 1)) * 100} aria-label="VRAM used" />
            <div className="text-muted-foreground">
              VRAM {(gpu.vram_used_mb / 1024).toFixed(1)} / {(gpu.vram_total_mb / 1024).toFixed(1)} GB
            </div>
            <div className="text-muted-foreground">
              Util {gpu.util_pct}% · {gpu.temp_c} °C
            </div>
          </>
        ) : (
          <div className="text-muted-foreground">GPU not visible</div>
        )}
      </CardContent>
    </Card>
  );
}

function LockCard({ lock }: { lock: SystemInfo["lock"] }) {
  const held = lock.holder_kind !== null;
  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle>GPU lock</CardTitle>
      </CardHeader>
      <CardContent className="text-sm">
        {held ? (
          <div>
            Held by {lock.holder_kind}{" "}
            {lock.holder_kind === "run" && lock.holder_id ? (
              <Link href={`/runs/${lock.holder_id}`} className="font-mono underline">
                {lock.holder_id}
              </Link>
            ) : (
              <span className="font-mono">{lock.holder_id}</span>
            )}
          </div>
        ) : (
          <div className="text-muted-foreground">Free</div>
        )}
      </CardContent>
    </Card>
  );
}

function DiskCard({ disk }: { disk: SystemInfo["disk"] }) {
  const tone = disk.block
    ? "border-red-500 bg-red-500/10"
    : disk.warn
      ? "border-amber-500 bg-amber-500/10"
      : "";
  return (
    <Card size="sm" data-testid="disk-card" className={cn(tone)}>
      <CardHeader>
        <CardTitle>Disk</CardTitle>
      </CardHeader>
      <CardContent className="text-sm">
        <div className="font-medium">{disk.free_gb.toFixed(1)} GB free</div>
        {disk.block ? (
          <div className="text-red-700 dark:text-red-300">Too low: uploads and new runs are blocked</div>
        ) : disk.warn ? (
          <div className="text-amber-700 dark:text-amber-300">Running low</div>
        ) : (
          <div className="text-muted-foreground">OK</div>
        )}
      </CardContent>
    </Card>
  );
}

function QueueCard({ queue }: { queue: SystemInfo["queue"] }) {
  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle>Queue</CardTitle>
      </CardHeader>
      <CardContent className="text-sm">
        <div>{queue.running} running</div>
        <div>{queue.queued} queued</div>
      </CardContent>
    </Card>
  );
}

export default function Dashboard() {
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [runs, setRuns] = useState<RunListItem[] | null>(null);
  const [models, setModels] = useState<ModelVersion[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    const refresh = async () => {
      const [sys, rs, ms] = await Promise.allSettled([getSystem(), listRuns(undefined, 10), listModels()]);
      if (!active) return;
      if (sys.status === "fulfilled") setSystem(sys.value);
      if (rs.status === "fulfilled") setRuns(rs.value);
      if (ms.status === "fulfilled") setModels(ms.value.filter((m) => m.champion));
      const failed = [sys, rs, ms].find((r) => r.status === "rejected");
      setError(
        failed && failed.status === "rejected"
          ? failed.reason instanceof Error
            ? failed.reason.message
            : String(failed.reason)
          : null,
      );
    };
    void refresh();
    const timer = setInterval(() => void refresh(), POLL_MS);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  return (
    <div className="space-y-5">
      <h1 className="text-lg font-semibold">Dashboard</h1>

      {error && (
        <div role="alert" className="rounded-md border border-red-500/50 bg-red-500/10 px-3 py-2 text-sm">
          Cannot reach the API ({error}). Retrying every 5 s; check that the SSH tunnel to port 8000 is up.
        </div>
      )}

      {system && (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <GpuCard gpu={system.gpu} />
          <LockCard lock={system.lock} />
          <DiskCard disk={system.disk} />
          <QueueCard queue={system.queue} />
        </div>
      )}

      <section>
        <h2 className="mb-2 text-sm font-semibold">Champions</h2>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Task</TableHead>
              <TableHead>Model</TableHead>
              <TableHead>Base</TableHead>
              <TableHead>Commercial</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {TASKS.map((task) => {
              const m = models?.find((x) => x.task === task);
              return (
                <TableRow key={task}>
                  <TableCell className="font-mono">{task}</TableCell>
                  {m ? (
                    <>
                      <TableCell>
                        <Link href={`/models/${m.id}`} className="underline">
                          {m.name} v{m.number}
                        </Link>
                      </TableCell>
                      <TableCell>{m.base_model}</TableCell>
                      <TableCell>{m.commercial_ok ? "OK" : "No"}</TableCell>
                    </>
                  ) : (
                    <TableCell colSpan={3} className="text-muted-foreground">
                      {models ? "No champion" : "..."}
                    </TableCell>
                  )}
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </section>

      <section>
        <h2 className="mb-2 text-sm font-semibold">Latest runs</h2>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Run</TableHead>
              <TableHead>Recipe</TableHead>
              <TableHead>State</TableHead>
              <TableHead>Created</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {runs?.length === 0 && (
              <TableRow>
                <TableCell colSpan={4} className="text-muted-foreground">
                  No runs yet
                </TableCell>
              </TableRow>
            )}
            {runs?.map((r) => (
              <TableRow key={r.id}>
                <TableCell>
                  <Link href={`/runs/${r.id}`} className="font-mono underline">
                    {r.id}
                  </Link>
                </TableCell>
                <TableCell>{r.recipe}</TableCell>
                <TableCell>
                  <Badge variant="outline" className={cn("border-transparent", stateColor(r.state))}>
                    {r.state}
                  </Badge>
                </TableCell>
                <TableCell className="text-muted-foreground">
                  {r.created_at ? new Date(r.created_at).toLocaleString() : ""}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </section>
    </div>
  );
}
