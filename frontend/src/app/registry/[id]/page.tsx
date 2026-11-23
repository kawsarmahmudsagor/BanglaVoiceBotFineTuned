"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { Crown } from "lucide-react";
import { PageHeader, Panel } from "@/components/matra";
import { ErrorNotice } from "@/components/notice";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ApiError, exportUrl, getModel, setChampion } from "@/lib/api";
import { formatBytes } from "@/lib/format";
import { formatMetric } from "@/lib/models";
import type { ModelVersion } from "@/lib/types";

const errText = (e: unknown) => (e instanceof ApiError ? e.detail : e instanceof Error ? e.message : String(e));

export default function ModelDetailPage() {
  const { id } = useParams<{ id: string }>();
  // keyed by id so model state never leaks across routes
  return <ModelDetailView key={id} id={id} />;
}

function ModelDetailView({ id }: { id: string }) {
  const [model, setModel] = useState<ModelVersion | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let active = true;
    getModel(Number(id))
      .then((m) => active && setModel(m))
      .catch((e) => active && setError(errText(e)));
    return () => {
      active = false;
    };
  }, [id]);

  const onChampion = async () => {
    if (!model) return;
    setBusy(true);
    setError(null);
    try {
      await setChampion(model.id);
      setModel(await getModel(model.id));
    } catch (e) {
      setError(errText(e));
    } finally {
      setBusy(false);
    }
  };

  if (!model) {
    return error ? (
      <div className="space-y-4">
        <PageHeader title={`Model ${id}`} />
        <ErrorNotice>Cannot load model ({error})</ErrorNotice>
      </div>
    ) : (
      <PageHeader title={`Model ${id}`}>Loading model…</PageHeader>
    );
  }

  const metrics = Object.entries(model.metrics).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
  return (
    <div className="space-y-5">
      <PageHeader
        title={<>{model.name} <span className="font-mono">v{model.number}</span></>}
        actions={
          <Button size="lg" className="px-3" disabled={busy || model.champion} onClick={onChampion}>
            {model.champion ? "Current champion" : "Set as champion"}
          </Button>
        }
      >
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
          <span className="font-mono">{model.task}</span>
          {model.champion && (
            <span className="inline-flex items-center gap-1 text-saffron">
              <Crown className="size-3.5" aria-hidden />
              Champion
            </span>
          )}
          <span className={model.commercial_ok ? "text-status-done" : "text-status-failed"}>
            {model.commercial_ok ? "commercial OK" : "non-commercial"}
          </span>
        </div>
      </PageHeader>
      {error && <ErrorNotice>{error}</ErrorNotice>}

      <Panel title="Provenance">
        <dl className="grid gap-3 text-sm sm:grid-cols-2">
          <div className="min-w-0">
            <dt className="text-[13px] text-muted-foreground">Base model</dt>
            <dd className="font-mono text-[13px] break-all">{model.base_model}</dd>
          </div>
          <div className="min-w-0">
            <dt className="text-[13px] text-muted-foreground">Source run</dt>
            <dd><Link href={`/runs/${encodeURIComponent(model.run_id)}`} className="font-mono text-[13px] underline">{model.run_id}</Link></dd>
          </div>
        </dl>
      </Panel>

      <div className="grid items-start gap-4 lg:grid-cols-2">
        <Panel title="Metrics">
          {metrics.length === 0 ? (
            <p className="text-sm text-muted-foreground">No metrics.</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead>Metric</TableHead>
                  <TableHead>Value</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {metrics.map(([k, v]) => (
                  <TableRow key={k}>
                    <TableCell className="font-mono">{k}</TableCell>
                    <TableCell className="font-mono">{formatMetric(v)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </Panel>

        <Panel title="Exports">
          {model.exports.length === 0 ? (
            <p className="text-sm text-muted-foreground">No exports.</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead>Format</TableHead>
                  <TableHead>Size</TableHead>
                  <TableHead>Serve with</TableHead>
                  <TableHead><span className="sr-only">Download</span></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {model.exports.map((x, i) => (
                  <TableRow key={i}>
                    <TableCell className="font-mono">{x.format}</TableCell>
                    <TableCell className="font-mono">{formatBytes(x.size_bytes)}</TableCell>
                    <TableCell>{x.serve_with ?? ""}</TableCell>
                    <TableCell>
                      <a href={exportUrl(model.id, i)} className="text-saffron underline">Download {x.format}</a>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </Panel>
      </div>
    </div>
  );
}
