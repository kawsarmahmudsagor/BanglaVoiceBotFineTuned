"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
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
      <div role="alert" className="rounded-md border border-red-500/50 bg-red-500/10 px-3 py-2 text-sm">
        Cannot load model ({error})
      </div>
    ) : (
      <p className="text-sm text-muted-foreground">Loading model...</p>
    );
  }

  const metrics = Object.entries(model.metrics).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-lg font-semibold">{model.name} v{model.number}</h1>
        <Badge variant="outline" className="font-mono">{model.task}</Badge>
        {model.champion && <Badge>champion</Badge>}
        <Badge variant={model.commercial_ok ? "secondary" : "outline"}>
          {model.commercial_ok ? "commercial OK" : "non-commercial"}
        </Badge>
        <Button className="ml-auto" disabled={busy || model.champion} onClick={onChampion}>
          Set champion
        </Button>
      </div>
      {error && (
        <div role="alert" className="rounded-md border border-red-500/50 bg-red-500/10 px-3 py-2 text-sm">
          {error}
        </div>
      )}
      <dl className="grid grid-cols-[8rem_1fr] gap-y-1 text-sm">
        <dt className="text-muted-foreground">Base model</dt>
        <dd className="font-mono">{model.base_model}</dd>
        <dt className="text-muted-foreground">Source run</dt>
        <dd>
          <Link href={`/runs/${encodeURIComponent(model.run_id)}`} className="font-mono underline">{model.run_id}</Link>
        </dd>
      </dl>

      <section className="space-y-2">
        <h2 className="text-sm font-medium">Metrics</h2>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Metric</TableHead>
              <TableHead>Value</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {metrics.length === 0 && (
              <TableRow><TableCell colSpan={2} className="text-muted-foreground">No metrics</TableCell></TableRow>
            )}
            {metrics.map(([k, v]) => (
              <TableRow key={k}>
                <TableCell className="font-mono">{k}</TableCell>
                <TableCell className="font-mono">{formatMetric(v)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </section>

      <section className="space-y-2">
        <h2 className="text-sm font-medium">Exports</h2>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Format</TableHead>
              <TableHead>Size</TableHead>
              <TableHead>Serve with</TableHead>
              <TableHead>Download</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {model.exports.length === 0 && (
              <TableRow><TableCell colSpan={4} className="text-muted-foreground">No exports</TableCell></TableRow>
            )}
            {model.exports.map((x, i) => (
              <TableRow key={i}>
                <TableCell className="font-mono">{x.format}</TableCell>
                <TableCell>{formatBytes(x.size_bytes)}</TableCell>
                <TableCell>{x.serve_with ?? ""}</TableCell>
                <TableCell>
                  <a href={exportUrl(model.id, i)} className="underline">Download</a>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </section>
    </div>
  );
}
