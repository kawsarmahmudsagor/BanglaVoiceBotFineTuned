"use client";

import Link from "next/link";
import { useParams, usePathname, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { ReportView } from "@/components/report-view";
import { SampleTable } from "@/components/sample-table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { editTranscript, getSamples, getVersion } from "@/lib/api";
import type { SamplesPage, VersionDetail } from "@/lib/types";

const PAGE = 50;
const SPLITS = ["train", "val", "test"];

function VersionView() {
  const { name: rawName, n: rawN } = useParams<{ name: string; n: string }>();
  const name = decodeURIComponent(rawName);
  const n = Number(rawN);
  const router = useRouter();
  const pathname = usePathname();
  const sp = useSearchParams();
  const split = sp.get("split") ?? "";
  const warning = sp.get("warning") ?? "";
  const page = Math.max(0, Number(sp.get("page") ?? 0) || 0);

  const [version, setVersion] = useState<VersionDetail | null>(null);
  const [samples, setSamples] = useState<SamplesPage | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    getVersion(name, n)
      .then((v) => active && setVersion(v))
      .catch((e) => active && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      active = false;
    };
  }, [name, n]);

  useEffect(() => {
    let active = true;
    getSamples(name, n, { offset: page * PAGE, limit: PAGE, split: split || undefined, warning: warning || undefined })
      .then((s) => active && setSamples(s))
      .catch((e) => active && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      active = false;
    };
  }, [name, n, page, split, warning]);

  function setQuery(patch: Record<string, string | null>) {
    const next = new URLSearchParams(sp.toString());
    for (const [k, v] of Object.entries(patch)) {
      if (v) next.set(k, v);
      else next.delete(k);
    }
    const q = next.toString();
    router.replace(q ? `${pathname}?${q}` : pathname);
  }

  async function onEdit(sampleId: string, text: string) {
    const child = await editTranscript(name, n, sampleId, text);
    router.push(`/datasets/${encodeURIComponent(name)}/v/${child.number}`);
  }

  const total = samples?.total ?? 0;
  const prov = version?.provenance;
  const enc = encodeURIComponent(name);

  return (
    <div className="space-y-5">
      <h1 className="text-lg font-semibold">
        <Link href={`/datasets/${enc}`} className="underline">{name}</Link> v{n}
        {version && <span className="ml-2 font-mono text-sm text-muted-foreground">{version.task}</span>}
      </h1>
      {error && (
        <div role="alert" className="rounded-md border border-red-500/50 bg-red-500/10 px-3 py-2 text-sm">
          {error}
        </div>
      )}

      {version && (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <Card size="sm">
            <CardHeader>
              <CardTitle>Provenance</CardTitle>
            </CardHeader>
            <CardContent className="space-y-1 text-sm">
              {prov ? (
                <>
                  <div className="flex gap-2">
                    <Badge variant="outline">{prov.license || "no license"}</Badge>
                    <Badge variant={prov.commercial_ok ? "secondary" : "destructive"}>
                      {prov.commercial_ok ? "commercial OK" : "non-commercial"}
                    </Badge>
                  </div>
                  <div>Source: {prov.source}</div>
                  <div>Consent: {prov.consent}</div>
                  {prov.notes && <div className="text-muted-foreground">{prov.notes}</div>}
                </>
              ) : (
                <div className="text-muted-foreground">No provenance recorded</div>
              )}
              {version.parent_number !== null && (
                <div className="text-muted-foreground">
                  Edited from{" "}
                  <Link href={`/datasets/${enc}/v/${version.parent_number}`} className="underline">
                    v{version.parent_number}
                  </Link>
                </div>
              )}
            </CardContent>
          </Card>
          <Card size="sm">
            <CardHeader>
              <CardTitle>Splits</CardTitle>
            </CardHeader>
            <CardContent className="space-y-1 text-sm">
              {Object.entries(version.splits).map(([k, v]) => (
                <div key={k}>
                  <span className="font-mono">{k}</span>: {v}
                </div>
              ))}
              <div className="text-muted-foreground">{version.n_samples} samples total</div>
            </CardContent>
          </Card>
        </div>
      )}

      {version && (
        <section className="space-y-2">
          <h2 className="text-sm font-semibold">Validation report</h2>
          <ReportView report={version.report} dataset={name} version={n} />
        </section>
      )}

      <section className="space-y-2">
        <div className="flex flex-wrap items-center gap-3 text-sm">
          <h2 className="font-semibold">Samples</h2>
          <label className="flex items-center gap-1">
            Split
            <select
              value={split}
              onChange={(e) => setQuery({ split: e.target.value || null, page: null })}
              className="h-8 rounded-md border bg-transparent px-2"
            >
              <option value="">all</option>
              {SPLITS.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </label>
          {warning && (
            <Badge variant="outline">
              warning: {warning}
              <button aria-label="Clear warning filter" onClick={() => setQuery({ warning: null, page: null })}>
                x
              </button>
            </Badge>
          )}
        </div>
        {version && samples && (
          <SampleTable
            task={version.task}
            dataset={name}
            version={n}
            rows={samples.rows}
            onEdit={version.task === "llm-sft" ? undefined : onEdit}
          />
        )}
        <div className="flex items-center gap-3 text-sm">
          <Button size="sm" variant="outline" disabled={page === 0} onClick={() => setQuery({ page: page > 1 ? String(page - 1) : null })}>
            Previous
          </Button>
          <span>
            {total === 0 ? "0" : `${page * PAGE + 1}-${Math.min((page + 1) * PAGE, total)}`} of {total}
          </span>
          <Button size="sm" variant="outline" disabled={(page + 1) * PAGE >= total} onClick={() => setQuery({ page: String(page + 1) })}>
            Next
          </Button>
        </div>
      </section>
    </div>
  );
}

export default function VersionPage() {
  return (
    <Suspense fallback={null}>
      <VersionView />
    </Suspense>
  );
}
