"use client";

import Link from "next/link";
import { useParams, usePathname, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { ReportView } from "@/components/report-view";
import { SampleTable } from "@/components/sample-table";
import { X } from "lucide-react";
import { PageHeader, Panel } from "@/components/matra";
import { ErrorNotice } from "@/components/notice";
import { Button } from "@/components/ui/button";
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
      <PageHeader title={<><Link href={`/datasets/${enc}`} className="hover:text-saffron">{name}</Link> <span className="font-mono">v{n}</span></>}>
        {version && <span className="font-mono">{version.task}</span>}
      </PageHeader>
      {error && <ErrorNotice>{error}</ErrorNotice>}

      {version && (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <Panel title="Provenance">
            <div className="space-y-1 text-sm">
              {prov ? (
                <>
                  <div className="flex flex-wrap gap-2">
                    <span className="rounded-md border border-border px-2 py-0.5 font-mono text-[13px]">{prov.license || "no license"}</span>
                    <span className={prov.commercial_ok ? "text-status-done" : "text-status-failed"}>
                      {prov.commercial_ok ? "commercial OK" : "non-commercial"}
                    </span>
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
            </div>
          </Panel>
          <Panel title="Splits">
            <div className="space-y-1 text-sm">
              {Object.entries(version.splits).map(([k, v]) => (
                <div key={k}>
                  <span className="font-mono">{k}</span>: <span className="font-mono">{v}</span>
                </div>
              ))}
              <div className="text-muted-foreground"><span className="font-mono">{version.n_samples}</span> samples total</div>
            </div>
          </Panel>
        </div>
      )}

      {version && (
        <Panel title="Validation report">
          <ReportView report={version.report} dataset={name} version={n} />
        </Panel>
      )}

      <Panel title="Samples">
        <div className="space-y-3">
        <div className="flex flex-wrap items-center gap-3 text-sm">
          <label className="flex items-center gap-2">
            Split
            <select
              value={split}
              onChange={(e) => setQuery({ split: e.target.value || null, page: null })}
              className="h-9 rounded-md border border-border bg-card px-2"
            >
              <option value="">all</option>
              {SPLITS.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </label>
          {warning && (
            <span className="inline-flex items-center gap-1 rounded-md border border-saffron/60 px-2 py-0.5 text-[13px]">
              warning: <span className="font-mono">{warning}</span>
              <button
                aria-label="Clear warning filter"
                className="inline-flex size-6 items-center justify-center rounded hover:bg-raised"
                onClick={() => setQuery({ warning: null, page: null })}
              >
                <X className="size-3.5" aria-hidden />
              </button>
            </span>
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
          <span className="font-mono">
            {total === 0 ? "0" : `${page * PAGE + 1}-${Math.min((page + 1) * PAGE, total)}`} of {total}
          </span>
          <Button size="sm" variant="outline" disabled={(page + 1) * PAGE >= total} onClick={() => setQuery({ page: String(page + 1) })}>
            Next
          </Button>
        </div>
        </div>
      </Panel>
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
