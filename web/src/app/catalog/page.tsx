"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Radio } from "lucide-react";
import { Suspense, useEffect, useMemo, useState } from "react";
import { LicensePill } from "@/components/catalog-detail";
import { FilterCheckbox, FilterSelect } from "@/components/filter-select";
import { PageHeader, Panel } from "@/components/matra";
import { EmptyState, ErrorNotice, WarningNotice } from "@/components/notice";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { listCatalog } from "@/lib/api";
import { BANGLA_LABEL, ROLE_LABEL, SECTIONS, filterCatalog, type CatalogFilter } from "@/lib/catalog";
import type { BanglaStatus, CatalogList, CatalogRole } from "@/lib/types";
import { cn } from "@/lib/utils";

const ROLES = Object.entries(ROLE_LABEL).map(([value, label]) => ({ value: value as CatalogRole, label }));
const BANGLA = Object.entries(BANGLA_LABEL).map(([value, label]) => ({ value: value as BanglaStatus, label }));

function CatalogView() {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  const section = SECTIONS.find((s) => s.task === params.get("task")) ?? SECTIONS[0];
  const filter: CatalogFilter = {
    role: (ROLES.find((r) => r.value === params.get("role"))?.value) ?? undefined,
    bangla: (BANGLA.find((b) => b.value === params.get("bangla"))?.value) ?? undefined,
    commercialOnly: params.get("commercial") === "1",
  };

  const [data, setData] = useState<CatalogList | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    listCatalog()
      .then((d) => active && setData(d))
      .catch((e) => active && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      active = false;
    };
  }, []);

  const set = (key: string, value: string | null) => {
    const q = new URLSearchParams(params.toString());
    if (value) q.set(key, value);
    else q.delete(key);
    router.replace(`${pathname}?${q.toString()}`, { scroll: false });
  };

  const counts = useMemo(
    () => Object.fromEntries(SECTIONS.map((s) => [s.task, data?.entries.filter((e) => e.task === s.task).length ?? 0])),
    [data],
  );
  const rows = useMemo(
    () => filterCatalog(data?.entries.filter((e) => e.task === section.task) ?? [], filter),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [data, section.task, filter.role, filter.bangla, filter.commercialOnly],
  );

  return (
    <div className="space-y-5">
      <PageHeader title="Model catalog">
        Base models you can fine-tune, with the data they need and what to train. Pick one and choose Fine-tune to start a recipe.
      </PageHeader>

      <div role="tablist" aria-label="Model type" className="flex gap-1 border-b border-border">
        {SECTIONS.map((s) => {
          const active = s.task === section.task;
          return (
            <button
              key={s.key}
              type="button"
              role="tab"
              aria-selected={active}
              onClick={() => set("task", s.task)}
              className={cn(
                "relative -mb-px flex h-10 items-center gap-2 px-4 text-sm transition-colors duration-150",
                active ? "font-semibold text-foreground" : "text-muted-foreground hover:text-foreground",
              )}
            >
              {s.label}
              <span className="font-mono text-[12px] text-muted-foreground">{counts[s.task]}</span>
              <span aria-hidden className={cn("absolute inset-x-2 bottom-0 h-0.5", active ? "bg-saffron" : "bg-transparent")} />
            </button>
          );
        })}
      </div>

      <div className="flex flex-wrap items-end gap-4">
        <FilterSelect label="Role" value={filter.role ?? ""} options={ROLES} onChange={(v) => set("role", v || null)} />
        <FilterSelect label="Bangla" value={filter.bangla ?? ""} options={BANGLA} onChange={(v) => set("bangla", v || null)} />
        <FilterCheckbox label="Commercial use only" checked={!!filter.commercialOnly} onChange={(v) => set("commercial", v ? "1" : null)} />
      </div>

      {error && <ErrorNotice>Cannot load the catalog ({error})</ErrorNotice>}
      {data && data.errors.length > 0 && (
        <WarningNotice>
          <p className="font-medium">{data.errors.length === 1 ? "One catalog file failed to load." : `${data.errors.length} catalog files failed to load.`}</p>
          <ul className="mt-1 space-y-1">
            {data.errors.map((e) => (
              <li key={e.file}>
                <span className="font-mono text-[13px]">{e.file}</span>
                <pre className="mt-0.5 max-h-24 overflow-auto font-mono text-[12px] whitespace-pre-wrap text-muted-foreground">{e.message}</pre>
              </li>
            ))}
          </ul>
        </WarningNotice>
      )}

      <Panel>
        {data && rows.length === 0 ? (
          <EmptyState>
            {data.entries.some((e) => e.task === section.task)
              ? "No models match these filters."
              : `No catalog entries for this section. Add a YAML file under catalog/${section.key}/.`}
          </EmptyState>
        ) : (
          <div role="tabpanel" aria-label={section.label}>
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead>Model</TableHead>
                  <TableHead>Params</TableHead>
                  <TableHead>Role</TableHead>
                  <TableHead>License</TableHead>
                  <TableHead>Bangla</TableHead>
                  <TableHead>Streaming</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((e) => (
                  <TableRow key={e.id}>
                    <TableCell className="max-w-72">
                      <Link href={`/catalog/${encodeURIComponent(e.id)}`} className="block truncate font-medium hover:text-saffron" title={e.name}>
                        {e.name}
                      </Link>
                      <div className="truncate font-mono text-[12px] text-muted-foreground" title={e.hf_repo}>{e.hf_repo}</div>
                    </TableCell>
                    <TableCell className="font-mono">{e.params}</TableCell>
                    <TableCell className={e.role === "primary" ? "text-saffron" : "text-muted-foreground"}>{ROLE_LABEL[e.role]}</TableCell>
                    <TableCell><LicensePill id={e.license.id} commercialOk={e.license.commercial_ok} /></TableCell>
                    <TableCell>{BANGLA_LABEL[e.bangla_status]}</TableCell>
                    <TableCell>
                      {e.streaming ? (
                        <span className="inline-flex items-center gap-1"><Radio className="size-3.5 text-saffron" aria-hidden />Yes</span>
                      ) : (
                        <span className="text-muted-foreground">No</span>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </Panel>
    </div>
  );
}

export default function CatalogPage() {
  return (
    <Suspense fallback={null}>
      <CatalogView />
    </Suspense>
  );
}
