"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { CatalogDetail } from "@/components/catalog-detail";
import { PageHeader } from "@/components/matra";
import { ErrorNotice } from "@/components/notice";
import { ApiError, getCatalogEntry } from "@/lib/api";
import type { CatalogEntry } from "@/lib/types";

export default function CatalogEntryPage() {
  const { id } = useParams<{ id: string }>();
  return <CatalogEntryView key={id} id={decodeURIComponent(id)} />;
}

function CatalogEntryView({ id }: { id: string }) {
  const [entry, setEntry] = useState<CatalogEntry | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    getCatalogEntry(id)
      .then((e) => active && setEntry(e))
      .catch((e) => {
        if (!active) return;
        setError(e instanceof ApiError && e.status === 404 ? `No catalog entry named ${id}.` : e instanceof Error ? e.message : String(e));
      });
    return () => {
      active = false;
    };
  }, [id]);

  if (error) {
    return (
      <div className="space-y-4">
        <PageHeader title={id} />
        <ErrorNotice>
          {error} <Link href="/catalog" className="underline">Back to the catalog</Link>
        </ErrorNotice>
      </div>
    );
  }
  if (!entry) return <PageHeader title={id} />;
  return (
    <div>
      <PageHeader title={entry.name} />
      <CatalogDetail entry={entry} />
    </div>
  );
}
