import Link from "next/link";
import { ExternalLink, Radio } from "lucide-react";
import type { ReactNode } from "react";
import { InlineMarkdown, Markdown } from "@/components/markdown";
import { Panel } from "@/components/matra";
import { buttonVariants } from "@/components/ui/button";
import { BANGLA_LABEL, ROLE_LABEL, sectionForTask } from "@/lib/catalog";
import type { CatalogEntry } from "@/lib/types";
import { cn } from "@/lib/utils";

export function Pill({ children, tone = "plain", title }: { children: ReactNode; tone?: "plain" | "good" | "bad" | "accent"; title?: string }) {
  return (
    <span
      title={title}
      className={cn(
        "inline-flex h-6 items-center gap-1 rounded-md border px-2 text-[13px] whitespace-nowrap",
        tone === "plain" && "border-border text-muted-foreground",
        tone === "good" && "border-status-done/50 text-status-done",
        tone === "bad" && "border-status-failed/50 text-status-failed",
        tone === "accent" && "border-saffron/60 text-saffron",
      )}
    >
      {children}
    </span>
  );
}

export function LicensePill({ id, commercialOk }: { id: string; commercialOk: boolean }) {
  return (
    <Pill tone={commercialOk ? "good" : "bad"} title={commercialOk ? "Commercial use allowed" : "Not cleared for commercial use"}>
      <span className="font-mono">{id}</span>
      <span aria-hidden>·</span>
      {commercialOk ? "commercial OK" : "non-commercial"}
    </Pill>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="min-w-0">
      <dt className="text-[13px] text-muted-foreground">{label}</dt>
      <dd className="mt-0.5 min-w-0 break-words">{children}</dd>
    </div>
  );
}

function KeyValueTable({ data, caption }: { data: Record<string, unknown>; caption: string }) {
  const rows = Object.entries(data);
  if (rows.length === 0) return null;
  return (
    <div className="overflow-x-auto rounded-md border border-border">
      <table className="w-full text-sm">
        <caption className="sr-only">{caption}</caption>
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k} className="border-b border-border last:border-0">
              <th scope="row" className="w-1/2 px-3 py-1.5 text-left font-mono text-[13px] font-normal text-muted-foreground">{k}</th>
              <td className="px-3 py-1.5 font-mono text-[13px] break-all">{typeof v === "object" ? JSON.stringify(v) : String(v)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function List({ items }: { items: string[] }) {
  return (
    <ul className="list-disc space-y-1.5 pl-5">
      {items.map((t, i) => (
        <li key={i} className="leading-relaxed">
          <InlineMarkdown text={t} />
        </li>
      ))}
    </ul>
  );
}

function Tags({ items, mono, highlightFirst }: { items: string[]; mono?: boolean; highlightFirst?: boolean }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((t, i) => (
        <span key={t} className={cn("rounded-md bg-raised px-2 py-0.5 text-[13px]", mono && "font-mono", highlightFirst && i === 0 && "ring-1 ring-saffron/60")}>
          {t}
        </span>
      ))}
    </div>
  );
}

export function CatalogHeaderMeta({ entry }: { entry: CatalogEntry }) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <a href={`https://huggingface.co/${entry.hf_repo}`} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 font-mono text-[13px] text-foreground hover:text-saffron">
        {entry.hf_repo}
        <ExternalLink className="size-3" aria-hidden />
        <span className="sr-only">(opens Hugging Face)</span>
      </a>
      <Pill tone={entry.role === "primary" ? "accent" : "plain"}>{ROLE_LABEL[entry.role]}</Pill>
      <LicensePill id={entry.license.id} commercialOk={entry.license.commercial_ok} />
      <Pill tone={entry.bangla.status === "native" || entry.bangla.status === "finetuned" ? "good" : "plain"}>{BANGLA_LABEL[entry.bangla.status]}</Pill>
      {entry.runtime.streaming && (
        <Pill>
          <Radio className="size-3" aria-hidden />
          Streaming
        </Pill>
      )}
    </div>
  );
}

export function CatalogDetail({ entry }: { entry: CatalogEntry }) {
  const { dataset, finetune, runtime, license } = entry;
  const section = sectionForTask(entry.task);
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <CatalogHeaderMeta entry={entry} />
        <Link href={`/train/new?catalog=${encodeURIComponent(entry.id)}`} className={cn(buttonVariants({ size: "lg" }), "px-4")}>
          Fine-tune
        </Link>
      </div>

      <Panel title="What it is">
        <div className="grid gap-5 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
          <Markdown text={entry.description} className="max-w-[72ch]" />
          <dl className="grid content-start gap-3 sm:grid-cols-2 lg:grid-cols-1">
            <Field label="Parameters"><span className="font-mono">{entry.params}</span></Field>
            {entry.derived_from && <Field label="Fine-tuned from"><span className="font-mono text-[13px]">{entry.derived_from}</span></Field>}
            <Field label="Runtime">
              <span className="font-mono text-[13px]">{runtime.engine}</span>
              {runtime.convert && <span className="text-muted-foreground"> after converting to <span className="font-mono">{runtime.convert}</span></span>}
            </Field>
            <Field label="Bangla">
              <InlineMarkdown text={entry.bangla.evidence} />
            </Field>
            <Field label="License">
              <a href={license.url} target="_blank" rel="noreferrer" className="font-mono text-[13px] underline underline-offset-2">{license.id}</a>
              {license.notes && <Markdown text={license.notes} className="mt-1 text-muted-foreground" />}
            </Field>
          </dl>
        </div>
      </Panel>

      <Panel
        title="Training data"
        actions={
          <Link href={`/datasets/new?task=${dataset.format}`} className="text-sm text-saffron underline underline-offset-2">
            Upload {dataset.format} dataset
          </Link>
        }
      >
        <div className="space-y-4">
          <dl className="grid gap-4 sm:grid-cols-2">
            <Field label="Required fields"><Tags items={dataset.required_fields} mono /></Field>
            <Field label="Optional fields">
              {dataset.optional_fields.length ? <Tags items={dataset.optional_fields} mono /> : <span className="text-muted-foreground">None</span>}
            </Field>
          </dl>
          <div>
            <div className="mb-1 text-[13px] text-muted-foreground">Example row ({section.label} upload format)</div>
            <pre className="overflow-x-auto rounded-md border border-border bg-ink p-3 font-mono text-[13px] whitespace-pre-wrap">{dataset.example}</pre>
          </div>
          <div>
            <div className="mb-1 text-[13px] text-muted-foreground">Recommended</div>
            <KeyValueTable data={dataset.recommended} caption="Recommended dataset size" />
          </div>
          {dataset.notes && <Markdown text={dataset.notes} className="max-w-[72ch]" />}
        </div>
      </Panel>

      <Panel title="Fine-tuning">
        <div className="space-y-5">
          <dl className="grid gap-4 sm:grid-cols-3">
            <Field label="Methods (first is recommended)"><Tags items={finetune.methods} mono highlightFirst /></Field>
            <Field label="Frameworks"><Tags items={finetune.frameworks} /></Field>
            <Field label="Watch these metrics"><Tags items={finetune.metrics} mono /></Field>
          </dl>
          <div>
            <h3 className="mb-2 text-sm font-semibold">What to train</h3>
            <List items={finetune.targets} />
          </div>
          <div>
            <h3 className="mb-2 text-sm font-semibold">Starting hyperparameters</h3>
            <KeyValueTable data={finetune.hyperparams} caption="Starting hyperparameters" />
          </div>
          <div>
            <h3 className="mb-2 text-sm font-semibold">Pitfalls</h3>
            <List items={finetune.pitfalls} />
          </div>
        </div>
      </Panel>

      <Panel title="Sources">
        <ul className="space-y-1.5">
          {entry.sources.map((s) => (
            <li key={s.url} className="min-w-0 break-words">
              <a href={s.url} target="_blank" rel="noreferrer" className="underline underline-offset-2 hover:text-saffron">{s.title}</a>
            </li>
          ))}
        </ul>
        <p className="mt-3 text-[13px] text-muted-foreground">Facts checked on <span className="font-mono">{entry.verified}</span>.</p>
      </Panel>
    </div>
  );
}
