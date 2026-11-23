"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Panel } from "@/components/matra";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { evalAudioUrl, getEvalSamples } from "@/lib/api";
import type { Task } from "@/lib/types";
import { wordDiff } from "@/lib/word-diff";

const PAGE = 50;
type Item = Record<string, unknown>;
const str = (v: unknown) => (typeof v === "string" ? v : v == null ? "" : String(v));

function lastUserTurn(prompt: unknown): string {
  if (!Array.isArray(prompt)) return str(prompt);
  const turns = prompt as { role?: string; content?: unknown }[];
  const t = [...turns].reverse().find((m) => m.role === "user") ?? turns[turns.length - 1];
  return t ? str(t.content) : "";
}

function Diff({ reference, hypothesis }: { reference: string; hypothesis: string }) {
  return (
    <span className="leading-relaxed">
      {wordDiff(reference, hypothesis).map((d, i) => (
        <span key={i}>
          {d.op === "eq" && d.ref}
          {d.op === "del" && <del className="text-status-failed">{d.ref}</del>}
          {d.op === "ins" && <ins className="text-status-done">{d.hyp}</ins>}
          {d.op === "sub" && (
            <>
              <del className="text-status-failed">{d.ref}</del> <ins className="text-status-done">{d.hyp}</ins>
            </>
          )}{" "}
        </span>
      ))}
    </span>
  );
}

/** `task` may be null (the run detail does not carry it): the item shape then decides the layout. */
export function EvalSamples({ runId, task }: { runId: string; task: Task | null }) {
  const [items, setItems] = useState<Item[]>([]);
  const [total, setTotal] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [exhausted, setExhausted] = useState(false);
  const [loading, setLoading] = useState(true);
  const [sortWer, setSortWer] = useState(false);
  const loadingRef = useRef(false);
  const itemsLen = useRef(0);

  const load = useCallback(async () => {
    if (loadingRef.current) return;
    loadingRef.current = true;
    setLoading(true);
    try {
      const page = await getEvalSamples(runId, itemsLen.current, PAGE);
      itemsLen.current += page.items.length;
      setExhausted(page.items.length === 0);
      setError(null);
      setItems((cur) => [...cur, ...page.items]);
      setTotal(page.total);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      loadingRef.current = false;
      setLoading(false);
    }
  }, [runId]);

  useEffect(() => {
    void load();
  }, [load]);

  const kind: Task | null = task ?? (items[0] ? ("wer" in items[0] ? "asr" : "audio" in items[0] ? "tts" : "llm-sft") : null);

  if (error && items.length === 0) return <p className="text-sm text-status-failed">Cannot load eval samples ({error})</p>;
  if (!loading && items.length === 0) return <p className="text-sm text-muted-foreground">No eval samples for this run.</p>;

  const rows = kind === "asr" && sortWer ? [...items].sort((a, b) => Number(b.wer) - Number(a.wer)) : items;

  return (
    <Panel title="Eval samples" actions={<span className="font-mono text-[13px] text-muted-foreground">{items.length} / {total}</span>}>
      {error && <p className="mb-3 text-sm text-status-failed">Cannot load more eval samples ({error})</p>}
      {kind === "tts" && (
        <ul className="space-y-3">
          {rows.map((r) => (
            <li key={str(r.sample_id)} className="flex min-w-0 flex-wrap items-center gap-3 text-sm">
              <span className="min-w-0 flex-1">{str(r.text)}</span>
              <audio controls preload="none" src={evalAudioUrl(runId, str(r.audio))} className="h-8" />
            </li>
          ))}
        </ul>
      )}
      {kind === "llm-sft" && (
        <Table>
          <TableHeader>
            <TableRow className="hover:bg-transparent"><TableHead>Prompt</TableHead><TableHead>Reference</TableHead><TableHead>Reply</TableHead></TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((r) => (
              <TableRow key={str(r.sample_id)}>
                <TableCell className="whitespace-pre-wrap align-top">{lastUserTurn(r.prompt)}</TableCell>
                <TableCell className="whitespace-pre-wrap align-top">{str(r.reference)}</TableCell>
                <TableCell className="whitespace-pre-wrap align-top">{str(r.reply)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
      {kind === "asr" && (
        <Table>
          <TableHeader>
            <TableRow className="hover:bg-transparent">
              <TableHead>Diff</TableHead>
              <TableHead>
                <button type="button" className="underline-offset-2 hover:underline" onClick={() => setSortWer((s) => !s)}>
                  WER{sortWer ? " ↓" : ""}
                </button>
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((r) => (
              <TableRow key={str(r.sample_id)}>
                <TableCell className="whitespace-normal align-top"><Diff reference={str(r.reference)} hypothesis={str(r.hypothesis)} /></TableCell>
                <TableCell className="font-mono align-top">{Number(r.wer).toFixed(3)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
      {items.length < total && !exhausted && (
        <Button variant="outline" className="mt-3" disabled={loading} onClick={() => void load()}>Load more</Button>
      )}
    </Panel>
  );
}
