"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { audioUrl } from "@/lib/api";
import { formatDuration } from "@/lib/format";
import type { SampleRow, Task } from "@/lib/types";

function compactMessages(json: unknown): string {
  try {
    const msgs = JSON.parse(String(json)) as { role?: string; content?: unknown }[];
    return msgs
      .map((m) => `${m.role ?? "?"}: ${typeof m.content === "string" ? m.content : JSON.stringify(m.content)}`)
      .join(" | ");
  } catch {
    return String(json ?? "");
  }
}

export function SampleTable({
  task, dataset, version, rows, onEdit,
}: {
  task: Task;
  dataset: string;
  version: number;
  rows: SampleRow[];
  onEdit?: (sampleId: string, text: string) => Promise<void> | void;
}) {
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const audio = task !== "llm-sft";

  async function save(id: string) {
    if (!onEdit) return;
    setSaving(true);
    setError(null);
    try {
      await onEdit(id, draft);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="space-y-2">
      {error && (
        <div role="alert" className="rounded-md border border-red-500/50 bg-red-500/10 px-3 py-2 text-sm">
          {error}
        </div>
      )}
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Sample</TableHead>
            <TableHead>Split</TableHead>
            {audio ? (
              <>
                <TableHead>File</TableHead>
                <TableHead>Text</TableHead>
                <TableHead>Duration</TableHead>
                <TableHead>Audio</TableHead>
                {onEdit && <TableHead />}
              </>
            ) : (
              <TableHead>Messages</TableHead>
            )}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.length === 0 && (
            <TableRow>
              <TableCell colSpan={7} className="text-muted-foreground">
                No samples
              </TableCell>
            </TableRow>
          )}
          {rows.map((r) => (
            <TableRow key={r.sample_id}>
              <TableCell className="font-mono text-xs">{r.sample_id}</TableCell>
              <TableCell>{r.split}</TableCell>
              {audio ? (
                <>
                  <TableCell className="text-xs">{r.file}</TableCell>
                  <TableCell className="max-w-md whitespace-normal">
                    {editing === r.sample_id ? (
                      <textarea
                        aria-label="Transcript"
                        className="w-full rounded-md border bg-transparent p-1 text-sm"
                        rows={2}
                        value={draft}
                        onChange={(e) => setDraft(e.target.value)}
                      />
                    ) : (
                      r.text
                    )}
                  </TableCell>
                  <TableCell>{typeof r.duration_s === "number" ? formatDuration(r.duration_s) : ""}</TableCell>
                  <TableCell>
                    <audio controls preload="none" src={audioUrl(dataset, version, r.sample_id)} />
                  </TableCell>
                  {onEdit && (
                    <TableCell className="space-x-1">
                      {editing === r.sample_id ? (
                        <>
                          <Button size="sm" disabled={saving || !draft.trim()} onClick={() => void save(r.sample_id)}>
                            Save
                          </Button>
                          <Button size="sm" variant="outline" disabled={saving} onClick={() => setEditing(null)}>
                            Cancel
                          </Button>
                        </>
                      ) : (
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => {
                            setEditing(r.sample_id);
                            setDraft(r.text ?? "");
                            setError(null);
                          }}
                        >
                          Edit
                        </Button>
                      )}
                    </TableCell>
                  )}
                </>
              ) : (
                <TableCell className="max-w-xl whitespace-normal text-xs">
                  {compactMessages(r.messages_json).slice(0, 300)}
                </TableCell>
              )}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
