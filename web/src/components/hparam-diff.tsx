import type { DiffRow } from "@/lib/compare";
import { cn } from "@/lib/utils";

function Value({ v }: { v: unknown }) {
  if (v === undefined) return <span className="text-muted-foreground italic">not set</span>;
  return <span className="break-all">{typeof v === "object" ? JSON.stringify(v) : String(v)}</span>;
}

function Rows({ rows, runIds }: { rows: DiffRow[]; runIds: string[] }) {
  return (
    <>
      {rows.map((r) => (
        <tr
          key={r.key}
          data-differs={r.differs ? "true" : "false"}
          className={cn("border-b border-border last:border-0", r.differs && "bg-saffron/10")}
        >
          <th scope="row" className="px-3 py-1.5 text-left align-top font-mono text-[13px] font-normal">
            <span className="break-all">{r.key}</span>
            {r.differs && <span className="ml-2 rounded bg-saffron/20 px-1.5 py-0.5 font-sans text-[12px] font-medium text-saffron">Differs</span>}
          </th>
          {runIds.map((id) => (
            <td key={id} className="px-3 py-1.5 align-top font-mono text-[13px]">
              {id in r.values ? <Value v={r.values[id]} /> : <span className="text-muted-foreground">no config</span>}
            </td>
          ))}
        </tr>
      ))}
    </>
  );
}

function Head({ runIds }: { runIds: string[] }) {
  return (
    <thead>
      <tr className="border-b border-border">
        <th scope="col" className="h-9 px-3 text-left text-[13px] font-medium text-muted-foreground">Setting</th>
        {runIds.map((id) => (
          <th key={id} scope="col" className="h-9 px-3 text-left font-mono text-[13px] font-medium text-muted-foreground">{id}</th>
        ))}
      </tr>
    </thead>
  );
}

/** Hyperparameter diff: differing settings first and labelled, identical ones folded away. */
export function HparamDiff({ rows, runIds }: { rows: DiffRow[]; runIds: string[] }) {
  const differ = rows.filter((r) => r.differs);
  const same = rows.filter((r) => !r.differs);
  return (
    <div className="space-y-3">
      {differ.length > 0 ? (
        <div className="overflow-x-auto rounded-md border border-border">
          <table className="w-full text-sm">
            <Head runIds={runIds} />
            <tbody><Rows rows={differ} runIds={runIds} /></tbody>
          </table>
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">No settings differ between these runs.</p>
      )}
      {same.length > 0 && (
        <details className="rounded-md border border-border">
          <summary className="cursor-pointer px-3 py-2 text-sm text-muted-foreground hover:text-foreground">{`Same in all runs (${same.length})`}</summary>
          <div className="overflow-x-auto border-t border-border">
            <table className="w-full text-sm">
              <Head runIds={runIds} />
              <tbody><Rows rows={same} runIds={runIds} /></tbody>
            </table>
          </div>
        </details>
      )}
    </div>
  );
}
