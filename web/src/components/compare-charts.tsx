"use client";

import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { MergedRow } from "@/lib/compare";
import { seriesColor } from "@/lib/palette";
import { CHART_CHROME, useThemeValue } from "@/lib/use-theme";

/** Shared legend: colour swatch + run id, so lines are never identified by colour alone. */
export function RunLegend({ runIds, missing = [] }: { runIds: string[]; missing?: string[] }) {
  const theme = useThemeValue();
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-[13px]">
      {runIds.map((id, i) => (
        <li key={id} className="flex items-center gap-1.5">
          <span aria-hidden className="inline-block h-0.5 w-4 rounded" style={{ background: seriesColor(i, theme), height: 3 }} />
          <span className="font-mono">{id}</span>
          {missing.includes(id) && <span className="text-muted-foreground">(no data)</span>}
        </li>
      ))}
    </ul>
  );
}

export function CompareCharts({ merged, runIds }: { merged: Record<string, MergedRow[]>; runIds: string[] }) {
  const theme = useThemeValue();
  const chrome = CHART_CHROME[theme];
  const keys = Object.keys(merged).sort();
  if (keys.length === 0) return <p className="text-sm text-muted-foreground">None of these runs has logged metrics yet.</p>;
  return (
    <div className="grid gap-4 xl:grid-cols-2">
      {keys.map((key) => {
        const rows = merged[key];
        const missing = runIds.filter((id) => !rows.some((r) => id in r));
        return (
          <figure key={key} className="min-w-0 rounded-md border border-border p-3">
            <figcaption className="mb-2 space-y-1.5">
              <div className="font-mono text-[13px] text-muted-foreground">{key}</div>
              <RunLegend runIds={runIds} missing={missing} />
            </figcaption>
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={rows} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                  <CartesianGrid stroke={chrome.grid} strokeDasharray="3 3" />
                  <XAxis dataKey="step" type="number" domain={["dataMin", "dataMax"]} fontSize={11} stroke={chrome.axis} tick={{ fill: chrome.axis }} />
                  <YAxis fontSize={11} width={48} domain={["auto", "auto"]} stroke={chrome.axis} tick={{ fill: chrome.axis }} />
                  <Tooltip
                    contentStyle={{ background: chrome.tooltipBg, border: `1px solid ${chrome.tooltipBorder}`, borderRadius: 6, color: chrome.text, fontSize: 12 }}
                    labelStyle={{ color: chrome.axis }}
                  />
                  {runIds.map((id, i) =>
                    missing.includes(id) ? null : (
                      <Line
                        key={id}
                        type="monotone"
                        dataKey={id}
                        name={id}
                        stroke={seriesColor(i, theme)}
                        strokeWidth={2}
                        dot={rows.length <= 1}
                        connectNulls
                        isAnimationActive={false}
                      />
                    ),
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </figure>
        );
      })}
    </div>
  );
}
