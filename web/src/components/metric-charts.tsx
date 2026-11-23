"use client";

import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { SeriesPoint } from "@/lib/metrics";
import { seriesColor } from "@/lib/palette";
import { CHART_CHROME, useThemeValue } from "@/lib/use-theme";

export function MetricCharts({ series, compact = false }: { series: Record<string, SeriesPoint[]>; compact?: boolean }) {
  const theme = useThemeValue();
  const chrome = CHART_CHROME[theme];
  const keys = Object.keys(series).sort();
  if (keys.length === 0) return <p className="text-sm text-muted-foreground">No metrics yet.</p>;
  return (
    <div className={compact ? "grid gap-3" : "grid gap-4 md:grid-cols-2"}>
      {keys.map((key, i) => (
        <figure key={key} className="min-w-0 rounded-md border border-border p-3">
          <figcaption className="mb-2 font-mono text-[13px] text-muted-foreground">{key}</figcaption>
          <div className={compact ? "h-32" : "h-48"}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={series[key]} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                <CartesianGrid stroke={chrome.grid} strokeDasharray="3 3" />
                <XAxis dataKey="step" type="number" domain={["dataMin", "dataMax"]} fontSize={11} stroke={chrome.axis} tick={{ fill: chrome.axis }} />
                <YAxis fontSize={11} width={48} domain={["auto", "auto"]} stroke={chrome.axis} tick={{ fill: chrome.axis }} />
                <Tooltip
                  contentStyle={{ background: chrome.tooltipBg, border: `1px solid ${chrome.tooltipBorder}`, borderRadius: 6, color: chrome.text, fontSize: 12 }}
                  labelStyle={{ color: chrome.axis }}
                />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke={seriesColor(i === 0 ? 1 : i, theme)}
                  strokeWidth={2}
                  dot={series[key].length <= 1}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </figure>
      ))}
    </div>
  );
}
