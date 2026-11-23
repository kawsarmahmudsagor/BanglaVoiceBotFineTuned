"use client";

import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { SeriesPoint } from "@/lib/metrics";

export function MetricCharts({ series }: { series: Record<string, SeriesPoint[]> }) {
  const keys = Object.keys(series).sort();
  if (keys.length === 0) return <p className="text-sm text-muted-foreground">No metrics yet</p>;
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {keys.map((key) => (
        <figure key={key} className="rounded-md border p-3">
          <figcaption className="mb-2 font-mono text-xs text-muted-foreground">{key}</figcaption>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={series[key]} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis dataKey="step" type="number" domain={["dataMin", "dataMax"]} fontSize={11} />
                <YAxis fontSize={11} width={48} domain={["auto", "auto"]} />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke="#3b82f6"
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
