import { useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { formatDuration } from "./ProgressRow";

type Range = [number, number];

/** Sorts ranges by start and merges overlapping or touching ones. */
export function mergeRanges(ranges: Range[]): Range[] {
  const sorted = ranges.map((r) => [...r] as Range).sort((a, b) => a[0] - b[0]);
  const out: Range[] = [];
  for (const [s, e] of sorted) {
    const last = out.at(-1);
    if (last && s <= last[1]) last[1] = Math.max(last[1], e);
    else out.push([s, e]);
  }
  return out;
}

const round1 = (n: number) => Math.round(n * 10) / 10;

interface Props {
  duration: number;
  playhead: number;
  ranges: Range[];
  onChange: (r: Range[]) => void;
  onSeek: (t: number) => void;
}

export function RangeEditor({ duration, playhead, ranges, onChange, onSeek }: Props) {
  const [pending, setPending] = useState<number | null>(null);
  const barRef = useRef<HTMLDivElement>(null);
  const clamp = (t: number) => Math.min(duration, Math.max(0, Number.isFinite(t) ? t : 0));
  const pct = (t: number) => (duration > 0 ? (clamp(t) / duration) * 100 : 0);

  function setEdge(i: number, edge: 0 | 1, t: number) {
    onChange(ranges.map((r, k) => (k === i ? ((edge === 0 ? [clamp(t), r[1]] : [r[0], clamp(t)]) as Range) : r)));
  }

  function markOut() {
    if (pending === null) return;
    const a = round1(pending);
    const b = round1(clamp(playhead));
    onChange([...ranges, a <= b ? [a, b] : [b, a]]);
    setPending(null);
  }

  function drag(i: number, edge: 0 | 1, e: ReactPointerEvent<HTMLElement>) {
    const bar = barRef.current;
    if (!bar) return;
    e.preventDefault();
    e.currentTarget.setPointerCapture?.(e.pointerId);
    const move = (ev: PointerEvent) => {
      const box = bar.getBoundingClientRect();
      if (box.width <= 0) return;
      setEdge(i, edge, round1(((ev.clientX - box.left) / box.width) * duration));
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }

  const kept = mergeRanges(ranges).reduce((n, [s, e]) => n + Math.max(0, e - s), 0);

  return (
    <div className="ranges">
      <div className="ranges__bar" ref={barRef} aria-hidden="true">
        {ranges.map(([s, e], i) => (
          <span key={i} className="ranges__span" style={{ left: `${pct(s)}%`, width: `${Math.max(0, pct(e) - pct(s))}%` }}>
            <i className="ranges__edge ranges__edge--start" onPointerDown={(ev) => drag(i, 0, ev)} />
            <i className="ranges__edge ranges__edge--end" onPointerDown={(ev) => drag(i, 1, ev)} />
          </span>
        ))}
        <span className="ranges__head" style={{ left: `${pct(playhead)}%` }} />
      </div>
      <div className="ranges__actions">
        <button type="button" className="btn btn--small" onClick={() => setPending(round1(clamp(playhead)))}>
          Mark in
        </button>
        <button type="button" className="btn btn--small" disabled={pending === null} onClick={markOut}>
          Mark out
        </button>
        {pending !== null && <span className="muted">In at {formatDuration(pending)}</span>}
        <span className="ranges__total">
          Kept: {formatDuration(kept)} of {formatDuration(duration)}
        </span>
      </div>
      {ranges.length === 0 && <p className="muted">No ranges yet. Mark in and out while the video plays.</p>}
      <ul className="ranges__list">
        {ranges.map(([s, e], i) => (
          <li key={i} className="ranges__row">
            <input
              className="input num"
              type="number"
              min={0}
              max={duration}
              step={0.1}
              aria-label={`Start of range ${i + 1}`}
              value={s}
              onChange={(ev) => setEdge(i, 0, ev.target.valueAsNumber)}
            />
            <input
              className="input num"
              type="number"
              min={0}
              max={duration}
              step={0.1}
              aria-label={`End of range ${i + 1}`}
              value={e}
              onChange={(ev) => setEdge(i, 1, ev.target.valueAsNumber)}
            />
            <button type="button" className="btn btn--small" onClick={() => onSeek(s)}>
              Play
            </button>
            <button type="button" className="btn btn--small btn--danger" onClick={() => onChange(ranges.filter((_, k) => k !== i))}>
              Delete
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
