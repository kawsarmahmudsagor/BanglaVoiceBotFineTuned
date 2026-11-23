/** Append a sample, keeping the newest `max`. Returns a new array. */
export function pushSample(hist: number[], v: number, max = 60): number[] {
  const next = [...hist, v];
  return next.length > max ? next.slice(next.length - max) : next;
}
