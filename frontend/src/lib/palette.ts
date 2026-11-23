import type { Theme } from "./theme";

/** Magma-style series ramp, ordered so neighbouring series differ in lightness. */
// Dark theme: magenta lifted from #B5367A to #D14A92 so it keeps 3:1 against the panels.
export const SERIES_DARK = ["#D14A92", "#F4B740", "#7A5CC7", "#F07D52", "#E7E9F0"];
// Light theme: saffron and orange darkened so lines keep at least 3:1 against the light background.
export const SERIES_LIGHT = ["#B5367A", "#9A6100", "#7A5CC7", "#C9542A", "#1A1F2C"];

export function seriesColor(i: number, theme: Theme): string {
  const ramp = theme === "light" ? SERIES_LIGHT : SERIES_DARK;
  return ramp[((i % ramp.length) + ramp.length) % ramp.length];
}
