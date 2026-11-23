import type { Theme } from "./theme";

/** Magma-style series ramp, ordered so neighbouring series differ in lightness. */
export const SERIES_DARK = ["#B5367A", "#F4B740", "#7A5CC7", "#F07D52", "#E7E9F0"];
export const SERIES_LIGHT = ["#B5367A", "#F4B740", "#7A5CC7", "#F07D52", "#1A1F2C"];

export function seriesColor(i: number, theme: Theme): string {
  const ramp = theme === "light" ? SERIES_LIGHT : SERIES_DARK;
  return ramp[((i % ramp.length) + ramp.length) % ramp.length];
}
