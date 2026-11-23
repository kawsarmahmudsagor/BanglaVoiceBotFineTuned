"use client";

import { useEffect, useState } from "react";
import type { Theme } from "./theme";

/** The theme currently applied to <html>, kept in sync when the toggle changes it. */
export function useThemeValue(): Theme {
  const [theme, setTheme] = useState<Theme>("dark");
  useEffect(() => {
    const root = document.documentElement;
    const read = () => setTheme(root.classList.contains("dark") ? "dark" : "light");
    read();
    const obs = new MutationObserver(read);
    obs.observe(root, { attributes: true, attributeFilter: ["class"] });
    return () => obs.disconnect();
  }, []);
  return theme;
}

/** Chart chrome colours (axes, grid, tooltip) per theme; SVG attributes cannot read CSS variables. */
export const CHART_CHROME: Record<Theme, { grid: string; axis: string; tooltipBg: string; tooltipBorder: string; text: string }> = {
  dark: { grid: "#353C52", axis: "#9BA2B6", tooltipBg: "#272D3E", tooltipBorder: "#353C52", text: "#E7E9F0" },
  light: { grid: "#D5D9E3", axis: "#5A6275", tooltipBg: "#FFFFFF", tooltipBorder: "#D5D9E3", text: "#1A1F2C" },
};
