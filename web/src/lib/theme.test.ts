import { describe, expect, it } from "vitest";
import { readTheme, writeTheme, THEME_SCRIPT } from "./theme";

const store = (v: string | null) => ({ getItem: () => v });

describe("readTheme", () => {
  it("defaults to dark when the key is missing", () => {
    expect(readTheme(store(null))).toBe("dark");
  });
  it("defaults to dark for an invalid value", () => {
    expect(readTheme(store("blue"))).toBe("dark");
  });
  it("defaults to dark when storage throws", () => {
    expect(readTheme({ getItem: () => { throw new Error("blocked"); } })).toBe("dark");
  });
  it("returns light when stored", () => {
    expect(readTheme(store("light"))).toBe("light");
  });
});

describe("writeTheme", () => {
  it("swallows storage errors", () => {
    expect(() => writeTheme("light", { setItem: () => { throw new Error("blocked"); } })).not.toThrow();
  });
  it("stores the value under the theme key", () => {
    const saved: Record<string, string> = {};
    writeTheme("light", { setItem: (k: string, v: string) => { saved[k] = v; } });
    expect(saved).toEqual({ theme: "light" });
  });
});

describe("THEME_SCRIPT", () => {
  it("reads the theme key and toggles the dark class", () => {
    expect(THEME_SCRIPT).toContain("'theme'");
    expect(THEME_SCRIPT).toContain("dark");
  });
});
