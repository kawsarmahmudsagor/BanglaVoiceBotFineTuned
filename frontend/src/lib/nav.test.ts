import { describe, expect, it } from "vitest";
import { NAV_GROUPS, breadcrumbs, isActive } from "./nav";

describe("NAV_GROUPS", () => {
  it("lists groups and pages in order", () => {
    expect(NAV_GROUPS.map((g) => g.label)).toEqual([null, "Prepare", "Training", "Ship"]);
    expect(NAV_GROUPS[2].items.map((i) => i.href)).toEqual(["/train", "/runs", "/runs/compare"]);
    expect(NAV_GROUPS.flatMap((g) => g.items.map((i) => i.href))).toEqual([
      "/", "/catalog", "/datasets", "/train", "/runs", "/runs/compare", "/registry",
    ]);
  });
});

describe("Train nav item", () => {
  it("is /train labelled Train, with no /recipes", () => {
    const items = NAV_GROUPS.flatMap((g) => g.items);
    expect(items).toContainEqual(expect.objectContaining({ href: "/train", label: "Train" }));
    expect(items.some((i) => i.href.startsWith("/recipes"))).toBe(false);
  });
});

describe("isActive", () => {
  it("matches root only on root", () => {
    expect(isActive("/", "/")).toBe(true);
    expect(isActive("/runs", "/")).toBe(false);
  });
  it("matches a section and its children", () => {
    expect(isActive("/runs/r-1", "/runs")).toBe(true);
    expect(isActive("/catalog/whisper-x", "/catalog")).toBe(true);
    expect(isActive("/runsx", "/runs")).toBe(false);
  });
  it("keeps Runs and Compare apart", () => {
    expect(isActive("/runs/compare", "/runs")).toBe(false);
    expect(isActive("/runs/compare", "/runs/compare")).toBe(true);
    expect(isActive("/runs/r-1", "/runs/compare")).toBe(false);
  });
});

describe("breadcrumbs", () => {
  it("labels known segments and shows dynamic ones verbatim", () => {
    expect(breadcrumbs("/catalog/whisper-x")).toEqual([
      { href: "/catalog", label: "Catalog" },
      { href: "/catalog/whisper-x", label: "whisper-x" },
    ]);
  });
  it("is Overview on the root", () => {
    expect(breadcrumbs("/")).toEqual([{ href: "/", label: "Overview" }]);
  });
  it("folds the v segment of a dataset version into one crumb, so no crumb points at a missing page", () => {
    expect(breadcrumbs("/datasets/bn-chat/v/3")).toEqual([
      { href: "/datasets", label: "Datasets" },
      { href: "/datasets/bn-chat", label: "bn-chat" },
      { href: "/datasets/bn-chat/v/3", label: "v3" },
    ]);
  });
  it("labels nested known segments", () => {
    expect(breadcrumbs("/runs/compare")).toEqual([
      { href: "/runs", label: "Runs" },
      { href: "/runs/compare", label: "Compare" },
    ]);
  });
});
