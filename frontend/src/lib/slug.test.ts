import { describe, expect, it } from "vitest";
import { isValidSlug } from "./slug";

describe("isValidSlug", () => {
  it("accepts lowercase hyphenated slugs", () => {
    expect(isValidSlug("bn-chat-1")).toBe(true);
    expect(isValidSlug("a")).toBe(true);
  });
  it("rejects uppercase, spaces, edge or double hyphens and empty", () => {
    for (const bad of ["", "Bn", "a b", "-a", "a-", "a--b", "a_b", "বাংলা"]) {
      expect(isValidSlug(bad)).toBe(false);
    }
  });
});
