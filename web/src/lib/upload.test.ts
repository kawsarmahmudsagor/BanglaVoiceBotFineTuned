import { describe, expect, it } from "vitest";
import { describeUploadError, ingestOutcome, uploadIdFromUrl } from "./upload";

const withResponse = (status: number, body: string) =>
  Object.assign(new Error("tus: unexpected response"), {
    originalResponse: { getStatus: () => status, getBody: () => body },
  });

describe("uploadIdFromUrl", () => {
  it("takes the last path segment of an absolute URL", () => {
    expect(uploadIdFromUrl("http://localhost:8000/api/uploads/abc123")).toBe("abc123");
  });
  it("ignores a trailing slash, query and fragment", () => {
    expect(uploadIdFromUrl("/api/uploads/abc123/")).toBe("abc123");
    expect(uploadIdFromUrl("http://h/api/uploads/abc123?x=1#y")).toBe("abc123");
  });
  it("returns empty string for empty input", () => {
    expect(uploadIdFromUrl("")).toBe("");
  });
});

describe("describeUploadError", () => {
  it("maps 507 to a disk-full message", () => {
    expect(describeUploadError(withResponse(507, ""))).toMatch(/disk is full/i);
  });
  it("maps 409 and includes the body", () => {
    const m = describeUploadError(withResponse(409, "offset mismatch"));
    expect(m).toContain("409");
    expect(m).toContain("offset mismatch");
  });
  it("truncates long bodies to about 300 chars", () => {
    const m = describeUploadError(withResponse(500, "x".repeat(5000)));
    expect(m.length).toBeLessThan(400);
  });
  it("handles errors without a response", () => {
    expect(describeUploadError(new Error("network down"))).toBe("Upload failed: network down");
  });
});

describe("ingestOutcome", () => {
  const base = { id: 1, report: null, error: null, dataset: "d", created_at: null };
  const v = { id: 1, number: 2, hash: "h", n_samples: 1, commercial_ok: true, parent_number: null, created_at: null };
  it("classifies poll results", () => {
    expect(ingestOutcome({ ...base, state: "running", version: null })).toBe("continue");
    expect(ingestOutcome({ ...base, state: "pending", version: null })).toBe("continue");
    expect(ingestOutcome({ ...base, state: "done", version: v })).toBe("done");
    expect(ingestOutcome({ ...base, state: "failed", version: null })).toBe("failed");
  });
  it("treats done without a version as broken, not endless polling", () => {
    expect(ingestOutcome({ ...base, state: "done", version: null })).toBe("broken");
  });
});
