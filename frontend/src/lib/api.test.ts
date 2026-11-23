import { afterEach, describe, expect, it, vi } from "vitest";
import {
  API_URL, ApiError, apiGet, apiPatch, apiPost, audioUrl, exportUrl, getSamples, listModels,
  listRuns, runEventsUrl, runExportFileUrl,
} from "./api";

function mockFetch(status: number, body: unknown) {
  const fn = vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

const asError = (e: unknown) => e as ApiError;

afterEach(() => vi.unstubAllGlobals());

describe("ApiError", () => {
  it("joins the errors array of a 422 by newline", async () => {
    mockFetch(422, { errors: ["a", "b"] });
    const err = await apiPost<never>("/api/recipes", {}).catch(asError);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(422);
    expect(err.detail).toBe("a\nb");
    expect(err.message).toBe("a\nb");
  });
  it("uses a string detail", async () => {
    mockFetch(404, { detail: "run 'x' not found" });
    const err = await apiGet<never>("/api/runs/x").catch(asError);
    expect(err.status).toBe(404);
    expect(err.detail).toBe("run 'x' not found");
  });
  it("formats a FastAPI validation detail list", async () => {
    mockFetch(422, { detail: [{ loc: ["body", "name"], msg: "bad slug", type: "x" }] });
    const err = await apiPost<never>("/api/datasets/ingest", {}).catch(asError);
    expect(err.detail).toBe("body.name: bad slug");
  });
  it("survives a non-JSON error body", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: false, status: 502, statusText: "Bad Gateway", json: async () => { throw new Error("no json"); },
    }));
    const err = await apiGet<never>("/api/system").catch(asError);
    expect(err.status).toBe(502);
    expect(err.detail).toBe("Bad Gateway");
  });
});

describe("requests", () => {
  it("GETs against API_URL", async () => {
    const fn = mockFetch(200, { ok: 1 });
    await expect(apiGet("/api/system")).resolves.toEqual({ ok: 1 });
    expect(fn).toHaveBeenCalledWith(`${API_URL}/api/system`, expect.anything());
  });
  it("POSTs and PATCHes JSON bodies", async () => {
    const fn = mockFetch(200, {});
    await apiPost("/api/x", { a: 1 });
    await apiPatch("/api/y", { b: 2 });
    const [, post] = fn.mock.calls[0];
    const [, patch] = fn.mock.calls[1];
    expect(post.method).toBe("POST");
    expect(post.body).toBe('{"a":1}');
    expect(post.headers["Content-Type"]).toBe("application/json");
    expect(patch.method).toBe("PATCH");
    expect(patch.body).toBe('{"b":2}');
  });
  it("POST without body sends none", async () => {
    const fn = mockFetch(200, {});
    await apiPost("/api/runs/r-1/cancel");
    expect(fn.mock.calls[0][1].body).toBeUndefined();
  });
  it("builds query strings and skips undefined", async () => {
    const fn = mockFetch(200, []);
    await listRuns("failed");
    await listModels();
    await getSamples("ds", 2, { offset: 10, split: "val" });
    expect(fn.mock.calls[0][0]).toBe(`${API_URL}/api/runs?state=failed`);
    expect(fn.mock.calls[1][0]).toBe(`${API_URL}/api/models`);
    expect(fn.mock.calls[2][0]).toBe(`${API_URL}/api/datasets/ds/versions/2/samples?offset=10&split=val`);
    await listRuns(undefined, 10);
    expect(fn.mock.calls[3][0]).toBe(`${API_URL}/api/runs?limit=10`);
  });
});

describe("url builders", () => {
  it("builds event, audio and export urls", () => {
    expect(runEventsUrl("r-1")).toBe(`${API_URL}/api/runs/r-1/events`);
    expect(audioUrl("my-set", 3, "s 1")).toBe(`${API_URL}/api/datasets/my-set/versions/3/audio/s%201`);
    expect(exportUrl(7, 0)).toBe(`${API_URL}/api/models/7/exports/0`);
  });
  it("prefixes run export manifest paths with export/ and encodes segments", () => {
    expect(runExportFileUrl("r-1", "gguf/model q4.gguf")).toBe(
      `${API_URL}/api/runs/r-1/files/export/gguf/model%20q4.gguf`,
    );
  });
});
