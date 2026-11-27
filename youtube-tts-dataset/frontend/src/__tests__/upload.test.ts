import { vi } from "vitest";
import { UploadError, uploadStudioFile } from "../api";

class FakeXhr {
  static last: FakeXhr;
  upload: { onprogress: ((e: { loaded: number; total: number }) => void) | null } = { onprogress: null };
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onabort: (() => void) | null = null;
  status = 0;
  responseText = "";
  method = "";
  url = "";
  sent: unknown = null;
  constructor() {
    FakeXhr.last = this;
  }
  open(m: string, u: string) {
    this.method = m;
    this.url = u;
  }
  send(b: unknown) {
    this.sent = b;
  }
  finish(status: number, body: unknown) {
    this.status = status;
    this.responseText = JSON.stringify(body);
    this.onload?.();
  }
}

beforeEach(() => vi.stubGlobal("XMLHttpRequest", FakeXhr));
afterEach(() => vi.unstubAllGlobals());

test("uploadStudioFile sends the raw file, reports progress and resolves the video id", async () => {
  const file = new File(["abc"], "a b.mp4");
  const onProgress = vi.fn();
  const p = uploadStudioFile("ds1", file, onProgress);
  const x = FakeXhr.last;
  expect(x.method).toBe("POST");
  expect(x.url).toBe("/api/datasets/ds1/studio/upload?filename=a%20b.mp4");
  expect(x.sent).toBe(file);
  x.upload.onprogress?.({ loaded: 1, total: 3 });
  expect(onProgress).toHaveBeenCalledWith(1, 3);
  x.finish(200, { video_id: "local_x" });
  await expect(p).resolves.toEqual({ video_id: "local_x" });
});

test("uploadStudioFile rejects a duplicate with status 409 and the existing video id", async () => {
  const p = uploadStudioFile("ds1", new File(["abc"], "a.mp4"), () => {});
  FakeXhr.last.finish(409, { detail: { error: "this file is already in the dataset", video_id: "local_x" } });
  const err = await p.catch((e: unknown) => e);
  expect(err).toBeInstanceOf(UploadError);
  expect(err).toMatchObject({ status: 409, video_id: "local_x", message: "this file is already in the dataset" });
});
