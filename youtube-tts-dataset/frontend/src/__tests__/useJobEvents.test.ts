import { act, renderHook } from "@testing-library/react";
import { useJobEvents } from "../useJobEvents";
import { STAGES, type DatasetDetail, type JobEvent } from "../types";

class MockWebSocket {
  static instances: MockWebSocket[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  closed = false;

  constructor(readonly url: string) {
    MockWebSocket.instances.push(this);
  }
  close() {
    this.closed = true;
  }
  // test helpers
  open() {
    this.onopen?.();
  }
  send(event: JobEvent) {
    this.onmessage?.({ data: JSON.stringify(event) });
  }
  drop() {
    this.onclose?.();
  }
}

const snapshot: DatasetDetail = {
  id: "ds1",
  name: "news",
  speaker: "",
  created: 0,
  videos: [
    {
      dataset_id: "ds1",
      id: "v1",
      url: "https://youtu.be/v1",
      title: "one",
      channel: null,
      upload_date: null,
      duration_s: 60,
      license: null,
      status: "running",
      error: null,
      warning: null,
      stages: Object.fromEntries(STAGES.map((s) => [s, false])) as DatasetDetail["videos"][0]["stages"],
      kept: 0,
      dropped: 0,
    },
  ],
};

const fetchMock = vi.fn(async () => ({ ok: true, status: 200, text: async () => JSON.stringify(snapshot) }));

beforeEach(() => {
  MockWebSocket.instances = [];
  fetchMock.mockClear();
  vi.stubGlobal("WebSocket", MockWebSocket);
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

const flush = () => act(async () => {});

test("useJobEvents refetches state after reconnect", async () => {
  vi.useFakeTimers();
  const { result } = renderHook(() => useJobEvents("ds1"));
  expect(MockWebSocket.instances).toHaveLength(1);

  act(() => MockWebSocket.instances[0].open());
  await flush();
  expect(fetchMock).toHaveBeenCalledTimes(1);
  expect(result.current.connected).toBe(true);
  expect(result.current.videos).toHaveLength(1);

  act(() => MockWebSocket.instances[0].drop());
  expect(result.current.connected).toBe(false);
  await act(async () => {
    await vi.advanceTimersByTimeAsync(1000);
  });
  expect(MockWebSocket.instances).toHaveLength(2);

  act(() => MockWebSocket.instances[1].open());
  await flush();
  expect(fetchMock).toHaveBeenCalledTimes(2);
  expect(fetchMock).toHaveBeenLastCalledWith("/api/datasets/ds1", expect.anything());
  expect(result.current.connected).toBe(true);
});

test("useJobEvents applies stage_done", async () => {
  const { result } = renderHook(() => useJobEvents("ds1"));
  act(() => MockWebSocket.instances[0].open());
  await flush();
  expect(result.current.videos[0].stages.download).toBe(false);

  act(() =>
    MockWebSocket.instances[0].send({ type: "stage_done", dataset_id: "ds1", video_id: "v1", stage: "download" }),
  );
  expect(result.current.videos[0].stages.download).toBe(true);
  expect(result.current.log.some((line) => line.includes("download"))).toBe(true);
});

test("useJobEvents tracks export progress, completion and failure", async () => {
  const { result } = renderHook(() => useJobEvents("ds1"));
  act(() => MockWebSocket.instances[0].open());
  await flush();
  const ws = MockWebSocket.instances[0];
  expect(result.current.exportState.status).toBe("idle");

  act(() => ws.send({ type: "progress", dataset_id: "ds1", stage: "export", done: 3, total: 10 }));
  expect(result.current.exportState).toMatchObject({ status: "running", done: 3, total: 10 });

  const summary = { out_dir: "C:/x", clips: 10, hours: 0.1, process_error: 0, empty_text_excluded: 0,
    validation_dropped: 1, unreviewed_translated_skipped: 0 };
  act(() => ws.send({ type: "stage_done", dataset_id: "ds1", stage: "export", summary }));
  expect(result.current.exportState).toMatchObject({ status: "done", done: 10, summary });

  act(() => ws.send({ type: "stage_failed", dataset_id: "ds1", stage: "export", error: "disk full" }));
  expect(result.current.exportState).toMatchObject({ status: "failed", error: "disk full" });
  // video events leave the export alone
  act(() => ws.send({ type: "stage_done", dataset_id: "ds1", video_id: "v1", stage: "download" }));
  expect(result.current.exportState.status).toBe("failed");
});
