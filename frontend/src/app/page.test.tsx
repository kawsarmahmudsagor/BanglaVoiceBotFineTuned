import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import Overview from "./page";

vi.mock("next/link", () => ({
  default: ({ href, children, ...rest }: { href: string; children: React.ReactNode }) => (
    <a href={href} {...rest}>{children}</a>
  ),
}));

const SYSTEM = {
  gpu: null,
  lock: { holder_kind: null, holder_id: null, acquired_at: null },
  disk: { free_gb: 12.3, warn: true, block: false },
  queue: { queued: 2, running: 1 },
};

const CHAMPION = {
  id: 7, name: "bn-whisper", number: 3, run_id: "r-1", task: "asr", base_model: "b", metrics: { wer: 0.12 },
  exports: [], commercial_ok: true, champion: true, created_at: null,
};

function route(handlers: Record<string, () => unknown>) {
  const fn = vi.fn(async (url: string) => {
    const path = new URL(url).pathname;
    const h = handlers[path];
    if (!h) throw new TypeError("Failed to fetch");
    return { ok: true, status: 200, json: async () => h() };
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

const BASE = {
  "/api/system": () => SYSTEM,
  "/api/runs": () => [],
  "/api/models": () => [],
  "/api/datasets": () => [],
};

beforeEach(() => vi.useFakeTimers());
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("Overview", () => {
  it("renders the Now panel and system state, polling every 5 s until unmounted", async () => {
    const fn = route(BASE);
    const { unmount } = render(<Overview />);
    await act(async () => {});
    expect(screen.getByRole("heading", { name: "Now" })).toBeInTheDocument();
    expect(screen.getByText("GPU not visible")).toBeInTheDocument();
    expect(screen.getByTestId("disk-card").dataset.level).toBe("warn");
    const systemCalls = () => fn.mock.calls.filter(([u]) => String(u).endsWith("/api/system")).length;
    expect(systemCalls()).toBe(1);
    const runCalls = fn.mock.calls.map(([u]) => String(u)).filter((u) => u.includes("/api/runs"));
    expect(runCalls).toHaveLength(1);
    expect(new URL(runCalls[0]).searchParams.get("limit")).toBe("10");
    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
    expect(systemCalls()).toBe(2);
    unmount();
    await act(async () => { await vi.advanceTimersByTimeAsync(15000); });
    expect(systemCalls()).toBe(2);
  });

  it("points to the catalog when there are no runs", async () => {
    route(BASE);
    const { container } = render(<Overview />);
    await act(async () => {});
    expect(container.textContent).toContain("No runs yet. Pick a model in Catalog to start a recipe.");
    expect(screen.getByRole("link", { name: "Catalog" })).toHaveAttribute("href", "/catalog");
  });

  it("links champions to the registry", async () => {
    route({ ...BASE, "/api/models": () => [CHAMPION] });
    render(<Overview />);
    await act(async () => {});
    expect(screen.getByRole("link", { name: /bn-whisper/ })).toHaveAttribute("href", "/registry/7");
  });

  it("shows a banner instead of crashing when the API is unreachable", async () => {
    route({});
    render(<Overview />);
    await act(async () => {});
    expect(screen.getByRole("alert")).toHaveTextContent("Cannot reach the API");
    expect(screen.getByRole("heading", { name: "Overview" })).toBeInTheDocument();
  });
});
