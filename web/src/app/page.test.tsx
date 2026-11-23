import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import Dashboard from "./page";

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

beforeEach(() => vi.useFakeTimers());
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("Dashboard", () => {
  it("renders system cards and polls every 5 s until unmounted", async () => {
    const fn = route({ "/api/system": () => SYSTEM, "/api/runs": () => [], "/api/models": () => [] });
    const { unmount } = render(<Dashboard />);
    await act(async () => {});
    expect(screen.getByText("GPU not visible")).toBeInTheDocument();
    expect(screen.getByTestId("disk-card").className).toContain("amber");
    const systemCalls = () => fn.mock.calls.filter(([u]) => String(u).endsWith("/api/system")).length;
    expect(systemCalls()).toBe(1);
    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
    expect(systemCalls()).toBe(2);
    unmount();
    await act(async () => { await vi.advanceTimersByTimeAsync(15000); });
    expect(systemCalls()).toBe(2);
  });

  it("shows a banner instead of crashing when the API is unreachable", async () => {
    route({});
    render(<Dashboard />);
    await act(async () => {});
    expect(screen.getByRole("alert")).toHaveTextContent("Cannot reach the API");
    expect(screen.getByText("Dashboard")).toBeInTheDocument();
  });
});
