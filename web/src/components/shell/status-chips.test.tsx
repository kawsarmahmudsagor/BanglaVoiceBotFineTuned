import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { SystemInfo } from "@/lib/types";
import { StatusChips } from "./status-chips";

vi.mock("next/link", () => ({
  default: ({ href, children, ...rest }: { href: string; children: React.ReactNode }) => (
    <a href={href} {...rest}>{children}</a>
  ),
}));

const BASE: SystemInfo = {
  gpu: { name: "RTX", vram_used_mb: 8192, vram_total_mb: 32768, util_pct: 41, temp_c: 60 },
  lock: { holder_kind: null, holder_id: null, acquired_at: null },
  disk: { free_gb: 100, warn: false, block: false },
  queue: { queued: 0, running: 0 },
};

describe("StatusChips", () => {
  it("shows API unreachable with the error as title", () => {
    render(<StatusChips system={null} error="Failed to fetch" history={[]} />);
    const chip = screen.getByText("API unreachable");
    expect(chip.closest("[title]")?.getAttribute("title")).toBe("Failed to fetch");
  });
  it("shows No GPU when gpu is null", () => {
    render(<StatusChips system={{ ...BASE, gpu: null }} error={null} history={[]} />);
    expect(screen.getByText("No GPU")).toBeInTheDocument();
  });
  it("shows utilisation, VRAM and a free lock", () => {
    render(<StatusChips system={BASE} error={null} history={[41]} />);
    expect(screen.getByText("41%")).toBeInTheDocument();
    expect(screen.getByText("8.0/32.0 GB")).toBeInTheDocument();
    expect(screen.getByText("GPU free")).toBeInTheDocument();
  });
  it("links the lock to the holding run", () => {
    render(
      <StatusChips system={{ ...BASE, lock: { holder_kind: "run", holder_id: "r-1", acquired_at: null } }} error={null} history={[]} />,
    );
    expect(screen.getByRole("link", { name: /r-1/ })).toHaveAttribute("href", "/runs/r-1");
  });
});
