import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { IngestingNote } from "./ingest-shared";
import type { IngestStatus } from "@/lib/types";

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }) }));

const base: IngestStatus = {
  id: 1, state: "running", report: null, version: null, error: null, dataset: "ds", created_at: null, progress: null,
};

describe("IngestingNote", () => {
  it("shows validation progress when the ingest reports it", () => {
    render(<IngestingNote phase="ingesting" ingest={{ ...base, progress: { done: 1200, total: 50000 } }} />);
    expect(screen.getByText(/Validating 1,?200 of 50,?000 files/)).toBeInTheDocument();
  });

  it("falls back to the state when there is no progress yet", () => {
    render(<IngestingNote phase="ingesting" ingest={base} />);
    expect(screen.getByText("Validating and ingesting (running)...")).toBeInTheDocument();
    expect(screen.queryByText(/of .* files/)).not.toBeInTheDocument();
  });

  it("renders nothing when idle", () => {
    const { container } = render(<IngestingNote phase="idle" ingest={base} />);
    expect(container).toBeEmptyDOMElement();
  });
});
