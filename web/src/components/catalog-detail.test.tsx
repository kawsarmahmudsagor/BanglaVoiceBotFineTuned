import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { CatalogEntry } from "@/lib/types";
import { CatalogDetail } from "./catalog-detail";

vi.mock("next/link", () => ({
  default: ({ href, children, ...rest }: { href: string; children: React.ReactNode }) => (
    <a href={href} {...rest}>{children}</a>
  ),
}));

function entryFixture(over: Partial<CatalogEntry> = {}): CatalogEntry {
  return {
    id: "whisper-x",
    task: "asr",
    name: "Whisper X",
    hf_repo: "org/whisper-x",
    derived_from: "openai/whisper",
    params: "809M",
    role: "primary",
    license: { id: "mit", commercial_ok: true, url: "https://example.com/l", notes: null },
    bangla: { status: "finetuned", evidence: "WER 11%", url: null },
    runtime: { engine: "faster_whisper", convert: "ct2", streaming: false },
    description: "A **Bangla** model.",
    dataset: {
      format: "asr",
      required_fields: ["file", "text"],
      optional_fields: ["speaker"],
      example: "clips/1.wav,আমি",
      recommended: { hours: "10-100" },
      notes: null,
    },
    finetune: {
      methods: ["full", "lora"],
      frameworks: ["transformers"],
      targets: ["Train the decoder"],
      hyperparams: { learning_rate: 1e-5 },
      metrics: ["wer"],
      pitfalls: ["Hallucination on silence"],
    },
    recipe_template: { backend: null, method: { type: "full" }, hparams: { learning_rate: 1e-5 }, export: ["ct2"] },
    sources: [{ title: "Card", url: "https://example.com" }],
    verified: "2026-11-23",
    ...over,
  };
}

describe("CatalogDetail", () => {
  it("renders the four sections in order", () => {
    render(<CatalogDetail entry={entryFixture()} />);
    const headings = screen.getAllByRole("heading", { level: 2 }).map((h) => h.textContent);
    expect(headings).toEqual(["What it is", "Training data", "Fine-tuning", "Sources"]);
  });

  it("links Fine-tune to a pre-filled recipe and the data format to upload", () => {
    render(<CatalogDetail entry={entryFixture()} />);
    expect(screen.getByRole("link", { name: "Fine-tune" })).toHaveAttribute("href", "/recipes/new?catalog=whisper-x");
    expect(screen.getByRole("link", { name: /Upload asr dataset/ })).toHaveAttribute("href", "/datasets/new?task=asr");
  });

  it("shows the example row in a pre block", () => {
    const { container } = render(<CatalogDetail entry={entryFixture()} />);
    expect(container.querySelector("pre")?.textContent).toBe("clips/1.wav,আমি");
  });

  it("renders raw HTML in markdown as text, never as elements", () => {
    const evil = "<script>alert(1)</script><img src=x onerror=alert(1)>";
    const { container } = render(<CatalogDetail entry={entryFixture({ description: evil })} />);
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("img")).toBeNull();
    expect(container.textContent).toContain("<script>alert(1)</script>");
  });
});
