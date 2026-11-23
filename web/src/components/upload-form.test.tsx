import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { UploadForm } from "./upload-form";

const query = { value: new URLSearchParams() };
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
  useSearchParams: () => query.value,
}));

describe("UploadForm", () => {
  it("preselects the task given as defaultTask", () => {
    query.value = new URLSearchParams();
    render(<UploadForm defaultTask="asr" />);
    expect(screen.getByLabelText("Task")).toHaveValue("asr");
  });

  it("falls back to the ?task= query, then llm-sft", () => {
    query.value = new URLSearchParams("task=tts");
    const { unmount } = render(<UploadForm />);
    expect(screen.getByLabelText("Task")).toHaveValue("tts");
    unmount();
    query.value = new URLSearchParams();
    render(<UploadForm />);
    expect(screen.getByLabelText("Task")).toHaveValue("llm-sft");
  });
});
