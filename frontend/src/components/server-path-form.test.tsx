import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "@/lib/api";
import { UploadForm } from "./upload-form";
import { ServerPathForm } from "./server-path-form";

const api = vi.hoisted(() => ({
  importDirs: vi.fn(),
  ingestLocal: vi.fn(),
  getIngest: vi.fn(),
  startIngest: vi.fn(),
}));
vi.mock("@/lib/api", async (orig) => ({ ...(await orig<typeof import("@/lib/api")>()), ...api }));
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
}));

const DIRS = {
  items: [
    { path: "bn-tts", task_hint: "tts", has_curation: true },
    { path: "misc/raw", task_hint: null, has_curation: false },
  ],
};

function fillRequired() {
  fireEvent.change(screen.getByLabelText("Dataset name"), { target: { value: "my-set" } });
  fireEvent.change(screen.getByLabelText("Source"), { target: { value: "lab" } });
  fireEvent.change(screen.getByLabelText("License"), { target: { value: "cc-by" } });
  fireEvent.change(screen.getByLabelText("Consent"), { target: { value: "signed" } });
}

describe("ServerPathForm", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    api.importDirs.mockResolvedValue(DIRS);
  });

  it("lists the import dirs", async () => {
    render(<ServerPathForm />);
    expect(await screen.findByRole("option", { name: "bn-tts" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "misc/raw" })).toBeInTheDocument();
  });

  it("choosing a dir fills the path and preselects the task hint", async () => {
    render(<ServerPathForm defaultTask="asr" />);
    await screen.findByRole("option", { name: "bn-tts" });
    expect(screen.getByLabelText("Task")).toHaveValue("asr");
    fireEvent.change(screen.getByLabelText("Import directory"), { target: { value: "bn-tts" } });
    expect(screen.getByLabelText("Path")).toHaveValue("bn-tts");
    expect(screen.getByLabelText("Task")).toHaveValue("tts");
    fireEvent.change(screen.getByLabelText("Import directory"), { target: { value: "misc/raw" } });
    expect(screen.getByLabelText("Path")).toHaveValue("misc/raw");
    expect(screen.getByLabelText("Task")).toHaveValue("tts");
  });

  it("submits the relative path, task and provenance", async () => {
    api.ingestLocal.mockResolvedValue({ ingest_id: 7 });
    api.getIngest.mockResolvedValue({ id: 7, state: "running", report: null, version: null });
    render(<ServerPathForm />);
    await screen.findByRole("option", { name: "bn-tts" });
    fireEvent.change(screen.getByLabelText("Import directory"), { target: { value: "bn-tts" } });
    fillRequired();
    fireEvent.click(screen.getByLabelText("Eval only (never used for training)"));
    fireEvent.click(screen.getByRole("button", { name: "Ingest from server path" }));
    await waitFor(() => expect(api.ingestLocal).toHaveBeenCalled());
    expect(api.ingestLocal).toHaveBeenCalledWith({
      path: "bn-tts",
      name: "my-set",
      task: "tts",
      eval_only: true,
      provenance: { source: "lab", license: "cc-by", consent: "signed", commercial_ok: false, notes: "" },
    });
    await waitFor(() => expect(api.getIngest).toHaveBeenCalledWith(7));
  });

  it("shows the API's error message on 403", async () => {
    api.ingestLocal.mockRejectedValue(new ApiError(403, "path is not allowed"));
    render(<ServerPathForm defaultTask="asr" />);
    fireEvent.change(screen.getByLabelText("Path"), { target: { value: "../etc" } });
    fillRequired();
    fireEvent.click(screen.getByRole("button", { name: "Ingest from server path" }));
    expect(await screen.findByText(/path is not allowed/)).toBeInTheDocument();
  });
});

describe("UploadForm mode toggle", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    api.importDirs.mockResolvedValue(DIRS);
  });

  it("defaults to Upload file and switches to From server path", async () => {
    render(<UploadForm />);
    expect(screen.getByRole("button", { name: "Upload file" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByLabelText(/^File/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "From server path" }));
    expect(await screen.findByLabelText("Path")).toBeInTheDocument();
    expect(screen.queryByLabelText(/^File/)).toBeNull();
  });
});
