import { act, fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { UploadDrawerProvider, useUploadDrawer } from "./upload-drawer";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
}));

function DatasetsPage() {
  const { openUpload } = useUploadDrawer();
  return <button onClick={() => openUpload("asr")}>Upload dataset</button>;
}
function RunsPage() {
  return <p>Runs</p>;
}

describe("UploadDrawerProvider", () => {
  it("keeps the upload form mounted when the page that opened it goes away", async () => {
    const { rerender } = render(
      <UploadDrawerProvider>
        <DatasetsPage />
      </UploadDrawerProvider>,
    );
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Upload dataset" }));
    });
    const task = screen.getByLabelText("Task");
    expect(task).toHaveValue("asr");

    // Navigating replaces the page under the shell; the provider (in the shell) stays.
    rerender(
      <UploadDrawerProvider>
        <RunsPage />
      </UploadDrawerProvider>,
    );
    expect(screen.getByText("Runs")).toBeInTheDocument();
    expect(task.isConnected).toBe(true);
  });
});
