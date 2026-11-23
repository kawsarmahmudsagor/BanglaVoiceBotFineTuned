import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import NewRecipePage from "./page";

const api = vi.hoisted(() => ({
  getRun: vi.fn(), getRecipe: vi.fn(), getRecipeVersion: vi.fn(), getCatalogEntry: vi.fn(), listBackends: vi.fn(),
}));
vi.mock("@/lib/api", () => api);
let query = "";
vi.mock("next/navigation", () => ({ useSearchParams: () => new URLSearchParams(query) }));
const seen = vi.hoisted(() => ({ initial: undefined as unknown }));
vi.mock("@/components/recipe-editor", () => ({
  RecipeEditor: ({ initial }: { initial?: unknown }) => {
    seen.initial = initial;
    return <div>editor</div>;
  },
}));

beforeEach(() => {
  Object.values(api).forEach((f) => f.mockReset());
  seen.initial = undefined;
});

describe("new training config prefill", () => {
  it("prefills from the run's recipe version and sets init_from", async () => {
    query = "from_run=r-1&checkpoint=checkpoint-300";
    api.getRun.mockResolvedValue({ id: "r-1", recipe: { name: "cfg", number: 3 } });
    api.getRecipeVersion.mockResolvedValue({ spec: { name: "cfg", task: "llm-sft", backend: "u", base_model: "m", init_from: { run: "old", checkpoint: "final" } } });
    render(<NewRecipePage />);
    await screen.findByText("editor");
    expect(api.getRecipeVersion).toHaveBeenCalledWith("cfg", 3);
    expect(seen.initial).toMatchObject({ name: "", base_model: "m", init_from: { run: "r-1", checkpoint: 300 } });
  });

  it("uses final for the final checkpoint", async () => {
    query = "from_run=r-1&checkpoint=final";
    api.getRun.mockResolvedValue({ id: "r-1", recipe: { name: "cfg", number: 3 } });
    api.getRecipeVersion.mockResolvedValue({ spec: { name: "cfg", base_model: "m" } });
    render(<NewRecipePage />);
    await screen.findByText("editor");
    expect(seen.initial).toMatchObject({ init_from: { run: "r-1", checkpoint: "final" } });
  });

  it("shows an error when the run cannot be loaded", async () => {
    query = "from_run=r-1&checkpoint=final";
    api.getRun.mockRejectedValue(new Error("gone"));
    render(<NewRecipePage />);
    await waitFor(() => expect(screen.getByText(/Cannot preload r-1 \(gone\)/)).toBeInTheDocument());
  });
});