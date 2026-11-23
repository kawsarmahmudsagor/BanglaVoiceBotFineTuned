import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { RUN_STATES } from "@/lib/types";
import { StatusBadge } from "./status-badge";

describe("StatusBadge", () => {
  it.each(RUN_STATES)("renders an icon and the label for %s", (state) => {
    const { container } = render(<StatusBadge state={state} />);
    expect(screen.getByText(state)).toBeInTheDocument();
    expect(container.querySelector("svg")).not.toBeNull();
  });
});
