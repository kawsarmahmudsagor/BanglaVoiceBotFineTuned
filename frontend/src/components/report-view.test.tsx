import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ReportView } from "./report-view";

vi.mock("next/link", () => ({
  default: ({ href, children, ...rest }: { href: string; children: React.ReactNode }) => (
    <a href={href} {...rest}>{children}</a>
  ),
}));

const REPORT = {
  errors: [
    { code: "bad_json", message: "line 1: bad", sample_id: null },
    { code: "bad_json", message: "line 2: bad", sample_id: null },
  ],
  warnings: [
    { code: "too_short", message: "short", sample_id: "abc" },
    { code: "no_tools", message: "none", sample_id: "def" },
    { code: "no_tools", message: "none", sample_id: "ghi" },
  ],
  stats: { n_samples: 10 },
};

describe("ReportView", () => {
  it("shows counts and groups issues by code", () => {
    render(<ReportView report={REPORT} />);
    expect(screen.getByText("2 errors")).toBeInTheDocument();
    expect(screen.getByText("3 warnings")).toBeInTheDocument();
    expect(screen.getByText("no_tools")).toBeInTheDocument();
    expect(screen.getByText("bad_json")).toBeInTheDocument();
    expect(screen.getByText("n_samples")).toBeInTheDocument();
  });

  it("uses singular and links sample ids to the filtered samples", () => {
    render(
      <ReportView
        report={{ errors: [], warnings: [REPORT.warnings[0]] }}
        dataset="ds"
        version={3}
      />,
    );
    expect(screen.getByText("0 errors")).toBeInTheDocument();
    expect(screen.getByText("1 warning")).toBeInTheDocument();
    const link = screen.getByRole("link", { name: "abc" });
    expect(link).toHaveAttribute("href", "/datasets/ds/v/3?warning=too_short");
  });

  it("renders nothing useful-but-safe for a null report", () => {
    render(<ReportView report={null} />);
    expect(screen.getByText("No report")).toBeInTheDocument();
  });
});
