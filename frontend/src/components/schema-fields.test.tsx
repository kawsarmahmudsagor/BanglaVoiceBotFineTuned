import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SchemaFields, normalizeType } from "./schema-fields";

describe("normalizeType", () => {
  it("handles plain, array and nullable types", () => {
    expect(normalizeType({ type: "integer" })).toEqual({ type: "integer", nullable: false });
    expect(normalizeType({ type: ["string", "null"] })).toEqual({ type: "string", nullable: true });
    expect(normalizeType({ enum: ["a", null] })).toEqual({ type: "string", nullable: true });
    expect(normalizeType({})).toEqual({ type: "unknown", nullable: false });
  });
});

describe("SchemaFields", () => {
  it("renders a number input with minimum and emits the updated object", () => {
    const onChange = vi.fn();
    const schema = { type: "object", properties: { steps: { type: "integer", minimum: 1, description: "Number of steps" } } };
    render(<SchemaFields schema={schema} value={{}} onChange={onChange} />);
    const input = screen.getByLabelText("steps") as HTMLInputElement;
    expect(input.type).toBe("number");
    expect(input.min).toBe("1");
    expect(screen.getByText("Number of steps")).toBeInTheDocument();
    fireEvent.change(input, { target: { value: "5" } });
    expect(onChange).toHaveBeenCalledWith({ steps: 5 });
  });

  it("removes the key when a number input is emptied", () => {
    const onChange = vi.fn();
    const schema = { type: "object", properties: { steps: { type: "integer" }, seed: { type: "integer" } } };
    render(<SchemaFields schema={schema} value={{ steps: 3, seed: 1 }} onChange={onChange} />);
    fireEvent.change(screen.getByLabelText("steps"), { target: { value: "" } });
    expect(onChange).toHaveBeenCalledWith({ seed: 1 });
  });

  it("renders a select for an enum", () => {
    const onChange = vi.fn();
    const schema = { type: "object", properties: { mode: { type: "string", enum: ["a", "b"] } } };
    render(<SchemaFields schema={schema} value={{}} onChange={onChange} />);
    const select = screen.getByLabelText("mode") as HTMLSelectElement;
    expect(select.tagName).toBe("SELECT");
    fireEvent.change(select, { target: { value: "b" } });
    expect(onChange).toHaveBeenCalledWith({ mode: "b" });
  });

  it("maps the none option of a nullable enum to null", () => {
    const onChange = vi.fn();
    const schema = {
      type: "object",
      properties: { fail_phase: { type: ["string", "null"], enum: ["train", "eval", null] } },
    };
    render(<SchemaFields schema={schema} value={{ fail_phase: "train" }} onChange={onChange} />);
    const select = screen.getByLabelText("fail_phase") as HTMLSelectElement;
    expect(Array.from(select.options).map((o) => o.text)).toEqual(["none", "train", "eval"]);
    fireEvent.change(select, { target: { value: "" } });
    expect(onChange).toHaveBeenCalledWith({ fail_phase: null });
  });

  it("renders a switch for booleans and a fieldset for nested objects", () => {
    const onChange = vi.fn();
    const schema = {
      type: "object",
      properties: {
        oom: { type: "boolean" },
        lora: { type: "object", properties: { r: { type: "integer" } } },
      },
    };
    render(<SchemaFields schema={schema} value={{ lora: { r: 2 } }} onChange={onChange} />);
    fireEvent.click(screen.getByLabelText("oom"));
    expect(onChange).toHaveBeenLastCalledWith({ lora: { r: 2 }, oom: true });
    fireEvent.change(screen.getByLabelText("r"), { target: { value: "8" } });
    expect(onChange).toHaveBeenLastCalledWith({ lora: { r: 8 } });
    expect(screen.getByText("lora").tagName).toBe("LEGEND");
  });

  it("shows a read-only JSON preview for unknown types and skips excluded keys", () => {
    const schema = { type: "object", properties: { type: { type: "string" }, weird: { type: "array", items: {} } } };
    render(<SchemaFields schema={schema} value={{ weird: [1, 2] }} onChange={() => {}} exclude={["type"]} />);
    expect(screen.queryByLabelText("type")).toBeNull();
    expect(screen.getByText("[1,2]")).toBeInTheDocument();
  });
});
