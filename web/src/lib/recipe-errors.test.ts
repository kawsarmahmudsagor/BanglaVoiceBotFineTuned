import { describe, expect, it } from "vitest";
import { groupErrors } from "./recipe-errors";

describe("groupErrors", () => {
  it("routes validation messages to the field they are about", () => {
    const g = groupErrors([
      "name: name must be a kebab-case slug",
      "unknown backend 'x'",
      "task 'asr' not supported by backend 'fake'",
      "method 'qlora' not supported",
      "export format 'gguf' not supported",
      "dataset 'a@v1' not found",
      "dataset 'a' is eval-only and cannot be used for training",
      "hparams.lr: 'x' is not of type 'number'",
      "base_model: Field required",
      "base_model_meta.license: Field required",
      "something else",
    ]);
    expect(g.name).toEqual(["name: name must be a kebab-case slug"]);
    expect(g.backend).toEqual(["unknown backend 'x'", "task 'asr' not supported by backend 'fake'"]);
    expect(g.method).toEqual(["method 'qlora' not supported"]);
    expect(g.export).toEqual(["export format 'gguf' not supported"]);
    expect(g.train).toEqual(["dataset 'a@v1' not found", "dataset 'a' is eval-only and cannot be used for training"]);
    expect(g.hparams).toEqual(["hparams.lr: 'x' is not of type 'number'"]);
    expect(g.base_model).toEqual(["base_model: Field required"]);
    expect(g.license).toEqual(["base_model_meta.license: Field required"]);
    expect(g.general).toEqual(["something else"]);
  });

  it("sends eval dataset errors to the eval field", () => {
    expect(groupErrors(["datasets.eval: invalid dataset ref 'x'"]).eval).toEqual(["datasets.eval: invalid dataset ref 'x'"]);
  });
});
