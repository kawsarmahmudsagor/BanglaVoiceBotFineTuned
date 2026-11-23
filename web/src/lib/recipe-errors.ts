export type RecipeField =
  | "name" | "backend" | "method" | "base_model" | "license" | "train" | "eval" | "export" | "hparams" | "general";

const RULES: [RegExp, RecipeField][] = [
  [/^name\b/, "name"],
  [/^unknown backend|not supported by backend|^backend\b/, "backend"],
  [/^method\b/, "method"],
  [/^export\b|^export format/, "export"],
  [/^datasets\.eval\b/, "eval"],
  [/^datasets\b|^dataset '/, "train"],
  [/^hparams\b/, "hparams"],
  [/^base_model_meta\b/, "license"],
  [/^base_model\b/, "base_model"],
];

/** Route server validation messages to the form field they describe, so they show inline. */
export function groupErrors(errors: string[]): Partial<Record<RecipeField, string[]>> {
  const out: Partial<Record<RecipeField, string[]>> = {};
  for (const e of errors) {
    const field = RULES.find(([re]) => re.test(e))?.[1] ?? "general";
    (out[field] ??= []).push(e);
  }
  return out;
}
