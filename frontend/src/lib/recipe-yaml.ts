import { parse, stringify } from "yaml";

export type YamlResult = { ok: true; spec: Record<string, unknown> } | { ok: false; error: string };

export function specToYaml(spec: Record<string, unknown>): string {
  return stringify(spec);
}

export function yamlToSpec(text: string): YamlResult {
  try {
    const doc: unknown = parse(text);
    if (doc === null || typeof doc !== "object" || Array.isArray(doc)) {
      return { ok: false, error: "YAML must be a mapping at the top level" };
    }
    return { ok: true, spec: doc as Record<string, unknown> };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
}
