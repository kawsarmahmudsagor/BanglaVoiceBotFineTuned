"use client";

import { useId } from "react";
import { Input } from "@/components/ui/input";

export type JsonSchema = {
  type?: string | string[];
  enum?: unknown[];
  properties?: Record<string, JsonSchema>;
  minimum?: number;
  maximum?: number;
  description?: string;
  default?: unknown;
};
type Obj = Record<string, unknown>;

export function normalizeType(s: JsonSchema): { type: string; nullable: boolean } {
  const types = Array.isArray(s.type) ? s.type : s.type ? [s.type] : [];
  const nullable = types.includes("null") || (s.enum?.includes(null) ?? false);
  const main = types.find((t) => t !== "null");
  if (main) return { type: main, nullable };
  if (s.properties) return { type: "object", nullable };
  if (s.enum) return { type: "string", nullable };
  return { type: "unknown", nullable };
}

const SELECT_CLASS =
  "h-8 w-full rounded-lg border border-input bg-transparent px-2 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 dark:bg-input/30";

function setKey(obj: Obj, key: string, v: unknown): Obj {
  const next = { ...obj };
  if (v === undefined) delete next[key];
  else next[key] = v;
  return next;
}

function Field({
  id, name, schema, value, onChange,
}: { id: string; name: string; schema: JsonSchema; value: unknown; onChange: (v: unknown) => void }) {
  const { type, nullable } = normalizeType(schema);
  const help = schema.description ? <p className="text-xs text-muted-foreground">{schema.description}</p> : null;
  const placeholder = schema.default !== undefined ? String(schema.default) : undefined;

  if (type === "object" && schema.properties) {
    return (
      <fieldset className="space-y-2 rounded-md border p-3">
        <legend className="px-1 text-sm font-medium">{name}</legend>
        {help}
        <SchemaFields
          schema={schema}
          value={(value && typeof value === "object" ? value : {}) as Obj}
          onChange={(v) => onChange(v)}
          idPrefix={id}
        />
      </fieldset>
    );
  }

  let control: React.ReactNode;
  if (schema.enum) {
    const options = schema.enum.filter((o) => o !== null);
    const current = value === null || value === undefined ? "" : String(value);
    control = (
      <select
        id={id}
        className={SELECT_CLASS}
        value={current}
        onChange={(e) => {
          if (e.target.value === "") return onChange(nullable ? null : undefined);
          onChange(options.find((o) => String(o) === e.target.value));
        }}
      >
        <option value="">{nullable ? "none" : "(unset)"}</option>
        {options.map((o) => (
          <option key={String(o)} value={String(o)}>{String(o)}</option>
        ))}
      </select>
    );
  } else if (type === "integer" || type === "number") {
    control = (
      <Input
        id={id}
        type="number"
        min={schema.minimum}
        max={schema.maximum}
        step={type === "integer" ? 1 : "any"}
        placeholder={placeholder}
        value={typeof value === "number" ? value : ""}
        onChange={(e) => {
          const raw = e.target.value;
          onChange(raw === "" || Number.isNaN(Number(raw)) ? undefined : Number(raw));
        }}
      />
    );
  } else if (type === "string") {
    control = (
      <Input
        id={id}
        type="text"
        placeholder={placeholder}
        value={typeof value === "string" ? value : ""}
        onChange={(e) => onChange(e.target.value === "" ? undefined : e.target.value)}
      />
    );
  } else if (type === "boolean") {
    return (
      <div className="space-y-1">
        <label htmlFor={id} className="flex items-center gap-2 text-sm">
          <input id={id} type="checkbox" role="switch" checked={value === true} onChange={(e) => onChange(e.target.checked)} />
          {name}
        </label>
        {help}
      </div>
    );
  } else {
    return (
      <div className="space-y-1">
        <span className="text-sm">{name}</span>
        <pre className="overflow-auto rounded-md bg-muted p-2 font-mono text-xs">{JSON.stringify(value ?? null)}</pre>
        {help}
      </div>
    );
  }
  return (
    <div className="space-y-1">
      <label htmlFor={id} className="text-sm">{name}</label>
      {control}
      {help}
    </div>
  );
}

export function SchemaFields({
  schema, value, onChange, exclude = [], idPrefix,
}: {
  schema: JsonSchema;
  value: Obj;
  onChange: (v: Obj) => void;
  exclude?: string[];
  idPrefix?: string;
}) {
  const generated = useId();
  const prefix = idPrefix ?? generated;
  const props = Object.entries(schema.properties ?? {}).filter(([k]) => !exclude.includes(k));
  return (
    <div className="space-y-3">
      {props.map(([key, sub]) => (
        <Field
          key={key}
          id={`${prefix}-${key}`}
          name={key}
          schema={sub}
          value={value[key]}
          onChange={(v) => onChange(setKey(value, key, v))}
        />
      ))}
    </div>
  );
}
