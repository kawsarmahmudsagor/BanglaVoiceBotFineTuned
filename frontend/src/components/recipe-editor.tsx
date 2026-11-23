"use client";

import { useRouter } from "next/navigation";
import { useEffect, useId, useMemo, useRef, useState, type ReactNode } from "react";
import { Panel } from "@/components/matra";
import { ErrorNotice, WarningNotice } from "@/components/notice";
import { SchemaFields, type JsonSchema } from "@/components/schema-fields";
import { SplitFields, type SplitDataset } from "@/components/split-fields";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { getDataset, getVersion, listBackends, listDatasets, saveRecipe, validateRecipe } from "@/lib/api";
import { groupErrors, type RecipeField } from "@/lib/recipe-errors";
import { DEFAULT_SPLIT, ratiosValid } from "@/lib/splits";
import { specToYaml, yamlToSpec } from "@/lib/recipe-yaml";
import type { DatasetDetail, DatasetListItem, RecipeBody, SplitConfig, Task } from "@/lib/types";
import { cn } from "@/lib/utils";

type Spec = Record<string, unknown>;
interface TaskCap { methods: string[]; export_formats: string[]; schema: JsonSchema }
interface BackendCaps { name: string; label: string; tasks: Record<string, TaskCap> }

const TASKS: Task[] = ["llm-sft", "asr", "tts"];
const SELECT_CLASS = "h-9 w-full rounded-md border border-input bg-card px-2 text-sm";
const EMPTY_SPEC: Spec = {
  name: "",
  task: "llm-sft",
  backend: "",
  base_model: "",
  base_model_meta: { license: "", commercial_ok: false },
  datasets: { train: "", eval: null },
  method: { type: "" },
  hparams: {},
  export: [],
};

const asObj = (v: unknown): Spec => (v && typeof v === "object" && !Array.isArray(v) ? (v as Spec) : {});
const str = (v: unknown): string => (typeof v === "string" ? v : "");
const errMsg = (e: unknown) => (e instanceof Error ? e.message : String(e));

function FieldErrors({ id, errors }: { id: string; errors?: string[] }) {
  if (!errors?.length) return null;
  return (
    <ul id={id} className="space-y-0.5 text-[13px] text-status-failed">
      {errors.map((e, i) => <li key={i}>{e}</li>)}
    </ul>
  );
}

/** A labelled field with its own validation messages underneath. */
function Field({
  label,
  errors,
  children,
}: {
  label: string;
  errors?: string[];
  children: (props: { id: string; "aria-invalid"?: boolean; "aria-describedby"?: string }) => ReactNode;
}) {
  const id = useId();
  const errId = `${id}-err`;
  const bad = !!errors?.length;
  return (
    <div className="space-y-1">
      <label htmlFor={id} className="block text-[13px] text-muted-foreground">{label}</label>
      {children({ id, "aria-invalid": bad || undefined, "aria-describedby": bad ? errId : undefined })}
      <FieldErrors id={errId} errors={errors} />
    </div>
  );
}

/** Keeps a value the backend does not list yet (for example a catalog pre-fill) selectable. */
function withCurrent(options: string[], current: string): string[] {
  return current && !options.includes(current) ? [current, ...options] : options;
}

export function RecipeEditor({ initial, notice }: { initial?: Spec; notice?: string | null }) {
  const router = useRouter();
  const [spec, setSpec] = useState<Spec>(initial ?? EMPTY_SPEC);
  const [yamlText, setYamlText] = useState(() => specToYaml(initial ?? EMPTY_SPEC));
  const [yamlError, setYamlError] = useState<string | null>(null);
  const [backends, setBackends] = useState<BackendCaps[]>([]);
  const [datasets, setDatasets] = useState<DatasetListItem[]>([]);
  const [details, setDetails] = useState<Record<string, DatasetDetail>>({});
  const [errors, setErrors] = useState<string[]>([]);
  const [validating, setValidating] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const counter = useRef(0);
  // True while the latest spec change came from typing in the YAML panel.
  const fromYaml = useRef(false);

  useEffect(() => {
    let active = true;
    Promise.all([listBackends(), listDatasets()])
      .then(([b, d]) => {
        if (!active) return;
        setBackends(b as unknown as BackendCaps[]);
        setDatasets(d);
      })
      .catch((e) => active && setLoadError(errMsg(e)));
    return () => {
      active = false;
    };
  }, []);

  const task = str(spec.task) as Task;
  const taskDatasets = useMemo(() => datasets.filter((d) => d.task === task), [datasets, task]);

  useEffect(() => {
    let active = true;
    Promise.all(taskDatasets.map((d) => getDataset(d.name)))
      .then((all) => active && setDetails(Object.fromEntries(all.map((d) => [d.name, d]))))
      .catch((e) => active && setLoadError(errMsg(e)));
    return () => {
      active = false;
    };
  }, [taskDatasets]);

  const refs = (list: DatasetListItem[]) =>
    list.flatMap((d) =>
      (details[d.name]?.versions ?? [])
        .slice()
        .sort((a, b) => b.number - a.number)
        .map((v) => `${d.name}@v${v.number}`),
    );
  const trainRefs = refs(taskDatasets.filter((d) => !d.eval_only));
  const evalRefs = refs(taskDatasets);
  const firstTrain = trainRefs[0] ?? "";

  // Form edits rewrite the YAML panel; YAML edits leave the user's text alone.
  const updateSpec = (fn: (s: Spec) => Spec) => {
    fromYaml.current = false;
    setSpec(fn);
  };
  useEffect(() => {
    if (!fromYaml.current) {
      setYamlText(specToYaml(spec));
      setYamlError(null);
    }
  }, [spec]);

  const setField = (k: string, v: unknown) => updateSpec((s) => ({ ...s, [k]: v }));
  const datasetsObj = asObj(spec.datasets);
  const metaObj = asObj(spec.base_model_meta);
  const methodObj = asObj(spec.method);

  // default the train dataset to the latest version once versions are known
  useEffect(() => {
    if (firstTrain) {
      updateSpec((s) =>
        str(asObj(s.datasets).train) ? s : { ...s, datasets: { ...asObj(s.datasets), train: firstTrain } },
      );
    }
  }, [firstTrain, spec.task]);

  // The chosen train dataset's split info feeds the "Data splits" group.
  const trainRef = str(datasetsObj.train);
  const [splitDataset, setSplitDataset] = useState<SplitDataset | null>(null);
  const [splitLoadError, setSplitLoadError] = useState<string | null>(null);
  useEffect(() => {
    setSplitDataset(null);
    setSplitLoadError(null);
    const m = /^(.+)@v(\d+)$/.exec(trainRef);
    if (!m) {
      setSplitDataset(null);
      return;
    }
    let active = true;
    getVersion(m[1], Number(m[2]))
      .then((v) => active && setSplitDataset({ split_source: v.split_source, given_splits: v.given_splits, n_samples: v.n_samples }))
      .catch((e) => active && setSplitLoadError(errMsg(e)));
    return () => {
      active = false;
    };
  }, [trainRef]);
  const splitValue: SplitConfig = {
    ...DEFAULT_SPLIT,
    ...asObj(datasetsObj.split),
    ratios: { ...DEFAULT_SPLIT.ratios, ...asObj(asObj(datasetsObj.split).ratios) },
  } as SplitConfig;
  const splitError = ratiosValid(splitValue.ratios);

  const taskBackends = backends.filter((b) => b.tasks?.[task]);
  const cap = backends.find((b) => b.name === spec.backend)?.tasks?.[task];

  function changeTask(t: string) {
    updateSpec((s) => ({
      ...s, task: t, backend: "", datasets: { train: "", eval: null }, method: { type: "" }, hparams: {}, export: [],
    }));
  }
  function changeBackend(name: string) {
    const c = backends.find((b) => b.name === name)?.tasks?.[task];
    updateSpec((s) => {
      const keepMethod = c?.methods.includes(str(asObj(s.method).type));
      return {
        ...s,
        backend: name,
        method: keepMethod ? s.method : { type: c?.methods[0] ?? "" },
        export: (Array.isArray(s.export) ? (s.export as string[]) : []).filter((f) => c?.export_formats.includes(f)),
      };
    });
  }

  function onYamlChange(text: string) {
    setYamlText(text);
    const r = yamlToSpec(text);
    if (r.ok) {
      setYamlError(null);
      fromYaml.current = true;
      setSpec(r.spec);
    } else {
      setYamlError(r.error);
    }
  }

  // debounced server validation; out-of-order responses are discarded
  const key = JSON.stringify(spec);
  useEffect(() => {
    if (yamlError) {
      counter.current++;
      setValidating(false);
      return;
    }
    setValidating(true);
    const id = ++counter.current;
    const t = setTimeout(() => {
      validateRecipe({ spec: JSON.parse(key) })
        .then((r) => {
          if (id !== counter.current) return;
          setErrors(r.errors);
          setValidating(false);
        })
        .catch((e) => {
          if (id !== counter.current) return;
          setErrors([errMsg(e)]);
          setValidating(false);
        });
    }, 500);
    return () => clearTimeout(t);
  }, [key, yamlError]);

  async function save() {
    setSaveError(null);
    setSaving(true);
    try {
      const body: RecipeBody = { spec };
      const saved = await saveRecipe(body);
      router.push(`/recipes/${encodeURIComponent(saved.name)}`);
    } catch (e) {
      setSaveError(errMsg(e));
      setSaving(false);
    }
  }

  // An empty name is not a mistake yet; ask for it instead of flagging it.
  const unnamed = str(spec.name) === "";
  const shown = validating ? [] : unnamed ? errors.filter((e) => !e.startsWith("name")) : errors;
  const grouped = groupErrors(shown);
  const err = (f: RecipeField) => grouped[f];
  const canSave = !saving && !validating && errors.length === 0 && !yamlError && !splitError;
  const exportsSel = Array.isArray(spec.export) ? (spec.export as string[]) : [];
  const schemaProps = cap?.schema?.properties ?? {};
  const methods = withCurrent(cap?.methods ?? [], str(methodObj.type));
  // The backend's formats, plus any pre-filled format it does not offer (shown so it can be unticked).
  const capFormats = cap?.export_formats ?? [];
  const exportFormats = [...capFormats, ...exportsSel.filter((f) => !capFormats.includes(f))];

  return (
    <div className="space-y-4">
      {notice && <WarningNotice>{notice}</WarningNotice>}
      {loadError && <ErrorNotice>Cannot load backends or datasets ({loadError})</ErrorNotice>}

      <div className="grid items-start gap-4 lg:grid-cols-2">
        <Panel title="Form">
          <div className="grid gap-3">
            <Field label="Name" errors={err("name")}>
              {(p) => <Input {...p} value={str(spec.name)} onChange={(e) => setField("name", e.target.value)} placeholder="my-recipe" />}
            </Field>
            <div className="grid gap-3 sm:grid-cols-2">
              <Field label="Task">
                {(p) => (
                  <select {...p} className={SELECT_CLASS} value={task} onChange={(e) => changeTask(e.target.value)}>
                    {TASKS.map((t) => <option key={t} value={t}>{t}</option>)}
                  </select>
                )}
              </Field>
              <Field label="Backend" errors={err("backend")}>
                {(p) => (
                  <select {...p} className={SELECT_CLASS} value={str(spec.backend)} onChange={(e) => changeBackend(e.target.value)}>
                    <option value="">(select)</option>
                    {taskBackends.map((b) => <option key={b.name} value={b.name}>{b.label || b.name}</option>)}
                  </select>
                )}
              </Field>
            </div>
            <Field label="Method" errors={err("method")}>
              {(p) => (
                <select {...p} className={SELECT_CLASS} value={str(methodObj.type)} onChange={(e) => setField("method", { ...methodObj, type: e.target.value })}>
                  <option value="">(select)</option>
                  {methods.map((m) => <option key={m} value={m}>{m}</option>)}
                </select>
              )}
            </Field>
            <Field label="Base model" errors={err("base_model")}>
              {(p) => <Input {...p} className="font-mono" value={str(spec.base_model)} onChange={(e) => setField("base_model", e.target.value)} />}
            </Field>
            <div className="grid grid-cols-[1fr_auto] items-start gap-3">
              <Field label="Base model license" errors={err("license")}>
                {(p) => <Input {...p} className="font-mono" value={str(metaObj.license)} onChange={(e) => setField("base_model_meta", { ...metaObj, license: e.target.value })} />}
              </Field>
              <label className="flex h-9 items-center gap-2 self-end text-sm">
                <input
                  type="checkbox"
                  className="size-4 accent-[var(--saffron)]"
                  checked={metaObj.commercial_ok === true}
                  onChange={(e) => setField("base_model_meta", { ...metaObj, commercial_ok: e.target.checked })}
                />
                Commercial use OK
              </label>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <Field label="Train dataset" errors={err("train")}>
                {(p) => (
                  <select {...p} className={SELECT_CLASS} value={str(datasetsObj.train)} onChange={(e) => setField("datasets", { ...datasetsObj, train: e.target.value })}>
                    <option value="">(select)</option>
                    {withCurrent(trainRefs, str(datasetsObj.train)).map((r) => <option key={r} value={r}>{r}</option>)}
                  </select>
                )}
              </Field>
              <Field label="Eval dataset (optional)" errors={err("eval")}>
                {(p) => (
                  <select {...p} className={SELECT_CLASS} value={str(datasetsObj.eval)} onChange={(e) => setField("datasets", { ...datasetsObj, eval: e.target.value || null })}>
                    <option value="">none</option>
                    {withCurrent(evalRefs, str(datasetsObj.eval)).map((r) => <option key={r} value={r}>{r}</option>)}
                  </select>
                )}
              </Field>
            </div>
            <fieldset className="space-y-1">
              <legend className="text-[13px] text-muted-foreground">Export formats</legend>
              {exportFormats.length === 0 && <p className="text-sm text-muted-foreground">Pick a backend to see its export formats.</p>}
              {exportFormats.map((f) => (
                <label key={f} className="mr-4 inline-flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    className="size-4 accent-[var(--saffron)]"
                    checked={exportsSel.includes(f)}
                    onChange={(e) => setField("export", e.target.checked ? [...exportsSel, f] : exportsSel.filter((x) => x !== f))}
                  />
                  <span className="font-mono">{f}</span>
                </label>
              ))}
              <FieldErrors id="export-errors" errors={err("export")} />
            </fieldset>
            {schemaProps.method && (
              <SchemaFields
                schema={schemaProps.method}
                value={methodObj}
                exclude={["type"]}
                onChange={(m) => setField("method", { ...m, type: methodObj.type })}
              />
            )}
            <fieldset className="space-y-2 rounded-md border border-border p-3">
              <legend className="px-1 text-sm font-medium">Hyperparameters</legend>
              {schemaProps.hparams ? (
                <SchemaFields schema={schemaProps.hparams} value={asObj(spec.hparams)} onChange={(h) => setField("hparams", h)} />
              ) : (
                <p className="text-sm text-muted-foreground">Edit them in the YAML panel, or pick a backend for a form.</p>
              )}
              <FieldErrors id="hparams-errors" errors={err("hparams")} />
            </fieldset>
            <SplitFields
              value={splitValue}
              dataset={splitDataset}
              loadError={splitLoadError}
              onChange={(v) => setField("datasets", { ...datasetsObj, split: v })}
            />
          </div>
        </Panel>

        <Panel title="YAML">
          <textarea
            aria-label="Recipe YAML"
            aria-invalid={!!yamlError || undefined}
            className={cn(
              "h-[32rem] w-full resize-y rounded-md border border-input bg-ink p-3 font-mono text-[13px] leading-relaxed",
              yamlError && "border-status-failed",
            )}
            value={yamlText}
            onChange={(e) => onYamlChange(e.target.value)}
            spellCheck={false}
          />
          {yamlError && <ErrorNotice className="mt-2 font-mono text-xs">YAML parse error: {yamlError}</ErrorNotice>}
        </Panel>
      </div>

      <div aria-live="polite" className="space-y-2">
        {validating && <p className="text-sm text-muted-foreground">Validating…</p>}
        {!validating && grouped.general && (
          <ErrorNotice>
            <ul className="list-disc pl-4">{grouped.general.map((e, i) => <li key={i}>{e}</li>)}</ul>
          </ErrorNotice>
        )}
        {!validating && shown.length > 0 && !grouped.general && (
          <p className="text-sm text-status-failed">Fix the {shown.length === 1 ? "problem" : `${shown.length} problems`} marked above to save.</p>
        )}
        {!validating && unnamed && <p className="text-sm text-muted-foreground">Name the recipe to save it.</p>}
        {saveError && <ErrorNotice>{saveError}</ErrorNotice>}
      </div>
      <Button size="lg" className="px-4" onClick={() => void save()} disabled={!canSave}>Save recipe</Button>
    </div>
  );
}
