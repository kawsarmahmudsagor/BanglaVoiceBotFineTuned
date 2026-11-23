"use client";

import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { SchemaFields, type JsonSchema } from "@/components/schema-fields";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { getDataset, listBackends, listDatasets, saveRecipe, validateRecipe } from "@/lib/api";
import { specToYaml, yamlToSpec } from "@/lib/recipe-yaml";
import type { DatasetDetail, DatasetListItem, RecipeBody, Task } from "@/lib/types";

type Spec = Record<string, unknown>;
interface TaskCap { methods: string[]; export_formats: string[]; schema: JsonSchema }
interface BackendCaps { name: string; label: string; tasks: Record<string, TaskCap> }

const TASKS: Task[] = ["llm-sft", "asr", "tts"];
const SELECT_CLASS =
  "h-8 w-full rounded-lg border border-input bg-transparent px-2 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 dark:bg-input/30";
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

function Labeled({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block space-y-1 text-sm">
      <span>{label}</span>
      {children}
    </label>
  );
}

export function RecipeEditor({ initial }: { initial?: Spec }) {
  const router = useRouter();
  const [spec, setSpec] = useState<Spec>(initial ?? EMPTY_SPEC);
  const [tab, setTab] = useState<"form" | "yaml">("form");
  const [yamlText, setYamlText] = useState("");
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

  const setField = (k: string, v: unknown) => setSpec((s) => ({ ...s, [k]: v }));
  const datasetsObj = asObj(spec.datasets);
  const metaObj = asObj(spec.base_model_meta);
  const methodObj = asObj(spec.method);

  // default the train dataset to the latest version once versions are known
  useEffect(() => {
    if (firstTrain) {
      setSpec((s) =>
        str(asObj(s.datasets).train) ? s : { ...s, datasets: { ...asObj(s.datasets), train: firstTrain } },
      );
    }
  }, [firstTrain, spec.task]);

  const taskBackends = backends.filter((b) => b.tasks?.[task]);
  const cap = backends.find((b) => b.name === spec.backend)?.tasks?.[task];

  function changeTask(t: string) {
    setSpec((s) => ({
      ...s, task: t, backend: "", datasets: { train: "", eval: null }, method: { type: "" }, hparams: {}, export: [],
    }));
  }
  function changeBackend(name: string) {
    const c = backends.find((b) => b.name === name)?.tasks?.[task];
    setSpec((s) => ({
      ...s, backend: name, method: { type: c?.methods[0] ?? "" }, hparams: {}, export: c?.export_formats.slice(0, 1) ?? [],
    }));
  }

  // debounced server validation; out-of-order responses are discarded
  const key = tab === "yaml" ? `y:${yamlText}` : `s:${JSON.stringify(spec)}`;
  useEffect(() => {
    if (key.startsWith("y:") && yamlError) {
      counter.current++;
      setValidating(false);
      setErrors([]);
      return;
    }
    setValidating(true);
    const id = ++counter.current;
    const body: RecipeBody = key.startsWith("y:") ? { yaml: key.slice(2) } : { spec: JSON.parse(key.slice(2)) };
    const t = setTimeout(() => {
      validateRecipe(body)
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

  function switchTab(next: "form" | "yaml") {
    if (next === "yaml" && tab !== "yaml") {
      setYamlText(specToYaml(spec));
      setYamlError(null);
    }
    setTab(next);
  }
  function onYamlChange(text: string) {
    setYamlText(text);
    const r = yamlToSpec(text);
    if (r.ok) {
      setYamlError(null);
      setSpec(r.spec);
    } else {
      setYamlError(r.error);
    }
  }

  async function save() {
    setSaveError(null);
    setSaving(true);
    try {
      const body: RecipeBody = tab === "yaml" ? { yaml: yamlText } : { spec };
      const saved = await saveRecipe(body);
      router.push(`/recipes/${encodeURIComponent(saved.name)}`);
    } catch (e) {
      setSaveError(errMsg(e));
      setSaving(false);
    }
  }

  const canSave = !saving && !validating && errors.length === 0 && !(tab === "yaml" && yamlError);
  const exportsSel = Array.isArray(spec.export) ? (spec.export as string[]) : [];
  const schemaProps = cap?.schema?.properties ?? {};

  return (
    <div className="space-y-4">
      {loadError && (
        <div role="alert" className="rounded-md border border-red-500/50 bg-red-500/10 px-3 py-2 text-sm">
          Cannot load backends or datasets ({loadError})
        </div>
      )}
      <div role="tablist" className="flex gap-2">
        {(["form", "yaml"] as const).map((t) => (
          <Button key={t} role="tab" aria-selected={tab === t} variant={tab === t ? "default" : "outline"} onClick={() => switchTab(t)}>
            {t === "form" ? "Form" : "YAML"}
          </Button>
        ))}
      </div>

      {tab === "yaml" ? (
        <div className="space-y-2">
          <textarea
            aria-label="Recipe YAML"
            className="h-96 w-full rounded-lg border border-input bg-transparent p-2 font-mono text-sm"
            value={yamlText}
            onChange={(e) => onYamlChange(e.target.value)}
            spellCheck={false}
          />
          {yamlError && (
            <div role="alert" className="rounded-md border border-red-500/50 bg-red-500/10 px-3 py-2 font-mono text-xs">
              YAML parse error: {yamlError}
            </div>
          )}
        </div>
      ) : (
        <div className="grid max-w-2xl gap-3">
          <Labeled label="Name">
            <Input value={str(spec.name)} onChange={(e) => setField("name", e.target.value)} placeholder="my-recipe" />
          </Labeled>
          <Labeled label="Task">
            <select className={SELECT_CLASS} value={task} onChange={(e) => changeTask(e.target.value)}>
              {TASKS.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </Labeled>
          <Labeled label="Backend">
            <select className={SELECT_CLASS} value={str(spec.backend)} onChange={(e) => changeBackend(e.target.value)}>
              <option value="">(select)</option>
              {taskBackends.map((b) => <option key={b.name} value={b.name}>{b.label || b.name}</option>)}
            </select>
          </Labeled>
          <Labeled label="Method">
            <select
              className={SELECT_CLASS}
              value={str(methodObj.type)}
              onChange={(e) => setField("method", { ...methodObj, type: e.target.value })}
            >
              <option value="">(select)</option>
              {(cap?.methods ?? []).map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
          </Labeled>
          <Labeled label="Base model">
            <Input value={str(spec.base_model)} onChange={(e) => setField("base_model", e.target.value)} />
          </Labeled>
          <div className="grid grid-cols-[1fr_auto] items-end gap-3">
            <Labeled label="Base model license">
              <Input value={str(metaObj.license)} onChange={(e) => setField("base_model_meta", { ...metaObj, license: e.target.value })} />
            </Labeled>
            <label className="flex items-center gap-2 pb-1.5 text-sm">
              <input
                type="checkbox"
                checked={metaObj.commercial_ok === true}
                onChange={(e) => setField("base_model_meta", { ...metaObj, commercial_ok: e.target.checked })}
              />
              Commercial use OK
            </label>
          </div>
          <Labeled label="Train dataset">
            <select
              className={SELECT_CLASS}
              value={str(datasetsObj.train)}
              onChange={(e) => setField("datasets", { ...datasetsObj, train: e.target.value })}
            >
              <option value="">(select)</option>
              {str(datasetsObj.train) && !trainRefs.includes(str(datasetsObj.train)) && (
                <option value={str(datasetsObj.train)}>{str(datasetsObj.train)}</option>
              )}
              {trainRefs.map((r) => <option key={r} value={r}>{r}</option>)}
            </select>
          </Labeled>
          <Labeled label="Eval dataset (optional)">
            <select
              className={SELECT_CLASS}
              value={str(datasetsObj.eval)}
              onChange={(e) => setField("datasets", { ...datasetsObj, eval: e.target.value || null })}
            >
              <option value="">none</option>
              {str(datasetsObj.eval) && !evalRefs.includes(str(datasetsObj.eval)) && (
                <option value={str(datasetsObj.eval)}>{str(datasetsObj.eval)}</option>
              )}
              {evalRefs.map((r) => <option key={r} value={r}>{r}</option>)}
            </select>
          </Labeled>
          <fieldset className="space-y-1">
            <legend className="text-sm">Export formats</legend>
            {(cap?.export_formats ?? []).map((f) => (
              <label key={f} className="mr-4 inline-flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={exportsSel.includes(f)}
                  onChange={(e) => setField("export", e.target.checked ? [...exportsSel, f] : exportsSel.filter((x) => x !== f))}
                />
                {f}
              </label>
            ))}
          </fieldset>
          {schemaProps.method && (
            <SchemaFields
              schema={schemaProps.method}
              value={methodObj}
              exclude={["type"]}
              onChange={(m) => setField("method", { ...m, type: methodObj.type })}
            />
          )}
          {schemaProps.hparams && (
            <fieldset className="space-y-2 rounded-md border p-3">
              <legend className="px-1 text-sm font-medium">Hyperparameters</legend>
              <SchemaFields schema={schemaProps.hparams} value={asObj(spec.hparams)} onChange={(h) => setField("hparams", h)} />
            </fieldset>
          )}
        </div>
      )}

      <div aria-live="polite" className="space-y-1">
        {validating && <p className="text-sm text-muted-foreground">Validating...</p>}
        {!validating && errors.length > 0 && (
          <ul role="alert" className="list-disc rounded-md border border-red-500/50 bg-red-500/10 py-2 pl-7 pr-3 text-sm">
            {errors.map((e, i) => <li key={i}>{e}</li>)}
          </ul>
        )}
        {saveError && (
          <div role="alert" className="rounded-md border border-red-500/50 bg-red-500/10 px-3 py-2 text-sm">{saveError}</div>
        )}
      </div>
      <Button onClick={() => void save()} disabled={!canSave}>Save</Button>
    </div>
  );
}
