import { useEffect, useState, type FormEvent } from "react";
import { getSettings, putSettings, testLlm, updateDataset } from "../api";
import type { AppSettings, DatasetSummary, ModelSlotInfo, StageModels, Thresholds } from "../types";

type Draft = Record<keyof Thresholds, string>;

const FIELDS: { key: keyof Thresholds; label: string; unit?: string; step: string; hint: string }[] = [
  { key: "min_bn_ratio", label: "Minimum Bangla script share", step: "0.05",
    hint: "Transcripts with less Bangla script than this are broken, so the clip is dropped." },
  { key: "min_chars_per_s", label: "Slowest speaking rate", unit: "chars/s", step: "0.5",
    hint: "Fewer characters per second than this means words were missed, so the clip is dropped." },
  { key: "max_chars_per_s", label: "Fastest speaking rate", unit: "chars/s", step: "0.5",
    hint: "More than this means the model made text up, so the clip is dropped." },
  { key: "max_word_repeat", label: "Repeated word limit", step: "1",
    hint: "One word repeated this many times in a row is treated as a transcription loop." },
  { key: "dnsmos_min", label: "Minimum DNSMOS", step: "0.1", hint: "Overall quality score, 1 to 5." },
  { key: "speaker_sim_min", label: "Minimum speaker similarity", step: "0.01",
    hint: "Clips less like the main speaker than this are dropped." },
  { key: "clip_min_s", label: "Shortest clip", unit: "s", step: "0.1", hint: "" },
  { key: "clip_max_s", label: "Longest clip", unit: "s", step: "0.1", hint: "" },
  { key: "piece_max_s", label: "Longest piece", unit: "s", step: "0.5", hint: "Audio is transcribed in pieces up to this long." },
  { key: "piece_gap_s", label: "Shortest pause that splits pieces", unit: "s", step: "0.05", hint: "" },
  { key: "tail_min_s", label: "Shortest leftover clip", unit: "s", step: "0.5", hint: "Must not be longer than the shortest clip." },
  { key: "disagree_cer", label: "Model disagreement", step: "0.05",
    hint: "Models A and B differing by more than this flag the clip for review." },
  { key: "tie_margin", label: "Tie margin", step: "0.01", hint: "Score gap under this counts as a tie; model B wins." },
  { key: "demucs_segment_s", label: "Demucs segment", unit: "s", step: "0.5",
    hint: "Shorter segments use less GPU memory." },
];

const MODEL_LABEL: Record<keyof StageModels, string> = {
  separate: "Vocal separation",
  vad: "Speech detection",
  dnsmos: "Audio quality",
  speaker: "Speaker check",
  model_a: "Transcription (model A)",
  model_b: "Transcription (model B)",
  llm: "Punctuation",
};

const toDraft = (t: Thresholds) =>
  Object.fromEntries(Object.entries(t).map(([k, v]) => [k, String(v)])) as Draft;

/** Field errors for the threshold draft; empty when it can be saved. */
export function validateThresholds(d: Draft): Partial<Record<keyof Thresholds, string>> {
  const err: Partial<Record<keyof Thresholds, string>> = {};
  const n = (k: keyof Thresholds) => Number(d[k]);
  for (const { key } of FIELDS) {
    if (d[key].trim() === "" || !Number.isFinite(n(key)) || n(key) < 0) err[key] = "Enter a number of 0 or more.";
  }
  if (!err.min_chars_per_s && !err.max_chars_per_s && n("min_chars_per_s") >= n("max_chars_per_s"))
    err.max_chars_per_s = "Must be faster than the slowest rate.";
  if (!err.clip_min_s && !err.clip_max_s && n("clip_min_s") >= n("clip_max_s"))
    err.clip_max_s = "Must be longer than the shortest clip.";
  if (!err.tail_min_s && !err.clip_min_s && n("tail_min_s") > n("clip_min_s"))
    err.tail_min_s = "Must not be longer than the shortest clip.";
  if (!err.demucs_segment_s && n("demucs_segment_s") <= 0) err.demucs_segment_s = "Must be more than 0.";
  return err;
}

function message(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function SpeakerForm({ dataset, onSaved }: { dataset: DatasetSummary; onSaved: () => void }) {
  const [speaker, setSpeaker] = useState(dataset.speaker);
  const [status, setStatus] = useState<{ text: string; error: boolean } | null>(null);
  const [saving, setSaving] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setSaving(true);
    setStatus(null);
    try {
      await updateDataset(dataset.id, speaker);
      setStatus({ text: "Speaker saved.", error: false });
      onSaved();
    } catch (err) {
      setStatus({ text: message(err), error: true });
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="panel" aria-labelledby="speaker-title">
      <h2 id="speaker-title">
        Speaker for <span lang="bn">{dataset.name}</span>
      </h2>
      <form className="settings-inline" onSubmit={submit}>
        <label className="field">
          Name in metadata.csv
          <input className="input" value={speaker} onChange={(e) => setSpeaker(e.target.value)} placeholder={dataset.name}
            autoComplete="off" aria-describedby="speaker-hint" />
        </label>
        <button type="submit" className="btn" disabled={saving || speaker.trim() === dataset.speaker}>
          Save speaker
        </button>
      </form>
      <p id="speaker-hint" className="field__hint settings-hint">
        Leave empty to use the dataset name.
      </p>
      {status && (
        <p className={`notice${status.error ? " notice--error" : ""}`} role={status.error ? "alert" : "status"}>
          {status.text}
        </p>
      )}
    </section>
  );
}

function ModelSelect({ slot, label, info, onSaved, onError }: {
  slot: "model_a" | "model_b";
  label: string;
  info: ModelSlotInfo;
  onSaved: (s: AppSettings) => void;
  onError: (text: string) => void;
}) {
  const [saving, setSaving] = useState(false);
  async function change(value: string) {
    setSaving(true);
    try {
      onSaved(await putSettings({ [slot]: value }));
    } catch (err) {
      onError(message(err));
    } finally {
      setSaving(false);
    }
  }
  return (
    <label className="field">
      {label}
      <select className="input" value={info.current} disabled={saving} onChange={(e) => void change(e.target.value)}>
        {info.options.map((o) => (
          <option key={o.id} value={o.id} disabled={!o.available}>
            {o.available ? o.label : `${o.label} (missing: ${o.expected})`}
          </option>
        ))}
      </select>
    </label>
  );
}

interface Props {
  dataset: DatasetSummary | null;
  onDatasetChanged: () => void;
}

export function Settings({ dataset, onDatasetChanged }: Props) {
  const [loaded, setLoaded] = useState<AppSettings | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [llmHost, setLlmHost] = useState("");
  const [llmModel, setLlmModel] = useState("");
  const [cookies, setCookies] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState<{ text: string; error: boolean } | null>(null);
  const [cookiesError, setCookiesError] = useState<string | null>(null);
  const [showErrors, setShowErrors] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ ok: boolean; text: string } | null>(null);

  function apply(s: AppSettings) {
    setLoaded(s);
    setDraft(toDraft(s.thresholds));
    setLlmHost(s.llm_host);
    setLlmModel(s.llm_model);
    setCookies(s.cookies_path);
  }

  const load = () => {
    setLoadError(null);
    getSettings().then(apply, (e: unknown) => setLoadError(message(e)));
  };
  useEffect(load, []);

  const errors = draft ? validateThresholds(draft) : {};
  const hasErrors = Object.keys(errors).length > 0;

  async function save(e: FormEvent) {
    e.preventDefault();
    if (!draft) return;
    setSaveStatus(null);
    setCookiesError(null);
    if (hasErrors) {
      setShowErrors(true);
      return;
    }
    setSaving(true);
    try {
      const thresholds = Object.fromEntries(Object.entries(draft).map(([k, v]) => [k, Number(v)])) as unknown as Thresholds;
      apply(await putSettings({ thresholds, cookies_path: cookies.trim(), llm_host: llmHost, llm_model: llmModel }));
      setShowErrors(false);
      setSaveStatus({ text: "Settings saved. They apply to clips processed from now on.", error: false });
    } catch (err) {
      const text = message(err);
      if (text.startsWith("cookies file")) setCookiesError(text);
      else setSaveStatus({ text, error: true });
    } finally {
      setSaving(false);
    }
  }

  async function test() {
    setTesting(true);
    setTestResult(null);
    try {
      // the test uses the saved host and model, so save edited ones first
      if (loaded && (llmHost.trim() !== loaded.llm_host || llmModel.trim() !== loaded.llm_model)) {
        const s = await putSettings({ llm_host: llmHost, llm_model: llmModel });
        setLoaded(s);
        setLlmHost(s.llm_host);
        setLlmModel(s.llm_model);
      }
      const r = await testLlm();
      setTestResult(r.ok ? { ok: true, text: "Connected. The model answered." } : { ok: false, text: r.error ?? "Not reachable." });
    } catch (err) {
      setTestResult({ ok: false, text: message(err) });
    } finally {
      setTesting(false);
    }
  }

  if (loadError) {
    return (
      <div className="banner" role="alert">
        <p>Could not load settings: {loadError}</p>
        <button type="button" className="btn btn--small" onClick={load}>
          Try again
        </button>
      </div>
    );
  }
  if (!loaded || !draft) return <div className="panel empty">Loading settings…</div>;

  return (
    <div className="settings">
      <form className="settings__main" onSubmit={save} noValidate>
        <section className="panel" aria-labelledby="thr-title">
          <h2 id="thr-title">Thresholds</h2>
          <div className="thr-grid">
            {FIELDS.map((f) => {
              const err = showErrors ? errors[f.key] : undefined;
              const hintId = `thr-${f.key}-hint`;
              const errId = `thr-${f.key}-error`;
              return (
                <label key={f.key} className="field">
                  <span>
                    {f.label}
                    {f.unit && <span className="field__hint"> ({f.unit})</span>}
                  </span>
                  <input className="input num" type="number" inputMode="decimal" min="0" step={f.step}
                    value={draft[f.key]} onChange={(e) => setDraft({ ...draft, [f.key]: e.target.value })}
                    aria-invalid={err ? true : undefined}
                    aria-describedby={[f.hint && hintId, err && errId].filter(Boolean).join(" ") || undefined} />
                  {f.hint && (
                    <span id={hintId} className="field__hint">
                      {f.hint}
                    </span>
                  )}
                  {err && (
                    <span id={errId} className="field__error">
                      {err}
                    </span>
                  )}
                </label>
              );
            })}
          </div>
        </section>

        <section className="panel" aria-labelledby="tm-title">
          <h2 id="tm-title">Transcription models</h2>
          <div className="settings-pair">
            <ModelSelect slot="model_a" label="Model A" info={loaded.transcription.a} onSaved={setLoaded}
              onError={(text) => setSaveStatus({ text, error: true })} />
            <ModelSelect slot="model_b" label="Model B" info={loaded.transcription.b} onSaved={setLoaded}
              onError={(text) => setSaveStatus({ text, error: true })} />
          </div>
          <p className="field__hint settings-hint">
            Takes effect from the next transcription run. Clips already transcribed keep their text.
          </p>
        </section>

        <section className="panel" aria-labelledby="llm-title">
          <h2 id="llm-title">LLM</h2>
          <div className="settings-pair">
            <label className="field">
              Host
              <input className="input" value={llmHost} onChange={(e) => setLlmHost(e.target.value)} spellCheck={false}
                autoComplete="off" placeholder="http://localhost:11434" />
            </label>
            <label className="field">
              Model
              <input className="input" value={llmModel} onChange={(e) => setLlmModel(e.target.value)} spellCheck={false}
                autoComplete="off" />
            </label>
          </div>
          <div className="settings-test">
            <button type="button" className="btn" onClick={() => void test()} disabled={testing}>
              {testing ? "Testing…" : "Test connection"}
            </button>
            {testResult && (
              <span className={`test-result test-result--${testResult.ok ? "ok" : "fail"}`} role="status">
                {testResult.text}
              </span>
            )}
          </div>
        </section>

        <section className="panel" aria-labelledby="cookies-title">
          <h2 id="cookies-title">YouTube cookies</h2>
          <label className="field">
            <span>
              Path to cookies.txt <span className="field__hint">optional, for age-restricted videos; empty turns it off</span>
            </span>
            <input className="input" value={cookies} onChange={(e) => setCookies(e.target.value)} spellCheck={false}
              autoComplete="off" placeholder="C:\path\to\cookies.txt" aria-invalid={cookiesError ? true : undefined}
              aria-describedby={cookiesError ? "cookies-error" : undefined} />
            {cookiesError && (
              <span id="cookies-error" className="field__error" role="alert">
                {cookiesError}
              </span>
            )}
          </label>
        </section>

        <div className="settings__save">
          <button type="submit" className="btn btn--primary" disabled={saving}>
            {saving ? "Saving…" : "Save settings"}
          </button>
          {showErrors && hasErrors && (
            <span className="field__error" role="alert">
              Fix the thresholds marked above.
            </span>
          )}
          {saveStatus && (
            <span className={saveStatus.error ? "field__error" : "muted"} role={saveStatus.error ? "alert" : "status"}>
              {saveStatus.text}
            </span>
          )}
        </div>
      </form>

      <aside className="settings__side">
        {dataset ? (
          <SpeakerForm key={dataset.id} dataset={dataset} onSaved={onDatasetChanged} />
        ) : (
          <section className="panel">
            <h2>Speaker</h2>
            <p className="muted">Pick a dataset above to set the speaker name written to its metadata.csv.</p>
          </section>
        )}
        <section className="panel" aria-labelledby="models-title">
          <h2 id="models-title">Models</h2>
          <dl className="models">
            {(Object.keys(MODEL_LABEL) as (keyof StageModels)[]).map((k) => (
              <div key={k}>
                <dt>{MODEL_LABEL[k]}</dt>
                <dd>{loaded.models[k]}</dd>
              </div>
            ))}
          </dl>
        </section>
      </aside>
    </div>
  );
}
