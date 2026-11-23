# Model catalog and web UI redesign: design

Date: 2026-11-23. Builds on the core platform (spec `2026-11-23-bangla-voice-mlops-design.md`, merged as `d7eda72`). Research behind the catalog entries: `docs/context/stt-tts-model-survey.md`.

## 1. Purpose

Two changes to the platform's web app.

1. **Model catalog.** A browsable list of base models that can be fine-tuned, in three sections: LLM, STT, TTS. For each model it states what the model is, what training data it needs (mapped to the platform's upload formats), and what to target when fine-tuning it. A "Fine-tune" action starts a recipe from the entry.
2. **Web UI redesign.** Replace the current plain shadcn pages with a proper MLOps console, drawing on MLflow, Weights & Biases, Vertex AI Model Garden and Hugging Face. It adds a new app shell, a visual identity, a filterable runs table, run comparison, and a registry view.

### 1.1 Users

The sole user is the ML engineer who runs and evaluates fine-tuning on the VM. Colleagues who prepare Bangla data may read catalog pages to learn the dataset structure a model needs.

### 1.2 Success criteria

1. Each catalog entry answers, on one page: what is this model, is it licensed for commercial use, how well does it handle Bangla, what dataset do I upload, and what do I train.
2. "Fine-tune" on a catalog entry opens the recipe editor with the base model, license metadata, method and starting hyperparameters filled in.
3. Adding a model to the catalog means adding one YAML file; no code change.
4. Any 2–4 runs can be compared: overlaid metric charts, a hyperparameter diff and final eval metrics.
5. The UI passes the checks in §6.4 in both dark and light themes and at 375 px width.

### 1.3 Out of scope

Playground, voice call, real training backends, editing catalog entries in the UI, multi-user features.

## 2. Catalog data

### 2.1 Files

One YAML file per model at `catalog/<section>/<id>.yaml`, where `<section>` is `llm`, `stt` or `tts` and `<id>` equals the file's `id` field. The files live in git and change through commits; the UI is read-only.

### 2.2 Schema

`CatalogEntry` is a Pydantic model in `shared/mlops_shared/catalog.py`. All fields are required unless marked optional.

| Field | Type | Meaning |
|---|---|---|
| `id` | kebab-case slug, matches the file name | Stable identifier. |
| `task` | `llm-sft` \| `asr` \| `tts` | The platform task (same values as recipes). Section `llm` holds `llm-sft`, `stt` holds `asr`, `tts` holds `tts`; a mismatch is a validation error. |
| `name` | string | Display name. |
| `hf_repo` | string | Hugging Face repo id of the weights. |
| `derived_from` | string, optional | Upstream base when the entry is a fine-tune of another model. |
| `params` | string | Parameter count as written on the model card, e.g. `809M`, `4B`. |
| `role` | `primary` \| `comparison` \| `benchmark` | Shortlist role from the core spec §2. `benchmark` means not for production. |
| `license` | `{id: str, commercial_ok: bool, url: str, notes: str (optional)}` | `id` and `commercial_ok` feed the recipe's `base_model_meta`. |
| `bangla` | `{status, evidence: str, url: str (optional)}` | `status` is `native` (the model card lists Bangla), `finetuned` (this checkpoint is a Bangla fine-tune), `related` (no Bangla, but related Indic languages), or `none`. |
| `runtime` | `{engine: str, convert: str (optional), streaming: bool}` | How the model runs in the voice stack, e.g. RealtimeSTT `faster_whisper` with `convert: ct2`, or `llama-server` with `convert: gguf`. |
| `description` | markdown string | What the model is, in 1–3 short paragraphs. |
| `dataset` | object, below | The training data the model needs. |
| `finetune` | object, below | What to target when fine-tuning. |
| `recipe_template` | `{backend: str (optional), method: dict with "type", hparams: dict, export: list[str]}` | Values copied into a new recipe. |
| `sources` | list of `{title: str, url: str}`, at least one | Where the facts came from. |
| `verified` | date | When the facts were last checked. |

`dataset`:

| Field | Type | Meaning |
|---|---|---|
| `format` | `llm-sft` \| `asr` \| `tts` | Must equal `task`. Points at the platform upload format (core spec §4.2). |
| `required_fields` | list of strings | Columns or keys that must be present. |
| `optional_fields` | list of strings | Columns or keys the model can use. |
| `example` | string | One example row: a JSONL line, or a `metadata.csv` line. |
| `recommended` | dict of string to string | Sizing guidance, e.g. `hours: "100-500"`, `sample_rate: "16000"`, `clip_seconds: "1-30"`, `speakers: "1"`, `conversations: "5k-20k"`. |
| `notes` | markdown string, optional | Text normalisation, G2P, consent or format caveats. |

`finetune`:

| Field | Type | Meaning |
|---|---|---|
| `methods` | list of strings | E.g. `full`, `lora`, `qlora`. The first is the recommended one. |
| `frameworks` | list of strings | E.g. `unsloth`, `transformers`, `nemo`, `piper-train`. |
| `targets` | list of markdown strings | What to train and what to freeze: modules, tokenizer, G2P, speaker handling. |
| `hyperparams` | dict | Starting values, shown as a table. |
| `metrics` | list of strings | What to watch, e.g. `wer`, `cer`, `mos`, `judge_score`. |
| `pitfalls` | list of markdown strings | Known ways the fine-tune fails. |

Unknown fields are rejected, so typos surface as validation errors.

### 2.3 Initial entries

Every fact in an entry is checked against the model card or the source it cites, and `verified` is set to the date of that check. Where a source is ambiguous (for example a license that differs between Hugging Face and AIKosh), `license.notes` says so and `commercial_ok` is `false` until confirmed.

| Section | Entries (role) |
|---|---|
| LLM | Qwen3-4B-Instruct-2507 (primary), Gemma 3 4B-it (comparison), TituLM (comparison) |
| STT | Whisper large-v3-turbo (comparison), mozilla-ai `whisper-large-v3-turbo-bn` (primary), Meta Omnilingual ASR (comparison), SPRING Lab Bengali streaming Zipformer (primary, streaming), IndicConformer (primary), Nemotron 3.5 ASR streaming 0.6B (comparison), wav2vec2 XLS-R 300M (comparison), MMS-1B-all (benchmark) |
| TTS | IndicF5 (primary), Indic Parler-TTS (primary), Qwen3-TTS 0.6B Base (comparison), Kokoro-82M (comparison), Piper VITS (comparison), XTTS v2 (benchmark), MMS-TTS ben (benchmark) |

### 2.4 API

- `GET /api/catalog?task=<task>` returns `{entries: [CatalogSummary], errors: [{file, message}]}`. `CatalogSummary` holds `id, task, name, hf_repo, params, role, license, bangla.status, runtime.streaming`.
- `GET /api/catalog/{id}` returns the full entry; 404 if unknown.
- The catalog directory is a setting, `CATALOG_DIR`, defaulting to `<repo>/catalog`, set the same way as `RECIPES_DIR`. Files are read on each request (the catalog is small); a file that fails to parse or validate goes into `errors` and does not stop the others from loading.

## 3. Catalog UI

- **List** `/catalog`, with tabs LLM, STT, TTS (`?task=`). A filterable table with columns name, params, role, license with a `commercial_ok` badge, Bangla status, and streaming. Filters: role, `commercial_ok`, Bangla status. Any load `errors` show as a warning panel listing the file and message.
- **Detail** `/catalog/{id}`, in this order:
  1. Header: name, `hf_repo` link, role, license, Bangla status, streaming badges, and the Fine-tune button.
  2. "What it is": `description`, `params`, `derived_from`, runtime.
  3. "Training data": format (linked to the datasets upload page for that task), required and optional fields, the example row in a code block, the recommended sizes as a table, and notes.
  4. "Fine-tuning": methods, frameworks, targets, a starting-hyperparameter table, metrics, and pitfalls.
  5. Sources and the verified date.
- **Fine-tune** links to `/recipes/new?catalog=<id>`. The recipe editor fetches the entry and pre-fills `task`, `base_model` (= `hf_repo`), `base_model_meta` (= `license.id`, `license.commercial_ok`), `method`, `hparams`, `export` and, if set, `backend`. If the backend is unset or not in `/api/backends`, the editor shows: "No installed backend supports this model yet. Pick a backend or add one under backends/."
- **Navigation**: the existing "Models" page becomes **Registry**. Its routes move from `/models` to `/registry` and `/registry/{id}`, and `/models` redirects there. The API routes `/api/models` stay as they are.

## 4. Visual system

### 4.1 Identity

The subject is a workbench for teaching machines to hear and speak Bangla. Two things from that subject shape the look: the **matra** (মাত্রা), the headline bar Bengali letters hang from, and the **spectrogram**, the way speech looks to an engineer.

### 4.2 Color tokens

Defined as CSS variables in `globals.css` and mapped to the existing shadcn token names (`--background`, `--card`, `--primary`, …) so components keep working.

| Token | Dark | Light | Use |
|---|---|---|---|
| `ink` | `#161A24` | `#F6F7FA` | Page background |
| `slate` | `#1E2331` | `#FFFFFF` | Panels, table rows |
| `raised` | `#272D3E` | `#EEF0F5` | Popovers, hover |
| `rule` | `#353C52` | `#D5D9E3` | Borders and the matra line |
| `paper` | `#E7E9F0` | `#1A1F2C` | Primary text |
| `muted` | `#9BA2B6` | `#5A6275` | Secondary text |
| `saffron` | `#F4B740` | `#A86A00` | The single interactive accent: primary buttons (dark text on the dark-theme fill), focus ring, active nav |

Status colors are separate from the accent and always paired with an icon and a label: running `#5CC8E0`, done `#62D394`, failed `#F06B7E`, queued and cancelled in `muted`. The light theme uses darker shades of each status color, chosen to meet 4.5:1 contrast on `slate`.

Charts use a magma-style ramp of five series colors, ordered so adjacent series differ in lightness: `#B5367A`, `#F4B740`, `#7A5CC7`, `#F07D52`, `#E7E9F0` (dark theme; the light theme swaps the last for `#1A1F2C`).

Dark is the default. A theme toggle in the top bar switches to light; the choice is stored in `localStorage` and applied before first paint to avoid a flash.

### 4.3 Type

- **Hind Siliguri** (Google Fonts via `next/font`, weights 400/500/600) for all UI text. It is drawn for both Bangla and Latin, so Bangla transcripts and English labels share weight and height. It replaces Geist Sans and Noto Sans Bengali.
- **JetBrains Mono** (400/500) only for run IDs, metric values, hyperparameters, code and logs, with tabular figures.
- Scale: 13, 14, 16, 20, 26 px. Dense table body is 14 px; page titles 26 px, weight 600. Sentence case throughout; no all-caps labels.

### 4.4 The matra line

The one bold element.

- Every page title and panel heading sits under a 2 px `rule` bar that spans its container, like letters hanging from মাত্রা.
- The active nav item shows a saffron segment on its bar.
- On a run page, the run's progress (phase and step fraction) is drawn as a saffron fill along the header's matra line.

Everything else stays quiet: panels have borders and no shadows; radius is 6 px on controls and 10 px on panels; there is no motion on load. Motion is 150–200 ms and only in response to user actions (drawer open, tab change, row expand), and none at all under `prefers-reduced-motion`.

## 5. App shell and pages

### 5.1 Shell

- **Left rail**, 220 px, collapsible to a 56 px icon rail (state stored in `localStorage`). Groups:
  - Overview
  - Prepare: Catalog, Datasets, Recipes
  - Train: Runs, Compare
  - Ship: Registry
- **Top bar**:
  - breadcrumbs;
  - status chips from `/api/system`, polled every 5 s: GPU utilization and VRAM with a 60-sample history strip, GPU lock holder (linked to the holding run, or "free"), and API health (red with the error text if `/api/system` fails);
  - a `Ctrl+K` (`Cmd+K` on macOS) command palette;
  - the theme toggle.
- **Command palette**, using `cmdk`: navigate to any page, and search runs (by ID or recipe), catalog entries, recipes and datasets. Results load from the list endpoints when the palette opens.
- **Below 768 px** the rail becomes a drawer opened from the top bar. Tables scroll horizontally inside their panel; the page itself never scrolls sideways.

### 5.2 Pages

| Page | Content |
|---|---|
| Overview `/` | A "Now" panel first: the active run (live loss chart and progress on its matra line), queued runs, and the GPU lock. Then recent runs (10), the champion per task, and recent datasets. Empty states name the next step, e.g. "No runs yet. Pick a model in Catalog to start a recipe." |
| Catalog | §3. |
| Datasets | Table: name, task, latest version, size, `commercial_ok`, eval-only. Upload opens in a side drawer, and tus progress stays visible while you browse. The version page keeps the current report, splits and samples, with an inline audio player for ASR and TTS rows. |
| Recipes | Table. The editor has a YAML panel and a schema-driven form panel side by side (stacked below 1024 px), with inline validation next to each field. |
| Runs `/runs` | Filterable table: state, task, recipe, backend, `commercial_ok`, created date. Filters are kept in the URL query. Each row has a checkbox; with 2–4 rows selected, "Compare" opens the compare page. |
| Run `/runs/{id}` | Header: ID, state, recipe, and Cancel, Resume and Promote actions, with the phase timeline and progress on the matra line. Tabs: Metrics, Logs, Config, Eval, Artifacts. |
| Compare `/runs/compare?ids=a,b,c` | One chart per metric key, with each run overlaid in its ramp color and a shared legend. A hyperparameter diff table: keys that differ are highlighted and listed first; identical keys sit in a collapsed group. Final eval metrics side by side, with the best value per metric marked (direction per metric: loss, WER and CER lower is better; others higher). Fewer than 2 or more than 4 IDs, or an unknown ID, shows an inline error naming the problem. |
| Registry `/registry` | One lane each for LLM, STT and TTS, with the champion pinned at the top. A versions table shows eval metrics, `commercial_ok` and the source run. Selecting 2–4 versions and choosing "Compare" opens the compare page with their source runs. |

### 5.3 Data flow

The only new backend endpoints are the catalog ones (§2.4). Run filtering and comparison happen in the browser on `/api/runs`, `/api/runs/{id}` and `/api/runs/{id}/metrics`. This is enough for one user and hundreds of runs.

## 6. Testing

### 6.1 Python (pytest)

- `CatalogEntry` accepts a valid entry, and rejects: an unknown field, a `task` and section mismatch, a `dataset.format` not equal to `task`, an `id` that does not match the file name, and a non-slug `id`.
- The loader returns valid entries plus an `errors` item for a broken file.
- The API list filters by task, the detail endpoint returns 404 for an unknown id, and summaries carry the documented fields.
- Every file under the repo's `catalog/` validates.

### 6.2 Web (vitest)

- Catalog to recipe prefill: field mapping, and the warning when the backend is missing.
- Run filters and their round trip through the URL query.
- Hyperparameter diff: nested keys, keys missing on one side, differing versus identical.
- Metric merge for compare, and best-value marking by metric direction.
- Compare ID validation: fewer than 2, more than 4, unknown ID.
- Nav groups and active state.
- Command-palette result building.
- Component tests: catalog detail sections render from an entry, and the compare diff table highlights differing keys.

### 6.3 Existing tests

All existing pytest and vitest tests keep passing; tests that reference `/models` routes are updated to `/registry`.

### 6.4 UI checks before completion

Run the app with `run-local.bat`. Screenshot every page in dark and light at 1440 px and 375 px, and check against the ui-ux-pro-max pre-delivery checklist:

- text contrast of at least 4.5:1;
- visible keyboard focus on every interactive element;
- no horizontal page scroll at 375 px;
- `prefers-reduced-motion` respected;
- icon buttons have labels;
- status is never shown by color alone.

`npm run typecheck`, `npm run build` and `uv run pytest -q` pass.
