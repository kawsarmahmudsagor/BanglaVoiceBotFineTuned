# Model Catalog and Web UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a YAML-backed model catalog (LLM, STT and TTS sections, with "Fine-tune" leading to a pre-filled recipe) and redesign the web app as a dark-first MLOps console with run comparison and a registry view.

**Architecture:**
- **Catalog backend.** Catalog files under `catalog/{llm,stt,tts}/` are validated by a shared Pydantic schema and served read-only by two new FastAPI routes.
- **Web shell.** The Next.js app gets a new shell (grouped left rail, top bar with live system chips, `Ctrl+K` palette) and a token-based visual system built around the "matra line".
- **Browser-side features.** Run filtering and comparison run in the browser on the existing run and recipe endpoints.

**Tech Stack:** Python 3.12, Pydantic v2, FastAPI, PyYAML, pytest. Next.js 15.5, React 19, Tailwind 4, shadcn (Base UI), recharts, `cmdk`, `react-markdown`, vitest with Testing Library.

**Spec:** `docs/superpowers/specs/2026-11-23-model-catalog-ui-redesign-design.md` (research: `docs/context/stt-tts-model-survey.md`).

## Global Constraints

- **No vendored edits.** Do not modify `C:\Users\BS23-DESKTOP-00038\Projects\LocalVoiceAgent` or any vendored RealtimeSTT/RealtimeTTS code.
- **Git safety.**
  - Never commit `.env` or secrets.
  - Stage files by explicit path; never `git add -A` or `git add .`.
  - Never use `--no-verify`.
- **Commit messages:** subject `feat(<area>): …`, `fix(<area>): …` or `docs: …`, ending with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Subagents use the trailer of the model they actually are.
- **Catalog files.**
  - Path: `catalog/<section>/<id>.yaml`.
  - Section-to-task mapping: `llm`→`llm-sft`, `stt`→`asr`, `tts`→`tts`.
  - Unknown fields are rejected.
  - `verified: 2026-11-23` or the actual check date.
  - Conflicting license sources mean `commercial_ok: false` plus `license.notes`.
- **Color tokens (dark / light):**
  - ink `#161A24`/`#F6F7FA`
  - slate `#1E2331`/`#FFFFFF`
  - raised `#272D3E`/`#EEF0F5`
  - rule `#353C52`/`#D5D9E3`
  - paper `#E7E9F0`/`#1A1F2C`
  - muted `#9BA2B6`/`#5A6275`
  - saffron `#F4B740`/`#A86A00`
- **Status colors (dark):** running `#5CC8E0`, done `#62D394`, failed `#F06B7E`, queued and cancelled = muted. Light shades are darker variants meeting 4.5:1 on slate. Status is always icon + label + color.
- **Chart ramp (dark):** `#B5367A`, `#F4B740`, `#7A5CC7`, `#F07D52`, `#E7E9F0`. Light replaces the last with `#1A1F2C`.
- **Fonts:**
  - Hind Siliguri 400/500/600 (subsets `bengali`, `latin`) for all UI text.
  - JetBrains Mono 400/500 only for IDs, metric values, hyperparameters, code and logs, with `font-variant-numeric: tabular-nums`.
  - Type scale 13/14/16/20/26 px. Sentence case; no all-caps labels.
- **Shapes and motion:**
  - Radius 6 px on controls, 10 px on panels. Borders, no shadows on panels.
  - No motion on page load. 150–200 ms transitions only in response to user actions; none under `prefers-reduced-motion`.
- **Theme:** dark by default. The choice is stored in `localStorage` key `theme` (`"light"` | `"dark"`) and applied by an inline script before first paint. Every `localStorage` access is wrapped in try/catch.
- **Width:** no horizontal page scroll at 375 px. Tables scroll inside their panel.
- **Verification before claiming a task done:** `uv run pytest -q` (Python tasks) or `npm run test && npm run typecheck` in `web/` (web tasks), all passing.

## Review Focus

1. **Untrusted markdown.** A catalog `description`, `targets`, `pitfalls` or `notes` containing raw HTML (`<script>`, `<img onerror>`) must render as text, never as HTML. Test in Task 9.
2. **Partial metrics in Compare.** Compared runs may have no metrics yet, different metric keys, or a recipe version that fails to load. Compare must show "No data" for that run or metric and still render the others. Tests in Task 12.
3. **No GPU or no system endpoint.** If `/api/system` fails or returns `gpu: null` (no NVIDIA GPU on a dev box), the top-bar chips show "No GPU" or a red "API unreachable" chip with the error. The shell never throws. Test in Task 7.
4. **Blocked storage.** If `localStorage` throws (private mode or blocked storage), the theme defaults to dark and the rail to expanded, without a crash. Tests in Task 6 and Task 7.
5. **Long values at 375 px.** Long Bangla text, model names and run IDs truncate with a `title` tooltip inside table cells and never widen the page. Checked in Task 15's screenshots; table cells use `max-w-*` + `truncate` from Task 6's `DataTable` styling.

---

### Task 1: Catalog schema and loader

**Files:**
- Create: `shared/mlops_shared/catalog.py`
- Test: `shared/tests/test_catalog.py`

**Interfaces:**
- Produces:
  - `SECTION_TASK: dict[str, str] = {"llm": "llm-sft", "stt": "asr", "tts": "tts"}`
  - Pydantic models, all `model_config = ConfigDict(extra="forbid")`:
    - `License(id: str, commercial_ok: bool, url: str, notes: str | None = None)`
    - `Bangla(status: Literal["native","finetuned","related","none"], evidence: str, url: str | None = None)`
    - `Runtime(engine: str, convert: str | None = None, streaming: bool)`
    - `DatasetGuide(format: Literal["llm-sft","asr","tts"], required_fields: list[str], optional_fields: list[str] = [], example: str, recommended: dict[str, str], notes: str | None = None)`
    - `FinetuneGuide(methods: list[str] (min 1), frameworks: list[str], targets: list[str], hyperparams: dict, metrics: list[str], pitfalls: list[str])`
    - `RecipeTemplate(backend: str | None = None, method: dict, hparams: dict, export: list[str] = [])`. `method` must contain `"type"`.
    - `Source(title: str, url: str)`
    - `CatalogEntry(id, task: Literal["llm-sft","asr","tts"], name, hf_repo, derived_from: str | None = None, params: str, role: Literal["primary","comparison","benchmark"], license: License, bangla: Bangla, runtime: Runtime, description: str, dataset: DatasetGuide, finetune: FinetuneGuide, recipe_template: RecipeTemplate, sources: list[Source] (min 1), verified: datetime.date)`
    - `id` is validated with `_SLUG_RE.fullmatch` imported from `mlops_shared.recipes`. A model validator requires `dataset.format == task`.
  - `CatalogError(BaseModel): file: str; message: str`
  - `load_catalog(root: Path) -> tuple[list[CatalogEntry], list[CatalogError]]`.
    - Reads `root/<section>/*.yaml` for each key of `SECTION_TASK`.
    - These problems become `CatalogError`s:
      - YAML parse error;
      - `ValidationError`;
      - `id != file stem`;
      - `task != SECTION_TASK[section]`.
    - `CatalogError.file` is the path relative to `root`, in posix form.
    - Valid entries are sorted by `(task, role order primary<comparison<benchmark, name)`.
    - A missing `root` returns `([], [])`.
  - `catalog_summary(e: CatalogEntry) -> dict` with keys `id, task, name, hf_repo, params, role, license (dict), bangla_status, streaming`.

- [ ] **Step 1: Write the failing tests** in `shared/tests/test_catalog.py`.
  - Use a `valid_entry()` dict helper (`task="asr"`, `dataset.format="asr"`) and a `write(root, section, stem, data)` helper.
  - Tests:
    - `test_valid_entry_loads`: one file under `stt/`; one entry and no errors.
    - `test_unknown_field_rejected`: adds `foo: 1`; one error whose `file == "stt/<stem>.yaml"`.
    - `test_task_section_mismatch`: an `asr` entry under `tts/` gives an error mentioning `section`.
    - `test_dataset_format_must_equal_task`: `dataset.format="tts"` gives an error.
    - `test_id_must_match_filename`: `id="a"` in `b.yaml` gives an error.
    - `test_non_slug_id_rejected`: `id="Bad_Id"` and `"a\n"`.
    - `test_bad_yaml_reported_others_load`: one broken file and one valid file give 1 entry and 1 error.
    - `test_missing_root_is_empty`.
    - `test_sorted_by_task_role_name`.
    - `test_summary_fields`: exact key set.
- [ ] **Step 2: Run** `uv run pytest shared/tests/test_catalog.py -q`. Expected: FAIL (`ModuleNotFoundError: mlops_shared.catalog`).
- [ ] **Step 3: Implement** `shared/mlops_shared/catalog.py` per Interfaces. Use `yaml.safe_load`. Error messages: `str(exc)` for validation errors; `"id 'x' does not match file name 'y'"`; `"task 'asr' does not belong in section 'tts'"`.
- [ ] **Step 4: Run** `uv run pytest shared/tests/test_catalog.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `shared/mlops_shared/catalog.py shared/tests/test_catalog.py` as `feat(shared): catalog entry schema and loader`.

### Task 2: Catalog API, `CATALOG_DIR`, and task on the runs list

**Files:**
- Create: `api/mlops_api/routers/catalog.py`, `api/tests/test_catalog_api.py`, `catalog/.gitkeep`
- Modify:
  - `api/mlops_api/settings.py`: add `catalog_dir: Path = REPO_ROOT / "catalog"`.
  - `api/mlops_api/app.py`: include the router.
  - `api/mlops_api/routers/runs.py` `list_runs`: add `"task": rv.spec.get("task")`.
  - `conftest.py`: `catalog_dir=tmp_path / "catalog"` in the `settings` fixture.
  - `docker-compose.yml` api service: env `CATALOG_DIR: ${REPO_ROOT}/catalog` and volume `./catalog:${REPO_ROOT}/catalog:ro`.
  - `run-local.bat`: `set "CATALOG_DIR=%ROOT%\catalog"`.
  - `.env.example`: a commented `# CATALOG_DIR=` line explaining the default.
  - `api/tests/test_runs_api.py`: assert `task` is in list items.

**Interfaces:**
- Consumes: `load_catalog`, `catalog_summary` (Task 1).
- Produces:
  - `GET /api/catalog?task=<llm-sft|asr|tts>` returns `{"entries": [summary], "errors": [{"file","message"}]}`. An invalid `task` gives 422. `errors` is always the full list, unfiltered.
  - `GET /api/catalog/{entry_id}` returns `CatalogEntry.model_dump(mode="json")`; 404 `"catalog entry 'x' not found"`.
  - Run list items gain `task: str | null`.

- [ ] **Step 1: Write failing tests** in `api/tests/test_catalog_api.py`, writing YAML into `settings.catalog_dir`. Reuse a valid-entry helper by copying the dict builder; do not import from `shared/tests`.
  - `test_list_all_and_errors`: one valid file and one broken file give 1 entry and 1 error.
  - `test_list_filters_by_task`.
  - `test_invalid_task_422`.
  - `test_detail_and_404`.
  - `test_summary_has_documented_fields`.
  - In `test_runs_api.py`, assert `item["task"] == "llm-sft"` in the existing list test.
- [ ] **Step 2: Run** `uv run pytest api/tests/test_catalog_api.py api/tests/test_runs_api.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the router (`APIRouter(prefix="/api")`; reads `request.app.state.settings.catalog_dir` on each request), the settings field, the app include, the runs field, and the compose, bat and env edits.
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** the listed files as `feat(api): read-only catalog endpoints and run task field`.

### Task 3: LLM catalog entries

**Files:**
- Create: `catalog/llm/qwen3-4b-instruct-2507.yaml`, `catalog/llm/gemma-3-4b-it.yaml`, `catalog/llm/titulm.yaml`
- Create: `shared/tests/test_repo_catalog.py`

**Interfaces:**
- Consumes: `load_catalog` (Task 1).
- Produces: `test_repo_catalog_valid`: `load_catalog(REPO/"catalog")` returns no errors and at least one entry. Later content tasks rely on this test.

- [ ] **Step 1: Write** `shared/tests/test_repo_catalog.py`: `errors == []` and `len(entries) >= 1`. With an empty catalog it fails on the length check. Run it and record the failure.
- [ ] **Step 2: Research each model on the web.** Read the Hugging Face model card and the license file (WebSearch, or the web-fetch agent) for: exact `hf_repo`, parameter count, license id, commercial terms, and Bangla capability with evidence.
  - TituLM: pick the current Hishab TituLM release on Hugging Face and record which one.
  - Gemma: license id `gemma`. `commercial_ok` is true only if the Gemma Terms allow commercial use, with a note on the prohibited-use policy.
- [ ] **Step 3: Write the three files.**
  - Role and `bangla.status`:
    - Qwen3-4B-Instruct-2507: role `primary`; `bangla.status` from its card's language list (`native` if Bengali is listed, else `related`).
    - Gemma 3 4B-it: role `comparison`.
    - TituLM: role `comparison`, `bangla.status: native`.
  - `runtime`: `{engine: llama-server, convert: gguf, streaming: true}`.
  - `dataset`: format `llm-sft`, `required_fields: [messages]`, `optional_fields: [tools, tags]`. The example is one Bangla e-commerce chat JSONL line with a plain-text assistant reply. `recommended: {conversations: "2k-20k", turns: "1-8", language: "bn (Bangla script), code-mixed allowed"}`. Notes: Bangla normalisation, no markdown in replies.
  - `finetune`:
    - `methods: [qlora, lora, full]`, `frameworks: [unsloth, transformers]`.
    - `targets`: LoRA on all attention and MLP projections, r 16–32. Train on assistant turns only. Keep the chat template. Check tokenizer efficiency on Bangla (tokens per word).
    - `hyperparams`: a learning rate of 2e-4 (LoRA), 2–3 epochs, max_seq_len 2048, and so on.
    - `metrics: [eval_loss, judge_score, refusal_rate]`.
    - `pitfalls`: catastrophic forgetting of instruction following, template mismatch at serving, and markdown leaking into replies.
  - `recipe_template`: `method: {type: qlora}`, `hparams` from above, `export: [gguf]`, no `backend`.
  - Every claim needs a `sources` URL.
- [ ] **Step 4: Run** `uv run pytest shared/tests/test_repo_catalog.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** the 3 YAML files and the test as `feat(catalog): LLM entries`.

### Task 4: STT catalog entries

**Files:** Create `catalog/stt/` with:
- `whisper-large-v3-turbo.yaml`
- `whisper-large-v3-turbo-bn.yaml`
- `omnilingual-asr.yaml`
- `bengali-zipformer-streaming.yaml`
- `indicconformer.yaml`
- `nemotron-3-5-asr-streaming.yaml`
- `wav2vec2-xls-r-300m.yaml`
- `mms-1b-all.yaml`

**Interfaces:** Consumes the Task 1 schema and the Task 3 repo test.

- [ ] **Step 1: Research each model on the web** (model card plus license). Start from `docs/context/stt-tts-model-survey.md`, which already cites sources.
  - Settle the open items:
    - the exact SPRING Lab repo id and license;
    - the IndicConformer repo for Bengali and its license;
    - which Omnilingual size to list (CTC 300M or 1B; record the choice);
    - the exact MMS-1B-all license (CC-BY-NC means `benchmark`, `commercial_ok: false`).
  - Roles follow spec §2.3.
- [ ] **Step 2: Write the 8 files.**
  - `dataset`: format `asr`, `required_fields: [file, text]`, `optional_fields: [region, channel, noise, speaker]`. The example is a `metadata.csv` line with a Bangla transcript.
  - `recommended` per model, from the survey's data-size table, plus `sample_rate: "16000"` and `clip_seconds: "1-30"`.
  - `finetune.targets` per model:
    - **Whisper:** full fine-tune or LoRA on the decoder plus top encoder layers; language token `bn`; convert to CT2 for RealtimeSTT.
    - **Zipformer:** fine-tune with icefall; keep the BPE model if it already covers Bengali.
    - **IndicConformer / Nemotron:** NeMo; Nemotron needs a new tokenizer per NeMo docs, re-initialising the decoder and joint network, with the Malay-collapse warning.
    - **XLS-R:** CTC head on Bengali characters.
    - **MMS:** adapter per language.
  - `metrics: [wer, cer]`, plus `rtf` for streaming models.
  - `runtime.engine` is the RealtimeSTT engine name where one exists (`faster_whisper`, `omnilingual_asr`, `sherpa_onnx_nemotron`). Otherwise use `custom` and say so in `description`.
- [ ] **Step 3: Run** `uv run pytest shared/tests/test_repo_catalog.py -q`. Expected: PASS.
- [ ] **Step 4: Commit** the 8 files as `feat(catalog): STT entries`.

### Task 5: TTS catalog entries

**Files:** Create `catalog/tts/` with:
- `indicf5.yaml`
- `indic-parler-tts.yaml`
- `qwen3-tts-0-6b-base.yaml`
- `kokoro-82m.yaml`
- `piper-vits.yaml`
- `xtts-v2.yaml`
- `mms-tts-ben.yaml`

**Interfaces:** Consumes the Task 1 schema and the Task 3 repo test.

- [ ] **Step 1: Research each model on the web**, starting from the survey.
  - Indic Parler-TTS: Apache-2.0 on Hugging Face vs MIT on AIKosh means `commercial_ok: false` plus `license.notes`.
  - IndicF5: MIT, gated.
  - XTTS v2: CPML, `benchmark`.
  - MMS-TTS: check the license; CC-BY-NC means `benchmark`.
- [ ] **Step 2: Write the 7 files.**
  - `dataset`: format `tts`, `required_fields: [file, text, speaker]`, `optional_fields: [style]`. The example is a `metadata.csv` line.
  - `recommended`: `hours` per the survey table, `speakers`, sample rate per model (24000 for IndicF5 and Qwen3-TTS, 44100 for Parler, 22050 for Piper; verify each), and `clip_seconds: "2-15"`.
  - `finetune.targets` per model:
    - **Parler:** fine-tune the decoder; description prompts for speaker and style.
    - **IndicF5:** F5 fine-tune with reference audio and text pairs.
    - **Qwen3-TTS:** official `sft_12hz.py`, single speaker, Base model only; new language unproven.
    - **Kokoro:** StyleTTS2 community recipe; needs a Bangla G2P.
    - **Piper:** `piper1-gpl` from a checkpoint; espeak-ng `bn`.
    - **XTTS:** BnTTS precedent of 3.85k h.
  - `metrics: [mos, cer_via_asr, speaker_similarity, rtf]`.
  - `runtime.engine`: the RealtimeTTS engine class where one exists (`ParlerEngine`, `QwenEngine`, `KokoroEngine`, `PiperEngine`, `CoquiEngine`), else `custom`.
- [ ] **Step 3: Run** `uv run pytest shared/tests/test_repo_catalog.py -q`. Expected: PASS.
- [ ] **Step 4: Commit** the 7 files as `feat(catalog): TTS entries`.

### Task 6: Visual foundation

**Files:**
- Modify:
  - `web/src/app/globals.css`: tokens; `:root` = light, `.dark` = dark; mapped to the shadcn variable names.
  - `web/src/app/layout.tsx`: fonts and the theme script.
  - `web/package.json`: add `cmdk`, `react-markdown`.
- Create:
  - `web/src/lib/theme.ts`
  - `web/src/lib/theme.test.ts`
  - `web/src/lib/palette.ts`
  - `web/src/components/matra.tsx`
  - `web/src/components/status-badge.tsx`
  - `web/src/components/status-badge.test.tsx`

**Interfaces:**
- Produces:
  - `theme.ts`:
    - `type Theme = "light" | "dark"`
    - `readTheme(storage?: Pick<Storage,"getItem">): Theme`: returns `"dark"` if the key is missing, invalid, or the access throws.
    - `writeTheme(t: Theme, storage?)`: swallows errors.
    - `THEME_SCRIPT: string`: an inline script that adds class `dark` unless the stored value is `"light"`.
  - `palette.ts`:
    - `SERIES_DARK: string[]`, `SERIES_LIGHT: string[]` (the 5-color ramps from Global Constraints).
    - `seriesColor(i: number, theme: Theme): string`: cycles through the ramp.
  - `matra.tsx`:
    - `<PageHeader title: string; actions?: ReactNode; progress?: number /* 0..1 */; children?>`: a title row hanging from a 2 px `rule` bar. When `progress` is set, the bar carries a saffron fill of that width, with `role="progressbar"`, `aria-valuenow` in percent and `aria-label="Progress"`.
    - `<Panel title?: string; actions?; children>`: a bordered 10 px panel whose heading hangs from a matra bar.
  - `status-badge.tsx`: `<StatusBadge state: RunState>`. Renders a lucide icon (running: `Loader`, done: `CheckCircle2`, failed: `XCircle`, queued: `Clock`, cancelled: `Ban`, other active phases: `Loader`), the state label text, and the status color class.

- [ ] **Step 1: Write failing tests.**
  - `theme.test.ts`: `readTheme` returns dark for missing, for `"blue"`, and when `getItem` throws; returns light for `"light"`. `writeTheme` does not throw when `setItem` throws.
  - `status-badge.test.tsx`: each `RUN_STATES` value renders its label text and an `svg`.
- [ ] **Step 2: Run** `npm run test` in `web/`. Expected: FAIL (modules missing).
- [ ] **Step 3: Install and implement.**
  - `npm install cmdk@^1 react-markdown@^10`.
  - Implement `theme.ts`, `palette.ts`, `matra.tsx`, `status-badge.tsx`.
  - Replace the tokens in `globals.css` with the Global Constraints values. Keep the shadcn variable names; map `--primary` to saffron and `--ring` to saffron. Add `.font-mono { font-variant-numeric: tabular-nums }` and the reduced-motion rule (`@media (prefers-reduced-motion: reduce) { *,*::before,*::after { transition-duration: 0s !important; animation: none !important } }`).
  - In `layout.tsx`: `Hind_Siliguri({subsets:["bengali","latin"], weight:["400","500","600"], variable:"--font-sans-ui"})` and `JetBrains_Mono({subsets:["latin"], weight:["400","500"], variable:"--font-mono-ui"})`, wired to `--font-sans` and `--font-mono`. Remove the Geist and Noto fonts. Use `THEME_SCRIPT`.
- [ ] **Step 4: Run** `npm run test && npm run typecheck`. Expected: PASS.
- [ ] **Step 5: Commit** the listed files plus `web/package-lock.json` as `feat(web): visual tokens, fonts, theme and matra primitives`.

### Task 7: App shell and Registry route

**Files:**
- Create:
  - `web/src/components/shell/sidebar.tsx`
  - `web/src/components/shell/top-bar.tsx`
  - `web/src/components/shell/status-chips.tsx`
  - `web/src/components/shell/breadcrumbs.tsx`
  - `web/src/components/shell/app-shell.tsx`
  - `web/src/lib/nav.ts`
  - `web/src/lib/nav.test.ts`
  - `web/src/lib/gpu-history.ts`
  - `web/src/lib/gpu-history.test.ts`
  - `web/src/components/shell/status-chips.test.tsx`
- Move: `web/src/app/models/` to `web/src/app/registry/` (`page.tsx`, `[id]/page.tsx`); update internal links (`page.tsx` overview, `runs/[id]/page.tsx` promote redirect).
- Modify:
  - `web/next.config.ts`: `redirects()` for `/models` → `/registry` and `/models/:id` → `/registry/:id` (permanent false).
  - `web/src/app/layout.tsx`: use `<AppShell>`.
- Delete: `web/src/components/nav.tsx`. Move its tests' intent to `nav.test.ts`.

**Interfaces:**
- Consumes: `PageHeader`, `readTheme`/`writeTheme` (Task 6); `getSystem` (existing).
- Produces:
  - `nav.ts`:
    - `NAV_GROUPS: {label: string | null; items: {href; label; icon: LucideIcon}[]}[]`, in this order:
      - `null`: Overview `/`
      - Prepare: Catalog `/catalog`, Datasets `/datasets`, Recipes `/recipes`
      - Train: Runs `/runs`, Compare `/runs/compare`
      - Ship: Registry `/registry`
    - `isActive(pathname, href)`: same rule as today, except that `/runs/compare` must not mark Runs active, and Runs must not be active on `/runs/compare`.
    - `breadcrumbs(pathname): {href; label}[]`: maps known segments to labels; dynamic segments are shown verbatim.
  - `gpu-history.ts`: `pushSample(hist: number[], v: number, max = 60): number[]`, immutable.
  - `<StatusChips system: SystemInfo | null; error: string | null>`:
    - error set: a red chip "API unreachable" with `title=error`;
    - `gpu` null: "No GPU";
    - otherwise: util %, VRAM used/total GB, and a lock chip (a link to `/runs/<holder_id>` when `holder_kind==="run"`, else "GPU free").
  - `<AppShell>`: the rail (collapsible to icons; collapsed state in `localStorage` key `rail`, try/catch) and the top bar with breadcrumbs, chips (polling `getSystem` every 5000 ms), a palette slot and the theme toggle. Below 768 px the rail is a drawer opened by a labelled menu button.

- [ ] **Step 1: Write failing tests.**
  - `nav.test.ts`: group order; `isActive` cases including `/runs/compare` vs `/runs/r-1`; `breadcrumbs("/catalog/whisper-x")` returns `[{"/catalog","Catalog"},{"/catalog/whisper-x","whisper-x"}]`.
  - `gpu-history.test.ts`: caps at max; does not mutate its input.
  - `status-chips.test.tsx`: error shows "API unreachable"; `gpu: null` shows "No GPU"; a run holder renders a link to `/runs/r-1`.
- [ ] **Step 2: Run** `npm run test`. Expected: FAIL.
- [ ] **Step 3: Implement** the shell, the move, the redirects and the link updates. Update `page.test.tsx` and `api.test.ts` only where they reference the `/models` *page* routes; the API paths stay `/api/models`.
- [ ] **Step 4: Run** `npm run test && npm run typecheck && npm run build`. Expected: PASS.
- [ ] **Step 5: Commit** the listed paths (use `git add` on each moved and deleted path explicitly) as `feat(web): app shell, status chips, registry route`.

### Task 8: Command palette

**Files:**
- Create: `web/src/components/shell/command-palette.tsx`, `web/src/lib/palette-search.ts`, `web/src/lib/palette-search.test.ts`
- Modify: `web/src/components/shell/top-bar.tsx`: mount the palette with a labelled trigger button showing the `Ctrl K` hint.

**Interfaces:**
- Consumes: `NAV_GROUPS` (Task 7); `listRuns`, `listRecipes`, `listDatasets` (existing).
- Produces (this task adds the catalog client and types; Task 9 reuses them):
  - `palette-search.ts`: `buildItems(src: {runs; recipes; datasets; catalog}): {group: "Pages"|"Runs"|"Catalog"|"Recipes"|"Datasets"; label: string; href: string; keywords: string[]}[]`.
    - Run items are labelled `"<id> · <recipe> v<n>"` and their keywords include the state.
    - Catalog items are labelled with the entry name and keywords `[id, hf_repo, task]`.
  - `listCatalog(task?: Task)` in `api.ts` returns `CatalogList` (`{entries: CatalogSummary[]; errors: CatalogLoadError[]}`); types go in `types.ts`.
- [ ] **Step 1: Write failing tests:** `buildItems` covers every page in `NAV_GROUPS`; run labels and hrefs; catalog keywords; empty sources give only Pages.
- [ ] **Step 2: Run** `npm run test`. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - Use the `cmdk` `Command.Dialog`, opened by `Ctrl+K`/`Cmd+K` (and closed by Esc).
  - Load sources when the palette opens. Failed sources are skipped with a one-line note inside the palette.
  - Selecting an item calls `router.push(href)`.
- [ ] **Step 4: Run** `npm run test && npm run typecheck`. Expected: PASS.
- [ ] **Step 5: Commit** as `feat(web): command palette`.

### Task 9: Catalog pages

**Files:**
- Create:
  - `web/src/lib/catalog.ts`
  - `web/src/lib/catalog.test.ts`
  - `web/src/app/catalog/page.tsx`
  - `web/src/app/catalog/[id]/page.tsx`
  - `web/src/components/catalog-detail.tsx`
  - `web/src/components/catalog-detail.test.tsx`
  - `web/src/components/markdown.tsx`
- Modify: `web/src/lib/types.ts` (add `CatalogEntry` mirroring Task 1; `CatalogSummary`, `CatalogList` and `CatalogLoadError` come from Task 8), `web/src/lib/api.ts` (`getCatalogEntry(id)`).

**Interfaces:**
- Consumes: `PageHeader`, `Panel` (Task 6); `listCatalog` (Task 8).
- Produces:
  - `catalog.ts`:
    - `SECTIONS = [{key:"llm", task:"llm-sft", label:"LLM"}, {key:"stt", task:"asr", label:"STT"}, {key:"tts", task:"tts", label:"TTS"}]`
    - `filterCatalog(entries, f: {role?: Role; commercialOnly?: boolean; bangla?: BanglaStatus}): CatalogSummary[]`
    - `BANGLA_LABEL: Record<BanglaStatus,string>` = `{native:"Bangla native", finetuned:"Bangla fine-tune", related:"Related languages only", none:"No Bangla"}`
  - `<Markdown text: string>`: `react-markdown` with no `rehype-raw`, so raw HTML renders as text, and links with `rel="noreferrer" target="_blank"`.
  - `<CatalogDetail entry: CatalogEntry>`: sections in spec §3 order, under the headings "What it is", "Training data", "Fine-tuning", "Sources".
    - The example row is in a `<pre>`.
    - The "Fine-tune" button links to `/recipes/new?catalog=<id>`.
    - "Training data" links to `/datasets/new?task=<task>`.
- [ ] **Step 1: Write failing tests.**
  - `catalog.test.ts`: `filterCatalog` by role, `commercialOnly` and bangla status.
  - `catalog-detail.test.tsx`:
    - the four headings render in order;
    - the Fine-tune href;
    - the example is in a `pre`;
    - a `description` of `"<script>alert(1)</script><img src=x onerror=alert(1)>"` produces no `script` or `img` element, and the literal text is visible.
- [ ] **Step 2: Run** `npm run test`. Expected: FAIL.
- [ ] **Step 3: Implement the pages.**
  - **List:** tabs from `SECTIONS` (state in `?task=`), filters (state in the query), and a table with columns name (link), params (mono), role, a license badge with `commercial_ok`, Bangla label, and streaming. Load `errors` show in a warning `Panel` listing file and message. The empty state reads "No catalog entries for this section. Add a YAML file under catalog/<key>/."
  - **Detail:** fetch the entry and render `CatalogDetail`. A 404 shows "No catalog entry named <id>."
- [ ] **Step 4: Run** `npm run test && npm run typecheck`. Expected: PASS.
- [ ] **Step 5: Commit** as `feat(web): catalog list and detail pages`.

### Task 10: Recipe prefill from catalog, and Recipes pages restyle

**Files:**
- Create: `web/src/lib/catalog-prefill.ts`, `web/src/lib/catalog-prefill.test.ts`
- Modify:
  - `web/src/app/recipes/new/page.tsx`: handle `?catalog=`.
  - `web/src/components/recipe-editor.tsx`: accept `notice?: string` and render it above the form; layout as side-by-side YAML and form panels, stacked below 1024 px; field errors inline.
  - `web/src/app/recipes/page.tsx`, `web/src/app/recipes/[name]/page.tsx`: `PageHeader`, `Panel`, restyled table.

**Interfaces:**
- Consumes: `getCatalogEntry` (Task 9), `listBackends` (existing).
- Produces:
  - `catalogToSpec(entry: CatalogEntry): Record<string, unknown>`. It returns `{name:"", task, backend: recipe_template.backend ?? "", base_model: hf_repo, base_model_meta:{license: license.id, commercial_ok: license.commercial_ok}, datasets:{train:"", eval:null}, method, hparams, export}`.
  - `backendNotice(backend: string, installed: string[]): string | null`. When `backend` is empty or not installed it returns the spec text, "No installed backend supports this model yet. Pick a backend or add one under backends/."; otherwise `null`.
- [ ] **Step 1: Write failing tests:** the `catalogToSpec` mapping (deep equality on a fixture); `backendNotice` for empty, missing and installed backends.
- [ ] **Step 2: Run** `npm run test`. Expected: FAIL.
- [ ] **Step 3: Implement.** The new-recipe page loads the entry, then the backends, and passes `initial={catalogToSpec(e)}` and `notice`. The header reads "New recipe (from catalog: <name>)". A load failure shows the existing alert pattern.
- [ ] **Step 4: Run** `npm run test && npm run typecheck`. Expected: PASS.
- [ ] **Step 5: Commit** as `feat(web): fine-tune from catalog and recipe pages restyle`.

### Task 11: Runs table with filters and selection, and run page redesign

**Files:**
- Create: `web/src/lib/run-filters.ts`, `web/src/lib/run-filters.test.ts`
- Modify: `web/src/lib/types.ts` (`RunListItem.task: Task | null`), `web/src/app/runs/page.tsx`, `web/src/app/runs/[id]/page.tsx`, `web/src/components/phase-timeline.tsx`

**Interfaces:**
- Consumes: `StatusBadge`, `PageHeader` with `progress` (Task 6).
- Produces:
  - `run-filters.ts`:
    - `RunFilter = {state?: RunState[]; task?: Task; recipe?: string; backend?: string; commercialOnly?: boolean; from?: string; to?: string}` (`from`/`to` are ISO dates, inclusive, compared against `created_at`).
    - `applyRunFilter(runs, f)`
    - `filterToQuery(f): URLSearchParams` and `queryToFilter(q: URLSearchParams): RunFilter`, which round-trip. `state` is comma-joined; unknown states are dropped.
    - `compareHref(ids: string[]): string` returns `/runs/compare?ids=a,b`.
  - `runProgress(status: RunStatus | null): number | undefined`: the fraction `step/total_steps` when both are present, else undefined.
- [ ] **Step 1: Write failing tests:** each filter dimension; the round trip `queryToFilter(filterToQuery(f))` equals `f`; unknown state dropped; date bounds inclusive; `compareHref`; `runProgress` with and without totals. Check the `RunStatus` field names in `types.ts` first and use them.
- [ ] **Step 2: Run** `npm run test`. Expected: FAIL.
- [ ] **Step 3: Implement the pages.**
  - **Runs page:** filter bar synced to the URL via `router.replace`; a table with a checkbox column, ID (mono), recipe vN, task, backend, `StatusBadge`, `commercial_ok`, created. A "Compare (n)" button is enabled for 2–4 selected rows and goes to `compareHref`.
  - **Run page:** `PageHeader` with ID, `StatusBadge`, recipe link, the Cancel/Resume/Promote actions, and `progress={runProgress(...)}`. The phase timeline sits under it. Tabs: Metrics, Logs, Config (recipe version YAML via `getRecipeVersion`), Eval, Artifacts. Existing behaviour (SSE, cancel, resume, promote to `/registry/<id>`) is unchanged.
- [ ] **Step 4: Run** `npm run test && npm run typecheck`. Expected: PASS.
- [ ] **Step 5: Commit** as `feat(web): filterable runs table and run page redesign`.

### Task 12: Compare page

**Files:**
- Create:
  - `web/src/lib/compare.ts`
  - `web/src/lib/compare.test.ts`
  - `web/src/app/runs/compare/page.tsx`
  - `web/src/components/hparam-diff.tsx`
  - `web/src/components/hparam-diff.test.tsx`
  - `web/src/components/compare-charts.tsx`

**Interfaces:**
- Consumes: `getRun`, `getRunMetrics` (existing; check their names in `api.ts`), `getRecipeVersion`, `seriesColor` (Task 6).
- Produces:
  - `parseCompareIds(q: string | null, known?: Set<string>): {ids: string[]} | {error: string}`.
    - Fewer than 2 gives `"Pick at least 2 runs to compare."`; more than 4 gives `"Compare up to 4 runs at a time."`.
    - Duplicates are removed first.
    - IDs missing from `known` (when given) give `"Unknown run: <id>"`.
  - `flatten(obj, prefix = ""): Record<string, unknown>`, with dot-joined keys.
  - `diffHparams(specs: Record<string, Record<string, unknown> | null>): {key: string; values: Record<string, unknown>; differs: boolean}[]`.
    - Input is keyed by run ID. It flattens `hparams`, `method` (prefixed `method.`), `base_model` and `datasets.*`.
    - A key missing on one side shows as `undefined` and `differs: true`.
    - Rows are sorted differing first, then by key.
    - A `null` spec (failed to load) contributes nothing.
  - `mergeMetrics(byRun: Record<string, RunMetric[]>): Record<string /* split/name */, {step: number; [runId: string]: number}[]>`.
    - One series per metric, keyed `"<split>/<name>"`.
    - Rows are sorted by step. A run without that metric has no field in the row.
  - `metricDirection(name: string): "min" | "max"`: `min` when the name matches `/loss|wer|cer|error|rtf/i`, else `max`.
  - `bestRun(values: Record<string, number | undefined>, dir): string | null`. Ties return the first in input order; all-undefined returns null.
  - `<HparamDiff rows>`: differing rows highlighted with `data-differs="true"` and a "Differs" text marker, not color alone. Identical rows sit in a collapsed `<details>` labelled "Same in all runs (n)".
  - `<CompareCharts merged; runIds>`: one recharts LineChart per metric key with a legend. A metric with no points for a run shows that run in the legend as "<id> (no data)".
- [ ] **Step 1: Write failing tests.**
  - `parseCompareIds`: all cases, including duplicates.
  - `diffHparams`: nested keys, a key missing on one side, identical values, a null spec.
  - `mergeMetrics`: different keys per run, unaligned steps, a run with zero metrics.
  - `metricDirection` cases.
  - `bestRun`: min, max, tie, all-undefined.
  - `hparam-diff.test.tsx`: differing rows are marked and listed first; identical rows are inside `details`.
- [ ] **Step 2: Run** `npm run test`. Expected: FAIL.
- [ ] **Step 3: Implement the page.**
  - Read `ids` and validate with `parseCompareIds`, using the known set from `listRuns`. Errors show inline with a link back to Runs.
  - Load, per run: `getRun`, metrics from offset 0, and the recipe version spec. Any individual failure is recorded per run and shown as a "Could not load <id>: <msg>" note, while the other runs still render.
  - Sections, in order:
    1. charts;
    2. hyperparameter diff;
    3. a final eval metrics table with the best value per metric marked by a "Best" text badge plus bold.
- [ ] **Step 4: Run** `npm run test && npm run typecheck`. Expected: PASS.
- [ ] **Step 5: Commit** as `feat(web): compare runs`.

### Task 13: Registry lanes and Overview redesign

**Files:**
- Modify: `web/src/app/registry/page.tsx`, `web/src/app/registry/[id]/page.tsx`, `web/src/app/page.tsx`, `web/src/app/page.test.tsx`, `web/src/lib/models.ts`, `web/src/lib/models.test.ts`

**Interfaces:**
- Consumes: `groupByTask`, `TASK_ORDER` (existing); `compareHref` (Task 11); `PageHeader`, `Panel`, `StatusBadge` (Task 6); `seriesColor`.
- Produces: `laneOrder(models: ModelVersion[]): ModelVersion[]`, which puts the champion first and the rest by `created_at` descending. Add it to `models.ts`.
- [ ] **Step 1: Write failing tests.**
  - `models.test.ts`: `laneOrder` champion first, then newest first.
  - Update `page.test.tsx`:
    - the Overview renders a "Now" panel heading;
    - with no runs it shows "No runs yet. Pick a model in Catalog to start a recipe." with a link to `/catalog`;
    - champion links point to `/registry/<id>`.
- [ ] **Step 2: Run** `npm run test`. Expected: FAIL.
- [ ] **Step 3: Implement the pages.**
  - **Registry:** one `Panel` lane per task, labelled LLM, STT, TTS, ordered by `laneOrder`. A versions table with a checkbox, name vN, champion star with a "Champion" label, key metrics (mono), `commercial_ok` and a source-run link. "Compare (n)" for 2–4 selected goes to `compareHref(source run ids)`.
  - **Model detail:** restyled with `PageHeader` and `Panel`; behaviour unchanged.
  - **Overview:**
    1. The "Now" panel: the active run (first run with an active state) with `StatusBadge`, a compact live loss chart (metrics polled every 5 s while active), and progress. Then the queued count and the lock holder.
    2. Recent runs (limit 10).
    3. Champions per task.
    4. Recent datasets (5).
- [ ] **Step 4: Run** `npm run test && npm run typecheck`. Expected: PASS.
- [ ] **Step 5: Commit** as `feat(web): registry lanes and overview redesign`.

### Task 14: Datasets pages restyle and upload drawer

**Files:**
- Create: `web/src/components/drawer.tsx`
- Modify:
  - `web/src/app/datasets/page.tsx`
  - `web/src/app/datasets/new/page.tsx`: keep it working as a full page for deep links, and read `?task=` to preselect the task.
  - `web/src/app/datasets/[name]/page.tsx`
  - `web/src/app/datasets/[name]/v/[n]/page.tsx`
  - `web/src/components/upload-form.tsx`: accept `defaultTask?: Task`.
  - `web/src/components/upload-form.test.tsx`: create if missing.

**Interfaces:**
- Consumes: `PageHeader`, `Panel` (Task 6).
- Produces:
  - `<Drawer open; onOpenChange; title; children>`: a Base UI Dialog styled as a right-side sheet, with a labelled close button.
  - `UploadForm` `defaultTask` preselects the task.
- [ ] **Step 1: Write a failing test:** `UploadForm` with `defaultTask="asr"` has the `asr` option selected.
- [ ] **Step 2: Run** `npm run test`. Expected: FAIL.
- [ ] **Step 3: Implement the pages.**
  - **Datasets list:** a table with columns name, task, latest version, size, `commercial_ok`, eval-only. An "Upload dataset" button opens the `Drawer` with `UploadForm`. The upload keeps running and its progress stays visible in the drawer if it is closed and reopened (keep the form mounted and hide it with CSS).
  - **Version page:** restyled panels; the audio player stays inline per sample row.
- [ ] **Step 4: Run** `npm run test && npm run typecheck`. Expected: PASS.
- [ ] **Step 5: Commit** as `feat(web): datasets restyle and upload drawer`.

### Task 15: UI verification pass and docs

**Files:**
- Modify: any web file needing fixes found here; `README.md` (one short "Model catalog" subsection: where entries live, how to add one, which tests validate them).
- Create: `docs/superpowers/ui-checks/2026-11-23/` with the screenshots (PNG, committed as review evidence) and `checklist.md`.

- [ ] **Step 1: Seed data.** Start the app with `run-local.bat sqlite`. Seed by uploading the README's `smoke.jsonl`, creating recipe `smoke` (backend `fake`), and running it to `done`. Promote it, run a second recipe variant with `steps: 30`, and leave one run cancelled.
- [ ] **Step 2: Take screenshots** with Playwright (`npx playwright@1 screenshot`, or a short script run from the scratchpad directory) of:
  - `/`, `/catalog`, `/catalog/whisper-large-v3-turbo-bn`, `/datasets`, a dataset version, `/recipes/new?catalog=qwen3-4b-instruct-2507`, `/runs`, a run page, `/runs/compare?ids=<two ids>`, `/registry`;
  - each in dark and light, at 1440×900 and 375×812.
- [ ] **Step 3: Check every screenshot** and record pass or fail per item in `checklist.md`:
  - text contrast ≥ 4.5:1 (spot-check the muted-on-slate and saffron-on-ink pairs with a contrast calculation);
  - keyboard focus visible (tab through `/runs` and the palette);
  - no horizontal page scroll at 375 px (`document.documentElement.scrollWidth <= 375`);
  - reduced motion (emulate `prefers-reduced-motion: reduce`, no transitions);
  - icon buttons have accessible names;
  - status never shown by color alone.
- [ ] **Step 4: Fix** every failure, re-screenshot it, and update `checklist.md`.
- [ ] **Step 5: Run** `uv run pytest -q`, then `npm run test && npm run typecheck && npm run build` in `web/`. Expected: all pass. Paste the summary lines into `checklist.md`.
- [ ] **Step 6: Commit** the fixes, screenshots, `checklist.md` and `README.md` as `docs: UI verification for catalog and redesign`.
