# Training Backends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make LLM, STT and TTS models trainable for real. Each run converts the task's standard dataset into its model's format (cached), then trains, evaluates and exports through four backends: `unsloth`, `hf`, `nemo`, `f5`.

**Architecture:**
- Shared, dependency-light code goes in `app.core`:
  - `prepare` (cache contract)
  - `text` (`bn_normalize`)
  - `metrics`
  - `evaluation` (eval loops over injected model functions)
  - `backend_kit` (phase CLI, reporting, signals, checkpoints)
- Heavy shared helpers live in `app.core.backend_kit.<sub>` submodules. Only backends import them.
- Each backend is a separate uv project under `backend/workers/backends/<name>` with its own Linux lockfile. It contains one converter per model family plus train/eval/export modules. Each module imports heavy libraries inside functions, so light tests run without torch.
- The worker and API change only slightly:
  - MLflow params for prepared hashes
  - `HF_HOME`
  - three read-only endpoints
  - two UI panels

**Tech Stack:**
- Python 3.12, pydantic 2, pyarrow, numpy/soundfile/soxr
- torch 2.7.1+cu128, transformers, TRL, Unsloth, NeMo, f5-tts, parler-tts, CTranslate2
- FastAPI
- Next.js + vitest

**Spec:** `docs/superpowers/specs/2026-11-23-training-backends-design.md` (parent: `docs/superpowers/specs/2026-11-23-bangla-voice-mlops-design.md`)

**Execution order:** Tasks 1–9, then 31–32 (per-run splits, spec §15), then 28–30 (warm start, spec §14), then 10–27. Backend tasks carry **Warm start** lines that implement spec §14.2–14.7 for that backend.

## Global Constraints

**Python, packaging and environments**
- Python `>=3.12,<3.13` everywhere.
- Base install of `bangla-voice-backend` stays `["pydantic>=2.7", "pyyaml"]`. New extras:
  - `prepare = ["pyarrow"]`
  - `audio = ["numpy", "soundfile", "soxr"]`
- `app.core.contract`, `recipes`, `catalog`, `prepare`, `text`, `metrics`, `evaluation` and `backend_kit` (the package `__init__`, not submodules) must import without loading `fastapi`, `sqlalchemy`, `pyarrow`, `mlflow`, `torch`, `transformers` or `numpy`.
- Every backend pins `torch==2.7.1` from the `https://download.pytorch.org/whl/cu128` index. Lockfiles are Linux-only: `[tool.uv] environments = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]`.
- Each new backend directory is added to `[tool.uv.workspace] exclude` in `backend/pyproject.toml`.
- Backend modules import torch, transformers, unsloth, nemo, f5_tts, parler_tts, ctranslate2 and datasets **inside functions only**. Light tests must run in the backend workspace venv without them.
- Gated repos need `HF_TOKEN`. The prepare-phase message is exactly `no access to <repo>: accept its terms on Hugging Face and set HF_TOKEN`.
- Weights cache: `HF_HOME=$DATA_ROOT/models/hf`.
- Audio converters wrap each `load_audio` call. A decode error skips the row with reason `undecodable audio: <ExceptionType>` (spec §10). Each audio converter task adds a light test for it.

**Prepare cache**
- Prepared cache path: `<prepared_root>/<dataset_hash>/<spec_hash>/`, where `spec_hash` = first 16 hex chars of sha256 over canonical JSON (`sort_keys=True, separators=(",", ":"), ensure_ascii=False`) of `PrepSpec.model_dump()`.
- Default `max_skip_share = 0.05`. Recipe override: `hparams.prepare.max_skip_share`.

**Training and eval runtime**
- NaN stop: 3 consecutive non-finite logged losses. Message `loss is NaN; lower learning_rate`.
- OOM: print `RuntimeError: CUDA out of memory` to stderr and exit 1.
- SIGTERM: checkpoint, then exit 143.
- A checkpoint counts only with a `.complete` marker file.
- Eval caps:
  - LLM: 200 generated prompts, greedy, 512 new tokens.
  - TTS: 20 synthesized sentences.
- Baseline metrics are prefixed `base/`. `hparams.eval.baseline: false` skips them.
- Export: formats are independent. Exit 0 if at least one succeeded.

**Recipe hyperparameter names (all backends)**
- `learning_rate`, `epochs`, `batch_size`, `grad_accum`, `warmup_steps`, `max_steps` (optional; overrides epochs, used by smoke tests), `seed`
- LLM: `max_seq_len`, `lora_r`, `lora_alpha`
- f5: `batch_size_frames`
- `eval` object (`baseline`, `max_samples`), `prepare` object (`max_skip_share`)

**Tests and commits**
- Local test command for the shared code: `cd backend && uv run pytest`.
- Local light test command for a backend: `bash scripts/test_backend.sh <name>` (Task 9).
- GPU and heavy tests run on the VM only: `bash scripts/test_backend.sh <name> --heavy`.
- Commit messages end with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## VM Gates

The executor cannot run CUDA work on Windows. These steps are marked **VM GATE**. At a VM GATE the executor:
1. pushes the branch,
2. stops and asks the user to run the listed command on the VM,
3. continues only after the user reports output.

If a VM GATE install check fails, do not work around it:
1. Record the exact error in the plan's progress notes.
2. Skip that backend's remaining tasks.
3. Continue with the next backend (spec §12).

## Review Focus

1. **Two runs prepare the same dataset × spec at once.** Expect one build; the other waits and gets a cache hit. Test: `test_concurrent_builds_share_one_result` (Task 1).
2. **Text with characters outside a fixed vocabulary** (MMS `ben` adapter, NeMo tokenizer, F5 vocab). Expect the row to be skipped with a reason naming the characters, not a crash or a silent `<unk>`. Tests: `test_mms_skips_out_of_vocab_rows` (Task 18), `test_nemo_skips_unk_rows` (Task 21), `test_f5_appends_new_chars_without_reordering` (Task 24).
3. **Resume after cancel when the last checkpoint save was interrupted.** Expect resume from the previous complete checkpoint. Test: `test_latest_checkpoint_ignores_incomplete` (Task 4).
4. **Eval audio name containing `../`, or a symlink pointing out of `eval/audio`.** Expect 404. Test: `test_eval_audio_rejects_traversal` (Task 7).
5. **ASR eval set with slice columns only on some rows** (e.g. `region` missing for half). Expect slice WER over rows that have the value, with no `wer/region=None` key. Test: `test_asr_eval_slices_skip_missing` (Task 5).

---

## File Structure

```
backend/
  pyproject.toml                         + extras prepare/audio, dev deps, workspace exclude
  app/core/
    contract.py                          + RunDir paths: prepared_refs, eval samples/audio; EvalSample, PreparedRef
    prepare.py                           PrepSpec, iter_rows, load_splits, prepared_dir, ensure_prepared
    text.py                              bn_normalize, BN_NORMALIZE_VERSION
    metrics.py                           wer, cer, bangla_script_share, has_markdown, has_emoji
    evaluation.py                        run_llm_eval, run_asr_eval, run_tts_eval, baseline cache
    backend_kit/__init__.py              main, Reporter, StopFlag, latest_checkpoint, mark_complete, NanGuard, max_skip_share
    backend_kit/hf_access.py             check_hf_access (huggingface_hub, lazy)
    backend_kit/audio.py                 load_audio, write_wav, duration_s (numpy/soundfile/soxr)
    backend_kit/hf.py                    ContractCallback (transformers)
    backend_kit/lightning.py             ContractLightningCallback (lightning)
    backend_kit/whisper.py               whisper converter + transcribe helper (transformers)
  app/routers/runs.py                    + /eval/samples, /eval/audio/{name}, /prepared
  workers/executor.py, tracking.py       + Forwarder.params, HF_HOME env
  workers/backends/
    unsloth/  hf/  nemo/  f5/
      pyproject.toml  capabilities.json  README.md
      mlops_backend/__main__.py          backend_kit.main(HANDLERS)
      mlops_backend/models.py            MODELS: dict[hf_repo, ModelInfo]
      mlops_backend/prepare/<family>.py  converters
      mlops_backend/train.py  eval.py  export.py
      tests/  (light by default; @pytest.mark.heavy for torch/GPU)
  tests/core/test_prepare.py test_text.py test_metrics.py test_evaluation.py test_backend_kit.py test_repo_catalog.py(+)
frontend/src/
  lib/api.ts, lib/types.ts               + getEvalSamples, getPrepared, evalAudioUrl
  components/prepared-card.tsx           Prepared data card
  components/eval-samples.tsx            LLM table / ASR diff / TTS players
  lib/word-diff.ts                       word-level diff for ASR
  app/runs/[id]/page.tsx                 wire both + base/ deltas
scripts/
  test_backend.sh                        light/heavy backend tests
  check_backend.sh                       VM install + CUDA import check
  smoke_backends.sh                      end-to-end tiny run per backend
  smoke/fixtures/                        tiny llm .jsonl, asr .zip, tts .zip
```

---

## Part A — Shared core

### Task 1: Prepare contract

**Files:**
- Modify: `backend/pyproject.toml` (extras `prepare = ["pyarrow"]`, `audio = ["numpy", "soundfile", "soxr"]`; add `soxr` to the `dev` group)
- Create: `backend/app/core/prepare.py`
- Modify: `backend/app/core/contract.py` (add `PreparedRef`, `RunDir.prepared_refs_path`, `write_prepared_refs`, `read_prepared_refs`)
- Modify: `backend/tests/core/test_import_isolation.py`
- Test: `backend/tests/core/test_prepare.py`

**Interfaces:**
- Produces:
  ```python
  class PrepSpec(BaseModel): family: str; version: int; params: dict
      def hash(self) -> str                       # 16 hex chars, canonical JSON (Global Constraints)
  class Skip(BaseModel): sample_id: str; split: str; reason: str
  class BuildResult(BaseModel): counts: dict[str, int]; skipped: list[Skip] = []; files: list[str] = []
  class PrepInfo(BaseModel):
      spec: PrepSpec; spec_hash: str; dataset_hash: str; counts: dict[str, int]
      skipped: list[Skip]; files: list[str]; built_at: float; build_s: float
      path: str; cache_hit: bool = False          # path/cache_hit set on return, not trusted from disk
  class PrepareError(RuntimeError)
  def iter_rows(ds: DatasetRef, split: str) -> Iterator[dict]
  def load_splits(ds: DatasetRef, splits: list[str]) -> dict[str, list[dict]]
  def prepared_dir(cfg: RunConfig, ds: DatasetRef, spec: PrepSpec) -> Path
  def ensure_prepared(cfg: RunConfig, ds: DatasetRef, spec: PrepSpec,
                      build: Callable[[dict[str, list[dict]], Path], BuildResult], *,
                      splits: list[str] | None = None, max_skip_share: float = 0.05) -> PrepInfo
  # contract.py
  class PreparedRef(BaseModel): role: str; info: PrepInfo-shaped dict   # stored as dict to keep contract.py free of prepare import
  RunDir.prepared_refs_path -> path / "prepared_refs.json"
  def write_prepared_refs(rd: RunDir, refs: list[dict]) -> None   # atomic
  def read_prepared_refs(rd: RunDir) -> list[dict]                 # [] when missing/invalid
  ```
- Row shape from `iter_rows`:
  - `llm-sft` rows: `messages` (decoded from `messages_json`) and `tools` (decoded from `tools_json`, `None` if null).
  - Audio rows: `audio_path` = `Path(ds.root) / "files" / row["file"]` as a string.
  - All other columns pass through unchanged. Rows are filtered by `row["split"] == split`.

- [ ] **Step 1: Write the failing tests** in `backend/tests/core/test_prepare.py`. Build a tiny manifest with pyarrow in a `tmp_path` fixture: 3 audio rows across `train`/`val`, plus one llm dataset with `messages_json`.

```python
def test_spec_hash_is_stable_and_order_independent():
    a = PrepSpec(family="f", version=1, params={"b": 1, "a": "বাংলা"})
    b = PrepSpec(family="f", version=1, params={"a": "বাংলা", "b": 1})
    assert a.hash() == b.hash() and len(a.hash()) == 16

def test_iter_rows_filters_split_and_resolves_audio(audio_ds):
    rows = list(iter_rows(audio_ds, "train"))
    assert {r["split"] for r in rows} == {"train"}
    assert Path(rows[0]["audio_path"]) == Path(audio_ds.root) / "files" / rows[0]["file"]

def test_iter_rows_decodes_messages(llm_ds):
    row = next(iter_rows(llm_ds, "train"))
    assert row["messages"][0]["role"] in {"system", "user"}

def test_miss_builds_then_hit_reuses(cfg, audio_ds):
    calls = []
    def build(rows, out):
        calls.append(1); (out / "x.txt").write_text("ok")
        return BuildResult(counts={k: len(v) for k, v in rows.items()}, files=["x.txt"])
    first = ensure_prepared(cfg, audio_ds, SPEC, build)
    second = ensure_prepared(cfg, audio_ds, SPEC, build)
    assert calls == [1] and first.cache_hit is False and second.cache_hit is True
    assert Path(second.path) == prepared_dir(cfg, audio_ds, SPEC)
    assert (Path(second.path) / "prepared.json").exists()

def test_build_exception_leaves_no_target_and_no_tmp(cfg, audio_ds):
    def boom(rows, out): (out / "half").write_text("x"); raise ValueError("bad")
    with pytest.raises(ValueError):
        ensure_prepared(cfg, audio_ds, SPEC, boom)
    parent = prepared_dir(cfg, audio_ds, SPEC).parent
    assert not prepared_dir(cfg, audio_ds, SPEC).exists()
    assert not list(parent.glob("*.tmp-*"))

def test_stale_tmp_dirs_are_removed(cfg, audio_ds):
    target = prepared_dir(cfg, audio_ds, SPEC)
    stale = target.with_name(target.name + ".tmp-99999"); stale.mkdir(parents=True)
    ensure_prepared(cfg, audio_ds, SPEC, ok_build)
    assert not stale.exists()

def test_skip_share_above_threshold_fails_and_caches_nothing(cfg, audio_ds):
    def skip_all(rows, out):
        return BuildResult(counts={}, skipped=[Skip(sample_id=r["sample_id"], split=s, reason="too long")
                                               for s, rs in rows.items() for r in rs])
    with pytest.raises(PrepareError, match=r"100\.0% of rows skipped .*too long"):
        ensure_prepared(cfg, audio_ds, SPEC, skip_all)
    assert not prepared_dir(cfg, audio_ds, SPEC).exists()

def test_concurrent_builds_share_one_result(cfg, audio_ds, tmp_path):
    # two subprocesses call ensure_prepared with a build that sleeps 1 s and appends to a counter file
    ...
    assert counter.read_text().count("built") == 1
    assert sorted(hits) == [False, True]
```

The `PrepareError` message format is `"{pct:.1f}% of rows skipped (limit {limit:.1f}%): {top3}"`. `top3` = the three most common reasons, as `"<reason> ×<n>"` joined by `"; "`.

Also extend `test_import_isolation.py`:
- Import `app.core.prepare`.
- Add `"torch", "transformers", "numpy"` to `HEAVY`.

- [ ] **Step 2: Run tests, expect failures**

Run: `cd backend && uv run pytest tests/core/test_prepare.py tests/core/test_import_isolation.py -v`
Expected: FAIL (`ModuleNotFoundError: app.core.prepare`).

- [ ] **Step 3: Implement `app/core/prepare.py`**
- `pyarrow.parquet` is imported inside `iter_rows`.
- Use `ensure_prepared` from spec §4.2:
  - lock file `<target>.lock` via `fcntl.flock` on POSIX, `msvcrt.locking` on Windows (small helper `_FileLock` context manager in the same module)
  - check again for a hit after taking the lock
  - remove `<target>.tmp-*`
  - build into `<target>.tmp-<pid>`
  - compute the skip share as `len(skipped) / max(1, total_input_rows)`
  - write `prepared.json` (PrepInfo without `path`/`cache_hit`)
  - `os.replace` to the target
- On any exception, `shutil.rmtree` the tmp dir, then re-raise.
- `splits` defaults to `ds.splits`.

- [ ] **Step 4: Implement `PreparedRef` helpers in `contract.py`.** Reuse `_atomic_write` semantics for a JSON list. Add `"prepared_refs_path"` next to the other `RunDir` paths.

- [ ] **Step 5: Run tests, expect pass**

Run: `cd backend && uv run pytest tests/core -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock backend/app/core/prepare.py backend/app/core/contract.py backend/tests/core
git commit -m "feat(core): prepare cache contract with atomic build and skip threshold"
```

### Task 2: `bn_normalize`

**Files:**
- Create: `backend/app/core/text.py`
- Test: `backend/tests/core/test_text.py`

**Interfaces:**
- Produces:
  - `BN_NORMALIZE_VERSION: int = 1`
  - `bn_normalize(text: str, *, mode: Literal["asr", "tts"]) -> str`
  - `number_to_bangla_words(n: int) -> str`

- [ ] **Step 1: Write table-driven failing tests**

```python
CASES_BOTH = [
    ("কি\u200bছু", "কিছু"),                      # zero-width space removed
    ("র\u200dযাব", "র\u200dযাব"),                # ZWJ kept between Bangla letters
    ("আমি   যাব।", "আমি যাব।"),                 # whitespace collapsed
    ("“হ্যাঁ”", "\"হ্যাঁ\""),                     # quotes unified
    ("যাব|", "যাব।"),                           # pipe-as-danda unified
]
CASES_ASR = [
    ("Hello, আমি যাব।", "hello আমি যাব"),        # lowercase latin, strip punctuation
    ("১২৩ টাকা", "১২৩ টাকা"),                   # digits kept, not expanded
]
CASES_TTS = [
    ("১২৩", "একশো তেইশ"),
    ("123", "একশো তেইশ"),
    ("৳৫০০", "পাঁচশো টাকা"),
    ("২৫%", "পঁচিশ শতাংশ"),
    ("১০:৩০", "দশটা ত্রিশ মিনিট"),
    ("০৫/০১/২০২৬", "পাঁচ জানুয়ারি দুই হাজার ছাব্বিশ"),
    ("৩.৫", "তিন দশমিক পাঁচ"),
    ("আমি যাব।", "আমি যাব।"),                   # sentence punctuation kept
]
def test_nfc(): assert bn_normalize("\u0995\u09bc", mode="asr") == unicodedata.normalize("NFC", "\u0995\u09bc")
def test_number_words_range():
    assert number_to_bangla_words(0) == "শূন্য"
    assert number_to_bangla_words(100000) == "এক লাখ"
    assert number_to_bangla_words(10000000) == "এক কোটি"
```

`asr` mode converts Latin digits to Bengali digits, so `"123"` becomes `"১২৩"`. Add that case.

- [ ] **Step 2: Run, expect FAIL**

`cd backend && uv run pytest tests/core/test_text.py -v`

- [ ] **Step 3: Implement**
- Rule order:
  1. NFC
  2. zero-width removal, keeping U+200C/U+200D only when both neighbours are in U+0980–U+09FF
  3. quotes and danda
  4. digits
  5. mode-specific rules
  6. whitespace collapse
- `number_to_bangla_words` uses the full 0–99 Bangla word table (each of 0–99 has an irregular word). Then hundreds `শো` (with `একশো`, `দুইশো`…), thousands `হাজার`, lakhs `লাখ`, crores `কোটি`, recursively for crore multiples.
- Month names: `জানুয়ারি … ডিসেম্বর`.
- Years read as a plain number.

- [ ] **Step 4: Run, expect PASS**

- [ ] **Step 5: Commit**

`git commit -m "feat(core): bn_normalize for asr and tts"`

### Task 3: Rule metrics

**Files:**
- Create: `backend/app/core/metrics.py`
- Test: `backend/tests/core/test_metrics.py`

**Interfaces:**
- Produces:
  - `wer(refs: list[str], hyps: list[str]) -> float`: corpus-level, total edits / total ref words
  - `cer(refs, hyps) -> float`: characters, spaces excluded
  - `sample_wer(ref: str, hyp: str) -> float`
  - `bangla_script_share(text: str) -> float`: Bangla letters / all letters, `0.0` if no letters
  - `has_markdown(text: str) -> bool`
  - `has_emoji(text: str) -> bool`

- [ ] **Step 1: Failing tests**

```python
def test_wer_known(): assert wer(["আমি ভাত খাই"], ["আমি খাই"]) == pytest.approx(1/3)
def test_wer_corpus_weighted(): assert wer(["a b", "c d e f"], ["a b", "c x e f"]) == pytest.approx(1/6)
def test_cer_ignores_spaces(): assert cer(["ab cd"], ["ab ce"]) == pytest.approx(1/4)
def test_wer_empty_ref_counts_insertions(): assert sample_wer("", "x y") == 2.0
def test_script_share(): assert bangla_script_share("আমি ok") == pytest.approx(3/5)
@pytest.mark.parametrize("t", ["**bold**", "# Title", "- item\n- item", "`code`", "[a](b)"])
def test_markdown(t): assert has_markdown(t)
def test_plain_bangla_not_markdown(): assert not has_markdown("দাম ৫০০ টাকা - ঠিক আছে")
def test_emoji(): assert has_emoji("ঠিক আছে 😊") and not has_emoji("ঠিক আছে")
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement.** Levenshtein over token lists, O(n·m) DP, no dependencies. Letters means `str.isalpha()` code points. Emoji = code points in U+1F300–U+1FAFF, U+2600–U+27BF, U+1F600–U+1F64F, U+1F680–U+1F6FF.

- [ ] **Step 4: Run, expect PASS. Step 5: Commit** `feat(core): rule metrics (wer, cer, script share, markdown, emoji)`

### Task 4: Backend runtime kit

**Files:**
- Create: `backend/app/core/backend_kit/__init__.py`
- Create: `backend/app/core/backend_kit/hf_access.py`
- Create: `backend/app/core/backend_kit/audio.py`
- Modify: `backend/app/core/contract.py`:
  - `RunDir.eval_samples_path` (`eval/samples.jsonl`)
  - `RunDir.eval_audio_dir` (`eval/audio`)
  - `append_eval_sample(rd, sample: dict)`
  - `read_eval_samples(rd, offset: int, limit: int) -> tuple[list[dict], int]` (returns rows and the total count)
- Test: `backend/tests/core/test_backend_kit.py`, `backend/tests/core/test_audio.py`

**Interfaces:**
- Produces:
  ```python
  Handler = Callable[[RunDir, RunConfig, "Reporter"], int]
  def main(handlers: dict[Phase, Handler], argv: list[str] | None = None) -> NoReturn
      # argv: ["<phase>", "--run-dir", DIR]; unknown phase -> exit 2
      # exception -> status.message = f"{type(e).__name__}: {first line}", traceback to stderr, exit 1
      # torch.cuda.OutOfMemoryError (matched by class name "OutOfMemoryError") -> print
      #   "RuntimeError: CUDA out of memory" to stderr, exit 1
      # PrepareError -> status.message = str(e), exit 1
  class Reporter:
      def __init__(self, rd: RunDir, clock: Callable[[], float] = time.time)
      def status(self, phase: Phase, step: int = 0, total_steps: int | None = None, message: str = "") -> None
          # writes at most once per 1.0 s unless phase changed or step == total_steps; eta from step rate
      def metric(self, step: int, split: str, name: str, value: float) -> None
  class StopFlag:                       # SIGTERM handler sets .requested; install() registers it
      requested: bool; def install(self) -> "StopFlag"
  class NanGuard:                       # feed(loss) -> raises TrainingDiverged after 3 consecutive non-finite
      def feed(self, loss: float) -> None
  class TrainingDiverged(RuntimeError)  # message "loss is NaN; lower learning_rate"
  def mark_complete(ckpt_dir: Path) -> None          # writes ckpt_dir/".complete"
  def latest_checkpoint(rd: RunDir) -> Path | None   # newest (by trailing int in name) dir under checkpoints/ with .complete
  def max_skip_share(cfg: RunConfig) -> float         # hparams.prepare.max_skip_share or 0.05
  def eval_opts(cfg: RunConfig) -> tuple[bool, int | None]   # (baseline, max_samples)
  # hf_access.py
  def check_hf_access(repo: str) -> None   # huggingface_hub.auth_check; on GatedRepoError/401 raise PrepareError(exact message)
  # audio.py
  def load_audio(path: str | Path, sr: int) -> "np.ndarray"   # mono float32, resampled with soxr
  def write_wav(path: Path, audio: "np.ndarray", sr: int) -> None   # PCM_16
  def duration_s(path: str | Path) -> float                   # soundfile.info
  ```

- [ ] **Step 1: Failing tests** (`test_backend_kit.py`)

```python
def test_main_dispatches_and_exits_with_handler_code(run_dir_with_config): ...  # handler returns 7 -> SystemExit(7)
def test_main_exception_writes_status_message(run_dir_with_config):
    def bad(rd, cfg, rep): raise ValueError("first line\nsecond")
    with pytest.raises(SystemExit) as e: main({Phase.train: bad}, ["train", "--run-dir", str(run_dir_with_config)])
    assert e.value.code == 1
    assert read_status(RunDir(run_dir_with_config)).message == "ValueError: first line"
def test_main_oom_prints_marker(run_dir_with_config, capsys):
    class OutOfMemoryError(RuntimeError): ...
    def oom(rd, cfg, rep): raise OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB")
    with pytest.raises(SystemExit): main({Phase.train: oom}, ["train", "--run-dir", str(run_dir_with_config)])
    assert "RuntimeError: CUDA out of memory" in capsys.readouterr().err
def test_reporter_rate_limits(tmp_path):
    t = [0.0]; rep = Reporter(RunDir(tmp_path), clock=lambda: t[0])
    rep.status(Phase.train, 1, 10); t[0] = 0.5; rep.status(Phase.train, 2, 10)
    assert read_status(RunDir(tmp_path)).step == 1
    t[0] = 1.2; rep.status(Phase.train, 3, 10)
    st = read_status(RunDir(tmp_path)); assert st.step == 3 and st.eta_s == pytest.approx(2.8, rel=0.05)
def test_reporter_writes_final_step_immediately(tmp_path): ...
def test_nan_guard_trips_on_third():
    g = NanGuard(); g.feed(float("nan")); g.feed(float("inf"))
    with pytest.raises(TrainingDiverged, match="loss is NaN; lower learning_rate"): g.feed(float("nan"))
def test_nan_guard_resets_on_finite(): ...
def test_latest_checkpoint_ignores_incomplete(tmp_path):
    rd = RunDir(tmp_path); rd.ensure()
    for n in (100, 200): (rd.checkpoints_dir / f"checkpoint-{n}").mkdir()
    mark_complete(rd.checkpoints_dir / "checkpoint-100")
    assert latest_checkpoint(rd).name == "checkpoint-100"
def test_stop_flag_on_sigterm(): ...  # POSIX only: pytest.mark.skipif(sys.platform == "win32")
def test_max_skip_share_override(): ...
def test_eval_samples_roundtrip(tmp_path): ...  # append 3, read offset=1 limit=1 -> ([row2], 3)
```

`test_audio.py` (uses the `audio` extra, present in the dev env):
- `test_load_audio_resamples_and_downmixes`: write 44.1 kHz stereo, load at 16 kHz, check shape and dtype.
- `test_write_wav_roundtrip`.

- [ ] **Step 2: Run, expect FAIL.** `cd backend && uv run pytest tests/core/test_backend_kit.py tests/core/test_audio.py -v`

- [ ] **Step 3: Implement.**
- `main` uses `argparse`, then `read_config`.
- The exception branch writes status with the phase currently being run.
- Extend `test_import_isolation.py` to import `app.core.backend_kit`, still with no heavy modules. `audio.py` and `hf_access.py` are submodules and are not imported by `__init__`.

- [ ] **Step 4: Run, expect PASS. Step 5: Commit** `feat(core): backend_kit runtime (phase CLI, reporter, signals, checkpoints, audio helpers)`

### Task 5: Shared eval loops

**Files:**
- Create: `backend/app/core/evaluation.py`
- Test: `backend/tests/core/test_evaluation.py`

**Interfaces:**
- Consumes: Task 3 metrics, Task 2 `bn_normalize`, Task 4 `append_eval_sample`, `Reporter`.
- Produces (model calls are injected, so backends supply only the model-specific function):
  ```python
  def run_llm_eval(rd, rep, rows: list[dict], generate: Callable[[list[dict]], str], *,
                   loss: float | None, prefix: str = "", max_samples: int = 200) -> dict[str, float]
      # rows: llm rows (messages); prompt = messages up to the last assistant turn; reference = last assistant content
      # metrics: eval_loss, perplexity (=exp(loss)) when loss given; bangla_script_share, markdown_rate,
      #          emoji_rate, avg_reply_chars over generated replies. Samples written only when prefix == "".
  def run_asr_eval(rd, rep, rows: list[dict], transcribe: Callable[[list[str]], list[str]], *,
                   prefix: str = "", batch_size: int = 16, max_samples: int | None = None) -> dict[str, float]
      # wer, cer after bn_normalize(mode="asr"); "wer/<col>=<value>" for col in (region, channel, noise)
      # rtf = wall time / total audio seconds (rows carry duration_s from the manifest)
  def run_tts_eval(rd, rep, rows: list[dict], synthesize: Callable[[str, dict], tuple["np.ndarray", int]], *,
                   loss: float | None, prefix: str = "", max_samples: int = 20) -> dict[str, float]
      # writes eval/audio/<sample_id>.wav via backend_kit.audio.write_wav (lazy import); eval_loss, rtf
  def baseline_metrics(cfg: RunConfig, ds: DatasetRef, eval_key: dict, compute: Callable[[], dict[str, float]]) -> dict[str, float]
      # cache file: <prepared_root>/_baselines/<sha16(base_model, ds.hash, eval_key)>.json; returns keys prefixed "base/"
  def merge_results(rd: RunDir, *parts: dict[str, float]) -> None    # writes eval/results.json (EvalResults)
  ```

- [ ] **Step 1: Failing tests** (stub `generate`/`transcribe`/`synthesize`)

```python
def test_llm_eval_rule_metrics(rd, rep):
    rows = [{"sample_id": "a", "messages": [{"role": "user", "content": "দাম?"}, {"role": "assistant", "content": "৫০০ টাকা"}]}]
    m = run_llm_eval(rd, rep, rows, lambda msgs: "**দাম** ৫০০ টাকা 😊", loss=0.5)
    assert m["markdown_rate"] == 1.0 and m["emoji_rate"] == 1.0
    assert m["perplexity"] == pytest.approx(math.exp(0.5))
    s = read_eval_samples(rd, 0, 10)[0][0]
    assert s == {"sample_id": "a", "prompt": rows[0]["messages"][:1], "reference": "৫০০ টাকা", "reply": "**দাম** ৫০০ টাকা 😊"}
def test_llm_eval_caps_samples(rd, rep): ...   # 250 rows, generate called 200 times
def test_asr_eval_normalizes_before_scoring(rd, rep):
    rows = [{"sample_id": "a", "audio_path": "x", "text": "আমি যাব।", "duration_s": 2.0}]
    m = run_asr_eval(rd, rep, rows, lambda paths: ["আমি যাব"])
    assert m["wer"] == 0.0
def test_asr_eval_slices_skip_missing(rd, rep):
    rows = [{"sample_id": "a", "text": "x", "duration_s": 1, "audio_path": "a", "region": "dhaka"},
            {"sample_id": "b", "text": "y", "duration_s": 1, "audio_path": "b", "region": None}]
    m = run_asr_eval(rd, rep, rows, lambda p: ["x", "z"])
    assert m["wer/region=dhaka"] == 0.0 and not any("None" in k for k in m)
def test_prefix_applied_and_no_samples_for_baseline(rd, rep): ...  # prefix="base/" -> keys start "base/", samples.jsonl absent
def test_tts_eval_writes_wavs(rd, rep): ...   # synthesize returns np.zeros(24000), 24000 -> eval/audio/a.wav exists, rtf present
def test_baseline_cached(cfg, ds):
    calls = []
    f = lambda: calls.append(1) or {"wer": 0.4}
    assert baseline_metrics(cfg, ds, {"k": 1}, f) == {"base/wer": 0.4}
    baseline_metrics(cfg, ds, {"k": 1}, f); assert calls == [1]
```

- [ ] **Step 2: Run, expect FAIL. Step 3: Implement.** No numpy import at module level (the type hint is a string). **Step 4: PASS. Step 5: Commit** `feat(core): shared llm/asr/tts eval loops and baseline cache`

### Task 6: Worker — prepared hashes to MLflow, `HF_HOME`

**Files:**
- Modify: `backend/workers/executor.py`:
  - add `params(self, params: dict) -> None` to the `Forwarder` protocol and to `NullForwarder`
  - after a successful `prepare` phase, forward `{"prepared.<role>.spec_hash": ..., "prepared.<role>.cache_hit": ...}` from `read_prepared_refs`
  - `_run_phase` env adds `HF_HOME=str(settings.hf_home)`, unless `HF_HOME` is already set in the environment
- Modify: `backend/workers/tracking.py`: `MlflowForwarder.params` logs a param batch (same truncation as `start`)
- Modify: `backend/app/settings.py`: `hf_home` property, `data_root / "models" / "hf"`
- Modify: `.env.example`: add `HF_TOKEN=` with comment `# needed for gated weights: IndicF5, Gemma 3, IndicConformer`
- Test: `backend/tests/workers/test_executor.py`, `backend/tests/workers/test_tracking.py`, `backend/tests/app/test_settings.py`

- [ ] **Step 1: Failing tests**
- `test_prepared_refs_forwarded_after_prepare`:
  - Make the fake backend's `prepare` write `prepared_refs.json` when `hparams.write_prepared_refs` is true. This adds that hparam to the fake backend schema.
  - Use a recording forwarder and assert `params` received `prepared.train.spec_hash`.
- `test_hf_home_passed_to_backend`: the fake backend's `prepare` echoes `os.environ["HF_HOME"]` to stdout; assert `stdout.log` contains `models/hf`.
- `test_mlflow_params_batch`: use the existing fake client pattern.

- [ ] **Step 2: FAIL. Step 3: Implement. Step 4: PASS** (`cd backend && uv run pytest tests/workers workers/backends/fake/tests -v`). **Step 5: Commit** `feat(worker): forward prepared-data hashes to MLflow; set HF_HOME for backends`

### Task 7: API — prepared refs, eval samples, eval audio

**Files:**
- Modify: `backend/app/routers/runs.py`
- Test: `backend/tests/app/test_runs_api.py`

**Interfaces:**
- Produces (the frontend in Task 8 relies on these shapes):
  - `GET /api/runs/{id}/prepared` returns `{"refs": list[dict]}`
  - `GET /api/runs/{id}/eval/samples?offset=0&limit=50` returns `{"items": list[dict], "total": int}`. `limit` is between 1 and 500.
  - `GET /api/runs/{id}/eval/audio/{name}` returns a `FileResponse` with `audio/wav`. It returns 404 unless `name` matches `^[A-Za-z0-9._-]+\.wav$` and the resolved path is inside `eval/audio` (use the existing `inside()` helper on the resolved path).

- [ ] **Step 1: Failing tests**

```python
async def test_prepared_refs_empty_then_present(client, run_with_dir): ...
async def test_eval_samples_paged(client, run_with_dir): ...   # 3 samples, offset=1 limit=1 -> 1 item, total 3
async def test_eval_audio_serves_wav(client, run_with_dir): ...
@pytest.mark.parametrize("name", ["..%2Fconfig.json", "../config.json", "a.txt", "link.wav"])
async def test_eval_audio_rejects_traversal(client, run_with_dir, name): ...
    # "link.wav" is a symlink inside eval/audio pointing at config.json (skip symlink case on Windows if os.symlink raises)
```

- [ ] **Step 2: FAIL. Step 3: Implement. Step 4: PASS. Step 5: Commit** `feat(api): prepared refs, eval samples and eval audio endpoints`

### Task 8: UI — Prepared data card, Eval samples panel, baseline deltas

**Files:**
- Modify: `frontend/src/lib/types.ts`: `PreparedRef`, `EvalSamplesPage`
- Modify: `frontend/src/lib/api.ts`: `getPrepared(id)`, `getEvalSamples(id, offset, limit)`, `evalAudioUrl(id, name)`
- Create: `frontend/src/lib/word-diff.ts` + `word-diff.test.ts`
- Create: `frontend/src/components/prepared-card.tsx` + test
- Create: `frontend/src/components/eval-samples.tsx` + test
- Modify: `frontend/src/app/runs/[id]/page.tsx`:
  - the Eval tab shows a metrics table with a `base/` column and a delta
  - then `<EvalSamples task=… runId=…>`
  - the Config tab gets `<PreparedCard refs=…>`

**Interfaces:**
- Consumes: Task 7 response shapes.
- Produces:
  - `wordDiff(ref: string, hyp: string): {op: "eq" | "del" | "ins" | "sub", ref?: string, hyp?: string}[]`
  - `<PreparedCard refs: PreparedRef[]>`: per ref, shows the family, `spec_hash`, a cache badge "cache hit" or "built", counts per split, and the first 50 skipped rows with reasons. When there are no refs it shows "Prepared data appears after the prepare phase."
  - `<EvalSamples runId task>`:
    - `llm-sft`: table of prompt (last user turn), reference, reply
    - `asr`: table sortable by WER, with a diff (`del` struck through, `ins` underlined, `sub` shows both)
    - `tts`: list of text + `<audio controls src=evalAudioUrl(...)>`
    - Paged 50 rows at a time with a "Load more" button.
  - `metricRows(metrics): {name, value, base?: number, delta?: number}[]` in `frontend/src/lib/metrics.ts`. It pairs `x` with `base/x` and hides `base/*` from its own row.

- [ ] **Step 1: Failing vitest tests**
  - `wordDiff("আমি ভাত খাই", "আমি খাই")` → `[eq আমি, del ভাত, eq খাই]`
  - `metricRows({wer: 0.2, "base/wer": 0.3})` → `[{name: "wer", value: 0.2, base: 0.3, delta: -0.1}]`
  - PreparedCard renders "cache hit" and a skipped reason
  - EvalSamples (mocked api) renders audio elements for `tts` and the diff for `asr`

- [ ] **Step 2: FAIL.** Run from an upper-case `C:\` cwd: `cd C:\Users\BS23-DESKTOP-00038\Projects\BanglaVoiceBotFineTuned\frontend; npm test`

- [ ] **Step 3: Implement.** Use the existing `Table`, `Panel` and `Badge` components and the existing styling conventions.

- [ ] **Step 4: PASS + `npm run typecheck` + `npm run build`**

- [ ] **Step 5: Commit** `feat(web): prepared data card, eval samples panel, baseline deltas`

### Task 9: Backend scaffolding helpers

**Files:**
- Create: `scripts/test_backend.sh`
  - `test_backend.sh <name> [--heavy]`
  - Light mode runs: `uv run --project backend --extra server --extra audio --with pytest env PYTHONPATH=backend/workers/backends/<name> pytest backend/workers/backends/<name>/tests -m "not heavy"`
  - `--heavy` runs `uv run --project backend/workers/backends/<name> pytest backend/workers/backends/<name>/tests` (VM only)
- Create: `scripts/check_backend.sh`
  - `check_backend.sh <name>`
  - runs `uv sync --project backend/workers/backends/<name>`
  - then `uv run --project … python -m mlops_backend.check`, which prints `torch <ver> cuda <bool> capability <maj.min>` and the backend's key import versions, and exits 1 if CUDA is unavailable or an import fails
- Create: `backend/workers/backends/conftest_shared.py`
  - pytest fixtures reused by all four backends (each backend's `tests/conftest.py` does `from conftest_shared import *` via `pytest_plugins`)
  - `run_dir(tmp_path, task, base_model, hparams)` builds a RunDir + `config.json` with a tiny dataset manifest (pyarrow) of 4 rows
  - `tiny_audio_ds(sr=16000)` writes sine-wave WAVs
  - registers the `heavy` marker
- Modify: `scripts/check_vm.sh`: check `cmake`, `build-essential` (`gcc`, `make`) and `git` are on PATH (llama.cpp build for GGUF)

- [ ] **Step 1:** Write the scripts.
- [ ] **Step 2:** Smoke-check the light runner against the fake backend: `bash scripts/test_backend.sh fake`. Expected: the fake tests pass (they are light).
- [ ] **Step 3: Commit** `build: backend test/check scripts and shared backend test fixtures`

---

## Part B — `unsloth` backend

Model registry (`mlops_backend/models.py`):

```python
@dataclass(frozen=True)
class ModelInfo: task: str; family: str; gated: bool; chat_template: str | None = None
MODELS = {
  "Qwen/Qwen3-4B-Instruct-2507":             ModelInfo("llm-sft", "chat", False, "qwen3"),
  "google/gemma-3-4b-it":                    ModelInfo("llm-sft", "chat", True,  "gemma3"),
  "hishab/titulm-llama-3.2-3b-v2.0":         ModelInfo("llm-sft", "chat", False, "llama-3.2"),
  "openai/whisper-large-v3-turbo":           ModelInfo("asr", "whisper", False),
  "mozilla-ai/whisper-large-v3-turbo-bn":    ModelInfo("asr", "whisper", False),
}
```

### Task 10: unsloth scaffold + VM install check

**Files:**
- Create: `backend/workers/backends/unsloth/pyproject.toml`
  - name `mlops-backend-unsloth`
  - deps: `bangla-voice-backend[prepare,audio]` (path `../../..`, editable), `torch==2.7.1`, `unsloth`, `trl`, `transformers`, `datasets`, `peft`, `bitsandbytes`, `accelerate`, `huggingface_hub`, `ctranslate2`
  - torch index per Global Constraints; Linux-only `environments`
  - pytest in a dev group
- Create: `backend/workers/backends/unsloth/capabilities.json`
  - `name: "unsloth"`, `label: "Unsloth (LLM SFT, Whisper LoRA)"`
  - `llm-sft`: methods `lora`, `qlora`, `full`; export `merged-safetensors`, `gguf`
  - `asr`: methods `lora`; export `merged-safetensors`, `ct2`
  - JSON Schema properties:
    - `base_model` `enum` = that task's `MODELS` keys
    - `method.type` enum
    - `method.gguf_quant` enum `["q4_k_m", "q8_0", "f16"]`, default `q4_k_m`
    - hparams: names from Global Constraints with bounds `learning_rate` 1e-6–1e-2, `epochs` 1–50, `batch_size` 1–64, `grad_accum` 1–64, `max_seq_len` 256–8192, `lora_r` 4–256, `lora_alpha` 4–512, `max_steps` ≥ 1
    - `full` requires `hparams.gradient_checkpointing: true` (use JSON Schema `if`/`then`)
- Create: `mlops_backend/__init__.py`, `__main__.py` (`main(HANDLERS)`), `check.py`, `models.py`
- Create: `mlops_backend/phases.py`: `HANDLERS` maps each phase to a function that dispatches on `MODELS[cfg.base_model].family` (`chat` → `llm`, `whisper` → `asr`)
- Create: `tests/conftest.py`, `tests/test_capabilities.py`
- Modify: `backend/pyproject.toml`: add `"workers/backends/unsloth"` to `exclude`
- Create: `backend/app/core/backend_kit/hf.py` with `ContractCallback(TrainerCallback)`:
  - `__init__(self, rep: Reporter, stop: StopFlag, nan: NanGuard, total_steps: int)`
  - `on_log`: `rep.metric(state.global_step, "train", k, v)` for numeric `loss`, `learning_rate`, `grad_norm`; `eval_loss` goes to split `val`; `nan.feed(loss)`
  - `on_step_end`: `rep.status(...)`; when `stop.requested`, set `control.should_save = True` and `control.should_training_stop = True`
  - `on_save`: `mark_complete(<output_dir>/checkpoint-<step>)`
  - Transformers is imported at module level here; that's allowed for submodules.

- [ ] **Step 1: Failing light tests:**
  - `test_capabilities_schema_valid`: `jsonschema.Draft202012Validator.check_schema` for each task schema
  - `test_base_model_enum_matches_registry`
  - `test_phase_dispatch_unknown_model_fails`: a config with `base_model: "x/y"` exits 1 with message `unsupported base_model 'x/y' for backend unsloth`

- [ ] **Step 2: FAIL** (`bash scripts/test_backend.sh unsloth`). **Step 3: Implement.** Run `uv lock --project backend/workers/backends/unsloth`. If resolution fails because a dependency requires a different torch, set the torch version the resolver demands and add a comment `# torch differs from 2.7.1: <reason>`. Do the same in the spec's §13 open items. **Step 4: PASS.**

- [ ] **Step 5: Commit** `feat(unsloth): backend scaffold, capabilities, ContractCallback`

- [ ] **Step 6: VM GATE**: `bash scripts/check_backend.sh unsloth`. Expected: `torch 2.7.1+cu128 cuda True capability 12.0` and the unsloth version. On failure, follow the VM Gates rule.

### Task 11: `chat` converter

**Files:**
- Create: `backend/workers/backends/unsloth/mlops_backend/prepare/chat.py`
- Test: `backend/workers/backends/unsloth/tests/test_prepare_chat.py`

**Interfaces:**
- Produces:
  - `chat_spec(cfg) -> PrepSpec`: `family="chat"`, `version=1`, params `base_model`, `chat_template`, `max_seq_len`
  - `build_chat(tokenizer, max_seq_len: int) -> Callable[[dict[str, list[dict]], Path], BuildResult]`
    - writes `<out>/<split>.jsonl` with `{"input_ids": [...], "labels": [...]}`. Labels are `-100` everywhere except assistant-turn tokens. JSONL is used, not an HF dataset, so light tests need no `datasets`; the train step loads it with `datasets.load_dataset("json")`.
    - masking: render the conversation once with `tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)`; render each prefix ending before an assistant turn and each prefix ending after it; tokenize; mark label spans between those offsets
    - rows over `max_seq_len` tokens are skipped with reason `"{n} tokens > max_seq_len {max}"`
  - `prepare_llm(rd, cfg, rep) -> int`:
    - `check_hf_access` if gated
    - load the tokenizer with `AutoTokenizer.from_pretrained` (lazy)
    - `ensure_prepared` for the train dataset (`train`, `val`) and the eval dataset if set
    - `write_prepared_refs`

- [ ] **Step 1: Failing light tests** with a fake tokenizer class in the test (`apply_chat_template` joins `"<|role|>content"`; `__call__` splits on characters):
  - `test_labels_mask_non_assistant`: decode the labels ≠ -100 → exactly the assistant contents
  - `test_multi_turn_masks_each_assistant_turn`
  - `test_overlong_rows_skipped_with_reason`
  - `test_tools_passed_to_template`

  Heavy test (`@pytest.mark.heavy`): `test_qwen3_real_template_masks_assistant_only` with the real Qwen3 tokenizer.

- [ ] **Step 2: FAIL. Step 3: Implement. Step 4: light PASS. Step 5: Commit** `feat(unsloth): chat converter with assistant-only labels`

### Task 12: LLM training (LoRA / QLoRA / full) with resume

**Warm start (spec §14):**
- Before loading, call `init_source(cfg, rd)` (Task 28). If it returns `kind == "resume"`, keep the existing resume path. If `kind == "init"`, load per the spec §14.2 table:
  - `hf-peft` + lora/qlora: `FastLanguageModel.from_pretrained(<init path>)`. Unsloth loads base + adapter; make the adapter trainable.
  - `hf-peft` + full: load, `merge_and_unload()`.
  - `hf-full`: `from_pretrained(<init path>)` as the base, then add a new adapter for lora/qlora.
- Factor the choice into `plan_init(meta: dict | None, method: str) -> Literal["base", "attach_adapter", "merge_adapter", "full_weights", "full_then_adapter"]` with a light test per combination.
- Every complete checkpoint gets `write_init_meta` (format `hf-peft` for lora/qlora, `hf-full` for full).

**Files:**
- Create: `backend/workers/backends/unsloth/mlops_backend/llm.py`: `train_llm(rd, cfg, rep) -> int`, `_trainer_args(cfg, rd) -> dict`
- Test: `backend/workers/backends/unsloth/tests/test_llm_train.py`

**Interfaces:**
- Consumes: `read_prepared_refs` (train ref path), `ContractCallback`, `StopFlag`, `NanGuard`, `latest_checkpoint`.
- Behaviour:
  - Load with `FastLanguageModel.from_pretrained(base_model, max_seq_length=max_seq_len, load_in_4bit=(method=="qlora"), full_finetuning=(method=="full"))`.
  - LoRA/QLoRA: `FastLanguageModel.get_peft_model(model, r=lora_r, lora_alpha=lora_alpha, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], use_gradient_checkpointing="unsloth", random_state=seed)`.
  - `SFTTrainer` gets the pre-tokenized dataset (`dataset_kwargs={"skip_prepare_dataset": True}`) and a padding collator (`DataCollatorForSeq2Seq`, labels pad -100).
  - `TrainingArguments`:
    - `output_dir=rd.checkpoints_dir`
    - `save_steps=max(50, total//10)`, `save_total_limit=3`
    - `logging_steps=10`, `eval_strategy="steps"`, `eval_steps=save_steps`
    - `bf16=True`
    - `optim="adamw_8bit"` for full, else `"adamw_torch"`
    - `report_to=[]`, `seed`
  - `trainer.train(resume_from_checkpoint=str(latest_checkpoint(rd)) if cfg.resume and latest_checkpoint(rd) else None)`
  - Afterwards: `trainer.save_model(rd.checkpoints_dir / "final")`, `mark_complete(... / "final")`, and the tokenizer saved alongside.
  - If `stop.requested` after `train()` returns, return 143.
  - `TrainingDiverged` propagates to `main`, which writes the message.

- [ ] **Step 1: Failing light test:** `test_trainer_args_from_hparams`. `_trainer_args` maps hparams to `TrainingArguments` kwargs: `max_steps` overrides `num_train_epochs`, the 8-bit optimiser is used for full, `output_dir` is the checkpoints dir. Heavy test: `test_llm_train_20_steps_tiny` uses `hparams.max_steps=20` on Qwen3 with a 16-row fixture, then asserts `checkpoints/final/.complete` and `metrics.jsonl` has a `loss`.

- [ ] **Step 2: FAIL. Step 3: Implement. Step 4: light PASS. Step 5: Commit** `feat(unsloth): llm sft training with resume, stop and nan guard`

### Task 13: LLM eval

**Warm start:** if `cfg.init_from` is set, `compute` for `baseline_metrics` loads the init checkpoint through `plan_init` instead of the raw base model. The `eval_key` includes `{"init": [source_run, checkpoint]}`.

**Files:**
- Modify: `backend/workers/backends/unsloth/mlops_backend/llm.py`: `eval_llm(rd, cfg, rep) -> int`
- Test: `backend/workers/backends/unsloth/tests/test_llm_eval.py`

**Behaviour:**
- Eval rows come from `iter_rows`:
  - eval dataset: splits from its `DatasetRef.splits`
  - otherwise: train dataset `val`
- Loss: `Trainer.evaluate()` on the prepared eval/val JSONL.
- `generate`: `FastLanguageModel.for_inference(model)`, `tokenizer.apply_chat_template(prompt, add_generation_prompt=True)`, greedy, `max_new_tokens=512`.
- `run_llm_eval` runs for the fine-tuned model (from `checkpoints/final`).
- If `eval_opts(cfg)[0]` is true, `baseline_metrics(...)` runs with a `compute` that loads the base model the same way.
- `merge_results`.

- [ ] **Step 1: Failing light test:** `test_eval_rows_fallback_to_val` (no eval dataset → train `val` rows). Heavy: `test_llm_eval_writes_results_and_samples`.
- [ ] **Step 2–4.** **Step 5: Commit** `feat(unsloth): llm eval with loss, rule metrics and baseline`

### Task 14: LLM export (merged safetensors, GGUF)

**Files:**
- Create: `backend/workers/backends/unsloth/mlops_backend/export.py`: `export_all(rd, cfg, rep, exporters: dict[str, Callable[[Path], list[ExportArtifact]]]) -> int`, `llm_exporters(rd, cfg) -> dict`
- Test: `backend/workers/backends/unsloth/tests/test_export.py`

**Interfaces:**
- `export_all` runs every format in `cfg.export` (unknown format → logged failure). Each runs in its own try block; tracebacks go to stderr.
- It writes the manifest with only the successful artifacts.
- `rep.status(Phase.export, message=...)`:
  - all succeeded: `"exported: a, b"`
  - some failed: `"exported: a; failed: b"`
  - none succeeded: return 1
- `export_all` is reused by Tasks 15, 18–20, 23, 25: copy it into each backend? **No.** Move it into `app.core.backend_kit` as `export_all` (pure Python), in this task, with its light test in `backend/tests/core/test_backend_kit.py`.
- LLM exporters:
  - `merged-safetensors`: `model.save_pretrained_merged(dir, tokenizer, save_method="merged_16bit")`, artifact `serve_with="transformers"`
  - `gguf`: `model.save_pretrained_gguf(dir, tokenizer, quantization_method=method.gguf_quant)`, artifact `serve_with="llama-server"`. The artifact path is the produced `.gguf` file.

- [ ] **Step 1: Failing light tests** in `backend/tests/core/test_backend_kit.py`:
  - `test_export_all_partial_failure`: one exporter raises → exit 0, manifest has one artifact, status message `"exported: merged-safetensors; failed: gguf"`
  - `test_export_all_all_fail_exits_1`
  - `test_export_all_unknown_format_is_a_failure`

  Heavy: `test_llm_export_gguf_file_exists`.
- [ ] **Step 2–4.** **Step 5: Commit** `feat: shared export_all with partial failure; unsloth llm exporters`
- [ ] **Step 6: VM GATE**: `bash scripts/test_backend.sh unsloth --heavy`. Expected: chat + llm heavy tests pass.

### Task 15: Whisper converter + Whisper LoRA (unsloth)

**Warm start:** Whisper LoRA uses the same `plan_init` rules as Task 12. Write init meta on every complete checkpoint. The baseline loads the init checkpoint when set.

**Files:**
- Create: `backend/app/core/backend_kit/whisper.py` (shared by unsloth and hf; imports transformers):
  - `whisper_spec(cfg) -> PrepSpec`: family `whisper`, version 1, params `base_model`, `sample_rate=16000`, `bn_normalize_version`, `max_label_tokens=448`
  - `build_whisper(processor) -> build fn`:
    - writes `<out>/<split>/` as an HF `datasets.Dataset.save_to_disk` with `input_features` (from `processor.feature_extractor`; mel bins follow the model, 128 for large-v3-turbo) and `labels` (from `processor.tokenizer` with `language="bengali", task="transcribe"`, text = `bn_normalize(text, mode="asr")`)
    - skip reasons: `"duration {d:.1f} s > 30 s"` and `"{n} label tokens > 448"`
  - `make_transcriber(model, processor, device) -> Callable[[list[str]], list[str]]`: batched `generate(language="bengali", task="transcribe")`
  - `prepare_whisper(rd, cfg, rep) -> int`
- Create: `backend/workers/backends/unsloth/mlops_backend/asr.py`:
  - `train_asr`: `FastModel.from_pretrained(base_model, auto_model=WhisperForConditionalGeneration, whisper_language="Bengali", whisper_task="transcribe")`, `FastModel.get_peft_model(r=lora_r, lora_alpha=lora_alpha, target_modules=["q_proj","v_proj"])`, `Seq2SeqTrainer` with the same args and callback as Task 12
  - `eval_asr`: `run_asr_eval` + baseline
  - `asr_exporters`:
    - `merged-safetensors` (merge LoRA → `save_pretrained`)
    - `ct2`: `ct2-transformers-converter --model <merged> --output_dir <dir> --quantization float16 --copy_files tokenizer.json preprocessor_config.json`, run with `subprocess` from the same venv; `serve_with="faster-whisper"`. If `merged-safetensors` was not requested, merge into a temp dir first.
- Test: `backend/workers/backends/unsloth/tests/test_prepare_whisper.py` (heavy-marked; needs transformers), `tests/test_asr_light.py`

- [ ] **Step 1: Failing tests:**
  - light `test_whisper_spec_includes_normalizer_version`
  - heavy `test_whisper_build_skips_long_audio` (35 s sine → skipped with reason)
  - heavy `test_whisper_lora_20_steps`
- [ ] **Step 2–4. Step 5: Commit** `feat: shared whisper converter; unsloth whisper lora train/eval/export`
- [ ] **Step 6: VM GATE**: `bash scripts/test_backend.sh unsloth --heavy`

---

## Part C — `hf` backend

Registry:

```python
MODELS = {
  "openai/whisper-large-v3-turbo":        ModelInfo("asr", "whisper", False),
  "mozilla-ai/whisper-large-v3-turbo-bn": ModelInfo("asr", "whisper", False),
  "facebook/wav2vec2-xls-r-300m":         ModelInfo("asr", "ctc", False),
  "facebook/mms-1b-all":                  ModelInfo("asr", "ctc-mms", False),
  "ai4bharat/indic-parler-tts":           ModelInfo("tts", "parler", False),
}
```

### Task 16: hf scaffold + VM install check

Same shape as Task 10:
- Name `mlops-backend-hf`.
- Deps: `bangla-voice-backend[prepare,audio]`, `torch==2.7.1`, `torchaudio==2.7.1`, `transformers`, `datasets`, `accelerate`, `evaluate`, `huggingface_hub`, `ctranslate2`, `parler-tts @ git+https://github.com/huggingface/parler-tts`, `descript-audio-codec`.
- Capabilities:
  - `asr`: methods `full`, `adapter`; export `safetensors`, `ct2`, `adapter-safetensors`
  - `tts`: method `full`; export `safetensors`
- Schema `if`/`then` rules:
  - `adapter` only for `facebook/mms-1b-all`
  - `ct2` only for whisper models
  - `parler` hparams add `description_template` (string, default `"{speaker} speaks clearly in a studio-quality recording."`)
- Add to `exclude`.

- [ ] **Step 1–4** as Task 10 (tests `test_capabilities_schema_valid`, `test_base_model_enum_matches_registry`, `test_adapter_only_for_mms`, schema-level via jsonschema).
- [ ] **Step 5: Commit** `feat(hf): backend scaffold and capabilities`
- [ ] **Step 6: VM GATE**: `bash scripts/check_backend.sh hf`. If parler-tts pins a transformers version that conflicts with whisper/ctc needs, record it. Proceed with whisper/ctc and drop `tts` from `capabilities.json`, noting it in spec §13.

### Task 17: Whisper full fine-tune (hf)

**Warm start:** `hf-full` init → `WhisperForConditionalGeneration.from_pretrained(<init path>)`. Write init meta (`hf-full`) on every complete checkpoint, `final` included. The baseline loads the init checkpoint when set.

**Files:** `backend/workers/backends/hf/mlops_backend/whisper_full.py`
- `train`: `WhisperForConditionalGeneration.from_pretrained`, `Seq2SeqTrainer`, `gradient_checkpointing=True`, `ContractCallback`
- `eval`: `make_transcriber` + `run_asr_eval` + baseline
- exporters `safetensors` (`save_pretrained` + processor) and `ct2` (same command as Task 15)

**Test:** heavy `test_whisper_full_20_steps`; light `test_dispatch_whisper_family`.

- [ ] Steps 1–4. **Step 5: Commit** `feat(hf): whisper full fine-tune`

### Task 18: `ctc` converter + wav2vec2 / MMS adapter

**Warm start:**
- wav2vec2: when `cfg.init_from` is set, the vocab is `<init path>/vocab.json` in `fixed` mode (spec §14.6), and the model loads from the init path.
- MMS: init format `hf-mms-adapter` → load base, `init_adapter_layers()`, load `adapter.ben.safetensors`.
- Write init meta (`hf-full` or `hf-mms-adapter`) on every complete checkpoint.
- Light test: `test_ctc_vocab_from_init_when_warm_start`.

**Files:**
- Create: `backend/workers/backends/hf/mlops_backend/prepare/ctc.py`:
  - `build_ctc_vocab(texts: Iterable[str]) -> dict[str, int]`: characters of `bn_normalize(t, mode="asr")`, sorted; `" "` → `"|"`; then `[UNK]`, `[PAD]` last
  - `ctc_spec(cfg, vocab: dict) -> PrepSpec`: params include `sha16(vocab json)`
  - `build_ctc(vocab, mode: Literal["new", "fixed"]) -> build fn`:
    - writes `<out>/vocab.json` and `<out>/<split>.jsonl` of `{"audio_path", "duration_s", "text"}`
    - audio is resampled lazily at train time, so the light build needs no audio library
    - in `fixed` mode, rows with characters outside the vocab are skipped with reason `"characters not in vocab: <chars>"`
  - `mms_vocab(processor) -> dict`: `processor.tokenizer.set_target_lang("ben")` vocabulary (heavy)
- Create: `backend/workers/backends/hf/mlops_backend/ctc.py`:
  - wav2vec2: `Wav2Vec2ForCTC.from_pretrained(base, vocab_size=len(vocab), pad_token_id=…, ctc_loss_reduction="mean")`, `freeze_feature_encoder()`, `Trainer`, data collator that loads audio with `load_audio(path, 16000)` and pads
  - MMS: `Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="ben", ignore_mismatched_sizes=True)`, `model.init_adapter_layers()`, freeze all but adapter params (`model.freeze_base_model()`, unfreeze `adapter` params)
  - eval: greedy CTC decode transcriber → `run_asr_eval` + baseline
  - export:
    - `safetensors`: model + processor
    - `adapter-safetensors` (MMS): `safetensors.torch.save_file(adapter_state, "adapter.ben.safetensors")`
- Test: `tests/test_prepare_ctc.py` (light)

```python
def test_vocab_maps_space_to_pipe_and_adds_specials():
    v = build_ctc_vocab(["আমি যাব"])
    assert "|" in v and " " not in v and list(v)[-2:] == ["[UNK]", "[PAD]"]
def test_vocab_is_deterministic(): assert build_ctc_vocab(["খা", "আ"]) == build_ctc_vocab(["আ", "খা"])
def test_mms_skips_out_of_vocab_rows(tmp_path):
    res = build_ctc({"আ": 0, "|": 1, "[UNK]": 2, "[PAD]": 3}, "fixed")({"train": [row("আ আ"), row("আক")]}, tmp_path)
    assert res.counts["train"] == 1 and res.skipped[0].reason == "characters not in vocab: ক"
```

Heavy: `test_wav2vec2_20_steps`, `test_mms_adapter_only_trains_adapter` (asserts the count of params with `requires_grad` equals the adapter param count).

- [ ] Steps 1–4. **Step 5: Commit** `feat(hf): ctc converter, wav2vec2 and mms adapter training`

### Task 19: `parler` converter + Indic Parler-TTS training

**Warm start:** `hf-full` init → `ParlerTTSForConditionalGeneration.from_pretrained(<init path>)`. Write init meta on complete checkpoints. The baseline loads the init checkpoint when set.

**Files:**
- Create: `backend/workers/backends/hf/mlops_backend/prepare/parler.py`:
  - `describe(row, template) -> str`: formats `template` with `speaker`, `style` (default `""`, extra spaces collapsed)
  - `build_parler(feature_extractor, audio_encoder, prompt_tokenizer, description_tokenizer, template) -> build fn`:
    - per row: `load_audio(path, 44100)` → DAC codes via `audio_encoder.encode`
    - `prompt_input_ids` from `bn_normalize(text, mode="tts")`
    - `input_ids` from the description
    - saves `datasets` to disk per split
  - spec params: `base_model`, `sample_rate=44100`, `bn_normalize_version`, `description_template`
- Create: `backend/workers/backends/hf/mlops_backend/parler.py`:
  - train: `ParlerTTSForConditionalGeneration.from_pretrained`; loop is a `Trainer` subclass whose `compute_loss` calls `model(input_ids, attention_mask, prompt_input_ids, prompt_attention_mask, labels=codes)` (the training-script objective). Audio encoder frozen; `ContractCallback`.
  - eval: `eval_loss` from `Trainer.evaluate`; `synthesize(text, row)` = `model.generate(input_ids=desc, prompt_input_ids=prompt)` → `(audio, 44100)`; `run_tts_eval` (no baseline for TTS loss; baseline still on when enabled, for `rtf`)
  - export: `safetensors` (`save_pretrained` + tokenizer + feature extractor)
- Test: light `test_describe_template` (`describe({"speaker": "rina", "style": None}, DEFAULT) == "rina speaks clearly in a studio-quality recording."`); heavy `test_parler_20_steps_and_synthesize`.

- [ ] Steps 1–4. **Step 5: Commit** `feat(hf): parler converter and indic parler-tts training`
- [ ] **Step 6: VM GATE**: `bash scripts/test_backend.sh hf --heavy`

---

## Part D — `nemo` backend

Registry:

```python
MODELS = {
  "ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large": ModelInfo("asr", "nemo-manifest", True),
  "bengaliAI/BanglaConformer":                             ModelInfo("asr", "nemo-manifest", False),
  "nvidia/nemotron-3.5-asr-streaming-0.6b":                ModelInfo("asr", "nemo-manifest", False),
}
```

### Task 20: nemo scaffold + Lightning callback + VM install check

**Files:**
- As in Task 10:
  - name `mlops-backend-nemo`
  - deps: `bangla-voice-backend[prepare,audio]`, `torch==2.7.1`, `nemo_toolkit[asr]`, `lightning`, `huggingface_hub`
  - capabilities: `asr` / method `full` / export `nemo`
  - hparams: `learning_rate`, `epochs`, `batch_size`, `grad_accum`, `max_steps`, `seed`, `eval`, `prepare`
- Create: `backend/app/core/backend_kit/lightning.py` with `ContractLightningCallback(Callback)`:
  - `__init__(rep, stop, nan, total_steps)`
  - `on_train_batch_end`: metrics `train_loss` from `trainer.callback_metrics`, NaN guard, status; when stop is requested, `trainer.should_stop = True` and a checkpoint is saved
  - `on_validation_end`: `val_loss`/`val_wer` → split `val`
  - `on_save_checkpoint`: write `.complete` next to the saved `.ckpt` (directory per checkpoint: `checkpoints/step-<n>/model.ckpt`)

- [ ] Steps 1–4 (light: capabilities tests). **Step 5: Commit** `feat(nemo): backend scaffold and lightning contract callback`
- [ ] **Step 6: VM GATE**: `bash scripts/check_backend.sh nemo`

### Task 21: `nemo-manifest` converter

**Files:**
- Create: `backend/workers/backends/nemo/mlops_backend/prepare/manifest.py`:
  - `build_manifest(encode: Callable[[str], list[int]] | None, unk_id: int | None) -> build fn`
    - per row: `load_audio(path, 16000)` → `write_wav(<out>/audio/<sample_id>.wav, …, 16000)`
    - manifest line `{"audio_filepath": abs, "duration": seconds (3 decimals), "text": bn_normalize(text, mode="asr")}` in `<out>/<split>.jsonl`
    - if `encode` is given and `unk_id in encode(text)`, skip with reason `"tokenizer maps characters to <unk>: <chars>"` (chars found by encoding each character alone)
  - `prepare_nemo(rd, cfg, rep)`: `check_hf_access` if gated; restore the model on CPU to get `model.tokenizer` (`text_to_ids`, `unk_id`); `ensure_prepared`
- Test: `tests/test_prepare_manifest.py` (light; uses the `audio` extra)

```python
def test_manifest_lines_resampled(tiny_audio_ds_44k, tmp_path): ...   # audio/ files are 16 kHz; duration matches
def test_nemo_skips_unk_rows(tmp_path):
    enc = lambda t: [0 if c != "ক" else 9 for c in t]
    res = build_manifest(enc, 9)({"train": [arow("আম"), arow("কম")]}, tmp_path)
    assert res.counts["train"] == 1 and res.skipped[0].reason == "tokenizer maps characters to <unk>: ক"
```

- [ ] Steps 1–4. **Step 5: Commit** `feat(nemo): manifest converter with unk guard`

### Task 22: NeMo training / eval / export (3 models)

**Warm start:**
- `nemo` init (`final/model.nemo`) → `ASRModel.restore_from(<init path>/model.nemo)`.
- `lightning-ckpt` init → load the base model, then `model.load_state_dict(torch.load(ckpt)["state_dict"])`.
- Write init meta: `nemo` for final, `lightning-ckpt` for intermediate.
- The baseline loads the init checkpoint when set.
- Light test: `test_nemo_init_path_choice`.

**Files:**
- Create: `backend/workers/backends/nemo/mlops_backend/asr.py`:
  - `load_model(base_model)`: `ASRModel.from_pretrained(model_name=base)`, falling back to `restore_from(hf_hub_download(base, <the .nemo file in the repo>))`
  - `apply_data_config(model, prepared_dir, cfg)`: `model.setup_training_data({"manifest_filepath": train.jsonl, "sample_rate": 16000, "batch_size": …, "shuffle": True, "num_workers": 4})`, same for validation
  - For hybrid (`EncDecHybridRNNTCTCBPEModel`), keep `aux_ctc` weight from the checkpoint config. For Nemotron, do not modify `encoder` streaming config keys (`att_context_size`, `conv_context_size`).
  - `train`: `lightning.Trainer(devices=1, accelerator="gpu", precision="bf16-mixed", max_epochs=epochs, max_steps=max_steps or -1, accumulate_grad_batches=grad_accum, callbacks=[ContractLightningCallback, ModelCheckpoint(dirpath=checkpoints, every_n_train_steps=…)], logger=False, enable_checkpointing=True)`, `ckpt_path=latest .ckpt` on resume; then `model.save_to(checkpoints/final/model.nemo)` + `mark_complete`
  - `eval`: `model.transcribe(paths, batch_size=…)` (returns `Hypothesis` objects or strings; take `.text`) → `run_asr_eval` + baseline
  - `export`: `nemo` artifact = copy of `final/model.nemo`, `serve_with="nemo"`
- Test: light `test_data_config_paths`, `test_streaming_keys_untouched` (mock config object as `dict`); heavy `test_indicconformer_20_steps`, `test_nemotron_20_steps`.

- [ ] Steps 1–4. **Step 5: Commit** `feat(nemo): conformer/nemotron training, eval and .nemo export`
- [ ] **Step 6: VM GATE**: `bash scripts/test_backend.sh nemo --heavy`

---

## Part E — `f5` backend

Registry: `{"ai4bharat/IndicF5": ModelInfo("tts", "f5", True)}`

### Task 23: f5 scaffold + VM install check

As Task 10:
- name `mlops-backend-f5`
- deps: `bangla-voice-backend[prepare,audio]`, `torch==2.7.1`, `torchaudio==2.7.1`, `f5-tts`, `huggingface_hub`, `transformers` (for `ContractCallback` reuse only if the trainer is HF-based; F5's trainer is accelerate-based, so this backend uses a step hook — see Task 25)
- capabilities: `tts` / `full` / export `f5-ckpt`
- hparams: `learning_rate`, `epochs`, `batch_size_frames` (default 3200), `grad_accum`, `warmup_steps` (default 1000), `max_steps`, `seed`, `eval`, `prepare`

- [ ] Steps 1–4. **Step 5: Commit** `feat(f5): backend scaffold`
- [ ] **Step 6: VM GATE**: `bash scripts/check_backend.sh f5`

### Task 24: `f5` converter (vocab extend, reference clips)

**Warm start:** when `cfg.init_from` is set, `build_f5`'s base vocab is `<init path>/vocab.txt` (spec §14.6). The init vocab hash goes into `PrepSpec.params`. Light test: `test_f5_vocab_extends_init_vocab`.

**Files:**
- Create: `backend/workers/backends/f5/mlops_backend/prepare/f5.py`:
  - `extend_vocab(base: list[str], texts: Iterable[str]) -> list[str]`: `base` unchanged, then new characters in first-seen order
  - `pick_reference(rows) -> dict[speaker, row]`: median `duration_s` per speaker; ties broken by `sample_id`
  - `build_f5(base_vocab: list[str]) -> build fn`:
    - `<out>/wavs/<sample_id>.wav` at 24 kHz
    - `<out>/<split>/raw.arrow` via pyarrow, columns `audio_path`, `text`, `duration`; `bn_normalize(mode="tts")` text
    - `<out>/<split>/duration.json` (`{"duration": [...]}`)
    - `<out>/vocab.txt` (one char per line)
    - `<out>/refs/<speaker>.wav` + `<out>/refs/<speaker>.txt`
  - `prepare_f5`: `check_hf_access`, download the IndicF5 `vocab.txt` via `hf_hub_download`, `ensure_prepared`
- Test: `tests/test_prepare_f5.py` (light)

```python
def test_f5_appends_new_chars_without_reordering():
    assert extend_vocab(["আ", "ম"], ["মাআক", "খ"]) == ["আ", "ম", "া", "ক", "খ"]
def test_reference_is_median_clip_per_speaker(): ...
def test_build_writes_24k_audio_and_duration_json(tiny_tts_ds, tmp_path): ...
```

- [ ] Steps 1–4. **Step 5: Commit** `feat(f5): converter with vocab extension and reference clips`

### Task 25: F5 training / eval / export

**Warm start:**
- `f5-pt` init → load `<init path>/model_last.pt`.
- Embedding expansion applies relative to the init vocab.
- Train writes `checkpoints/final/` (`model_last.pt` + `vocab.txt` + init meta `f5-pt`).
- The baseline loads the init checkpoint when set.

**Files:**
- Create: `backend/workers/backends/f5/mlops_backend/tts.py`:
  - Load the model as `f5_tts.infer.utils_infer.load_model(DiT, model_cfg, ckpt_path, vocab_file=<prepared>/vocab.txt)`. If the vocab grew, expand `transformer.text_embed.text_embed` rows for the appended characters, copying the initial rows from the mean embedding.
  - Train with `f5_tts.model.Trainer(model, epochs, learning_rate, num_warmup_updates=warmup_steps, save_per_updates=…, checkpoint_path=rd.checkpoints_dir, batch_size_per_gpu=batch_size_frames, batch_size_type="frame", grad_accumulation_steps=grad_accum, logger=None)` and `trainer.train(train_dataset, resumable_with_seed=seed)`.
  - Progress and metrics: wrap `trainer.accelerator.log` (or subclass `Trainer` and override its per-update hook) to call `Reporter`, `NanGuard` and check `StopFlag`. On stop, `trainer.save_checkpoint(update, last=True)` then exit 143. Mark `model_last.pt`'s directory complete after each save: checkpoints are saved into `checkpoints/update-<n>/`, and `model_last.pt` is copied there.
  - Eval: `synthesize(text, row)` = `utils_infer.infer_process(ref_audio, ref_text, text, model, vocoder)` using the speaker's reference clip → `(audio, 24000)`; `run_tts_eval` (+ baseline with the original IndicF5 checkpoint).
  - Export `f5-ckpt`: directory with `model.safetensors` (EMA weights), `vocab.txt`, `refs/`, and `serve_with="f5-tts"`.
- Test: light `test_embedding_expansion_shape` (pure-tensor helper `expand_rows(weight: "torch.Tensor", n_new: int)`, heavy-marked since it needs torch); heavy `test_f5_20_updates_and_synthesize`.

- [ ] Steps 1–4. **Step 5: Commit** `feat(f5): indicf5 fine-tune, eval and export`
- [ ] **Step 6: VM GATE**: `bash scripts/test_backend.sh f5 --heavy`

---

## Part F — Catalog, smoke, docs

### Task 26: Catalog recipe templates match backends

**Files:**
- Modify the `recipe_template` blocks for all in-scope catalog entries (spec §2.1): set `backend`, rename hparams to the Global Constraints names (`max_epochs` → `epochs`, `max_seq_len` kept), set `export` to supported formats.

| Catalog entry | `export` |
|---|---|
| LLMs | `[gguf, merged-safetensors]` |
| whisper (unsloth LoRA template) | `[ct2]` |
| wav2vec2 | `[safetensors]` |
| mms | `[adapter-safetensors]` |
| conformers | `[nemo]` |
| parler | `[safetensors]` |
| indicf5 | `[f5-ckpt]` |

  Whisper templates use backend `hf` with method `full`, matching their current `method`. Out-of-scope entries keep `backend: null`.
- Modify: `backend/tests/core/test_repo_catalog.py`: add `test_recipe_templates_validate_against_backend_schemas`. For each entry with `recipe_template.backend` set, load that backend's `capabilities.json` and validate `{base_model: hf_repo, method, hparams, export}` with `jsonschema`. Assert that the export formats are listed and the method is in `methods`.

- [ ] **Step 1:** Write the test. **Step 2:** FAIL. **Step 3:** Edit the YAML. **Step 4:** PASS (`cd backend && uv run pytest tests/core/test_repo_catalog.py -v`). **Step 5: Commit** `feat(catalog): recipe templates target real backends`

### Task 27: Smoke script, fixtures, README

**Files:**
- Create: `scripts/smoke/fixtures/llm.jsonl` (16 Bangla conversations), `scripts/smoke/make_audio_fixtures.py` (generates `asr.zip`/`tts.zip` with 16 sine-tone clips + `metadata.csv` at run time; nothing binary committed)
- Create: `scripts/smoke_backends.sh [backend...]` (default: all four). For each backend:
  1. ingest the matching fixture via `POST /api/datasets/ingest` with provenance `source=smoke, license=internal, consent=true, commercial_ok=false`
  2. poll the ingest
  3. save a recipe from the backend's first model with `hparams.max_steps=20`, `eval.max_samples=4`
  4. create the run and poll `GET /api/runs/{id}` until terminal
  5. assert:
     - `state == "done"`
     - `GET /prepared` non-empty
     - `eval.metrics` contains the task key: `perplexity`, `wer` or `rtf`
     - each export artifact `path` exists on disk
  6. start a second run of the same recipe with `init_from: {run: <first run id>, checkpoint: final}` and apply the same checks to it, plus `parent_run_id` set
  7. print `PASS <backend>/<model>` and `PASS <backend>/<model> warm-start`, or `FAIL ... <reason>`; exit non-zero on any failure
- Modify: `README.md`: section "Training backends" covering setup (`scripts/setup_venvs.sh`), `HF_TOKEN`, the per-backend check, the smoke script, and the dataset-format → prepared-format table (spec §4.4)

- [ ] **Step 1:** Write the fixtures and script; `bash -n scripts/smoke_backends.sh` passes. **Step 2: Commit** `test: end-to-end smoke script for training backends; README`
- [ ] **Step 3: VM GATE**: `bash scripts/setup_venvs.sh && bash scripts/smoke_backends.sh`. Expected: `PASS` per backend/model.

---

## Part G — Warm start (spec §14). Execute after Task 9, before Task 10.

### Task 28: `init_from` contract, validation, lineage, `init/` linking

**Files:**
- Modify: `backend/app/core/recipes.py`:
  - `InitFrom(BaseModel)`: `run: str | None = None`, `checkpoint: Literal["final", "latest"] | int = "final"`, `model_version: str | None = None`. A validator requires exactly one of `run`/`model_version`, and a non-default `checkpoint` only with `run`.
  - `RecipeSpec.init_from: InitFrom | None = None`.
  - `INIT_METHOD_PAIRS: frozenset[tuple[str, str]]` from spec §14.2: `(lora,lora) (lora,qlora) (qlora,lora) (qlora,qlora) (lora,full) (qlora,full) (full,lora) (full,qlora) (full,full) (adapter,adapter)`.
- Modify: `backend/app/core/contract.py`:
  - `InitRef(BaseModel)`: `path: str`, `source_run: str`, `checkpoint: str`, `method: str`, `format: str`
  - `RunConfig.init_from: InitRef | None = None`
  - `RunDir.init_dir` (`init/`)
- Modify: `backend/app/core/backend_kit/__init__.py`:
  - `INIT_META = "init_meta.json"`
  - `write_init_meta(ckpt_dir: Path, *, backend: str, base_model: str, method: str, format: str, step: int) -> None`
  - `read_init_meta(ckpt_dir: Path) -> dict | None`
  - `resolve_checkpoint(rd: RunDir, which: str | int) -> Path | None`: `"final"` = `checkpoints/final` if complete; `"latest"` = `latest_checkpoint(rd)`; int = the complete dir whose trailing int equals it
  - `@dataclass InitChoice`: `kind: Literal["resume", "init"]`, `path: Path`, `meta: dict | None`
  - `init_source(cfg: RunConfig, rd: RunDir) -> InitChoice | None`:
    - own `latest_checkpoint` when `cfg.resume` and one exists → `resume`
    - else `cfg.init_from` → `init` with `path=rd.init_dir`, `meta=read_init_meta(rd.init_dir)`
    - else `None`
- Create: `backend/app/runs_init.py` (server side, async, DB):
  - `async resolve_init(session, spec: RecipeSpec, settings) -> tuple[tuple[Run, Path, InitRef] | None, list[str]]`
  - Validates per spec §14.3. A `model_version` ref (`<name>@v<N>`) resolves to its run with checkpoint `final`.
  - The source method is read from the resolved checkpoint's `init_meta.json`.
  - Error strings:
    - `init_from: run <id> not found`
    - `init_from: model version <ref> not found`
    - `init_from: run <id> is <field> '<a>', recipe is <field> '<b>'` (field ∈ task, backend, base_model)
    - `init_from: method '<src>' cannot start a '<dst>' run`
    - `init_from: run <id> has no complete checkpoint '<which>'`
  - `link_tree(src: Path, dst: Path) -> None`: hard-link every file, falling back to `shutil.copy2` on `OSError`; build into `dst.tmp` then `os.replace`.
- Modify: `backend/app/routers/recipes.py`: `check_recipe` appends `resolve_init` errors.
- Modify: `backend/app/routers/runs.py`:
  - `create_run`: resolve init again (422 on errors); `commercial_ok` ANDs the parent run's flag; set `Run.parent_run_id`/`parent_checkpoint`
  - run detail JSON adds `parent_run_id`, `parent_checkpoint`
- Modify: `backend/workers/executor.py`:
  - `build_config` sets `init_from` (`InitRef` with `path=str(rd.init_dir)`)
  - before the first phase, when `init_from` is set and `rd.init_dir` does not exist, `link_tree(source checkpoint, rd.init_dir)`
  - a missing source checkpoint at that point fails the run with the matching `init_from:` message
- Create: `backend/alembic/versions/0002_run_parent.py`: nullable `parent_run_id` (FK `run.id`) and `parent_checkpoint` (String) on `run`; upgrade and downgrade
- Modify: `backend/app/models.py` (two columns)
- Modify: the fake backend's train writes `checkpoints/final/` with `.complete` and `write_init_meta(..., method=cfg.method["type"], format="fake")`, so warm start works end to end in tests. Add `"none"` to the allowed pairs only for the fake backend: `INIT_METHOD_PAIRS` plus `("none", "none")`.
- Test: `backend/tests/core/test_recipes.py`, `backend/tests/core/test_backend_kit.py`, `backend/tests/app/test_runs_init.py`, `backend/tests/workers/test_executor.py`, the existing migration test module

**Interfaces:**
- Produces (used by Tasks 12–25, 29, 30):
  - `InitFrom`, `InitRef`, `RunConfig.init_from`
  - `write_init_meta`, `read_init_meta`, `resolve_checkpoint`, `init_source`, `InitChoice`
  - `resolve_init`, `link_tree`
  - `Run.parent_run_id`, `Run.parent_checkpoint`

- [ ] **Step 1: Failing tests**

```python
def test_init_from_requires_exactly_one_source(): ...          # both / neither -> ValidationError
def test_init_from_checkpoint_only_with_run(): ...
def test_resolve_checkpoint_final_latest_step(tmp_path): ...   # incomplete dirs ignored; 300 picks checkpoint-300
def test_init_source_prefers_own_checkpoint_on_resume(tmp_path): ...
def test_init_source_uses_init_dir_when_no_own_checkpoint(tmp_path): ...
async def test_resolve_init_rejects_backend_mismatch(session, settings, done_run):
    # errors == [f"init_from: run {done_run.id} is backend 'fake', recipe is backend 'other'"]
async def test_resolve_init_rejects_method_pair(...): ...       # source "adapter" -> new "full"
async def test_resolve_init_model_version_uses_final(...): ...
async def test_resolve_init_missing_checkpoint(...): ...        # "init_from: run r-… has no complete checkpoint 'latest'"
def test_link_tree_survives_source_delete(tmp_path): ...
async def test_create_run_sets_parent_and_ands_commercial_ok(...): ...   # parent False -> child False
async def test_executor_links_init_dir_and_sets_config(...): ...         # fake run 2 from run 1: config.json init_from.path == <run2>/init, init/ has files
def test_migration_0002_upgrade_downgrade(...): ...
```

- [ ] **Step 2: FAIL. Step 3: Implement. Step 4: PASS** (`cd backend && uv run pytest`). **Step 5: Commit** `feat: warm start contract, validation, lineage and init linking`

### Task 29: Checkpoints and init-sources API

**Files:**
- Modify: `backend/app/routers/runs.py`:
  - `GET /api/runs/{id}/checkpoints` returns `{"items": [{"name", "step", "complete", "init_meta"}]}`
  - `final` comes first, then the rest newest first
  - `step` is `init_meta.step`, else the trailing int, else `null`
- Create: `backend/app/routers/init_sources.py` (wired in `backend/app/app.py`):
  - `GET /api/init-sources?task=&backend=&base_model=` (all required) returns:
    `{"runs": [{"id", "recipe", "state", "finished_at", "checkpoints": [complete items only, same shape]}], "model_versions": [{"ref": "<name>@v<N>", "run_id", "metrics"}]}`
  - Runs are filtered to matching task/backend/base_model with at least one complete checkpoint, newest first, max 50.
- Test: `backend/tests/app/test_runs_api.py`, `backend/tests/app/test_init_sources_api.py`

- [ ] **Step 1: Failing tests:**
  - `test_checkpoints_lists_complete_and_incomplete`
  - `test_checkpoints_final_first`
  - `test_init_sources_filters_by_backend_and_model`
  - `test_init_sources_excludes_runs_without_complete_checkpoint`
  - `test_init_sources_lists_model_versions`
- [ ] **Step 2–4.** **Step 5: Commit** `feat(api): run checkpoints and init sources`

### Task 30: UI — Start-from picker, checkpoints tab, parent link

**Files:**
- Modify: `frontend/src/lib/types.ts`, `frontend/src/lib/api.ts`: `getCheckpoints(id)`, `getInitSources(q)`, `InitFrom` type
- Modify: `frontend/src/lib/recipe-yaml.ts`: round-trip `init_from`
- Create: `frontend/src/components/start-from-picker.tsx` + test:
  - a select with "Base model (default)", each run's complete checkpoints labelled `<run id> · <checkpoint> · step N`, and model versions `<name>@vN`
  - `onChange(InitFrom | null)`
  - it refetches when task/backend/base_model change and clears a selection that is no longer listed
- Modify: `frontend/src/components/recipe-editor.tsx`: render the picker under the base model field
- Modify: `frontend/src/app/runs/[id]/page.tsx`:
  - a "Checkpoints" tab: table of name, step, complete badge, and a "New run from this checkpoint" button (only on complete rows) that navigates to the recipe editor with `?from_run=<id>&checkpoint=<name>`
  - the header shows `Started from <parent run link> · <checkpoint>` when `parent_run_id` is set
- Modify: the recipe creation page: with `from_run` in the query, prefill from that run's recipe version and set `init_from`
- Tests (vitest):
  - `recipe-yaml` round-trips `init_from`
  - the picker lists options from mocked `getInitSources` and emits `{run, checkpoint}`
  - the run page renders the button only for complete checkpoints
  - the prefill sets `init_from`

- [ ] **Step 1: Failing tests. Step 2: FAIL. Step 3: Implement. Step 4: PASS + `npm run typecheck` + `npm run build`** (run from an upper-case `C:\` cwd). **Step 5: Commit** `feat(web): start-from picker, checkpoints tab, parent link`

---

## Part H — Per-run splits (spec §15). Execute after Task 9, before Task 28.

### Task 31: Split declaration at ingest, `SplitConfig`, per-run assignment

**Files:**
- Create: `backend/app/core/splits.py`:
  - `class SplitRatios(BaseModel)`: `train: float = 0.9`, `val: float = 0.05`, `test: float = 0.05`, with the spec §15.4 validator
  - `class SplitConfig(BaseModel)`: `use_dataset_splits: bool = True`, `ratios: SplitRatios = SplitRatios()`
  - `SPLIT_ALIASES: dict[str, str]` (spec §15.2)
  - `normalize_split(value: str | None) -> str | None`: raises `ValueError` for unknown values
  - `assign(rows: list[dict], cfg: SplitConfig) -> list[str]`: spec §15.5
  - `class EmptySplitError(ValueError)`: message `split '<name>' is empty: add data or change datasets.split.ratios`
- Modify: `backend/app/core/contract.py`: `DatasetRef.split: SplitConfig | None = None` (import from `app.core.splits`, which is pure)
- Modify: `backend/app/core/recipes.py`: `RecipeDatasets.split: SplitConfig = SplitConfig()`
- Modify: `backend/app/core/prepare.py`:
  - `iter_rows(ds, split)`: when `ds.split` is set, compute `assign` over all rows once (cache per `(manifest, split config)` within the process), then filter. When `ds.split is None`, yield every row for any requested split in `ds.splits` (eval-only behaviour).
  - `prepared_dir`/`ensure_prepared`: the directory hash is the hash of `{"spec": spec.model_dump(), "split": ds.split.model_dump() if ds.split else None}`. The stored `spec_hash` is that value.
  - `EmptySplitError` from `assign` becomes a `PrepareError` with the same message.
- Modify: `backend/app/core/evaluation.py`: `baseline_metrics` key includes `ds.split`.
- Modify: `backend/app/datasets/validate_llm.py`, `validate_audio.py`, `ingest.py`:
  - accept the `"split"` field / `split` column (remove `split` from `_RESERVED`)
  - accept the file layouts of spec §15.2: `llm-sft` now accepts `.jsonl` or `.zip`; audio zips accept `metadata_<split>.csv`
  - store the normalised split, or null
  - errors exactly as spec §15.2
  - stop calling `assign_split`; delete it and its test, replaced by `tests/core/test_splits.py`
- Modify: `backend/app/datasets/versions.py` and `backend/app/models.py`: `DatasetVersion.split_source: str = "none"`, `given_splits: dict`. Transcript edits keep both.
- Create: `backend/alembic/versions/0003_dataset_splits.py` (after `0002_run_parent`; if Task 28 has not run yet, chain from `0001` and Task 28 renumbers its migration to chain after this one. Rule: migrations form one linear chain in commit order.)
- Modify: `backend/app/routers/datasets.py`:
  - the version detail returns `split_source` and `given_splits`, plus `default_split_counts` (counts under `SplitConfig()`)
  - the sample listing's `split` filter and the per-row `split` field use `assign(rows, SplitConfig())`
- Modify: `backend/workers/executor.py` `build_config`: train ref `split=spec.datasets.split`; eval ref `split=None` if eval-only, else `spec.datasets.split`
- Tests:
  - `backend/tests/core/test_splits.py`
  - `backend/tests/core/test_prepare.py` (+ split cases)
  - `backend/tests/app/datasets/test_ingest.py`, `test_validate_llm.py`, `test_validate_audio.py`
  - `backend/tests/app/test_datasets_api.py`
  - `backend/tests/workers/test_executor.py`
  - the migration test module

**Interfaces:**
- Produces (Task 32 and every backend task rely on these):
  - `SplitConfig`, `SplitRatios`, `assign`, `normalize_split`, `EmptySplitError`
  - `DatasetRef.split`
  - `RecipeDatasets.split`
  - dataset version JSON fields `split_source`, `given_splits`, `default_split_counts`

- [ ] **Step 1: Failing tests**

```python
def rows(labels): return [{"sample_id": f"{i:016x}", "split": l} for i, l in enumerate(labels)]

def test_no_labels_90_5_5():
    out = assign(rows([None] * 20000), SplitConfig())
    assert abs(out.count("train") / 20000 - 0.9) < 0.02 and abs(out.count("val") / 20000 - 0.05) < 0.02
def test_all_given_kept():
    labels = ["train", "val", "test"] * 10
    assert assign(rows(labels), SplitConfig()) == labels
def test_train_and_test_given_carves_val_from_train():
    labels = ["train"] * 9000 + ["test"] * 1000
    out = assign(rows(labels), SplitConfig())
    assert out[9000:] == ["test"] * 1000
    assert abs(out[:9000].count("val") / 9000 - 0.05 / 0.95) < 0.02 and "test" not in out[:9000]
def test_only_test_given_rest_split_90_5():
    labels = ["test"] * 1000 + [None] * 9000
    out = assign(rows(labels), SplitConfig())
    assert out[:1000] == ["test"] * 1000
    assert abs(out[1000:].count("train") / 9000 - 0.9 / 0.95) < 0.02
def test_ignore_dataset_splits():
    labels = ["test"] * 10000
    out = assign(rows(labels), SplitConfig(use_dataset_splits=False))
    assert abs(out.count("train") / 10000 - 0.9) < 0.02
def test_deterministic_and_order_independent():
    r = rows([None] * 500)
    a = dict(zip([x["sample_id"] for x in r], assign(r, SplitConfig())))
    rr = list(reversed(r))
    assert dict(zip([x["sample_id"] for x in rr], assign(rr, SplitConfig()))) == a
def test_empty_split_error():
    with pytest.raises(EmptySplitError, match="split 'val' is empty"):
        assign(rows([None]), SplitConfig())   # one row cannot fill train and val
def test_zero_ratio_split_may_be_empty():
    assign(rows([None] * 10), SplitConfig(ratios=SplitRatios(train=0.9, val=0.1, test=0.0)))
@pytest.mark.parametrize("bad", [dict(train=0.5, val=0.2, test=0.2), dict(train=0, val=0.5, test=0.5), dict(train=1.1, val=-0.1, test=0)])
def test_ratio_validation(bad):
    with pytest.raises(ValidationError): SplitRatios(**bad)
@pytest.mark.parametrize("v,exp", [("Validation", "val"), ("dev", "val"), ("TRAINING", "train"), ("testing", "test"), ("", None), (None, None)])
def test_normalize_split(v, exp): assert normalize_split(v) == exp
def test_normalize_split_unknown(): ...   # "hold" -> ValueError
```

  Ingest tests:
  - `test_llm_split_field`
  - `test_llm_zip_file_layout`
  - `test_llm_zip_without_split_files_errors`
  - `test_audio_split_column`
  - `test_audio_metadata_split_files`
  - `test_audio_mixed_metadata_layouts_errors`
  - `test_unknown_split_value_errors` (exact message `split value 'hold' on line 2: use train, valid or test`)
  - `test_split_source_and_given_splits_recorded`

  Prepare tests:
  - `test_iter_rows_applies_split_config`
  - `test_prepared_dir_differs_when_only_ratios_differ`
  - `test_eval_only_ref_yields_all_rows`

  Executor test:
  - `test_build_config_passes_split_config`

  Migration test:
  - `test_migration_0003_upgrade_downgrade`

  API test:
  - `test_version_detail_reports_split_source_and_default_counts`

- [ ] **Step 2: FAIL. Step 3: Implement. Step 4: PASS** (`cd backend && uv run pytest`). **Step 5: Commit** `feat: per-run train/valid/test splits with dataset-provided defaults`

### Task 32: UI — Data splits group, dataset split display

**Files:**
- Modify: `frontend/src/lib/types.ts`: `SplitConfig`; `DatasetVersion` gains `split_source`, `given_splits`, `default_split_counts`
- Modify: `frontend/src/lib/recipe-yaml.ts`: round-trip `datasets.split`
- Create: `frontend/src/lib/splits.ts`:
  - `previewCounts(given: Record<string, number>, n: number, cfg: SplitConfig): {train: number, val: number, test: number}`. It is an approximation, like the server's proportional rule (spec §15.5), used for the live preview only.
  - `ratiosValid(r): string | null`: returns the error text `Ratios must add up to 100%`, or `Train must be above 0%`
- Create: `frontend/src/components/split-fields.tsx` + test:
  - props: `value: SplitConfig`, `onChange`, `dataset: {split_source, given_splits, n_samples} | null`
  - shows the "Use the dataset's splits" toggle only when `split_source === "dataset"`, with its counts
  - percentage inputs train / valid / test (90 / 5 / 5), disabled while the toggle is on
  - the live preview line `≈ train n · valid n · test n`
  - the inline error from `ratiosValid`
- Modify: `frontend/src/components/recipe-editor.tsx`: render `<SplitFields>` in the hyperparameters area, bound to `datasets.split`; block save while `ratiosValid` returns an error
- Modify: `frontend/src/app/datasets/[name]/v/[n]/page.tsx`: the split summary line per spec §15.7
- Tests (vitest):
  - `splits.test.ts`: preview for none-given / partial / all-given, and the validation messages
  - `split-fields.test.tsx`: toggle hidden without dataset splits, inputs disabled while on, error shown at 60/20/10
  - `recipe-yaml` round-trips `datasets.split`
  - the dataset page shows both summary variants

- [ ] **Step 1: Failing tests. Step 2: FAIL. Step 3: Implement. Step 4: PASS + `npm run typecheck` + `npm run build`** (run from an upper-case `C:\` cwd). **Step 5: Commit** `feat(web): data splits fields in recipe form; dataset split summary`
