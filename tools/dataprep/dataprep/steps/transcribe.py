"""`transcribe` step: ASR labeller hypotheses and CER against the cleaned transcript (spec 6.2).

A labeller is a callable `transcribe(paths: list[Path]) -> list[str]`. Specs: `whisper-hf:<hf id or local dir>`
(transformers pipeline; a platform run's exported model dir works) and `nemo:<hf id or .nemo path>`.
Output is `cer_<slug>.parquet` (`id, hyp, cer, error`); CER is `app.core.metrics.cer` over
`bn_normalize(mode="asr")` of both sides. Models load lazily; tests inject `_labeller`.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from app.core.metrics import cer
from app.core.text import bn_normalize

from ..select import cer_step_name
from ..workdir import StepFile, WorkDir
from ._runner import run_step

VERSION = 1
COLUMNS = ["id", "hyp", "cer", "error"]
BATCH_SIZE = 16
SAMPLE_RATE = 16000

Labeller = Callable[[list[Path]], list[str]]
_labeller: Labeller | None = None  # injected by tests
_labellers: dict[tuple[str, int], Labeller] = {}
CHECKPOINT_BATCHES = 64


def step_name(labeller: str) -> str:
    return cer_step_name(labeller)  # the one slug function; select reads the same file


def _texts(out) -> list[str]:
    """NeMo returns list[str], list[Hypothesis], or a (best, all) tuple of such lists (RNNT models)."""
    if isinstance(out, tuple):
        out = out[0]
    return [(h if isinstance(h, str) else h.text).strip() for h in out]


def _whisper(model: str, batch_size: int) -> Labeller:
    from ..roundtrip_cosyvoice import load_asr_pipeline

    asr = load_asr_pipeline(model)
    kwargs = {"language": "bengali", "task": "transcribe"}

    def transcribe(paths: list[Path]) -> list[str]:
        import librosa

        wavs = [{"raw": librosa.load(str(p), sr=SAMPLE_RATE, mono=True)[0], "sampling_rate": SAMPLE_RATE} for p in paths]
        out = asr(wavs, batch_size=batch_size, chunk_length_s=30, generate_kwargs=kwargs)
        return [o["text"].strip() for o in out]

    return transcribe


def _nemo(model: str, batch_size: int) -> Labeller:
    import nemo.collections.asr as nemo_asr

    if model.endswith(".nemo") or Path(model).is_file():
        asr = nemo_asr.models.ASRModel.restore_from(model)
    else:
        asr = nemo_asr.models.ASRModel.from_pretrained(model)
    asr.eval()

    def transcribe(paths: list[Path]) -> list[str]:
        return _texts(asr.transcribe([str(p) for p in paths], batch_size=batch_size))

    return transcribe


def make_labeller(spec: str, *, batch_size: int = BATCH_SIZE) -> Labeller:
    """The real labeller for `whisper-hf:<model>` or `nemo:<model>`. Imports torch/transformers/nemo lazily."""
    kind, _, model = spec.partition(":")
    if kind == "whisper-hf" and model:
        return _whisper(model, batch_size)
    if kind == "nemo" and model:
        return _nemo(model, batch_size)
    raise ValueError(f"unknown labeller {spec!r}; use whisper-hf:<hf id or dir> or nemo:<hf id or .nemo path>")


def get_labeller(spec: str, *, batch_size: int = BATCH_SIZE) -> Labeller:
    if _labeller is not None:
        return _labeller
    key = (spec, batch_size)
    if key not in _labellers:
        _labellers[key] = make_labeller(spec, batch_size=batch_size)
    return _labellers[key]


def _result(row: dict, ref: str, hyp: str) -> dict:
    value = cer([bn_normalize(ref, mode="asr")], [bn_normalize(hyp, mode="asr")])
    return {"id": row["id"], "hyp": hyp, "cer": float(value), "error": None}


def _err(row: dict, e: Exception | str) -> dict:
    return {"id": row["id"], "error": e if isinstance(e, str) else f"{type(e).__name__}: {e}"}


def _transcribe_rows(rows: list[dict], refs: dict[str, str], labeller: Labeller) -> list[dict]:
    """One batch; if the batch call fails, retry per row so one bad file does not poison the batch."""
    try:
        hyps = labeller([Path(r["src_path"]) for r in rows])
        if len(hyps) != len(rows):
            raise ValueError(f"labeller returned {len(hyps)} texts for {len(rows)} files")
        return [_result(r, refs[r["id"]], h) for r, h in zip(rows, hyps)]
    except Exception as e:  # noqa: BLE001
        if len(rows) == 1:
            return [_err(rows[0], e)]
    return [x for r in rows for x in _transcribe_rows([r], refs, labeller)]


def run(
    wd: WorkDir, spec: str, *, batch_size: int = BATCH_SIZE, force: bool = False, retry_errors: bool = False,
    only_profile: str | None = None,
) -> int:
    only_ids = None
    if only_profile:
        from ..profiles import load_profile
        from ..select import eligible, resolve_skips

        profile = load_profile(only_profile)
        # The cer filter needs this step's output, so it is always skipped here (spec 5.2 step 2).
        skip, notes = resolve_skips(wd, profile, spec, frozenset({"cer"}), speaker_if_missing=True)
        for note in notes:
            print(f"transcribe: {note}")
        only_ids = set(eligible(wd, profile, spec, skip_columns=skip)["id"])
    labeller = get_labeller(spec, batch_size=batch_size)  # fail fast on a missing extra or model
    text = StepFile.read(wd.step_path("text"), step="text", version=1, params={})
    info = text.set_index("id")[["text_clean", "empty_asr", "error"]].to_dict("index")
    refs = {i: v["text_clean"] for i, v in info.items()}

    def rows_fn(rows: list[dict]) -> list[dict]:
        out, todo = [], []
        for r in rows:
            t = info.get(r["id"])
            if t is None or pd.notna(t["error"]):
                out.append(_err(r, "no reference text"))
            elif bool(t["empty_asr"]):
                out.append(_err(r, "empty reference"))
            else:
                todo.append(r)
        for i in range(0, len(todo), batch_size):
            out.extend(_transcribe_rows(todo[i : i + batch_size], refs, labeller))
        failed = sum(1 for o in out if o["error"])
        print(f"{step_name(spec)}: {len(out) - failed} ok, {failed} failed")
        return out

    return run_step(
        wd, step=step_name(spec), version=VERSION, columns=COLUMNS, row_fn=lambda r: r, input_cols=["id", "src_path"],
        dtypes={"hyp": "object", "cer": "float64"}, workers=1, force=force, retry_errors=retry_errors,
        only_ids=only_ids, rows_fn=rows_fn, checkpoint_every=batch_size * CHECKPOINT_BATCHES,
    )


def _cmd(args: argparse.Namespace) -> int:
    from ..select import SelectError
    from ..workdir import StaleStepError

    try:
        n = run(
            WorkDir(args.work), args.labeller, batch_size=args.batch_size, force=args.force,
            retry_errors=args.retry_errors, only_profile=args.only_profile,
        )
    except (SelectError, StaleStepError, RuntimeError, ImportError, ValueError, OSError) as e:
        print(f"transcribe: {e}")
        return 1
    print(f"transcribe: {n} rows computed")
    return 0


def add_parser(commands) -> None:
    p = commands.add_parser("transcribe", help="ASR labeller hypotheses and CER (needs the labeller extra)")
    p.add_argument("--work", type=Path, default=Path("var/dataprep/bengaliai"), help="work dir")
    p.add_argument("--labeller", required=True, help="whisper-hf:<hf id or local dir> or nemo:<hf id or .nemo path>")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="clips per labeller call")
    p.add_argument("--force", action="store_true", help="recompute all rows")
    p.add_argument("--retry-errors", action="store_true", help="recompute rows whose error is set")
    p.add_argument(
        "--only-profile", default=None, metavar="NAME",
        help="transcribe only clips passing this profile's other filters (spec 5.2 step 2)",
    )
    p.set_defaults(func=_cmd)
