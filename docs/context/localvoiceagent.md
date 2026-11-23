# LocalVoiceAgent: integration context

Source: `C:\Users\BS23-DESKTOP-00038\Projects\LocalVoiceAgent` (read 2026-11-23, latest branch `feat/call-UI-integration`).
Its own `CLAUDE.md` and `README.md` are authoritative; this note only records the seams this project plugs into.

## What it is

A local voice bot for measuring latency: RealtimeSTT -> LLM -> RealtimeTTS, FastAPI server (`server/voicebot/`) + React/Vite UI (`web/`).
Runs on the same Ubuntu 24.04 VM (RTX 5090 32 GB, `sm_120`, `torch==2.7.1+cu128`), opened from the Windows PC over an SSH tunnel.
Per-turn timing spans: `silence_wait_ms`, `stt_ms`, `llm_ttft_ms`, `filter_ms`, `tts_first_audio_ms`, `playback_ms`, `perceived_ms`, `total_ms` -> `logs/turns.jsonl`.

## Seams

| Concern | Where | Contract |
|---|---|---|
| LLM | `server/voicebot/llm/base.py`, `llm/factory.py` | `LLMBackend` protocol: `start()`, `stream(messages, GenerationParams, stop_event) -> Iterator[str]`, `close()`. Selected by `LLM_BACKEND` = `llamacpp` (GGUF via llama-server Docker image) or `transformers` (HF checkpoint, 4-bit bnb, not yet run on VM). |
| STT / TTS selection | `server/voicebot/catalog.py`, `catalog.toml`, `bench/pool.toml` | `Entry(slug, kind=stt/tts, role=final/realtime, engine, model, device, family, options, streaming, vram_mb, fetch, runtime, group, voice)`. `fetch` accepts `auto`, `hf:<repo>[:<file>]`, `manual:<note>`, etc. |
| TTS engine build | `server/voicebot/tts_engines.py` | `build_engine(entry)` -> `getattr(RealtimeTTS, entry.engine)(**kwargs)`. A new engine (e.g. F5/IndicF5) needs a RealtimeTTS engine class. |
| STT build | `server/voicebot/stt.py` | RealtimeSTT recorder; default engine faster-whisper (CTranslate2). A fine-tuned Whisper must be converted to CT2 to use it. |
| Benchmarks | `server/voicebot/bench/` (`run_stt.py`, `run_tts.py`, `matrix.py`) | WER + latency per catalog entry. |

## Constraints that carry over

- Do not edit vendored `RealtimeSTT/` / `RealtimeTTS/` (see its `VENDORED.md`).
- Tests run without GPU, Docker, network or models.
- Browser audio only on the VM (no mic/speakers).

## Export formats implied for fine-tuned checkpoints

- LLM -> GGUF (Unsloth exports directly) for `llamacpp`, or merged HF weights for `transformers`.
- Whisper STT -> CTranslate2 for faster-whisper.
- TTS -> depends on engine; IndicF5/F5 has no RealtimeTTS engine today.
