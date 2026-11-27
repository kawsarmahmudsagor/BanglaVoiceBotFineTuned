# stt-playground

Gradio UI to check Bangla STT quality and latency for the two model families the platform fine-tunes:

- **Whisper large-v3-turbo**: the same engine as LocalVoiceAgent's RealtimeSTT pipeline (faster-whisper on
  CTranslate2, language forced to `bn`, silero VAD, beam 5).
- **Qwen3-ASR**: transformers, bf16, greedy, with the same prompt as the qwen-asr backend's eval
  (`language Bengali<asr_text>`, read from `backend/workers/backends/qwen-asr/mlops_backend/prompt.py`).

The model family is detected from the checkpoint (its `config.json` `model_type`), so one box takes either.

**Model dropdown** (`registry.py`): the base models (`openai/whisper-large-v3-turbo`,
`mozilla-ai/whisper-large-v3-turbo-bn`, `SayedShaun/Qwen3-ASR-0.6B-Bengali`) plus every ASR run under
`/srv/mlops/runs` (override: `STT_RUNS_ROOT`), one entry each: its export (`ct2` / `safetensors`, written after
the run finished, labelled with the run's eval WER on its own eval set), or its `final` checkpoint when it has no
export. Selecting an entry loads it; "Rescan runs" picks up new runs. Whisper `checkpoint-N` dirs hold no tokenizer, so the base model's tokenizer is staged next to
links to the checkpoint (`~/.cache/stt-playground/staged`) before the CT2 conversion. The checkpoint box still
takes any path or repo id.

**Log**: every model load, file transcription and live turn is appended to `logs/playground.jsonl` (override:
`STT_LOG`), one JSON object per line, durations in seconds: model label, checkpoint, family, engine, device, GPU,
compute type, decoding/beam/VAD, audio length, latency (mean, best, every run), RTF, transcript, reference,
WER/CER; live turns add silence wait, hand-off and speech-end-to-text. `pandas.read_json(path, lines=True)` reads it.

    ./run.sh                                       # http://localhost:7860
    STT_CHECKPOINT=/srv/mlops/.../ckpt ./run.sh    # your fine-tuned checkpoint
    PORT=7861 ./run.sh

- Whisper: an HF repo id, an HF-format dir (converted to CT2 once, cached in `~/.cache/stt-playground/ct2`), or a
  CT2 dir (the training platform's `ct2` export).
- Qwen3-ASR: an HF repo id, a training checkpoint dir, or the platform's `safetensors` export dir. Compute type,
  beam and VAD do not apply.
- Mic or file upload; paste a reference transcript to get WER/CER (`bn_normalize`, asr mode).
- Latency = full transcription incl. VAD and decoding, mean of N runs after warm-up.
  RTF = latency / audio length. It excludes RealtimeSTT's endpointing wait (silence timeout).
- Latency rises while a training run shares the GPU.
- Whisper token cap: faster-whisper passes `max_length=448` to CTranslate2, which then stops after 224 generated
  tokens. Bangla costs ~16 tokens per second of speech, so stock faster-whisper cuts any window over ~14 s
  mid-sentence. The playground raises it to 440 new tokens (`WHISPER_MAX_LENGTH`, default 880); set
  `WHISPER_MAX_LENGTH=448` to reproduce the stock behaviour. A window still holds at most ~27 s of Bangla speech.
- Whisper window cap: with VAD on, the batched pipeline (the recorder's, and now the file tab's) merges speech into
  windows of up to 30 s. The Bangla fine-tunes, trained on single-sentence clips, end the transcript at a pause
  inside a long window and drop the phrase after it. The playground caps windows at 10 s (`WHISPER_WINDOW_S`; 30 =
  stock). Both settings are logged (`whisper_max_length`, `window_s`).
- Live turns save the audio sent to the engine to `logs/audio/*.wav` (path in the log's `audio_file`).
- Safeguards (`whisper_chunking.py`, on by default, `WHISPER_SAFEGUARDS=0` to turn off): unbroken speech over 10 s
  is split at its quietest point with 1 s overlap and the duplicated words are removed; a window that hit the token
  cap or gave fewer than 12 tokens/s is decoded again as 5 s windows. Each call logs a `whisper_windows` record.
- All Whisper problems, causes, fixes and settings for a production voice bot: `docs/whisper-specs.md`.

## Live tab (turn-end timing)

Uses LocalVoiceAgent's own recorder: RealtimeSTT `AudioToTextRecorder` (`language=bn`), fed from the browser mic
via `feed_audio` (the reference's `AUDIO_IO=browser`). Whisper runs on its `faster_whisper` engine; Qwen3-ASR runs on
`playground_qwen3_asr`, an engine `qwen_engine.py` adds to RealtimeSTT's registry. Turn ends after
`post_speech_silence_duration` (default 0.6 s = the reference's `STT_SILENCE_SEC`; override with env or slider),
then it transcribes automatically. Stage marks use the same callbacks as the reference's `timing_callbacks`:
`silence wait` (speech_end -> endpoint) + `hand-off` (endpoint -> stt_start) + `STT` (stt_start -> text)
= `speech end -> text`. Press Start (mic needs localhost or HTTPS, so use the SSH tunnel to localhost).

**Streaming toggle** (`live_stream.py`, off by default, `STT_STREAM=1` to start with it on): the "Stream" checkbox
switches between RealtimeSTT's turn-end decode and decoding chunks while the speaker talks. It applies from the next
turn, with no restart. In streaming mode a chunk is committed at every pause of at least the chunk pause (slider,
default 0.3 s, `STT_STREAM_PAUSE_S`, capped at 75% of the turn-end silence) after 1.5 s of audio, or forced after
10 s with a 1 s overlap. Each chunk is decoded by the recorder's own engine. At turn end only the rest is decoded, so
`STT` drops to the decode of the last few words, or to ~0 when the turn ended on a chunk. The turns table shows the
mode and the chunk count. The log adds `streaming`, `chunks` (each chunk's span, reason, decode time and text),
`flush_wait_s` and `flush_decode_s`. Chunks are decoded without the text before them, so a word at a chunk edge can
come out differently. See `model-specification/STT/whisper-specs.md`, P7.
