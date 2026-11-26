# whisper-stt-playground

Gradio UI to check Whisper large-v3-turbo Bangla STT quality and latency, using the same engine as
LocalVoiceAgent's RealtimeSTT pipeline (faster-whisper on CTranslate2, language forced to `bn`, silero VAD).
Default checkpoint: `openai/whisper-large-v3-turbo` (the base you fine-tune). Preset dropdown also has `mozilla-ai/whisper-large-v3-turbo-bn`; point the box at your own checkpoint after training.

    ./run.sh                                       # http://localhost:7860
    STT_CHECKPOINT=/srv/mlops/.../ckpt ./run.sh    # your fine-tuned checkpoint
    PORT=7861 ./run.sh

- Checkpoint box takes an HF repo id, an HF-format dir (converted to CT2 once, cached in
  `~/.cache/whisper-stt-playground/ct2`), or a CT2 dir (the training platform's `ct2` export).
- Mic or file upload; paste a reference transcript to get WER/CER (`bn_normalize`, asr mode).
- Latency = full `transcribe()` incl. audio decode, VAD and decoding, mean of N runs after warm-up.
  RTF = latency / audio length. It excludes RealtimeSTT's endpointing wait (silence timeout).

## Live tab (turn-end timing)

Uses LocalVoiceAgent's own recorder: RealtimeSTT `AudioToTextRecorder` (faster_whisper engine, `language=bn`),
fed from the browser mic via `feed_audio` (the reference's `AUDIO_IO=browser`). Turn ends after
`post_speech_silence_duration` (default 0.6 s = the reference's `STT_SILENCE_SEC`; override with env or slider),
then it transcribes automatically. Stage marks use the same callbacks as the reference's `timing_callbacks`:
`silence wait` (speech_end -> endpoint) + `hand-off` (endpoint -> stt_start) + `STT` (stt_start -> text)
= `speech end -> text`. Press Start (mic needs localhost or HTTPS, so use the SSH tunnel to localhost).
