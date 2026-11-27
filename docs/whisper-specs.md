# Bangla Whisper in a voice bot: problems, fixes and required settings

This spec records every problem found while testing Bangla Whisper speech-to-text in `stt-playground`, what caused
it, and the setting or code that fixes it. Apply all of it when Whisper is deployed in a real voice bot (for example
LocalVoiceAgent). Each fix is needed for a specific reason. The reference implementation is in
`stt-playground/stt_ui.py` (`_raise_whisper_token_cap`) and `stt-playground/whisper_chunking.py`.

Tested 2026-11-27 on an RTX 5090, faster-whisper 1.2.1, CTranslate2 4.9.1, RealtimeSTT from the LocalVoiceAgent
fork. Models tested: our fine-tunes `r-854260e0` (bengaliai-200h) and `r-5af3e11f` (OpenSLR, continued from
r-854260e0), `mozilla-ai/whisper-large-v3-turbo-bn`, and `openai/whisper-large-v3-turbo`.

## Pipeline these rules apply to

```
mic -> RealtimeSTT AudioToTextRecorder (silero VAD, turn ends after post_speech_silence_duration)
    -> faster_whisper engine -> BatchedInferencePipeline.transcribe(vad_filter=True, batch_size=16)
    -> CTranslate2 Whisper.generate (one decode per window)
```

The recorder uses `batch_size=16` by default, which sends every turn through faster-whisper's
`BatchedInferencePipeline`. This path always decodes with `without_timestamps=True`, so a window is decoded once,
and text that is not produced in that pass is lost.

## Summary of required settings

| Setting | Stock value | Required value | Fixes |
|---|---|---|---|
| `WhisperModel.max_length` | 448 (hardcoded) | **880** | P1: transcripts cut at 224 tokens |
| Decoding window length | 30 s | **10 s**, split at pauses | P2: phrase after a pause dropped |
| Forced split of unbroken speech | hard cut | **quietest point + 1 s overlap, remove duplicate words** | P3: words broken at a split |
| Retry of a cut window | none | **re-decode as 5 s windows if it hit the token cap or < 12 tokens/s** | P4: remaining cut windows |
| `ensure_sentence_ends_with_period` | `True` | `False` for Bangla, or log the raw text | P5: cuts look like finished sentences |
| Per-turn audio and window log | none | **save the WAV and the window plan per turn** | P6: cuts cannot be diagnosed from text |
| Streaming decode (optional) | decode whole turn at turn end | **chunks at 0.3 s pauses or every 10 s, flush rest at turn end** | P7: STT latency grows with turn length |
| faster-whisper, ctranslate2 versions | any | **pin 1.2.1 and 4.9.1**; re-run the checks after any upgrade | P1 depends on internals |

The environment variables in the playground are `WHISPER_MAX_LENGTH`, `WHISPER_WINDOW_S`, `WHISPER_RETRY_WINDOW_S`,
`WHISPER_OVERLAP_S`, `WHISPER_MIN_TOK_PER_S`, and `WHISPER_SAFEGUARDS` (0 turns off P3 and P4).

## P1. Transcripts stop at exactly 224 tokens

**Symptom.** Long turns end in the middle of a word, often in the middle of a letter (the text ends in `�`). It
happens with every Whisper model, in both the batched (live) and the plain (file) decoding paths.

**Cause.** faster-whisper sets `self.max_length = 448` in `WhisperModel.__init__` and passes it to CTranslate2.
CTranslate2 halves it before decoding. From `src/models/whisper.cc`, `WhisperReplica::generate`:

```cpp
decoding_options.max_length = std::min(total_max_length / 2, total_max_length - start_step);
```

So the model can generate at most min(448 / 2, 448 - 3) = **224 tokens per window**. The Python docstring only says
`Maximum generation length` and does not mention the halving. faster-whisper's `max_new_tokens` cannot get past this:
it raises `ValueError` when the prompt plus `max_new_tokens` exceeds 448.

**Why only Bangla.** Whisper's tokenizer has few Bangla merges. Each Bangla letter is 3 bytes in UTF-8, so Bangla
costs about 11 tokens per word. Measured: 23 to 31 tokens per second of speech in a window, and about 16 per second
over a whole turn including pauses. English costs about 3 tokens per second. 224 tokens hold about 14 s of Bangla
but more than 60 s of English, which is why LocalVoiceAgent never showed this in English.

**Evidence.**
- Every cut output was exactly 224 tokens, on all four models and in every decoding mode.
- Raw CTranslate2: `max_length=448` gave 224 tokens, and `max_length=300` gave 150.
- With `max_length=600`, the same 18.4 s turn ended by itself at 287 tokens with the full sentence.

**Fix.** Set `max_length = 880` on every `WhisperModel` after construction. The cap becomes min(440, 877) = 440 new
tokens. Added to the 4 prompt tokens, that is 444 decoder positions, inside the model's `max_target_positions` of
448. Do not go above 880: the decoder has no positions past 448.

```python
import faster_whisper

_orig_init = faster_whisper.WhisperModel.__init__

def _init(self, *a, **kw):
    _orig_init(self, *a, **kw)
    self.max_length = 880   # CTranslate2 generates up to max_length // 2 = 440 tokens

faster_whisper.WhisperModel.__init__ = _init
```

Apply the patch at import time in the module that builds the recorder. RealtimeSTT loads the model in a spawned child
process, which re-imports the main module, so a module-level patch runs in the child too.

**Cost.** None. It only removes a limit.

## P2. The phrase after a pause is dropped

**Symptom.** With P1 fixed, a turn can still lose its last phrase or sentence, cleanly at a word boundary and well
under the token cap. Example: a 14.8 s turn lost `সেটা জন্যই মিল্লি টেস্ট করা`, which came after a 0.4 s pause.

**Cause.** With VAD on, the batched pipeline merges speech into windows of up to 30 s (`chunk_length`). Our
fine-tunes and mozilla-ai were trained on single-sentence clips of 2 to 6 s. Every training example was one sentence,
then end-of-text. Inside a long window, the model predicts end-of-text at a pause and drops the speech after it.

**Evidence.**
- The audio holds the phrase: Qwen3-ASR transcribes it from the same WAV.
- Whisper transcribes it when given only the last 4 s.
- Whisper drops it with VAD on or off, on all three Bangla Whisper models.
- With windows of 10 s or less, all three models keep it.

**Fix.** Decode in windows of at most **10 s**, built from VAD speech regions and split at pauses. A pause of about
1 s or more always ends a window. Each window then looks like a training clip. All windows still go through one
batched decode call, so latency does not grow with the number of windows. Minimal form, with stock faster-whisper:

```python
segments, info = pipeline.transcribe(audio, language="bn", vad_filter=True, batch_size=16, chunk_length=10)
```

The full version, with P3 and P4, is `whisper_chunking.plan_windows` and `whisper_chunking.transcribe`. It runs
silero VAD with `min_silence_duration_ms=160` and the default `speech_pad_ms=400`, merges regions greedily up to
10 s, and passes the windows to `BatchedInferencePipeline.transcribe` as `clip_timestamps`.

**Cost.** One extra VAD pass, a few milliseconds.

**Limit.** A 10 s window reduces this risk but does not remove it. Early end-of-text can still happen at a pause
inside a short window. P4 catches those cases.

## P3. Unbroken speech longer than a window

**Symptom.** If a speaker talks for more than 10 s without a pause, the window has to be cut inside speech, and a
word can break across the cut.

**Fix.**
1. Cut at the lowest-energy 20 ms frame in the last 2 s before the 10 s limit. That point is most likely a gap
   between words.
2. Start the next window 1 s before the cut, so both windows hear the words around it.
3. When joining the two texts, find the longest run of up to 8 words at the end of the first text that matches the
   start of the second. Match after removing punctuation, with a fuzzy ratio of at least 0.75 (the cut word decodes
   slightly differently in each window). Drop that run from the second text.

Code: `whisper_chunking.plan_windows` (`_quietest_split`, `overlap_prev`) and `whisper_chunking.join_overlap`.

**Cost.** About 1 s of extra audio per forced split, decoded in the same batch, so about 0 to 20 ms.

**Limit.** Duplicate removal is imperfect when the words at the cut decode differently in the two windows. In a test
with 4 s windows forced on synthetic joined audio, some splits left a duplicated or garbled word. Real turns with
10 s windows rarely need a forced split: none of the recorded turns did.

## P4. Detecting and retrying a window that was still cut

**Symptom.** Any window that P1 to P3 did not save: a cap hit on very fast or long speech, or an early end-of-text
inside a 10 s window.

**Fix.** After the batched decode, check each window:
- **Token cap:** the window produced at least `cap - 2` tokens (438 with P1, 222 stock).
- **Too few tokens:** the window is at least 2.5 s long and produced fewer than **12 tokens per second of window**.
  Complete Bangla windows measured 23 to 31 tokens per second. Windows that had lost whole sentences measured 8 to 10.

A flagged window is decoded again as 5 s sub-windows, with the P2 and P3 rules. The retry replaces the original only
if it produced more tokens and its compression ratio is at most 2.4 (faster-whisper's repetition-loop threshold).
Code: `whisper_chunking._suspicious` and the retry loop in `whisper_chunking.transcribe`.

**Evidence.**
- 25 s of joined dataset clips: all three 10 s windows were flagged (8.3 to 10.5 tokens/s), and the retries
  recovered the missing sentences (100 to 241, 59 to 160, and 72 to 139 tokens).
- With the stock 224 cap and 30 s windows, the retry alone recovered the full text of both recorded turns.
- No retries fired on the two real recorded turns at default settings.

**Cost.** Nothing on a normal turn: the checks use token counts that are already in each segment. A flagged turn pays
one extra decode of the flagged windows, about 100 to 600 ms on this GPU.

**Not used:** a check for speech after the last decoded word. It needs `word_timestamps=True`, an alignment pass on
every turn. Its alignment heads come from the base model and may be off after fine-tuning.

## P5. The recorder adds a period to every result

**Symptom.** Cut text such as `মাঝে-মধ` arrives as `মাঝে-মধ.`, so a cut looks like a finished sentence.

**Cause.** RealtimeSTT `AudioToTextRecorder(ensure_sentence_ends_with_period=True)` is the default. Bangla sentences
end with `।`, not `.`, so the added `.` is wrong anyway.

**Fix.** Set `ensure_sentence_ends_with_period=False` for Bangla, or at least log the text before this step. The
playground leaves it on to match LocalVoiceAgent's behaviour.

## P6. Diagnostics

Text alone could not tell a cut audio buffer from a decoding problem. Log the following for every turn:
- **The audio sent to the engine.** RealtimeSTT keeps it in `recorder.last_transcription_bytes` (16 kHz float32).
  The playground writes it to `logs/audio/*.wav`.
- **The window plan.** Start, end and token count of each window, which splits were forced, and every retry with its
  reason. The playground writes this as a `whisper_windows` record in `logs/playground.jsonl`.
- **The settings in effect:** `whisper_max_length` and `window_s`.

With those, a cut is easy to classify. If the WAV holds the missing words, the problem is decoding (P1 to P4). If it
doesn't, the problem is endpointing or the audio buffer.

## P7. Optional: decode chunks while the speaker talks (streaming mode)

**Problem.** RealtimeSTT decodes the whole turn after the turn-end silence. The STT time after the endpoint is the
decode of the whole turn, about 0.35 s for a 15 to 18 s Bangla turn, and it grows with the turn length.

**Fix (optional, a different design).** Keep the recorder for capture, VAD and turn end, so endpointing does not
change. While a turn is recorded, a side thread reads the recorder's frame buffer every 100 ms
(`RealtimeSTT.core.recording_buffers.snapshot_frames(recorder)`, int16) and commits a chunk when:
1. the speaker paused for at least the **chunk pause (0.3 s)** after at least **1.5 s** of uncommitted audio. The
   chunk ends in the pause. The pause is found with silero VAD (`min_silence_duration_ms` = chunk pause,
   `speech_pad_ms=100`). The chunk pause must be shorter than the turn-end silence; the playground caps it at 75%;
2. 10 s pass without such a pause. The chunk ends at the quietest 20 ms in the last 2 s, the next chunk starts 1 s
   earlier, and the duplicated words are removed as in P3.

Each chunk is decoded by the recorder's own engine, `recorder.perform_final_transcription(audio_bytes=chunk)`, so the
same model in the same child process decodes it, and no second copy is loaded. At turn end the main loop calls
`recorder.wait_audio()` instead of `recorder.text()`: it waits for the turn without decoding it. It then waits for a
chunk decode still in flight and decodes only the uncommitted rest, skipping it when VAD finds no speech in it.
`recorder.text()` is `wait_audio()` plus `transcribe()`, so a turn-end-mode turn can be decoded from the same loop
with `recorder.transcribe()`.

Joining the chunks:
- Strip the period that `ensure_sentence_ends_with_period` adds to every chunk except the last (P5).
- After a forced split, the overlap can start in the middle of a word (`করতেছিলাম` decodes as `আছিলাম`). The duplicate
  removal must also compare the start of the next chunk with the same-length end of the previous text, not only
  whole words.

**Cost.** One silero VAD pass over at most 10 s of audio every 100 ms on the CPU, and the GPU decodes while the user
speaks. Chunks are decoded without the text before them, so a word at a chunk edge can come out differently than in
a whole-turn decode. In the tests the text matched the turn-end decode apart from such words.

**Measured (RTX 5090, r-5af3e11f, beam 5, turn-end silence 0.6 s, real turns fed at real-time pace).**

| Turn | Turn-end mode: STT, speech end to text | Streaming: chunks, STT, speech end to text |
|---|---|---|
| 18.4 s | 375 ms, 1081 ms | 3 chunks (all at pauses), 2 ms, 609 ms |
| 14.8 s | 341 to 346 ms, 947 to 955 ms | 2 chunks (one forced split), 3 to 4 ms, 609 to 610 ms |
| 21 s | 373 ms, 981 ms | 3 chunks, 2 ms, 607 ms |

With streaming, almost all of the speech-end-to-text time is the turn-end silence wait. The playground's live tab has a
toggle for this mode (applied per turn), and the log records `streaming`, `chunks` (start, end, reason, decode
time and text of each chunk), `flush_wait_s` and `flush_decode_s`.

## Required checks after an upgrade or a model change

The P1 fix depends on CTranslate2's halving rule and on faster-whisper's hardcoded 448. Re-run these checks after
upgrading either library, and after switching to a new checkpoint.

1. **Token cap.** Decode an 18 s Bangla turn with `max_length=448`: the output must stop at exactly 224 tokens. With
   880, it must end naturally, under 440 tokens. If the 448 result changes, the halving rule changed: recompute the
   value.
2. **Pause drop.** Decode a turn with a pause before the last phrase. The phrase must appear with 10 s windows.
3. **Latency.** A normal 15 to 18 s turn must cost about the same with and without the safeguards (measured 0.37 vs
   0.37 s and 0.32 vs 0.31 s on the RTX 5090).
4. **No false retries.** Normal turns must log `retries: []`.

The recorded test turns are `stt-playground/logs/audio/20261127-054133-755.wav` (18.4 s) and
`20261127-055458-311.wav` (14.8 s). The log directory is gitignored, so copy them somewhere durable if they are
needed later.

## Measured latency (RTX 5090, r-854260e0, beam 5, mean of 3 runs)

| Audio | max_length 880, `chunk_length=10` only | With P2 to P4 | Notes |
|---|---|---|---|
| 18.4 s real turn | 0.37 s | 0.37 s | no retry |
| 14.8 s real turn | 0.31 s | 0.32 s | no retry |
| 25 s joined clips | 0.30 s, sentences missing | 0.89 s, complete | 3 windows retried |
| 18.4 s turn, stock cap 224, 30 s window | cut at 224 tokens | 0.64 s, complete | retry on token cap |

## Longer term: training

- **Long clips (optional, fixes P2 at the source).** Join 2 to 4 same-speaker clips into 10 to 25 s training
  examples. Keep labels under 440 tokens, about 27 s of Bangla: the recipe's `MAX_LABEL_TOKENS = 448` drops longer
  ones.
- **Timestamps (optional).** The current recipe trains with `<|notimestamps|>`
  (`backend/app/core/backend_kit/whisper.py`, `set_prefix_tokens(language, task)`), so the models output one segment
  per window. Timestamped labels can be built from joined clips, because each clip's start and end are known. That
  lets plain sequential decoding resume after a cut, but it does not help the batched pipeline. Mix about 50% with
  timestamps and 50% without.
- **Neither replaces P1.** The 224-token cap is a decoding setting, and no training changes it.

## What not to rely on

- **The openai base model** (`openai/whisper-large-v3-turbo`) is weak in Bangla. On long or short windows it loops
  (`কেন কেন কেন…`) or switches script (Latin, Khmer). Don't use it to judge the pipeline.
- **`max_new_tokens`** cannot raise the cap: faster-whisper rejects values whose total exceeds 448.
- **Plain long-form decoding with timestamps** cannot resume after a cut with the current models: they emit one
  segment per window, so there is no timestamp to resume from.
