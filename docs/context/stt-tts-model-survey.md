# STT and TTS models from LocalVoiceAgent: open weights, fine-tuning and Bangla

Researched 2026-11-23. The model list comes from LocalVoiceAgent's `server/voicebot/catalog.toml` and `bench/pool.toml`. License information comes from the vendored docs (`RealtimeSTT/docs/licenses.md`, `RealtimeTTS/README.md`, `RealtimeTTS/docs/engines/qwen.md`, `RealtimeTTS/docs/engines/inflect.md`). The vendored docs say nothing about fine-tuning or Bangla, so those columns were checked on the web (sources at the end). Upstream model cards change; re-check before committing to a model.

## Can fine-tuning teach a model Bangla?

Yes. "Does not support Bangla" means the model starts further away, not that it is ruled out. How far depends on three things.

1. **Can the model represent Bangla text?**
   - Byte-level tokenizers (Whisper, Qwen3-ASR, Qwen3-TTS) can encode any script, including Bengali, at the cost of more tokens per word.
   - Fixed vocabularies (Parakeet's SentencePiece, built for European languages) map Bengali characters to `<unk>`. The tokenizer must be replaced, which re-initialises the decoder and joint network, close to training them from scratch.
   - Phoneme front-ends (Kokoro, Piper, Inflect) need a Bangla grapheme-to-phoneme (G2P) step. espeak-ng has Bengali; Kokoro's misaki does not.
2. **How much transfers?** STT encoders learn general acoustics, and Bangla overlaps a lot with Hindi and other Indic languages. TTS audio codecs and vocoders are mostly language-agnostic, so the model mainly has to learn the text-to-sound mapping and Bangla prosody.
3. **How much data?** Rough rules of thumb, not measurements:

| Starting point | STT | TTS (one speaker) |
|---|---|---|
| Already knows Bangla, needs polishing | 10–100 h | 1–10 h |
| Knows the script or related languages, not Bangla | 100–500 h | 20–50 h |
| New tokenizer or new script | 1000+ h | 50+ h, plus a new G2P |

Use full fine-tuning rather than LoRA when adding a language; LoRA works well for adapting a language the model already knows. Small models (Kokoro, 82M parameters) have little spare capacity and may degrade when a language is added.

## STT

| Model (LocalVoiceAgent slug) | Weights and license | Fine-tuning | Bangla |
|---|---|---|---|
| Whisper `small.en`, `tiny.en` (`whisper-small-en`, `whisper-tiny-en-rt`) | Open, MIT | Yes (HF transformers, Unsloth) | No, English only. Do not use for Bangla. |
| Whisper `large-v3-turbo` (`whisper-large-v3-turbo`) | Open, MIT | Yes. Proven for Bengali: `mozilla-ai/whisper-large-v3-turbo-bn` went from 78.8% to 11.05% WER on Common Voice Bengali. Convert the result to CTranslate2 for RealtimeSTT's faster_whisper engine. | Listed as a Whisper language, but zero-shot WER is above 80% in one paper. |
| Qwen3-ASR-1.7B (`qwen3-asr-1-7b`) | Open, Apache-2.0 | Yes. The official `qwen-asr` package supports SFT on one or more GPUs. | Not among its 30 languages (GitHub issue #203 requests it). Hindi is supported. |
| Nemotron 3.5 ASR streaming 0.6B (`nemotron-rt`, sherpa int8 bundle) | Open, OpenMDW-1.1 (community ports list it as commercial OK; not confirmed on NVIDIA's own card) | Yes, with NeMo on the original `nvidia/nemotron-3.5-asr-streaming-0.6b` checkpoint, not the int8 ONNX bundle. NVIDIA publishes a guide for fine-tuning it to a new language; a paper adapted it to Kikuyu, Dholuo and Kalenjin. | Not found among its ~40 locales. Hindi is supported. |
| Parakeet TDT 0.6B v3 (`transcribe-cpp-parakeet-q4`, `sherpa-parakeet-v3`) | Open, CC-BY-4.0 (attribution required) | Only from the NeMo checkpoint; the GGUF and ONNX copies are inference only. Bangla needs a new tokenizer, which resets the decoder and joint network. A community Malay run that swapped the tokenizer collapsed; Malay then worked with the original tokenizer, which Bengali script cannot do. | No, 25 European languages. |
| Kroko EN 128 (`kroko-en-128-rt`) | Community models CC-BY-SA; free license keys non-commercial | No public training code found. The project mentions "DIY training guides" and paid fine-tuning. | No |

## TTS

| Model (LocalVoiceAgent slug) | Weights and license | Fine-tuning | Bangla |
|---|---|---|---|
| Qwen3-TTS 0.6B Base (`qwen-cuda`, run as GGUF Q8_0 through qwentts.cpp) | Open, Apache-2.0 | Yes. Official `finetuning/` folder in QwenLM/Qwen3-TTS (`prepare_data.py`, then `sft_12hz.py`). Base models only, single speaker only. Train the HF weights, then convert back to GGUF. | No, 10 languages. One community note says a new language may need codec vocabulary changes (unverified). |
| Kokoro-82M (`kokoro`) | Open, Apache-2.0 | No official training code. Community StyleTTS2-based recipes exist (kikiri-tts / kokoro-deutsch), with Indic precedents: a Kannada fine-tune on about 11 h and a Marathi one. The main bottleneck is a G2P for the language. | No |
| Inflect Micro-v2 (`inflect-cuda`) | Open, Apache-2.0 (its phonemizer dependency is GPLv3+) | The model card describes an adaptation toolkit for a fixed voice, labelled experimental. A community trainer exists (`scrappylabsai/inflect-trainer`). | No |
| Piper `en_US-lessac-medium` (`piper`) | Code: MIT for the old rhasspy/piper, GPL-3.0 for piper1-gpl. The voice's license follows its dataset; check the lessac model card. | Yes. `piper1-gpl` trains from a base checkpoint (`piper.train fit`) and exports to ONNX. VITS fine-tunes well on little data. | No Bengali voice found. IndicTTS has about 12 h of studio Bangla per voice as a possible corpus (license not checked). |
| Chatterbox multilingual (`chatterbox`) | Open, MIT | No official recipe. A community guide adds a language by extending `SUPPORTED_LANGUAGES` and fine-tuning the T3 model, with 10–50 h of audio. | No, 23 languages. Hindi is supported. |
| XTTS v2 (`coqui-xtts-v2`) | Open, Coqui Public Model License (non-commercial; Coqui shut down, so no commercial license can be bought) | Yes (Coqui trainer). BnTTS built Bangla on it, but with 3.85k hours of continued pretraining. | No, 17 languages. Hindi is supported. |

## Recommendations

- **STT:** fine-tune Whisper `large-v3-turbo` first: MIT, Unsloth support, proven Bengali results, and it drops into RealtimeSTT after CT2 conversion. `mozilla-ai/whisper-large-v3-turbo-bn` is a baseline to compare against or start from. Nemotron 3.5 is a second track for low-latency streaming partials.
- **TTS:** no model in the LocalVoiceAgent pool speaks Bangla.
  - Qwen3-TTS has an official fine-tune path and already runs in RealtimeTTS, but Bangla would be a new language and is unproven.
  - Kokoro and Piper are cheaper bets with Indic precedents.
  - Other Bangla starting points: `bn-parler-tts` (an Indic Parler-TTS fine-tune) and IndicF5 (see `localvoiceagent.md`; it has no RealtimeTTS engine yet).
- **Licensing:** avoid XTTS v2 and Kroko Community models for production. In the platform they must carry `commercial_ok = false`.

## Sources

- Qwen3-ASR: https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf, https://github.com/QwenLM/Qwen3-ASR/issues/203, https://pypi.org/project/qwen-asr/, https://qwen.ai/blog?id=qwen3asr
- Qwen3-TTS: https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/HEAD/README.md, https://mintlify.com/QwenLM/Qwen3-TTS/advanced/fine-tuning, https://www.mintlify.com/QwenLM/Qwen3-TTS/concepts/languages
- Whisper Bengali: https://huggingface.co/mozilla-ai/whisper-large-v3-turbo-bn, https://huggingface.co/BuzzASR/bengali, https://huggingface.co/KhushiDS/whisper-large-v3-Bengali
- Nemotron 3.5: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b, https://huggingface.co/blog/nvidia/fine-tuning-nemotron-35-asr, https://arxiv.org/pdf/2607.18912, https://together.ai/models/nvidia-nemotron-35-asr
- Parakeet and NeMo: https://docs.nvidia.com/nemo/speech/3.0.0/asr/fine_tuning.html, https://huggingface.co/Kim-el/failed-swapped-to-malay-tokenizer, https://build.nvidia.com/nvidia/parakeet-tdt-0_6b/modelcard
- Kroko: https://cloudfare-proxy.pages.dev/Banafo/Kroko-ASR (mirror of the Hugging Face page)
- Kokoro: https://huggingface.co/hexgrad/Kokoro-82M/discussions/10, https://huggingface.co/Sammy-Mohan/Kokoro-82M-Kannada, https://huggingface.co/Thorsten-Voice/Kokoro, https://awesome.ecosyste.ms/projects/github.com%2Fshreyaskarnik%2Fbol-tts-marathi
- Inflect: https://huggingface.co/owensong/Inflect-Micro-v2, https://huggingface.co/scrappylabsai/clara-voice-1
- Chatterbox: https://resemble.ai/introducing-chatterbox-multilingual-open-source-tts-for-23-languages, https://www.mintlify.com/yocxy2/chatterboxyocxy/api/supported-languages
- XTTS v2 and BnTTS: https://huggingface.co/coqui/XTTS-v2/raw/v2.0.3/README.md, https://arxiv.org/html/2502.05729v1
- Bangla TTS data and models: https://arxiv.org/pdf/2609.29146, https://huggingface.co/Tanmoyshome/bn-parler-tts, https://huggingface.co/quangdung/Piper_checkpoint (Piper fine-tune workflow example)
