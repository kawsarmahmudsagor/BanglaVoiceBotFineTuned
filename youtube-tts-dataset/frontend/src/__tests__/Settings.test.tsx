import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { vi } from "vitest";
import { Settings } from "../pages/Settings";
import * as api from "../api";
import type { AppSettings } from "../types";

vi.mock("../api");

const settings: AppSettings = {
  thresholds: {
    min_bn_ratio: 0.5, min_chars_per_s: 3, max_chars_per_s: 30, max_word_repeat: 6, dnsmos_min: 3,
    speaker_sim_min: 0.6, clip_min_s: 15, clip_max_s: 25, piece_max_s: 10, piece_gap_s: 0.3, tail_min_s: 5,
    disagree_cer: 0.25, tie_margin: 0.05, demucs_segment_s: 7,
  },
  cookies_path: "",
  llm_host: "http://localhost:11434",
  llm_model: "gemma",
  models: { separate: "s", vad: "v", dnsmos: "d", speaker: "sp", model_a: "Qwen", model_b: "Whisper", llm: "gemma" },
  transcription: {
    a: { current: "qwen3-asr-bn", options: [{ id: "qwen3-asr-bn", label: "Qwen3 ASR Bangla", available: true, expected: "C:\a" }] },
    b: {
      current: "whisper-bn-v3",
      options: [
        { id: "whisper-bn-v3", label: "Whisper v3 Bangla", available: true, expected: "C:\b" },
        { id: "whisper-bn-mozilla", label: "Whisper Mozilla", available: false, expected: "C:\moz" },
      ],
    },
  },
};

beforeEach(() => {
  vi.mocked(api.getSettings).mockResolvedValue(settings);
  vi.mocked(api.putSettings).mockResolvedValue(settings);
});

test("model dropdowns show current values and disable missing options", async () => {
  render(<Settings dataset={null} onDatasetChanged={() => {}} />);
  const a = (await screen.findByLabelText("Model A")) as HTMLSelectElement;
  const b = screen.getByLabelText("Model B") as HTMLSelectElement;
  expect(a.value).toBe("qwen3-asr-bn");
  expect(b.value).toBe("whisper-bn-v3");
  const missing = screen.getByRole("option", { name: /missing: C:\moz/ }) as HTMLOptionElement;
  expect(missing.disabled).toBe(true);
  expect(screen.getByText(/Takes effect from the next transcription run/)).toBeTruthy();
  expect(document.body.textContent).not.toMatch(/translated/i);
});

test("changing Model B saves only that field", async () => {
  vi.mocked(api.getSettings).mockResolvedValue({
    ...settings,
    transcription: { ...settings.transcription, b: { ...settings.transcription.b,
      options: settings.transcription.b.options.map((o) => ({ ...o, available: true })) } },
  });
  render(<Settings dataset={null} onDatasetChanged={() => {}} />);
  const b = await screen.findByLabelText("Model B");
  fireEvent.change(b, { target: { value: "whisper-bn-mozilla" } });
  await waitFor(() => expect(api.putSettings).toHaveBeenCalledWith({ model_b: "whisper-bn-mozilla" }));
});
