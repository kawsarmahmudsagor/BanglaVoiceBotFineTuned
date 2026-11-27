import { act, fireEvent, render, screen } from "@testing-library/react";
import { ReviewCard, wordDiff } from "../components/ReviewCard";
import type { ReviewClip } from "../types";

// jsdom has no Web Audio or canvas: the waveform is drawn in the browser only.
vi.mock("../components/Waveform", () => ({ Waveform: () => <div data-testid="waveform" /> }));

const base: ReviewClip = {
  dataset_id: "ds1",
  id: "v1_0003",
  video_id: "v1",
  idx: 3,
  start_s: 12,
  end_s: 16.5,
  dnsmos: 3.4,
  speaker_sim: 0.8,
  drop_reason: null,
  text_a: "আমি ভাত খাই",
  text_b: "আমি ভাত খেয়েছি",
  cer: 0.12,
  text_en: null,
  text_translated: null,
  text_final: "আমি ভাত খাই।",
  text_source: "stt",
  status: "flagged",
  no_punct: 0,
  flag: null,
  text_pick: null,
  score_a: null,
  score_b: null,
  reason: "cer",
};

function setup(clip: ReviewClip = base) {
  const props = {
    clip,
    onApprove: vi.fn(),
    onReject: vi.fn(),
    onNext: vi.fn(),
    onPrev: vi.fn(),
    onEdit: vi.fn(),
  };
  render(<ReviewCard {...props} />);
  return props;
}

afterEach(() => {
  vi.useRealTimers();
});

// approve/reject/save settle asynchronously, so let their state updates land inside act
const press = (target: Element, init: KeyboardEventInit) =>
  act(async () => {
    fireEvent.keyDown(target, init);
  });

test("Enter approves", async () => {
  const p = setup();
  await press(document.body, { key: "Enter" });
  expect(p.onApprove).toHaveBeenCalledWith("আমি ভাত খাই।");
});

test("R rejects", async () => {
  const p = setup();
  await press(document.body, { key: "r" });
  expect(p.onReject).toHaveBeenCalledTimes(1);
});

test("J and K navigate", () => {
  const p = setup();
  fireEvent.keyDown(document.body, { key: "j" });
  fireEvent.keyDown(document.body, { key: "k" });
  fireEvent.keyDown(document.body, { key: "J" });
  expect(p.onNext).toHaveBeenCalledTimes(2);
  expect(p.onPrev).toHaveBeenCalledTimes(1);
});

test("keys ignored while typing in the text box", async () => {
  const p = setup();
  const box = screen.getByRole("textbox", { name: /final text/i });
  box.focus();
  for (const key of ["Enter", "r", "j", "k", " "]) await press(box, { key });
  expect(p.onApprove).not.toHaveBeenCalled();
  expect(p.onReject).not.toHaveBeenCalled();
  expect(p.onNext).not.toHaveBeenCalled();
  expect(p.onPrev).not.toHaveBeenCalled();
  await press(box, { key: "Enter", ctrlKey: true });
  expect(p.onApprove).toHaveBeenCalledTimes(1);
});

test("edit is saved after debounce", async () => {
  vi.useFakeTimers();
  const p = setup();
  const box = screen.getByRole("textbox", { name: /final text/i });
  fireEvent.change(box, { target: { value: "আমি ভাত" } });
  fireEvent.change(box, { target: { value: "আমি ভাত খেয়েছি।" } });
  act(() => {
    vi.advanceTimersByTime(499);
  });
  expect(p.onEdit).not.toHaveBeenCalled();
  await act(async () => {
    vi.advanceTimersByTime(1);
  });
  expect(p.onEdit).toHaveBeenCalledTimes(1);
  expect(p.onEdit).toHaveBeenCalledWith("আমি ভাত খেয়েছি।");
});

test("translated clip shows English and translation", () => {
  setup({
    ...base,
    reason: "translated",
    status: "needs_review",
    text_source: "translated",
    text_en: "I eat rice.",
    text_translated: "আমি ভাত খাই।",
  });
  expect(screen.getByText("I eat rice.")).toBeInTheDocument();
  expect(screen.getByLabelText(/LLM translation/i)).toHaveTextContent("আমি ভাত খাই।");
});

test("approve error is shown inline", async () => {
  const p = setup({ ...base, text_final: "", text_a: "" });
  p.onApprove.mockRejectedValueOnce(new Error("cannot approve a clip with empty text"));
  await press(document.body, { key: "Enter" });
  expect(screen.getByRole("alert")).toHaveTextContent("cannot approve a clip with empty text");
});

test("word diff marks tokens only in A and only in B", () => {
  const { a, b } = wordDiff("আমি ভাত খাই", "আমি ভাত খেয়েছি");
  expect(a).toEqual([
    { text: "আমি", changed: false },
    { text: "ভাত", changed: false },
    { text: "খাই", changed: true },
  ]);
  expect(b.filter((t) => t.changed).map((t) => t.text)).toEqual(["খেয়েছি"]);
});
