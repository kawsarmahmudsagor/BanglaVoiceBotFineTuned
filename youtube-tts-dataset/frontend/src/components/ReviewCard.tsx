import { useEffect, useRef, useState } from "react";
import { clipAudioUrl, patchClip } from "../api";
import type { ReviewClip, ReviewReason } from "../types";
import { formatDuration } from "./ProgressRow";
import { Waveform } from "./Waveform";

export const SAVE_DEBOUNCE_MS = 500;

export const REASON_LABEL: Record<ReviewReason, string> = {
  cer: "Flagged",
  translated: "Translated",
  pending_llm: "Waiting for LLM",
  auto: "Auto-accepted",
  disagree: "Models disagree",
  review: "Needs review",
};

export interface DiffToken {
  text: string;
  changed: boolean;
}

// Punctuation is ignored when matching words: model B and the LLM add it, model A may not.
const PUNCT = /[।॥,.?!;:"'“”‘’()[\]{}\-–—]/g;
const tokens = (s: string | null) => (s ?? "").trim().split(/\s+/).filter(Boolean);
const key = (t: string) => t.replace(PUNCT, "");

/** Word-level diff of A and B: a token is `changed` when it is not part of their longest common subsequence. */
export function wordDiff(a: string | null, b: string | null): { a: DiffToken[]; b: DiffToken[] } {
  const ta = tokens(a);
  const tb = tokens(b);
  const ka = ta.map(key);
  const kb = tb.map(key);
  // lcs[i][j] = LCS length of ka[i..] and kb[j..]
  const lcs = Array.from({ length: ta.length + 1 }, () => new Array<number>(tb.length + 1).fill(0));
  for (let i = ta.length - 1; i >= 0; i--)
    for (let j = tb.length - 1; j >= 0; j--)
      lcs[i][j] = ka[i] === kb[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
  const sameA = new Array<boolean>(ta.length).fill(false);
  const sameB = new Array<boolean>(tb.length).fill(false);
  for (let i = 0, j = 0; i < ta.length && j < tb.length; ) {
    if (ka[i] === kb[j]) {
      sameA[i++] = true;
      sameB[j++] = true;
    } else if (lcs[i + 1][j] >= lcs[i][j + 1]) i++;
    else j++;
  }
  return {
    a: ta.map((text, i) => ({ text, changed: !sameA[i] })),
    b: tb.map((text, i) => ({ text, changed: !sameB[i] })),
  };
}

function DiffText({ parts, side }: { parts: DiffToken[]; side: "a" | "b" }) {
  if (!parts.length) return <span className="muted">No text</span>;
  return (
    <>
      {parts.map((t, i) => (
        <span key={i}>
          {i > 0 && " "}
          {t.changed ? <mark className={`diff diff--${side}`}>{t.text}</mark> : t.text}
        </span>
      ))}
    </>
  );
}

function message(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

type SaveState = "idle" | "pending" | "saving" | "saved" | "error";

interface Props {
  clip: ReviewClip;
  /** Saves `text` as the final text and approves; a rejection is shown inline (e.g. 422 for empty text). */
  onApprove: (text: string) => unknown;
  onReject: () => unknown;
  onNext: () => void;
  onPrev: () => void;
  /** Saves the final text; called 500 ms after the last keystroke. */
  onEdit: (text: string) => unknown;
  videoTitle?: string | null;
  /** Called with the server's clip after a model swap. */
  onSwapped?: (clip: ReviewClip) => void;
}

/**
 * One clip to review. Keys: Space play/pause, Enter approve, R reject, J next, K previous. They are ignored
 * while the final-text box has focus, except Ctrl+Enter which approves. Render with `key={clip.id}`.
 */
export function ReviewCard({ clip, onApprove, onReject, onNext, onPrev, onEdit, videoTitle, onSwapped }: Props) {
  const [text, setText] = useState(
    clip.text_final ?? (clip.text_pick === "b" ? clip.text_b : clip.text_a) ?? "",
  );
  const [save, setSave] = useState<SaveState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [audio, setAudio] = useState<HTMLAudioElement | null>(null);
  const boxRef = useRef<HTMLTextAreaElement>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const unsaved = useRef<string | null>(null);
  const mounted = useRef(true);

  const props = useRef({ onApprove, onReject, onNext, onPrev, onEdit });
  props.current = { onApprove, onReject, onNext, onPrev, onEdit };
  const textRef = useRef(text);
  textRef.current = text;

  function saveNow(value: string) {
    clearTimeout(timer.current);
    unsaved.current = null;
    setSave("saving");
    let result: unknown;
    try {
      result = props.current.onEdit(value);
    } catch (e) {
      result = Promise.reject(e);
    }
    Promise.resolve(result).then(
      () => {
        if (mounted.current) setSave("saved");
      },
      (e: unknown) => {
        if (!mounted.current) return;
        setSave("error");
        setError(`Edit not saved: ${message(e)}`);
      },
    );
  }

  function flush() {
    if (unsaved.current !== null) saveNow(unsaved.current);
  }

  function change(value: string) {
    setText(value);
    setError(null);
    setSave("pending");
    unsaved.current = value;
    clearTimeout(timer.current);
    timer.current = setTimeout(() => saveNow(value), SAVE_DEBOUNCE_MS);
  }

  async function decide(kind: "approve" | "reject") {
    if (busy) return;
    setBusy(true);
    setError(null);
    const pending = unsaved.current;
    try {
      if (kind === "approve") {
        // approving saves the text in the same request
        clearTimeout(timer.current);
        unsaved.current = null;
        await props.current.onApprove(textRef.current);
      } else {
        flush();
        await props.current.onReject();
      }
    } catch (e) {
      if (!mounted.current) return;
      setError(message(e));
      if (kind === "approve" && pending !== null) saveNow(textRef.current); // the edit still needs saving
    } finally {
      if (mounted.current) setBusy(false);
    }
  }

  const [pick, setPick] = useState(clip.text_pick);
  async function swap(which: "a" | "b") {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      const res = await patchClip(clip.dataset_id, clip.id, { text_pick: which });
      if (!mounted.current) return;
      clearTimeout(timer.current);
      unsaved.current = null;
      setText(res.text_final ?? "");
      setPick(res.text_pick ?? which);
      setSave("idle");
      onSwapped?.(res);
    } catch (e) {
      if (mounted.current) setError(`Could not swap the text: ${message(e)}`);
    } finally {
      if (mounted.current) setBusy(false);
    }
  }

  function togglePlay() {
    if (!audio) return;
    if (audio.paused) audio.play().catch((e: unknown) => setError(`Could not play the clip: ${message(e)}`));
    else audio.pause();
  }

  // unmount (next clip): keep pending edits
  useEffect(() => {
    mounted.current = true; // StrictMode runs effects twice
    return () => {
      mounted.current = false;
      if (unsaved.current !== null) {
        clearTimeout(timer.current);
        void Promise.resolve(props.current.onEdit(unsaved.current)).catch(() => {});
      }
    };
  }, []);

  const keyHandler = useRef<(e: KeyboardEvent) => void>(() => {});
  keyHandler.current = (e: KeyboardEvent) => {
    const t = e.target instanceof HTMLElement ? e.target : null;
    if (t && t === boxRef.current) {
      if (e.key === "Enter" && e.ctrlKey && !e.repeat) {
        e.preventDefault();
        void decide("approve");
      }
      return;
    }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const tag = t?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || t?.isContentEditable) return;
    const onControl = tag === "BUTTON" || tag === "A" || tag === "SUMMARY"; // Space/Enter activate them
    switch (e.key) {
      case " ":
        if (onControl) return;
        e.preventDefault();
        togglePlay();
        break;
      case "Enter":
        if (onControl || e.repeat) return;
        e.preventDefault();
        void decide("approve");
        break;
      case "r":
      case "R":
        if (!e.repeat) void decide("reject");
        break;
      case "j":
      case "J":
        props.current.onNext();
        break;
      case "k":
      case "K":
        props.current.onPrev();
        break;
    }
  };
  useEffect(() => {
    const on = (e: KeyboardEvent) => keyHandler.current(e);
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, []);

  // text_b exists only in datasets transcribed before model B was removed
  const hasB = clip.text_b != null;
  const diff = wordDiff(clip.text_a, clip.text_b);
  const translated = clip.text_source === "translated" || clip.reason === "translated" || clip.reason === "pending_llm";
  const titleId = `clip-${clip.id}-title`;
  const errorId = `clip-${clip.id}-error`;
  const saveText: Record<SaveState, string> = {
    idle: "",
    pending: "Unsaved changes",
    saving: "Saving…",
    saved: "Saved",
    error: "Not saved",
  };

  return (
    <article className="rcard panel" aria-labelledby={titleId}>
      <header className="rcard__head">
        <div className="rcard__title">
          <h2 id={titleId}>
            Clip {clip.idx + 1}{" "}
            <span className="muted num">
              {formatDuration(clip.orig_start_s ?? clip.start_s)} to {formatDuration(clip.orig_end_s ?? clip.end_s)}
            </span>
          </h2>
          <span className="rcard__video" lang="bn" title={clip.video_id}>
            {videoTitle || clip.video_id}
          </span>
        </div>
        <span className={`pill pill--reason-${clip.reason}`}>{REASON_LABEL[clip.reason]}</span>
        {(clip.status === "approved" || clip.status === "rejected") && (
          <span className={`pill pill--${clip.status}`}>{clip.status}</span>
        )}
        <dl className="rcard__metrics num">
          {clip.cer != null && (
            <div>
              <dt>CER</dt>
              <dd>{`${Math.round(clip.cer * 100)}%`}</dd>
            </div>
          )}
          <div>
            <dt>DNSMOS</dt>
            <dd>{clip.dnsmos == null ? "–" : clip.dnsmos.toFixed(2)}</dd>
          </div>
          <div>
            <dt>Length</dt>
            <dd>{(clip.end_s - clip.start_s).toFixed(1)} s</dd>
          </div>
        </dl>
      </header>

      <div className="rcard__player">
        <button type="button" className="btn btn--play" onClick={togglePlay}
          aria-label={playing ? "Pause clip" : "Play clip"} aria-keyshortcuts="Space">
          <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
            {playing ? <path d="M4 2h3v12H4zM9 2h3v12H9z" fill="currentColor" /> : <path d="M4 2l10 6-10 6z" fill="currentColor" />}
          </svg>
        </button>
        <Waveform datasetId={clip.dataset_id} clipId={clip.id} audio={audio} />
        <audio ref={setAudio} src={clipAudioUrl(clip.dataset_id, clip.id)} preload="auto"
          onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)} />
      </div>

      <div className="rcard__compare">
        <section aria-label="Transcript">
          <h3>
            {clip.text_pick != null ? "Model A" : "Transcript"}
            {pick === "a" && <span className="pill pill--approved"> kept</span>}
          </h3>
          <p lang="bn">
            {hasB ? <DiffText parts={diff.a} side="a" /> : clip.text_a || <span className="muted">No text</span>}
          </p>
        </section>
        {hasB && (
          <section aria-label="Model B transcript">
            <h3>
              Model B
              {pick === "b" && <span className="pill pill--approved"> kept</span>}
            </h3>
            <p lang="bn">
              <DiffText parts={diff.b} side="b" />
            </p>
          </section>
        )}
      </div>

      {clip.text_pick != null && (
        <div className="rcard__swap">
          <button type="button" className="btn btn--small" disabled={busy || !clip.text_a} onClick={() => void swap("a")}>
            Use model A
          </button>
          <button type="button" className="btn btn--small" disabled={busy || !clip.text_b} onClick={() => void swap("b")}>
            Use model B
          </button>
          <p className="field__hint">Punctuation is not re-applied to the swapped text.</p>
        </div>
      )}

      {translated && (
        <div className="rcard__compare rcard__compare--translated">
          <section aria-label="English transcript">
            <h3>English (Whisper)</h3>
            <p lang="en">{clip.text_en || <span className="muted">Not transcribed yet</span>}</p>
          </section>
          <section aria-label="LLM translation">
            <h3>LLM translation</h3>
            <p lang="bn">{clip.text_translated || <span className="muted">Not translated yet</span>}</p>
          </section>
        </div>
      )}
      {clip.reason === "pending_llm" && (
        <p className="rcard__note">
          The LLM was not reachable when this clip was translated. Type the Bangla text yourself, or reject the clip.
        </p>
      )}

      <label className="field rcard__final">
        <span className="rcard__final-label">
          Final text
          <span className={`rcard__save rcard__save--${save}`} aria-live="polite">
            {saveText[save]}
          </span>
        </span>
        <textarea ref={boxRef} className="textarea" lang="bn" value={text} onChange={(e) => change(e.target.value)}
          aria-describedby={error ? errorId : undefined} aria-invalid={error ? true : undefined} rows={3} />
      </label>
      {error && (
        <p id={errorId} className="notice notice--error rcard__error" role="alert">
          {error}
        </p>
      )}

      <div className="rcard__actions">
        <button type="button" className="btn btn--primary" disabled={busy} onClick={() => void decide("approve")}
          aria-keyshortcuts="Enter Control+Enter">
          Approve <kbd>Enter</kbd>
        </button>
        <button type="button" className="btn btn--danger" disabled={busy} onClick={() => void decide("reject")}
          aria-keyshortcuts="R">
          Reject <kbd>R</kbd>
        </button>
        <span className="rcard__nav">
          <button type="button" className="btn" onClick={onPrev} aria-keyshortcuts="K">
            Previous <kbd>K</kbd>
          </button>
          <button type="button" className="btn" onClick={onNext} aria-keyshortcuts="J">
            Next <kbd>J</kbd>
          </button>
        </span>
      </div>
      <p className="rcard__hint muted">
        <kbd>Space</kbd> plays the clip. In the text box, <kbd>Ctrl</kbd>+<kbd>Enter</kbd> approves. Edits save on
        their own.
      </p>
    </article>
  );
}
