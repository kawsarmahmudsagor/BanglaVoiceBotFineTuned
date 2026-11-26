import { useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import { getClipAudio } from "../api";

const BAR_W = 2;
const BAR_GAP = 1;

type AudioContextCtor = typeof AudioContext;

function audioContextCtor(): AudioContextCtor | undefined {
  const w = window as unknown as { AudioContext?: AudioContextCtor; webkitAudioContext?: AudioContextCtor };
  return w.AudioContext ?? w.webkitAudioContext;
}

/** Max absolute sample per bucket, scaled so the loudest bucket is 1. */
export function computePeaks(samples: Float32Array, buckets: number): Float32Array {
  const out = new Float32Array(buckets);
  const size = samples.length / buckets;
  let max = 0;
  for (let i = 0; i < buckets; i++) {
    let m = 0;
    const end = Math.min(samples.length, Math.floor((i + 1) * size));
    for (let j = Math.floor(i * size); j < end; j++) {
      const v = Math.abs(samples[j]);
      if (v > m) m = v;
    }
    out[i] = m;
    if (m > max) max = m;
  }
  if (max > 0) for (let i = 0; i < buckets; i++) out[i] /= max;
  return out;
}

interface Props {
  datasetId: string;
  clipId: string;
  /** The clip's <audio> element: drives the playhead and receives seeks. */
  audio: HTMLAudioElement | null;
  height?: number;
}

/** Peaks of the decoded clip drawn on a canvas; the played part is filled with the accent colour. */
export function Waveform({ datasetId, clipId, audio, height = 64 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [samples, setSamples] = useState<Float32Array | null>(null);
  const [failed, setFailed] = useState(false);
  const [width, setWidth] = useState(0);
  const [played, setPlayed] = useState(0);

  // decode
  useEffect(() => {
    setSamples(null);
    setFailed(false);
    const Ctor = audioContextCtor();
    if (!Ctor) {
      setFailed(true);
      return;
    }
    let cancelled = false;
    getClipAudio(datasetId, clipId)
      .then(async (buf) => {
        const ctx = new Ctor();
        try {
          return await ctx.decodeAudioData(buf);
        } finally {
          void ctx.close();
        }
      })
      .then((decoded) => {
        if (!cancelled) setSamples(decoded.getChannelData(0));
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId, clipId]);

  // width follows the container
  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    setWidth(el.clientWidth);
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // playhead: animation frames while playing, one update on seek/pause
  useEffect(() => {
    if (!audio) return;
    let frame = 0;
    const update = () => setPlayed(audio.duration > 0 ? audio.currentTime / audio.duration : 0);
    const tick = () => {
      update();
      frame = requestAnimationFrame(tick);
    };
    const onPlay = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(tick);
    };
    const onStop = () => {
      cancelAnimationFrame(frame);
      update();
    };
    audio.addEventListener("play", onPlay);
    audio.addEventListener("pause", onStop);
    audio.addEventListener("ended", onStop);
    audio.addEventListener("seeked", onStop);
    audio.addEventListener("loadedmetadata", onStop);
    return () => {
      cancelAnimationFrame(frame);
      audio.removeEventListener("play", onPlay);
      audio.removeEventListener("pause", onStop);
      audio.removeEventListener("ended", onStop);
      audio.removeEventListener("seeked", onStop);
      audio.removeEventListener("loadedmetadata", onStop);
    };
  }, [audio]);

  const bars = Math.max(1, Math.floor(width / (BAR_W + BAR_GAP)));
  const peaks = useMemo(() => (samples ? computePeaks(samples, bars) : null), [samples, bars]);

  // draw
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !peaks || width <= 0) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const g = canvas.getContext("2d");
    if (!g) return;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, width, height);
    const css = getComputedStyle(canvas);
    const playedColor = css.getPropertyValue("--fill").trim() || "#0b8a64";
    const restColor = css.getPropertyValue("--muted").trim() || "#56635c";
    const mid = height / 2;
    const playedBars = played * bars;
    for (let i = 0; i < bars; i++) {
      const h = Math.max(1, peaks[i] * (height - 4));
      g.fillStyle = i < playedBars ? playedColor : restColor;
      g.globalAlpha = i < playedBars ? 1 : 0.45;
      g.fillRect(i * (BAR_W + BAR_GAP), mid - h / 2, BAR_W, h);
    }
    g.globalAlpha = 1;
  }, [peaks, bars, width, height, played]);

  function seek(e: MouseEvent<HTMLCanvasElement>) {
    if (!audio || !(audio.duration > 0)) return;
    const rect = e.currentTarget.getBoundingClientRect();
    audio.currentTime = ((e.clientX - rect.left) / rect.width) * audio.duration;
  }

  return (
    <div className="wave" style={{ height }}>
      <canvas ref={canvasRef} className="wave__canvas" onClick={seek} aria-hidden="true" />
      {!samples && (
        <span className="wave__note">{failed ? "Waveform unavailable" : "Loading waveform…"}</span>
      )}
    </div>
  );
}
